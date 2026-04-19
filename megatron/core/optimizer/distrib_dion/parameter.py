"""Dion parameter-path helpers for the distributed optimizer."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Callable, FrozenSet, Iterable, List, Optional, Tuple

import torch
import torch.distributed as dist

from ...distributed.param_and_grad_buffer import _HandleGroup
from ...fp8_utils import is_float8tensor
from .sharding import (
    DionShardLayout,
    compute_fs_flat_segments,
    compute_fs_shard_range,
    get_fs_split_dim,
    get_tp_split_dim,
    is_tp_enabled,
    fs_shard_view_2d,
)


def annotate_dion_candidates(module: torch.nn.Module) -> None:
    """Mark all local parameters as potential Dion candidates."""
    for param in module.parameters():
        param.dion_candidate = True


def is_dion_param(param: torch.Tensor, param_name: Optional[str] = None) -> bool:
    """Return True iff this parameter should use Dion FS sharding."""
    del param_name
    if getattr(param, "use_dion", None) is False:
        return False
    if not getattr(param, "dion_candidate", False):
        return False
    if param.ndim != 2:
        return False
    if getattr(param, "is_embedding_or_output_parameter", False):
        return False
    if getattr(param, "is_lm_head_parameter", False):
        return False
    if is_float8tensor(param):
        return False
    return True


def is_moe_expert_param(param: torch.Tensor, param_name: Optional[str] = None) -> bool:
    """Return True iff this parameter is a MoE expert weight."""
    num_local_experts = getattr(param, "num_local_experts", None)
    if num_local_experts is not None and int(num_local_experts) > 1:
        return True

    resolved_name = param_name or getattr(param, "_param_name", None)
    if resolved_name and ".experts." in resolved_name:
        return True

    return False


@dataclass(frozen=True)
class DionBucketEntry:
    """Bucket-local Dion shard layout for one parameter."""

    param: torch.nn.Parameter
    shard_layout: DionShardLayout
    size_per_rank: int
    shard_capacity: int
    shard_offset: int
    canonical_bucket_start: int
    canonical_bucket_end: int
    canonical_rank_flat_segments: Tuple[Tuple[Tuple[int, int], ...], ...]
    grad_rank_flat_segments: Tuple[Tuple[Tuple[int, int], ...], ...]
    rank_split_ranges: Tuple[Tuple[int, int], ...]

    @property
    def local_shape(self) -> Tuple[int, int]:
        return self.shard_layout.local_shape

    @property
    def global_shape(self) -> Tuple[int, int]:
        return self.shard_layout.global_shape

    @property
    def fs_shard_dim(self) -> int:
        return int(self.shard_layout.fs_shard_dim)

    @property
    def start_idx(self) -> int:
        return int(self.shard_layout.start_idx)

    @property
    def end_idx(self) -> int:
        return int(self.shard_layout.end_idx)

    @property
    def local_numel(self) -> int:
        return int(self.shard_layout.local_numel)


@dataclass(frozen=True)
class DionBucketLayout:
    """Bucket-local Dion transport layout under stock DO control."""

    entries: Tuple[DionBucketEntry, ...]
    shard_size: int
    gathered_numel: int
    grad_gathered_numel: int
    param_ids: FrozenSet[int]

    @property
    def has_params(self) -> bool:
        return bool(self.entries)

    @property
    def entry_count(self) -> int:
        return len(self.entries)

    @property
    def max_shard_capacity(self) -> int:
        return max((int(entry.shard_capacity) for entry in self.entries), default=0)


def build_dion_entries(
    *,
    bucket,
    param_map,
    dion_static_info_by_param,
    fs_size: int,
    fs_rank: int,
    grad_shard_group_size: int,
) -> tuple[DionBucketLayout | None, dict, int]:
    """Build bucket-local Dion shard layouts without mutating parent `param_map`."""
    if grad_shard_group_size <= 0:
        raise RuntimeError(
            f"[Dion] invalid Dion grad shard group size: {grad_shard_group_size}"
        )
    if grad_shard_group_size % fs_size != 0:
        raise RuntimeError(
            "[Dion] Dion grad shard group is incompatible with FS topology "
            f"(grad_group={grad_shard_group_size}, fs_size={fs_size})"
        )
    grad_ranks_per_fs_rank = grad_shard_group_size // fs_size

    entries: List[DionBucketEntry] = []
    dion_shard_layout_by_param = {}
    shard_offset = 0

    for param in param_map.keys():
        if not getattr(param, "is_dion_param", False):
            continue

        static_info = dion_static_info_by_param.get(param)
        if static_info is None:
            continue

        global_shape = tuple(static_info["global_shape"])
        fs_shard_dim = int(static_info["fs_shard_dim"])
        m, n = tuple(int(dim) for dim in param.shape)

        split_size = m if fs_shard_dim == 0 else n
        start_idx, end_idx = compute_fs_shard_range(split_size, fs_size, fs_rank)
        local_split_size = int(end_idx) - int(start_idx)
        size_per_rank = math.ceil(split_size / fs_size)

        if fs_shard_dim == 0:
            local_shape = (local_split_size, n)
            shard_capacity = size_per_rank * n
        else:
            local_shape = (m, local_split_size)
            shard_capacity = m * size_per_rank

        if not hasattr(bucket, "param_to_index") or param not in bucket.param_to_index:
            raise RuntimeError(
                "[Dion] missing canonical bucket span for Dion param "
                f"param={getattr(param, '_param_name', f'id_{id(param)}')}"
            )

        canonical_bucket_start, canonical_bucket_end = bucket.param_to_index[param]
        canonical_bucket_start = int(canonical_bucket_start)
        canonical_bucket_end = int(canonical_bucket_end)

        canonical_rank_flat_segments = []
        rank_split_ranges = []
        for rank_i in range(fs_size):
            rank_start, rank_end = compute_fs_shard_range(split_size, fs_size, rank_i)
            rank_split_ranges.append((int(rank_start), int(rank_end)))
            canonical_rank_flat_segments.append(
                compute_fs_flat_segments(
                    full_start=canonical_bucket_start,
                    m=m,
                    n=n,
                    fs_shard_dim=fs_shard_dim,
                    start_idx=rank_start,
                    end_idx=rank_end,
                )
            )
        grad_rank_flat_segments = []
        for grad_rank in range(grad_shard_group_size):
            grad_fs_rank = int(grad_rank // grad_ranks_per_fs_rank)
            grad_rank_flat_segments.append(canonical_rank_flat_segments[grad_fs_rank])

        shard_layout = DionShardLayout(
            local_shape=tuple(int(dim) for dim in local_shape),
            global_shape=tuple(int(dim) for dim in global_shape),
            fs_shard_dim=fs_shard_dim,
            start_idx=int(start_idx),
            end_idx=int(end_idx),
            per_expert_global_shape=static_info.get("per_expert_global_shape"),
        )
        dion_shard_layout_by_param[param] = shard_layout
        entries.append(
            DionBucketEntry(
                param=param,
                shard_layout=shard_layout,
                size_per_rank=int(size_per_rank),
                shard_capacity=int(shard_capacity),
                shard_offset=int(shard_offset),
                canonical_bucket_start=canonical_bucket_start,
                canonical_bucket_end=canonical_bucket_end,
                canonical_rank_flat_segments=tuple(
                    tuple((int(seg_start), int(seg_end)) for seg_start, seg_end in rank_segments)
                    for rank_segments in canonical_rank_flat_segments
                ),
                grad_rank_flat_segments=tuple(
                    tuple((int(seg_start), int(seg_end)) for seg_start, seg_end in rank_segments)
                    for rank_segments in grad_rank_flat_segments
                ),
                rank_split_ranges=tuple(rank_split_ranges),
            )
        )
        shard_offset += int(shard_capacity)

    bucket_layout = None
    if entries:
        bucket_layout = DionBucketLayout(
            entries=tuple(entries),
            shard_size=int(shard_offset),
            gathered_numel=int(shard_offset) * int(fs_size),
            grad_gathered_numel=int(shard_offset) * int(grad_shard_group_size),
            param_ids=frozenset(id(entry.param) for entry in entries),
        )

    return bucket_layout, dion_shard_layout_by_param, len(entries)


def select_non_dion_bucket_params_(*, bucket_params, dion_layout: DionBucketLayout | None):
    """Return bucket params that are not carried by Dion transport layout entries."""
    dion_param_ids = frozenset() if dion_layout is None else dion_layout.param_ids
    return [param for param in bucket_params if id(param) not in dion_param_ids]


def bucket_rank_range_(full_start: int, full_end: int, shard_size: int, rank: int) -> tuple[int, int]:
    """Return the intersection of one full bucket span with one stock local shard."""
    shard_abs_start = rank * shard_size
    shard_abs_end = shard_abs_start + shard_size
    local_abs_start = max(int(full_start), shard_abs_start)
    local_abs_end = min(int(full_end), shard_abs_end)
    if local_abs_end <= local_abs_start:
        return 0, 0
    return local_abs_start - shard_abs_start, local_abs_end - shard_abs_start


def param_rank_range_(full_start: int, full_end: int, shard_size: int, rank: int) -> tuple[int, int]:
    """Return the intersection of one full bucket span in param-local coordinates."""
    shard_abs_start = rank * shard_size
    shard_abs_end = shard_abs_start + shard_size
    local_abs_start = max(int(full_start), shard_abs_start)
    local_abs_end = min(int(full_end), shard_abs_end)
    if local_abs_end <= local_abs_start:
        return 0, 0
    return local_abs_start - int(full_start), local_abs_end - int(full_start)


def collect_dion_bucket_params(dion_layout: DionBucketLayout | None) -> list[torch.nn.Parameter]:
    """Return unique bucket-local Dion params in canonical entry order."""
    if dion_layout is None or not dion_layout.has_params:
        return []
    params = []
    seen_param_ids = set()
    for entry in dion_layout.entries:
        if entry.param is None or id(entry.param) in seen_param_ids:
            continue
        seen_param_ids.add(id(entry.param))
        params.append(entry.param)
    return params


def serialize_dion_bucket_gather_layout_(dion_layout: DionBucketLayout) -> tuple[int, ...]:
    """Serialize bucket-global Dion gather invariants for cross-rank validation."""
    payload: list[int] = [
        int(dion_layout.entry_count),
        int(dion_layout.shard_size),
        int(dion_layout.gathered_numel),
        int(dion_layout.max_shard_capacity),
    ]
    for entry in dion_layout.entries:
        payload.extend(
            [
                int(entry.shard_offset),
                int(entry.shard_capacity),
                int(entry.canonical_bucket_start),
                int(entry.canonical_bucket_end),
                int(entry.fs_shard_dim),
                int(entry.size_per_rank),
            ]
        )
        payload.append(len(entry.canonical_rank_flat_segments))
        for rank_segments in entry.canonical_rank_flat_segments:
            payload.append(len(rank_segments))
            for seg_start, seg_end in rank_segments:
                payload.extend([int(seg_start), int(seg_end)])
    return tuple(payload)


def set_dion_local_shard_(
    *,
    entry,
    full_view_2d: torch.Tensor,
    update_data_shard: Callable[[torch.nn.Parameter, torch.Tensor], None],
    param_name: Callable[[torch.nn.Parameter], str],
) -> None:
    """Set one Dion param's canonical local shard view from a full bucket view."""
    local_target = fs_shard_view_2d(
        full_view_2d,
        int(entry.fs_shard_dim),
        int(entry.start_idx),
        int(entry.end_idx),
    )
    local_numel = int(entry.local_numel)
    if local_numel != local_target.numel():
        raise RuntimeError(
            "[Dion] local restore shard size mismatch "
            f"param={param_name(entry.param)} "
            f"source={local_numel} target={int(local_target.numel())}"
        )
    update_data_shard(entry.param, local_target)
    entry.param._fs_shard = local_target


def restore_dion_local_shards_from_bucket(
    *,
    dion_layout: DionBucketLayout | None,
    get_full_view_2d: Callable[[object], torch.Tensor],
    update_data_shard: Callable[[torch.nn.Parameter, torch.Tensor], None],
    param_name: Callable[[torch.nn.Parameter], str],
) -> None:
    """Restore all Dion local shard aliases from canonical bucket.param_data views."""
    if dion_layout is None or not dion_layout.has_params:
        return
    for entry in dion_layout.entries:
        set_dion_local_shard_(
            entry=entry,
            full_view_2d=get_full_view_2d(entry),
            update_data_shard=update_data_shard,
            param_name=param_name,
        )


def _restore_dion_bucket(
    *,
    bucket,
    prepared_entries: Iterable[object],
    gathered_buffer: torch.Tensor,
    shard_group_size: int,
    update_data_shard: Callable[[torch.nn.Parameter, torch.Tensor], None],
    param_name: Callable[[torch.nn.Parameter], str],
) -> None:
    """Restore canonical bucket storage from one gathered Dion shard buffer."""
    if gathered_buffer.dim() != 2:
        raise RuntimeError(
            f"[Dion] gathered Dion bucket buffer must be 2D, got shape={tuple(gathered_buffer.shape)}"
        )
    if gathered_buffer.size(0) != shard_group_size:
        raise RuntimeError(
            "[Dion] gathered Dion bucket buffer group-size mismatch "
            f"bucket={getattr(bucket, 'bucket_id', -1)} "
            f"buffer_group={int(gathered_buffer.size(0))} expected_group={int(shard_group_size)}"
        )

    for entry, full_view_2d in prepared_entries:
        m, n = (int(dim) for dim in full_view_2d.shape)
        fs_shard_dim = int(entry.fs_shard_dim)
        shard_start = int(entry.shard_offset)
        shard_end = shard_start + int(entry.shard_capacity)
        for rank_i, (rank_start, rank_end) in enumerate(entry.rank_split_ranges):
            rank_source = gathered_buffer[rank_i, shard_start:shard_end]
            rank_start = int(rank_start)
            rank_end = int(rank_end)
            local_split_size = rank_end - rank_start
            if local_split_size <= 0:
                continue

            if fs_shard_dim == 0:
                local_numel = local_split_size * n
                local_source = rank_source[:local_numel].view(local_split_size, n)
                full_view_2d[rank_start:rank_end, :].copy_(local_source)
            else:
                local_numel = m * local_split_size
                local_source = rank_source[:local_numel].view(m, local_split_size)
                full_view_2d[:, rank_start:rank_end].copy_(local_source)
        set_dion_local_shard_(
            entry=entry,
            full_view_2d=full_view_2d,
            update_data_shard=update_data_shard,
            param_name=param_name,
        )


logger = logging.getLogger(__name__)


class CallbackHandle:
    """Waitable handle that runs a local callback after the collective completes."""

    def __init__(self, work, callback=None):
        self._work = work
        self._callback = callback

    def wait(self):
        if self._work is not None:
            self._work.wait()
        if self._callback is not None:
            self._callback()
        self._work = None
        self._callback = None


class BucketGatherHandle:
    """Preserve mixed-bucket canonical restore ordering behind one wait handle."""

    def __init__(self, non_dion_handle, dion_handle):
        self._non_dion_handle = non_dion_handle
        self._dion_handle = dion_handle

    def wait(self):
        if self._non_dion_handle is not None:
            self._non_dion_handle.wait()
        if self._dion_handle is not None:
            self._dion_handle.wait()
        self._non_dion_handle = None
        self._dion_handle = None


def check_shard_identity(
    *,
    optimizer,
    model_float16_groups,
    main_shard_groups,
) -> None:
    """Verify shard params still match optimizer param group objects by data_ptr."""
    mismatch_count = 0
    match_count = 0

    opt_param_by_ptr = {}
    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            if param is None or param.numel() == 0:
                continue
            opt_param_by_ptr[param.data_ptr()] = param

    for group_index, (model_group, shard_param_group) in enumerate(
        zip(model_float16_groups, main_shard_groups)
    ):
        for param_index, (_, shard_param) in enumerate(zip(model_group, shard_param_group)):
            if shard_param is None:
                continue
            if shard_param.numel() == 0:
                continue

            shard_ptr = shard_param.data_ptr()
            opt_param = opt_param_by_ptr.get(shard_ptr)

            if opt_param is None:
                if mismatch_count < 5:
                    logger.error(
                        "[IDENTITY MISMATCH] main_shard_groups[%s][%s] (shape=%s, ptr=%s) NOT FOUND in optimizer.param_groups! Dion updates will be LOST!",
                        group_index,
                        param_index,
                        shard_param.shape,
                        shard_ptr,
                    )
                mismatch_count += 1
            elif opt_param is not shard_param:
                if mismatch_count < 5:
                    logger.error(
                        "[Dion] main_shard_groups[%s][%s] object mismatch: same data_ptr=%s but different object. id(shard)=%s, id(opt)=%s",
                        group_index,
                        param_index,
                        shard_ptr,
                        id(shard_param),
                        id(opt_param),
                    )
                mismatch_count += 1
            else:
                match_count += 1

    if mismatch_count > 0:
        raise RuntimeError(
            "[Dion] main_shard_groups identity mismatch with optimizer.param_groups: "
            f"mismatches={mismatch_count} matched={match_count}"
        )


def build_bucket_param_map(
    cls,
    parent_result,
    ordered_params,
    dp_group,
    dp_rank,
    bucket_index,
    param_index_map,
    bucket_offset: int,
    bucket_size: int,
    bucket_param_to_index=None,
    param_to_name=None,
):
    """Rebuild the parent DO param_map in canonical bucket-param order."""
    del bucket_param_to_index, param_to_name
    from collections import OrderedDict
    from ..distrib_optimizer import DistributedOptimizer, Range

    parent_param_map = parent_result["param_map"]
    dp_world_size = dp_group.size()
    if bucket_size % dp_world_size != 0:
        raise RuntimeError(
            f"[Dion] bucket_size must be divisible by dp_size for canonical param_map "
            f"(bucket={bucket_index}, bucket_size={bucket_size}, dp_size={dp_world_size})"
        )

    max_gbuf_range_size = bucket_size // dp_world_size
    gbuf_world_start = dp_rank * max_gbuf_range_size
    gbuf_world_end = min(bucket_size, gbuf_world_start + max_gbuf_range_size)
    gbuf_world_range = Range(
        gbuf_world_start + bucket_offset,
        gbuf_world_end + bucket_offset,
    )
    reconstructed_param_map = DistributedOptimizer._build_model_gbuf_param_range_map(
        param_index_map,
        gbuf_world_range,
        bucket_offset,
    )

    canonical_param_map = OrderedDict()
    for param in ordered_params:
        parent_info = parent_param_map.get(param)
        reconstructed_info = reconstructed_param_map.get(
            param,
            {
                "param": Range(0, 0),
                "gbuf_world": Range(0, 0),
                "gbuf_local": Range(0, 0),
                "gbuf_world_in_bucket": Range(0, 0),
            },
        )
        chosen_info = parent_info or reconstructed_info
        canonical_param_map[param] = chosen_info

    parent_result["param_map"] = canonical_param_map
    return canonical_param_map


def mark_bucket_dion_params(cls, param_map, param_to_name, fs_size):
    """Classify bucket params and build static Dion metadata once."""
    del cls, fs_size
    from .. import parallel_state

    dion_param_count = 0
    dion_static_info_by_param = {}

    for param in param_map.keys():
        param_name = None
        if param_to_name is not None and param in param_to_name:
            param_name = param_to_name[param]
        if param_name:
            param._param_name = param_name

        param.is_dion_param = is_dion_param(param, param_name)

        is_expert = is_moe_expert_param(param, param_name)
        raw_tp_split_dim = get_tp_split_dim(param)
        has_tp = is_tp_enabled(param)
        if is_expert and has_tp:
            tp_world_size = parallel_state.get_expert_tensor_parallel_world_size()
        else:
            tp_world_size = parallel_state.get_tensor_model_parallel_world_size() if has_tp else 1
        tp_shard_dim = raw_tp_split_dim if has_tp and tp_world_size > 1 else -1

        if not param.is_dion_param:
            continue

        dion_param_count += 1
        m, n = param.shape
        fs_shard_dim = get_fs_split_dim(tp_shard_dim)

        if tp_shard_dim == 0:
            global_m = m * tp_world_size
            global_n = n
        elif tp_shard_dim == 1:
            global_m = m
            global_n = n * tp_world_size
        else:
            global_m = m
            global_n = n

        num_local_experts = getattr(param, "num_local_experts", None)
        if num_local_experts is not None and num_local_experts > 1:
            if tp_shard_dim == 0:
                per_expert_global_shape = (global_m // num_local_experts, global_n)
            elif tp_shard_dim == 1:
                per_expert_global_shape = (global_m, global_n // num_local_experts)
            else:
                per_expert_global_shape = (global_m, global_n)
        else:
            per_expert_global_shape = None

        dion_static_info_by_param[param] = {
            "is_dion": True,
            "global_shape": (global_m, global_n),
            "fs_shard_dim": fs_shard_dim,
            "tp_shard_dim": tp_shard_dim,
            "per_expert_global_shape": per_expert_global_shape,
        }

    return dion_param_count, dion_static_info_by_param


def bucket_param_view(bucket, param) -> Optional[torch.Tensor]:
    """Return the canonical full-param view for one bucket param."""
    if bucket is None or getattr(bucket, "param_data", None) is None:
        return None
    if not hasattr(bucket, "param_to_index") or param not in bucket.param_to_index:
        return None
    start, end = bucket.param_to_index[param]
    return bucket.param_data.view(-1)[start:end].view(param.data.shape)


def bucket_full_param_view_2d(bucket, param, entry) -> Optional[torch.Tensor]:
    """Return the canonical full 2D bucket view for one Dion param."""
    if bucket is None or getattr(bucket, "param_data", None) is None:
        return None
    if not hasattr(bucket, "param_to_index") or param not in bucket.param_to_index:
        return None
    start, end = bucket.param_to_index[param]
    full_flat = bucket.param_data.view(-1)[start:end]
    global_shape = entry.global_shape
    fs_shard_dim = int(entry.fs_shard_dim)
    if fs_shard_dim == 0:
        full_shape = (int(global_shape[0]), int(param.shape[1]))
    else:
        full_shape = (int(param.shape[0]), int(global_shape[1]))
    expected_numel = int(full_shape[0]) * int(full_shape[1])
    if full_flat.numel() != expected_numel:
        raise RuntimeError(
            "[Dion] canonical full-param view size mismatch "
            f"for param={getattr(param, 'shape', None)} bucket={getattr(bucket, 'bucket_id', -1)} "
            f"view_numel={int(full_flat.numel())} expected_numel={expected_numel} "
            f"full_shape={full_shape}"
        )
    return full_flat.view(full_shape)


def set_bucket_param_views(
    optimizer,
    bucket,
    *,
    copy_data: bool,
    params: Optional[list[torch.nn.Parameter]] = None,
) -> None:
    """Ensure selected params in `bucket` alias the bucket's canonical param buffer."""
    if bucket is None or getattr(bucket, "param_data", None) is None:
        return
    params_to_set = list(bucket.params) if params is None else list(params)
    source_copies = {}
    if copy_data:
        for param in params_to_set:
            expected_view = bucket_param_view(bucket, param)
            if expected_view is None:
                continue
            if (
                param.data.shape == expected_view.shape
                and param.data.data_ptr() == expected_view.data_ptr()
            ):
                continue
            if param.data.numel() != expected_view.numel():
                continue
            source_copies[param] = param.data.view(expected_view.shape).clone()
    for param in params_to_set:
        expected_view = bucket_param_view(bucket, param)
        if expected_view is None:
            continue
        if (
            param.data.shape == expected_view.shape
            and param.data.data_ptr() == expected_view.data_ptr()
        ):
            continue
        if copy_data and param in source_copies:
            expected_view.copy_(source_copies[param])
        param.data = expected_view


def check_bucket_param_views(
    optimizer,
    bucket,
    *,
    context: str,
    params: Optional[list[torch.nn.Parameter]] = None,
) -> None:
    """Verify selected param views still alias the canonical bucket buffer."""
    if bucket is None or not getattr(bucket, "_dion_requires_param_sync_check", False):
        return
    if getattr(bucket, "param_data", None) is None:
        raise RuntimeError(
            f"[Dion] {context}: bucket {getattr(bucket, 'bucket_id', -1)} missing bucket.param_data"
        )
    params_to_check = list(bucket.params) if params is None else list(params)
    for param in params_to_check:
        expected_view = bucket_param_view(bucket, param)
        if expected_view is None:
            continue
        if (
            param.data.shape != expected_view.shape
            or param.data.data_ptr() != expected_view.data_ptr()
        ):
            param_name = optimizer._lookup_param_name(getattr(optimizer, "param_to_name", None), param)
            if param_name is None:
                param_name = optimizer._find_param_name(param) or f"id_{id(param)}"
            logger.error(
                "[DION_PARAM_BUFFER_VIEW_MISMATCH] context=%s bucket_id=%s param=%s param_ptr=%s bucket_ptr=%s",
                context,
                getattr(bucket, "bucket_id", -1),
                param_name,
                param.data.data_ptr(),
                expected_view.data_ptr(),
            )
            raise RuntimeError(
                f"[Dion] {context}: param.data no longer aliases bucket.param_data for {param_name}"
            )


def attach_dion_bucket_layout(
    optimizer,
    *,
    gbuf_idx: int,
    buffer,
    bucket,
    dion_bucket_layout: DionBucketLayout,
) -> None:
    """Validate that stored Dion entries already target the current runtime bucket."""
    if not hasattr(bucket, "param_to_index") or bucket.param_to_index is None:
        raise RuntimeError(
            f"[Dion] buffer={gbuf_idx} bucket={bucket.bucket_id} missing bucket.param_to_index"
        )

    for entry in dion_bucket_layout.entries:
        entry_param = entry.param
        if entry_param not in bucket.param_to_index:
            buffer_name_map = getattr(buffer, "param_to_name", None)
            param_name = optimizer._lookup_param_name(buffer_name_map, entry_param)
            raise RuntimeError(
                f"[Dion] buffer={gbuf_idx} bucket={bucket.bucket_id} stored param "
                f"{param_name or f'id_{id(entry_param)}'} "
                "is not present in the current runtime bucket"
            )
    bucket.dion_layout = dion_bucket_layout
    for entry in dion_bucket_layout.entries:
        optimizer._dion_buckets_by_param[entry.param] = bucket
        optimizer._dion_entries_by_param[entry.param] = entry


def init_dion_bucket(
    optimizer,
    *,
    gbuf_idx: int,
    buffer,
    bucket,
    dion_bucket_layout: DionBucketLayout,
    fs_group,
) -> None:
    """Configure one bucket that contains at least one Dion param."""
    attach_dion_bucket_layout(
        optimizer,
        gbuf_idx=gbuf_idx,
        buffer=buffer,
        bucket=bucket,
        dion_bucket_layout=dion_bucket_layout,
    )
    optimizer._init_bucket_comm(bucket, fs_group)
    bucket.dion_optimizer = optimizer
    bucket._dion_requires_param_sync_check = True
    bucket._dion_full_param_ready = True
    set_bucket_param_views(optimizer, bucket, copy_data=True)


def init_non_dion_bucket(optimizer, *, gbuf_idx: int, buffer, bucket, fs_group) -> None:
    """Configure one bucket that has no Dion layout."""
    has_dion = any(getattr(param, "is_dion_param", False) for param in bucket.params)
    if has_dion:
        name_map = getattr(optimizer, "param_to_name", {}) or getattr(buffer, "param_to_name", {})
        dion_params = []
        for param in bucket.params:
            if getattr(param, "is_dion_param", False):
                param_name = optimizer._lookup_param_name(name_map, param)
                dion_params.append((id(param), param_name or f"id_{id(param)}", tuple(param.shape)))
        logger.error(
            f"[Dion] Buffer {gbuf_idx} Bucket {bucket.bucket_id} has Dion params but no dion_bucket_layout. params={dion_params}"
        )
        raise RuntimeError(
            f"[Dion] Buffer {gbuf_idx} Bucket {bucket.bucket_id} has Dion params but no dion_bucket_layout"
        )

    bucket.dion_layout = None
    optimizer._init_bucket_comm(bucket, fs_group)
    bucket.dion_optimizer = optimizer
    bucket._dion_requires_param_sync_check = False
    bucket._dion_full_param_ready = True


def bucket_dion_full_view(optimizer, bucket, entry: DionBucketEntry) -> torch.Tensor:
    """Return the canonical full-param view for one Dion entry."""
    full_view_2d = bucket_full_param_view_2d(bucket, entry.param, entry)
    if full_view_2d is None:
        raise RuntimeError(
            "[Dion] canonical FS gather requires bucket.param_data view "
            f"for param={optimizer._param_name(entry.param) or f'id_{id(entry.param)}'} "
            f"bucket={getattr(bucket, 'bucket_id', -1)}"
        )
    return full_view_2d


def fill_dion_shard_buffer(
    optimizer,
    *,
    entry: DionBucketEntry,
    full_view_2d: torch.Tensor,
    shard_buffer: torch.Tensor,
) -> None:
    """Pack one Dion local shard into a gather input buffer."""
    canonical_local_source = fs_shard_view_2d(
        full_view_2d,
        int(entry.fs_shard_dim),
        int(entry.start_idx),
        int(entry.end_idx),
    )
    bound_data_shard = optimizer._get_data_shard(entry.param)
    if bound_data_shard is not None:
        if bound_data_shard.numel() != canonical_local_source.numel():
            raise RuntimeError(
                "[Dion] canonical FS gather source size mismatch "
                f"param={optimizer._param_name(entry.param) or f'id_{id(entry.param)}'} "
                f"bound={int(bound_data_shard.numel())} canonical={int(canonical_local_source.numel())}"
            )
        if bound_data_shard.data_ptr() != canonical_local_source.data_ptr():
            raise RuntimeError(
                "[Dion] canonical FS gather source mismatch "
                f"for param={optimizer._param_name(entry.param) or f'id_{id(entry.param)}'}: "
                "registered data_shard no longer aliases bucket.param_data canonical FS view"
            )
    local_numel = int(entry.local_numel)
    if canonical_local_source.numel() != local_numel:
        raise RuntimeError(
            "[Dion] local gather source size mismatch "
            f"param={optimizer._param_name(entry.param) or f'id_{id(entry.param)}'} "
            f"source={int(canonical_local_source.numel())} expected={local_numel}"
        )
    shard_buffer.zero_()
    shard_buffer[:local_numel].copy_(canonical_local_source.reshape(-1)[:local_numel])


def prepare_dion_bucket_gather(optimizer, bucket) -> Tuple[torch.Tensor, List[Tuple[DionBucketEntry, torch.Tensor]]]:
    """Build one bucket-local Dion shard buffer in canonical entry order."""
    dion_layout = getattr(bucket, "dion_layout", None)
    if dion_layout is None or not dion_layout.has_params:
        return (
            torch.empty(
                0,
                dtype=bucket.param_data.dtype,
                device=torch.cuda.current_device(),
            ),
            [],
        )

    prepared_buffer = torch.empty(
        int(dion_layout.shard_size),
        dtype=bucket.param_data.dtype,
        device=torch.cuda.current_device(),
    )
    prepared_entries = []
    for entry in dion_layout.entries:
        full_view_2d = bucket_dion_full_view(optimizer, bucket, entry)
        shard_start = int(entry.shard_offset)
        shard_end = shard_start + int(entry.shard_capacity)
        fill_dion_shard_buffer(
            optimizer,
            entry=entry,
            full_view_2d=full_view_2d,
            shard_buffer=prepared_buffer[shard_start:shard_end],
        )
        prepared_entries.append((entry, full_view_2d))
    return prepared_buffer, prepared_entries


def restore_bucket(
    optimizer,
    *,
    bucket,
    prepared_entries,
    gathered_buffer: torch.Tensor,
    shard_group_size: int,
) -> None:
    """Restore canonical bucket.param_data from one bucket-wise gathered Dion shard buffer."""
    _restore_dion_bucket(
        bucket=bucket,
        prepared_entries=prepared_entries,
        gathered_buffer=gathered_buffer,
        shard_group_size=shard_group_size,
        update_data_shard=optimizer._update_data_shard,
        param_name=lambda param: optimizer._param_name(param) or f'id_{id(param)}',
    )


def bucket_local_shard_group(optimizer, bucket):
    """Resolve the stock DO local-shard group that owns one bucket."""
    dp_group = getattr(bucket, "intra_distributed_optimizer_instance_group", None)
    if dp_group is not None:
        return (
            dp_group,
            int(getattr(bucket, "intra_distributed_optimizer_instance_size", dp_group.size())),
            int(getattr(bucket, "intra_distributed_optimizer_instance_rank", dp_group.rank())),
        )

    if not hasattr(optimizer, "per_model_bucket_groups"):
        return None, None, None

    for bucket_groups in optimizer.per_model_bucket_groups.values():
        for bucket_group in bucket_groups:
            if bucket not in bucket_group.buckets:
                continue
            dp_group = getattr(bucket_group, "intra_distributed_optimizer_instance_group", None)
            if dp_group is None:
                return None, None, None
            return (
                dp_group,
                int(bucket_group.intra_distributed_optimizer_instance_size),
                int(bucket_group.intra_distributed_optimizer_instance_rank),
            )

    return None, None, None


def all_gather_non_dion_bucket(optimizer, bucket, async_op=False):
    """Gather mixed-bucket non-Dion params back into canonical bucket.param_data."""
    dion_layout = getattr(bucket, "dion_layout", None)
    if dion_layout is None or not bucket.has_non_dion_params:
        return None

    dp_group, dp_size, dp_rank = bucket_local_shard_group(optimizer, bucket)
    if dp_group is None:
        raise RuntimeError(
            "[Dion] mixed non-Dion all-gather requires the standard local-shard group."
        )
    if dp_size == 1:
        return None

    shard_size = bucket.param_data.numel() // dp_size
    if shard_size <= 0:
        return None

    input_shard = torch.zeros(
        shard_size,
        dtype=bucket.param_data.dtype,
        device=torch.cuda.current_device(),
    )
    non_dion_params = list(
        select_non_dion_bucket_params_(
            bucket_params=bucket.params_list,
            dion_layout=dion_layout,
        )
    )

    for param in non_dion_params:
        full_start, full_end = bucket.param_to_index[param]
        bucket_start, _ = bucket_rank_range_(full_start, full_end, shard_size, dp_rank)
        param_start, param_end = param_rank_range_(full_start, full_end, shard_size, dp_rank)
        actual_size = max(0, int(param_end) - int(param_start))
        if actual_size <= 0:
            continue

        full_view = bucket_param_view(bucket, param)
        if full_view is None:
            param_name = optimizer._param_name(param)
            raise RuntimeError(
                "[Dion] mixed non-Dion all-gather requires canonical bucket.param_data view "
                f"for param={param_name or f'id_{id(param)}'} "
                f"bucket={getattr(bucket, 'bucket_id', -1)}"
            )
        local_shard = full_view.view(-1)[int(param_start):int(param_end)]
        input_shard[int(bucket_start) : int(bucket_start) + actual_size].copy_(local_shard)

    return dist.all_gather_into_tensor(
        output_tensor=bucket.param_data,
        input_tensor=input_shard,
        group=dp_group,
        async_op=async_op,
    )


def all_gather_dion_bucket(
    optimizer,
    bucket,
    async_op=False,
    prepared_gather=None,
):
    """Gather one bucket-local Dion shard buffer back into canonical bucket.param_data."""
    dion_layout = getattr(bucket, "dion_layout", None)
    if dion_layout is None or not dion_layout.has_params:
        return None

    shard_group = getattr(bucket, "dion_shard_group", None)
    if shard_group is None:
        shard_group = optimizer.fs_group
    shard_group_size = optimizer._group_size(shard_group) if shard_group is not None else 1
    if prepared_gather is None:
        prepared_buffer, prepared_entries = prepare_dion_bucket_gather(optimizer, bucket)
    else:
        prepared_buffer, prepared_entries = prepared_gather

    if shard_group_size == 1:
        def _restore_callback():
            restore_bucket(
                optimizer,
                bucket=bucket,
                prepared_entries=prepared_entries,
                gathered_buffer=prepared_buffer.view(1, -1),
                shard_group_size=1,
            )

        if async_op:
            return CallbackHandle(None, _restore_callback)

        _restore_callback()
        return None
    if shard_group is None:
        return None

    layout_len = int(dion_layout.entry_count)
    bucket_shard_size = int(dion_layout.shard_size)
    expected_gathered_numel = int(dion_layout.gathered_numel)
    if expected_gathered_numel != bucket_shard_size * shard_group_size:
        raise RuntimeError(
            "[Dion] FS gather shard size invariant mismatch "
            f"bucket={getattr(bucket, 'bucket_id', -1)} "
            f"shard_size={bucket_shard_size} fs_size={shard_group_size} "
            f"expected_total={expected_gathered_numel}"
        )

    if not getattr(bucket, "_dion_ag_invariants_verified", False):
        local = torch.tensor(
            [layout_len, bucket_shard_size],
            device=torch.cuda.current_device(),
            dtype=torch.int64,
        )
        local_min = local.clone()
        local_max = local.clone()
        dist.all_reduce(local_min, op=dist.ReduceOp.MIN, group=shard_group)
        dist.all_reduce(local_max, op=dist.ReduceOp.MAX, group=shard_group)
        if not torch.equal(local_min, local_max):
            logger.error(
                "[Dion] FS AG invariants mismatch (bucket_id=%s): local(layout_len=%s, shard_size=%s) "
                "min=%s max=%s",
                getattr(bucket, "bucket_id", -1),
                layout_len,
                bucket_shard_size,
                tuple(int(x) for x in local_min.tolist()),
                tuple(int(x) for x in local_max.tolist()),
            )
            raise RuntimeError("Dion FS all-gather invariants mismatch across FS ranks")

        bucket._dion_ag_invariants_verified = True
        bucket._dion_ag_verified_layout_len = layout_len
        bucket._dion_ag_verified_shard_size = bucket_shard_size

    gathered_buffer = torch.empty(
        expected_gathered_numel,
        dtype=bucket.param_data.dtype,
        device=torch.cuda.current_device(),
    )
    work = dist.all_gather_into_tensor(
        output_tensor=gathered_buffer,
        input_tensor=prepared_buffer,
        group=shard_group,
        async_op=async_op,
    )

    def _restore_callback():
        restore_bucket(
            optimizer,
            bucket=bucket,
            prepared_entries=prepared_entries,
            gathered_buffer=gathered_buffer.view(shard_group_size, bucket_shard_size),
            shard_group_size=shard_group_size,
        )

    if async_op:
        return CallbackHandle(work, _restore_callback)

    _restore_callback()
    return None


def all_gather_bucket_params(optimizer, bucket, async_op=False):
    """Gather all custom bucket params that do not follow the pure standard DO path."""
    dion_layout = getattr(bucket, "dion_layout", None)
    has_dion = dion_layout is not None and dion_layout.has_params
    has_mixed_non_dion = has_dion and bucket.has_non_dion_params

    prepared_dion_entries = None
    if has_mixed_non_dion:
        prepared_dion_entries = prepare_dion_bucket_gather(optimizer, bucket)
        mixed_non_dion_handle = all_gather_non_dion_bucket(optimizer, bucket, async_op=async_op)
        dion_handle = all_gather_dion_bucket(
            optimizer,
            bucket,
            async_op=async_op,
            prepared_gather=prepared_dion_entries,
        )
        if async_op and (mixed_non_dion_handle is not None or dion_handle is not None):
            return BucketGatherHandle(mixed_non_dion_handle, dion_handle)
        return None

    handles = []
    dion_handle = all_gather_dion_bucket(
        optimizer,
        bucket,
        async_op=async_op,
    )
    if dion_handle is not None:
        handles.append(dion_handle)
    mixed_non_dion_handle = all_gather_non_dion_bucket(optimizer, bucket, async_op=async_op)
    if mixed_non_dion_handle is not None:
        handles.append(mixed_non_dion_handle)
    if async_op and handles:
        return _HandleGroup(handles)
    return None
