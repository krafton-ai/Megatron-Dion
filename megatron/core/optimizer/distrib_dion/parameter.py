"""Dion parameter-path helpers for the distributed optimizer."""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from typing import Callable, FrozenSet, Iterable, List, Optional, Tuple

import torch
import torch.distributed as dist

from ...distributed.param_and_grad_buffer import _HandleGroup
from ...fp8_utils import is_float8tensor
from ...transformer.fsdp_dtensor_checkpoint import get_expert_index_from_key
from .sharding import (
    DionShardLayout,
    compute_fs_flat_segments,
    compute_fs_shard_range,
    get_fs_split_dim,
    get_tp_split_dim,
    is_tp_enabled,
    fs_shard_view_2d,
)


def mark_dion_candidates(module: torch.nn.Module) -> None:
    """Mark all local parameters as potential Dion candidates."""
    for param in module.parameters():
        param.dion_candidate = True


def is_dion_param(param: torch.Tensor, param_name: Optional[str] = None) -> bool:
    """Return True iff this parameter should use Dion FS sharding."""
    resolved_name = param_name or getattr(param, "_param_name", None)
    if getattr(param, "use_dion", None) is False:
        return False
    if not getattr(param, "dion_candidate", False):
        return False
    if param.ndim != 2:
        return False
    if getattr(param, "sequence_parallel", False):
        return False
    if getattr(param, "average_gradients_across_tp_domain", False):
        return False
    if getattr(param, "is_embedding_or_output_parameter", False):
        return False
    if getattr(param, "is_lm_head_parameter", False):
        return False
    if is_float8tensor(param):
        return False
    if is_combined_grouped_mlp_param(param, resolved_name):
        return False
    return True


def is_moe_expert_param(param: torch.Tensor, param_name: Optional[str] = None) -> bool:
    """Return True iff this parameter is a MoE expert weight."""
    if not getattr(param, "allreduce", True):
        return True

    num_local_experts = getattr(param, "num_local_experts", None)
    if num_local_experts is not None and int(num_local_experts) > 1:
        return True

    resolved_name = param_name or getattr(param, "_param_name", None)
    if resolved_name and ".experts." in resolved_name:
        return True

    return False


def is_combined_grouped_mlp_param(param: torch.Tensor, param_name: Optional[str] = None) -> bool:
    """Return whether this is a legacy GroupedMLP tensor containing multiple experts."""
    num_local_experts = getattr(param, "num_local_experts", None)
    if num_local_experts is None or int(num_local_experts) <= 1:
        return False
    resolved_name = param_name or getattr(param, "_param_name", None) or ""
    leaf_name = os.path.basename(str(resolved_name).replace(".", os.sep))
    if leaf_name not in {"weight1", "weight2"}:
        return False
    if ".linear_fc" in resolved_name or ".local_experts." in resolved_name:
        return False
    return True


def has_explicit_expert_index(param_name: Optional[str]) -> bool:
    """Return whether MCore names this tensor as one explicit local expert."""
    if not param_name:
        return False
    try:
        return get_expert_index_from_key(param_name) is not None
    except AssertionError:
        return False


@dataclass(frozen=True)
class DionShardEntry:
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

    entries: Tuple[DionShardEntry, ...]
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


@dataclass(frozen=True)
class StandardParamGatherRoute:
    """Bucket all-gather layout for standard params inside a mixed bucket."""

    group_size: int
    group_rank: int
    standard_shard_size: int
    standard_numel: int
    rank_segments: Tuple[Tuple[Tuple[int, int, int], ...], ...]


@dataclass
class ParamGatherBuffer:
    tensor: torch.Tensor
    active: bool = False


def _param_gather_device_key(device: torch.device) -> Tuple[str, int]:
    device = torch.device(device)
    device_index = -1 if device.index is None else int(device.index)
    return device.type, device_index


def _param_gather_cache(bucket) -> tuple[dict, dict]:
    cache = getattr(bucket, "_dion_param_gather_buffers", None)
    if cache is None:
        cache = {}
        bucket._dion_param_gather_buffers = cache
    refs = getattr(bucket, "_dion_param_gather_buffer_refs", None)
    if refs is None:
        refs = {}
        bucket._dion_param_gather_buffer_refs = refs
    return cache, refs


def _acquire_param_gather_tensor(
    bucket,
    *,
    name: str,
    shape,
    dtype,
    device,
    route_key,
    zero: bool = False,
) -> torch.Tensor:
    shape = tuple(int(dim) for dim in shape)
    device = torch.device(device)
    key = (
        str(name),
        shape,
        dtype,
        _param_gather_device_key(device),
        tuple(route_key),
    )
    cache, refs = _param_gather_cache(bucket)
    tensors = cache.get(key)
    if tensors is None:
        tensors = []
        cache[key] = tensors
    for item in tensors:
        if item.active:
            continue
        tensor = item.tensor
        if tuple(tensor.shape) != shape or tensor.dtype != dtype or tensor.device != device:
            continue
        item.active = True
        refs[id(tensor)] = item
        if zero:
            tensor.zero_()
        return tensor

    tensor = torch.empty(shape, dtype=dtype, device=device)
    item = ParamGatherBuffer(tensor=tensor, active=True)
    tensors.append(item)
    refs[id(tensor)] = item
    if zero:
        tensor.zero_()
    return tensor


def _release_param_gather_tensor(bucket, tensor: torch.Tensor | None) -> None:
    if tensor is None:
        return
    refs = getattr(bucket, "_dion_param_gather_buffer_refs", None)
    if refs is None:
        return
    item = refs.get(id(tensor))
    if item is not None:
        item.active = False


def _group_ranks(group) -> Tuple[int, ...]:
    if group is None:
        if not dist.is_initialized():
            return (0,)
        return tuple(int(rank) for rank in range(dist.get_world_size()))
    return tuple(int(rank) for rank in dist.get_process_group_ranks(group))


def _groups_have_same_order(first_group, second_group) -> bool:
    if first_group is None or second_group is None:
        return first_group is second_group
    return _group_ranks(first_group) == _group_ranks(second_group)


def resolve_grad_rank_to_fs_rank(
    *,
    grad_group,
    fs_group,
    fs_size: int,
    bucket_id: int,
) -> Tuple[int, ...]:
    """Return the FS rank selected by each grad reduce-scatter output rank."""
    fs_size = int(fs_size)
    if fs_size <= 0:
        raise RuntimeError(f"[Dion] invalid FS size for bucket={bucket_id}: {fs_size}")

    grad_group_ranks = _group_ranks(grad_group)
    grad_group_size = len(grad_group_ranks)
    if grad_group_size <= 0:
        raise RuntimeError(f"[Dion] empty grad sync group for bucket={bucket_id}")
    if grad_group_size % fs_size != 0:
        raise RuntimeError(
            "[Dion] grad sync group is incompatible with FS topology "
            f"bucket={bucket_id} grad_group_size={grad_group_size} fs_size={fs_size} "
            f"grad_group_ranks={grad_group_ranks}"
        )
    if fs_size == 1:
        return tuple(0 for _ in grad_group_ranks)
    if fs_group is None:
        raise RuntimeError(
            "[Dion] missing FS group for multi-rank Dion grad sync "
            f"bucket={bucket_id} grad_group_ranks={grad_group_ranks} fs_size={fs_size}"
        )

    fs_group_ranks = _group_ranks(fs_group)
    if len(fs_group_ranks) != fs_size:
        raise RuntimeError(
            "[Dion] FS group size mismatch for Dion grad sync "
            f"bucket={bucket_id} fs_size={fs_size} fs_group_ranks={fs_group_ranks}"
        )

    grad_rank_by_global = {rank: rank_idx for rank_idx, rank in enumerate(grad_group_ranks)}
    missing_fs_ranks = tuple(rank for rank in fs_group_ranks if rank not in grad_rank_by_global)
    if missing_fs_ranks:
        raise RuntimeError(
            "[Dion] FS group must be contained in the grad sync group "
            f"bucket={bucket_id} missing_fs_ranks={missing_fs_ranks} "
            f"fs_group_ranks={fs_group_ranks} grad_group_ranks={grad_group_ranks}"
        )

    fs_rank_by_global = {rank: fs_rank for fs_rank, rank in enumerate(fs_group_ranks)}
    if all(rank in fs_rank_by_global for rank in grad_group_ranks):
        return tuple(int(fs_rank_by_global[rank]) for rank in grad_group_ranks)

    same_fs_rank_count = grad_group_size // fs_size
    fs_positions = tuple(int(grad_rank_by_global[rank]) for rank in fs_group_ranks)
    if len(set(fs_positions)) != fs_size:
        raise RuntimeError(
            "[Dion] duplicate FS rank position in Dion grad sync group "
            f"bucket={bucket_id} fs_group_ranks={fs_group_ranks} "
            f"grad_group_ranks={grad_group_ranks} fs_positions={fs_positions}"
        )

    fs_stride = int(fs_positions[1]) - int(fs_positions[0])
    if fs_stride <= 0:
        raise RuntimeError(
            "[Dion] cannot prove FS rank order from authoritative grad/FS groups "
            f"bucket={bucket_id} fs_group_ranks={fs_group_ranks} "
            f"grad_group_ranks={grad_group_ranks} fs_positions={fs_positions} "
            f"same_fs_rank_count={same_fs_rank_count}"
        )
    fs_position_offset = int(fs_positions[0])
    expected_fs_positions = tuple(
        fs_position_offset + fs_rank * fs_stride for fs_rank in range(fs_size)
    )
    fs_cycle = int(fs_stride) * int(fs_size)
    cycle_offset = fs_position_offset % fs_cycle
    if (
        fs_positions != expected_fs_positions
        or cycle_offset >= fs_stride
        or grad_group_size % fs_cycle != 0
    ):
        raise RuntimeError(
            "[Dion] cannot prove FS rank order from authoritative grad/FS groups "
            f"bucket={bucket_id} fs_group_ranks={fs_group_ranks} "
            f"grad_group_ranks={grad_group_ranks} fs_positions={fs_positions} "
            f"same_fs_rank_count={same_fs_rank_count}"
        )
    fs_cycle_start = fs_position_offset - (fs_position_offset % fs_stride)
    grad_rank_to_fs_rank = [
        int(((grad_rank - fs_cycle_start) // fs_stride) % fs_size)
        for grad_rank in range(grad_group_size)
    ]

    if any(fs_rank < 0 for fs_rank in grad_rank_to_fs_rank):
        raise RuntimeError(
            "[Dion] incomplete FS rank mapping for Dion grad sync "
            f"bucket={bucket_id} grad_rank_to_fs_rank={tuple(grad_rank_to_fs_rank)} "
            f"grad_group_ranks={grad_group_ranks}"
        )
    return tuple(int(fs_rank) for fs_rank in grad_rank_to_fs_rank)


def build_dion_shard_entries(
    *,
    bucket,
    param_map,
    dion_info_by_param,
    fs_size: int,
    fs_rank: int,
    grad_shard_group_size: int,
    grad_rank_to_fs_rank: Tuple[int, ...],
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
    if len(grad_rank_to_fs_rank) != int(grad_shard_group_size):
        raise RuntimeError(
            "[Dion] Dion grad rank mapping size mismatch "
            f"grad_group={grad_shard_group_size} mapping={tuple(grad_rank_to_fs_rank)}"
        )
    for grad_rank, grad_fs_rank in enumerate(grad_rank_to_fs_rank):
        if int(grad_fs_rank) < 0 or int(grad_fs_rank) >= int(fs_size):
            raise RuntimeError(
                "[Dion] Dion grad rank maps outside FS range "
                f"grad_rank={grad_rank} fs_rank={int(grad_fs_rank)} fs_size={fs_size}"
            )
    entries: List[DionShardEntry] = []
    dion_shard_layout_by_param = {}
    shard_offset = 0

    for param in param_map.keys():
        if not getattr(param, "is_dion_param", False):
            continue

        static_info = dion_info_by_param.get(param)
        if static_info is None:
            continue

        global_shape = tuple(static_info["global_shape"])
        fs_shard_dim = int(static_info["fs_shard_dim"])
        m_local, n_local = tuple(int(dim) for dim in param.shape)

        split_size = m_local if fs_shard_dim == 0 else n_local
        start_idx, end_idx = compute_fs_shard_range(split_size, fs_size, fs_rank)
        local_split_size = int(end_idx) - int(start_idx)
        if local_split_size <= 0:
            raise RuntimeError(
                "[DION_EMPTY_FS_SHARD] "
                f"param={getattr(param, '_param_name', f'id_{id(param)}')} "
                f"global_shape={global_shape} fs_shard_dim={fs_shard_dim} "
                f"fs_size={fs_size} fs_rank={fs_rank} range=({start_idx}, {end_idx})"
            )
        size_per_rank = math.ceil(split_size / fs_size)

        if fs_shard_dim == 0:
            local_shape = (local_split_size, n_local)
            shard_capacity = size_per_rank * n_local
        else:
            local_shape = (m_local, local_split_size)
            shard_capacity = m_local * size_per_rank

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
                    m=m_local,
                    n=n_local,
                    fs_shard_dim=fs_shard_dim,
                    start_idx=rank_start,
                    end_idx=rank_end,
                )
            )
        grad_rank_flat_segments = []
        for grad_rank in range(grad_shard_group_size):
            grad_fs_rank = int(grad_rank_to_fs_rank[grad_rank])
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
            DionShardEntry(
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


def bucket_rank_range(full_start: int, full_end: int, shard_size: int, rank: int) -> tuple[int, int]:
    """Return the intersection of one full bucket span with one stock local shard."""
    shard_abs_start = rank * shard_size
    shard_abs_end = shard_abs_start + shard_size
    local_abs_start = max(int(full_start), shard_abs_start)
    local_abs_end = min(int(full_end), shard_abs_end)
    if local_abs_end <= local_abs_start:
        return 0, 0
    return local_abs_start - shard_abs_start, local_abs_end - shard_abs_start


def param_rank_range(full_start: int, full_end: int, shard_size: int, rank: int) -> tuple[int, int]:
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


def serialize_bucket_gather_layout(dion_layout: DionBucketLayout) -> tuple[int, ...]:
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


def validate_bucket_gather_layout(
    *,
    bucket,
    dion_layout: DionBucketLayout,
    shard_group,
) -> None:
    """Validate that all shard ranks agree on one Dion bucket gather layout."""
    if shard_group is None:
        return
    shard_group_size = int(dist.get_world_size(shard_group))
    if shard_group_size <= 1:
        return

    local_layout = serialize_bucket_gather_layout(dion_layout)
    gathered_layouts = [None for _ in range(shard_group_size)]
    dist.all_gather_object(gathered_layouts, local_layout, group=shard_group)
    mismatched_ranks = [
        rank for rank, gathered_layout in enumerate(gathered_layouts) if gathered_layout != local_layout
    ]
    if mismatched_ranks:
        raise RuntimeError(
            "[DION_BUCKET_GATHER_LAYOUT_MISMATCH] "
            f"rank={dist.get_rank()} bucket={getattr(bucket, 'bucket_id', -1)} "
            f"group_rank={dist.get_rank(shard_group)} mismatched_ranks={tuple(mismatched_ranks)} "
            f"group_ranks={tuple(dist.get_process_group_ranks(shard_group))}"
        )


def write_dion_shard_(
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


def restore_dion_shards_from_bucket_(
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
        write_dion_shard_(
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
        m_local, n_local = (int(dim) for dim in full_view_2d.shape)
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
                local_numel = local_split_size * n_local
                local_source = rank_source[:local_numel].view(local_split_size, n_local)
                full_view_2d[rank_start:rank_end, :].copy_(local_source)
            else:
                local_numel = m_local * local_split_size
                local_source = rank_source[:local_numel].view(m_local, local_split_size)
                full_view_2d[:, rank_start:rank_end].copy_(local_source)
        write_dion_shard_(
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

    def __init__(self, standard_handle, dion_handle):
        self._standard_handle = standard_handle
        self._dion_handle = dion_handle

    def wait(self):
        if self._standard_handle is not None:
            self._standard_handle.wait()
        if self._dion_handle is not None:
            self._dion_handle.wait()
        self._standard_handle = None
        self._dion_handle = None


def assert_shard_aliased(
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


def mark_dion_bucket_params(cls, param_map, param_to_name, fs_size):
    """Classify bucket params and build static Dion metadata once."""
    del cls, fs_size
    from .. import parallel_state

    dion_param_count = 0
    dion_info_by_param = {}

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
        m_local, n_local = param.shape
        fs_shard_dim = get_fs_split_dim(tp_shard_dim)

        if tp_shard_dim == 0:
            m_global = m_local * tp_world_size
            n_global = n_local
        elif tp_shard_dim == 1:
            m_global = m_local
            n_global = n_local * tp_world_size
        else:
            m_global = m_local
            n_global = n_local

        num_local_experts = getattr(param, "num_local_experts", None)
        is_explicit_expert = has_explicit_expert_index(param_name)
        if num_local_experts is not None and num_local_experts > 1 and not is_explicit_expert:
            if tp_shard_dim == 0:
                per_expert_global_shape = (m_global // num_local_experts, n_global)
            elif tp_shard_dim == 1:
                per_expert_global_shape = (m_global, n_global // num_local_experts)
            else:
                per_expert_global_shape = (m_global, n_global)
        else:
            per_expert_global_shape = None

        dion_info_by_param[param] = {
            "is_dion": True,
            "global_shape": (m_global, n_global),
            "fs_shard_dim": fs_shard_dim,
            "tp_shard_dim": tp_shard_dim,
            "per_expert_global_shape": per_expert_global_shape,
        }

    return dion_param_count, dion_info_by_param


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
    if bucket is None or not getattr(bucket, "_tracks_dion_param_views", False):
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


def attach_dion_bucket_layout_(
    optimizer,
    *,
    gbuf_idx: int,
    buffer,
    bucket,
    dion_layout: DionBucketLayout,
) -> None:
    """Validate that stored Dion entries already target the current runtime bucket."""
    if not hasattr(bucket, "param_to_index") or bucket.param_to_index is None:
        raise RuntimeError(
            f"[Dion] buffer={gbuf_idx} bucket={bucket.bucket_id} missing bucket.param_to_index"
        )

    for entry in dion_layout.entries:
        entry_param = entry.param
        if entry_param not in bucket.param_to_index:
            buffer_name_map = getattr(buffer, "param_to_name", None)
            param_name = optimizer._lookup_param_name(buffer_name_map, entry_param)
            raise RuntimeError(
                f"[Dion] buffer={gbuf_idx} bucket={bucket.bucket_id} stored param "
                f"{param_name or f'id_{id(entry_param)}'} "
                "is not present in the current runtime bucket"
            )
    bucket.dion_layout = dion_layout
    for entry in dion_layout.entries:
        optimizer._dion_buckets_by_param[entry.param] = bucket
        optimizer._dion_entries_by_param[entry.param] = entry


def init_dion_bucket(
    optimizer,
    *,
    gbuf_idx: int,
    buffer,
    bucket,
    dion_layout: DionBucketLayout,
    fs_group,
) -> None:
    """Configure one bucket that contains at least one Dion param."""
    attach_dion_bucket_layout_(
        optimizer,
        gbuf_idx=gbuf_idx,
        buffer=buffer,
        bucket=bucket,
        dion_layout=dion_layout,
    )
    validate_bucket_gather_layout(
        bucket=bucket,
        dion_layout=dion_layout,
        shard_group=fs_group,
    )
    optimizer._init_bucket_comm(bucket, fs_group)
    bucket.dion_optimizer = optimizer
    bucket._tracks_dion_param_views = True
    bucket._dion_full_param_ready = True
    set_bucket_param_views(optimizer, bucket, copy_data=True)


def init_standard_bucket(optimizer, *, gbuf_idx: int, buffer, bucket, fs_group) -> None:
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
            f"[Dion] Buffer {gbuf_idx} Bucket {bucket.bucket_id} has Dion params but no Dion layout. params={dion_params}"
        )
        raise RuntimeError(
            f"[Dion] Buffer {gbuf_idx} Bucket {bucket.bucket_id} has Dion params but no Dion layout"
        )

    bucket.dion_layout = None
    optimizer._init_bucket_comm(bucket, fs_group)
    bucket.dion_optimizer = optimizer
    bucket._tracks_dion_param_views = False
    bucket._dion_full_param_ready = True


def bucket_dion_param_view(optimizer, bucket, entry: DionShardEntry) -> torch.Tensor:
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
    entry: DionShardEntry,
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
    shard_buffer[:local_numel].copy_(canonical_local_source.reshape(-1)[:local_numel])
    if local_numel < shard_buffer.numel():
        shard_buffer[local_numel:].zero_()


def prepare_dion_param_gather(optimizer, bucket) -> Tuple[torch.Tensor, List[Tuple[DionShardEntry, torch.Tensor]]]:
    """Build one bucket-local Dion shard buffer in canonical entry order."""
    dion_layout = getattr(bucket, "dion_layout", None)
    if dion_layout is None or not dion_layout.has_params:
        return (
            torch.empty(
                0,
                dtype=bucket.param_data.dtype,
                device=bucket.param_data.device,
            ),
            [],
        )

    prepared_buffer = _acquire_param_gather_tensor(
        bucket,
        name="dion_input",
        shape=(int(dion_layout.shard_size),),
        dtype=bucket.param_data.dtype,
        device=bucket.param_data.device,
        route_key=(
            int(dion_layout.entry_count),
            int(dion_layout.shard_size),
            int(dion_layout.max_shard_capacity),
        ),
    )
    prepared_entries = []
    for entry in dion_layout.entries:
        full_view_2d = bucket_dion_param_view(optimizer, bucket, entry)
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


def restore_bucket_param_data_(
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


def get_bucket_shard_group(optimizer, bucket):
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


def _build_standard_rank_segments(
    *,
    bucket,
    group_size: int,
    shard_size: int,
) -> tuple[Tuple[Tuple[int, int, int], ...], int]:
    dion_param_ids = bucket.dion_param_ids
    rank_segments = []
    max_rank_numel = 0
    for group_rank in range(int(group_size)):
        rank_start = int(group_rank) * int(shard_size)
        rank_end = rank_start + int(shard_size)
        cursor = 0
        segments = []
        for param in bucket.params_list:
            if id(param) in dion_param_ids:
                continue
            param_start, param_end = bucket.param_to_index[param]
            source_start = max(int(param_start), rank_start)
            source_end = min(int(param_end), rank_end)
            if source_end <= source_start:
                continue
            target_start = cursor
            cursor += source_end - source_start
            segments.append((int(source_start), int(source_end), int(target_start)))
        max_rank_numel = max(max_rank_numel, cursor)
        rank_segments.append(tuple(segments))
    return tuple(rank_segments), int(max_rank_numel)


def _get_standard_param_gather_route(
    *,
    bucket,
    group_size: int,
    group_rank: int,
    standard_shard_size: int,
) -> StandardParamGatherRoute:
    cached = getattr(bucket, "_standard_param_gather_route", None)
    if (
        cached is not None
        and int(cached.group_size) == int(group_size)
        and int(cached.group_rank) == int(group_rank)
        and int(cached.standard_shard_size) == int(standard_shard_size)
    ):
        return cached

    rank_segments, standard_numel = _build_standard_rank_segments(
        bucket=bucket,
        group_size=int(group_size),
        shard_size=int(standard_shard_size),
    )
    route = StandardParamGatherRoute(
        group_size=int(group_size),
        group_rank=int(group_rank),
        standard_shard_size=int(standard_shard_size),
        standard_numel=int(standard_numel),
        rank_segments=rank_segments,
    )
    bucket._standard_param_gather_route = route
    return route


def _standard_param_gather_route(optimizer, bucket):
    dion_layout = getattr(bucket, "dion_layout", None)
    if dion_layout is None or not bucket.has_standard_params:
        return None, None, None

    dp_group, dp_size, dp_rank = get_bucket_shard_group(optimizer, bucket)
    if dp_group is None:
        raise RuntimeError(
            "[Dion] mixed standard all-gather requires the standard local-shard group."
        )
    if dp_size == 1:
        return None, dp_group, dp_size

    if bucket.param_data.numel() % dp_size != 0:
        raise RuntimeError(
            "[Dion] mixed standard all-gather requires a padded standard bucket "
            f"bucket={getattr(bucket, 'bucket_id', -1)} "
            f"bucket_numel={int(bucket.param_data.numel())} group_size={int(dp_size)}"
        )
    shard_size = bucket.param_data.numel() // dp_size
    if shard_size <= 0:
        return None, dp_group, dp_size

    route = _get_standard_param_gather_route(
        bucket=bucket,
        group_size=dp_size,
        group_rank=dp_rank,
        standard_shard_size=shard_size,
    )
    return route, dp_group, dp_size


def fill_standard_param_gather(bucket, target: torch.Tensor, route: StandardParamGatherRoute) -> None:
    target.zero_()
    for source_start, source_end, target_start in route.rank_segments[int(route.group_rank)]:
        source_start = int(source_start)
        source_end = int(source_end)
        target_start = int(target_start)
        target_end = target_start + source_end - source_start
        target[target_start:target_end].copy_(
            bucket.param_data.view(-1)[source_start:source_end]
        )


def prepare_standard_param_gather(optimizer, bucket):
    """Build the minimal standard all-gather input for a mixed bucket."""
    route, dp_group, _ = _standard_param_gather_route(optimizer, bucket)
    if route is None or int(route.standard_numel) <= 0:
        return None, route, dp_group

    input_shard = _acquire_param_gather_tensor(
        bucket,
        name="standard_input",
        shape=(int(route.standard_numel),),
        dtype=bucket.param_data.dtype,
        device=bucket.param_data.device,
        route_key=(
            int(route.group_size),
            int(route.group_rank),
            int(route.standard_shard_size),
            int(route.standard_numel),
        ),
    )
    fill_standard_param_gather(bucket, input_shard, route)
    return input_shard, route, dp_group


def restore_standard_param_gather_(bucket, gathered_buffer: torch.Tensor, route) -> None:
    """Restore gathered mixed-bucket standard params into canonical bucket.param_data."""
    if route is None or int(route.standard_numel) <= 0:
        return
    expected_shape = (int(route.group_size), int(route.standard_numel))
    if gathered_buffer.dim() == 1:
        gathered_view = gathered_buffer.view(*expected_shape)
    elif tuple(gathered_buffer.shape) == expected_shape:
        gathered_view = gathered_buffer
    else:
        raise RuntimeError(
            "[Dion] mixed standard all-gather output shape mismatch "
            f"bucket={getattr(bucket, 'bucket_id', -1)} "
            f"shape={tuple(gathered_buffer.shape)} expected={expected_shape}"
        )
    flat_param_data = bucket.param_data.view(-1)
    for group_rank, segments in enumerate(route.rank_segments):
        rank_source = gathered_view[int(group_rank)]
        for source_start, source_end, target_start in segments:
            source_start = int(source_start)
            source_end = int(source_end)
            target_start = int(target_start)
            target_end = target_start + source_end - source_start
            flat_param_data[source_start:source_end].copy_(rank_source[target_start:target_end])


def all_gather_standard_params_(optimizer, bucket, async_op=False, prepared_gather=None):
    """Gather mixed-bucket standard params back into canonical bucket.param_data."""
    if prepared_gather is None:
        input_shard, route, dp_group = prepare_standard_param_gather(optimizer, bucket)
    else:
        input_shard, route, dp_group = prepared_gather

    if input_shard is None or route is None or int(route.standard_numel) <= 0:
        return None

    gathered_buffer = _acquire_param_gather_tensor(
        bucket,
        name="standard_output",
        shape=(int(route.group_size) * int(route.standard_numel),),
        dtype=bucket.param_data.dtype,
        device=bucket.param_data.device,
        route_key=(
            int(route.group_size),
            int(route.standard_shard_size),
            int(route.standard_numel),
        ),
    )
    try:
        work = dist.all_gather_into_tensor(
            output_tensor=gathered_buffer,
            input_tensor=input_shard,
            group=dp_group,
            async_op=async_op,
        )
    except Exception:
        _release_param_gather_tensor(bucket, input_shard)
        _release_param_gather_tensor(bucket, gathered_buffer)
        raise

    def _restore_callback(_input_shard=input_shard, _gathered_buffer=gathered_buffer):
        try:
            restore_standard_param_gather_(bucket, _gathered_buffer, route)
        finally:
            _release_param_gather_tensor(bucket, _input_shard)
            _release_param_gather_tensor(bucket, _gathered_buffer)

    if async_op:
        return CallbackHandle(work, _restore_callback)

    _restore_callback()
    return None


def all_gather_dion_params_(
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
        prepared_buffer, prepared_entries = prepare_dion_param_gather(optimizer, bucket)
    else:
        prepared_buffer, prepared_entries = prepared_gather

    if shard_group_size == 1:
        def _restore_callback(_prepared_buffer=prepared_buffer):
            try:
                restore_bucket_param_data_(
                    optimizer,
                    bucket=bucket,
                    prepared_entries=prepared_entries,
                    gathered_buffer=_prepared_buffer.view(1, -1),
                    shard_group_size=1,
                )
            finally:
                _release_param_gather_tensor(bucket, _prepared_buffer)

        if async_op:
            return CallbackHandle(None, _restore_callback)

        _restore_callback()
        return None
    if shard_group is None:
        _release_param_gather_tensor(bucket, prepared_buffer)
        return None

    bucket_shard_size = int(dion_layout.shard_size)
    expected_gathered_numel = int(dion_layout.gathered_numel)
    if expected_gathered_numel != bucket_shard_size * shard_group_size:
        _release_param_gather_tensor(bucket, prepared_buffer)
        raise RuntimeError(
            "[Dion] FS gather shard size invariant mismatch "
            f"bucket={getattr(bucket, 'bucket_id', -1)} "
            f"shard_size={bucket_shard_size} fs_size={shard_group_size} "
            f"expected_total={expected_gathered_numel}"
        )

    gathered_buffer = _acquire_param_gather_tensor(
        bucket,
        name="dion_output",
        shape=(expected_gathered_numel,),
        dtype=bucket.param_data.dtype,
        device=bucket.param_data.device,
        route_key=(
            int(shard_group_size),
            int(bucket_shard_size),
            int(dion_layout.entry_count),
        ),
    )
    try:
        work = dist.all_gather_into_tensor(
            output_tensor=gathered_buffer,
            input_tensor=prepared_buffer,
            group=shard_group,
            async_op=async_op,
        )
    except Exception:
        _release_param_gather_tensor(bucket, prepared_buffer)
        _release_param_gather_tensor(bucket, gathered_buffer)
        raise

    def _restore_callback(_prepared_buffer=prepared_buffer, _gathered_buffer=gathered_buffer):
        try:
            restore_bucket_param_data_(
                optimizer,
                bucket=bucket,
                prepared_entries=prepared_entries,
                gathered_buffer=_gathered_buffer.view(shard_group_size, bucket_shard_size),
                shard_group_size=shard_group_size,
            )
        finally:
            _release_param_gather_tensor(bucket, _prepared_buffer)
            _release_param_gather_tensor(bucket, _gathered_buffer)

    if async_op:
        return CallbackHandle(work, _restore_callback)

    _restore_callback()
    return None


def prepare_mixed_param_gather(
    optimizer,
    bucket,
    *,
    dion_layout: DionBucketLayout,
    standard_route: StandardParamGatherRoute,
):
    group_size = int(standard_route.group_size)
    dion_size = int(dion_layout.shard_size)
    standard_numel = int(standard_route.standard_numel)
    payload_size = dion_size + standard_numel
    input_payload = _acquire_param_gather_tensor(
        bucket,
        name="mixed_input",
        shape=(payload_size,),
        dtype=bucket.param_data.dtype,
        device=bucket.param_data.device,
        route_key=(int(group_size), int(dion_size), int(standard_numel)),
    )
    prepared_entries = []
    try:
        for entry in dion_layout.entries:
            full_view_2d = bucket_dion_param_view(optimizer, bucket, entry)
            shard_start = int(entry.shard_offset)
            shard_end = shard_start + int(entry.shard_capacity)
            fill_dion_shard_buffer(
                optimizer,
                entry=entry,
                full_view_2d=full_view_2d,
                shard_buffer=input_payload[shard_start:shard_end],
            )
            prepared_entries.append((entry, full_view_2d))
        fill_standard_param_gather(
            bucket,
            input_payload[dion_size:payload_size],
            standard_route,
        )
    except Exception:
        _release_param_gather_tensor(bucket, input_payload)
        raise
    return input_payload, prepared_entries


def _all_gather_mixed_payload_(
    optimizer,
    bucket,
    *,
    async_op: bool,
    input_payload: torch.Tensor,
    prepared_entries,
    standard_route: StandardParamGatherRoute,
    dp_group,
    group_size: int,
    dion_size: int,
    standard_numel: int,
):
    dion_end = int(dion_size)
    payload_size = dion_end + int(standard_numel)
    gathered_buffer = _acquire_param_gather_tensor(
        bucket,
        name="mixed_output",
        shape=(int(group_size) * payload_size,),
        dtype=bucket.param_data.dtype,
        device=bucket.param_data.device,
        route_key=(int(group_size), int(dion_size), int(standard_numel)),
    )
    try:
        work = dist.all_gather_into_tensor(
            output_tensor=gathered_buffer,
            input_tensor=input_payload,
            group=dp_group,
            async_op=async_op,
        )
    except Exception:
        _release_param_gather_tensor(bucket, input_payload)
        _release_param_gather_tensor(bucket, gathered_buffer)
        raise

    def _restore_callback(_input_payload=input_payload, _gathered_buffer=gathered_buffer):
        try:
            gathered_view = _gathered_buffer.view(int(group_size), payload_size)
            restore_bucket_param_data_(
                optimizer,
                bucket=bucket,
                prepared_entries=prepared_entries,
                gathered_buffer=gathered_view[:, :dion_end],
                shard_group_size=int(group_size),
            )
            restore_standard_param_gather_(
                bucket,
                gathered_view[:, dion_end:payload_size],
                standard_route,
            )
        finally:
            _release_param_gather_tensor(bucket, _input_payload)
            _release_param_gather_tensor(bucket, _gathered_buffer)

    if async_op:
        return CallbackHandle(work, _restore_callback)

    _restore_callback()
    return None


def all_gather_bucket_params_(optimizer, bucket, async_op=False):
    """Gather all Dion bucket params that do not follow the pure standard DO path."""
    dion_layout = getattr(bucket, "dion_layout", None)
    has_dion = dion_layout is not None and dion_layout.has_params
    has_mixed_standard_params = has_dion and bucket.has_standard_params

    prepared_dion_entries = None
    if has_mixed_standard_params:
        standard_route, dp_group, _ = _standard_param_gather_route(optimizer, bucket)
        shard_group = getattr(bucket, "dion_shard_group", None)
        if shard_group is None:
            shard_group = optimizer.fs_group
        shard_group_size = optimizer._group_size(shard_group) if shard_group is not None else 1
        if (
            standard_route is not None
            and int(standard_route.standard_numel) > 0
            and int(standard_route.group_size) == int(shard_group_size)
            and _groups_have_same_order(dp_group, shard_group)
        ):
            dion_size = int(dion_layout.shard_size)
            standard_numel = int(standard_route.standard_numel)
            expected_gathered_numel = int(dion_layout.gathered_numel)
            if expected_gathered_numel != dion_size * int(standard_route.group_size):
                raise RuntimeError(
                    "[Dion] combined mixed-bucket gather size invariant mismatch "
                    f"bucket={getattr(bucket, 'bucket_id', -1)} "
                    f"shard_size={dion_size} group_size={int(standard_route.group_size)} "
                    f"expected_total={expected_gathered_numel}"
                )
            input_payload, prepared_entries = prepare_mixed_param_gather(
                optimizer,
                bucket,
                dion_layout=dion_layout,
                standard_route=standard_route,
            )
            return _all_gather_mixed_payload_(
                optimizer,
                bucket,
                async_op=async_op,
                input_payload=input_payload,
                prepared_entries=prepared_entries,
                standard_route=standard_route,
                dp_group=dp_group,
                group_size=int(standard_route.group_size),
                dion_size=dion_size,
                standard_numel=standard_numel,
            )

        prepared_dion_entries = prepare_dion_param_gather(optimizer, bucket)
        prepared_standard_gather = prepare_standard_param_gather(optimizer, bucket)
        standard_handle = all_gather_standard_params_(
            optimizer,
            bucket,
            async_op=async_op,
            prepared_gather=prepared_standard_gather,
        )
        dion_handle = all_gather_dion_params_(
            optimizer,
            bucket,
            async_op=async_op,
            prepared_gather=prepared_dion_entries,
        )
        if async_op and (standard_handle is not None or dion_handle is not None):
            return BucketGatherHandle(standard_handle, dion_handle)
        return None

    handles = []
    dion_handle = all_gather_dion_params_(
        optimizer,
        bucket,
        async_op=async_op,
    )
    if dion_handle is not None:
        handles.append(dion_handle)
    standard_handle = all_gather_standard_params_(optimizer, bucket, async_op=async_op)
    if standard_handle is not None:
        handles.append(standard_handle)
    if async_op and handles:
        return _HandleGroup(handles)
    return None
