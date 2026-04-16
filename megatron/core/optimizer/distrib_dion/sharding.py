"""TP/FS sharding helpers for distributed Dion."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch
import torch.distributed as dist

from ..dion.state import p_is_fs_sharded, p_is_tp_sharded


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DionShardLayout:
    """Minimal Dion shard layout for one logical model parameter."""

    local_shape: Tuple[int, int]
    global_shape: Tuple[int, int]
    fs_shard_dim: int
    start_idx: int
    end_idx: int
    per_expert_global_shape: Optional[Tuple[int, int]] = None

    @property
    def local_numel(self) -> int:
        return int(self.local_shape[0]) * int(self.local_shape[1])


def get_tp_split_dim(param: torch.Tensor) -> int:
    """Return TP split dim in Dion naming convention."""
    return getattr(param, "partition_dim", -1)


def is_tp_enabled(param: torch.Tensor) -> bool:
    """Return True iff this parameter participates in Tensor Parallelism."""
    return bool(getattr(param, "tensor_model_parallel", False))


def compute_fs_shard_range(global_size: int, fs_size: int, fs_rank: int) -> Tuple[int, int]:
    """Compute the canonical FS shard range for one rank."""
    if fs_size <= 0:
        raise ValueError(f"fs_size must be positive, got {fs_size}")
    if fs_rank < 0 or fs_rank >= fs_size:
        raise ValueError(f"fs_rank must be in [0, {fs_size}), got {fs_rank}")

    size_per_rank = global_size // fs_size
    remainder = global_size % fs_size

    if fs_rank < remainder:
        start_idx = fs_rank * (size_per_rank + 1)
        end_idx = start_idx + size_per_rank + 1
    else:
        start_idx = remainder * (size_per_rank + 1) + (fs_rank - remainder) * size_per_rank
        end_idx = start_idx + size_per_rank

    return start_idx, end_idx


def get_fs_split_dim(tp_shard_dim: int) -> int:
    """Return the FS split dim orthogonal to TP."""
    if tp_shard_dim == 0:
        return 1
    if tp_shard_dim == 1:
        return 0
    return 0


def compute_local_shape(
    m: int,
    n: int,
    start_idx: int,
    end_idx: int,
    fs_shard_dim: int,
) -> Tuple[int, int]:
    """Return local (m, n) shape after FS sharding."""
    local_split_size = end_idx - start_idx
    if fs_shard_dim == 0:
        return (local_split_size, n)
    return (m, local_split_size)


def slice_fs_shard_2d(
    tensor2d: torch.Tensor,
    fs_shard_dim: int,
    start_idx: int,
    end_idx: int,
) -> torch.Tensor:
    """Return a view for the FS shard slice of a 2D tensor."""
    if fs_shard_dim == 0:
        return tensor2d[start_idx:end_idx, :]
    return tensor2d[:, start_idx:end_idx]


def write_fs_shard_2d(
    dst2d: torch.Tensor,
    fs_shard_dim: int,
    start_idx: int,
    end_idx: int,
    src2d: torch.Tensor,
) -> None:
    """In-place write `src2d` into `dst2d` at the FS shard slice."""
    if fs_shard_dim == 0:
        dst2d[start_idx:end_idx, :].copy_(src2d)
    else:
        dst2d[:, start_idx:end_idx].copy_(src2d)


def compute_fs_flat_segments(
    *,
    full_start: int,
    m: int,
    n: int,
    fs_shard_dim: int,
    start_idx: int,
    end_idx: int,
) -> Tuple[Tuple[int, int], ...]:
    """Return canonical flat-buffer segments for one FS shard of a 2D parameter."""
    if start_idx < 0 or end_idx < start_idx:
        raise ValueError(f"invalid FS shard range: start_idx={start_idx} end_idx={end_idx}")
    if fs_shard_dim == 0:
        return ((full_start + start_idx * n, full_start + end_idx * n),)

    width = end_idx - start_idx
    segments = []
    for row_idx in range(m):
        row_base = full_start + row_idx * n
        segments.append((row_base + start_idx, row_base + start_idx + width))
    return tuple(segments)


def resolve_tp_group(
    dist_meta,
    *,
    require_in_distributed: bool,
    group_size_fn: Callable,
) -> Optional[torch.distributed.ProcessGroup]:
    """Return the authoritative TP group from adapter-owned metadata."""
    param_name = getattr(dist_meta, "param_name", "") if dist_meta is not None else ""
    param_uid = getattr(dist_meta, "param_uid", None) if dist_meta is not None else None
    tp_group = getattr(dist_meta, "tp_group", None) if dist_meta is not None else None
    meta_tp_world_size = int(getattr(dist_meta, "tp_world_size", 1)) if dist_meta is not None else 1
    meta_tp_rank = int(getattr(dist_meta, "tp_rank", -1)) if dist_meta is not None else -1

    if tp_group is None:
        if require_in_distributed and meta_tp_world_size > 1:
            raise RuntimeError(
                "[DION_MISSING_TP_GROUP_META] "
                f"rank={dist.get_rank()} param={param_name} param_uid={param_uid} "
                f"meta_tp_world_size={meta_tp_world_size} meta_tp_rank={meta_tp_rank}"
            )
        return None

    actual_tp_world_size = group_size_fn(tp_group)
    actual_tp_rank = dist.get_rank(tp_group)
    if actual_tp_world_size != meta_tp_world_size or actual_tp_rank != meta_tp_rank:
        raise RuntimeError(
            "[DION_INCONSISTENT_TP_GROUP_META] "
            f"rank={dist.get_rank()} param={param_name} param_uid={param_uid} "
            f"meta_tp_world_size={meta_tp_world_size} actual_tp_world_size={actual_tp_world_size} "
            f"meta_tp_rank={meta_tp_rank} actual_tp_rank={actual_tp_rank}"
        )
    if actual_tp_world_size <= 1:
        return None
    return tp_group


def resolve_fs_group(
    dist_meta,
    *,
    require_in_distributed: bool,
    group_size_fn: Callable,
) -> Optional[torch.distributed.ProcessGroup]:
    """Return the authoritative FS group from adapter-owned metadata."""
    param_name = getattr(dist_meta, "param_name", "") if dist_meta is not None else ""
    param_uid = getattr(dist_meta, "param_uid", None) if dist_meta is not None else None
    fs_group = getattr(dist_meta, "fs_group", None) if dist_meta is not None else None
    meta_fs_world_size = int(getattr(dist_meta, "fs_world_size", 1)) if dist_meta is not None else 1
    meta_fs_rank = int(getattr(dist_meta, "fs_rank", -1)) if dist_meta is not None else -1

    if fs_group is None:
        if require_in_distributed and meta_fs_world_size > 1:
            raise RuntimeError(
                "[DION_MISSING_FS_GROUP_META] "
                f"rank={dist.get_rank()} param={param_name} param_uid={param_uid} "
                f"meta_fs_world_size={meta_fs_world_size} meta_fs_rank={meta_fs_rank}"
            )
        return None

    actual_fs_world_size = group_size_fn(fs_group)
    actual_fs_rank = dist.get_rank(fs_group)
    if actual_fs_world_size != meta_fs_world_size or actual_fs_rank != meta_fs_rank:
        raise RuntimeError(
            "[DION_INCONSISTENT_FS_GROUP_META] "
            f"rank={dist.get_rank()} param={param_name} param_uid={param_uid} "
            f"meta_fs_world_size={meta_fs_world_size} actual_fs_world_size={actual_fs_world_size} "
            f"meta_fs_rank={meta_fs_rank} actual_fs_rank={actual_fs_rank}"
        )
    if actual_fs_world_size <= 1:
        return None
    return fs_group


def resolve_ortho_group(
    config,
    dist_meta,
    *,
    use_fs_collectives: bool,
    resolve_tp_group_fn: Callable,
    resolve_fs_group_fn: Callable,
):
    """Return the authoritative orthogonalization group from adapter metadata."""
    use_tp_shard = bool(getattr(config, "use_tp_shard", False))

    if p_is_tp_sharded(config, use_tp_shard=use_tp_shard):
        return resolve_tp_group_fn(dist_meta, require_in_distributed=True)
    if use_fs_collectives and p_is_fs_sharded(config):
        return resolve_fs_group_fn(dist_meta, require_in_distributed=True)
    return None


def create_fs_shard(optimizer, model_param, shard_layout: DionShardLayout):
    """Create the local FS shard view from `model_param`."""
    start_idx = int(shard_layout.start_idx)
    end_idx = int(shard_layout.end_idx)
    fs_shard_dim = int(shard_layout.fs_shard_dim)

    shard = slice_fs_shard_2d(model_param.detach(), fs_shard_dim, start_idx, end_idx)
    shard._model_param = model_param
    return shard


def prepare_fs_shard(optimizer, model_param, shard):
    """Attach FS shard to model_param for optimizer state."""
    shard_layout = optimizer._param_shard_layout(model_param)
    if shard_layout is not None:
        expected_view = slice_fs_shard_2d(
            model_param.data,
            int(shard_layout.fs_shard_dim),
            int(shard_layout.start_idx),
            int(shard_layout.end_idx),
        )
        if shard.shape != expected_view.shape or shard.data_ptr() != expected_view.data_ptr():
            param_name = getattr(model_param, '_param_name', f'id_{id(model_param)}')
            logger.error(
                "[DION_FS_ALIAS_MISMATCH] param=%s shard_shape=%s expected_shape=%s shard_ptr=%s expected_ptr=%s",
                param_name,
                tuple(shard.shape),
                tuple(expected_view.shape),
                shard.data_ptr(),
                expected_view.data_ptr(),
            )
    model_param._fs_shard = shard


def register_dion_shard(
    optimizer,
    model_param: torch.nn.Parameter,
    data_shard: torch.Tensor,
    opt_shard: torch.Tensor,
    shard_layout: DionShardLayout,
) -> None:
    """Register all shard info for a Dion parameter in one call."""
    optimizer._shards_by_param[model_param] = (data_shard, opt_shard)
    optimizer._shard_layouts_by_param[model_param] = shard_layout


def get_data_shard(optimizer, model_param: torch.nn.Parameter) -> Optional[torch.Tensor]:
    """Get data shard (FP16) for a model parameter."""
    info = optimizer._shards_by_param.get(model_param)
    return info[0] if info else None


def get_opt_shard(optimizer, model_param: torch.nn.Parameter) -> Optional[torch.Tensor]:
    """Get optimizer shard (FP32) for a model parameter."""
    info = optimizer._shards_by_param.get(model_param)
    return info[1] if info else None


def update_data_shard(
    optimizer, model_param: torch.nn.Parameter, new_data_shard: torch.Tensor
) -> None:
    """Update data shard for a model parameter."""
    info = optimizer._shards_by_param.get(model_param)
    if info is not None:
        _, opt_shard = info
        optimizer._shards_by_param[model_param] = (new_data_shard, opt_shard)


def update_opt_shard(
    optimizer, model_param: torch.nn.Parameter, new_opt_shard: torch.Tensor
) -> None:
    """Update optimizer shard for a model parameter."""
    info = optimizer._shards_by_param.get(model_param)
    if info is not None:
        data_shard, _ = info
        optimizer._shards_by_param[model_param] = (data_shard, new_opt_shard)


def param_shard_layout(optimizer, model_param: torch.nn.Parameter) -> Optional[DionShardLayout]:
    """Return typed Dion shard layout for one model param, if any."""
    return optimizer._shard_layouts_by_param.get(model_param)
