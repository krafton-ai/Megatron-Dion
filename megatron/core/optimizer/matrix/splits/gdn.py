"""GDN in-projection split helpers for matrix optimizers.

GatedDeltaNet stores ``[query, key, value, z, beta, alpha]`` in one
ColumnParallelLinear ``in_proj``.  Unlike grouped QKV, TP row sharding keeps
that order rank-locally, so helpers here treat split sizes as global child row
counts and project to rank-local rows when TP owns the row axis.
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import torch

from .axis import (
    axis_shape,
    axis_tensor,
    copy_split_axis,
    dist_meta_for_split_axis,
    get_split_axis_from_dist_meta,
    get_split_axis_from_param,
    get_split_axis_from_state,
    normalize_split_axis,
    original_shape,
    original_tensor,
    resolve_split_axis,
)


GDN_CHILD_KINDS: Tuple[str, str, str, str, str, str] = (
    "query",
    "key",
    "value",
    "z",
    "beta",
    "alpha",
)


def is_gdn_param(param: torch.Tensor) -> bool:
    """Return whether a param is tagged as fused GDN in-projection."""
    return bool(getattr(param, "is_gdn", False))


def _normalize_gdn_split_shapes(
    split_shapes,
    *,
    context: str,
) -> Tuple[int, int, int, int, int, int]:
    split_shapes = tuple(int(dim) for dim in split_shapes)
    if len(split_shapes) != 6 or any(dim <= 0 for dim in split_shapes):
        raise RuntimeError(
            "[MATRIX_INVALID_GDN_SPLIT_SHAPES] "
            f"context={context} split_shapes={split_shapes}"
        )
    if split_shapes[0] != split_shapes[1]:
        raise RuntimeError(
            "[MATRIX_INVALID_GDN_QK_SHAPE] "
            f"context={context} split_shapes={split_shapes}"
        )
    if split_shapes[2] != split_shapes[3]:
        raise RuntimeError(
            "[MATRIX_INVALID_GDN_VZ_SHAPE] "
            f"context={context} split_shapes={split_shapes}"
        )
    if split_shapes[4] != split_shapes[5]:
        raise RuntimeError(
            "[MATRIX_INVALID_GDN_BETA_ALPHA_SHAPE] "
            f"context={context} split_shapes={split_shapes}"
        )
    return split_shapes


def get_gdn_split_shapes(param: torch.Tensor) -> Tuple[int, int, int, int, int, int]:
    """Return global child row sizes from a tagged GDN in-projection parameter."""
    split_shapes = getattr(param, "gdn_split_shapes", None)
    if split_shapes is None:
        model_param = getattr(param, "_model_param", None)
        split_shapes = getattr(model_param, "gdn_split_shapes", None)
    if split_shapes is None:
        raise RuntimeError(
            "[MATRIX_GDN_SPLIT_SHAPES_MISSING] "
            f"param={getattr(param, '_param_name', '') or id(param)}"
        )
    return _normalize_gdn_split_shapes(
        split_shapes,
        context=f"param={getattr(param, '_param_name', '') or id(param)}",
    )


def get_gdn_split_axis(param: torch.Tensor) -> int:
    """Return the fused GDN split axis for a tagged parameter."""
    return get_split_axis_from_param(param, "gdn_split_axis", default=0)


def get_gdn_split_shapes_from_dist_meta(
    dist_meta,
) -> Optional[Tuple[int, int, int, int, int, int]]:
    """Return validated GDN split sizes from distributed metadata if present."""
    if dist_meta is None:
        return None
    split_shapes = getattr(dist_meta, "gdn_split_shapes", None)
    if split_shapes is None:
        return None
    return _normalize_gdn_split_shapes(
        split_shapes,
        context=(
            f"dist_meta.param_uid={getattr(dist_meta, 'param_uid', None)} "
            f"dist_meta.param_name={getattr(dist_meta, 'param_name', '')}"
        ),
    )


def get_gdn_split_axis_from_dist_meta(dist_meta) -> int:
    """Return the GDN split axis from distributed metadata."""
    return get_split_axis_from_dist_meta(dist_meta, "gdn_split_axis", default=0)


def get_gdn_split_shapes_from_state(
    optimizer_state: Optional[dict],
) -> Optional[Tuple[int, int, int, int, int, int]]:
    """Return validated GDN split sizes from persistent optimizer state if present."""
    if not optimizer_state or not bool(optimizer_state.get("gdn_split_gdn", False)):
        return None
    split_shapes = optimizer_state.get("gdn_split_shapes", None)
    if split_shapes is None:
        raise RuntimeError("[MATRIX_GDN_SPLIT_STATE_MISSING_SHAPES]")
    return _normalize_gdn_split_shapes(split_shapes, context="optimizer_state")


def get_gdn_split_axis_from_state(optimizer_state: Optional[dict]) -> int:
    """Return the GDN split axis from optimizer state."""
    return get_split_axis_from_state(optimizer_state, "gdn_split_axis", default=0)


def resolve_gdn_split_shapes(
    *,
    param: Optional[torch.Tensor] = None,
    optimizer_state: Optional[dict] = None,
    dist_meta=None,
) -> Optional[Tuple[int, int, int, int, int, int]]:
    """Resolve GDN split shapes from state, dist-meta, then param attrs."""
    split_shapes = get_gdn_split_shapes_from_state(optimizer_state)
    if split_shapes is not None:
        return split_shapes
    split_shapes = get_gdn_split_shapes_from_dist_meta(dist_meta)
    if split_shapes is not None:
        return split_shapes
    if param is None:
        return None
    try:
        return get_gdn_split_shapes(param)
    except RuntimeError:
        return None


def resolve_gdn_split_axis(
    *,
    param: Optional[torch.Tensor] = None,
    optimizer_state: Optional[dict] = None,
    dist_meta=None,
) -> int:
    """Resolve the GDN split axis from state, dist-meta, then param attrs."""
    return resolve_split_axis(
        attr="gdn_split_axis",
        param=param,
        optimizer_state=optimizer_state,
        dist_meta=dist_meta,
        default=0,
    )


def copy_gdn_split_metadata(destination_tensor: torch.Tensor, source_tensor: torch.Tensor) -> None:
    """Copy fused-GDN split metadata when the source tensor is tagged as GDN."""
    if not is_gdn_param(source_tensor) and not hasattr(source_tensor, "gdn_split_shapes"):
        return
    destination_tensor.is_gdn = True
    destination_tensor.gdn_split_shapes = get_gdn_split_shapes(source_tensor)
    copy_split_axis(destination_tensor, source_tensor, "gdn_split_axis")


def gdn_child_name(parent_name: str, child_kind: str) -> str:
    """Return the optimizer-only child name for one fused GDN child."""
    if child_kind not in GDN_CHILD_KINDS:
        raise RuntimeError(f"[MATRIX_INVALID_GDN_CHILD_KIND] child_kind={child_kind!r}")
    return f"{parent_name}::{child_kind}"


def gdn_state_key(prefix: str, child_kind: str) -> str:
    """Return a stable parent-state key for one GDN child field."""
    if child_kind not in GDN_CHILD_KINDS:
        raise RuntimeError(f"[MATRIX_INVALID_GDN_CHILD_KIND] child_kind={child_kind!r}")
    return f"gdn_{child_kind}_{prefix}"


def gdn_child_param_uid(parent_uid, child_kind: str):
    """Return a stable optimizer-only child identity derived from the parent uid."""
    if child_kind not in GDN_CHILD_KINDS:
        raise RuntimeError(f"[MATRIX_INVALID_GDN_CHILD_KIND] child_kind={child_kind!r}")
    if parent_uid is None:
        raise RuntimeError("[MATRIX_GDN_CHILD_UID_REQUIRES_PARENT_UID]")
    if isinstance(parent_uid, tuple):
        return (*parent_uid, ("gdn_child", child_kind))
    return (parent_uid, ("gdn_child", child_kind))


def _gdn_child_index(child_kind: str) -> int:
    try:
        return GDN_CHILD_KINDS.index(child_kind)
    except ValueError as exc:
        raise RuntimeError(
            f"[MATRIX_INVALID_GDN_CHILD_KIND] child_kind={child_kind!r}"
        ) from exc


def _split_range(size: int, world_size: int, rank: int) -> Tuple[int, int]:
    if world_size <= 0:
        raise RuntimeError(f"[MATRIX_INVALID_GDN_WORLD_SIZE] world_size={world_size}")
    if rank < 0 or rank >= world_size:
        raise RuntimeError(
            f"[MATRIX_INVALID_GDN_RANK] rank={rank} world_size={world_size}"
        )
    size_per_rank = int(size) // int(world_size)
    remainder = int(size) % int(world_size)
    if rank < remainder:
        start = rank * (size_per_rank + 1)
        end = start + size_per_rank + 1
    else:
        start = remainder * (size_per_rank + 1) + (rank - remainder) * size_per_rank
        end = start + size_per_rank
    return int(start), int(end)


def _tp_row_world_size(dist_meta) -> int:
    if dist_meta is None:
        return 1
    if int(getattr(dist_meta, "tp_shard_dim", -1)) == 0:
        return max(1, int(getattr(dist_meta, "tp_world_size", 1)))
    return 1


def _uses_rank_local_rows(dist_meta) -> bool:
    return _tp_row_world_size(dist_meta) > 1


def _validate_orthogonal_row_shards(dist_meta, *, context: str) -> None:
    if dist_meta is None:
        return
    fs_row = (
        int(getattr(dist_meta, "fs_shard_dim", -1)) == 0
        and int(getattr(dist_meta, "fs_world_size", 1)) > 1
    )
    tp_row = (
        int(getattr(dist_meta, "tp_shard_dim", -1)) == 0
        and int(getattr(dist_meta, "tp_world_size", 1)) > 1
    )
    if fs_row and tp_row:
        raise RuntimeError(
            "[MATRIX_GDN_NON_ORTHOGONAL_ROW_SHARDS] "
            f"context={context} "
            f"param_uid={getattr(dist_meta, 'param_uid', None)} "
            f"param_name={getattr(dist_meta, 'param_name', '')}"
        )


def _local_split_shapes(
    split_shapes: Tuple[int, int, int, int, int, int],
    dist_meta,
    *,
    context: str,
) -> Tuple[int, int, int, int, int, int]:
    tp_world_size = _tp_row_world_size(dist_meta)
    if tp_world_size <= 1:
        return split_shapes
    if any(int(dim) % tp_world_size != 0 for dim in split_shapes):
        raise RuntimeError(
            "[MATRIX_GDN_TP_SPLIT_SHAPES_NOT_DIVISIBLE] "
            f"context={context} split_shapes={split_shapes} tp_world_size={tp_world_size} "
            f"param_uid={getattr(dist_meta, 'param_uid', None)} "
            f"param_name={getattr(dist_meta, 'param_name', '')}"
        )
    return tuple(int(dim) // tp_world_size for dim in split_shapes)


def _global_row_count(dist_meta, *, context: str) -> int:
    parent_global_shape = getattr(dist_meta, "global_shape", None)
    if parent_global_shape is None or len(parent_global_shape) != 2:
        raise RuntimeError(
            "[MATRIX_GDN_MISSING_GLOBAL_SHAPE] "
            f"context={context} "
            f"param_uid={getattr(dist_meta, 'param_uid', None)} "
            f"param_name={getattr(dist_meta, 'param_name', '')}"
        )
    return int(parent_global_shape[0])


def validate_gdn_split_shapes_for_rows(
    split_shapes: Tuple[int, int, int, int, int, int],
    *,
    rows: int,
    context: str,
) -> None:
    """Validate that a global parent row count matches these GDN child sizes."""
    split_shapes = _normalize_gdn_split_shapes(split_shapes, context=context)
    if int(rows) != int(sum(split_shapes)):
        raise RuntimeError(
            "[MATRIX_GDN_LAYOUT_MISMATCH] "
            f"context={context} rows={rows} split_shapes={split_shapes}"
        )


def _parent_row_range(
    *,
    local_rows: int,
    split_shapes: Tuple[int, int, int, int, int, int],
    dist_meta,
    context: str,
) -> tuple[int, int, Tuple[int, int, int, int, int, int]]:
    """Return parent row interval and active split sizes in the local coordinate space."""
    if dist_meta is None:
        return 0, int(local_rows), split_shapes

    _validate_orthogonal_row_shards(dist_meta, context=context)

    if _uses_rank_local_rows(dist_meta):
        local_shapes = _local_split_shapes(split_shapes, dist_meta, context=context)
        if int(sum(local_shapes)) != int(local_rows):
            raise RuntimeError(
                "[MATRIX_GDN_LOCAL_TP_ROW_MISMATCH] "
                f"context={context} local_rows={local_rows} local_split_shapes={local_shapes} "
                f"split_shapes={split_shapes} param_uid={getattr(dist_meta, 'param_uid', None)} "
                f"param_name={getattr(dist_meta, 'param_name', '')}"
            )
        return 0, int(local_rows), local_shapes

    fs_shard_dim = int(getattr(dist_meta, "fs_shard_dim", -1))
    fs_world_size = int(getattr(dist_meta, "fs_world_size", 1))
    if fs_shard_dim == 0 and fs_world_size > 1:
        fs_rank = int(getattr(dist_meta, "fs_rank", -1))
        if fs_rank < 0:
            return -1, -1, split_shapes
        parent_row_start = int(getattr(dist_meta, "fs_start_idx", -1))
        parent_row_end = int(getattr(dist_meta, "fs_end_idx", -1))
        if parent_row_start < 0 or parent_row_end < parent_row_start:
            raise RuntimeError(
                "[MATRIX_GDN_MISSING_FS_RANGE] "
                f"context={context} "
                f"param_uid={getattr(dist_meta, 'param_uid', None)} "
                f"param_name={getattr(dist_meta, 'param_name', '')}"
            )
        if parent_row_end - parent_row_start != int(local_rows):
            raise RuntimeError(
                "[MATRIX_GDN_LOCAL_ROW_RANGE_MISMATCH] "
                f"context={context} local_rows={local_rows} "
                f"parent_row_range=({parent_row_start}, {parent_row_end})"
            )
        return parent_row_start, parent_row_end, split_shapes

    parent_global_rows = _global_row_count(dist_meta, context=context)
    if int(parent_global_rows) != int(sum(split_shapes)):
        raise RuntimeError(
            "[MATRIX_GDN_GLOBAL_LAYOUT_MISMATCH] "
            f"context={context} parent_global_rows={parent_global_rows} "
            f"split_shapes={split_shapes}"
        )
    if int(local_rows) != int(parent_global_rows):
        raise RuntimeError(
            "[MATRIX_GDN_LOCAL_ROW_RANGE_MISMATCH] "
            f"context={context} local_rows={local_rows} parent_global_rows={parent_global_rows}"
        )
    return 0, int(local_rows), split_shapes


def _child_segments(
    *,
    parent_row_start: int,
    parent_row_end: int,
    split_shapes: Tuple[int, ...],
    child_kind: str,
) -> list[tuple[int, int, int, int]]:
    if parent_row_start < 0 or parent_row_end <= parent_row_start:
        return []
    child_index = _gdn_child_index(child_kind)
    child_start = sum(int(split_shapes[idx]) for idx in range(child_index))
    child_end = child_start + int(split_shapes[child_index])
    overlap_start = max(int(parent_row_start), child_start)
    overlap_end = min(int(parent_row_end), child_end)
    if overlap_end <= overlap_start:
        return []
    child_local_start = int(overlap_start - child_start)
    child_local_end = int(overlap_end - child_start)
    source_start = int(overlap_start - parent_row_start)
    source_end = int(overlap_end - parent_row_start)
    return [(source_start, source_end, child_local_start, child_local_end)]


def _child_row_count(
    *,
    parent_row_start: int,
    parent_row_end: int,
    split_shapes: Tuple[int, ...],
    child_kind: str,
) -> int:
    return sum(
        int(source_end - source_start)
        for source_start, source_end, _, _ in _child_segments(
            parent_row_start=parent_row_start,
            parent_row_end=parent_row_end,
            split_shapes=split_shapes,
            child_kind=child_kind,
        )
    )


def gdn_child_row_range(
    *,
    parent_row_start: int,
    parent_row_end: int,
    split_shapes: Tuple[int, int, int, int, int, int],
    child_kind: str,
) -> Optional[Tuple[int, int]]:
    """Project a global contiguous GDN parent row interval to one child interval."""
    segments = _child_segments(
        parent_row_start=parent_row_start,
        parent_row_end=parent_row_end,
        split_shapes=split_shapes,
        child_kind=child_kind,
    )
    if not segments:
        return None
    return int(segments[0][2]), int(segments[-1][3])


def gdn_child_rank_row_range(
    *,
    split_shapes: Tuple[int, int, int, int, int, int],
    child_kind: str,
    world_size: int,
    rank: int,
) -> Optional[Tuple[int, int]]:
    """Return one TP rank's child row interval for GDN rank-local row layout."""
    child_rows = int(split_shapes[_gdn_child_index(child_kind)])
    start, end = _split_range(child_rows, int(world_size), int(rank))
    if end <= start:
        return None
    return int(start), int(end)


def gdn_child_global_shape(
    parent_global_shape: Tuple[int, int],
    split_shapes: Tuple[int, int, int, int, int, int],
    child_kind: str,
    split_axis: int = 0,
) -> Tuple[int, int]:
    """Return the global 2D shape for one GDN child."""
    split_axis = normalize_split_axis(split_axis)
    parent_rows, parent_cols = axis_shape(parent_global_shape, split_axis)
    validate_gdn_split_shapes_for_rows(
        split_shapes,
        rows=parent_rows,
        context="global_shape",
    )
    return original_shape((int(split_shapes[_gdn_child_index(child_kind)]), parent_cols), split_axis)


def gdn_child_local_shape(
    parent_local_shape: Tuple[int, int],
    split_shapes: Tuple[int, int, int, int, int, int],
    child_kind: str,
    dist_meta=None,
    split_axis: int = 0,
) -> Tuple[int, int]:
    """Return the local 2D shape for one GDN child."""
    split_axis = normalize_split_axis(split_axis)
    axis_meta = dist_meta_for_split_axis(dist_meta, split_axis)
    local_rows, local_cols = axis_shape(parent_local_shape, split_axis)
    parent_row_start, parent_row_end, active_shapes = _parent_row_range(
        local_rows=local_rows,
        split_shapes=split_shapes,
        dist_meta=axis_meta,
        context=f"local_shape:{child_kind}",
    )
    child_rows = _child_row_count(
        parent_row_start=parent_row_start,
        parent_row_end=parent_row_end,
        split_shapes=active_shapes,
        child_kind=child_kind,
    )
    if child_rows <= 0:
        raise RuntimeError(
            "[MATRIX_GDN_EMPTY_CHILD_LOCAL_SHAPE] "
            f"child_kind={child_kind} parent_local_shape={parent_local_shape}"
        )
    return original_shape((child_rows, local_cols), split_axis)


def gdn_child_has_local_overlap(
    split_shapes: Tuple[int, int, int, int, int, int],
    dist_meta,
    child_kind: str,
    split_axis: int = 0,
) -> bool:
    """Return whether one GDN child has a non-empty local shard on this rank."""
    split_axis = normalize_split_axis(split_axis)
    dist_meta = dist_meta_for_split_axis(dist_meta, split_axis)
    if dist_meta is None:
        return True
    parent_global_rows = _global_row_count(dist_meta, context=f"has_overlap:{child_kind}")
    validate_gdn_split_shapes_for_rows(
        split_shapes,
        rows=parent_global_rows,
        context=f"has_overlap:{child_kind}",
    )
    fs_shard_dim = int(getattr(dist_meta, "fs_shard_dim", -1))
    fs_world_size = int(getattr(dist_meta, "fs_world_size", 1))
    if fs_shard_dim == 0 and fs_world_size > 1 and int(getattr(dist_meta, "fs_rank", -1)) < 0:
        return False

    local_shape = getattr(dist_meta, "local_shape", None) or getattr(dist_meta, "shape", None)
    local_rows = int(local_shape[0]) if local_shape is not None else int(
        getattr(dist_meta, "fs_end_idx", 0)
    ) - int(getattr(dist_meta, "fs_start_idx", 0))
    parent_row_start, parent_row_end, active_shapes = _parent_row_range(
        local_rows=local_rows,
        split_shapes=split_shapes,
        dist_meta=dist_meta,
        context=f"has_overlap:{child_kind}",
    )
    return _child_row_count(
        parent_row_start=parent_row_start,
        parent_row_end=parent_row_end,
        split_shapes=active_shapes,
        child_kind=child_kind,
    ) > 0


def _shares_storage(lhs: torch.Tensor, rhs: torch.Tensor) -> bool:
    if lhs.numel() == 0 or rhs.numel() == 0:
        return False
    return lhs.untyped_storage().data_ptr() == rhs.untyped_storage().data_ptr()


def _uses_same_parent_range(
    dest: torch.Tensor,
    child: torch.Tensor,
    segments: list[tuple[int, int, int, int]],
) -> bool:
    if len(segments) != 1 or dest.numel() == 0 or child.numel() == 0:
        return False
    source_start, source_end, _, _ = segments[0]
    if int(source_end - source_start) != int(child.size(0)):
        return False
    if int(child.size(1)) != int(dest.size(1)):
        return False
    if child.dtype != dest.dtype or child.device != dest.device:
        return False
    if child.untyped_storage().data_ptr() != dest.untyped_storage().data_ptr():
        return False
    if tuple(int(stride) for stride in child.stride()) != tuple(
        int(stride) for stride in dest.stride()
    ):
        return False
    expected_offset = int(dest.storage_offset()) + int(source_start) * int(dest.stride(0))
    return int(child.storage_offset()) == expected_offset


def extract_gdn_child(
    tensor: torch.Tensor,
    split_shapes: Tuple[int, int, int, int, int, int],
    child_kind: str,
    dist_meta=None,
    split_axis: int = 0,
) -> torch.Tensor:
    """Read one GDN child from a fused in-projection tensor."""
    if tensor.ndim != 2:
        raise RuntimeError(
            "[MATRIX_GDN_READ_REQUIRES_2D] "
            f"child_kind={child_kind} shape={tuple(int(dim) for dim in tensor.shape)}"
        )
    split_axis = normalize_split_axis(split_axis)
    axis_meta = dist_meta_for_split_axis(dist_meta, split_axis)
    tensor_axis = axis_tensor(tensor, split_axis)
    rows, cols = int(tensor_axis.size(0)), int(tensor_axis.size(1))
    parent_row_start, parent_row_end, active_shapes = _parent_row_range(
        local_rows=rows,
        split_shapes=split_shapes,
        dist_meta=axis_meta,
        context=f"extract:{child_kind}",
    )
    segments = _child_segments(
        parent_row_start=parent_row_start,
        parent_row_end=parent_row_end,
        split_shapes=active_shapes,
        child_kind=child_kind,
    )
    if not segments:
        raise RuntimeError(
            "[MATRIX_GDN_EMPTY_CHILD_LOCAL_SHAPE] "
            f"child_kind={child_kind} tensor_shape={tuple(int(dim) for dim in tensor.shape)}"
        )
    source_start, source_end, _, _ = segments[0]
    return original_tensor(
        tensor_axis.narrow(0, source_start, source_end - source_start).contiguous(),
        split_axis,
    )


def scatter_gdn_child_(
    dest: torch.Tensor,
    child: torch.Tensor,
    split_shapes: Tuple[int, int, int, int, int, int],
    child_kind: str,
    dist_meta=None,
    split_axis: int = 0,
) -> None:
    """Write one GDN child back into the fused parent tensor."""
    if dest.ndim != 2 or child.ndim != 2:
        raise RuntimeError(
            "[MATRIX_GDN_SCATTER_REQUIRES_2D] "
            f"child_kind={child_kind} dest_shape={tuple(int(dim) for dim in dest.shape)} "
            f"child_shape={tuple(int(dim) for dim in child.shape)}"
        )
    split_axis = normalize_split_axis(split_axis)
    axis_meta = dist_meta_for_split_axis(dist_meta, split_axis)
    dest_axis = axis_tensor(dest, split_axis)
    child_axis = axis_tensor(child, split_axis).contiguous()
    rows, cols = int(dest_axis.size(0)), int(dest_axis.size(1))
    parent_row_start, parent_row_end, active_shapes = _parent_row_range(
        local_rows=rows,
        split_shapes=split_shapes,
        dist_meta=axis_meta,
        context=f"scatter:{child_kind}",
    )
    segments = _child_segments(
        parent_row_start=parent_row_start,
        parent_row_end=parent_row_end,
        split_shapes=active_shapes,
        child_kind=child_kind,
    )
    expected_child_rows = sum(
        int(source_end - source_start) for source_start, source_end, _, _ in segments
    )
    expected_child_shape = (expected_child_rows, cols)
    if tuple(int(dim) for dim in child_axis.shape) != expected_child_shape:
        raise RuntimeError(
            "[MATRIX_GDN_SCATTER_CHILD_SHAPE_MISMATCH] "
            f"child_kind={child_kind} expected_child_shape={expected_child_shape} "
            f"child_shape={tuple(int(dim) for dim in child.shape)}"
        )
    child_source = child_axis
    if _uses_same_parent_range(dest_axis, child_source, segments):
        return
    if _shares_storage(dest_axis, child_source):
        child_source = child_source.clone()
    source_start, source_end, _, _ = segments[0]
    dest_axis.narrow(0, source_start, source_end - source_start).copy_(child_source)


def iter_gdn_child_kinds() -> Iterable[str]:
    """Yield GDN child kinds in canonical checkpoint-compatible order."""
    return GDN_CHILD_KINDS
