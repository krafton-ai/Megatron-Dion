"""QKV virtual-split helpers for Dion/Muon-style fused QKV weights."""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import torch


QKV_CHILD_KINDS: Tuple[str, str, str] = ("q", "k", "v")


def is_qkv_param(param: torch.Tensor) -> bool:
    """Return whether a param is tagged as fused QKV."""
    return bool(getattr(param, "is_qkv", False))


def _normalize_qkv_split_shapes(split_shapes, *, context: str) -> Tuple[int, int, int]:
    """Validate and normalize one QKV split-shape tuple."""
    split_shapes = tuple(int(dim) for dim in split_shapes)
    if len(split_shapes) != 3 or any(dim <= 0 for dim in split_shapes):
        raise RuntimeError(
            "[DION_INVALID_QKV_SPLIT_SHAPES] "
            f"context={context} "
            f"split_shapes={split_shapes}"
        )
    return split_shapes


def get_qkv_split_shapes(param: torch.Tensor) -> Tuple[int, int, int]:
    """Return per-group QKV split sizes from a tagged fused QKV parameter."""
    split_shapes = getattr(param, "qkv_split_shapes", None)
    if split_shapes is None:
        model_param = getattr(param, "_model_param", None)
        split_shapes = getattr(model_param, "qkv_split_shapes", None)
    if split_shapes is None:
        raise RuntimeError(
            "[DION_QKV_SPLIT_SHAPES_MISSING] "
            f"param={getattr(param, '_param_name', '') or id(param)}"
        )
    return _normalize_qkv_split_shapes(
        split_shapes,
        context=f"param={getattr(param, '_param_name', '') or id(param)}",
    )


def get_qkv_split_shapes_from_dist_meta(dist_meta) -> Optional[Tuple[int, int, int]]:
    """Return validated QKV split sizes from distributed metadata if present."""
    if dist_meta is None:
        return None
    split_shapes = getattr(dist_meta, "qkv_split_shapes", None)
    if split_shapes is None:
        return None
    return _normalize_qkv_split_shapes(
        split_shapes,
        context=(
            f"dist_meta.param_uid={getattr(dist_meta, 'param_uid', None)} "
            f"dist_meta.param_name={getattr(dist_meta, 'param_name', '')}"
        ),
    )


def get_qkv_split_shapes_from_state(optimizer_state: Optional[dict]) -> Optional[Tuple[int, int, int]]:
    """Return validated QKV split sizes from persistent optimizer state if present."""
    if not optimizer_state or not bool(optimizer_state.get("qkv_split_qkv", False)):
        return None
    split_shapes = optimizer_state.get("qkv_split_shapes", None)
    if split_shapes is None:
        raise RuntimeError("[DION_QKV_SPLIT_STATE_MISSING_SHAPES]")
    return _normalize_qkv_split_shapes(split_shapes, context="optimizer_state")


def resolve_qkv_split_shapes(
    *,
    param: Optional[torch.Tensor] = None,
    optimizer_state: Optional[dict] = None,
    dist_meta=None,
) -> Optional[Tuple[int, int, int]]:
    """Resolve QKV split shapes from state, dist-meta, then param attrs."""
    split_shapes = get_qkv_split_shapes_from_state(optimizer_state)
    if split_shapes is not None:
        return split_shapes
    split_shapes = get_qkv_split_shapes_from_dist_meta(dist_meta)
    if split_shapes is not None:
        return split_shapes
    if param is None:
        return None
    try:
        return get_qkv_split_shapes(param)
    except RuntimeError:
        return None


def copy_qkv_split_metadata(destination_tensor: torch.Tensor, source_tensor: torch.Tensor) -> None:
    """Copy fused-QKV split metadata when the source tensor is tagged as QKV."""
    if not is_qkv_param(source_tensor) and not hasattr(source_tensor, "qkv_split_shapes"):
        return
    destination_tensor.qkv_split_shapes = get_qkv_split_shapes(source_tensor)


def qkv_child_name(parent_name: str, child_kind: str) -> str:
    """Return the optimizer-only child name for one fused QKV child."""
    if child_kind not in QKV_CHILD_KINDS:
        raise RuntimeError(f"[DION_INVALID_QKV_CHILD_KIND] child_kind={child_kind!r}")
    return f"{parent_name}::{child_kind}"


def qkv_state_key(prefix: str, child_kind: str) -> str:
    """Return a stable parent-state key for one QKV child field."""
    if child_kind not in QKV_CHILD_KINDS:
        raise RuntimeError(f"[DION_INVALID_QKV_CHILD_KIND] child_kind={child_kind!r}")
    return f"qkv_{child_kind}_{prefix}"


def qkv_child_param_uid(parent_uid, child_kind: str):
    """Return a stable optimizer-only child identity derived from the parent uid."""
    if child_kind not in QKV_CHILD_KINDS:
        raise RuntimeError(f"[DION_INVALID_QKV_CHILD_KIND] child_kind={child_kind!r}")
    if parent_uid is None:
        raise RuntimeError("[DION_QKV_CHILD_UID_REQUIRES_PARENT_UID]")
    if isinstance(parent_uid, tuple):
        return (*parent_uid, ("qkv_child", child_kind))
    return (parent_uid, ("qkv_child", child_kind))


def _qkv_child_index(child_kind: str) -> int:
    if child_kind == "q":
        return 0
    if child_kind == "k":
        return 1
    if child_kind == "v":
        return 2
    raise RuntimeError(f"[DION_INVALID_QKV_CHILD_KIND] child_kind={child_kind!r}")


def _validate_qkv_layout_rows(*, rows: int, split_shapes: Tuple[int, int, int], context: str) -> int:
    total_per_group = int(sum(split_shapes))
    if rows <= 0:
        raise RuntimeError(f"[DION_INVALID_QKV_ROW_COUNT] context={context} rows={rows}")
    if rows % total_per_group != 0:
        raise RuntimeError(
            "[DION_QKV_LOCAL_LAYOUT_MISMATCH] "
            f"context={context} rows={rows} total_per_group={total_per_group} "
            f"split_shapes={split_shapes} "
            "dion_split_qkv requires an integral global grouped-QKV row layout."
        )
    return rows // total_per_group


def validate_qkv_split_shapes_for_rows(
    split_shapes: Tuple[int, int, int],
    *,
    rows: int,
    context: str,
) -> None:
    """Validate that a row count can represent grouped QKV with these split sizes."""
    _validate_qkv_layout_rows(rows=int(rows), split_shapes=split_shapes, context=context)


def _split_range(size: int, world_size: int, rank: int) -> Tuple[int, int]:
    if world_size <= 0:
        raise RuntimeError(f"[DION_INVALID_QKV_WORLD_SIZE] world_size={world_size}")
    if rank < 0 or rank >= world_size:
        raise RuntimeError(
            f"[DION_INVALID_QKV_RANK] rank={rank} world_size={world_size}"
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


def _global_row_count(dist_meta, *, context: str) -> int:
    parent_global_shape = getattr(dist_meta, "global_shape", None)
    if parent_global_shape is None or len(parent_global_shape) != 2:
        raise RuntimeError(
            "[DION_QKV_MISSING_GLOBAL_SHAPE] "
            f"context={context} "
            f"param_uid={getattr(dist_meta, 'param_uid', None)} "
            f"param_name={getattr(dist_meta, 'param_name', '')}"
        )
    return int(parent_global_shape[0])


def _parent_row_range(
    *,
    local_rows: int,
    dist_meta,
    context: str,
) -> Tuple[int, int]:
    if dist_meta is None:
        return 0, int(local_rows)

    fs_shard_dim = int(getattr(dist_meta, "fs_shard_dim", -1))
    fs_world_size = int(getattr(dist_meta, "fs_world_size", 1))
    if fs_shard_dim == 0 and fs_world_size > 1:
        fs_rank = int(getattr(dist_meta, "fs_rank", -1))
        if fs_rank < 0:
            return -1, -1
        parent_row_start = int(getattr(dist_meta, "fs_start_idx", -1))
        parent_row_end = int(getattr(dist_meta, "fs_end_idx", -1))
        if parent_row_start < 0 or parent_row_end < parent_row_start:
            raise RuntimeError(
                "[DION_QKV_MISSING_FS_RANGE] "
                f"context={context} "
                f"param_uid={getattr(dist_meta, 'param_uid', None)} "
                f"param_name={getattr(dist_meta, 'param_name', '')}"
            )
        if parent_row_end - parent_row_start != int(local_rows):
            raise RuntimeError(
                "[DION_QKV_LOCAL_ROW_RANGE_MISMATCH] "
                f"context={context} local_rows={local_rows} "
                f"parent_row_range=({parent_row_start}, {parent_row_end})"
            )
        return parent_row_start, parent_row_end

    tp_shard_dim = int(getattr(dist_meta, "tp_shard_dim", -1))
    tp_world_size = int(getattr(dist_meta, "tp_world_size", 1))
    if tp_shard_dim == 0 and tp_world_size > 1:
        tp_rank = int(getattr(dist_meta, "tp_rank", -1))
        parent_global_rows = _global_row_count(dist_meta, context=context)
        parent_row_start, parent_row_end = _split_range(
            parent_global_rows,
            tp_world_size,
            tp_rank,
        )
        if parent_row_end - parent_row_start != int(local_rows):
            raise RuntimeError(
                "[DION_QKV_LOCAL_ROW_RANGE_MISMATCH] "
                f"context={context} local_rows={local_rows} "
                f"parent_row_range=({parent_row_start}, {parent_row_end})"
            )
        return parent_row_start, parent_row_end

    return 0, int(local_rows)


def _child_segments(
    *,
    parent_row_start: int,
    parent_row_end: int,
    split_shapes: Tuple[int, int, int],
    child_kind: str,
) -> list[tuple[int, int, int, int]]:
    """Map one parent row interval to source and child row intervals."""
    if parent_row_start < 0 or parent_row_end <= parent_row_start:
        return []
    total_per_group = int(sum(split_shapes))
    child_index = _qkv_child_index(child_kind)
    child_rows_per_group = int(split_shapes[child_index])
    child_offset = sum(int(split_shapes[idx]) for idx in range(child_index))
    first_group = int(parent_row_start) // total_per_group
    last_group = (int(parent_row_end) - 1) // total_per_group

    segments: list[tuple[int, int, int, int]] = []
    for group_idx in range(first_group, last_group + 1):
        parent_child_start = group_idx * total_per_group + child_offset
        parent_child_end = parent_child_start + child_rows_per_group
        overlap_start = max(int(parent_row_start), parent_child_start)
        overlap_end = min(int(parent_row_end), parent_child_end)
        if overlap_end <= overlap_start:
            continue
        child_start = group_idx * child_rows_per_group + (
            overlap_start - parent_child_start
        )
        child_end = child_start + (overlap_end - overlap_start)
        source_start = overlap_start - int(parent_row_start)
        source_end = overlap_end - int(parent_row_start)
        if segments and segments[-1][3] != child_start:
            raise RuntimeError(
                "[DION_QKV_CHILD_NONCONTIGUOUS_LOCAL_RANGE] "
                f"child_kind={child_kind} parent_row_range=({parent_row_start}, {parent_row_end}) "
                f"previous_child_end={segments[-1][3]} next_child_start={child_start}"
            )
        segments.append((int(source_start), int(source_end), int(child_start), int(child_end)))
    return segments


def _child_row_count(
    *,
    parent_row_start: int,
    parent_row_end: int,
    split_shapes: Tuple[int, int, int],
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


def qkv_child_row_range(
    *,
    parent_row_start: int,
    parent_row_end: int,
    split_shapes: Tuple[int, int, int],
    child_kind: str,
) -> Optional[Tuple[int, int]]:
    """Project a fused-QKV parent row interval to one child row interval."""
    segments = _child_segments(
        parent_row_start=parent_row_start,
        parent_row_end=parent_row_end,
        split_shapes=split_shapes,
        child_kind=child_kind,
    )
    if not segments:
        return None
    return int(segments[0][2]), int(segments[-1][3])


def qkv_child_global_shape(
    parent_global_shape: Tuple[int, int],
    split_shapes: Tuple[int, int, int],
    child_kind: str,
) -> Tuple[int, int]:
    """Return the global 2D shape for one Q/K/V child."""
    parent_rows, parent_cols = (int(parent_global_shape[0]), int(parent_global_shape[1]))
    num_query_groups = _validate_qkv_layout_rows(
        rows=parent_rows,
        split_shapes=split_shapes,
        context="global_shape",
    )
    child_rows = int(split_shapes[_qkv_child_index(child_kind)]) * int(num_query_groups)
    return (child_rows, parent_cols)


def qkv_child_local_shape(
    parent_local_shape: Tuple[int, int],
    split_shapes: Tuple[int, int, int],
    child_kind: str,
    dist_meta=None,
) -> Tuple[int, int]:
    """Return the local 2D shape for one Q/K/V child."""
    local_rows, local_cols = (int(parent_local_shape[0]), int(parent_local_shape[1]))
    parent_row_start, parent_row_end = _parent_row_range(
        local_rows=local_rows,
        dist_meta=dist_meta,
        context=f"local_shape:{child_kind}",
    )
    child_rows = _child_row_count(
        parent_row_start=parent_row_start,
        parent_row_end=parent_row_end,
        split_shapes=split_shapes,
        child_kind=child_kind,
    )
    if child_rows <= 0:
        raise RuntimeError(
            "[DION_QKV_EMPTY_CHILD_LOCAL_SHAPE] "
            f"child_kind={child_kind} parent_local_shape={parent_local_shape}"
        )
    return (child_rows, local_cols)


def qkv_child_has_local_overlap(
    split_shapes: Tuple[int, int, int],
    dist_meta,
    child_kind: str,
) -> bool:
    """Return whether one Q/K/V child has a non-empty local shard on this rank."""
    if dist_meta is None:
        return True

    parent_global_rows = _global_row_count(dist_meta, context=f"has_overlap:{child_kind}")
    total_per_group = int(sum(int(dim) for dim in split_shapes))
    if total_per_group <= 0 or parent_global_rows % total_per_group != 0:
        raise RuntimeError(
            "[DION_QKV_GLOBAL_LAYOUT_MISMATCH] "
            f"child_kind={child_kind} parent_global_rows={parent_global_rows} "
            f"split_shapes={split_shapes}"
        )

    fs_shard_dim = int(getattr(dist_meta, "fs_shard_dim", -1))
    fs_world_size = int(getattr(dist_meta, "fs_world_size", 1))
    if fs_shard_dim == 0 and fs_world_size > 1 and int(getattr(dist_meta, "fs_rank", -1)) < 0:
        return False

    local_shape = getattr(dist_meta, "local_shape", None) or getattr(dist_meta, "shape", None)
    local_rows = int(local_shape[0]) if local_shape is not None else int(
        getattr(dist_meta, "fs_end_idx", 0)
    ) - int(getattr(dist_meta, "fs_start_idx", 0))
    parent_row_start, parent_row_end = _parent_row_range(
        local_rows=local_rows,
        dist_meta=dist_meta,
        context=f"has_overlap:{child_kind}",
    )
    return _child_row_count(
        parent_row_start=parent_row_start,
        parent_row_end=parent_row_end,
        split_shapes=split_shapes,
        child_kind=child_kind,
    ) > 0


def _shares_storage(lhs: torch.Tensor, rhs: torch.Tensor) -> bool:
    """Return whether two tensors are backed by the same storage."""
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


def extract_qkv_child(
    tensor: torch.Tensor,
    split_shapes: Tuple[int, int, int],
    child_kind: str,
    dist_meta=None,
) -> torch.Tensor:
    """Read one Q/K/V child from a fused QKV tensor into a contiguous 2D tensor."""
    if tensor.ndim != 2:
        raise RuntimeError(
            "[DION_QKV_READ_REQUIRES_2D] "
            f"child_kind={child_kind} shape={tuple(int(dim) for dim in tensor.shape)}"
        )
    rows, cols = int(tensor.size(0)), int(tensor.size(1))
    parent_row_start, parent_row_end = _parent_row_range(
        local_rows=rows,
        dist_meta=dist_meta,
        context=f"extract:{child_kind}",
    )
    segments = _child_segments(
        parent_row_start=parent_row_start,
        parent_row_end=parent_row_end,
        split_shapes=split_shapes,
        child_kind=child_kind,
    )
    if not segments:
        raise RuntimeError(
            "[DION_QKV_EMPTY_CHILD_LOCAL_SHAPE] "
            f"child_kind={child_kind} tensor_shape={tuple(int(dim) for dim in tensor.shape)}"
        )
    if len(segments) == 1:
        source_start, source_end, _, _ = segments[0]
        return tensor.narrow(0, source_start, source_end - source_start).contiguous()

    child_rows = sum(int(source_end - source_start) for source_start, source_end, _, _ in segments)
    child = tensor.new_empty((child_rows, cols))
    child_cursor = 0
    for source_start, source_end, _, _ in segments:
        rows_in_segment = int(source_end - source_start)
        child.narrow(0, child_cursor, rows_in_segment).copy_(
            tensor.narrow(0, source_start, rows_in_segment)
        )
        child_cursor += rows_in_segment
    return child


def scatter_qkv_child_(
    dest: torch.Tensor,
    child: torch.Tensor,
    split_shapes: Tuple[int, int, int],
    child_kind: str,
    dist_meta=None,
) -> None:
    """Write one Q/K/V child back into the fused parent tensor."""
    if dest.ndim != 2 or child.ndim != 2:
        raise RuntimeError(
            "[DION_QKV_SCATTER_REQUIRES_2D] "
            f"child_kind={child_kind} dest_shape={tuple(int(dim) for dim in dest.shape)} "
            f"child_shape={tuple(int(dim) for dim in child.shape)}"
        )
    rows, cols = int(dest.size(0)), int(dest.size(1))
    parent_row_start, parent_row_end = _parent_row_range(
        local_rows=rows,
        dist_meta=dist_meta,
        context=f"scatter:{child_kind}",
    )
    segments = _child_segments(
        parent_row_start=parent_row_start,
        parent_row_end=parent_row_end,
        split_shapes=split_shapes,
        child_kind=child_kind,
    )
    expected_child_rows = sum(
        int(source_end - source_start) for source_start, source_end, _, _ in segments
    )
    expected_child_shape = (expected_child_rows, cols)
    if tuple(int(dim) for dim in child.shape) != expected_child_shape:
        raise RuntimeError(
            "[DION_QKV_SCATTER_CHILD_SHAPE_MISMATCH] "
            f"child_kind={child_kind} expected_child_shape={expected_child_shape} "
            f"child_shape={tuple(int(dim) for dim in child.shape)}"
        )
    child_source = child.contiguous()
    if _uses_same_parent_range(dest, child_source, segments):
        return
    if _shares_storage(dest, child_source):
        child_source = child_source.clone()
    child_cursor = 0
    for source_start, source_end, _, _ in segments:
        rows_in_segment = int(source_end - source_start)
        dest.narrow(0, source_start, rows_in_segment).copy_(
            child_source.narrow(0, child_cursor, rows_in_segment)
        )
        child_cursor += rows_in_segment


def iter_qkv_child_kinds() -> Iterable[str]:
    """Yield QKV child kinds in canonical order."""
    return QKV_CHILD_KINDS
