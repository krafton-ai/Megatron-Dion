"""Two-child split helpers for fused linear weights."""

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


DEFAULT_LINEAR_CHILD_KINDS: Tuple[str, str] = ("gate", "up")


def is_linear_split_param(param: torch.Tensor) -> bool:
    """Return whether a param is tagged as a two-child fused linear."""
    return bool(getattr(param, "is_linear_split", False))


def _normalize_linear_child_kinds(child_kinds=None, *, context: str) -> Tuple[str, str]:
    if child_kinds is None:
        return DEFAULT_LINEAR_CHILD_KINDS
    child_kinds = tuple(str(kind) for kind in child_kinds)
    if len(child_kinds) != 2 or any(not kind for kind in child_kinds):
        raise RuntimeError(
            "[MATRIX_INVALID_LINEAR_CHILD_KINDS] "
            f"context={context} child_kinds={child_kinds}"
        )
    if child_kinds[0] == child_kinds[1]:
        raise RuntimeError(
            "[MATRIX_DUPLICATE_LINEAR_CHILD_KIND] "
            f"context={context} child_kinds={child_kinds}"
        )
    return child_kinds


def _linear_child_kinds_from_source(source) -> Optional[Tuple[str, str]]:
    child_kinds = getattr(source, "linear_child_kinds", None)
    if child_kinds is None:
        return None
    return _normalize_linear_child_kinds(
        child_kinds,
        context=f"source={getattr(source, 'param_name', '') or getattr(source, '_param_name', '') or id(source)}",
    )


def get_linear_child_kinds_from_dist_meta(dist_meta) -> Tuple[str, str]:
    """Return child kind names for a split-linear distributed metadata object."""
    if dist_meta is None:
        return DEFAULT_LINEAR_CHILD_KINDS
    return _normalize_linear_child_kinds(
        getattr(dist_meta, "linear_child_kinds", DEFAULT_LINEAR_CHILD_KINDS),
        context=(
            f"dist_meta.param_uid={getattr(dist_meta, 'param_uid', None)} "
            f"dist_meta.param_name={getattr(dist_meta, 'param_name', '')}"
        ),
    )


def get_linear_child_kinds_from_state(optimizer_state: Optional[dict]) -> Tuple[str, str]:
    """Return child kind names from persistent optimizer state."""
    if not optimizer_state:
        return DEFAULT_LINEAR_CHILD_KINDS
    return _normalize_linear_child_kinds(
        optimizer_state.get("linear_child_kinds", DEFAULT_LINEAR_CHILD_KINDS),
        context="optimizer_state",
    )


def get_linear_child_kinds(param: torch.Tensor) -> Tuple[str, str]:
    """Return child kind names from a tagged split-linear parameter."""
    child_kinds = _linear_child_kinds_from_source(param)
    if child_kinds is not None:
        return child_kinds
    model_param = getattr(param, "_model_param", None)
    child_kinds = _linear_child_kinds_from_source(model_param)
    if child_kinds is not None:
        return child_kinds
    return DEFAULT_LINEAR_CHILD_KINDS


def get_linear_partition_stride(source) -> int:
    """Return the local TP child-layout stride for a split-linear tensor."""
    stride = getattr(source, "linear_partition_stride", None)
    if stride is None:
        model_param = getattr(source, "_model_param", None)
        stride = getattr(model_param, "linear_partition_stride", None)
    if stride is None:
        stride = getattr(source, "partition_stride", 1)
    stride = int(stride)
    if stride <= 0:
        raise RuntimeError(
            "[MATRIX_INVALID_LINEAR_PARTITION_STRIDE] "
            f"stride={stride} source={getattr(source, '_param_name', '') or id(source)}"
        )
    return stride


def _normalize_linear_split_rows(split_rows, *, context: str) -> Tuple[int, int]:
    """Validate and normalize one two-child split tuple."""
    split_rows = tuple(int(dim) for dim in split_rows)
    if len(split_rows) != 2 or any(dim <= 0 for dim in split_rows):
        raise RuntimeError(
            "[MATRIX_INVALID_LINEAR_SPLIT_ROWS] "
            f"context={context} "
            f"split_rows={split_rows}"
        )
    return split_rows


def get_linear_split_rows_from_dist_meta(dist_meta) -> Optional[Tuple[int, int]]:
    """Return validated split sizes from distributed metadata if present."""
    if dist_meta is None:
        return None
    split_rows = getattr(dist_meta, "linear_split_rows", None)
    if split_rows is None:
        return None
    return _normalize_linear_split_rows(
        split_rows,
        context=(
            f"dist_meta.param_uid={getattr(dist_meta, 'param_uid', None)} "
            f"dist_meta.param_name={getattr(dist_meta, 'param_name', '')}"
        ),
    )


def get_linear_split_axis_from_dist_meta(dist_meta) -> int:
    """Return the fused linear split axis from distributed metadata."""
    return get_split_axis_from_dist_meta(dist_meta, "linear_split_axis", default=0)


def get_linear_split_rows(param, *, global_rows: Optional[int] = None) -> Optional[Tuple[int, int]]:
    """Return explicit child split sizes from a tagged fused linear parameter."""
    if not is_linear_split_param(param) and not hasattr(param, "linear_split_rows"):
        return None
    split_rows = getattr(param, "linear_split_rows", None)
    if split_rows is None:
        raise RuntimeError(
            "[MATRIX_LINEAR_SPLIT_PARAM_MISSING_ROWS] "
            f"param={getattr(param, '_param_name', '') or id(param)}"
        )
    split_rows = _normalize_linear_split_rows(
        split_rows,
        context=f"param={getattr(param, '_param_name', '') or id(param)}",
    )
    if global_rows is not None and int(sum(split_rows)) != int(global_rows):
        raise RuntimeError(
            "[MATRIX_LINEAR_SPLIT_ROWS_MISMATCH] "
            f"param={getattr(param, '_param_name', '') or id(param)} "
            f"split_rows={split_rows} global_rows={int(global_rows)}"
        )
    return split_rows


def get_linear_split_axis(param: torch.Tensor) -> int:
    """Return the fused linear split axis for a tagged parameter."""
    return get_split_axis_from_param(param, "linear_split_axis", default=0)


def get_linear_split_rows_from_state(optimizer_state: Optional[dict]) -> Optional[Tuple[int, int]]:
    """Return validated split sizes from persistent optimizer state if present."""
    if not optimizer_state or not bool(optimizer_state.get("linear_split_linear", False)):
        return None
    split_rows = optimizer_state.get("linear_split_rows", None)
    if split_rows is None:
        raise RuntimeError("[MATRIX_LINEAR_SPLIT_STATE_MISSING_ROWS]")
    return _normalize_linear_split_rows(split_rows, context="optimizer_state")


def get_linear_split_axis_from_state(optimizer_state: Optional[dict]) -> int:
    """Return the fused linear split axis from optimizer state."""
    return get_split_axis_from_state(optimizer_state, "linear_split_axis", default=0)


def resolve_linear_split_rows(
    *,
    optimizer_state: Optional[dict] = None,
    dist_meta=None,
) -> Optional[Tuple[int, int]]:
    """Resolve split sizes from state, then distributed metadata."""
    split_rows = get_linear_split_rows_from_state(optimizer_state)
    if split_rows is not None:
        return split_rows
    return get_linear_split_rows_from_dist_meta(dist_meta)


def resolve_linear_split_axis(
    *,
    param: Optional[torch.Tensor] = None,
    optimizer_state: Optional[dict] = None,
    dist_meta=None,
) -> int:
    """Resolve the fused linear split axis from state, dist-meta, then param attrs."""
    return resolve_split_axis(
        attr="linear_split_axis",
        param=param,
        optimizer_state=optimizer_state,
        dist_meta=dist_meta,
        default=0,
    )


def resolve_linear_child_kinds(
    *,
    param: Optional[torch.Tensor] = None,
    optimizer_state: Optional[dict] = None,
    dist_meta=None,
) -> Tuple[str, str]:
    """Resolve split-linear child kind names from state, dist-meta, then param attrs."""
    if optimizer_state and "linear_child_kinds" in optimizer_state:
        return get_linear_child_kinds_from_state(optimizer_state)
    if dist_meta is not None and hasattr(dist_meta, "linear_child_kinds"):
        return get_linear_child_kinds_from_dist_meta(dist_meta)
    if param is not None:
        return get_linear_child_kinds(param)
    return DEFAULT_LINEAR_CHILD_KINDS


def copy_linear_split_metadata(destination_tensor: torch.Tensor, source_tensor: torch.Tensor) -> None:
    """Copy fused linear split metadata."""
    if not is_linear_split_param(source_tensor) and not hasattr(source_tensor, "linear_split_rows"):
        return
    destination_tensor.is_linear_split = True
    destination_tensor.linear_split_rows = get_linear_split_rows(source_tensor)
    destination_tensor.linear_child_kinds = get_linear_child_kinds(source_tensor)
    destination_tensor.linear_partition_stride = get_linear_partition_stride(source_tensor)
    copy_split_axis(destination_tensor, source_tensor, "linear_split_axis")


def linear_child_name(parent_name: str, child_kind: str) -> str:
    """Return the optimizer-only child name for one fused linear child."""
    return f"{parent_name}::{child_kind}"


def linear_state_key(prefix: str, child_kind: str) -> str:
    """Return a stable parent-state key for one split-linear child field."""
    return f"linear_{child_kind}_{prefix}"


def linear_child_param_uid(parent_uid, child_kind: str):
    """Return a stable optimizer-only child identity derived from the parent uid."""
    if parent_uid is None:
        raise RuntimeError("[MATRIX_LINEAR_CHILD_UID_REQUIRES_PARENT_UID]")
    if isinstance(parent_uid, tuple):
        return (*parent_uid, ("linear_child", child_kind))
    return (parent_uid, ("linear_child", child_kind))


def _linear_child_index(child_kind: str, child_kinds=None) -> int:
    child_kinds = _normalize_linear_child_kinds(
        child_kinds,
        context=f"child_kind={child_kind!r}",
    )
    if child_kind == child_kinds[0]:
        return 0
    if child_kind == child_kinds[1]:
        return 1
    raise RuntimeError(
        "[MATRIX_INVALID_LINEAR_CHILD_KIND] "
        f"child_kind={child_kind!r} child_kinds={child_kinds}"
    )


def linear_child_global_shape(
    parent_global_shape: Tuple[int, int],
    split_rows: Tuple[int, int],
    child_kind: str,
    split_axis: int = 0,
    child_kinds=None,
) -> Tuple[int, int]:
    """Return the global 2D shape for one split-linear child."""
    split_axis = normalize_split_axis(split_axis)
    parent_rows, parent_cols = axis_shape(parent_global_shape, split_axis)
    if parent_rows != int(sum(split_rows)):
        raise RuntimeError(
            "[MATRIX_LINEAR_GLOBAL_ROWS_MISMATCH] "
            f"parent_global_shape={parent_global_shape} split_rows={split_rows}"
        )
    child_rows = int(split_rows[_linear_child_index(child_kind, child_kinds)])
    return original_shape((child_rows, parent_cols), split_axis)


def linear_child_row_range(
    *,
    parent_row_start: int,
    parent_row_end: int,
    split_rows: Tuple[int, int],
    child_kind: str,
    child_kinds=None,
) -> Optional[Tuple[int, int]]:
    """Project a fused linear parent row interval to one child row interval."""
    child_index = _linear_child_index(child_kind, child_kinds)
    child_row_start = 0 if child_index == 0 else int(split_rows[0])
    child_row_end = child_row_start + int(split_rows[child_index])
    overlap_start = max(int(parent_row_start), child_row_start)
    overlap_end = min(int(parent_row_end), child_row_end)
    if overlap_end <= overlap_start:
        return None
    return int(overlap_start - child_row_start), int(overlap_end - child_row_start)


def _direct_linear_rows(
    *,
    local_rows: int,
    split_rows: Tuple[int, int],
    child_kind: str,
    child_kinds=None,
) -> Tuple[int, int]:
    expected_rows = int(sum(split_rows))
    if int(local_rows) != expected_rows:
        raise RuntimeError(
            "[MATRIX_LINEAR_LOCAL_ROWS_MISMATCH] "
            f"local_rows={local_rows} split_rows={split_rows} child_kind={child_kind}"
        )
    child_index = _linear_child_index(child_kind, child_kinds)
    child_rows = int(split_rows[child_index])
    if child_index == 0:
        return 0, child_rows
    return int(split_rows[0]), child_rows


def _split_range(size: int, world_size: int, rank: int) -> Tuple[int, int]:
    if world_size <= 0:
        raise RuntimeError(f"[MATRIX_INVALID_LINEAR_WORLD_SIZE] world_size={world_size}")
    if rank < 0 or rank >= world_size:
        raise RuntimeError(
            f"[MATRIX_INVALID_LINEAR_RANK] rank={rank} world_size={world_size}"
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


def _parent_row_range(*, local_rows: int, split_rows: Tuple[int, int], dist_meta, context: str):
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
                "[MATRIX_LINEAR_MISSING_FS_RANGE] "
                f"context={context} child_rows={split_rows} "
                f"param_uid={getattr(dist_meta, 'param_uid', None)} "
                f"param_name={getattr(dist_meta, 'param_name', '')}"
            )
        if parent_row_end - parent_row_start != int(local_rows):
            raise RuntimeError(
                "[MATRIX_LINEAR_LOCAL_ROW_RANGE_MISMATCH] "
                f"context={context} local_rows={local_rows} "
                f"parent_row_range=({parent_row_start}, {parent_row_end})"
            )
        return parent_row_start, parent_row_end

    tp_shard_dim = int(getattr(dist_meta, "tp_shard_dim", -1))
    tp_world_size = int(getattr(dist_meta, "tp_world_size", 1))
    if tp_shard_dim == 0 and tp_world_size > 1:
        tp_rank = int(getattr(dist_meta, "tp_rank", -1))
        parent_global_shape = getattr(dist_meta, "global_shape", None)
        if parent_global_shape is None or len(parent_global_shape) != 2:
            raise RuntimeError(
                "[MATRIX_LINEAR_MISSING_GLOBAL_SHAPE] "
                f"context={context} "
                f"param_uid={getattr(dist_meta, 'param_uid', None)} "
                f"param_name={getattr(dist_meta, 'param_name', '')}"
            )
        parent_row_start, parent_row_end = _split_range(
            int(parent_global_shape[0]),
            tp_world_size,
            tp_rank,
        )
        if parent_row_end - parent_row_start != int(local_rows):
            raise RuntimeError(
                "[MATRIX_LINEAR_LOCAL_ROW_RANGE_MISMATCH] "
                f"context={context} local_rows={local_rows} "
                f"parent_row_range=({parent_row_start}, {parent_row_end})"
            )
        return parent_row_start, parent_row_end

    expected_rows = int(sum(split_rows))
    if int(local_rows) != expected_rows:
        raise RuntimeError(
            "[MATRIX_LINEAR_LOCAL_ROWS_MISMATCH] "
            f"context={context} local_rows={local_rows} split_rows={split_rows} "
            f"param_uid={getattr(dist_meta, 'param_uid', None)} "
            f"param_name={getattr(dist_meta, 'param_name', '')}"
        )
    return 0, int(local_rows)


def _linear_partition_stride(dist_meta) -> int:
    return int(getattr(dist_meta, "linear_partition_stride", 1)) if dist_meta is not None else 1


def _linear_child_segments(
    *,
    local_rows: int,
    split_rows: Tuple[int, int],
    dist_meta,
    child_kind: str,
    context: str,
    child_kinds=None,
) -> list[tuple[int, int, int, int]]:
    """Map one parent local tensor to source and child row intervals."""
    child_index = _linear_child_index(child_kind, child_kinds)
    tp_shard_dim = int(getattr(dist_meta, "tp_shard_dim", -1)) if dist_meta is not None else -1
    tp_world_size = int(getattr(dist_meta, "tp_world_size", 1)) if dist_meta is not None else 1
    partition_stride = _linear_partition_stride(dist_meta)

    if tp_shard_dim == 0 and tp_world_size > 1:
        if partition_stride == len(split_rows):
            tp_rank = int(getattr(dist_meta, "tp_rank", -1))
            source_cursor = 0
            segments: list[tuple[int, int, int, int]] = []
            for split_index, child_global_rows in enumerate(split_rows):
                child_start, child_end = _split_range(int(child_global_rows), tp_world_size, tp_rank)
                child_local_rows = int(child_end - child_start)
                if split_index == child_index and child_local_rows > 0:
                    segments.append(
                        (
                            int(source_cursor),
                            int(source_cursor + child_local_rows),
                            int(child_start),
                            int(child_end),
                        )
                    )
                source_cursor += child_local_rows
            if source_cursor != int(local_rows):
                raise RuntimeError(
                    "[MATRIX_LINEAR_STRIDED_TP_LOCAL_ROWS_MISMATCH] "
                    f"context={context} local_rows={local_rows} expected_rows={source_cursor} "
                    f"tp_world_size={tp_world_size} tp_rank={tp_rank} split_rows={split_rows}"
                )
            return segments

        if partition_stride != 1:
            raise RuntimeError(
                "[MATRIX_LINEAR_UNSUPPORTED_TP_PARTITION_STRIDE] "
                f"context={context} partition_stride={partition_stride} split_rows={split_rows}"
            )

    parent_row_start, parent_row_end = _parent_row_range(
        local_rows=local_rows,
        split_rows=split_rows,
        dist_meta=dist_meta,
        context=context,
    )
    if parent_row_start < 0:
        return []
    if parent_row_start == 0 and parent_row_end == int(sum(split_rows)):
        child_source_start, child_rows = _direct_linear_rows(
            local_rows=local_rows,
            split_rows=split_rows,
            child_kind=child_kind,
            child_kinds=child_kinds,
        )
        return [
            (
                int(child_source_start),
                int(child_source_start + child_rows),
                0,
                int(child_rows),
            )
        ]

    child_range = linear_child_row_range(
        parent_row_start=parent_row_start,
        parent_row_end=parent_row_end,
        split_rows=split_rows,
        child_kind=child_kind,
        child_kinds=child_kinds,
    )
    if child_range is None:
        return []
    child_start, child_end = child_range
    child_row_start = 0 if child_index == 0 else int(split_rows[0])
    overlap_start = child_start + child_row_start
    overlap_end = child_end + child_row_start
    source_start = overlap_start - int(parent_row_start)
    return [
        (
            int(source_start),
            int(source_start + overlap_end - overlap_start),
            int(child_start),
            int(child_end),
        )
    ]


def linear_child_has_local_overlap(
    split_rows: Tuple[int, int],
    dist_meta,
    child_kind: str,
    split_axis: int = 0,
    child_kinds=None,
) -> bool:
    """Return whether one split-linear child has a non-empty local shard on this rank."""
    split_axis = normalize_split_axis(split_axis)
    dist_meta = dist_meta_for_split_axis(dist_meta, split_axis)
    if dist_meta is None:
        return True

    fs_shard_dim = int(getattr(dist_meta, "fs_shard_dim", -1))
    fs_world_size = int(getattr(dist_meta, "fs_world_size", 1))
    if fs_shard_dim == 0 and fs_world_size > 1 and int(getattr(dist_meta, "fs_rank", -1)) < 0:
        return False

    local_shape = getattr(dist_meta, "local_shape", None) or getattr(dist_meta, "shape", None)
    if local_shape is None:
        local_rows = int(getattr(dist_meta, "fs_end_idx", 0)) - int(
            getattr(dist_meta, "fs_start_idx", 0)
        )
    else:
        local_rows = int(local_shape[0])
    segments = _linear_child_segments(
        local_rows=local_rows,
        split_rows=split_rows,
        dist_meta=dist_meta,
        child_kind=child_kind,
        context=f"has_overlap:{child_kind}",
        child_kinds=child_kinds,
    )
    return any(source_end > source_start for source_start, source_end, _, _ in segments)


def linear_child_local_shape(
    parent_local_shape: Tuple[int, int],
    split_rows: Tuple[int, int],
    dist_meta,
    child_kind: str,
    split_axis: int = 0,
    child_kinds=None,
) -> Tuple[int, int]:
    """Return the local 2D shape for one split-linear child."""
    split_axis = normalize_split_axis(split_axis)
    axis_meta = dist_meta_for_split_axis(dist_meta, split_axis)
    local_rows, local_cols = axis_shape(parent_local_shape, split_axis)
    segments = _linear_child_segments(
        local_rows=local_rows,
        split_rows=split_rows,
        dist_meta=axis_meta,
        child_kind=child_kind,
        context=f"local_shape:{child_kind}",
        child_kinds=child_kinds,
    )
    child_rows = sum(int(source_end - source_start) for source_start, source_end, _, _ in segments)
    if child_rows <= 0:
        raise RuntimeError(
            "[MATRIX_LINEAR_EMPTY_CHILD_LOCAL_SHAPE] "
            f"parent_local_shape={parent_local_shape} child_kind={child_kind}"
        )
    return original_shape((child_rows, local_cols), split_axis)


def read_linear_child(
    tensor: torch.Tensor,
    split_rows: Tuple[int, int],
    dist_meta,
    child_kind: str,
    split_axis: int = 0,
    child_kinds=None,
) -> torch.Tensor:
    """Read one split-linear child from a fused parent tensor into a contiguous 2D tensor."""
    if tensor.ndim != 2:
        raise RuntimeError(
            "[MATRIX_LINEAR_READ_REQUIRES_2D] "
            f"child_kind={child_kind} shape={tuple(int(dim) for dim in tensor.shape)}"
        )
    split_axis = normalize_split_axis(split_axis)
    axis_meta = dist_meta_for_split_axis(dist_meta, split_axis)
    tensor_axis = axis_tensor(tensor, split_axis)
    local_rows = int(tensor_axis.size(0))
    segments = _linear_child_segments(
        local_rows=local_rows,
        split_rows=split_rows,
        dist_meta=axis_meta,
        child_kind=child_kind,
        context=f"read:{child_kind}",
        child_kinds=child_kinds,
    )
    if not segments:
        raise RuntimeError(
            "[MATRIX_LINEAR_EMPTY_CHILD_LOCAL_SHAPE] "
            f"child_kind={child_kind} tensor_shape={tuple(int(dim) for dim in tensor.shape)}"
        )
    if len(segments) == 1:
        source_start, source_end, _, _ = segments[0]
        return original_tensor(
            tensor_axis.narrow(0, source_start, source_end - source_start).contiguous(),
            split_axis,
        )

    child_rows = sum(int(source_end - source_start) for source_start, source_end, _, _ in segments)
    child = tensor_axis.new_empty((child_rows, int(tensor_axis.size(1))))
    child_cursor = 0
    for source_start, source_end, _, _ in segments:
        rows_in_segment = int(source_end - source_start)
        child.narrow(0, child_cursor, rows_in_segment).copy_(
            tensor_axis.narrow(0, source_start, rows_in_segment)
        )
        child_cursor += rows_in_segment
    return original_tensor(child, split_axis)


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


def write_linear_child_(
    dest: torch.Tensor,
    child: torch.Tensor,
    split_rows: Tuple[int, int],
    dist_meta,
    child_kind: str,
    split_axis: int = 0,
    child_kinds=None,
) -> None:
    """Write one split-linear child tensor back into the fused parent tensor."""
    if dest.ndim != 2 or child.ndim != 2:
        raise RuntimeError(
            "[MATRIX_LINEAR_WRITE_REQUIRES_2D] "
            f"child_kind={child_kind} dest_shape={tuple(int(dim) for dim in dest.shape)} "
            f"child_shape={tuple(int(dim) for dim in child.shape)}"
        )
    split_axis = normalize_split_axis(split_axis)
    axis_meta = dist_meta_for_split_axis(dist_meta, split_axis)
    dest_axis = axis_tensor(dest, split_axis)
    child_axis = axis_tensor(child, split_axis).contiguous()
    expected_shape = linear_child_local_shape(
        parent_local_shape=(int(dest.size(0)), int(dest.size(1))),
        split_rows=split_rows,
        dist_meta=dist_meta,
        child_kind=child_kind,
        split_axis=split_axis,
        child_kinds=child_kinds,
    )
    if tuple(int(dim) for dim in child.shape) != expected_shape:
        raise RuntimeError(
            "[MATRIX_LINEAR_CHILD_SHAPE_MISMATCH] "
            f"child_kind={child_kind} expected_shape={expected_shape} "
            f"child_shape={tuple(int(dim) for dim in child.shape)}"
        )

    local_rows = int(dest_axis.size(0))
    segments = _linear_child_segments(
        local_rows=local_rows,
        split_rows=split_rows,
        dist_meta=axis_meta,
        child_kind=child_kind,
        context=f"write:{child_kind}",
        child_kinds=child_kinds,
    )
    child_source = child_axis
    if _uses_same_parent_range(dest_axis, child_source, segments):
        return
    if _shares_storage(dest_axis, child_source):
        child_source = child_source.clone()
    child_cursor = 0
    for source_start, source_end, _, _ in segments:
        rows_in_segment = int(source_end - source_start)
        dest_axis.narrow(0, source_start, rows_in_segment).copy_(
            child_source.narrow(0, child_cursor, rows_in_segment)
        )
        child_cursor += rows_in_segment


def iter_linear_child_kinds(child_kinds=None) -> Iterable[str]:
    """Yield split-linear child kinds in canonical order."""
    return _normalize_linear_child_kinds(child_kinds, context="iter_linear_child_kinds")
