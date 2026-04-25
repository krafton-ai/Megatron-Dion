"""Gate/up child helpers for Dion fused linear_fc1 weights."""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import torch


LINEAR_CHILD_KINDS: Tuple[str, str] = ("gate", "up")


def is_linear_fc1_param(param: torch.Tensor) -> bool:
    """Return whether a param is tagged as fused linear_fc1."""
    return bool(getattr(param, "is_linear_fc1", False))


def _normalize_linear_split_rows(split_rows, *, context: str) -> Tuple[int, int]:
    """Validate and normalize one gate/up split tuple."""
    split_rows = tuple(int(dim) for dim in split_rows)
    if len(split_rows) != 2 or any(dim <= 0 for dim in split_rows):
        raise RuntimeError(
            "[DION_INVALID_LINEAR_SPLIT_ROWS] "
            f"context={context} "
            f"split_rows={split_rows}"
        )
    return split_rows


def get_linear_split_rows_from_dist_meta(dist_meta) -> Optional[Tuple[int, int]]:
    """Return validated gate/up row sizes from distributed metadata if present."""
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


def get_linear_split_rows(param, *, global_rows: Optional[int] = None) -> Optional[Tuple[int, int]]:
    """Return explicit gate/up row sizes from a tagged fused linear_fc1 parameter."""
    if not is_linear_fc1_param(param):
        return None
    split_rows = getattr(param, "linear_split_rows", None)
    if split_rows is None:
        raise RuntimeError(
            "[DION_LINEAR_SPLIT_PARAM_MISSING_ROWS] "
            f"param={getattr(param, '_param_name', '') or id(param)}"
        )
    split_rows = _normalize_linear_split_rows(
        split_rows,
        context=f"param={getattr(param, '_param_name', '') or id(param)}",
    )
    if global_rows is not None and int(sum(split_rows)) != int(global_rows):
        raise RuntimeError(
            "[DION_LINEAR_SPLIT_ROWS_MISMATCH] "
            f"param={getattr(param, '_param_name', '') or id(param)} "
            f"split_rows={split_rows} global_rows={int(global_rows)}"
        )
    return split_rows


def get_linear_split_rows_from_state(optimizer_state: Optional[dict]) -> Optional[Tuple[int, int]]:
    """Return validated gate/up row sizes from persistent optimizer state if present."""
    if not optimizer_state or not bool(optimizer_state.get("linear_split_linear", False)):
        return None
    split_rows = optimizer_state.get("linear_split_rows", None)
    if split_rows is None:
        raise RuntimeError("[DION_LINEAR_SPLIT_STATE_MISSING_ROWS]")
    return _normalize_linear_split_rows(split_rows, context="optimizer_state")


def resolve_linear_split_rows(
    *,
    optimizer_state: Optional[dict] = None,
    dist_meta=None,
) -> Optional[Tuple[int, int]]:
    """Resolve gate/up row sizes from state, then distributed metadata."""
    split_rows = get_linear_split_rows_from_state(optimizer_state)
    if split_rows is not None:
        return split_rows
    return get_linear_split_rows_from_dist_meta(dist_meta)


def linear_child_name(parent_name: str, child_kind: str) -> str:
    """Return the optimizer-only child name for one fused linear_fc1 child."""
    if child_kind not in LINEAR_CHILD_KINDS:
        raise RuntimeError(f"[DION_INVALID_LINEAR_CHILD_KIND] child_kind={child_kind!r}")
    return f"{parent_name}::{child_kind}"


def linear_state_key(prefix: str, child_kind: str) -> str:
    """Return a stable parent-state key for one gate/up child field."""
    if child_kind not in LINEAR_CHILD_KINDS:
        raise RuntimeError(f"[DION_INVALID_LINEAR_CHILD_KIND] child_kind={child_kind!r}")
    return f"linear_{child_kind}_{prefix}"


def linear_child_param_uid(parent_uid, child_kind: str):
    """Return a stable optimizer-only child identity derived from the parent uid."""
    if child_kind not in LINEAR_CHILD_KINDS:
        raise RuntimeError(f"[DION_INVALID_LINEAR_CHILD_KIND] child_kind={child_kind!r}")
    if parent_uid is None:
        raise RuntimeError("[DION_LINEAR_CHILD_UID_REQUIRES_PARENT_UID]")
    if isinstance(parent_uid, tuple):
        return (*parent_uid, ("linear_child", child_kind))
    return (parent_uid, ("linear_child", child_kind))


def _linear_child_index(child_kind: str) -> int:
    if child_kind == "gate":
        return 0
    if child_kind == "up":
        return 1
    raise RuntimeError(f"[DION_INVALID_LINEAR_CHILD_KIND] child_kind={child_kind!r}")


def linear_child_global_shape(
    parent_global_shape: Tuple[int, int],
    split_rows: Tuple[int, int],
    child_kind: str,
) -> Tuple[int, int]:
    """Return the global 2D shape for one gate/up child."""
    parent_rows, parent_cols = (int(parent_global_shape[0]), int(parent_global_shape[1]))
    if parent_rows != int(sum(split_rows)):
        raise RuntimeError(
            "[DION_LINEAR_GLOBAL_ROWS_MISMATCH] "
            f"parent_global_shape={parent_global_shape} split_rows={split_rows}"
        )
    child_rows = int(split_rows[_linear_child_index(child_kind)])
    return (child_rows, parent_cols)


def _direct_linear_rows(
    *,
    local_rows: int,
    split_rows: Tuple[int, int],
    child_kind: str,
) -> Tuple[int, int]:
    expected_rows = int(sum(split_rows))
    if int(local_rows) != expected_rows:
        raise RuntimeError(
            "[DION_LINEAR_LOCAL_ROWS_MISMATCH] "
            f"local_rows={local_rows} split_rows={split_rows} child_kind={child_kind}"
        )
    child_rows = int(split_rows[_linear_child_index(child_kind)])
    if child_kind == "gate":
        return 0, child_rows
    return int(split_rows[0]), child_rows


def _split_range(size: int, world_size: int, rank: int) -> Tuple[int, int]:
    if world_size <= 0:
        raise RuntimeError(f"[DION_INVALID_LINEAR_WORLD_SIZE] world_size={world_size}")
    if rank < 0 or rank >= world_size:
        raise RuntimeError(
            f"[DION_INVALID_LINEAR_RANK] rank={rank} world_size={world_size}"
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
                "[DION_LINEAR_MISSING_FS_RANGE] "
                f"context={context} child_rows={split_rows} "
                f"param_uid={getattr(dist_meta, 'param_uid', None)} "
                f"param_name={getattr(dist_meta, 'param_name', '')}"
            )
        if parent_row_end - parent_row_start != int(local_rows):
            raise RuntimeError(
                "[DION_LINEAR_LOCAL_ROW_RANGE_MISMATCH] "
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
                "[DION_LINEAR_MISSING_GLOBAL_SHAPE] "
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
                "[DION_LINEAR_LOCAL_ROW_RANGE_MISMATCH] "
                f"context={context} local_rows={local_rows} "
                f"parent_row_range=({parent_row_start}, {parent_row_end})"
            )
        return parent_row_start, parent_row_end

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
) -> list[tuple[int, int, int, int]]:
    """Map one parent local tensor to source and child row intervals."""
    child_index = _linear_child_index(child_kind)
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
                    "[DION_LINEAR_STRIDED_TP_LOCAL_ROWS_MISMATCH] "
                    f"context={context} local_rows={local_rows} expected_rows={source_cursor} "
                    f"tp_world_size={tp_world_size} tp_rank={tp_rank} split_rows={split_rows}"
                )
            return segments

        if partition_stride != 1:
            raise RuntimeError(
                "[DION_LINEAR_UNSUPPORTED_TP_PARTITION_STRIDE] "
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
        )
        return [
            (
                int(child_source_start),
                int(child_source_start + child_rows),
                0,
                int(child_rows),
            )
        ]

    child_row_start = 0 if child_index == 0 else int(split_rows[0])
    child_row_end = child_row_start + int(split_rows[child_index])
    overlap_start = max(int(parent_row_start), child_row_start)
    overlap_end = min(int(parent_row_end), child_row_end)
    if overlap_end <= overlap_start:
        return []
    source_start = overlap_start - int(parent_row_start)
    child_start = overlap_start - child_row_start
    return [
        (
            int(source_start),
            int(source_start + overlap_end - overlap_start),
            int(child_start),
            int(child_start + overlap_end - overlap_start),
        )
    ]


def linear_child_has_local_overlap(
    split_rows: Tuple[int, int],
    dist_meta,
    child_kind: str,
) -> bool:
    """Return whether one gate/up child has a non-empty local shard on this rank."""
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
    )
    return any(source_end > source_start for source_start, source_end, _, _ in segments)


def linear_child_local_shape(
    parent_local_shape: Tuple[int, int],
    split_rows: Tuple[int, int],
    dist_meta,
    child_kind: str,
) -> Tuple[int, int]:
    """Return the local 2D shape for one gate/up child."""
    local_rows, local_cols = (int(parent_local_shape[0]), int(parent_local_shape[1]))
    segments = _linear_child_segments(
        local_rows=local_rows,
        split_rows=split_rows,
        dist_meta=dist_meta,
        child_kind=child_kind,
        context=f"local_shape:{child_kind}",
    )
    child_rows = sum(int(source_end - source_start) for source_start, source_end, _, _ in segments)
    if child_rows <= 0:
        raise RuntimeError(
            "[DION_LINEAR_EMPTY_CHILD_LOCAL_SHAPE] "
            f"parent_local_shape={parent_local_shape} child_kind={child_kind}"
        )
    return (child_rows, local_cols)


def read_linear_child(
    tensor: torch.Tensor,
    split_rows: Tuple[int, int],
    dist_meta,
    child_kind: str,
) -> torch.Tensor:
    """Read one gate/up child from a fused linear_fc1 tensor into a contiguous 2D tensor."""
    if tensor.ndim != 2:
        raise RuntimeError(
            "[DION_LINEAR_READ_REQUIRES_2D] "
            f"child_kind={child_kind} shape={tuple(int(dim) for dim in tensor.shape)}"
        )
    local_rows = int(tensor.size(0))
    segments = _linear_child_segments(
        local_rows=local_rows,
        split_rows=split_rows,
        dist_meta=dist_meta,
        child_kind=child_kind,
        context=f"read:{child_kind}",
    )
    if not segments:
        raise RuntimeError(
            "[DION_LINEAR_EMPTY_CHILD_LOCAL_SHAPE] "
            f"child_kind={child_kind} tensor_shape={tuple(int(dim) for dim in tensor.shape)}"
        )
    if len(segments) == 1:
        source_start, source_end, _, _ = segments[0]
        return tensor.narrow(0, source_start, source_end - source_start).contiguous()

    child_rows = sum(int(source_end - source_start) for source_start, source_end, _, _ in segments)
    child = tensor.new_empty((child_rows, int(tensor.size(1))))
    child_cursor = 0
    for source_start, source_end, _, _ in segments:
        rows_in_segment = int(source_end - source_start)
        child.narrow(0, child_cursor, rows_in_segment).copy_(
            tensor.narrow(0, source_start, rows_in_segment)
        )
        child_cursor += rows_in_segment
    return child


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
) -> None:
    """Write one gate/up child tensor back into the fused parent tensor."""
    if dest.ndim != 2 or child.ndim != 2:
        raise RuntimeError(
            "[DION_LINEAR_WRITE_REQUIRES_2D] "
            f"child_kind={child_kind} dest_shape={tuple(int(dim) for dim in dest.shape)} "
            f"child_shape={tuple(int(dim) for dim in child.shape)}"
        )
    expected_shape = linear_child_local_shape(
        parent_local_shape=(int(dest.size(0)), int(dest.size(1))),
        split_rows=split_rows,
        dist_meta=dist_meta,
        child_kind=child_kind,
    )
    if tuple(int(dim) for dim in child.shape) != expected_shape:
        raise RuntimeError(
            "[DION_LINEAR_CHILD_SHAPE_MISMATCH] "
            f"child_kind={child_kind} expected_shape={expected_shape} "
            f"child_shape={tuple(int(dim) for dim in child.shape)}"
        )

    local_rows = int(dest.size(0))
    segments = _linear_child_segments(
        local_rows=local_rows,
        split_rows=split_rows,
        dist_meta=dist_meta,
        child_kind=child_kind,
        context=f"write:{child_kind}",
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


def iter_linear_child_kinds() -> Iterable[str]:
    """Yield linear child kinds in canonical order."""
    return LINEAR_CHILD_KINDS
