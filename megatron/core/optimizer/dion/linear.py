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
    child_kind: str,
) -> Tuple[int, int]:
    if local_rows % 2 != 0:
        raise RuntimeError(
            "[DION_LINEAR_LOCAL_ROWS_MUST_BE_EVEN] "
            f"local_rows={local_rows} child_kind={child_kind}"
        )
    half_rows = local_rows // 2
    if half_rows <= 0:
        raise RuntimeError(
            "[DION_LINEAR_EMPTY_CHILD_LOCAL_SHAPE] "
            f"local_rows={local_rows} child_kind={child_kind}"
        )
    if child_kind == "gate":
        return 0, half_rows
    return half_rows, half_rows


def _row_shard_linear_rows(
    *,
    parent_row_start: int,
    parent_row_end: int,
    split_rows: Tuple[int, int],
    child_kind: str,
) -> Tuple[int, int]:
    child_index = _linear_child_index(child_kind)
    child_row_start = 0 if child_index == 0 else int(split_rows[0])
    child_row_end = child_row_start + int(split_rows[child_index])
    overlap_start = max(parent_row_start, child_row_start)
    overlap_end = min(parent_row_end, child_row_end)
    overlap_rows = overlap_end - overlap_start
    if overlap_rows <= 0:
        raise RuntimeError(
            "[DION_LINEAR_EMPTY_CHILD_LOCAL_SHAPE] "
            f"parent_row_start={parent_row_start} parent_row_end={parent_row_end} "
            f"child_kind={child_kind} split_rows={split_rows}"
        )
    return overlap_start - parent_row_start, overlap_rows


def linear_child_has_local_overlap(
    split_rows: Tuple[int, int],
    dist_meta,
    child_kind: str,
) -> bool:
    """Return whether one gate/up child has a non-empty local shard on this rank."""
    fs_shard_dim = int(getattr(dist_meta, "fs_shard_dim", -1)) if dist_meta is not None else -1
    fs_world_size = int(getattr(dist_meta, "fs_world_size", 1)) if dist_meta is not None else 1
    if fs_shard_dim != 0 or fs_world_size <= 1:
        return True

    parent_row_start = int(getattr(dist_meta, "fs_start_idx", -1))
    parent_row_end = int(getattr(dist_meta, "fs_end_idx", -1))
    if parent_row_start < 0 or parent_row_end < parent_row_start:
        raise RuntimeError(
            "[DION_LINEAR_MISSING_FS_RANGE] "
            f"child_kind={child_kind} "
            f"param_uid={getattr(dist_meta, 'param_uid', None)} "
            f"param_name={getattr(dist_meta, 'param_name', '')}"
        )

    child_index = _linear_child_index(child_kind)
    child_row_start = 0 if child_index == 0 else int(split_rows[0])
    child_row_end = child_row_start + int(split_rows[child_index])
    return min(parent_row_end, child_row_end) > max(parent_row_start, child_row_start)


def linear_child_local_shape(
    parent_local_shape: Tuple[int, int],
    split_rows: Tuple[int, int],
    dist_meta,
    child_kind: str,
) -> Tuple[int, int]:
    """Return the local 2D shape for one gate/up child."""
    local_rows, local_cols = (int(parent_local_shape[0]), int(parent_local_shape[1]))
    tp_shard_dim = int(getattr(dist_meta, "tp_shard_dim", -1)) if dist_meta is not None else -1
    tp_world_size = int(getattr(dist_meta, "tp_world_size", 1)) if dist_meta is not None else 1
    fs_shard_dim = int(getattr(dist_meta, "fs_shard_dim", -1)) if dist_meta is not None else -1
    fs_world_size = int(getattr(dist_meta, "fs_world_size", 1)) if dist_meta is not None else 1

    if tp_shard_dim == 0 and tp_world_size > 1:
        _, child_rows = _direct_linear_rows(local_rows=local_rows, child_kind=child_kind)
        return (child_rows, local_cols)

    if fs_shard_dim == 0 and fs_world_size > 1:
        parent_row_start = int(getattr(dist_meta, "fs_start_idx", -1))
        parent_row_end = int(getattr(dist_meta, "fs_end_idx", -1))
        if parent_row_start < 0 or parent_row_end < parent_row_start:
            raise RuntimeError(
                "[DION_LINEAR_MISSING_FS_RANGE] "
                f"child_kind={child_kind} "
                f"param_uid={getattr(dist_meta, 'param_uid', None)} "
                f"param_name={getattr(dist_meta, 'param_name', '')}"
            )
        _, child_rows = _row_shard_linear_rows(
            parent_row_start=parent_row_start,
            parent_row_end=parent_row_end,
            split_rows=split_rows,
            child_kind=child_kind,
        )
        return (child_rows, local_cols)

    _, child_rows = _direct_linear_rows(local_rows=local_rows, child_kind=child_kind)
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
    tp_shard_dim = int(getattr(dist_meta, "tp_shard_dim", -1)) if dist_meta is not None else -1
    tp_world_size = int(getattr(dist_meta, "tp_world_size", 1)) if dist_meta is not None else 1
    fs_shard_dim = int(getattr(dist_meta, "fs_shard_dim", -1)) if dist_meta is not None else -1
    fs_world_size = int(getattr(dist_meta, "fs_world_size", 1)) if dist_meta is not None else 1

    if tp_shard_dim == 0 and tp_world_size > 1:
        start_row, child_rows = _direct_linear_rows(local_rows=local_rows, child_kind=child_kind)
        return tensor.narrow(0, start_row, child_rows).contiguous()

    if fs_shard_dim == 0 and fs_world_size > 1:
        parent_row_start = int(getattr(dist_meta, "fs_start_idx", -1))
        parent_row_end = int(getattr(dist_meta, "fs_end_idx", -1))
        start_row, child_rows = _row_shard_linear_rows(
            parent_row_start=parent_row_start,
            parent_row_end=parent_row_end,
            split_rows=split_rows,
            child_kind=child_kind,
        )
        return tensor.narrow(0, start_row, child_rows).contiguous()

    start_row, child_rows = _direct_linear_rows(local_rows=local_rows, child_kind=child_kind)
    return tensor.narrow(0, start_row, child_rows).contiguous()


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
    tp_shard_dim = int(getattr(dist_meta, "tp_shard_dim", -1)) if dist_meta is not None else -1
    tp_world_size = int(getattr(dist_meta, "tp_world_size", 1)) if dist_meta is not None else 1
    fs_shard_dim = int(getattr(dist_meta, "fs_shard_dim", -1)) if dist_meta is not None else -1
    fs_world_size = int(getattr(dist_meta, "fs_world_size", 1)) if dist_meta is not None else 1

    if tp_shard_dim == 0 and tp_world_size > 1:
        start_row, child_rows = _direct_linear_rows(local_rows=local_rows, child_kind=child_kind)
        dest.narrow(0, start_row, child_rows).copy_(child)
        return

    if fs_shard_dim == 0 and fs_world_size > 1:
        parent_row_start = int(getattr(dist_meta, "fs_start_idx", -1))
        parent_row_end = int(getattr(dist_meta, "fs_end_idx", -1))
        start_row, child_rows = _row_shard_linear_rows(
            parent_row_start=parent_row_start,
            parent_row_end=parent_row_end,
            split_rows=split_rows,
            child_kind=child_kind,
        )
        dest.narrow(0, start_row, child_rows).copy_(child)
        return

    start_row, child_rows = _direct_linear_rows(local_rows=local_rows, child_kind=child_kind)
    dest.narrow(0, start_row, child_rows).copy_(child)


def iter_linear_child_kinds() -> Iterable[str]:
    """Yield linear child kinds in canonical order."""
    return LINEAR_CHILD_KINDS
