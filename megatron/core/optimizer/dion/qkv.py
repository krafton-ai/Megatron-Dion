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
            "dion_split_qkv currently requires a clean local grouped-QKV row layout."
        )
    return rows // total_per_group


def qkv_child_global_shape(
    parent_global_shape: Tuple[int, int],
    split_shapes: Tuple[int, int, int],
    child_kind: str,
) -> Tuple[int, int]:
    """Return the logical global 2D shape for one Q/K/V child."""
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
) -> Tuple[int, int]:
    """Return the local packed 2D shape for one Q/K/V child."""
    local_rows, local_cols = (int(parent_local_shape[0]), int(parent_local_shape[1]))
    local_query_groups = _validate_qkv_layout_rows(
        rows=local_rows,
        split_shapes=split_shapes,
        context="local_shape",
    )
    child_rows = int(split_shapes[_qkv_child_index(child_kind)]) * int(local_query_groups)
    return (child_rows, local_cols)


def _qkv_child_slice_bounds(
    split_shapes: Tuple[int, int, int],
    child_kind: str,
) -> Tuple[int, int, int]:
    child_index = _qkv_child_index(child_kind)
    child_rows = int(split_shapes[child_index])
    start = sum(int(split_shapes[idx]) for idx in range(child_index))
    end = start + child_rows
    return start, end, child_rows


def _shares_storage(lhs: torch.Tensor, rhs: torch.Tensor) -> bool:
    """Return whether two tensors are backed by the same storage."""
    if lhs.numel() == 0 or rhs.numel() == 0:
        return False
    return lhs.untyped_storage().data_ptr() == rhs.untyped_storage().data_ptr()


def extract_qkv_child(
    tensor: torch.Tensor,
    split_shapes: Tuple[int, int, int],
    child_kind: str,
) -> torch.Tensor:
    """Pack one logical Q/K/V child from a fused QKV tensor into a contiguous 2D tensor."""
    if tensor.ndim != 2:
        raise RuntimeError(
            "[DION_QKV_PACK_REQUIRES_2D] "
            f"child_kind={child_kind} shape={tuple(int(dim) for dim in tensor.shape)}"
        )
    rows, cols = int(tensor.size(0)), int(tensor.size(1))
    num_query_groups = _validate_qkv_layout_rows(
        rows=rows,
        split_shapes=split_shapes,
        context=f"extract:{child_kind}",
    )
    start, end, child_rows_per_group = _qkv_child_slice_bounds(split_shapes, child_kind)
    grouped = tensor.contiguous().view(num_query_groups, int(sum(split_shapes)), cols)
    child = grouped[:, start:end, :].contiguous().reshape(
        num_query_groups * child_rows_per_group,
        cols,
    )
    return child


def scatter_qkv_child_(
    dest: torch.Tensor,
    child: torch.Tensor,
    split_shapes: Tuple[int, int, int],
    child_kind: str,
) -> None:
    """Scatter one packed Q/K/V child back into the fused parent tensor."""
    if dest.ndim != 2 or child.ndim != 2:
        raise RuntimeError(
            "[DION_QKV_SCATTER_REQUIRES_2D] "
            f"child_kind={child_kind} dest_shape={tuple(int(dim) for dim in dest.shape)} "
            f"child_shape={tuple(int(dim) for dim in child.shape)}"
        )
    rows, cols = int(dest.size(0)), int(dest.size(1))
    num_query_groups = _validate_qkv_layout_rows(
        rows=rows,
        split_shapes=split_shapes,
        context=f"scatter:{child_kind}",
    )
    start, end, child_rows_per_group = _qkv_child_slice_bounds(split_shapes, child_kind)
    expected_child_shape = (num_query_groups * child_rows_per_group, cols)
    if tuple(int(dim) for dim in child.shape) != expected_child_shape:
        raise RuntimeError(
            "[DION_QKV_SCATTER_CHILD_SHAPE_MISMATCH] "
            f"child_kind={child_kind} expected_child_shape={expected_child_shape} "
            f"child_shape={tuple(int(dim) for dim in child.shape)}"
        )
    if not dest.is_contiguous():
        raise RuntimeError(
            "[DION_QKV_SCATTER_REQUIRES_CONTIGUOUS_DEST] "
            f"child_kind={child_kind} dest_shape={tuple(int(dim) for dim in dest.shape)}"
        )
    grouped_dest = dest.view(num_query_groups, int(sum(split_shapes)), cols)
    grouped_child = child.contiguous().view(num_query_groups, child_rows_per_group, cols)
    if _shares_storage(dest, child):
        # Split-QKV can write a child tensor back into the same fused storage
        # it was read from; clone first so values stay stable.
        grouped_child = grouped_child.clone()
    grouped_dest[:, start:end, :].copy_(grouped_child)


def iter_qkv_child_kinds() -> Iterable[str]:
    """Yield QKV child kinds in canonical order."""
    return QKV_CHILD_KINDS
