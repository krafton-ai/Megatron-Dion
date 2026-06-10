"""Shared axis helpers for optimizer-only fused-parameter splits."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Optional, Tuple

import torch


def normalize_split_axis(split_axis: int | None) -> int:
    """Return a validated 2D split axis."""
    split_axis = 0 if split_axis is None else int(split_axis)
    if split_axis not in (0, 1):
        raise RuntimeError(f"[MATRIX_INVALID_SPLIT_AXIS] split_axis={split_axis}")
    return split_axis


def get_split_axis_from_param(param: torch.Tensor, attr: str, *, default: int = 0) -> int:
    """Return a layout split axis from a param or its owning model param."""
    split_axis = getattr(param, attr, None)
    if split_axis is None:
        model_param = getattr(param, "_model_param", None)
        split_axis = getattr(model_param, attr, None)
    return normalize_split_axis(default if split_axis is None else split_axis)


def get_split_axis_from_dist_meta(dist_meta, attr: str, *, default: int = 0) -> int:
    """Return a layout split axis from distributed metadata."""
    if dist_meta is None:
        return normalize_split_axis(default)
    return normalize_split_axis(getattr(dist_meta, attr, default))


def get_split_axis_from_state(state: Optional[dict], attr: str, *, default: int = 0) -> int:
    """Return a layout split axis from optimizer state."""
    if not state:
        return normalize_split_axis(default)
    return normalize_split_axis(state.get(attr, default))


def resolve_split_axis(
    *,
    attr: str,
    param: Optional[torch.Tensor] = None,
    optimizer_state: Optional[dict] = None,
    dist_meta=None,
    default: int = 0,
) -> int:
    """Resolve a split axis from state, dist-meta, then param attrs."""
    if optimizer_state and attr in optimizer_state:
        return get_split_axis_from_state(optimizer_state, attr, default=default)
    if dist_meta is not None and hasattr(dist_meta, attr):
        return get_split_axis_from_dist_meta(dist_meta, attr, default=default)
    if param is not None:
        return get_split_axis_from_param(param, attr, default=default)
    return normalize_split_axis(default)


def copy_split_axis(destination_tensor: torch.Tensor, source_tensor: torch.Tensor, attr: str) -> None:
    """Copy a layout split axis if present on the source tensor."""
    if hasattr(source_tensor, attr):
        setattr(destination_tensor, attr, normalize_split_axis(getattr(source_tensor, attr)))
        return
    model_param = getattr(source_tensor, "_model_param", None)
    if model_param is not None and hasattr(model_param, attr):
        setattr(destination_tensor, attr, normalize_split_axis(getattr(model_param, attr)))


def axis_shape(shape: Tuple[int, int], split_axis: int) -> Tuple[int, int]:
    """Return shape in split-axis coordinates."""
    split_axis = normalize_split_axis(split_axis)
    shape = tuple(int(dim) for dim in shape)
    if len(shape) != 2:
        raise RuntimeError(f"[MATRIX_SPLIT_AXIS_REQUIRES_2D] shape={shape}")
    return shape if split_axis == 0 else (shape[1], shape[0])


def original_shape(axis_shape_: Tuple[int, int], split_axis: int) -> Tuple[int, int]:
    """Return original tensor shape from split-axis coordinates."""
    return axis_shape(axis_shape_, split_axis)


def axis_tensor(tensor: torch.Tensor, split_axis: int) -> torch.Tensor:
    """Return a tensor view whose row axis is the fused split axis."""
    split_axis = normalize_split_axis(split_axis)
    if tensor.ndim != 2:
        raise RuntimeError(
            "[MATRIX_SPLIT_AXIS_REQUIRES_2D] "
            f"shape={tuple(int(dim) for dim in tensor.shape)}"
        )
    return tensor if split_axis == 0 else tensor.transpose(0, 1)


def original_tensor(tensor: torch.Tensor, split_axis: int) -> torch.Tensor:
    """Return a contiguous tensor in original coordinates."""
    split_axis = normalize_split_axis(split_axis)
    return tensor if split_axis == 0 else tensor.transpose(0, 1).contiguous()


def _swap_dim(dim: int) -> int:
    if int(dim) == 0:
        return 1
    if int(dim) == 1:
        return 0
    return int(dim)


def dist_meta_for_split_axis(dist_meta, split_axis: int):
    """Return metadata viewed with split axis as row axis."""
    split_axis = normalize_split_axis(split_axis)
    if dist_meta is None or split_axis == 0:
        return dist_meta

    attrs = dict(getattr(dist_meta, "__dict__", {}))
    if not attrs:
        return dist_meta

    for attr in ("shape", "local_shape", "global_shape", "per_expert_global_shape"):
        value = attrs.get(attr, None)
        if value is not None and len(value) == 2:
            attrs[attr] = (int(value[1]), int(value[0]))

    attrs["fs_shard_dim"] = _swap_dim(attrs.get("fs_shard_dim", -1))
    attrs["tp_shard_dim"] = _swap_dim(attrs.get("tp_shard_dim", -1))
    return SimpleNamespace(**attrs)


def project_child_shard_range(
    *,
    parent_start: int,
    parent_end: int,
    shard_dim: int,
    split_axis: int,
    project_split_axis_range,
) -> Optional[Tuple[int, int]]:
    """Project a parent shard interval to a child interval.

    When the shard axis is not the fused split axis, every child owns the same
    parent interval along that orthogonal axis.
    """
    split_axis = normalize_split_axis(split_axis)
    if int(shard_dim) == split_axis:
        return project_split_axis_range(int(parent_start), int(parent_end))
    return int(parent_start), int(parent_end)
