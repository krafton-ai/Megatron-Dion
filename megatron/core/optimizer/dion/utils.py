"""Utility functions for Dion optimizer."""

from typing import Optional, Tuple

import torch
from .types import DionDistMeta


def get_global_shape(
    dist_meta: Optional[DionDistMeta],
    local_m: int,
    local_n: int,
) -> Tuple[int, int]:
    """Get fully global shape (FS and TP restored) for rank calculation.

    dist_meta.global_shape is already fully global when set by DistributedOptimizer.

    Args:
        dist_meta: Distribution metadata for the parameter
        local_m: Local m dimension
        local_n: Local n dimension

    Returns:
        Tuple of (global_m, global_n)
    """
    if dist_meta is not None:
        if getattr(dist_meta, 'global_shape', None) is not None:
            return tuple(dist_meta.global_shape)
        if getattr(dist_meta, 'is_dion_param', False):
            raise RuntimeError(
                "Dion distributed param is missing global_shape required for "
                f"LR/rank scaling: local_shape=({local_m}, {local_n}) "
                f"param={getattr(dist_meta, 'param_name', '')}"
            )
    # Local non-distributed case.
    return (local_m, local_n)


def str_to_dtype(dtype_val) -> Optional[torch.dtype]:
    """Convert string dtype to torch.dtype if needed.

    Args:
        dtype_val: String dtype name or torch.dtype

    Returns:
        torch.dtype or None
    """
    if dtype_val is None:
        return None
    if isinstance(dtype_val, torch.dtype):
        return dtype_val
    # Handle string dtype (from config YAML)
    dtype_map = {
        'float32': torch.float32,
        'float': torch.float32,
        'fp32': torch.float32,
        'float16': torch.float16,
        'fp16': torch.float16,
        'half': torch.float16,
        'bfloat16': torch.bfloat16,
        'bf16': torch.bfloat16,
    }
    if isinstance(dtype_val, str):
        dtype_lower = dtype_val.lower()
        if dtype_lower.startswith("torch."):
            dtype_lower = dtype_lower.split(".", 1)[1]
        if dtype_lower in dtype_map:
            return dtype_map[dtype_lower]
        raise ValueError(f"Unknown dtype string: {dtype_val}")
    return dtype_val


def format_meta_id(dist_meta: Optional[DionDistMeta]) -> dict[str, object]:
    """Return a compact logical parameter identifier for reporting."""
    if dist_meta is None:
        return {"param_uid": None, "param_name": ""}
    return {
        "param_uid": getattr(dist_meta, "param_uid", None),
        "param_name": getattr(dist_meta, "param_name", ""),
    }
