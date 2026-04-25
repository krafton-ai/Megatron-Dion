"""Shared Dion optimizer functions."""

from typing import Optional, Tuple

import torch
from .types import DionDistMeta


def get_global_shape(
    dist_meta: Optional[DionDistMeta],
    m_local: int,
    n_local: int,
) -> Tuple[int, int]:
    """Get the global shape used by Dion math.

    For expert parameters, `per_expert_global_shape` is the matrix shape
    that defines the Dion object. For non-expert parameters, this falls back to
    the full global shape.

    Args:
        dist_meta: Distribution metadata for the parameter
        m_local: Local m dimension
        n_local: Local n dimension

    Returns:
        Tuple `(m_global, n_global)`
    """
    if dist_meta is not None:
        if getattr(dist_meta, 'per_expert_global_shape', None) is not None:
            return tuple(dist_meta.per_expert_global_shape)
        if getattr(dist_meta, 'global_shape', None) is not None:
            return tuple(dist_meta.global_shape)
        if getattr(dist_meta, 'is_dion_param', False):
            raise RuntimeError(
                "Dion distributed param is missing global_shape needed to compute "
                f"LR/rank scaling: local_shape=({m_local}, {n_local}) "
                f"param={getattr(dist_meta, 'param_name', '')}"
            )
    # Local non-distributed case.
    return (m_local, n_local)


def get_local_shape(
    dist_meta: Optional[DionDistMeta],
    m_local: int,
    n_local: int,
) -> Tuple[int, int]:
    """Return the Dion-object local matrix shape.

    For combined expert tensors, the physical shard may contain multiple local
    experts while the Dion object is one expert matrix. When available,
    `dist_meta.local_shape` is the authoritative object-local shape.
    """
    if dist_meta is not None and getattr(dist_meta, "local_shape", None) is not None:
        return tuple(int(dim) for dim in dist_meta.local_shape)
    return (m_local, n_local)


def has_multiple_local_experts(dist_meta: Optional[DionDistMeta]) -> bool:
    """Return whether one local tensor holds multiple expert Dion objects."""
    if dist_meta is None:
        return False
    return (
        int(getattr(dist_meta, "expert_axis", -1)) in (0, 1)
        and int(getattr(dist_meta, "num_local_experts", 1)) > 1
        and int(getattr(dist_meta, "local_expert_index", -1)) >= 0
    )


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
    """Return a compact parameter identifier for reporting."""
    if dist_meta is None:
        return {"param_uid": None, "param_name": ""}
    return {
        "param_uid": getattr(dist_meta, "param_uid", None),
        "param_name": getattr(dist_meta, "param_name", ""),
    }
