"""Utility functions for Dion optimizer."""

from typing import Optional, Tuple

import torch
from torch import Tensor

from .types import MegatronDionDistMeta


def get_global_shape(
    meta: Optional[MegatronDionDistMeta],
    local_m: int,
    local_n: int,
) -> Tuple[int, int]:
    """Get fully global shape (FS and TP restored) for rank calculation.

    meta.global_shape is already fully global when set by DistributedOptimizer.

    Args:
        meta: Distribution metadata for the parameter
        local_m: Local m dimension
        local_n: Local n dimension

    Returns:
        Tuple of (global_m, global_n)
    """
    if meta is not None and getattr(meta, 'global_shape', None) is not None:
        return tuple(meta.global_shape)
    # Fallback: use local shape (no sharding)
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
        if dtype_lower in dtype_map:
            return dtype_map[dtype_lower]
        raise ValueError(f"Unknown dtype string: {dtype_val}")
    return dtype_val


def infer_local_2d_shape(
    p: Tensor,
    meta: MegatronDionDistMeta,
) -> Optional[Tuple[int, int]]:
    """Infer local 2D shape from flattened parameter in Distributed mode.

    Args:
        p: Parameter tensor (may be flattened)
        meta: Distribution metadata for the parameter

    Returns:
        Tuple of (m_local, n_local) or None if not a valid 2D parameter
    """
    if meta is None or meta.global_shape is None or len(meta.global_shape) != 2:
        return None

    m_g, n_g = meta.global_shape  # noqa: F841
    elems = p.numel()

    if meta.tp_split_dim == 1:
        if meta.shape is None or len(meta.shape) != 2:
            raise RuntimeError("meta.shape is missing for tp_split_dim == 1")
        n_local = meta.shape[1]
        if elems % n_local != 0:
            raise RuntimeError(f"FS slice size {elems} not divisible by n_local {n_local}")
        m_local = elems // n_local
        return m_local, n_local
    elif meta.tp_split_dim == 0:
        if meta.shape is None or len(meta.shape) != 2:
            raise RuntimeError("meta.shape is missing for tp_split_dim == 0")
        m_local_pre_fs = meta.shape[0]
        if elems % m_local_pre_fs != 0:
            raise RuntimeError(f"FS slice size {elems} not divisible by m_local_pre_fs {m_local_pre_fs}")
        n_local = elems // m_local_pre_fs
        return m_local_pre_fs, n_local
    else:
        n_local = n_g
        if elems % n_local != 0:
            raise RuntimeError(f"FS slice size {elems} not divisible by n_global {n_local}")
        m_local = elems // n_local
        return m_local, n_local
