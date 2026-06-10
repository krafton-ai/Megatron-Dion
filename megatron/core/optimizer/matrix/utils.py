"""Small shared utilities for matrix-aware optimizers."""

from __future__ import annotations

import os
from typing import Optional

import torch


def env_flag(name: str, default: bool = False) -> bool:
    """Return a boolean environment flag."""
    value = os.getenv(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in ("1", "true", "yes", "on")


def str_to_dtype(dtype_val) -> Optional[torch.dtype]:
    """Convert string dtype names from configs into torch dtypes."""
    if dtype_val is None:
        return None
    if isinstance(dtype_val, torch.dtype):
        return dtype_val
    if isinstance(dtype_val, str):
        dtype_lower = dtype_val.lower()
        if dtype_lower.startswith("torch."):
            dtype_lower = dtype_lower.split(".", 1)[1]
        dtype_map = {
            "float32": torch.float32,
            "float": torch.float32,
            "fp32": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
            "half": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        if dtype_lower in dtype_map:
            return dtype_map[dtype_lower]
        raise ValueError(f"Unknown dtype string: {dtype_val}")
    return dtype_val
