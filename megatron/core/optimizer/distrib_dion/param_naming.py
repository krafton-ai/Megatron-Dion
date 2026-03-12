"""Parameter naming helpers for diagnostics/logging."""

from __future__ import annotations

from typing import Callable, Optional

import torch


def get_param_name(param: torch.Tensor, fallback_style: str = "shape") -> str:
    """Best-effort stable name for a parameter.

    This is only used for logs/diagnostics, so it must never throw.
    """
    name = getattr(param, "_param_name", None)
    if name is not None:
        return name

    if fallback_style == "id":
        return f"id_{id(param)}"
    if fallback_style == "both":
        return f"shape={param.shape}"
    # "shape" default
    return str(param.shape)


def get_optimizer_param_name(
    param: torch.Tensor,
    *,
    primary_name_fn: Optional[Callable[[torch.Tensor], str]] = None,
    param_to_name=None,
    buffers=None,
) -> str:
    """Best-effort parameter name for optimizer/debug plumbing.

    Order of preference:
    1. `primary_name_fn` (for example `DistributedOptimizer._param_name`)
    2. optimizer-level `param_to_name`
    3. buffer-level `param_to_name`
    4. `id_<ptr>` fallback
    """
    if primary_name_fn is not None:
        try:
            return primary_name_fn(param)
        except Exception:
            pass

    try:
        if param_to_name is not None and param in param_to_name:
            return param_to_name[param]
    except Exception:
        pass

    try:
        for buffer in buffers or []:
            if hasattr(buffer, "param_to_name") and param in buffer.param_to_name:
                return buffer.param_to_name[param]
    except Exception:
        pass

    return f"id_{id(param)}"
