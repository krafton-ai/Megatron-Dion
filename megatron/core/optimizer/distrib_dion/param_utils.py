"""Small parameter helpers used by Dion distributed optimizer plumbing."""

from __future__ import annotations

import torch


def get_tp_split_dim(param: torch.Tensor) -> int:
    """Return TP split dim in Dion naming convention.

    Megatron stores TP split information via the `partition_dim` attribute.
    """
    return getattr(param, "partition_dim", -1)


def is_tp_enabled(param: torch.Tensor) -> bool:
    """Return True iff this parameter participates in Tensor Parallelism."""
    return bool(getattr(param, "tensor_model_parallel", False))

