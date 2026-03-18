"""Dion parameter classification.

Keep this logic centralized so DDP, distributed optimizer setup, and focused
validation scripts all classify the same params the same way.
"""

from __future__ import annotations

from typing import Optional

import torch

from ...fp8_utils import is_float8tensor
def annotate_dion_candidates(module: torch.nn.Module) -> None:
    """Mark all local parameters as potential Dion candidates.

    The final Dion decision remains centralized in `is_dion_param()`, which
    applies the dimensionality / dtype / manual-disable checks. Marking every
    parameter here lets the runtime opt any eligible 2D parameter into Dion,
    including embeddings, lm heads, projector layers, and other trainable 2D
    surfaces.
    """
    for param in module.parameters():
        param.dion_candidate = True


def is_dion_param(param: torch.Tensor, param_name: Optional[str] = None) -> bool:
    """Return True iff this parameter should use Dion FS sharding.

    Rule:
    - Manual override: `param.use_dion is False` disables Dion.
    - Must be 2D.
    - Exclude embedding/output and lm-head surfaces (scalar path).
    - Exclude FP8 tensors (handled by standard DO path).
    """
    if getattr(param, "use_dion", None) is False:
        return False

    if not getattr(param, "dion_candidate", False):
        return False

    if param.ndim != 2:
        return False

    if getattr(param, "is_embedding_or_output_parameter", False):
        return False

    if getattr(param, "is_lm_head_parameter", False):
        return False

    if is_float8tensor(param):
        return False

    return True


def is_moe_expert_param(param: torch.Tensor, param_name: Optional[str] = None) -> bool:
    """Return True iff this parameter is a MoE expert weight.

    This is intentionally distinct from Megatron-Core's expert-parallel routing
    (`allreduce=False`). In EP=1, expert weights still participate in standard
    dense communication groups, but they remain MoE expert parameters for Dion
    math such as per-expert LR scaling and sparse zero-grad handling.
    """
    num_local_experts = getattr(param, "num_local_experts", None)
    if num_local_experts is not None and int(num_local_experts) > 1:
        return True

    resolved_name = param_name or getattr(param, "_param_name", None)
    if resolved_name and ".experts." in resolved_name:
        return True

    return False
