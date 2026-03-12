"""Dion vs non-Dion parameter classification.

Keep this logic centralized so classification is consistent across the
distributed optimizer plumbing.
"""

from __future__ import annotations

from typing import Optional

import torch

from ...fp8_utils import is_float8tensor


# Keywords that exclude a 2D parameter from Dion classification.
#
# Note: router and experts are included in Dion (intentionally not excluded).
DION_EXCLUDE_KEYWORDS = [
    "embedding",
    "word_embeddings",
    "position_embeddings",
    "output_layer",
    "lm_head",
    "vocab",
    "norm",
    "layernorm",
    "rmsnorm",
    "groupnorm",
    "batchnorm",
]


def is_dion_param(param: torch.Tensor, param_name: Optional[str] = None) -> bool:
    """Return True iff this parameter should use Dion FS sharding.

    Rule:
    - Manual override: `param.use_dion is False` disables Dion.
    - Must be 2D.
    - Exclude FP8 tensors (handled by standard DO path).
    - Exclude known embedding/output/norm parameters by name.
    """
    if getattr(param, "use_dion", None) is False:
        return False

    if param.ndim != 2:
        return False

    if is_float8tensor(param):
        return False

    if param_name:
        name_lower = param_name.lower()
        for keyword in DION_EXCLUDE_KEYWORDS:
            if keyword in name_lower:
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
