"""Dion parameter classification helpers."""

from __future__ import annotations

from typing import Optional

import torch

from ..matrix.parameter import (
    is_matrix_param,
    is_vocab_param,
    mark_matrix_bucket_params,
    prepare_matrix_params,
)


def prepare_dion_params(module: torch.nn.Module) -> None:
    """Prepare local parameters for Dion routing."""
    prepare_matrix_params(module)
    for param in module.parameters():
        param.use_dion = is_dion_matrix_param(param)


def is_dion_matrix_param(param: torch.Tensor, param_name: Optional[str] = None) -> bool:
    """Return True iff this parameter should use Dion matrix updates."""
    if getattr(param, "use_dion", None) is False:
        return False
    return is_matrix_param(param, param_name) and not is_vocab_param(param)


def mark_dion_bucket_params(param_map, param_to_name, fs_size: int, *, tp_group=None):
    """Classify bucket params and build static Dion metadata once."""
    dion_param_count, matrix_info_by_param = mark_matrix_bucket_params(
        param_map=param_map,
        param_to_name=param_to_name,
        fs_size=fs_size,
        include_vocab=False,
        tp_group=tp_group,
    )
    for param in param_map.keys():
        use_dion = (
            getattr(param, "use_dion", None) is not False
            and bool(getattr(param, "is_matrix_param", False))
        )
        param.is_dion_param = bool(use_dion)
        param.use_dion = bool(use_dion)
        param.is_matrix_param = bool(param.is_dion_param)
        if not param.is_dion_param and param in matrix_info_by_param:
            matrix_info_by_param.pop(param, None)
            dion_param_count = max(0, int(dion_param_count) - 1)
    return dion_param_count, matrix_info_by_param
