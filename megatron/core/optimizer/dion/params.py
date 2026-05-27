"""Dion parameter classification helpers."""

from __future__ import annotations

from typing import Optional

import torch

from ... import parallel_state
from ...fp8_utils import is_float8tensor
from ..matrix.parameter import (
    has_explicit_expert_index,
    is_combined_grouped_mlp_param,
    is_moe_expert_param,
    is_unindexed_multi_local_expert_param,
)
from ..matrix.sharding import get_fs_split_dim, get_tp_split_dim, is_tp_enabled


def mark_dion_candidates(module: torch.nn.Module) -> None:
    """Mark all local parameters as potential Dion candidates."""
    for param in module.parameters():
        param.dion_candidate = True
        param.matrix_optimizer_candidate = True


def is_dion_param(param: torch.Tensor, param_name: Optional[str] = None) -> bool:
    """Return True iff this parameter should use Dion matrix updates."""
    resolved_name = param_name or getattr(param, "_param_name", None)
    if getattr(param, "use_dion", None) is False:
        return False
    if not getattr(param, "matrix_optimizer_candidate", getattr(param, "dion_candidate", False)):
        return False
    if param.ndim != 2:
        return False
    if getattr(param, "sequence_parallel", False):
        return False
    if getattr(param, "average_gradients_across_tp_domain", False):
        return False
    if getattr(param, "is_embedding_or_output_parameter", False):
        return False
    if getattr(param, "is_lm_head_parameter", False):
        return False
    if is_float8tensor(param):
        return False
    if is_combined_grouped_mlp_param(param, resolved_name):
        return False
    if is_unindexed_multi_local_expert_param(param, resolved_name):
        return False
    return True


def mark_dion_bucket_params(param_map, param_to_name, fs_size: int):
    """Classify bucket params and build static Dion metadata once."""
    fs_size = int(fs_size)
    if fs_size <= 0:
        raise RuntimeError(f"[Dion] invalid FS size while marking bucket params: {fs_size}")

    dion_param_count = 0
    dion_info_by_param = {}

    for param in param_map.keys():
        param_name = None
        if param_to_name is not None and param in param_to_name:
            param_name = param_to_name[param]
        if param_name:
            param._param_name = param_name

        fallback_to_scalar = (
            is_combined_grouped_mlp_param(param, param_name)
            or is_unindexed_multi_local_expert_param(param, param_name)
        )
        param.is_dion_param = is_dion_param(param, param_name)
        param.is_matrix_param = bool(param.is_dion_param)
        if not param.is_dion_param and fallback_to_scalar:
            param.dion_candidate = False
            param.matrix_optimizer_candidate = False

        is_expert = is_moe_expert_param(param, param_name)
        raw_tp_split_dim = get_tp_split_dim(param)
        has_tp = is_tp_enabled(param)
        if is_expert and has_tp:
            tp_world_size = parallel_state.get_expert_tensor_parallel_world_size()
        else:
            tp_world_size = parallel_state.get_tensor_model_parallel_world_size() if has_tp else 1
        tp_shard_dim = raw_tp_split_dim if has_tp and tp_world_size > 1 else -1

        if not param.is_dion_param:
            continue

        m_local, n_local = (int(dim) for dim in param.shape)
        fs_shard_dim = get_fs_split_dim(tp_shard_dim)
        split_size = m_local if fs_shard_dim == 0 else n_local
        if fs_size > 1 and split_size < fs_size:
            param.is_dion_param = False
            param.is_matrix_param = False
            param.dion_candidate = False
            param.matrix_optimizer_candidate = False
            continue

        dion_param_count += 1

        if tp_shard_dim == 0:
            m_global = m_local * tp_world_size
            n_global = n_local
        elif tp_shard_dim == 1:
            m_global = m_local
            n_global = n_local * tp_world_size
        else:
            m_global = m_local
            n_global = n_local

        num_local_experts = getattr(param, "num_local_experts", None)
        is_explicit_expert = has_explicit_expert_index(param_name)
        if num_local_experts is not None and num_local_experts > 1 and not is_explicit_expert:
            if tp_shard_dim == 0:
                per_expert_global_shape = (m_global // num_local_experts, n_global)
            elif tp_shard_dim == 1:
                per_expert_global_shape = (m_global, n_global // num_local_experts)
            else:
                per_expert_global_shape = (m_global, n_global)
        else:
            per_expert_global_shape = None

        dion_info_by_param[param] = {
            "is_dion": True,
            "global_shape": (m_global, n_global),
            "fs_shard_dim": fs_shard_dim,
            "tp_shard_dim": tp_shard_dim,
            "per_expert_global_shape": per_expert_global_shape,
        }

    return dion_param_count, dion_info_by_param
