"""Factory helpers for distributed MCore ARO."""

from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist

from megatron.core.optimizer.matrix.topology import get_matrix_replica_group, resolve_fs_group
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from megatron.core.optimizer_param_scheduler import ParamGroupOverride


def get_aro_param_override(
    config: OptimizerConfig,
    param: torch.nn.Parameter,
    param_override: Optional[ParamGroupOverride],
    param_name: Optional[str] = None,
) -> Optional[ParamGroupOverride]:
    """Return ARO-specific param group overrides.

    Full-model ARO applies matrix updates to eligible 2D parameters, including
    embedding and LM-head tensors, and does not need an optimizer-specific LR
    override.
    """
    del config, param, param_override, param_name
    return None


def _resolve_aro_fs_group(
    *,
    dense_fs_group: Optional[torch.distributed.ProcessGroup],
    pure_data_parallel_group: Optional[torch.distributed.ProcessGroup],
    is_expert_parallel: bool,
    requested_fs_size: int,
    requested_rp_size: int,
) -> Optional[torch.distributed.ProcessGroup]:
    fs_group = resolve_fs_group(
        dense_fs_group=dense_fs_group,
        pure_data_parallel_group=pure_data_parallel_group,
        is_expert_parallel=is_expert_parallel,
        requested_fs_world_size=requested_fs_size,
        requested_rp_world_size=requested_rp_size,
        optimizer_name="ARO optimizer",
    )
    if fs_group is not None:
        fs_size = len(dist.get_process_group_ranks(fs_group))
        if not is_expert_parallel and fs_size != int(requested_fs_size):
            raise RuntimeError(
                "ARO FS topology mismatch while constructing DistributedAroOptimizer. "
                f"requested_fs={requested_fs_size} actual_fs_group_size={fs_size} "
                f"(fs_group_ranks={dist.get_process_group_ranks(fs_group)})."
            )
    return fs_group


def build_aro_optimizer(
    *,
    config,
    param_groups,
    data_parallel_group=None,
    pure_data_parallel_group=None,
    dense_fs_group=None,
    aro_tp_group=None,
    pg_collection=None,
    is_expert_parallel: bool = False,
):
    """Build the inner MegatronAro optimizer used by local and distributed paths."""
    del data_parallel_group, pure_data_parallel_group, dense_fs_group, aro_tp_group
    del pg_collection, is_expert_parallel
    from megatron.core.optimizer.aro.algorithm import MegatronAro
    from megatron.core.optimizer.aro.types import AroMixedPrecisionConfig

    mixed_precision_config = AroMixedPrecisionConfig(
        momentum_dtype=getattr(config, "aro_momentum_dtype", None),
        rotation_dtype=getattr(config, "aro_rotation_dtype", None),
    )
    use_distributed = bool(getattr(config, "use_distributed_optimizer", False))
    return MegatronAro(
        param_groups,
        lr=config.lr,
        momentum=config.aro_momentum,
        weight_decay=config.weight_decay,
        base_optimizer=config.aro_base_optimizer,
        sinkhorn_iters=config.aro_sinkhorn_iters,
        qr_backend=config.aro_qr_backend,
        scqr_eps=config.aro_scqr_eps,
        update_rms_scale=config.aro_update_rms_scale,
        scalar_optimizer=config.aro_scalar_optimizer,
        scalar_lr_scale=config.aro_scalar_lr_scale,
        betas=(config.aro_beta1, config.aro_beta2),
        scalar_eps=config.aro_scalar_eps,
        split_qkv=False if use_distributed else config.aro_split_qkv,
        split_linear=False if use_distributed else config.aro_split_linear,
        mixed_precision_config=mixed_precision_config,
    )


def build_aro_distributed_optimizer(
    *,
    optimizer_args,
    dense_fs_group=None,
    aro_tp_group=None,
    **kwargs,
):
    """Build a DistributedAroOptimizer with MCore distributed-optimizer ownership."""
    from .optimizer import DistributedAroOptimizer

    config = kwargs.pop("config", None)
    pure_data_parallel_group = kwargs.pop("pure_data_parallel_group", None)
    pg_collection = kwargs.pop("pg_collection", None)
    is_expert_parallel = bool(kwargs.pop("is_expert_parallel", False))
    requested_fs_size = 1
    requested_rp_size = 1
    if config is not None:
        requested_fs_size = int(getattr(config, "fully_shard_model_parallel_size", 1) or 1)
        requested_rp_size = int(getattr(config, "replicate_model_parallel_size", 1) or 1)
        fs_group = _resolve_aro_fs_group(
            dense_fs_group=dense_fs_group,
            pure_data_parallel_group=pure_data_parallel_group,
            is_expert_parallel=is_expert_parallel,
            requested_fs_size=requested_fs_size,
            requested_rp_size=requested_rp_size,
        )
        fs_size = 1 if fs_group is None else len(dist.get_process_group_ranks(fs_group))
        replica_group = get_matrix_replica_group(
            pg_collection,
            pure_data_parallel_group,
            requested_fs_size,
            requested_rp_size,
            is_expert_parallel,
            optimizer_name="ARO optimizer",
        )
        kwargs.setdefault("fully_shard_model_parallel_size", fs_size)
        kwargs.setdefault("replica_model_parallel_size", requested_rp_size)
        kwargs.setdefault("replica_group", replica_group)
    else:
        fs_group = dense_fs_group
    kwargs.setdefault("is_expert_aro", is_expert_parallel)
    return DistributedAroOptimizer(
        *optimizer_args,
        aro_fs_group=fs_group,
        aro_tp_group=aro_tp_group,
        pure_data_parallel_group=pure_data_parallel_group,
        **kwargs,
    )


__all__ = [
    "build_aro_distributed_optimizer",
    "build_aro_optimizer",
    "get_aro_param_override",
]
