import math
from typing import Any, Optional

import torch
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.core.optimizer_param_scheduler import ParamGroupOverride
from megatron.core.process_groups_config import ProcessGroupCollection

from ..optimizer_config import DionOptimizerConfig, OptimizerConfig


def get_dion_scalar_param_override(
    config: OptimizerConfig,
    param: torch.nn.Parameter,
    param_override: Optional[ParamGroupOverride],
) -> Optional[ParamGroupOverride]:
    """Return Dion scalar-path override for embedding/lm-head surfaces."""
    if not isinstance(config, DionOptimizerConfig):
        return None

    is_text_embedding = bool(getattr(param, "is_text_embedding_parameter", False))
    is_lm_head = bool(getattr(param, "is_lm_head_parameter", False))
    is_shared_embedding = bool(getattr(param, "shared_embedding", False))
    is_tied_embedding_output = is_shared_embedding or (is_text_embedding and is_lm_head)

    if not is_text_embedding and not is_lm_head:
        return None

    scalar_override: ParamGroupOverride = {"wd_mult": 0.0}

    if is_lm_head and not is_tied_embedding_output:
        if param.ndim < 2:
            raise RuntimeError(
                f"[DION_LM_HEAD_INVALID_DIM] expected ndim>=2 for lm_head, got shape={tuple(param.shape)}"
            )
        fan_in = int(param.shape[1])
        if fan_in <= 0:
            raise RuntimeError(
                f"[DION_LM_HEAD_INVALID_FAN_IN] lm_head fan_in must be > 0, got {fan_in}"
            )
        scale = math.sqrt(float(fan_in))
        current_max_lr = (
            param_override["max_lr"]
            if param_override is not None and "max_lr" in param_override
            else config.lr
        )
        current_min_lr = (
            param_override["min_lr"]
            if param_override is not None and "min_lr" in param_override
            else config.min_lr
        )
        scalar_override["max_lr"] = current_max_lr / scale
        if current_min_lr is not None:
            scalar_override["min_lr"] = current_min_lr / scale

    return scalar_override


def _get_dion_full_data_parallel_group(
    pg_collection: Optional[ProcessGroupCollection], is_expert_parallel: bool
) -> Optional[torch.distributed.ProcessGroup]:
    if is_expert_parallel:
        if pg_collection is not None and hasattr(pg_collection, 'expt_dp'):
            return pg_collection.expt_dp
        return parallel_state.get_expert_data_parallel_group(check_initialized=False)

    if pg_collection is not None and hasattr(pg_collection, 'dp_cp'):
        return pg_collection.dp_cp
    return parallel_state.get_data_parallel_group(with_context_parallel=True)


def build_megatron_dion(
    *,
    config: DionOptimizerConfig,
    param_groups,
    data_parallel_group: Optional[torch.distributed.ProcessGroup],
    pg_collection: Optional[ProcessGroupCollection],
    is_expert_parallel: bool,
):
    from megatron.core.optimizer.dion import DionMixedPrecisionConfig, MegatronDion

    mixed_precision_config = DionMixedPrecisionConfig(
        momentum_dtype=config.dion_momentum_dtype,
        Q_dtype=config.dion_Q_dtype,
    )

    user_fs_world_size = getattr(config, 'fully_shard_model_parallel_size', 1) or 1
    user_rp_world_size = getattr(config, 'replicate_model_parallel_size', 1) or 1

    if is_expert_parallel:
        fs_process_group = parallel_state.get_expert_data_parallel_group(
            partial_expert_data_parallel=True
        )
    else:
        fs_process_group = data_parallel_group

    if fs_process_group is not None:
        fs_world_ranks = dist.get_process_group_ranks(fs_process_group)
        fs_world_size = len(fs_world_ranks)
        if not is_expert_parallel and fs_world_size != user_fs_world_size:
            raise RuntimeError(
                "Dion FS topology mismatch while constructing MegatronDion. "
                f"requested_fs={user_fs_world_size} actual_fs_group_size={fs_world_size} "
                f"(fs_group_ranks={fs_world_ranks})."
            )
    else:
        fs_world_size = user_fs_world_size

    full_dp_group = _get_dion_full_data_parallel_group(pg_collection, is_expert_parallel)
    if full_dp_group is None:
        full_dp_group = fs_process_group
    if full_dp_group is not None:
        full_dp_ranks = dist.get_process_group_ranks(full_dp_group)
        full_dp_world_size = len(full_dp_ranks)
    else:
        full_dp_ranks = None
        full_dp_world_size = fs_world_size

    expected_full_dp_world_size = fs_world_size * user_rp_world_size
    if full_dp_world_size != expected_full_dp_world_size:
        raise RuntimeError(
            "Dion RP/FS topology mismatch while constructing MegatronDion. "
            f"requested_rp={user_rp_world_size} requested_fs={fs_world_size} "
            f"expected_full_dp={expected_full_dp_world_size} actual_full_dp={full_dp_world_size} "
            f"(full_dp_ranks={full_dp_ranks})."
        )

    rp_process_group = None
    if user_rp_world_size > 1:
        rp_process_group = parallel_state.get_inter_distributed_optimizer_instance_group(
            check_initialized=False
        )
        if rp_process_group is None:
            raise RuntimeError(
                "Dion RP>1 requires Megatron-Core inter distributed optimizer instance group, "
                f"but it is not initialized (requested_rp={user_rp_world_size})."
            )
        rp_world_ranks = dist.get_process_group_ranks(rp_process_group)
        rp_world_size = len(rp_world_ranks)
        if rp_world_size != user_rp_world_size:
            raise RuntimeError(
                "Dion RP topology mismatch while constructing MegatronDion. "
                f"requested_rp={user_rp_world_size} actual_rp_group_size={rp_world_size} "
                f"(rp_group_ranks={rp_world_ranks})."
            )

    tp_group = (
        parallel_state.get_expert_tensor_parallel_group()
        if is_expert_parallel
        and parallel_state.get_expert_tensor_parallel_group(check_initialized=False) is not None
        else parallel_state.get_tensor_model_parallel_group()
    )

    return MegatronDion(
        param_groups,
        lr=config.lr,
        mu=config.dion_momentum,
        weight_decay=config.weight_decay,
        rank_fraction=config.dion_rank_fraction,
        rank_multiple_of=config.dion_rank_multiple_of,
        epsilon=config.dion_epsilon,
        rcqr_oversample=config.dion_oversample,
        betas=(config.dion_beta1, config.dion_beta2),
        eps=config.dion_eps,
        scalar_optimizer=config.dion_scalar_optimizer,
        lr_scaling_rule=config.dion_lr_scaling,
        mixed_precision_config=mixed_precision_config,
        use_fs_collectives=config.dion_use_fs_collectives,
        use_compressed_comm=config.dion_use_compressed_comm,
        rp_group=rp_process_group,
        fs_group=fs_process_group,
        tp_group=tp_group,
        enable_async=True,
        local_batch_size=config.dion_local_batch_size,
        max_concurrent_tasks=config.dion_max_concurrent_tasks,
    )


def build_distributed_optimizer_for_dion(
    *,
    optimizer_args,
    config: DionOptimizerConfig,
    model_chunks,
    per_model_buffers,
    data_parallel_group: Optional[torch.distributed.ProcessGroup],
    data_parallel_group_gloo: Optional[torch.distributed.ProcessGroup],
    data_parallel_group_idx: Optional[int],
    distributed_optimizer_instance_id: int,
    pg_collection: Optional[ProcessGroupCollection],
    is_expert_parallel: bool,
):
    from megatron.core.optimizer.distrib_optimizer_for_dion import DistributedOptimizerForDion

    if data_parallel_group is not None:
        requested_fs_size = getattr(config, 'fully_shard_model_parallel_size', None) or 1
        fs_world_ranks = dist.get_process_group_ranks(data_parallel_group)
        fs_size = len(fs_world_ranks)
        if not is_expert_parallel and fs_size != requested_fs_size:
            raise RuntimeError(
                "Dion FS topology mismatch while constructing DistributedOptimizerForDion. "
                f"requested_fs={requested_fs_size} actual_fs_group_size={fs_size} "
                f"(fs_group_ranks={fs_world_ranks})."
            )
    else:
        fs_size = getattr(config, 'fully_shard_model_parallel_size', None) or 1

    return DistributedOptimizerForDion(
        *optimizer_args,
        model_chunks=model_chunks,
        per_model_buffers=per_model_buffers,
        data_parallel_group=data_parallel_group,
        full_data_parallel_group=_get_dion_full_data_parallel_group(
            pg_collection, is_expert_parallel
        ),
        data_parallel_group_gloo=data_parallel_group_gloo,
        data_parallel_group_idx=data_parallel_group_idx,
        distributed_optimizer_instance_id=distributed_optimizer_instance_id,
        fully_shard_model_parallel_size=fs_size,
        replica_model_parallel_size=(getattr(config, 'replicate_model_parallel_size', None) or 1),
    )
