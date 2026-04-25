import math
from typing import Optional

import torch
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.core.optimizer_param_scheduler import ParamGroupOverride
from megatron.core.process_groups_config import ProcessGroupCollection

from ..optimizer_config import DionOptimizerConfig, OptimizerConfig


def _validate_replica_group(
    *,
    selected_group: Optional[torch.distributed.ProcessGroup],
    pure_dp_group: Optional[torch.distributed.ProcessGroup],
    requested_rp_world_size: int,
    source: str,
    is_expert_parallel: bool,
) -> Optional[torch.distributed.ProcessGroup]:
    """Validate that the selected group is already the exact Dion replica domain."""
    if selected_group is None:
        return None

    selected_ranks = tuple(int(rank) for rank in dist.get_process_group_ranks(selected_group))
    _validate_context_parallel_excluded(
        selected_ranks=selected_ranks,
        source=f"{source}:replica",
    )
    if len(selected_ranks) != int(requested_rp_world_size):
        raise RuntimeError(
            "Dion replica group size mismatch after adapter resolution. "
            f"source={source} requested_rp={requested_rp_world_size} "
            f"actual_rp={len(selected_ranks)} selected_ranks={selected_ranks} "
            f"is_expert_parallel={int(bool(is_expert_parallel))}"
        )

    if pure_dp_group is not None:
        pure_dp_ranks = tuple(int(rank) for rank in dist.get_process_group_ranks(pure_dp_group))
        pure_dp_rank_set = set(pure_dp_ranks)
        leaked_ranks = tuple(rank for rank in selected_ranks if rank not in pure_dp_rank_set)
        if leaked_ranks:
            raise RuntimeError(
                "Dion replica group must already be the exact CP-excluded replica domain. "
                f"source={source} requested_rp={requested_rp_world_size} "
                f"selected_ranks={selected_ranks} pure_dp_ranks={pure_dp_ranks} "
                f"leaked_ranks={leaked_ranks} is_expert_parallel={int(bool(is_expert_parallel))}"
            )

    return selected_group


def _validate_context_parallel_excluded(
    *,
    selected_ranks: tuple[int, ...],
    source: str,
) -> None:
    """Fail fast if a Dion communication group contains the caller's CP peers."""
    if not dist.is_initialized():
        return
    cp_group = parallel_state.get_context_parallel_group(check_initialized=False)
    if cp_group is None:
        return
    cp_ranks = tuple(int(rank) for rank in dist.get_process_group_ranks(cp_group))
    if len(cp_ranks) <= 1:
        return
    global_rank = int(dist.get_rank())
    overlap = tuple(rank for rank in selected_ranks if rank in set(cp_ranks))
    if overlap != (global_rank,):
        raise RuntimeError(
            "Dion communication groups must exclude context-parallel peers. "
            f"source={source} selected_ranks={selected_ranks} "
            f"context_parallel_ranks={cp_ranks} overlap={overlap} "
            f"global_rank={global_rank}"
        )


def _validate_dense_group(
    *,
    selected_group: Optional[torch.distributed.ProcessGroup],
    pure_dp_group: Optional[torch.distributed.ProcessGroup],
    requested_world_size: int,
    source: str,
) -> Optional[torch.distributed.ProcessGroup]:
    """Validate an authoritative dense Dion group against the CP-excluded DP domain."""
    if int(requested_world_size) <= 1:
        return None
    if selected_group is None:
        raise RuntimeError(
            "Dion dense topology requires an authoritative Megatron-Core process group. "
            f"source={source} requested_world_size={requested_world_size}"
        )
    selected_ranks = tuple(int(rank) for rank in dist.get_process_group_ranks(selected_group))
    _validate_context_parallel_excluded(
        selected_ranks=selected_ranks,
        source=f"{source}:dense",
    )
    if len(selected_ranks) != int(requested_world_size):
        raise RuntimeError(
            "Dion dense group size mismatch. "
            f"source={source} requested_world_size={requested_world_size} "
            f"actual_world_size={len(selected_ranks)} selected_ranks={selected_ranks}"
        )
    if pure_dp_group is None:
        raise RuntimeError(
            "Dion dense topology requires the CP-excluded data-parallel group. "
            f"source={source} selected_ranks={selected_ranks}"
        )
    pure_dp_ranks = tuple(int(rank) for rank in dist.get_process_group_ranks(pure_dp_group))
    pure_dp_rank_set = set(pure_dp_ranks)
    leaked_ranks = tuple(rank for rank in selected_ranks if rank not in pure_dp_rank_set)
    if leaked_ranks:
        raise RuntimeError(
            "Dion dense group must be contained in the CP-excluded data-parallel domain. "
            f"source={source} selected_ranks={selected_ranks} "
            f"pure_dp_ranks={pure_dp_ranks} leaked_ranks={leaked_ranks}"
        )
    return selected_group


def get_dion_param_override(
    config: OptimizerConfig,
    param: torch.nn.Parameter,
    param_override: Optional[ParamGroupOverride],
    param_name: Optional[str] = None,
) -> Optional[ParamGroupOverride]:
    """Return Dion-specific elementwise-surface LR overrides."""
    if not isinstance(config, DionOptimizerConfig):
        return None

    name = param_name or getattr(param, "_param_name", "") or ""
    is_embedding_or_output = bool(getattr(param, "is_embedding_or_output_parameter", False))
    is_text_embedding = bool(getattr(param, "is_text_embedding_parameter", False)) or (
        is_embedding_or_output and name.endswith("embedding.word_embeddings.weight")
    )
    is_lm_head = bool(getattr(param, "is_lm_head_parameter", False)) or (
        is_embedding_or_output and name.endswith("output_layer.weight")
    )
    is_shared_embedding = bool(getattr(param, "shared_embedding", False))
    is_tied_embedding_output = is_shared_embedding or (is_text_embedding and is_lm_head)

    if not is_embedding_or_output and not is_text_embedding and not is_lm_head:
        return None

    scale_mode = getattr(config, "dion_scale_mode", None)

    if scale_mode == "spectral":
        return None

    if is_lm_head and not is_tied_embedding_output:
        if param.ndim < 2:
            raise RuntimeError(
                f"[DION_LM_HEAD_INVALID_DIM] expected ndim>=2 for lm_head, got shape={tuple(param.shape)}"
            )
        fan_in = int(param.shape[1])
        if (
            bool(getattr(param, "tensor_model_parallel", False))
            and int(getattr(param, "partition_dim", -1)) == 1
            and parallel_state.model_parallel_is_initialized()
        ):
            fan_in *= int(parallel_state.get_tensor_model_parallel_world_size())
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
        dion_param_override: ParamGroupOverride = {}
        dion_param_override["max_lr"] = current_max_lr / scale
        if current_min_lr is not None:
            dion_param_override["min_lr"] = current_min_lr / scale
        return dion_param_override

    return None


def _get_dion_replica_group(
    pg_collection: Optional[ProcessGroupCollection],
    pure_dp_group: Optional[torch.distributed.ProcessGroup],
    requested_fs_world_size: int,
    requested_rp_world_size: int,
    is_expert_parallel: bool,
) -> Optional[torch.distributed.ProcessGroup]:
    if requested_rp_world_size <= 1:
        return None

    if pg_collection is not None and hasattr(pg_collection, 'inter_dist_opt'):
        group = pg_collection.inter_dist_opt
        if group is not None and dist.get_world_size(group) == requested_rp_world_size:
            return _validate_replica_group(
                selected_group=group,
                pure_dp_group=pure_dp_group,
                requested_rp_world_size=requested_rp_world_size,
                source="pg_collection_inter_dist_opt",
                is_expert_parallel=is_expert_parallel,
            )

    group = parallel_state.get_inter_distributed_optimizer_instance_group(check_initialized=False)
    if group is not None and dist.get_world_size(group) == requested_rp_world_size:
        return _validate_replica_group(
            selected_group=group,
            pure_dp_group=pure_dp_group,
            requested_rp_world_size=requested_rp_world_size,
            source="runtime_inter_dist_opt",
            is_expert_parallel=is_expert_parallel,
        )

    if is_expert_parallel and pure_dp_group is not None:
        if dist.get_world_size(pure_dp_group) == requested_rp_world_size:
            return _validate_replica_group(
                selected_group=pure_dp_group,
                pure_dp_group=pure_dp_group,
                requested_rp_world_size=requested_rp_world_size,
                source="expert_pure_dp",
                is_expert_parallel=is_expert_parallel,
            )

    raise RuntimeError(
        "Dion RP>1 requires a replica group, but none could be resolved. "
        f"requested_fs={requested_fs_world_size} requested_rp={requested_rp_world_size} "
        f"is_expert_parallel={int(bool(is_expert_parallel))}"
    )


def _get_dense_fs_group(
    *,
    dense_fs_group: Optional[torch.distributed.ProcessGroup],
    pure_dp_group: Optional[torch.distributed.ProcessGroup],
    requested_fs_world_size: int,
    requested_rp_world_size: int,
) -> Optional[torch.distributed.ProcessGroup]:
    del requested_rp_world_size
    if requested_fs_world_size <= 1:
        return None
    if dense_fs_group is None:
        dense_fs_group = (
            parallel_state.get_intra_distributed_optimizer_instance_data_parallel_group(
                check_initialized=False
            )
        )
    return _validate_dense_group(
        selected_group=dense_fs_group,
        pure_dp_group=pure_dp_group,
        requested_world_size=requested_fs_world_size,
        source="intra_dist_opt_data_parallel",
    )


def _resolve_fs_group(
    *,
    dense_fs_group: Optional[torch.distributed.ProcessGroup],
    pure_data_parallel_group: Optional[torch.distributed.ProcessGroup],
    is_expert_parallel: bool,
    requested_fs_world_size: int,
    requested_rp_world_size: int,
) -> Optional[torch.distributed.ProcessGroup]:
    if is_expert_parallel:
        expert_fs_group = parallel_state.get_expert_data_parallel_group(
            partial_expert_data_parallel=True
        )
        if expert_fs_group is not None:
            expert_fs_ranks = tuple(
                int(rank) for rank in dist.get_process_group_ranks(expert_fs_group)
            )
            _validate_context_parallel_excluded(
                selected_ranks=expert_fs_ranks,
                source="expert_partial_data_parallel:fs",
            )
        return expert_fs_group
    return _get_dense_fs_group(
        dense_fs_group=dense_fs_group,
        pure_dp_group=pure_data_parallel_group,
        requested_fs_world_size=requested_fs_world_size,
        requested_rp_world_size=requested_rp_world_size,
    )


def build_dion_optimizer(
    *,
    config: DionOptimizerConfig,
    param_groups,
    data_parallel_group: Optional[torch.distributed.ProcessGroup],
    pure_data_parallel_group: Optional[torch.distributed.ProcessGroup],
    dense_fs_group: Optional[torch.distributed.ProcessGroup],
    dion_tp_group: Optional[torch.distributed.ProcessGroup],
    pg_collection: Optional[ProcessGroupCollection],
    is_expert_parallel: bool,
):
    from megatron.core.optimizer.dion import DionMixedPrecisionConfig, MegatronDion

    del data_parallel_group, dion_tp_group

    mixed_precision_config = DionMixedPrecisionConfig(
        momentum_dtype=config.dion_momentum_dtype,
        q_dtype=config.dion_q_dtype,
    )

    user_fs_world_size = getattr(config, 'fully_shard_model_parallel_size', 1) or 1
    user_rp_world_size = getattr(config, 'replicate_model_parallel_size', 1) or 1

    fs_group = _resolve_fs_group(
        dense_fs_group=dense_fs_group,
        pure_data_parallel_group=pure_data_parallel_group,
        is_expert_parallel=is_expert_parallel,
        requested_fs_world_size=user_fs_world_size,
        requested_rp_world_size=user_rp_world_size,
    )

    if fs_group is not None:
        fs_world_ranks = dist.get_process_group_ranks(fs_group)
        fs_world_size = len(fs_world_ranks)
        if not is_expert_parallel and fs_world_size != user_fs_world_size:
            raise RuntimeError(
                "Dion FS topology mismatch while constructing MegatronDion. "
                f"requested_fs={user_fs_world_size} actual_fs_group_size={fs_world_size} "
                f"(fs_group_ranks={fs_world_ranks})."
            )
    else:
        fs_world_size = user_fs_world_size

    replica_group = _get_dion_replica_group(
        pg_collection,
        pure_data_parallel_group,
        user_fs_world_size,
        user_rp_world_size,
        is_expert_parallel,
    )
    if replica_group is not None:
        replica_ranks = dist.get_process_group_ranks(replica_group)
        replica_world_size = len(replica_ranks)
    else:
        replica_ranks = None
        replica_world_size = 1

    expected_replica_world_size = user_rp_world_size
    if replica_world_size != expected_replica_world_size:
        raise RuntimeError(
            "Dion RP/FS topology mismatch while constructing MegatronDion. "
            f"requested_rp={user_rp_world_size} requested_fs={fs_world_size} "
            f"expected_replica={expected_replica_world_size} actual_replica={replica_world_size} "
            f"(replica_ranks={replica_ranks})."
        )

    rp_process_group = replica_group
    if user_rp_world_size > 1:
        rp_world_ranks = dist.get_process_group_ranks(rp_process_group)
        rp_world_size = len(rp_world_ranks)
        if rp_world_size != user_rp_world_size:
            raise RuntimeError(
                "Dion RP topology mismatch while constructing MegatronDion. "
                f"requested_rp={user_rp_world_size} actual_rp_group_size={rp_world_size} "
                f"(rp_group_ranks={rp_world_ranks})."
            )

    if not config.use_distributed_optimizer:
        raise RuntimeError(
            "MegatronDion requires --use-distributed-optimizer. "
            "Legacy ctor-time TP/RP/FS bootstrap is unsupported."
        )

    return MegatronDion(
        param_groups,
        lr=config.lr,
        mu=config.dion_momentum,
        weight_decay=config.weight_decay,
        rank_fraction=config.dion_rank_fraction,
        rank_multiple_of=config.dion_rank_multiple_of,
        epsilon=config.dion_normalize_eps,
        rcqr_oversample=config.dion_oversample,
        betas=(config.dion_beta1, config.dion_beta2),
        elementwise_eps=config.dion_elementwise_eps,
        elementwise_optimizer=config.dion_elementwise_optimizer,
        scale_mode=config.dion_scale_mode,
        extra_scale_factor=config.dion_extra_scale_factor,
        split_qkv=config.dion_split_qkv,
        split_linear=config.dion_split_linear,
        mixed_precision_config=mixed_precision_config,
        use_fs_collectives=config.dion_use_fs_collectives,
        use_low_rank_sync=config.dion_use_low_rank_sync,
        enable_async=True,
        max_concurrent_tasks=config.dion_max_concurrent_tasks,
    )


def build_dion_distributed_optimizer(
    *,
    optimizer_args,
    config: DionOptimizerConfig,
    model_chunks,
    per_model_buffers,
    data_parallel_group: Optional[torch.distributed.ProcessGroup],
    pure_data_parallel_group: Optional[torch.distributed.ProcessGroup],
    dense_fs_group: Optional[torch.distributed.ProcessGroup],
    dion_tp_group: Optional[torch.distributed.ProcessGroup],
    data_parallel_group_gloo: Optional[torch.distributed.ProcessGroup],
    data_parallel_group_idx: Optional[int],
    distributed_optimizer_instance_id: int,
    pg_collection: Optional[ProcessGroupCollection],
    is_expert_parallel: bool,
):
    from megatron.core.optimizer.dion_distrib_optimizer import DionDistributedOptimizer

    requested_fs_size = getattr(config, 'fully_shard_model_parallel_size', None) or 1
    requested_rp_size = getattr(config, 'replicate_model_parallel_size', None) or 1
    fs_group = _resolve_fs_group(
        dense_fs_group=dense_fs_group,
        pure_data_parallel_group=pure_data_parallel_group,
        is_expert_parallel=is_expert_parallel,
        requested_fs_world_size=requested_fs_size,
        requested_rp_world_size=requested_rp_size,
    )
    if fs_group is not None:
        fs_world_ranks = dist.get_process_group_ranks(fs_group)
        fs_size = len(fs_world_ranks)
        if not is_expert_parallel and fs_size != requested_fs_size:
            raise RuntimeError(
                "Dion FS topology mismatch while constructing DionDistributedOptimizer. "
                f"requested_fs={requested_fs_size} actual_fs_group_size={fs_size} "
                f"(fs_group_ranks={fs_world_ranks})."
            )
    else:
        fs_size = requested_fs_size

    return DionDistributedOptimizer(
        *optimizer_args,
        model_chunks=model_chunks,
        per_model_buffers=per_model_buffers,
        data_parallel_group=data_parallel_group,
        pure_data_parallel_group=pure_data_parallel_group,
        dion_fs_group=fs_group,
        dion_tp_group=dion_tp_group,
        replica_group=_get_dion_replica_group(
            pg_collection,
            pure_data_parallel_group,
            requested_fs_size,
            requested_rp_size,
            is_expert_parallel,
        ),
        data_parallel_group_gloo=data_parallel_group_gloo,
        data_parallel_group_idx=data_parallel_group_idx,
        distributed_optimizer_instance_id=distributed_optimizer_instance_id,
        fully_shard_model_parallel_size=fs_size,
        replica_model_parallel_size=requested_rp_size,
        is_expert_dion=is_expert_parallel,
    )
