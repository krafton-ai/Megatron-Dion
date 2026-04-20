import math
from typing import Any, Optional

import torch
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.core.optimizer_param_scheduler import ParamGroupOverride
from megatron.core.process_groups_config import ProcessGroupCollection

from ..optimizer_config import DionOptimizerConfig, OptimizerConfig


_DENSE_DION_GROUP_CACHE: dict[tuple[int, ...], torch.distributed.ProcessGroup] = {}


def _validate_exact_replica_group(
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


def _get_or_create_cached_group(ranks: tuple[int, ...]) -> Optional[torch.distributed.ProcessGroup]:
    """Return a cached process group for one deterministic dense-Dion mesh group."""
    if len(ranks) <= 1:
        return None
    cached_group = _DENSE_DION_GROUP_CACHE.get(ranks)
    if cached_group is not None:
        return cached_group
    group = dist.new_group(ranks=list(ranks))
    _DENSE_DION_GROUP_CACHE[ranks] = group
    return group


def _resolve_dense_mesh_groups(
    *,
    pure_dp_group: Optional[torch.distributed.ProcessGroup],
    requested_fs_world_size: int,
    requested_rp_world_size: int,
) -> tuple[Optional[torch.distributed.ProcessGroup], Optional[torch.distributed.ProcessGroup]]:
    """Resolve dense Dion FS/RP groups directly from the global dense pure-DP mesh."""
    if pure_dp_group is None:
        raise RuntimeError(
            "Dion dense FS/RP mesh requires the pure data-parallel group, but it is not initialized."
        )

    fs_size = max(int(requested_fs_world_size), 1)
    rp_size = max(int(requested_rp_world_size), 1)
    expected_pure_dp_world_size = fs_size * rp_size
    tp_size = int(parallel_state.get_tensor_model_parallel_world_size())
    pp_size = int(parallel_state.get_pipeline_model_parallel_world_size())
    cp_size = int(parallel_state.get_context_parallel_world_size())
    order = parallel_state.get_parallel_group_order()
    dense_rank_generator = parallel_state.RankGenerator(
        tp=tp_size,
        ep=1,
        dp=expected_pure_dp_world_size,
        pp=pp_size,
        cp=cp_size,
        order=order,
        rank_offset=0,
    )
    dense_pure_dp_rank_sets = [
        tuple(int(rank) for rank in rank_set) for rank_set in dense_rank_generator.get_ranks("dp")
    ]
    current_pure_dp_ranks = tuple(
        sorted(int(rank) for rank in dist.get_process_group_ranks(pure_dp_group))
    )
    pure_dp_world_size = len(current_pure_dp_ranks)
    if pure_dp_world_size != expected_pure_dp_world_size:
        raise RuntimeError(
            "Dion dense FS/RP topology mismatch while resolving mesh groups. "
            f"requested_fs={requested_fs_world_size} requested_rp={requested_rp_world_size} "
            f"expected_pure_dp_group_size={expected_pure_dp_world_size} "
            f"actual_pure_dp_group_size={pure_dp_world_size} "
            f"(pure_dp_group_ranks={current_pure_dp_ranks})."
        )

    global_rank = int(dist.get_rank())
    if global_rank not in current_pure_dp_ranks:
        raise RuntimeError(
            "Current rank is missing from the pure data-parallel group while resolving Dion dense mesh. "
            f"global_rank={global_rank} pure_dp_group_ranks={current_pure_dp_ranks}"
        )
    if current_pure_dp_ranks not in dense_pure_dp_rank_sets:
        raise RuntimeError(
            "Current pure data-parallel group is missing from the global dense DP mesh. "
            f"global_rank={global_rank} pure_dp_group_ranks={current_pure_dp_ranks} "
            f"dense_dp_rank_sets={dense_pure_dp_rank_sets}"
        )

    fs_rank_sets: list[tuple[int, ...]] = []
    rp_rank_sets: list[tuple[int, ...]] = []
    fs_ranks: tuple[int, ...] | None = None
    rp_ranks: tuple[int, ...] | None = None
    for pure_dp_ranks in dense_pure_dp_rank_sets:
        if global_rank in pure_dp_ranks:
            pure_dp_rank = pure_dp_ranks.index(global_rank)
            fs_block = pure_dp_rank // fs_size
            fs_offset = pure_dp_rank % fs_size
            fs_ranks = tuple(
                pure_dp_ranks[(fs_block * fs_size) : ((fs_block + 1) * fs_size)]
            )
            rp_ranks = tuple(
                pure_dp_ranks[fs_offset + replica_idx * fs_size] for replica_idx in range(rp_size)
            )
        if fs_size > 1:
            fs_rank_sets.extend(
                tuple(pure_dp_ranks[(block_idx * fs_size) : ((block_idx + 1) * fs_size)])
                for block_idx in range(rp_size)
            )
        if rp_size > 1:
            rp_rank_sets.extend(
                tuple(pure_dp_ranks[offset_idx + replica_idx * fs_size] for replica_idx in range(rp_size))
                for offset_idx in range(fs_size)
            )
    if global_rank in current_pure_dp_ranks and (fs_ranks is None or rp_ranks is None):
        raise RuntimeError(
            "Failed to resolve current dense FS/RP groups from the global dense DP mesh. "
            f"global_rank={global_rank} pure_dp_group_ranks={current_pure_dp_ranks}"
        )
    for rank_set in fs_rank_sets + rp_rank_sets:
        _get_or_create_cached_group(rank_set)

    return (
        _get_or_create_cached_group(fs_ranks),
        _get_or_create_cached_group(rp_ranks),
    )


def get_dion_param_override(
    config: OptimizerConfig,
    param: torch.nn.Parameter,
    param_override: Optional[ParamGroupOverride],
) -> Optional[ParamGroupOverride]:
    """Return the Dion param override for scalar embedding/lm-head params."""
    if not isinstance(config, DionOptimizerConfig):
        return None

    is_text_embedding = bool(getattr(param, "is_text_embedding_parameter", False))
    is_lm_head = bool(getattr(param, "is_lm_head_parameter", False))
    is_shared_embedding = bool(getattr(param, "shared_embedding", False))
    is_tied_embedding_output = is_shared_embedding or (is_text_embedding and is_lm_head)

    if not is_text_embedding and not is_lm_head:
        return None

    lr_scaling_rule = getattr(config, "dion_lr_scaling", None)

    # Moonlight keeps scalar surfaces on the shared base-lr schedule and standard wd policy.
    if lr_scaling_rule == "moonlight":
        return None

    if is_lm_head and not is_tied_embedding_output:
        dion_param_override: ParamGroupOverride = {}
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

    if (not is_expert_parallel) and pure_dp_group is not None:
        _, replica_group = _resolve_dense_mesh_groups(
            pure_dp_group=pure_dp_group,
            requested_fs_world_size=requested_fs_world_size,
            requested_rp_world_size=requested_rp_world_size,
        )
        if replica_group is not None:
            return _validate_exact_replica_group(
                selected_group=replica_group,
                pure_dp_group=pure_dp_group,
                requested_rp_world_size=requested_rp_world_size,
                source="dense_mesh",
                is_expert_parallel=is_expert_parallel,
            )

    if pg_collection is not None and hasattr(pg_collection, 'inter_dist_opt_group'):
        group = pg_collection.inter_dist_opt_group
        if group is not None and dist.get_world_size(group) == requested_rp_world_size:
            return _validate_exact_replica_group(
                selected_group=group,
                pure_dp_group=pure_dp_group,
                requested_rp_world_size=requested_rp_world_size,
                source="pg_collection_inter_dist_opt",
                is_expert_parallel=is_expert_parallel,
            )

    group = parallel_state.get_inter_distributed_optimizer_instance_group(check_initialized=False)
    if group is not None and dist.get_world_size(group) == requested_rp_world_size:
        return _validate_exact_replica_group(
            selected_group=group,
            pure_dp_group=pure_dp_group,
            requested_rp_world_size=requested_rp_world_size,
            source="runtime_inter_dist_opt",
            is_expert_parallel=is_expert_parallel,
        )

    if is_expert_parallel and pure_dp_group is not None:
        if dist.get_world_size(pure_dp_group) == requested_rp_world_size:
            return _validate_exact_replica_group(
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


def _get_dense_dion_fs_group(
    *,
    pure_dp_group: Optional[torch.distributed.ProcessGroup],
    requested_fs_world_size: int,
    requested_rp_world_size: int,
) -> Optional[torch.distributed.ProcessGroup]:
    if requested_fs_world_size <= 1:
        return None
    try:
        dion_fs_group = (
            parallel_state.get_intra_distributed_optimizer_instance_data_parallel_group(
                check_initialized=False
            )
        )
    except Exception:
        dion_fs_group = None
    if dion_fs_group is not None:
        dion_fs_world_size = dist.get_world_size(dion_fs_group)
        if dion_fs_world_size == requested_fs_world_size:
            return dion_fs_group
    dion_fs_group, _ = _resolve_dense_mesh_groups(
        pure_dp_group=pure_dp_group,
        requested_fs_world_size=requested_fs_world_size,
        requested_rp_world_size=requested_rp_world_size,
    )
    return dion_fs_group


def _resolve_dion_fs_group(
    *,
    pure_data_parallel_group: Optional[torch.distributed.ProcessGroup],
    is_expert_parallel: bool,
    requested_fs_world_size: int,
    requested_rp_world_size: int,
) -> Optional[torch.distributed.ProcessGroup]:
    if is_expert_parallel:
        return parallel_state.get_expert_data_parallel_group(partial_expert_data_parallel=True)
    return _get_dense_dion_fs_group(
        pure_dp_group=pure_data_parallel_group,
        requested_fs_world_size=requested_fs_world_size,
        requested_rp_world_size=requested_rp_world_size,
    )


def build_megatron_dion(
    *,
    config: DionOptimizerConfig,
    param_groups,
    data_parallel_group: Optional[torch.distributed.ProcessGroup],
    pure_data_parallel_group: Optional[torch.distributed.ProcessGroup],
    pg_collection: Optional[ProcessGroupCollection],
    is_expert_parallel: bool,
):
    from megatron.core.optimizer.dion import DionMixedPrecisionConfig, MegatronDion

    mixed_precision_config = DionMixedPrecisionConfig(
        momentum_dtype=config.dion_momentum_dtype,
        q_dtype=config.dion_q_dtype,
    )

    user_fs_world_size = getattr(config, 'fully_shard_model_parallel_size', 1) or 1
    user_rp_world_size = getattr(config, 'replicate_model_parallel_size', 1) or 1

    fs_group = _resolve_dion_fs_group(
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

    tp_group = (
        parallel_state.get_expert_tensor_parallel_group()
        if is_expert_parallel
        and parallel_state.get_expert_tensor_parallel_group(check_initialized=False) is not None
        else parallel_state.get_tensor_model_parallel_group()
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
        epsilon=config.dion_epsilon,
        rcqr_oversample=config.dion_oversample,
        betas=(config.dion_beta1, config.dion_beta2),
        eps=config.dion_eps,
        scalar_optimizer=config.dion_scalar_optimizer,
        lr_scaling_rule=config.dion_lr_scaling,
        split_qkv=config.dion_split_qkv,
        mixed_precision_config=mixed_precision_config,
        use_fs_collectives=config.dion_use_fs_collectives,
        use_compressed_comm=config.dion_use_compressed_comm,
        enable_async=True,
        max_concurrent_tasks=config.dion_max_concurrent_tasks,
    )


def build_distributed_optimizer_for_dion(
    *,
    optimizer_args,
    config: DionOptimizerConfig,
    model_chunks,
    per_model_buffers,
    data_parallel_group: Optional[torch.distributed.ProcessGroup],
    pure_data_parallel_group: Optional[torch.distributed.ProcessGroup],
    data_parallel_group_gloo: Optional[torch.distributed.ProcessGroup],
    data_parallel_group_idx: Optional[int],
    distributed_optimizer_instance_id: int,
    pg_collection: Optional[ProcessGroupCollection],
    is_expert_parallel: bool,
):
    from megatron.core.optimizer.distrib_optimizer_for_dion import DistributedOptimizerForDion

    requested_fs_size = getattr(config, 'fully_shard_model_parallel_size', None) or 1
    requested_rp_size = getattr(config, 'replicate_model_parallel_size', None) or 1
    fs_group = _resolve_dion_fs_group(
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
                "Dion FS topology mismatch while constructing DistributedOptimizerForDion. "
                f"requested_fs={requested_fs_size} actual_fs_group_size={fs_size} "
                f"(fs_group_ranks={fs_world_ranks})."
            )
    else:
        fs_size = requested_fs_size

    return DistributedOptimizerForDion(
        *optimizer_args,
        model_chunks=model_chunks,
        per_model_buffers=per_model_buffers,
        data_parallel_group=data_parallel_group,
        pure_data_parallel_group=pure_data_parallel_group,
        dion_fs_group=fs_group,
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
        dion_is_expert_parallel=is_expert_parallel,
    )
