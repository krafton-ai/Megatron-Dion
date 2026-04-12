"""Adapter-owned q-init and state-replica helpers for distributed Dion."""

from __future__ import annotations

from typing import Callable, List

import torch
import torch.distributed as dist

from ... import parallel_state
from ..dion.state import q_init_seed_from_logical_id, resolve_q_state_layout
from ..dion.types import DionQLayout, DionQInit, DionStepParam


def make_group_broadcast_fn_(process_group, *, group_size_fn: Callable):
    """Build a no-op or process-group broadcast callback."""
    if process_group is None:
        return lambda tensor: None

    if group_size_fn(process_group) <= 1:
        return lambda tensor: None

    group_ranks = dist.get_process_group_ranks(process_group)
    src_rank = int(group_ranks[0])

    def _broadcast_tensor(tensor):
        dist.broadcast(tensor, src=src_rank, group=process_group)

    return _broadcast_tensor


def resolve_base_training_seed_() -> int:
    """Return the topology-invariant base training seed used for logical Q init."""
    try:
        from megatron.training.global_vars import get_args

        args = get_args()
        return int(args.seed)
    except Exception:
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        return (int(torch.initial_seed()) - 100 * int(pp_rank)) % (2**63 - 1)


def _resolve_q_init(
    *,
    param,
    optim_group,
    dist_meta,
    rank_fraction_default: float,
    rank_multiple_of_default: int,
    base_training_seed: int,
    get_replicate_group_fn: Callable,
    make_group_broadcast_fn: Callable,
) -> DionQInit:
    """Return the adapter-authored q-init contract for one logical Dion param."""
    if dist_meta is None:
        raise RuntimeError(
            "[DION_MISSING_STATE_INIT_META] "
            f"rank={dist.get_rank()} param={getattr(param, '_param_name', '')}"
        )

    config = getattr(dist_meta, "param_config", None)
    if config is None:
        raise RuntimeError(
            "[DION_MISSING_STATE_INIT_PARAM_CONFIG] "
            f"rank={dist.get_rank()} param={getattr(dist_meta, 'param_name', '')} "
            f"param_uid={getattr(dist_meta, 'param_uid', None)}"
        )

    local_shape = tuple(int(dim) for dim in getattr(dist_meta, "shape", ()))
    if len(local_shape) != 2:
        raise RuntimeError(
            "[DION_INVALID_STATE_INIT_LOCAL_SHAPE] "
            f"rank={dist.get_rank()} param={getattr(dist_meta, 'param_name', '')} "
            f"param_uid={getattr(dist_meta, 'param_uid', None)} shape={local_shape}"
        )

    rank_fraction = float(optim_group.get("rank_fraction", rank_fraction_default))
    rank_multiple_of = int(optim_group.get("rank_multiple_of", rank_multiple_of_default))
    tp_world_size = int(getattr(dist_meta, "tp_world_size", 1))
    tp_rank = int(getattr(dist_meta, "tp_rank", 0))
    q_needs_tp_unshard = bool(getattr(config, "use_tp_shard", False))

    q_layout_spec = resolve_q_state_layout(
        local_shape[0],
        local_shape[1],
        config,
        tp_world_size=tp_world_size,
        q_needs_tp_unshard=q_needs_tp_unshard,
        global_shape=tuple(dist_meta.global_shape),
        rank_fraction=rank_fraction,
        rank_multiple_of=rank_multiple_of,
    )
    q_init_seed = q_init_seed_from_logical_id(
        base_seed=base_training_seed,
        dist_meta=dist_meta,
        q_global_shape=(q_layout_spec["q_base_global"], q_layout_spec["r_global"]),
        is_transposed=config.is_transposed,
    )
    q_local_layout = (
        "shard(0)" if bool(getattr(config, "use_fs_shard", False)) else "replicate",
        "shard(1)" if q_needs_tp_unshard and tp_world_size > 1 else "replicate",
    )
    q_gathered_layout = (
        q_local_layout[0],
        "replicate",
    )
    q_layout = DionQLayout(
        q_global_shape=(int(q_layout_spec["q_base_global"]), int(q_layout_spec["r_global"])),
        q_local_shape=tuple(int(dim) for dim in q_layout_spec["q_shape"]),
        q_gathered_shape=(
            int(q_layout_spec["q_base_local"]),
            int(q_layout_spec["r_global"]),
        ),
        q_base_global=int(q_layout_spec["q_base_global"]),
        q_base_local=int(q_layout_spec["q_base_local"]),
        r_global=int(q_layout_spec["r_global"]),
        r_local=int(q_layout_spec["r_local"]),
        q_local_layout=q_local_layout,
        q_gathered_layout=q_gathered_layout,
    )
    replicate_group = get_replicate_group_fn()
    return DionQInit(
        tp_world_size=tp_world_size,
        tp_rank=tp_rank,
        q_needs_tp_unshard=q_needs_tp_unshard,
        q_init_seed=int(q_init_seed),
        q_layout=q_layout,
        broadcast_q_fn=make_group_broadcast_fn(replicate_group),
    )


def _sync_q_replicas(
    *,
    dion_params: List[DionStepParam],
    state_replica_group,
    group_size_fn: Callable,
    make_group_broadcast_fn: Callable,
) -> None:
    """Synchronize freshly initialized Q across standard DO state replicas."""
    if state_replica_group is None or group_size_fn(state_replica_group) <= 1:
        return

    broadcast_q = make_group_broadcast_fn(state_replica_group)
    for step_param in dion_params:
        state = step_param.optimizer_state
        dist_meta = step_param.dist_meta
        if state is None:
            raise RuntimeError(
                "[DION_STATE_REPLICA_Q_SYNC_MISSING_STATE] "
                f"param={getattr(dist_meta, 'param_name', '') if dist_meta is not None else ''}"
            )
        if not state.get("_needs_state_replica_q_sync", False):
            continue
        q_state = state.get("Q", None)
        if q_state is None:
            raise RuntimeError(
                "[DION_STATE_REPLICA_Q_SYNC_MISSING_Q] "
                f"param={getattr(dist_meta, 'param_name', '') if dist_meta is not None else ''}"
            )
        broadcast_q(q_state)
        state["_needs_state_replica_q_sync"] = False
