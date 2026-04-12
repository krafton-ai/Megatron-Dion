"""Adapter-owned enablement orchestration helpers for distributed Dion."""

from __future__ import annotations

import traceback
from typing import Callable

import torch.distributed as dist
from .runtime_binding import _validate_step_groups


def enable_distributed_dion_(
    *,
    optimizer,
    global_rank: int,
    original_dp_group,
    data_parallel_group,
    tp_group,
    rp_group,
    fs_group,
    state_replica_group,
    expected_rp_size: int,
    build_dist_metas_fn: Callable,
    route_step_params_fn: Callable,
    group_size_fn: Callable,
    group_rank_fn: Callable,
    use_compressed_comm: bool,
    use_fs_collectives: bool,
    validate_enabled_rp_topology_fn: Callable,
    logger_error_fn: Callable,
):
    """Enable distributed Dion mode and return the published distributed metadata."""
    if dist.is_initialized() and dist.get_world_size() == 1:
        return None

    try:
        dist_metas_sharded = build_dist_metas_fn()
    except Exception as exc:
        logger_error_fn("[Dion] Global rank %s: Failed in _build_dist_metas: %s", global_rank, exc)
        logger_error_fn(traceback.format_exc())
        raise

    full_data_parallel_group = original_dp_group or data_parallel_group
    optimizer.enable_distributed_mode(
        route_step_params_fn=_validate_step_groups(
            full_data_parallel_group=full_data_parallel_group,
            tp_group=tp_group,
            rp_group=rp_group,
            fs_group=fs_group,
            state_replica_group=state_replica_group,
            route_step_params_fn=route_step_params_fn,
            group_size_fn=group_size_fn,
            group_rank_fn=group_rank_fn,
            use_compressed_comm=use_compressed_comm,
            use_fs_collectives=use_fs_collectives,
        ),
    )
    validate_enabled_rp_topology_fn(
        expected_rp_size=expected_rp_size,
        rp_group=rp_group,
        data_parallel_group=data_parallel_group,
        global_rank=global_rank,
        dist_metas_sharded=dist_metas_sharded,
        logger_error_fn=logger_error_fn,
    )
    return dist_metas_sharded
