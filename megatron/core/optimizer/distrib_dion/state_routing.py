"""Adapter-owned state remap and distributed routing helpers for Dion execution."""

from __future__ import annotations


def use_distributed_dion_update_(
    *,
    param,
    state,
    optim_group,
    dist_meta,
    global_rank: int,
) -> bool:
    """Return the adapter-owned distributed Dion routing decision."""
    if dist_meta is None:
        raise RuntimeError(
            "[DION_MISSING_DIST_META_FOR_UPDATE_ROUTING] "
            f"rank={global_rank} "
            f"param_name={getattr(param, '_param_name', '')} param_shape={tuple(param.shape)}"
        )

    is_dion_marked = bool(dist_meta.is_dion_param)
    is_2d_global = bool(dist_meta.global_shape is not None and len(dist_meta.global_shape) == 2)
    if is_dion_marked and not is_2d_global:
        raise RuntimeError(
            "[DION_INVALID_DIST_META_FOR_DION_ROUTING] "
            f"rank={global_rank} "
            f"param_uid={getattr(dist_meta, 'param_uid', None)} "
            f"param_name={getattr(dist_meta, 'param_name', '')} "
            f"global_shape={getattr(dist_meta, 'global_shape', None)}"
        )

    return (
        optim_group.get('algorithm', 'dion') == 'dion'
        and is_dion_marked
        and is_2d_global
        and 'Q' in state
    )


def get_or_initialize_optimizer_state_(
    *,
    optimizer,
    param,
    optim_group,
    dion_state_param_by_uid,
    dion_dist_meta_by_uid,
    init_optimizer_state_fn,
):
    """Own distributed state remap and metadata recovery at the adapter boundary."""
    dist_meta = optimizer.dist_metas.get(param, None)
    param_uid = getattr(param, "_dion_param_uid", None)
    if param_uid is None and dist_meta is not None:
        param_uid = dist_meta.param_uid

    if param not in optimizer.state:
        if param_uid is not None:
            old_param = dion_state_param_by_uid.get(param_uid)
            if old_param is not None and old_param is not param and old_param in optimizer.state:
                optimizer.state[param] = optimizer.state.pop(old_param)
            else:
                optimizer.state[param] = {}
            dion_state_param_by_uid[param_uid] = param
            if param not in optimizer.dist_metas and param_uid in dion_dist_meta_by_uid:
                optimizer.dist_metas[param] = dion_dist_meta_by_uid[param_uid]
        else:
            optimizer.state[param] = {}

    state = optimizer.state[param]
    if len(state) == 0:
        init_optimizer_state_fn(param, state, optim_group)
    return state
