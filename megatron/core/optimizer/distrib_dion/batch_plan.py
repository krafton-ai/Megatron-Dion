"""Batch grouping and batch assembly for distributed Dion."""

from __future__ import annotations

from typing import Callable, List

from .batch_binding import _build_batch_collectives, _resolve_batch_route
from .batch_order import (
    align_group_data_order_,
    sync_batch_keys_,
    validate_distributed_batch_group_cardinality_,
    _validate_param_batches,
)
from ..dion.batching import build_batch_key, group_items_by_batch_key, unique_preserve_order
from ..dion.state import require_2d_local_shape
from ..dion.types import DionBatchGroup, DionBatch, DionStepParam
from ..dion.batching import pad_batch


def _missing_local_shard_error(
    *,
    batch_key,
    batch_route,
    global_rank: int,
) -> None:
    raise RuntimeError(
        "[DION_MISSING_LOCAL_SHARD] "
        f"batch_key={batch_key} rank={global_rank} "
        f"sync_groups={list(batch_route.sync_groups) if batch_route is not None else []}"
    )


def _group_and_order_param_batches(
    *,
    routed_params: list[DionStepParam],
    batch_key_cache: dict,
    use_fs_collectives: bool,
    state_replica_group,
    global_rank: int,
    group_size_fn: Callable,
    get_replicate_group_fn: Callable,
    resolve_ortho_group_fn: Callable,
):
    """Group Dion params by batch key, then canonicalize and validate cross-rank order."""
    batch_groups = {}
    batch_items = []
    batch_keys = []
    for routed_param in routed_params:
        param = routed_param.param
        state = routed_param.optimizer_state
        dist_meta = routed_param.dist_meta
        config = routed_param.config
        local_shape = state.get('local_shape', None)
        if local_shape is None:
            local_shape = require_2d_local_shape(param, dist_meta)
            state['local_shape'] = local_shape
        batch_items.append(routed_param)
        batch_keys.append(build_batch_key(local_shape, config, routed_param.grad.dtype))

    for batch_key, grouped_items in group_items_by_batch_key(batch_items, batch_keys):
        batch_groups[batch_key] = DionBatchGroup(
            params=[item.param for item in grouped_items],
            grads=[item.grad for item in grouped_items],
            optimizer_states=[item.optimizer_state for item in grouped_items],
            optim_groups=[item.optim_group for item in grouped_items],
            configs=[item.config for item in grouped_items],
            dist_metas=[item.dist_meta for item in grouped_items],
        )

    local_batch_keys = list(batch_groups.keys())
    batch_route_map = {}
    grouped_batch_keys = {}
    for batch_key in local_batch_keys:
        batch_group = batch_groups[batch_key]
        batch_route = _resolve_batch_route(
            config=batch_group.configs[0],
            dist_meta=batch_group.dist_metas[0] if batch_group.dist_metas else None,
            use_fs_collectives=use_fs_collectives,
            state_replica_group=state_replica_group,
            group_size_fn=group_size_fn,
            get_replicate_group_fn=get_replicate_group_fn,
            resolve_ortho_group_fn=resolve_ortho_group_fn,
        )
        batch_route_map[batch_key] = batch_route
        if not batch_route.sync_groups:
            grouped_batch_keys.setdefault(None, (None, []))[1].append(batch_key)
            continue
        for sync_group in batch_route.sync_groups:
            grouped_batch_keys.setdefault(id(sync_group), (sync_group, []))[1].append(batch_key)

    all_batch_keys = []
    for sync_group, group_keys in grouped_batch_keys.values():
        if sync_group is not None:
            all_batch_keys.extend(
                sync_batch_keys_(
                    local_batch_keys=group_keys,
                    sync_group=sync_group,
                    batch_key_cache=batch_key_cache,
                )
            )
        else:
            all_batch_keys.extend(group_keys)
    all_batch_keys = unique_preserve_order(all_batch_keys)

    ordered_batches = []
    for batch_key in all_batch_keys:
        if batch_key not in batch_groups:
            batch_route = batch_route_map.get(batch_key)
            _missing_local_shard_error(
                batch_key=batch_key,
                batch_route=batch_route,
                global_rank=global_rank,
            )

        batch_group = batch_groups[batch_key]
        batch_route = batch_route_map[batch_key]
        validate_distributed_batch_group_cardinality_(
            batch_key=batch_key,
            batch_group=batch_group,
            batch_route=batch_route,
        )
        for sync_group in batch_route.sync_groups:
            align_group_data_order_(
                batch_group=batch_group,
                sync_group=sync_group,
            )
        _validate_param_batches(
            batch_key=batch_key,
            batch_group=batch_group,
            batch_route=batch_route,
        )
        ordered_batches.append((batch_key, batch_group, batch_route))

    return ordered_batches


def _build_dion_batches(
    *,
    dion_params: list[DionStepParam],
    use_fs_collectives: bool,
    state_replica_group,
    batch_key_cache: dict,
    global_rank: int,
    group_size_fn: Callable,
    get_replicate_group_fn: Callable,
    resolve_ortho_group_fn: Callable,
    resolve_tp_group_fn: Callable,
    resolve_fs_group_fn: Callable,
    resolve_device_mesh_fn: Callable,
) -> List[DionBatch]:
    """Build Dion batches from explicit runtime state."""
    ordered_batches = _group_and_order_param_batches(
        routed_params=dion_params,
        batch_key_cache=batch_key_cache,
        use_fs_collectives=use_fs_collectives,
        state_replica_group=state_replica_group,
        global_rank=global_rank,
        group_size_fn=group_size_fn,
        get_replicate_group_fn=get_replicate_group_fn,
        resolve_ortho_group_fn=resolve_ortho_group_fn,
    )
    return _assemble_dion_batches(
        ordered_batches=ordered_batches,
        use_fs_collectives=use_fs_collectives,
        resolve_tp_group_fn=resolve_tp_group_fn,
        resolve_fs_group_fn=resolve_fs_group_fn,
        resolve_device_mesh_fn=resolve_device_mesh_fn,
    )


def _assemble_dion_batches(
    *,
    ordered_batches,
    use_fs_collectives: bool,
    resolve_tp_group_fn: Callable,
    resolve_fs_group_fn: Callable,
    resolve_device_mesh_fn: Callable,
) -> List[DionBatch]:
    """Assemble fixed-size Dion batches from ordered batch groups."""
    dion_batches: List[DionBatch] = []
    global_param_offset = 0
    for batch_key, batch_group, batch_route in ordered_batches:
        local_shape = batch_key[0]
        m, n = local_shape
        batch_size = batch_route.batch_world_size
        local_num_params = len(batch_group.params or [])

        for batch_start in range(0, local_num_params, batch_size):
            batch_end = min(batch_start + batch_size, local_num_params)

            params = []
            param_shapes = []
            momentums = []
            q_tensors = []
            configs = []
            dist_metas = []
            optim_groups = []
            grads_to_process = []
            optimizer_states = []

            for idx in range(batch_start, batch_end):
                param = batch_group.params[idx]
                grad = batch_group.grads[idx]
                optimizer_state = batch_group.optimizer_states[idx]
                optim_group = batch_group.optim_groups[idx]
                config = batch_group.configs[idx]
                dist_meta = batch_group.dist_metas[idx]

                params.append(param)
                param_shapes.append((m, n))
                grads_to_process.append(grad.view(m, n))
                momentums.append(optimizer_state['momentum'].view(m, n))
                q_tensors.append(optimizer_state['Q'])
                configs.append(config)
                dist_metas.append(dist_meta)
                optim_groups.append(optim_group)
                optimizer_states.append(optimizer_state)

            real_batch_size = len(params)

            params = pad_batch(params, batch_size)
            grads_to_process = pad_batch(grads_to_process, batch_size)
            momentums = pad_batch(momentums, batch_size)
            q_tensors = pad_batch(q_tensors, batch_size)

            while len(param_shapes) < batch_size:
                param_shapes.append(param_shapes[0])
            while len(configs) < batch_size:
                configs.append(configs[0])
            while len(dist_metas) < batch_size:
                dist_metas.append(None)
            while len(optim_groups) < batch_size:
                optim_groups.append(optim_groups[0])

            batch_collectives = _build_batch_collectives(
                q_tensors=q_tensors,
                configs=configs,
                dist_metas=dist_metas,
                use_fs_collectives=use_fs_collectives,
                resolve_tp_group_fn=resolve_tp_group_fn,
                resolve_fs_group_fn=resolve_fs_group_fn,
                resolve_device_mesh_fn=resolve_device_mesh_fn,
            )
            dion_batches.append(
                DionBatch(
                    batch_key=batch_key,
                    params=params,
                    grads=grads_to_process,
                    momentums=momentums,
                    q_tensors=q_tensors,
                    configs=configs,
                    dist_metas=dist_metas,
                    optim_groups=optim_groups,
                    optimizer_states=optimizer_states,
                    param_shapes=tuple(param_shapes),
                    real_batch_size=real_batch_size,
                    global_param_offset=global_param_offset,
                    batch_route=batch_route,
                    batch_collectives=batch_collectives,
                )
            )
            global_param_offset += real_batch_size

        batch_group.params = []
        batch_group.grads = []
        batch_group.optimizer_states = []
        batch_group.optim_groups = []
        batch_group.configs = []
        batch_group.dist_metas = []

    return dion_batches
