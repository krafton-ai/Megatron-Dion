"""Batch assembly for distributed Dion."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Sequence, Tuple

import torch
import torch.distributed as dist

from ... import parallel_state
from ..dion.state import require_2d_local_shape
from ..dion.state import (
    is_fs_only_config,
    needs_fs_p_reduce,
    needs_tp_r_reduce,
    p_is_tp_sharded,
)
from ..dion.utils import get_local_shape, has_multiple_local_experts
from ..dion.types import (
    DionAxisCollective,
    DionBatch,
    DionBatchEntry,
    DionBatchCollectives,
    DionBatchGroup,
    DionStepParam,
)

def _missing_local_shard_error(*, batch_key, batch_group, global_rank: int) -> None:
    raise RuntimeError(
        "[DION_MISSING_LOCAL_SHARD] "
        f"batch_key={batch_key} rank={global_rank} "
        f"sync_groups={list(batch_group.sync_groups) if batch_group is not None else []}"
    )


def _normalize_shard_dim(dim, has_axis: bool) -> int:
    if dim is None and not has_axis:
        return -1
    if dim is None:
        raise RuntimeError(
            "[Dion] missing shard tensor dim for active sharded axis in batch key construction"
        )
    return int(dim)


def _normalize_shape(shape) -> tuple:
    if shape is None:
        return ()
    return tuple(int(dim) for dim in shape)


def _build_update_contract_key(
    optim_group: Dict[str, Any] | None,
    optimizer_state: Dict[str, Any] | None,
) -> tuple:
    if optim_group is None:
        lr = None
        weight_decay = None
        wd_mult = None
        mu = None
    else:
        lr = float(optim_group["lr"]) if "lr" in optim_group else None
        weight_decay = (
            float(optim_group["weight_decay"]) if "weight_decay" in optim_group else None
        )
        wd_mult = float(optim_group["wd_mult"]) if "wd_mult" in optim_group else None
        mu = float(optim_group["mu"]) if "mu" in optim_group else None
    r = int(optimizer_state.get("r", -1)) if optimizer_state is not None else -1
    return (lr, weight_decay, wd_mult, mu, r)


def build_batch_key(
    shape,
    cfg,
    dtype: torch.dtype,
    *,
    true_global_shape=None,
    per_expert_global_shape=None,
) -> tuple:
    """Build the canonical Dion batch key used by local/distributed runtimes."""
    if len(shape) == 2:
        resolved_shape = (int(shape[0]), int(shape[1]))
    else:
        resolved_shape = tuple(int(dim) for dim in shape)
    return (
        resolved_shape,
        bool(cfg.has_fs_shard),
        bool(getattr(cfg, "use_fs_shard", cfg.has_fs_shard)),
        bool(cfg.has_tp_shard),
        bool(getattr(cfg, "use_tp_shard", cfg.has_tp_shard)),
        bool(cfg.is_transposed),
        bool(cfg.compressed_all_reduce),
        _normalize_shard_dim(cfg.tp_shard_dim, cfg.has_tp_shard),
        _normalize_shard_dim(cfg.fs_shard_dim, cfg.has_fs_shard),
        dtype,
        _normalize_shape(true_global_shape),
        _normalize_shape(per_expert_global_shape),
    )


def unique_preserve_order(items: Sequence[Any]) -> list[Any]:
    """Deduplicate while preserving first-seen order."""
    return list(dict.fromkeys(items))


def _batch_key_sort_key(batch_key: tuple) -> str:
    """Return a deterministic sort key for one canonical batch key."""
    return repr(batch_key)


def _sync_group_batch_metadata(
    *,
    sync_group,
    local_batch_keys: Sequence[tuple],
    batch_group_by_key: Dict[tuple, DionBatchGroup],
) -> tuple[list[tuple], dict[tuple, int]]:
    """Return canonical cross-rank batch order plus per-key chunk multiplicity.

    The Dion update must not depend on local batch-key discovery order.
    Canonicalize the batch schedule over each concrete sync group, then validate that
    every participating rank contributes the same number of local params for each
    batch key.
    """
    if sync_group is None or dist.get_world_size(sync_group) <= 1:
        ordered = sorted(local_batch_keys, key=_batch_key_sort_key)
        multiplicity = {
            batch_key: len(batch_group_by_key[batch_key].params or [])
            for batch_key in ordered
        }
        return ordered, multiplicity

    local_counts = {
        batch_key: len(batch_group_by_key[batch_key].params or [])
        for batch_key in local_batch_keys
    }
    gathered = [None] * dist.get_world_size(sync_group)
    dist.all_gather_object(
        gathered,
        {batch_key: int(count) for batch_key, count in local_counts.items()},
        group=sync_group,
    )

    all_batch_keys = set()
    for rank_counts in gathered:
        all_batch_keys.update(rank_counts.keys())
    ordered = sorted(all_batch_keys, key=_batch_key_sort_key)

    multiplicity = {}
    for batch_key in ordered:
        counts = [int(rank_counts.get(batch_key, 0)) for rank_counts in gathered]
        max_count = max(counts)
        min_count = min(counts)
        if min_count != max_count:
            raise RuntimeError(
                "[DION_BATCH_KEY_MULTIPLICITY_MISMATCH] "
                f"rank={dist.get_rank()} batch_key={batch_key} counts={counts} "
                f"group_ranks={tuple(dist.get_process_group_ranks(sync_group))}"
            )
        multiplicity[batch_key] = max_count

    return ordered, multiplicity


def group_items_by_batch_key(
    items: Sequence[Any],
    batch_keys: Sequence[tuple],
) -> list[tuple[tuple, list[Any]]]:
    """Group items by canonical batch key while preserving first-seen key order."""
    if len(items) != len(batch_keys):
        raise RuntimeError(
            f"[Dion] item/key length mismatch for batching: items={len(items)} keys={len(batch_keys)}"
        )
    grouped: dict[tuple, list[Any]] = {}
    for item, batch_key in zip(items, batch_keys):
        grouped.setdefault(batch_key, []).append(item)
    return list(grouped.items())


def pad_batch(batch: List[torch.Tensor], batch_size: int) -> List[torch.Tensor]:
    """Pad with inert zero tensors so partial distributed batches remain numerically stable."""
    assert len(batch) > 0
    assert len(batch) <= batch_size
    while len(batch) < batch_size:
        batch.append(torch.zeros_like(batch[0]))
    return batch


def _local_expert_tensor_view(
    tensor: torch.Tensor,
    *,
    axis: int,
    num_local_experts: int,
    local_expert_index: int,
    local_shape: Tuple[int, int],
) -> torch.Tensor:
    if axis not in (0, 1):
        raise RuntimeError(f"[DION_INVALID_EXPERT_AXIS] axis={axis}")
    if num_local_experts <= 1:
        return tensor
    if local_expert_index < 0 or local_expert_index >= num_local_experts:
        raise RuntimeError(
            "[DION_INVALID_EXPERT_LOCAL_INDEX] "
            f"axis={axis} num_local_experts={num_local_experts} local_expert_index={local_expert_index}"
        )

    expected_axis_size = int(local_shape[axis]) * int(num_local_experts)
    current_axis = int(tensor.size(axis))
    if current_axis == int(local_shape[axis]):
        return tensor
    if current_axis != expected_axis_size:
        raise RuntimeError(
            "[DION_EXPERT_TENSOR_SHAPE_MISMATCH] "
            f"axis={axis} tensor_shape={tuple(int(dim) for dim in tensor.shape)} "
            f"local_shape={local_shape} num_local_experts={num_local_experts}"
        )

    local_extent = int(local_shape[axis])
    start = int(local_expert_index) * local_extent
    end = start + local_extent
    if axis == 0:
        return tensor[start:end, :]
    return tensor[:, start:end]


def _local_expert_q_view(
    q_tensor: torch.Tensor,
    *,
    config,
    dist_meta,
    local_shape: Tuple[int, int],
) -> torch.Tensor:
    if not has_multiple_local_experts(dist_meta):
        return q_tensor

    expert_axis = int(getattr(dist_meta, "expert_axis", -1))
    num_local_experts = int(getattr(dist_meta, "num_local_experts", 1))
    local_expert_index = int(getattr(dist_meta, "local_expert_index", -1))
    q_base_axis = 0 if bool(getattr(config, "is_transposed", False)) else 1
    expected_q_rows = int(local_shape[0] if q_base_axis == 0 else local_shape[1])

    if int(q_tensor.size(0)) == expected_q_rows:
        return q_tensor
    if expert_axis != q_base_axis:
        raise RuntimeError(
            "[DION_EXPERT_Q_SHAPE_MISMATCH] "
            f"q_shape={tuple(int(dim) for dim in q_tensor.shape)} local_shape={local_shape} "
            f"expert_axis={expert_axis} q_base_axis={q_base_axis}"
        )
    return _local_expert_tensor_view(
        q_tensor,
        axis=0,
        num_local_experts=num_local_experts,
        local_expert_index=local_expert_index,
        local_shape=(expected_q_rows, int(q_tensor.size(1))),
    )


def _local_expert_views(*, param, grad, optimizer_state, config, dist_meta):
    physical_shape = tuple(int(dim) for dim in optimizer_state["momentum"].shape)
    local_shape = get_local_shape(dist_meta, physical_shape[0], physical_shape[1])
    if not has_multiple_local_experts(dist_meta):
        return param, grad, optimizer_state["momentum"], optimizer_state["Q"], local_shape

    expert_axis = int(getattr(dist_meta, "expert_axis", -1))
    num_local_experts = int(getattr(dist_meta, "num_local_experts", 1))
    local_expert_index = int(getattr(dist_meta, "local_expert_index", -1))
    param_view = _local_expert_tensor_view(
        param,
        axis=expert_axis,
        num_local_experts=num_local_experts,
        local_expert_index=local_expert_index,
        local_shape=local_shape,
    )
    grad_view = _local_expert_tensor_view(
        grad.view(*physical_shape),
        axis=expert_axis,
        num_local_experts=num_local_experts,
        local_expert_index=local_expert_index,
        local_shape=local_shape,
    )
    momentum_view = _local_expert_tensor_view(
        optimizer_state["momentum"].view(*physical_shape),
        axis=expert_axis,
        num_local_experts=num_local_experts,
        local_expert_index=local_expert_index,
        local_shape=local_shape,
    )
    q_view = _local_expert_q_view(
        optimizer_state["Q"],
        config=config,
        dist_meta=dist_meta,
        local_shape=local_shape,
    )
    return param_view, grad_view, momentum_view, q_view, local_shape


def _validate_exact_replicate_group(*, replicate_group, validation_group) -> None:
    """Fail fast unless the adapter already resolved the exact Dion replica group."""
    if replicate_group is None:
        return

    if validation_group is None:
        raise RuntimeError(
            "[DION_MISSING_REPLICA_VALIDATION_GROUP] "
            f"rank={dist.get_rank()}"
        )

    replicate_ranks = dist.get_process_group_ranks(replicate_group)
    validation_ranks = tuple(dist.get_process_group_ranks(validation_group))
    validation_rank_set = set(validation_ranks)
    leaked_ranks = [rank for rank in replicate_ranks if rank not in validation_rank_set]
    if leaked_ranks:
        raise RuntimeError(
            "[DION_INVALID_REPLICA_GROUP_DOMAIN] "
            f"rank={dist.get_rank()} replicate_ranks={replicate_ranks} "
            f"validation_ranks={validation_ranks} leaked_ranks={leaked_ranks}"
        )


def resolve_batch_group(
    *,
    config,
    dist_meta,
    use_fs_collectives: bool,
    state_replica_group,
    replica_validation_group,
    group_size: Callable,
    get_replicate_group: Callable,
    resolve_ortho_group: Callable,
) -> DionBatchGroup:
    """Return the batch execution groups from adapter runtime state."""
    replicate_group = get_replicate_group()
    if replicate_group is not None and group_size(replicate_group) > 1:
        _validate_exact_replicate_group(
            replicate_group=replicate_group,
            validation_group=replica_validation_group,
        )
    ortho_group = resolve_ortho_group(config, dist_meta)
    fs_group = getattr(dist_meta, "fs_group", None) if dist_meta is not None else None
    tp_group = getattr(dist_meta, "tp_group", None) if dist_meta is not None else None
    use_tp_shard = bool(getattr(config, "use_tp_shard", False))

    sync_groups = []
    if (
        config.compressed_all_reduce
        and replicate_group is not None
        and group_size(replicate_group) > 1
    ):
        sync_groups.append(replicate_group)

    if (
        state_replica_group is not None
        and group_size(state_replica_group) > 1
        and all(id(state_replica_group) != id(existing) for existing in sync_groups)
    ):
        sync_groups.append(state_replica_group)

    if use_tp_shard:
        if tp_group is None:
            raise RuntimeError(
                "[DION_MISSING_BATCH_TP_GROUP] "
                f"rank={dist.get_rank()} param={getattr(dist_meta, 'param_name', '')} "
                f"param_uid={getattr(dist_meta, 'param_uid', None)}"
            )
        if group_size(tp_group) > 1 and all(id(tp_group) != id(existing) for existing in sync_groups):
            sync_groups.append(tp_group)
    if bool(getattr(config, "use_fs_shard", False)):
        if fs_group is None:
            raise RuntimeError(
                "[DION_MISSING_BATCH_FS_GROUP] "
                f"rank={dist.get_rank()} param={getattr(dist_meta, 'param_name', '')} "
                f"param_uid={getattr(dist_meta, 'param_uid', None)}"
            )
        if group_size(fs_group) > 1 and all(id(fs_group) != id(existing) for existing in sync_groups):
            sync_groups.append(fs_group)

    if use_tp_shard:
        batch_world_size = group_size(tp_group)
    elif bool(getattr(config, "use_fs_shard", False)):
        batch_world_size = group_size(fs_group)
    elif config.compressed_all_reduce and replicate_group is not None:
        batch_world_size = group_size(replicate_group)
    else:
        batch_world_size = group_size(replicate_group) if replicate_group is not None else 1

    q_norm_group = None
    if (
        use_fs_collectives
        and bool(getattr(config, "use_fs_shard", False))
        and fs_group is not None
        and group_size(fs_group) > 1
    ):
        q_norm_group = fs_group

    kernel_kind = "ddp"
    if (
        p_is_tp_sharded(config, use_tp_shard=use_tp_shard)
        and ortho_group is not None
        and group_size(ortho_group) > 1
    ):
        kernel_kind = "fsdp_tp"
    elif (
        use_fs_collectives
        and is_fs_only_config(config)
        and fs_group is not None
        and group_size(fs_group) > 1
    ):
        kernel_kind = "fsdp"

    compressed_replicate_group = None
    if (
        config.compressed_all_reduce
        and kernel_kind == "fsdp"
        and replicate_group is not None
        and group_size(replicate_group) > 1
    ):
        compressed_replicate_group = replicate_group

    return DionBatchGroup(
        sync_groups=tuple(sync_groups),
        kernel_kind=kernel_kind,
        replicate_group=replicate_group,
        ortho_group=ortho_group,
        q_norm_group=q_norm_group,
        compressed_replicate_group=compressed_replicate_group,
        batch_world_size=batch_world_size,
    )


def build_batch_collectives(
    *,
    q_tensors,
    configs,
    dist_metas,
    use_fs_collectives: bool,
    resolve_tp_group: Callable,
    resolve_fs_group: Callable,
) -> DionBatchCollectives:
    """Return the TP/FS collectives for one concrete batch."""
    tp_q_gather_groups = {}
    fs_p_reduce_groups = {}
    tp_r_reduce_groups = {}
    tp_q_reshard_groups = {}
    fs_group_info = None
    fs_indices = []
    ortho_group = None

    dist_metas = dist_metas or []
    q_tensors = q_tensors or []
    real_dist_meta_count = sum(1 for dist_meta in dist_metas if dist_meta is not None)
    template_dist_meta = next((dist_meta for dist_meta in dist_metas if dist_meta is not None), None)

    def register_ortho_group(process_group, *, idx: int, axis: str) -> None:
        nonlocal ortho_group
        if process_group is None:
            return
        world_size = dist.get_world_size(process_group)
        rank = dist.get_rank(process_group)
        if world_size <= 1:
            return
        group_key = (id(process_group), world_size, rank)
        if ortho_group is None:
            ortho_group = process_group
            return
        current_key = (
            id(ortho_group),
            dist.get_world_size(ortho_group),
            dist.get_rank(ortho_group),
        )
        if current_key != group_key:
            raise RuntimeError(
                "[DION_ORTHO_GROUP_MISMATCH] "
                f"rank={dist.get_rank()} idx={idx} axis={axis} "
                f"expected={current_key} got={group_key}"
            )

    for idx, config in enumerate(configs):
        dist_meta = dist_metas[idx] if idx < len(dist_metas) else None
        if dist_meta is None:
            if idx < real_dist_meta_count:
                raise RuntimeError(
                    "[DION_UNEXPECTED_MISSING_DIST_META_IN_REAL_ENTRY] "
                    f"rank={dist.get_rank()} idx={idx} real_dist_meta_count={real_dist_meta_count}"
                )
            dist_meta = template_dist_meta
        use_tp_shard = bool(getattr(config, "use_tp_shard", False))

        if use_tp_shard:
            tp_group = resolve_tp_group(dist_meta, expect_group=True)
            if tp_group is None or dist.get_world_size(tp_group) <= 1:
                raise RuntimeError(
                    "[DION_MISSING_TP_GROUP_FOR_BATCH_COLLECTIVES] "
                    f"rank={dist.get_rank()} idx={idx} "
                    f"param={getattr(dist_meta, 'param_name', '')} "
                    f"param_uid={getattr(dist_meta, 'param_uid', None)}"
                )
            tp_key = (id(tp_group), dist.get_world_size(tp_group), dist.get_rank(tp_group))
            if tp_key not in tp_q_reshard_groups:
                tp_q_reshard_groups[tp_key] = (tp_group, [])
            tp_q_reshard_groups[tp_key][1].append(idx)
            if idx < len(q_tensors):
                q_local = q_tensors[idx]
                q_key = (
                    id(tp_group),
                    dist.get_world_size(tp_group),
                    dist.get_rank(tp_group),
                    q_local.size(0),
                    q_local.size(1),
                    q_local.dtype,
                    q_local.device,
                )
                if q_key not in tp_q_gather_groups:
                    tp_q_gather_groups[q_key] = (tp_group, [])
                tp_q_gather_groups[q_key][1].append(idx)
            if needs_tp_r_reduce(config, use_tp_shard=use_tp_shard):
                if tp_key not in tp_r_reduce_groups:
                    tp_r_reduce_groups[tp_key] = (tp_group, [])
                tp_r_reduce_groups[tp_key][1].append(idx)
            register_ortho_group(tp_group, idx=idx, axis="tp")

        if needs_fs_p_reduce(config):
            fs_group = resolve_fs_group(dist_meta, expect_group=True)
            if fs_group is None or dist.get_world_size(fs_group) <= 1:
                raise RuntimeError(
                    "[DION_MISSING_FS_GROUP_FOR_BATCH_COLLECTIVES] "
                    f"rank={dist.get_rank()} idx={idx} "
                    f"param={getattr(dist_meta, 'param_name', '')} "
                    f"param_uid={getattr(dist_meta, 'param_uid', None)}"
                )
            fs_key = (id(fs_group), dist.get_world_size(fs_group), dist.get_rank(fs_group))
            if fs_key not in fs_p_reduce_groups:
                fs_p_reduce_groups[fs_key] = (fs_group, [])
            fs_p_reduce_groups[fs_key][1].append(idx)

        if use_fs_collectives and bool(getattr(config, "use_fs_shard", False)) and not use_tp_shard:
            fs_group = resolve_fs_group(dist_meta, expect_group=True)
            if fs_group is None or dist.get_world_size(fs_group) <= 1:
                raise RuntimeError(
                    "[DION_MISSING_FS_ONLY_ORTHO_GROUP] "
                    f"rank={dist.get_rank()} idx={idx} "
                    f"param={getattr(dist_meta, 'param_name', '')} "
                    f"param_uid={getattr(dist_meta, 'param_uid', None)}"
                )
            fs_key = (id(fs_group), dist.get_world_size(fs_group), dist.get_rank(fs_group))
            if fs_group_info is None:
                fs_group_info = fs_group
            else:
                current_key = (
                    id(fs_group_info),
                    dist.get_world_size(fs_group_info),
                    dist.get_rank(fs_group_info),
                )
                if current_key != fs_key:
                    raise RuntimeError(
                        "[DION_FS_ONLY_ORTHO_GROUP_MISMATCH] "
                        f"rank={dist.get_rank()} idx={idx} expected={current_key} got={fs_key}"
                    )
            fs_indices.append(idx)
            register_ortho_group(fs_group, idx=idx, axis="fs")

    def finalize(grouped_collectives):
        collectives = []
        for process_group, indices in grouped_collectives.values():
            collectives.append(
                DionAxisCollective(
                    indices=tuple(indices),
                    process_group=process_group,
                    world_size=dist.get_world_size(process_group),
                    rank=dist.get_rank(process_group),
                )
            )
        return tuple(collectives)

    fs_collective = None
    if fs_group_info is not None:
        fs_collective = DionAxisCollective(
            indices=tuple(fs_indices),
            process_group=fs_group_info,
            world_size=dist.get_world_size(fs_group_info),
            rank=dist.get_rank(fs_group_info),
        )

    return DionBatchCollectives(
        tp_q_gathers=finalize(tp_q_gather_groups),
        fs_p_collectives=finalize(fs_p_reduce_groups),
        tp_r_collectives=finalize(tp_r_reduce_groups),
        tp_q_reshards=finalize(tp_q_reshard_groups),
        fs_collective=fs_collective,
    )


def group_and_order_param_batches(
    *,
    routed_params: list[DionStepParam],
    batch_key_cache: dict,
    use_fs_collectives: bool,
    state_replica_group,
    replica_validation_group,
    global_rank: int,
    group_size: Callable,
    get_replicate_group: Callable,
    resolve_ortho_group: Callable,
):
    """Group Dion params by batch key, then canonicalize and validate cross-rank order."""
    batch_group_by_key = {}
    batch_items = []
    batch_keys = []
    for routed_param in routed_params:
        param = routed_param.param
        state = routed_param.optimizer_state
        dist_meta = routed_param.dist_meta
        config = routed_param.config
        local_shape = state.get("local_shape", None)
        if local_shape is None:
            local_shape = require_2d_local_shape(param, dist_meta)
            state["local_shape"] = local_shape
        true_global_shape = state.get("true_global_shape", None)
        if true_global_shape is None and dist_meta is not None:
            true_global_shape = getattr(dist_meta, "global_shape", None)
        per_expert_global_shape = state.get("per_expert_global_shape", None)
        if per_expert_global_shape is None and dist_meta is not None:
            per_expert_global_shape = getattr(dist_meta, "per_expert_global_shape", None)
        batch_items.append(routed_param)
        batch_keys.append(
            (
                build_batch_key(
                    local_shape,
                    config,
                    routed_param.grad.dtype,
                    true_global_shape=true_global_shape,
                    per_expert_global_shape=per_expert_global_shape,
                ),
                _build_update_contract_key(
                    routed_param.optim_group,
                    routed_param.optimizer_state,
                ),
            )
        )

    for batch_key, grouped_items in group_items_by_batch_key(batch_items, batch_keys):
        batch_group_by_key[batch_key] = DionBatchGroup(
            params=[item.param for item in grouped_items],
            grads=[item.grad for item in grouped_items],
            optimizer_states=[item.optimizer_state for item in grouped_items],
            optim_groups=[item.optim_group for item in grouped_items],
            configs=[item.config for item in grouped_items],
            dist_metas=[item.dist_meta for item in grouped_items],
            commit_updates=[item.commit_update for item in grouped_items],
        )

    local_batch_keys = list(batch_group_by_key.keys())
    grouped_batch_keys = {}
    for batch_key in local_batch_keys:
        batch_group = batch_group_by_key[batch_key]
        resolved_group = resolve_batch_group(
            config=batch_group.configs[0],
            dist_meta=batch_group.dist_metas[0] if batch_group.dist_metas else None,
            use_fs_collectives=use_fs_collectives,
            state_replica_group=state_replica_group,
            replica_validation_group=replica_validation_group,
            group_size=group_size,
            get_replicate_group=get_replicate_group,
            resolve_ortho_group=resolve_ortho_group,
        )
        batch_group.sync_groups = resolved_group.sync_groups
        batch_group.kernel_kind = resolved_group.kernel_kind
        batch_group.replicate_group = resolved_group.replicate_group
        batch_group.ortho_group = resolved_group.ortho_group
        batch_group.q_norm_group = resolved_group.q_norm_group
        batch_group.compressed_replicate_group = resolved_group.compressed_replicate_group
        batch_group.batch_world_size = resolved_group.batch_world_size
        if not batch_group.sync_groups:
            grouped_batch_keys.setdefault(None, (None, []))[1].append(batch_key)
            continue
        for sync_group in batch_group.sync_groups:
            grouped_batch_keys.setdefault(id(sync_group), (sync_group, []))[1].append(batch_key)

    all_batch_keys = []
    batch_multiplicity_by_key: dict[tuple, int] = {}
    for sync_group, group_keys in grouped_batch_keys.values():
        ordered_keys, multiplicity = _sync_group_batch_metadata(
            sync_group=sync_group,
            local_batch_keys=group_keys,
            batch_group_by_key=batch_group_by_key,
        )
        all_batch_keys.extend(ordered_keys)
        for batch_key, count in multiplicity.items():
            existing = batch_multiplicity_by_key.get(batch_key, count)
            if existing != count:
                raise RuntimeError(
                    "[DION_BATCH_KEY_MULTIPLICITY_INCONSISTENT_ACROSS_GROUPS] "
                    f"rank={dist.get_rank()} batch_key={batch_key} "
                    f"existing={existing} new={count}"
                )
            batch_multiplicity_by_key[batch_key] = count
    all_batch_keys = unique_preserve_order(all_batch_keys)

    ordered_batches = []
    for batch_key in all_batch_keys:
        if batch_key not in batch_group_by_key:
            batch_group = batch_group_by_key.get(batch_key)
            _missing_local_shard_error(
                batch_key=batch_key,
                batch_group=batch_group,
                global_rank=global_rank,
            )

        batch_group = batch_group_by_key[batch_key]
        batch_group.local_param_count = int(batch_multiplicity_by_key.get(batch_key, 0))
        ordered_batches.append((batch_key, batch_group))

    return ordered_batches


def _build_batch_entries(
    *,
    entries: list[DionBatchEntry],
    batch_size: int,
) -> dict:
    """Assemble typed batch entries into the final DionBatch contract."""
    real_batch_size = len(entries)
    if real_batch_size <= 0:
        raise RuntimeError("[DION_EMPTY_BATCH_ENTRIES]")

    padded_entries = list(entries)
    params = [entry.param for entry in entries]
    grads = [entry.grad.view(*entry.param_shape) for entry in entries]
    momentums = [entry.momentum.view(*entry.param_shape) for entry in entries]
    q_tensors = [entry.q_tensor for entry in entries]
    configs = [entry.config for entry in entries]
    dist_metas = [entry.dist_meta for entry in entries]
    optim_groups = [entry.optim_group for entry in entries]
    optimizer_states = [entry.optimizer_state for entry in entries]
    param_shapes = [entry.param_shape for entry in entries]

    params = pad_batch(params, batch_size)
    grads = pad_batch(grads, batch_size)
    momentums = pad_batch(momentums, batch_size)
    q_tensors = pad_batch(q_tensors, batch_size)

    template = entries[0]
    while len(configs) < batch_size:
        configs.append(template.config)
    while len(dist_metas) < batch_size:
        dist_metas.append(None)
    while len(optim_groups) < batch_size:
        optim_groups.append(template.optim_group)
    while len(optimizer_states) < batch_size:
        optimizer_states.append(None)
    while len(param_shapes) < batch_size:
        param_shapes.append(template.param_shape)
    while len(padded_entries) < batch_size:
        padded_index = len(padded_entries)
        padded_entries.append(
            DionBatchEntry(
                param=params[padded_index],
                grad=grads[padded_index],
                optimizer_state=None,
                optim_group=template.optim_group,
                config=template.config,
                dist_meta=None,
                momentum=momentums[padded_index],
                q_tensor=q_tensors[padded_index],
                param_shape=template.param_shape,
            )
        )

    return {
        "entries": tuple(padded_entries),
        "params": params,
        "grads": grads,
        "momentums": momentums,
        "q_tensors": q_tensors,
        "configs": configs,
        "dist_metas": dist_metas,
        "optim_groups": optim_groups,
        "optimizer_states": optimizer_states,
        "param_shapes": tuple(param_shapes),
        "real_batch_size": real_batch_size,
    }


def build_dion_batches(
    *,
    dion_params: list[DionStepParam],
    use_fs_collectives: bool,
    state_replica_group,
    replica_validation_group,
    batch_key_cache: dict,
    global_rank: int,
    group_size: Callable,
    get_replicate_group: Callable,
    resolve_ortho_group: Callable,
    resolve_tp_group: Callable,
    resolve_fs_group: Callable,
) -> List[DionBatch]:
    """Build Dion batches from explicit runtime state."""
    ordered_batches = group_and_order_param_batches(
        routed_params=dion_params,
        batch_key_cache=batch_key_cache,
        use_fs_collectives=use_fs_collectives,
        state_replica_group=state_replica_group,
        replica_validation_group=replica_validation_group,
        global_rank=global_rank,
        group_size=group_size,
        get_replicate_group=get_replicate_group,
        resolve_ortho_group=resolve_ortho_group,
    )

    dion_batches: List[DionBatch] = []
    global_param_offset = 0
    for batch_key, batch_group in ordered_batches:
        batch_size = batch_group.batch_world_size
        local_num_params = int(getattr(batch_group, "local_param_count", len(batch_group.params or [])))

        for batch_start in range(0, local_num_params, batch_size):
            batch_end = min(batch_start + batch_size, local_num_params)
            entries: list[DionBatchEntry] = []
            for idx in range(batch_start, batch_end):
                param = batch_group.params[idx]
                optimizer_state = batch_group.optimizer_states[idx]
                config = batch_group.configs[idx]
                dist_meta = batch_group.dist_metas[idx]
                param_view, grad_view, momentum_view, q_view, param_shape = _local_expert_views(
                    param=param,
                    grad=batch_group.grads[idx],
                    optimizer_state=optimizer_state,
                    config=config,
                    dist_meta=dist_meta,
                )
                entries.append(
                    DionBatchEntry(
                        param=param_view,
                        grad=grad_view,
                        optimizer_state=optimizer_state,
                        optim_group=batch_group.optim_groups[idx],
                        config=config,
                        dist_meta=dist_meta,
                        momentum=momentum_view,
                        q_tensor=q_view,
                        param_shape=param_shape,
                        commit_update=batch_group.commit_updates[idx]
                        if batch_group.commit_updates is not None
                        else None,
                    )
                )

            batch_data = _build_batch_entries(entries=entries, batch_size=batch_size)
            batch_collectives = build_batch_collectives(
                q_tensors=batch_data["q_tensors"],
                configs=batch_data["configs"],
                dist_metas=batch_data["dist_metas"],
                use_fs_collectives=use_fs_collectives,
                resolve_tp_group=resolve_tp_group,
                resolve_fs_group=resolve_fs_group,
            )
            dion_batches.append(
                DionBatch(
                    batch_key=batch_key,
                    entries=batch_data["entries"],
                    real_batch_size=batch_data["real_batch_size"],
                    global_param_offset=global_param_offset,
                    batch_group=batch_group,
                    batch_collectives=batch_collectives,
                )
            )
            global_param_offset += batch_data["real_batch_size"]

        batch_group.params = []
        batch_group.grads = []
        batch_group.optimizer_states = []
        batch_group.optim_groups = []
        batch_group.configs = []
        batch_group.dist_metas = []
        if hasattr(batch_group, "commit_updates"):
            batch_group.commit_updates = []

    return dion_batches
