"""Authoritative batch route helpers for distributed Dion execution."""

from __future__ import annotations

from typing import Callable

import torch
import torch.distributed as dist

from ... import parallel_state
from ..dion.state import (
    is_fs_only_config,
    needs_fs_p_reduce,
    needs_tp_r_reduce,
    p_is_fs_sharded,
    p_is_tp_sharded,
)
from ..dion.types import (
    DionAxisCollective,
    DionBatchCollectives,
    DionBatchRoute,
)


def _resolve_batch_route(
    *,
    config,
    dist_meta,
    use_fs_collectives: bool,
    state_replica_group,
    group_size_fn: Callable,
    get_replicate_group_fn: Callable,
    resolve_ortho_group_fn: Callable,
) -> DionBatchRoute:
    """Return the authoritative batch route from adapter runtime state."""
    replicate_group = get_replicate_group_fn()
    replicate_subset_ranks = None
    if replicate_group is not None and group_size_fn(replicate_group) > 1:
        replicate_subset_ranks = resolve_replicate_subset_ranks_(replicate_group)
    ortho_group = resolve_ortho_group_fn(config, dist_meta)
    outer_shard_group = (
        getattr(dist_meta, "outer_shard_group", None) if dist_meta is not None else None
    )
    tp_group = getattr(dist_meta, "tp_group", None) if dist_meta is not None else None
    use_tp_shard = bool(getattr(config, "use_tp_shard", False))

    sync_groups = []
    if (
        config.compressed_all_reduce
        and replicate_group is not None
        and group_size_fn(replicate_group) > 1
    ):
        sync_groups.append(replicate_group)

    if (
        state_replica_group is not None
        and group_size_fn(state_replica_group) > 1
        and all(id(state_replica_group) != id(existing) for existing in sync_groups)
    ):
        sync_groups.append(state_replica_group)

    if bool(getattr(config, "use_tp_shard", False)):
        if tp_group is None:
            raise RuntimeError(
                "[DION_MISSING_BATCH_TP_GROUP] "
                f"rank={dist.get_rank()} param={getattr(dist_meta, 'param_name', '')} "
                f"param_uid={getattr(dist_meta, 'param_uid', None)}"
            )
        if group_size_fn(tp_group) > 1 and all(id(tp_group) != id(existing) for existing in sync_groups):
            sync_groups.append(tp_group)
    if bool(getattr(config, "use_fs_shard", False)):
        if outer_shard_group is None:
            raise RuntimeError(
                "[DION_MISSING_BATCH_FS_GROUP] "
                f"rank={dist.get_rank()} param={getattr(dist_meta, 'param_name', '')} "
                f"param_uid={getattr(dist_meta, 'param_uid', None)}"
            )
        if group_size_fn(outer_shard_group) > 1 and all(
            id(outer_shard_group) != id(existing) for existing in sync_groups
        ):
            sync_groups.append(outer_shard_group)

    if bool(getattr(config, "use_tp_shard", False)):
        batch_world_size = group_size_fn(tp_group)
    elif bool(getattr(config, "use_fs_shard", False)):
        batch_world_size = group_size_fn(outer_shard_group)
    elif config.compressed_all_reduce and replicate_group is not None:
        batch_world_size = group_size_fn(replicate_group)
    else:
        batch_world_size = group_size_fn(replicate_group) if replicate_group is not None else 1

    q_norm_group = None
    if (
        use_fs_collectives
        and bool(getattr(config, "use_fs_shard", False))
        and outer_shard_group is not None
        and group_size_fn(outer_shard_group) > 1
    ):
        q_norm_group = outer_shard_group

    kernel_kind = "ddp"
    if (
        p_is_tp_sharded(config, use_tp_shard=use_tp_shard)
        and ortho_group is not None
        and group_size_fn(ortho_group) > 1
    ):
        kernel_kind = "fsdp_tp"
    elif (
        use_fs_collectives
        and is_fs_only_config(config)
        and outer_shard_group is not None
        and group_size_fn(outer_shard_group) > 1
    ):
        kernel_kind = "fsdp"

    compressed_replicate_group = None
    compressed_replicate_ranks = None
    if (
        config.compressed_all_reduce
        and kernel_kind == "fsdp"
        and replicate_group is not None
        and group_size_fn(replicate_group) > 1
    ):
        compressed_replicate_group = replicate_group
        compressed_replicate_ranks = replicate_subset_ranks

    return DionBatchRoute(
        sync_groups=tuple(sync_groups),
        kernel_kind=kernel_kind,
        replicate_group=replicate_group,
        replicate_subset_ranks=(
            tuple(replicate_subset_ranks) if replicate_subset_ranks is not None else None
        ),
        ortho_group=ortho_group,
        q_norm_group=q_norm_group,
        compressed_replicate_group=compressed_replicate_group,
        compressed_replicate_ranks=(
            tuple(compressed_replicate_ranks)
            if compressed_replicate_ranks is not None
            else None
        ),
        batch_world_size=batch_world_size,
    )


def resolve_fs_only_compressed_replicate_spec_(
    replicate_group,
):
    """Return the fs-only compressed replicate surface after excluding CP-only duplication."""
    subset_ranks = resolve_replicate_subset_ranks_(replicate_group)
    return replicate_group, subset_ranks


def resolve_replicate_subset_ranks_(
    replicate_group,
):
    """Return the true Dion replicate subset after excluding CP-only duplication."""
    if replicate_group is None:
        return None

    cp_group = parallel_state.get_context_parallel_group(check_initialized=False)
    cp_world_size = dist.get_world_size(cp_group) if cp_group is not None else 1
    if cp_world_size <= 1:
        return None

    pure_dp_group = parallel_state.get_data_parallel_group(
        with_context_parallel=False,
        partial_data_parallel=False,
    )
    if pure_dp_group is None:
        raise RuntimeError(
            "[DION_FSONLY_COMPRESSED_MISSING_PURE_DP_GROUP] "
            f"rank={dist.get_rank()} cp_world_size={cp_world_size}"
        )

    replicate_ranks = dist.get_process_group_ranks(replicate_group)
    pure_dp_ranks = set(dist.get_process_group_ranks(pure_dp_group))
    subset_ranks = [rank for rank in replicate_ranks if rank in pure_dp_ranks]
    if not subset_ranks:
        raise RuntimeError(
            "[DION_FSONLY_COMPRESSED_EMPTY_REPLICA_SUBSET] "
            f"rank={dist.get_rank()} replicate_ranks={replicate_ranks} "
            f"pure_dp_ranks={sorted(pure_dp_ranks)}"
        )
    if dist.get_rank() not in subset_ranks:
        raise RuntimeError(
            "[DION_FSONLY_COMPRESSED_RANK_NOT_IN_REPLICA_SUBSET] "
            f"rank={dist.get_rank()} replicate_ranks={replicate_ranks} "
            f"subset_ranks={subset_ranks}"
        )
    if len(subset_ranks) == len(replicate_ranks):
        return None
    return subset_ranks


def _build_batch_collectives(
    *,
    q_tensors,
    configs,
    dist_metas,
    use_fs_collectives: bool,
    resolve_tp_group_fn: Callable,
    resolve_fs_group_fn: Callable,
    resolve_device_mesh_fn: Callable,
) -> DionBatchCollectives:
    """Return the authoritative TP/FS collectives for one concrete batch."""
    tp_q_gather_groups = {}
    fs_p_reduce_groups = {}
    tp_r_reduce_groups = {}
    tp_q_reshard_groups = {}
    fs_orthogonalize_group = None
    fs_orthogonalize_indices = []
    orthogonalize_group = None
    orthogonalize_mesh_dim_name = None

    dist_metas = dist_metas or []
    q_tensors = q_tensors or []

    def register_ortho_group(process_group, *, mesh_dim_name: str, idx: int, tag: str) -> None:
        nonlocal orthogonalize_group, orthogonalize_mesh_dim_name
        if process_group is None:
            return
        world_size = dist.get_world_size(process_group)
        rank = dist.get_rank(process_group)
        if world_size <= 1:
            return
        group_key = (id(process_group), world_size, rank)
        if orthogonalize_group is None:
            orthogonalize_group = process_group
            orthogonalize_mesh_dim_name = mesh_dim_name
            return
        current_key = (
            id(orthogonalize_group),
            dist.get_world_size(orthogonalize_group),
            dist.get_rank(orthogonalize_group),
        )
        if current_key != group_key or orthogonalize_mesh_dim_name != mesh_dim_name:
            raise RuntimeError(
                "[DION_ORTHO_DEVICE_MESH_PLAN_MISMATCH] "
                f"rank={dist.get_rank()} idx={idx} tag={tag} "
                f"expected={(current_key, orthogonalize_mesh_dim_name)} "
                f"got={(group_key, mesh_dim_name)}"
            )

    for idx, config in enumerate(configs):
        dist_meta = dist_metas[idx] if idx < len(dist_metas) else None
        use_tp_shard = bool(getattr(config, "use_tp_shard", False))

        if use_tp_shard:
            tp_group = resolve_tp_group_fn(dist_meta, require_in_distributed=True)
            if tp_group is None or dist.get_world_size(tp_group) <= 1:
                raise RuntimeError(
                    "[DION_MISSING_TP_BINDING_FOR_BATCH_AXIS_PLAN] "
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
            register_ortho_group(tp_group, mesh_dim_name="ortho", idx=idx, tag="tp")

        if needs_fs_p_reduce(config):
            fs_group = resolve_fs_group_fn(dist_meta, require_in_distributed=True)
            if fs_group is None or dist.get_world_size(fs_group) <= 1:
                raise RuntimeError(
                    "[DION_MISSING_FS_BINDING_FOR_BATCH_AXIS_PLAN] "
                    f"rank={dist.get_rank()} idx={idx} "
                    f"param={getattr(dist_meta, 'param_name', '')} "
                    f"param_uid={getattr(dist_meta, 'param_uid', None)}"
                )
            fs_key = (id(fs_group), dist.get_world_size(fs_group), dist.get_rank(fs_group))
            if fs_key not in fs_p_reduce_groups:
                fs_p_reduce_groups[fs_key] = (fs_group, [])
            fs_p_reduce_groups[fs_key][1].append(idx)

        if (
            use_fs_collectives
            and bool(getattr(config, "use_fs_shard", False))
            and not use_tp_shard
        ):
            fs_group = resolve_fs_group_fn(dist_meta, require_in_distributed=True)
            if fs_group is None or dist.get_world_size(fs_group) <= 1:
                raise RuntimeError(
                    "[DION_MISSING_FS_ONLY_ORTHO_BINDING_FOR_BATCH_AXIS_PLAN] "
                    f"rank={dist.get_rank()} idx={idx} "
                    f"param={getattr(dist_meta, 'param_name', '')} "
                    f"param_uid={getattr(dist_meta, 'param_uid', None)}"
                )
            fs_only_key = (id(fs_group), dist.get_world_size(fs_group), dist.get_rank(fs_group))
            if fs_orthogonalize_group is None:
                fs_orthogonalize_group = fs_group
            else:
                current_key = (
                    id(fs_orthogonalize_group),
                    dist.get_world_size(fs_orthogonalize_group),
                    dist.get_rank(fs_orthogonalize_group),
                )
                if current_key != fs_only_key:
                    raise RuntimeError(
                        "[DION_FS_ONLY_ORTHO_PLAN_MISMATCH] "
                        f"rank={dist.get_rank()} idx={idx} expected={current_key} got={fs_only_key}"
                    )
            fs_orthogonalize_indices.append(idx)
            register_ortho_group(
                fs_group,
                mesh_dim_name="fs_only_ortho",
                idx=idx,
                tag="fs_only",
            )

    def _finalize(grouped_collectives):
        plans = []
        for process_group, indices in grouped_collectives.values():
            plans.append(
                DionAxisCollective(
                    indices=tuple(indices),
                    process_group=process_group,
                    world_size=dist.get_world_size(process_group),
                    rank=dist.get_rank(process_group),
                )
            )
        return tuple(plans)

    fs_orthogonalize = None
    if fs_orthogonalize_group is not None:
        fs_orthogonalize = DionAxisCollective(
            indices=tuple(fs_orthogonalize_indices),
            process_group=fs_orthogonalize_group,
            world_size=dist.get_world_size(fs_orthogonalize_group),
            rank=dist.get_rank(fs_orthogonalize_group),
        )

    orthogonalize_mesh = None
    if orthogonalize_group is not None:
        orthogonalize_mesh = resolve_device_mesh_fn(
            orthogonalize_group,
            orthogonalize_mesh_dim_name,
        )

    return DionBatchCollectives(
        tp_q_gathers=_finalize(tp_q_gather_groups),
        fs_p_collectives=_finalize(fs_p_reduce_groups),
        tp_r_collectives=_finalize(tp_r_reduce_groups),
        tp_q_reshards=_finalize(tp_q_reshard_groups),
        fs_orthogonalize=fs_orthogonalize,
        orthogonalize_mesh=orthogonalize_mesh,
    )
