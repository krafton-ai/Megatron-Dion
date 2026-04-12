"""Adapter-owned TP/FS group and device-mesh helpers for distributed Dion."""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

from ..dion.state import p_is_fs_sharded, p_is_tp_sharded


def _resolve_tp_group(
    meta,
    *,
    require_in_distributed: bool,
    group_size_fn: Callable,
) -> Optional[torch.distributed.ProcessGroup]:
    """Return the authoritative TP group from adapter-owned metadata."""
    param_name = getattr(meta, "param_name", "") if meta is not None else ""
    param_uid = getattr(meta, "param_uid", None) if meta is not None else None
    tp_group = getattr(meta, "tp_group", None) if meta is not None else None
    meta_tp_world_size = int(getattr(meta, "tp_world_size", 1)) if meta is not None else 1
    meta_tp_rank = int(getattr(meta, "tp_rank", -1)) if meta is not None else -1

    if tp_group is None:
        if require_in_distributed and meta_tp_world_size > 1:
            raise RuntimeError(
                "[DION_MISSING_TP_GROUP_META] "
                f"rank={dist.get_rank()} param={param_name} param_uid={param_uid} "
                f"meta_tp_world_size={meta_tp_world_size} meta_tp_rank={meta_tp_rank}"
            )
        return None

    actual_tp_world_size = group_size_fn(tp_group)
    actual_tp_rank = dist.get_rank(tp_group)
    if actual_tp_world_size != meta_tp_world_size or actual_tp_rank != meta_tp_rank:
        raise RuntimeError(
            "[DION_INCONSISTENT_TP_GROUP_META] "
            f"rank={dist.get_rank()} param={param_name} param_uid={param_uid} "
            f"meta_tp_world_size={meta_tp_world_size} actual_tp_world_size={actual_tp_world_size} "
            f"meta_tp_rank={meta_tp_rank} actual_tp_rank={actual_tp_rank}"
        )
    if actual_tp_world_size <= 1:
        return None
    return tp_group


def _resolve_fs_group(
    meta,
    *,
    require_in_distributed: bool,
    group_size_fn: Callable,
) -> Optional[torch.distributed.ProcessGroup]:
    """Return the authoritative FS group from adapter-owned metadata."""
    param_name = getattr(meta, "param_name", "") if meta is not None else ""
    param_uid = getattr(meta, "param_uid", None) if meta is not None else None
    outer_shard_group = getattr(meta, "outer_shard_group", None) if meta is not None else None
    meta_outer_shard_world_size = (
        int(getattr(meta, "fs_world_size", 1)) if meta is not None else 1
    )
    meta_outer_shard_rank = (
        int(getattr(meta, "fs_rank", -1)) if meta is not None else -1
    )

    if outer_shard_group is None:
        if require_in_distributed and meta_outer_shard_world_size > 1:
            raise RuntimeError(
                "[DION_MISSING_OUTER_SHARD_GROUP_META] "
                f"rank={dist.get_rank()} param={param_name} param_uid={param_uid} "
                "meta_outer_shard_world_size="
                f"{meta_outer_shard_world_size} meta_outer_shard_rank={meta_outer_shard_rank}"
            )
        return None

    actual_outer_shard_world_size = group_size_fn(outer_shard_group)
    actual_outer_shard_rank = dist.get_rank(outer_shard_group)
    if (
        actual_outer_shard_world_size != meta_outer_shard_world_size
        or actual_outer_shard_rank != meta_outer_shard_rank
    ):
        raise RuntimeError(
            "[DION_INCONSISTENT_OUTER_SHARD_GROUP_META] "
            f"rank={dist.get_rank()} param={param_name} param_uid={param_uid} "
            f"meta_outer_shard_world_size={meta_outer_shard_world_size} "
            f"actual_outer_shard_world_size={actual_outer_shard_world_size} "
            f"meta_outer_shard_rank={meta_outer_shard_rank} "
            f"actual_outer_shard_rank={actual_outer_shard_rank}"
        )
    if actual_outer_shard_world_size <= 1:
        return None
    return outer_shard_group


def resolve_ortho_group_(
    config,
    meta,
    *,
    use_fs_collectives: bool,
    resolve_tp_group_fn: Callable,
    resolve_fs_group_fn: Callable,
):
    """Return the authoritative orthogonalization group from adapter metadata."""
    use_tp_shard = bool(getattr(config, "use_tp_shard", False))

    if p_is_tp_sharded(config, use_tp_shard=use_tp_shard):
        return resolve_tp_group_fn(meta, require_in_distributed=True)

    if use_fs_collectives and p_is_fs_sharded(config):
        return resolve_fs_group_fn(meta, require_in_distributed=True)

    return None


def resolve_device_mesh_(
    group,
    mesh_dim_name: str,
    *,
    cache: dict,
) -> DeviceMesh:
    """Return the adapter-owned DeviceMesh for an existing process group."""
    if group is None:
        raise RuntimeError("[DION_DEVICE_MESH_MISSING_GROUP] process group is required")
    key = (id(group), mesh_dim_name)
    mesh = cache.get(key)
    if mesh is not None:
        return mesh
    group_ranks = dist.get_process_group_ranks(group)
    mesh = DeviceMesh.from_group(
        group=group,
        device_type="cuda",
        mesh=torch.tensor(group_ranks, dtype=torch.int64),
        mesh_dim_names=(mesh_dim_name,),
    )
    cache[key] = mesh
    return mesh
