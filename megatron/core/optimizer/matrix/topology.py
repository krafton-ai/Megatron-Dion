"""Process-group topology helpers for matrix-aware optimizers."""

from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.core.process_groups_config import ProcessGroupCollection


def _label(name: str) -> str:
    return str(name or "Matrix optimizer")


def validate_context_parallel_excluded(
    *,
    selected_ranks: tuple[int, ...],
    source: str,
    optimizer_name: str = "Matrix optimizer",
) -> None:
    """Fail fast if a matrix optimizer communication group contains CP peers."""
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
            f"{_label(optimizer_name)} communication groups must exclude context-parallel peers. "
            f"source={source} selected_ranks={selected_ranks} "
            f"context_parallel_ranks={cp_ranks} overlap={overlap} global_rank={global_rank}"
        )


def validate_replica_group(
    *,
    selected_group: Optional[torch.distributed.ProcessGroup],
    pure_dp_group: Optional[torch.distributed.ProcessGroup],
    requested_rp_world_size: int,
    source: str,
    is_expert_parallel: bool,
    optimizer_name: str = "Matrix optimizer",
) -> Optional[torch.distributed.ProcessGroup]:
    """Validate that the selected group is the exact matrix replica domain."""
    if selected_group is None:
        return None

    selected_ranks = tuple(int(rank) for rank in dist.get_process_group_ranks(selected_group))
    validate_context_parallel_excluded(
        selected_ranks=selected_ranks,
        source=f"{source}:replica",
        optimizer_name=optimizer_name,
    )
    if len(selected_ranks) != int(requested_rp_world_size):
        raise RuntimeError(
            f"{_label(optimizer_name)} replica group size mismatch after adapter resolution. "
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
                f"{_label(optimizer_name)} replica group must be contained in the CP-excluded "
                "data-parallel domain. "
                f"source={source} requested_rp={requested_rp_world_size} "
                f"selected_ranks={selected_ranks} pure_dp_ranks={pure_dp_ranks} "
                f"leaked_ranks={leaked_ranks} is_expert_parallel={int(bool(is_expert_parallel))}"
            )

    return selected_group


def validate_dense_group(
    *,
    selected_group: Optional[torch.distributed.ProcessGroup],
    pure_dp_group: Optional[torch.distributed.ProcessGroup],
    requested_world_size: int,
    source: str,
    optimizer_name: str = "Matrix optimizer",
) -> Optional[torch.distributed.ProcessGroup]:
    """Validate an authoritative dense matrix group against CP-excluded DP."""
    if int(requested_world_size) <= 1:
        return None
    if selected_group is None:
        raise RuntimeError(
            f"{_label(optimizer_name)} dense topology requires an authoritative "
            f"Megatron-Core process group. source={source} requested_world_size={requested_world_size}"
        )
    selected_ranks = tuple(int(rank) for rank in dist.get_process_group_ranks(selected_group))
    validate_context_parallel_excluded(
        selected_ranks=selected_ranks,
        source=f"{source}:dense",
        optimizer_name=optimizer_name,
    )
    if len(selected_ranks) != int(requested_world_size):
        raise RuntimeError(
            f"{_label(optimizer_name)} dense group size mismatch. "
            f"source={source} requested_world_size={requested_world_size} "
            f"actual_world_size={len(selected_ranks)} selected_ranks={selected_ranks}"
        )
    if pure_dp_group is None:
        raise RuntimeError(
            f"{_label(optimizer_name)} dense topology requires the CP-excluded data-parallel group. "
            f"source={source} selected_ranks={selected_ranks}"
        )
    pure_dp_ranks = tuple(int(rank) for rank in dist.get_process_group_ranks(pure_dp_group))
    pure_dp_rank_set = set(pure_dp_ranks)
    leaked_ranks = tuple(rank for rank in selected_ranks if rank not in pure_dp_rank_set)
    if leaked_ranks:
        raise RuntimeError(
            f"{_label(optimizer_name)} dense group must be contained in the CP-excluded "
            "data-parallel domain. "
            f"source={source} selected_ranks={selected_ranks} "
            f"pure_dp_ranks={pure_dp_ranks} leaked_ranks={leaked_ranks}"
        )
    return selected_group


def validate_expert_fs_group(
    *,
    selected_group: Optional[torch.distributed.ProcessGroup],
    requested_world_size: int,
    source: str,
    optimizer_name: str = "Matrix optimizer",
) -> Optional[torch.distributed.ProcessGroup]:
    """Validate the expert-local FS group selected by MCore."""
    if selected_group is None:
        if int(requested_world_size) <= 1:
            return None
        raise RuntimeError(
            f"{_label(optimizer_name)} expert topology requires an authoritative "
            f"Megatron-Core process group. source={source} requested_world_size={requested_world_size}"
        )

    selected_ranks = tuple(int(rank) for rank in dist.get_process_group_ranks(selected_group))
    validate_context_parallel_excluded(
        selected_ranks=selected_ranks,
        source=f"{source}:expert_fs",
        optimizer_name=optimizer_name,
    )
    if len(selected_ranks) != int(requested_world_size):
        raise RuntimeError(
            f"{_label(optimizer_name)} expert FS group size mismatch. "
            f"source={source} requested_fs={requested_world_size} "
            f"actual_fs={len(selected_ranks)} selected_ranks={selected_ranks}. "
            "Choose FS/RP/EP/ETP sizes whose expert data-parallel shard domain matches "
            "the requested matrix optimizer FS topology."
        )
    return selected_group


def get_matrix_replica_group(
    pg_collection: Optional[ProcessGroupCollection],
    pure_dp_group: Optional[torch.distributed.ProcessGroup],
    requested_fs_world_size: int,
    requested_rp_world_size: int,
    is_expert_parallel: bool,
    *,
    optimizer_name: str = "Matrix optimizer",
) -> Optional[torch.distributed.ProcessGroup]:
    """Return the active matrix optimizer replica group."""
    if requested_rp_world_size <= 1:
        return None

    if pg_collection is not None and hasattr(pg_collection, "inter_dist_opt"):
        group = pg_collection.inter_dist_opt
        if group is not None and dist.get_world_size(group) == requested_rp_world_size:
            return validate_replica_group(
                selected_group=group,
                pure_dp_group=pure_dp_group,
                requested_rp_world_size=requested_rp_world_size,
                source="pg_collection_inter_dist_opt",
                is_expert_parallel=is_expert_parallel,
                optimizer_name=optimizer_name,
            )

    group = parallel_state.get_inter_distributed_optimizer_instance_group(check_initialized=False)
    if group is not None and dist.get_world_size(group) == requested_rp_world_size:
        return validate_replica_group(
            selected_group=group,
            pure_dp_group=pure_dp_group,
            requested_rp_world_size=requested_rp_world_size,
            source="runtime_inter_dist_opt",
            is_expert_parallel=is_expert_parallel,
            optimizer_name=optimizer_name,
        )

    if is_expert_parallel and pure_dp_group is not None:
        if dist.get_world_size(pure_dp_group) == requested_rp_world_size:
            return validate_replica_group(
                selected_group=pure_dp_group,
                pure_dp_group=pure_dp_group,
                requested_rp_world_size=requested_rp_world_size,
                source="expert_pure_dp",
                is_expert_parallel=is_expert_parallel,
                optimizer_name=optimizer_name,
            )

    raise RuntimeError(
        f"{_label(optimizer_name)} RP>1 requires a replica group, but none could be resolved. "
        f"requested_fs={requested_fs_world_size} requested_rp={requested_rp_world_size} "
        f"is_expert_parallel={int(bool(is_expert_parallel))}"
    )


def get_dense_fs_group(
    *,
    dense_fs_group: Optional[torch.distributed.ProcessGroup],
    pure_dp_group: Optional[torch.distributed.ProcessGroup],
    requested_fs_world_size: int,
    requested_rp_world_size: int,
    optimizer_name: str = "Matrix optimizer",
) -> Optional[torch.distributed.ProcessGroup]:
    """Return the dense FS group selected by MCore."""
    del requested_rp_world_size
    if requested_fs_world_size <= 1:
        return None
    if dense_fs_group is None:
        dense_fs_group = parallel_state.get_intra_distributed_optimizer_instance_data_parallel_group(
            check_initialized=False
        )
    return validate_dense_group(
        selected_group=dense_fs_group,
        pure_dp_group=pure_dp_group,
        requested_world_size=requested_fs_world_size,
        source="intra_dist_opt_data_parallel",
        optimizer_name=optimizer_name,
    )


def resolve_fs_group(
    *,
    dense_fs_group: Optional[torch.distributed.ProcessGroup],
    pure_data_parallel_group: Optional[torch.distributed.ProcessGroup],
    is_expert_parallel: bool,
    requested_fs_world_size: int,
    requested_rp_world_size: int,
    optimizer_name: str = "Matrix optimizer",
) -> Optional[torch.distributed.ProcessGroup]:
    """Resolve the FS group for dense or expert matrix optimizer parameters."""
    if is_expert_parallel:
        expert_fs_group = parallel_state.get_expert_data_parallel_group(
            partial_expert_data_parallel=True
        )
        expert_fs_world_size = (
            requested_fs_world_size
            if expert_fs_group is None
            else len(dist.get_process_group_ranks(expert_fs_group))
        )
        return validate_expert_fs_group(
            selected_group=expert_fs_group,
            requested_world_size=expert_fs_world_size,
            source="expert_partial_data_parallel",
            optimizer_name=optimizer_name,
        )
    return get_dense_fs_group(
        dense_fs_group=dense_fs_group,
        pure_dp_group=pure_data_parallel_group,
        requested_fs_world_size=requested_fs_world_size,
        requested_rp_world_size=requested_rp_world_size,
        optimizer_name=optimizer_name,
    )


def resolve_fs_rp_topology(args, *, optimizer_name: str = "Matrix optimizer") -> tuple[int, int]:
    """Resolve default FS/RP topology from the CP-excluded DP domain."""
    requested_fs = getattr(args, "fully_shard_model_parallel_size", 1) or 1
    requested_rp = getattr(args, "replicate_model_parallel_size", 1) or 1
    dense_dp_cp_domain = args.world_size // (
        args.tensor_model_parallel_size * args.pipeline_model_parallel_size
    )
    context_parallel_size = getattr(args, "context_parallel_size", 1) or 1
    if dense_dp_cp_domain % context_parallel_size != 0:
        raise RuntimeError(
            f"{_label(optimizer_name)} topology is incompatible with context parallel partitioning. "
            f"dense_dp_cp_domain={dense_dp_cp_domain} context_parallel_size={context_parallel_size}"
        )
    dense_dp_domain = dense_dp_cp_domain // context_parallel_size

    fs_explicit = requested_fs != 1
    rp_explicit = requested_rp != 1

    if fs_explicit and rp_explicit:
        resolved_fs = requested_fs
        resolved_rp = requested_rp
    elif fs_explicit:
        resolved_fs = requested_fs
        if dense_dp_domain % resolved_fs != 0:
            raise RuntimeError(
                f"{_label(optimizer_name)} FS topology is incompatible with the standard "
                "Megatron-Core distributed-optimizer shard domain. "
                f"dense_dp_domain={dense_dp_domain} requested_fs={resolved_fs}"
            )
        resolved_rp = dense_dp_domain // resolved_fs
    elif rp_explicit:
        resolved_rp = requested_rp
        if dense_dp_domain % resolved_rp != 0:
            raise RuntimeError(
                f"{_label(optimizer_name)} RP topology is incompatible with the standard "
                "Megatron-Core distributed-optimizer shard domain. "
                f"dense_dp_domain={dense_dp_domain} requested_rp={resolved_rp}"
            )
        resolved_fs = dense_dp_domain // resolved_rp
    else:
        resolved_fs = dense_dp_domain
        resolved_rp = 1

    if dense_dp_domain % resolved_fs != 0:
        raise RuntimeError(
            f"{_label(optimizer_name)} FS topology is incompatible with the standard "
            "Megatron-Core distributed-optimizer shard domain. "
            f"dense_dp_domain={dense_dp_domain} requested_fs={resolved_fs}"
        )

    expected_rp = dense_dp_domain // resolved_fs
    if resolved_rp != expected_rp:
        raise RuntimeError(
            f"{_label(optimizer_name)} RP/FS topology is incompatible with the standard "
            "Megatron-Core distributed-optimizer instance topology. "
            f"dense_dp_domain={dense_dp_domain} requested_fs={resolved_fs} "
            f"requested_rp={resolved_rp} expected_rp={expected_rp}"
        )

    expert_model_parallel_size = getattr(args, "expert_model_parallel_size", 1) or 1
    expert_tensor_parallel_size = (
        getattr(args, "expert_tensor_parallel_size", None) or args.tensor_model_parallel_size
    )
    expert_tensor_model_pipeline_parallel_size = (
        expert_tensor_parallel_size * expert_model_parallel_size * args.pipeline_model_parallel_size
    )
    if args.world_size % expert_tensor_model_pipeline_parallel_size != 0:
        raise RuntimeError(
            f"{_label(optimizer_name)} topology is incompatible with the standard "
            "Megatron-Core expert parallel topology. "
            f"world_size={args.world_size} "
            f"expert_tensor_model_pipeline_parallel_size={expert_tensor_model_pipeline_parallel_size}"
        )
    expert_data_parallel_size = args.world_size // expert_tensor_model_pipeline_parallel_size
    if expert_data_parallel_size % expected_rp != 0:
        raise RuntimeError(
            f"{_label(optimizer_name)} RP/FS topology is incompatible with the standard "
            "Megatron-Core expert distributed-optimizer shard domain. "
            f"expert_data_parallel_size={expert_data_parallel_size} "
            f"requested_rp={expected_rp} requested_fs={resolved_fs} "
            f"tensor_model_parallel_size={args.tensor_model_parallel_size} "
            f"expert_tensor_parallel_size={expert_tensor_parallel_size} "
            f"expert_model_parallel_size={expert_model_parallel_size} "
            f"pipeline_model_parallel_size={args.pipeline_model_parallel_size}"
        )

    return resolved_fs, expected_rp


__all__ = [
    "get_dense_fs_group",
    "get_matrix_replica_group",
    "resolve_fs_group",
    "resolve_fs_rp_topology",
    "validate_context_parallel_excluded",
    "validate_dense_group",
    "validate_expert_fs_group",
    "validate_replica_group",
]
