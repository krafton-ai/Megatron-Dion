"""Factory helpers for distributed MCore Muon."""

from __future__ import annotations

import torch
import torch.distributed as dist

from megatron.core.optimizer.matrix.topology import (
    get_matrix_replica_group,
    resolve_fs_group,
)

from .optimizer import DistributedMuonOptimizer


def _resolve_muon_fs_group(
    *,
    dense_fs_group: torch.distributed.ProcessGroup | None,
    pure_data_parallel_group: torch.distributed.ProcessGroup | None,
    is_expert_parallel: bool,
    requested_fs_size: int,
    requested_rp_size: int,
) -> torch.distributed.ProcessGroup | None:
    fs_group = resolve_fs_group(
        dense_fs_group=dense_fs_group,
        pure_data_parallel_group=pure_data_parallel_group,
        is_expert_parallel=is_expert_parallel,
        requested_fs_world_size=requested_fs_size,
        requested_rp_world_size=requested_rp_size,
        optimizer_name="Muon optimizer",
    )
    if fs_group is not None:
        fs_size = len(dist.get_process_group_ranks(fs_group))
        if not is_expert_parallel and fs_size != int(requested_fs_size):
            raise RuntimeError(
                "Muon FS topology mismatch while constructing DistributedMuonOptimizer. "
                f"requested_fs={requested_fs_size} actual_fs_group_size={fs_size} "
                f"(fs_group_ranks={dist.get_process_group_ranks(fs_group)})."
            )
    return fs_group


def build_muon_distributed_optimizer(
    *,
    optimizer_args,
    dense_fs_group=None,
    muon_tp_group=None,
    **kwargs,
):
    """Build a DistributedMuonOptimizer with MCore distributed-optimizer ownership."""
    config = kwargs.pop("config", None)
    pure_data_parallel_group = kwargs.pop("pure_data_parallel_group", None)
    pg_collection = kwargs.pop("pg_collection", None)
    is_expert_parallel = bool(kwargs.pop("is_expert_parallel", False))
    requested_fs_size = 1
    requested_rp_size = 1
    if config is not None:
        requested_fs_size = int(getattr(config, "fully_shard_model_parallel_size", 1) or 1)
        requested_rp_size = int(getattr(config, "replicate_model_parallel_size", 1) or 1)
        # resolve_fs_group lives in the helper above; keep this factory as the
        # single place that wires FS/RP topology into DistributedMuonOptimizer.
        fs_group = _resolve_muon_fs_group(
            dense_fs_group=dense_fs_group,
            pure_data_parallel_group=pure_data_parallel_group,
            is_expert_parallel=is_expert_parallel,
            requested_fs_size=requested_fs_size,
            requested_rp_size=requested_rp_size,
        )
        fs_size = 1 if fs_group is None else len(dist.get_process_group_ranks(fs_group))
        replica_group = get_matrix_replica_group(
            pg_collection,
            pure_data_parallel_group,
            requested_fs_size,
            requested_rp_size,
            is_expert_parallel,
            optimizer_name="Muon optimizer",
        )
        kwargs.setdefault("fully_shard_model_parallel_size", fs_size)
        kwargs.setdefault("replica_model_parallel_size", requested_rp_size)
        kwargs.setdefault("replica_group", replica_group)
        kwargs.setdefault("muon_fs_mode", getattr(config, "muon_fs_mode", "blockwise"))
        kwargs.setdefault("muon_tp_mode", getattr(config, "muon_tp_mode", "blockwise"))
        kwargs.setdefault("muon_ns_backend", getattr(config, "muon_ns_backend", "standard"))
    else:
        fs_group = dense_fs_group
    kwargs.setdefault("is_expert_muon", is_expert_parallel)
    return DistributedMuonOptimizer(
        *optimizer_args,
        muon_fs_group=fs_group,
        muon_tp_group=muon_tp_group,
        pure_data_parallel_group=pure_data_parallel_group,
        **kwargs,
    )
