"""Muon distributed batch helpers."""

from __future__ import annotations

from collections import defaultdict

import torch.distributed as dist

from ..types import MuonBatch, MuonBatchEntry


def _group_key(group, *, rank_cache=None):
    if group is None:
        return None
    cache_key = id(group)
    if rank_cache is not None:
        cached = rank_cache.get(cache_key)
        if cached is not None and cached[0] is group:
            return cached[1]
    if dist.is_available() and dist.is_initialized():
        ranks = tuple(int(rank) for rank in dist.get_process_group_ranks(group))
    else:
        ranks = ("uninitialized",)
    if rank_cache is not None:
        rank_cache[cache_key] = (group, ranks)
    return ranks


def _batch_key(step_param, *, fs_mode: str, tp_mode: str, ns_backend: str, rank_cache=None):
    """Return a topology-aware key for a Muon batch."""
    dist_meta = step_param.dist_meta
    config = step_param.config
    param = step_param.param
    # Keep this key broad. Shape-specific grouping happens at execution time so
    # orthogonalization can batch same local-shape matrices even when their
    # logical shapes differ only for per-parameter LR scaling.
    return (
        getattr(config, "fs_mode", fs_mode),
        getattr(config, "tp_mode", tp_mode),
        getattr(config, "ns_backend", ns_backend),
        str(getattr(param, "dtype", None)),
        str(getattr(param, "device", None)),
        int(getattr(dist_meta, "fs_shard_dim", -1)),
        int(getattr(dist_meta, "fs_world_size", 1)),
        int(getattr(dist_meta, "tp_shard_dim", -1)),
        int(getattr(dist_meta, "tp_world_size", 1)),
        _group_key(getattr(dist_meta, "fs_group", None), rank_cache=rank_cache),
        _group_key(getattr(dist_meta, "tp_group", None), rank_cache=rank_cache),
    )


def _state_momentum(optimizer_state):
    if optimizer_state is None:
        return None
    momentum = optimizer_state.get("momentum_buffer")
    if momentum is None:
        momentum = optimizer_state.get("momentum")
    return momentum


def build_muon_batches(
    matrix_params,
    *,
    fs_mode: str = "blockwise",
    tp_mode: str = "blockwise",
    ns_backend: str = "standard",
    rank_cache=None,
):
    """Group routed Muon step params into stable batches."""
    groups = defaultdict(list)
    for step_param in matrix_params:
        groups[
            _batch_key(
                step_param,
                fs_mode=fs_mode,
                tp_mode=tp_mode,
                ns_backend=ns_backend,
                rank_cache=rank_cache,
            )
        ].append(step_param)

    batches = []
    for key in sorted(groups):
        entries = tuple(
            MuonBatchEntry(
                param=step_param.param,
                grad=step_param.grad,
                optimizer_state=step_param.optimizer_state,
                optim_group=step_param.optim_group,
                config=step_param.config,
                dist_meta=step_param.dist_meta,
                momentum=_state_momentum(step_param.optimizer_state),
                param_shape=tuple(
                    getattr(step_param.dist_meta, "shape", None) or step_param.param.shape
                ),
                global_shape=tuple(
                    getattr(step_param.dist_meta, "global_shape", None)
                    or getattr(step_param.dist_meta, "shape", None)
                    or step_param.param.shape
                ),
                commit_update=step_param.commit_update,
            )
            for step_param in groups[key]
        )
        batches.append(
            MuonBatch(
                batch_key=key,
                entries=entries,
                real_batch_size=len(entries),
            )
        )
    return batches
