"""ARO distributed batch-planning helpers."""

from __future__ import annotations

from collections import defaultdict

import torch.distributed as dist

from ..types import AroBatch, AroBatchEntry


def _group_ranks_key(group, rank_cache=None):
    if group is None:
        return None
    if rank_cache is not None:
        cache_key = id(group)
        cached = rank_cache.get(cache_key)
        if cached is not None and cached[0] is group:
            return cached[1]
    if dist.is_available() and dist.is_initialized():
        ranks = tuple(int(rank) for rank in dist.get_process_group_ranks(group))
    else:
        ranks = ("uninitialized",)
    if rank_cache is not None:
        rank_cache[id(group)] = (group, ranks)
    return ranks


def build_batch_key(step_param, *, rank_cache=None):
    """Return a topology-aware key for an ARO batch."""
    dist_meta = step_param.dist_meta
    config = step_param.config
    param = step_param.param
    return (
        str(getattr(param, "dtype", None)),
        str(getattr(param, "device", None)),
        int(getattr(dist_meta, "fs_shard_dim", -1)),
        int(getattr(dist_meta, "fs_world_size", 1)),
        int(getattr(dist_meta, "tp_shard_dim", -1)),
        int(getattr(dist_meta, "tp_world_size", 1)),
        getattr(config, "base_optimizer", "sinkhorn"),
        int(getattr(config, "sinkhorn_iters", 5)),
        getattr(config, "qr_backend", "scqr"),
        getattr(config, "orientation", getattr(dist_meta, "orientation", "normal")),
        _group_ranks_key(getattr(dist_meta, "fs_group", None), rank_cache),
        _group_ranks_key(getattr(dist_meta, "tp_group", None), rank_cache),
    )


def build_aro_batches(matrix_params, *, rank_cache=None):
    """Group routed ARO step params into stable batches."""
    groups = defaultdict(list)
    for step_param in matrix_params:
        groups[build_batch_key(step_param, rank_cache=rank_cache)].append(step_param)

    batches = []
    for key in sorted(groups):
        entries = tuple(
            AroBatchEntry(
                param=step_param.param,
                grad=step_param.grad,
                optimizer_state=step_param.optimizer_state,
                optim_group=step_param.optim_group,
                config=step_param.config,
                dist_meta=step_param.dist_meta,
                momentum=(step_param.optimizer_state or {}).get("momentum"),
                rotation=(step_param.optimizer_state or {}).get("rotation"),
                param_shape=tuple(
                    getattr(step_param.dist_meta, "shape", None) or step_param.param.shape
                ),
                global_shape=tuple(
                    getattr(step_param.dist_meta, "global_shape", None)
                    or getattr(step_param.dist_meta, "shape", None)
                    or step_param.param.shape
                ),
                orientation=(step_param.optimizer_state or {}).get(
                    "orientation",
                    getattr(step_param.dist_meta, "orientation", "normal"),
                ),
                commit_update=step_param.commit_update,
            )
            for step_param in groups[key]
        )
        batches.append(
            AroBatch(
                batch_key=key,
                entries=entries,
                real_batch_size=len(entries),
            )
        )
    return batches


__all__ = ["build_aro_batches", "build_batch_key"]
