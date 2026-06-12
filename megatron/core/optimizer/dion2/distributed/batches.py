"""Dion2 distributed batch helpers."""

from __future__ import annotations

from collections import defaultdict

import torch.distributed as dist

from ..state import get_global_shape
from ..types import Dion2Batch, Dion2BatchEntry


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
    """Return a topology-aware key for a Dion2 batch."""
    dist_meta = step_param.dist_meta
    config = step_param.config
    param = step_param.param
    return (
        getattr(config, "fs_mode", fs_mode),
        getattr(config, "tp_mode", tp_mode),
        getattr(config, "ns_backend", ns_backend),
        getattr(config, "select_dim", "auto"),
        getattr(config, "selection_policy", "local_shard"),
        float(getattr(config, "fraction", 0.25)),
        float(getattr(config, "ef_decay", 0.95)),
        getattr(config, "adjust_lr", "spectral_norm"),
        getattr(config, "scale_mode", "spectral"),
        float(getattr(config, "extra_scale_factor", 0.2)),
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
    momentum = optimizer_state.get("momentum")
    if momentum is None:
        momentum = optimizer_state.get("momentum_buffer")
    return momentum


def build_dion2_batches(
    matrix_params,
    *,
    fs_mode: str = "distributed",
    tp_mode: str = "distributed",
    ns_backend: str = "standard",
    rank_cache=None,
):
    """Group routed Dion2 step params into stable batches."""
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
        entries = []
        for step_param in groups[key]:
            param_shape = tuple(
                int(dim)
                for dim in (getattr(step_param.dist_meta, "shape", None) or step_param.param.shape)
            )
            global_shape = get_global_shape(
                step_param.dist_meta,
                int(param_shape[0]),
                int(param_shape[1]),
            )
            entries.append(
                Dion2BatchEntry(
                    param=step_param.param,
                    grad=step_param.grad,
                    optimizer_state=step_param.optimizer_state,
                    optim_group=step_param.optim_group,
                    config=step_param.config,
                    dist_meta=step_param.dist_meta,
                    momentum=_state_momentum(step_param.optimizer_state),
                    param_shape=param_shape,
                    global_shape=global_shape,
                    commit_update=step_param.commit_update,
                )
            )
        entries = tuple(entries)
        batches.append(
            Dion2Batch(
                batch_key=key,
                entries=entries,
                real_batch_size=len(entries),
            )
        )
    return batches


__all__ = ["build_dion2_batches"]
