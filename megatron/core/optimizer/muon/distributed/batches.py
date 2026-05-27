"""Muon distributed batch-planning helpers."""

from __future__ import annotations

from collections import defaultdict

from ..types import MuonBatch, MuonBatchEntry


def build_batch_key(
    *,
    shape=None,
    cfg=None,
    dtype=None,
    global_shape=None,
    per_expert_global_shape=None,
    tensor_row_shard_sizes=None,
    row_shard_sizes=None,
    param_uid=None,
    fs_group=None,
    tp_group=None,
    dist_meta=None,
):
    """Return a TP-rank-invariant key for a distributed Muon batch."""
    del tensor_row_shard_sizes, row_shard_sizes
    if dist_meta is not None:
        global_shape = global_shape or getattr(dist_meta, "global_shape", None)
        shape = shape or getattr(dist_meta, "shape", None)
        param_uid = param_uid or getattr(dist_meta, "param_uid", None)
        fs_group = fs_group or getattr(dist_meta, "fs_group", None)
        tp_group = tp_group or getattr(dist_meta, "tp_group", None)
    return (
        tuple(int(dim) for dim in (global_shape or ())),
        tuple(int(dim) for dim in (per_expert_global_shape or ())),
        tuple(int(dim) for dim in (shape or ())),
        getattr(cfg, "fs_mode", "blockwise"),
        getattr(cfg, "tp_mode", "blockwise"),
        getattr(cfg, "ns_backend", "standard"),
        getattr(cfg, "coefficient_type", "quintic"),
        int(getattr(cfg, "num_ns_steps", 5)),
        dtype,
        param_uid,
        id(fs_group) if fs_group is not None else None,
        id(tp_group) if tp_group is not None else None,
    )


def muon_batch_key(step_param, *, fs_mode: str, tp_mode: str, ns_backend: str):
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
        id(getattr(dist_meta, "fs_group", None))
        if getattr(dist_meta, "fs_group", None) is not None
        else None,
        id(getattr(dist_meta, "tp_group", None))
        if getattr(dist_meta, "tp_group", None) is not None
        else None,
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
):
    """Group routed Muon step params into stable batches."""
    groups = defaultdict(list)
    for step_param in matrix_params:
        groups[
            muon_batch_key(
                step_param,
                fs_mode=fs_mode,
                tp_mode=tp_mode,
                ns_backend=ns_backend,
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
