"""Muon backend for the matrix distributed optimizer."""

from __future__ import annotations

import torch

from ..matrix.backend import MatrixBackend, MatrixStateSpec
from .kernels import compute_muon_update
from .state import get_global_shape, state_backend_keys
from .types import MuonBatch, MuonBatchEntry, MuonParamConfig


class MuonBackend(MatrixBackend):
    """Muon policy behind the matrix backend boundary."""

    name = "muon"
    supports_fs = True
    supports_rp = True
    supports_tp = True
    supports_expert_parallel = True
    supports_split_qkv = True
    supports_split_qkvg = True
    supports_split_linear = True

    def refresh_state(self, adapter, *, param, state, optim_group, dist_meta) -> None:
        refresh = getattr(adapter, "_refresh_muon_step_metadata", None)
        if refresh is not None:
            refresh(
                param=param,
                optimizer_state=state,
                optim_group=optim_group,
                dist_meta=dist_meta,
            )

    def use_matrix(self, adapter, *, param, state, optim_group, dist_meta) -> bool:
        should_use = getattr(adapter, "_should_use_distributed_muon_update", None)
        if should_use is not None:
            return bool(should_use(param, state, optim_group, dist_meta))
        if dist_meta is not None and getattr(dist_meta, "is_muon_param", False):
            return bool(param.ndim == 2)
        return bool(param.ndim == 2 and optim_group.get("algorithm", "muon") == "muon")

    def split_children(self, adapter, *, param, grad, state, optim_group, config, dist_meta):
        expand = getattr(adapter, "_expand_split_muon_params", None)
        if expand is None:
            return None
        return expand(
            param=param,
            grad=grad,
            optimizer_state=state,
            optim_group=optim_group,
            config=config,
            dist_meta=dist_meta,
        )

    def sync_state(self, adapter, matrix_params) -> None:
        sync = getattr(adapter, "_sync_muon_state", None)
        if sync is not None:
            sync(matrix_params)

    def build_batches(self, adapter, matrix_params):
        build = getattr(adapter, "_build_muon_batches", None)
        if build is not None:
            return build(matrix_params)
        entries = []
        for step_param in matrix_params:
            param = step_param.param
            grad = step_param.grad
            state = step_param.optimizer_state or {}
            config = step_param.config or MuonParamConfig()
            dist_meta = step_param.dist_meta
            if param is None or grad is None:
                continue
            momentum = state.get("momentum_buffer")
            if momentum is None:
                momentum = state.get("momentum")
            if momentum is None:
                momentum = torch.zeros_like(param)
                state["momentum_buffer"] = momentum
            else:
                state.setdefault("momentum_buffer", momentum)
            m_local, n_local = int(param.size(-2)), int(param.size(-1))
            global_shape = get_global_shape(dist_meta, m_local, n_local)
            entries.append(
                MuonBatchEntry(
                    param=param,
                    grad=grad,
                    optimizer_state=state,
                    optim_group=step_param.optim_group,
                    config=config,
                    dist_meta=dist_meta,
                    momentum=momentum,
                    param_shape=(m_local, n_local),
                    global_shape=global_shape,
                    commit_update=step_param.commit_update,
                )
            )
        if not entries:
            return []
        return [
            MuonBatch(
                batch_key=("local",),
                entries=tuple(entries),
                real_batch_size=len(entries),
            )
        ]

    @torch.no_grad()
    def apply_batch(self, batch: MuonBatch) -> None:
        """Apply a local Muon batch directly."""
        for entry in batch.entries[: batch.real_batch_size]:
            if entry.param is None or entry.grad is None or entry.momentum is None:
                continue
            config = entry.config or MuonParamConfig()
            optim_group = entry.optim_group or {}
            lr = float(optim_group.get("lr", 0.0))
            weight_decay = float(optim_group.get("weight_decay", 0.0))
            if weight_decay != 0.0:
                entry.param.add_(entry.param, alpha=-lr * weight_decay)

            partition_dim = config.tp_shard_dim if config.use_tp_shard else None
            update = compute_muon_update(
                entry.grad,
                entry.momentum,
                beta=config.momentum_beta,
                nesterov=config.use_nesterov,
                ns_backend=config.ns_backend,
                coefficient_type=config.coefficient_type,
                num_ns_steps=config.num_ns_steps,
                eps=config.ns_epsilon,
                scale_mode=config.scale_mode,
                extra_scale_factor=config.extra_scale_factor,
                global_shape=entry.global_shape or None,
                tp_group=getattr(entry.dist_meta, "tp_group", None),
                partition_dim=partition_dim,
                tp_mode=config.tp_mode,
                gram_restart_iterations=config.gram_restart_iterations,
                gram_dtype=config.gram_dtype,
            )
            entry.param.add_(update.to(dtype=entry.param.dtype), alpha=-lr)
            if entry.commit_update is not None:
                entry.commit_update(entry.param, entry.momentum)

    def state_spec(self) -> MatrixStateSpec:
        return MatrixStateSpec(backend=self.name, state_keys=state_backend_keys())


__all__ = ["MuonBackend"]
