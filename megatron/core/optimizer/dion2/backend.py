"""Dion2 backend for the matrix distributed optimizer."""

from __future__ import annotations

import torch

from ..matrix.backend import MatrixBackend, MatrixStateSpec
from .kernels import dion2_update_tensor
from .state import get_global_shape, init_matrix_state, state_backend_keys
from .types import Dion2Batch, Dion2BatchEntry, Dion2ParamConfig


class Dion2Backend(MatrixBackend):
    """Dion2 policy behind the matrix backend boundary."""

    name = "dion2"
    supports_fs = True
    supports_rp = True
    supports_tp = True
    supports_expert_parallel = True
    supports_split_parameters = True

    def refresh_state(self, adapter, *, param, state, optim_group, dist_meta) -> None:
        refresh = getattr(adapter, "_refresh_dion2_step_metadata", None)
        if refresh is not None:
            refresh(
                param=param,
                optimizer_state=state,
                optim_group=optim_group,
                dist_meta=dist_meta,
            )

    def use_matrix(self, adapter, *, param, state, optim_group, dist_meta) -> bool:
        should_use = getattr(adapter, "_should_use_distributed_dion2_update", None)
        if should_use is not None:
            return bool(should_use(param, state, optim_group, dist_meta))
        return bool(
            param.ndim == 2
            and optim_group.get("algorithm", "dion2") == "dion2"
            and dist_meta is not None
            and (
                getattr(dist_meta, "is_dion2_param", False)
                or getattr(dist_meta, "is_muon_param", False)
            )
        )

    def split_children(self, adapter, *, param, grad, state, optim_group, config, dist_meta):
        expand = getattr(adapter, "_expand_split_dion2_params", None)
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
        sync = getattr(adapter, "_sync_dion2_state", None)
        if sync is not None:
            sync(matrix_params)

    def build_batches(self, adapter, matrix_params):
        build = getattr(adapter, "_build_dion2_batches", None)
        if build is not None:
            return build(matrix_params)

        entries = []
        for step_param in matrix_params:
            param = step_param.param
            grad = step_param.grad
            state = step_param.optimizer_state or {}
            config = step_param.config or Dion2ParamConfig()
            dist_meta = step_param.dist_meta
            if param is None or grad is None:
                continue
            init_matrix_state(param, state)
            momentum = state.get("momentum")
            if momentum is None:
                momentum = state.get("momentum_buffer")
                state["momentum"] = momentum
            m_local, n_local = int(param.size(-2)), int(param.size(-1))
            global_shape = get_global_shape(dist_meta, m_local, n_local)
            entries.append(
                Dion2BatchEntry(
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
            Dion2Batch(
                batch_key=("local",),
                entries=tuple(entries),
                real_batch_size=len(entries),
            )
        ]

    @torch.no_grad()
    def apply_batch(self, batch: Dion2Batch) -> None:
        """Apply a local Dion2 batch directly."""
        for entry in batch.entries[: batch.real_batch_size]:
            if entry.param is None or entry.grad is None or entry.momentum is None:
                continue
            config = entry.config or Dion2ParamConfig()
            optim_group = entry.optim_group or {}
            lr = float(optim_group.get("lr", 0.0))
            weight_decay = float(optim_group.get("weight_decay", 0.0))
            dion2_update_tensor(
                param=entry.param,
                grad=entry.grad.view_as(entry.param),
                momentum=entry.momentum.view_as(entry.param),
                config=config,
                global_shape=entry.global_shape or tuple(entry.param.shape),
                base_lr=lr,
                weight_decay=weight_decay,
                dist_meta=entry.dist_meta,
            )
            if entry.commit_update is not None:
                entry.commit_update(entry.param, entry.momentum)

    def state_spec(self) -> MatrixStateSpec:
        return MatrixStateSpec(backend=self.name, state_keys=state_backend_keys())


__all__ = ["Dion2Backend"]
