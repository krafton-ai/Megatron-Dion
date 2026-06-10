"""ARO backend for the matrix distributed optimizer."""

from __future__ import annotations

from collections import defaultdict

import torch

from ..matrix.backend import MatrixBackend, MatrixStateSpec
from .kernels import compute_aro_update, compute_aro_updates_batched
from .state import get_global_shape, init_matrix_state, state_backend_keys
from .types import AroBatch, AroBatchEntry, AroParamConfig


class AroBackend(MatrixBackend):
    """ARO policy behind the matrix backend boundary."""

    name = "aro"
    supports_fs = True
    supports_rp = True
    supports_tp = True
    supports_expert_parallel = True
    supports_split_parameters = True

    def refresh_state(self, adapter, *, param, state, optim_group, dist_meta) -> None:
        refresh = getattr(adapter, "_refresh_aro_step_metadata", None)
        if refresh is not None:
            refresh(
                param=param,
                optimizer_state=state,
                optim_group=optim_group,
                dist_meta=dist_meta,
            )

    def use_matrix(self, adapter, *, param, state, optim_group, dist_meta) -> bool:
        should_use = getattr(adapter, "_should_use_distributed_aro_update", None)
        if should_use is not None:
            return bool(should_use(param, state, optim_group, dist_meta))
        return bool(dist_meta is not None and getattr(dist_meta, "is_aro_param", False))

    def split_children(self, adapter, *, param, grad, state, optim_group, config, dist_meta):
        expand = getattr(adapter, "_expand_split_aro_params", None)
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
        sync = getattr(adapter, "_sync_aro_state", None)
        if sync is not None:
            sync(matrix_params)

    def build_batches(self, adapter, matrix_params):
        build = getattr(adapter, "_build_aro_batches", None)
        if build is not None:
            return build(matrix_params)

        entries = []
        for step_param in matrix_params:
            param = step_param.param
            grad = step_param.grad
            state = step_param.optimizer_state or {}
            config = step_param.config or AroParamConfig()
            dist_meta = step_param.dist_meta
            if param is None or grad is None:
                continue
            init_matrix_state(param, state, dist_meta=dist_meta)
            momentum = state["momentum"]
            rotation = state["rotation"]
            m_local, n_local = int(param.size(-2)), int(param.size(-1))
            global_shape = get_global_shape(dist_meta, m_local, n_local)
            entries.append(
                AroBatchEntry(
                    param=param,
                    grad=grad,
                    optimizer_state=state,
                    optim_group=step_param.optim_group,
                    config=config,
                    dist_meta=dist_meta,
                    momentum=momentum,
                    rotation=rotation,
                    param_shape=(m_local, n_local),
                    global_shape=global_shape,
                    orientation=state.get("orientation", config.orientation),
                    commit_update=step_param.commit_update,
                )
            )
        if not entries:
            return []
        return [
            AroBatch(
                batch_key=("local",),
                entries=tuple(entries),
                real_batch_size=len(entries),
            )
        ]

    @torch.no_grad()
    def apply_batch(self, batch: AroBatch) -> None:
        """Apply a local ARO batch directly."""
        grouped = defaultdict(list)
        decay_groups = defaultdict(list)
        ordered_keys = []
        for entry in batch.entries[: batch.real_batch_size]:
            if entry.param is None or entry.grad is None:
                continue
            state = entry.optimizer_state or {}
            config = entry.config or AroParamConfig()
            init_matrix_state(entry.param, state, dist_meta=entry.dist_meta)
            momentum = state["momentum"]
            rotation = state["rotation"]
            beta = float(config.momentum)
            momentum.mul_(beta).add_(entry.grad.view_as(momentum), alpha=1.0 - beta)
            lr = float((entry.optim_group or {}).get("lr", 0.0))
            weight_decay = float((entry.optim_group or {}).get("weight_decay", 0.0))
            if weight_decay != 0.0:
                factor = 1.0 - lr * weight_decay
                decay_groups[(entry.param.device, entry.param.dtype, float(factor))].append(
                    entry.param
                )
            orientation = state.get("orientation", config.orientation)
            key = (
                tuple(int(dim) for dim in state.get("global_shape", entry.global_shape)),
                orientation,
                str(momentum.dtype),
                str(rotation.dtype),
                str(momentum.device),
                int(rotation.size(1)),
                str(getattr(config, "base_optimizer", "sinkhorn")),
                int(getattr(config, "sinkhorn_iters", 5)),
                str(getattr(config, "qr_backend", "scqr")),
                float(getattr(config, "scqr_eps", 1e-6)),
                float(getattr(config, "update_rms_scale", 0.2)),
            )
            if key not in grouped:
                ordered_keys.append(key)
            grouped[key].append((entry, config, momentum, rotation, lr, orientation))

        for (_device, _dtype, factor), params in decay_groups.items():
            torch._foreach_mul_(params, float(factor))

        for key in ordered_keys:
            items = grouped[key]
            _, first_config, _momentum, _rotation, _lr, orientation = items[0]
            result = None
            if len(items) > 1:
                result = compute_aro_updates_batched(
                    momentums=[item[2] for item in items],
                    rotations=[item[3] for item in items],
                    config=first_config,
                    orientation=orientation,
                )
            if result is None:
                for entry, config, momentum, rotation, lr, orientation in items:
                    update, new_rotation = compute_aro_update(
                        momentum=momentum,
                        rotation=rotation,
                        config=config,
                        orientation=orientation,
                    )
                    rotation.copy_(new_rotation.to(dtype=rotation.dtype))
                    entry.param.add_(update.to(dtype=entry.param.dtype), alpha=-lr)
                    if entry.commit_update is not None:
                        entry.commit_update(entry.param, momentum)
                continue

            updates, new_rotations = result
            for (entry, _config, momentum, rotation, lr, _orientation), update, new_rotation in zip(
                items,
                updates,
                new_rotations,
            ):
                rotation.copy_(new_rotation.to(dtype=rotation.dtype))
                entry.param.add_(update.to(dtype=entry.param.dtype), alpha=-lr)
                if entry.commit_update is not None:
                    entry.commit_update(entry.param, momentum)

    def state_spec(self) -> MatrixStateSpec:
        return MatrixStateSpec(backend=self.name, state_keys=state_backend_keys())


__all__ = ["AroBackend"]
