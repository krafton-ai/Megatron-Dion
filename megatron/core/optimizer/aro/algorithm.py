"""MCore-native ARO optimizer."""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

import torch
from torch.optim.optimizer import Optimizer

from .kernels import compute_aro_update, compute_aro_updates_batched
from .scalar_opts import adamw_update_foreach_, lion_update_foreach_
from .state import init_matrix_state, is_aro_matrix_param
from .types import AroMixedPrecisionConfig, AroParamConfig


def _reject_local_split_parameters(split_parameters: bool) -> None:
    if split_parameters:
        raise ValueError(
            "ARO split-parameters requires --use-distributed-optimizer; "
            "the local ARO path does not expand optimizer-only split children."
        )


class MegatronAro(Optimizer):
    """ARO optimizer with Sinkhorn base and scalar fallback."""

    def __init__(
        self,
        params,
        lr: float = 3e-4,
        momentum: float = 0.95,
        weight_decay: float = 0.01,
        base_optimizer: str = "sinkhorn",
        sinkhorn_iters: int = 5,
        qr_backend: str = "scqr",
        scqr_eps: float = 1e-6,
        update_rms_scale: float = 0.2,
        scalar_optimizer: str = "adam",
        scalar_lr_scale: float = 1.0,
        betas: tuple[float, float] = (0.9, 0.95),
        scalar_eps: float = 1e-8,
        split_parameters: bool = False,
        mixed_precision_config: Optional[AroMixedPrecisionConfig] = None,
    ) -> None:
        _reject_local_split_parameters(split_parameters)
        if isinstance(params, (list, tuple)):
            for param_group in params:
                if isinstance(param_group, dict):
                    _reject_local_split_parameters(
                        bool(param_group.get("split_parameters", False))
                    )
                if isinstance(param_group, dict) and "wd_mult" in param_group:
                    base_weight_decay = float(param_group.get("weight_decay", weight_decay))
                    param_group["weight_decay"] = base_weight_decay * float(
                        param_group.get("wd_mult", 1.0)
                    )

        defaults = dict(
            lr=lr,
            momentum=float(momentum),
            weight_decay=weight_decay,
            base_optimizer=base_optimizer,
            sinkhorn_iters=int(sinkhorn_iters),
            qr_backend=qr_backend,
            scqr_eps=float(scqr_eps),
            update_rms_scale=float(update_rms_scale),
            scalar_optimizer=scalar_optimizer,
            scalar_lr_scale=float(scalar_lr_scale),
            betas=tuple(float(beta) for beta in betas),
            scalar_eps=float(scalar_eps),
            split_parameters=bool(split_parameters),
            algorithm="aro",
        )
        AroParamConfig(
            momentum=momentum,
            base_optimizer=base_optimizer,
            sinkhorn_iters=sinkhorn_iters,
            qr_backend=qr_backend,
            scqr_eps=scqr_eps,
            update_rms_scale=update_rms_scale,
            scalar_optimizer=scalar_optimizer,
            scalar_lr_scale=scalar_lr_scale,
            beta1=betas[0],
            beta2=betas[1],
            scalar_eps=scalar_eps,
            split_parameters=split_parameters,
        )
        super().__init__(params, defaults)
        self._mixed_precision_config = mixed_precision_config or AroMixedPrecisionConfig()
        self.dist_metas = {}

    @staticmethod
    def _matrix_batch_key(momentum, rotation, state: dict, config: AroParamConfig) -> tuple:
        return (
            tuple(int(dim) for dim in state["global_shape"]),
            str(state["orientation"]),
            str(momentum.dtype),
            str(rotation.dtype),
            str(momentum.device),
            int(rotation.size(1)),
            str(config.base_optimizer),
            int(config.sinkhorn_iters),
            str(config.qr_backend),
            float(config.scqr_eps),
            float(config.update_rms_scale),
        )

    @staticmethod
    def _sinkhorn_target_numel(momentum, state: dict) -> float:
        del momentum
        global_shape = tuple(int(dim) for dim in state["global_shape"])
        return float(int(global_shape[0]) * int(global_shape[1]))

    def _group_config(self, group: dict, state: dict) -> AroParamConfig:
        beta1, beta2 = group.get("betas", self.defaults["betas"])
        orientation = state.get("orientation", "normal")
        split_parameters = bool(
            group.get("split_parameters", self.defaults["split_parameters"])
        )
        _reject_local_split_parameters(split_parameters)
        return AroParamConfig(
            momentum=float(group.get("momentum", self.defaults["momentum"])),
            base_optimizer=group.get("base_optimizer", self.defaults["base_optimizer"]),
            sinkhorn_iters=int(group.get("sinkhorn_iters", self.defaults["sinkhorn_iters"])),
            qr_backend=group.get("qr_backend", self.defaults["qr_backend"]),
            scqr_eps=float(group.get("scqr_eps", self.defaults["scqr_eps"])),
            update_rms_scale=float(
                group.get("update_rms_scale", self.defaults["update_rms_scale"])
            ),
            scalar_optimizer=group.get("scalar_optimizer", self.defaults["scalar_optimizer"]),
            scalar_lr_scale=float(
                group.get("scalar_lr_scale", self.defaults["scalar_lr_scale"])
            ),
            beta1=float(beta1),
            beta2=float(beta2),
            scalar_eps=float(group.get("scalar_eps", self.defaults["scalar_eps"])),
            split_parameters=split_parameters,
            orientation=orientation,
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = float(group.get("lr", self.defaults["lr"]))
            weight_decay = float(group.get("weight_decay", self.defaults["weight_decay"]))
            matrix_groups = defaultdict(list)
            matrix_order = []
            decay_params = []
            scalar_items = []
            for param in group["params"]:
                grad = param.grad
                if grad is None:
                    grad = getattr(param, "decoupled_grad", None)
                if grad is None:
                    continue
                state = self.state[param]
                dist_meta = self.dist_metas.get(param)
                if is_aro_matrix_param(param, dist_meta):
                    init_matrix_state(
                        param,
                        state,
                        dist_meta=dist_meta,
                        mixed_precision_config=self._mixed_precision_config,
                    )
                    config = self._group_config(group, state)
                    momentum = state["momentum"]
                    momentum.mul_(float(config.momentum)).add_(
                        grad.view_as(momentum),
                        alpha=1.0 - float(config.momentum),
                    )
                    if weight_decay != 0.0:
                        decay_params.append(param)
                    rotation = state["rotation"]
                    key = self._matrix_batch_key(
                        momentum,
                        rotation,
                        state,
                        config,
                    )
                    if key not in matrix_groups:
                        matrix_order.append(key)
                    matrix_groups[key].append((param, state, momentum, rotation, config, lr))
                    continue

                beta1, beta2 = group.get("betas", self.defaults["betas"])
                scalar_optimizer = group.get(
                    "scalar_optimizer",
                    self.defaults["scalar_optimizer"],
                )
                if scalar_optimizer not in ("adam", "adamw", "lion"):
                    raise RuntimeError(
                        f"[ARO_INVALID_SCALAR_OPTIMIZER] scalar_optimizer={scalar_optimizer!r}"
                    )
                scalar_items.append(
                    (
                        scalar_optimizer,
                        param,
                        grad,
                        state,
                        float(beta1),
                        float(beta2),
                        float(group.get("scalar_eps", self.defaults["scalar_eps"])),
                        float(group.get("scalar_lr_scale", self.defaults["scalar_lr_scale"])),
                    )
                )
            if scalar_items:
                scalar_optimizer = scalar_items[0][0]
                params = [item[1] for item in scalar_items]
                grads = [item[2] for item in scalar_items]
                states = [item[3] for item in scalar_items]
                beta1 = scalar_items[0][4]
                beta2 = scalar_items[0][5]
                scalar_eps = scalar_items[0][6]
                scalar_lr_scale = scalar_items[0][7]
                if scalar_optimizer in ("adam", "adamw"):
                    adamw_update_foreach_(
                        params=params,
                        grads=grads,
                        states=states,
                        lr=lr,
                        weight_decay=weight_decay,
                        beta1=beta1,
                        beta2=beta2,
                        eps=scalar_eps,
                        lr_scale=scalar_lr_scale,
                    )
                else:
                    lion_update_foreach_(
                        params=params,
                        grads=grads,
                        states=states,
                        lr=lr,
                        weight_decay=weight_decay,
                        beta1=beta1,
                        beta2=beta2,
                        lr_scale=scalar_lr_scale,
                    )
            if decay_params:
                torch._foreach_mul_(decay_params, 1.0 - lr * weight_decay)
            for key in matrix_order:
                items = matrix_groups[key]
                _, first_state, _momentum, _rotation, first_config, _lr = items[0]
                result = None
                if len(items) > 1:
                    result = compute_aro_updates_batched(
                        momentums=[item[2] for item in items],
                        rotations=[item[3] for item in items],
                        config=first_config,
                        orientation=first_state["orientation"],
                        target_numels=[
                            self._sinkhorn_target_numel(item[2], item[1]) for item in items
                        ],
                    )
                if result is None:
                    for param, state, momentum, rotation, config, item_lr in items:
                        update, new_rotation = compute_aro_update(
                            momentum=momentum,
                            rotation=rotation,
                            config=config,
                            orientation=state["orientation"],
                            target_numel=self._sinkhorn_target_numel(momentum, state),
                        )
                        rotation.copy_(new_rotation.to(dtype=rotation.dtype))
                        param.add_(update.to(dtype=param.dtype), alpha=-item_lr)
                    continue

                updates, new_rotations = result
                for (
                    param,
                    _state,
                    _momentum,
                    rotation,
                    _config,
                    item_lr,
                ), update, new_rotation in zip(items, updates, new_rotations):
                    rotation.copy_(new_rotation.to(dtype=rotation.dtype))
                    param.add_(update.to(dtype=param.dtype), alpha=-item_lr)
        return loss


__all__ = ["MegatronAro"]
