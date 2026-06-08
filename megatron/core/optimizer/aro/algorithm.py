"""MCore-native ARO optimizer."""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

import torch
from torch.optim.optimizer import Optimizer

from .kernels import compute_aro_update, compute_aro_updates_batched
from .scalar_opts import adamw_update_, lion_update_
from .state import init_matrix_state, is_aro_matrix_param
from .types import AroMixedPrecisionConfig, AroParamConfig


def _reject_local_split_flags(split_qkv: bool, split_linear: bool) -> None:
    if split_qkv or split_linear:
        raise ValueError(
            "ARO split flags require --use-distributed-optimizer; "
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
        split_qkv: bool = False,
        split_linear: bool = False,
        mixed_precision_config: Optional[AroMixedPrecisionConfig] = None,
    ) -> None:
        _reject_local_split_flags(split_qkv, split_linear)
        if isinstance(params, (list, tuple)):
            for param_group in params:
                if isinstance(param_group, dict):
                    _reject_local_split_flags(
                        bool(param_group.get("split_qkv", False)),
                        bool(param_group.get("split_linear", False)),
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
            split_qkv=bool(split_qkv),
            split_linear=bool(split_linear),
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
            split_qkv=split_qkv,
            split_qkvg=split_qkv,
            split_linear=split_linear,
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

    def _group_config(self, group: dict, state: dict) -> AroParamConfig:
        beta1, beta2 = group.get("betas", self.defaults["betas"])
        orientation = state.get("orientation", "normal")
        split_qkv = bool(group.get("split_qkv", self.defaults["split_qkv"]))
        split_linear = bool(group.get("split_linear", self.defaults["split_linear"]))
        _reject_local_split_flags(split_qkv, split_linear)
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
            split_qkv=split_qkv,
            split_qkvg=split_qkv,
            split_linear=split_linear,
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
                        param.mul_(1.0 - lr * weight_decay)
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
                if scalar_optimizer in ("adam", "adamw"):
                    adamw_update_(
                        param=param,
                        grad=grad,
                        state=state,
                        lr=lr,
                        weight_decay=weight_decay,
                        beta1=float(beta1),
                        beta2=float(beta2),
                        eps=float(group.get("scalar_eps", self.defaults["scalar_eps"])),
                        lr_scale=float(
                            group.get("scalar_lr_scale", self.defaults["scalar_lr_scale"])
                        ),
                    )
                elif scalar_optimizer == "lion":
                    lion_update_(
                        param=param,
                        grad=grad,
                        state=state,
                        lr=lr,
                        weight_decay=weight_decay,
                        beta1=float(beta1),
                        beta2=float(beta2),
                        lr_scale=float(
                            group.get("scalar_lr_scale", self.defaults["scalar_lr_scale"])
                        ),
                    )
                else:
                    raise RuntimeError(
                        f"[ARO_INVALID_SCALAR_OPTIMIZER] scalar_optimizer={scalar_optimizer!r}"
                    )
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
                    )
                if result is None:
                    for param, state, momentum, rotation, config, item_lr in items:
                        update, new_rotation = compute_aro_update(
                            momentum=momentum,
                            rotation=rotation,
                            config=config,
                            orientation=state["orientation"],
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
