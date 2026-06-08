"""Scalar fallback update rules for ARO."""

from __future__ import annotations

import torch


@torch.no_grad()
def adamw_update_(
    *,
    param: torch.Tensor,
    grad: torch.Tensor,
    state: dict,
    lr: float,
    weight_decay: float,
    beta1: float,
    beta2: float,
    eps: float,
    lr_scale: float = 1.0,
) -> None:
    """Apply a decoupled AdamW-style scalar update."""
    if grad is None:
        return
    if weight_decay != 0.0:
        param.mul_(1.0 - float(lr) * float(weight_decay))
    exp_avg = state.get("exp_avg")
    if exp_avg is None or tuple(exp_avg.shape) != tuple(param.shape):
        exp_avg = torch.zeros_like(param)
        state["exp_avg"] = exp_avg
    exp_avg_sq = state.get("exp_avg_sq")
    if exp_avg_sq is None or tuple(exp_avg_sq.shape) != tuple(param.shape):
        exp_avg_sq = torch.zeros_like(param)
        state["exp_avg_sq"] = exp_avg_sq
    step = int(state.get("step", 0)) + 1
    state["step"] = step
    grad_f = grad.to(dtype=exp_avg.dtype)
    exp_avg.mul_(float(beta1)).add_(grad_f, alpha=1.0 - float(beta1))
    exp_avg_sq.mul_(float(beta2)).addcmul_(grad_f, grad_f, value=1.0 - float(beta2))
    bias_correction1 = 1.0 - float(beta1) ** step
    bias_correction2 = 1.0 - float(beta2) ** step
    denom = exp_avg_sq.sqrt().div_(bias_correction2**0.5).add_(float(eps))
    step_size = float(lr) * float(lr_scale) / bias_correction1
    param.addcdiv_(exp_avg.to(dtype=param.dtype), denom.to(dtype=param.dtype), value=-step_size)


@torch.no_grad()
def lion_update_(
    *,
    param: torch.Tensor,
    grad: torch.Tensor,
    state: dict,
    lr: float,
    weight_decay: float,
    beta1: float,
    beta2: float,
    lr_scale: float = 1.0,
) -> None:
    """Apply a decoupled Lion scalar update."""
    if grad is None:
        return
    if weight_decay != 0.0:
        param.mul_(1.0 - float(lr) * float(weight_decay))
    exp_avg = state.get("exp_avg")
    if exp_avg is None or tuple(exp_avg.shape) != tuple(param.shape):
        exp_avg = torch.zeros_like(param)
        state["exp_avg"] = exp_avg
    grad_f = grad.to(dtype=exp_avg.dtype)
    update = exp_avg.mul(float(beta1)).add(grad_f, alpha=1.0 - float(beta1)).sign()
    param.add_(update.to(dtype=param.dtype), alpha=-float(lr) * float(lr_scale))
    exp_avg.mul_(float(beta2)).add_(grad_f, alpha=1.0 - float(beta2))


__all__ = ["adamw_update_", "lion_update_"]
