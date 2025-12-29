"""Scalar optimizer functions for non-Dion parameters (AdamW, Lion)."""

import math

import torch
from torch import Tensor


def adamw_update(
    p: Tensor,
    grad: Tensor,
    state: dict,
    betas: tuple,
    eps: float,
    lr: float,
    weight_decay: float,
) -> None:
    """
    AdamW update for non-Dion parameters.

    Args:
        p: Parameter tensor
        grad: Gradient tensor
        state: Optimizer state dict for this parameter
        betas: (beta1, beta2) tuple
        eps: Epsilon for numerical stability
        lr: Learning rate
        weight_decay: Weight decay coefficient
    """
    beta1, beta2 = betas

    if 'exp_avg' not in state:
        state['exp_avg'] = torch.zeros_like(grad, dtype=torch.float32)
        state['exp_avg_sq'] = torch.zeros_like(grad, dtype=torch.float32)
        state['step'] = 0

    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

    state['step'] += 1
    step = state['step']

    # Update moments
    grad_fp32 = grad.float() if grad.dtype != torch.float32 else grad
    exp_avg.mul_(beta1).add_(grad_fp32, alpha=1 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(grad_fp32, grad_fp32, value=1 - beta2)

    # Bias correction
    bias_correction1 = 1 - beta1 ** step
    bias_correction2 = 1 - beta2 ** step

    # Weight decay
    p.mul_(1 - lr * weight_decay)

    # Update
    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
    step_size = lr / bias_correction1

    p.data.addcdiv_(exp_avg, denom, value=-step_size)


def lion_update(
    p: Tensor,
    grad: Tensor,
    state: dict,
    betas: tuple,
    lr: float,
    weight_decay: float,
) -> None:
    """Lion optimizer update (sign-based, momentum-only)."""
    beta1, beta2 = betas

    # Initialize Lion state (only momentum, no variance!)
    if 'momentum' not in state:
        state['momentum'] = torch.zeros_like(grad, dtype=torch.float32)

    if 'step' not in state:
        state['step'] = 0

    momentum = state['momentum']
    state['step'] += 1

    # Convert gradient to FP32 for momentum
    grad_fp32 = grad.float() if grad.dtype != torch.float32 else grad

    # Compute sign update
    # U = sign(beta1 * M + (1 - beta1) * G)
    update = momentum.mul(beta1).add_(grad_fp32, alpha=1 - beta1)
    update_sign = update.sign()

    # Update momentum with new gradient
    # M = beta2 * M + (1 - beta2) * G
    momentum.mul_(beta2).add_(grad_fp32, alpha=1 - beta2)

    # Apply weight decay
    p.mul_(1 - lr * weight_decay)

    # Weight update
    # X = X - lr * sign(update)
    p.add_(update_sign.to(p.dtype), alpha=-lr)
