"""Foreach-style scalar optimizer helpers for Dion scalar buckets."""

from __future__ import annotations

from typing import List

import torch
from torch import Tensor


def adamw_update_foreach(
    params: List[Tensor],
    grads: List[Tensor],
    first_moments: List[Tensor],
    second_moments: List[Tensor],
    *,
    lr: float,
    beta1: float,
    beta2: float,
    weight_decay: float,
    step: int,
    epsilon: float,
) -> None:
    """Reference-style foreach AdamW update on one homogeneous scalar bucket."""
    batch_size = len(params)
    if batch_size == 0:
        return
    if batch_size != len(grads) or batch_size != len(first_moments) or batch_size != len(second_moments):
        raise RuntimeError(
            "[DION_SCALAR_ADAMW_BUCKET_MISMATCH] "
            f"params={batch_size} grads={len(grads)} first_moments={len(first_moments)} second_moments={len(second_moments)}"
        )
    if step <= 0:
        raise RuntimeError(f"[DION_INVALID_SCALAR_ADAMW_STEP] step={step}")

    first_moment_dtype = first_moments[0].dtype
    second_moment_dtype = second_moments[0].dtype

    grads_for_first_moment = [grad.to(dtype=first_moment_dtype) for grad in grads]
    torch._foreach_lerp_(first_moments, grads_for_first_moment, [1.0 - beta1] * batch_size)

    grad_sq = torch._foreach_mul(grads_for_first_moment, grads_for_first_moment)
    grad_sq = [grad.to(dtype=second_moment_dtype) for grad in grad_sq]
    torch._foreach_lerp_(second_moments, grad_sq, [1.0 - beta2] * batch_size)

    bias_correction1 = 1.0 - beta1**step
    bias_correction2 = 1.0 - beta2**step
    bias_correction2_sqrt = bias_correction2**0.5

    denom = torch._foreach_sqrt(second_moments)
    torch._foreach_div_(denom, bias_correction2_sqrt)
    torch._foreach_add_(denom, [epsilon] * batch_size)

    if weight_decay != 0.0:
        torch._foreach_mul_(params, 1.0 - lr * weight_decay)

    updates = torch._foreach_div(first_moments, denom)
    torch._foreach_mul_(updates, lr / bias_correction1)
    torch._foreach_sub_(params, updates)


def lion_update_foreach(
    params: List[Tensor],
    grads: List[Tensor],
    first_moments: List[Tensor],
    *,
    lr: float,
    beta1: float,
    beta2: float,
    weight_decay: float,
) -> None:
    """Reference-style foreach Lion update on one homogeneous scalar bucket."""
    batch_size = len(params)
    if batch_size == 0:
        return
    if batch_size != len(grads) or batch_size != len(first_moments):
        raise RuntimeError(
            "[DION_SCALAR_LION_BUCKET_MISMATCH] "
            f"params={batch_size} grads={len(grads)} first_moments={len(first_moments)}"
        )

    first_moment_dtype = first_moments[0].dtype
    grads = [grad.to(dtype=first_moment_dtype) for grad in grads]

    updates = torch._foreach_lerp(first_moments, grads, [1.0 - beta1] * batch_size)
    torch._foreach_sign_(updates)
    torch._foreach_lerp_(first_moments, grads, [1.0 - beta2] * batch_size)

    if weight_decay != 0.0:
        torch._foreach_mul_(params, 1.0 - lr * weight_decay)

    torch._foreach_mul_(updates, lr)
    torch._foreach_sub_(params, updates)
