"""Scalar fallback update rules for ARO."""

from __future__ import annotations

from collections import OrderedDict

import torch

_SCALAR_FOREACH_TEMP_BYTES_CAP = 128 * 1024 * 1024


def _chunk_ranges(params: list[torch.Tensor], max_numel_per_chunk: int):
    start = 0
    chunk_numel = 0
    for idx, param in enumerate(params):
        numel = int(param.numel())
        if idx > start and chunk_numel + numel > int(max_numel_per_chunk):
            yield start, idx
            start = idx
            chunk_numel = 0
        chunk_numel += numel
    if start < len(params):
        yield start, len(params)


def _ensure_adamw_state(param: torch.Tensor, state: dict) -> tuple[torch.Tensor, torch.Tensor, int]:
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
    return exp_avg, exp_avg_sq, step


def _ensure_lion_state(param: torch.Tensor, state: dict) -> tuple[torch.Tensor, int]:
    exp_avg = state.get("exp_avg")
    if exp_avg is None or tuple(exp_avg.shape) != tuple(param.shape):
        exp_avg = torch.zeros_like(param)
        state["exp_avg"] = exp_avg
    step = int(state.get("step", 0)) + 1
    state["step"] = step
    return exp_avg, step


@torch.no_grad()
def adamw_update_foreach_(
    *,
    params: list[torch.Tensor],
    grads: list[torch.Tensor],
    states: list[dict],
    lr: float,
    weight_decay: float,
    beta1: float,
    beta2: float,
    eps: float,
    lr_scale: float = 1.0,
) -> None:
    if not params:
        return
    exp_avgs = []
    exp_avg_sqs = []
    steps = []
    for param, state in zip(params, states):
        exp_avg, exp_avg_sq, step = _ensure_adamw_state(param, state)
        exp_avgs.append(exp_avg)
        exp_avg_sqs.append(exp_avg_sq)
        steps.append(int(step))
    grouped = OrderedDict()
    for index, (param, grad, exp_avg, exp_avg_sq, step) in enumerate(
        zip(params, grads, exp_avgs, exp_avg_sqs, steps)
    ):
        key = (
            param.device,
            param.dtype,
            grad.dtype,
            exp_avg.dtype,
            exp_avg_sq.dtype,
            int(step),
        )
        grouped.setdefault(key, []).append(index)
    for indices in grouped.values():
        chunk_params = [params[index] for index in indices]
        max_numel = max(1, _SCALAR_FOREACH_TEMP_BYTES_CAP // max(1, chunk_params[0].element_size()))
        for start, end in _chunk_ranges(chunk_params, max_numel):
            local = indices[start:end]
            p = [params[index] for index in local]
            g = [grads[index].to(dtype=exp_avgs[index].dtype) for index in local]
            m = [exp_avgs[index] for index in local]
            v = [exp_avg_sqs[index] for index in local]
            step = steps[local[0]]
            torch._foreach_lerp_(m, g, [1.0 - float(beta1)] * len(local))
            g_sq = torch._foreach_mul(g, g)
            g_sq = [item.to(dtype=v[0].dtype) for item in g_sq]
            torch._foreach_lerp_(v, g_sq, [1.0 - float(beta2)] * len(local))
            bias_correction1 = 1.0 - float(beta1) ** int(step)
            bias_correction2 = 1.0 - float(beta2) ** int(step)
            denom = torch._foreach_sqrt(v)
            torch._foreach_div_(denom, bias_correction2**0.5)
            torch._foreach_add_(denom, [float(eps)] * len(local))
            updates = torch._foreach_div(m, denom)
            torch._foreach_mul_(updates, float(lr) * float(lr_scale) / bias_correction1)
            updates = [update.to(dtype=param.dtype) for update, param in zip(updates, p)]
            if float(weight_decay) != 0.0:
                torch._foreach_mul_(p, 1.0 - float(lr) * float(weight_decay))
            torch._foreach_sub_(p, updates)


@torch.no_grad()
def lion_update_foreach_(
    *,
    params: list[torch.Tensor],
    grads: list[torch.Tensor],
    states: list[dict],
    lr: float,
    weight_decay: float,
    beta1: float,
    beta2: float,
    lr_scale: float = 1.0,
) -> None:
    if not params:
        return
    exp_avgs = []
    for param, state in zip(params, states):
        exp_avg, _step = _ensure_lion_state(param, state)
        exp_avgs.append(exp_avg)
    grouped = OrderedDict()
    for index, (param, grad, exp_avg) in enumerate(zip(params, grads, exp_avgs)):
        key = (param.device, param.dtype, grad.dtype, exp_avg.dtype)
        grouped.setdefault(key, []).append(index)
    for indices in grouped.values():
        chunk_params = [params[index] for index in indices]
        max_numel = max(1, _SCALAR_FOREACH_TEMP_BYTES_CAP // max(1, chunk_params[0].element_size()))
        for start, end in _chunk_ranges(chunk_params, max_numel):
            local = indices[start:end]
            p = [params[index] for index in local]
            g = [grads[index].to(dtype=exp_avgs[index].dtype) for index in local]
            m = [exp_avgs[index] for index in local]
            updates = torch._foreach_lerp(m, g, [1.0 - float(beta1)] * len(local))
            torch._foreach_sign_(updates)
            updates = [update.to(dtype=param.dtype) for update, param in zip(updates, p)]
            if float(weight_decay) != 0.0:
                torch._foreach_mul_(p, 1.0 - float(lr) * float(weight_decay))
            torch._foreach_add_(p, updates, alpha=-float(lr) * float(lr_scale))
            torch._foreach_lerp_(m, g, [1.0 - float(beta2)] * len(local))


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
    exp_avg, exp_avg_sq, step = _ensure_adamw_state(param, state)
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
    exp_avg, _step = _ensure_lion_state(param, state)
    grad_f = grad.to(dtype=exp_avg.dtype)
    update = exp_avg.mul(float(beta1)).add(grad_f, alpha=1.0 - float(beta1)).sign()
    param.add_(update.to(dtype=param.dtype), alpha=-float(lr) * float(lr_scale))
    exp_avg.mul_(float(beta2)).add_(grad_f, alpha=1.0 - float(beta2))


__all__ = ["adamw_update_", "adamw_update_foreach_", "lion_update_", "lion_update_foreach_"]
