"""Foreach-style scalar optimizer helpers for Dion scalar buckets."""

from __future__ import annotations

from typing import Iterator, List, Tuple

import torch
from torch import Tensor

_SCALAR_FOREACH_TEMP_BYTES_CAP = 128 * 1024 * 1024


def _iter_chunk_slices_by_numel(params: List[Tensor], max_numel_per_chunk: int) -> Iterator[Tuple[int, int]]:
    """Yield contiguous chunk slices whose total tensor elements stay under the cap.

    Chunking is mathematically exact because scalar AdamW/Lion updates are tensor-separable:
    each tensor's state transition depends only on that tensor's local state and scalar
    hyperparameters, not on neighboring tensors in the same foreach list.
    """
    if max_numel_per_chunk <= 0:
        raise RuntimeError(
            "[DION_INVALID_SCALAR_CHUNK_CAP] "
            f"max_numel_per_chunk={max_numel_per_chunk}"
        )

    start = 0
    chunk_numel = 0
    for idx, param in enumerate(params):
        tensor_numel = int(param.numel())
        if tensor_numel <= 0:
            raise RuntimeError(
                "[DION_INVALID_SCALAR_TENSOR_NUMEL] "
                f"index={idx} numel={tensor_numel}"
            )
        if idx > start and chunk_numel + tensor_numel > max_numel_per_chunk:
            yield start, idx
            start = idx
            chunk_numel = 0
        chunk_numel += tensor_numel

    if start < len(params):
        yield start, len(params)


def _adamw_update_foreach_chunk(
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
    """Reference-style foreach AdamW update on one homogeneous scalar chunk."""
    batch_size = len(params)
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

    updates = torch._foreach_div(first_moments, denom)
    torch._foreach_mul_(updates, lr / bias_correction1)
    if weight_decay != 0.0:
        torch._foreach_mul_(params, 1.0 - lr * weight_decay)
    torch._foreach_sub_(params, updates)


def _lion_update_foreach_chunk(
    params: List[Tensor],
    grads: List[Tensor],
    first_moments: List[Tensor],
    *,
    lr: float,
    beta1: float,
    beta2: float,
    weight_decay: float,
) -> None:
    """Reference-style foreach Lion update on one homogeneous scalar chunk."""
    batch_size = len(params)
    first_moment_dtype = first_moments[0].dtype
    grads = [grad.to(dtype=first_moment_dtype) for grad in grads]

    updates = torch._foreach_lerp(first_moments, grads, [1.0 - beta1] * batch_size)
    torch._foreach_sign_(updates)
    torch._foreach_lerp_(first_moments, grads, [1.0 - beta2] * batch_size)

    torch._foreach_mul_(updates, lr)
    if weight_decay != 0.0:
        torch._foreach_mul_(params, 1.0 - lr * weight_decay)
    torch._foreach_sub_(params, updates)


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

    dtype_bytes = max(first_moments[0].element_size(), second_moments[0].element_size())
    max_numel_per_chunk = max(1, _SCALAR_FOREACH_TEMP_BYTES_CAP // dtype_bytes)

    for start, end in _iter_chunk_slices_by_numel(params, max_numel_per_chunk):
        _adamw_update_foreach_chunk(
            params[start:end],
            grads[start:end],
            first_moments[start:end],
            second_moments[start:end],
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            weight_decay=weight_decay,
            step=step,
            epsilon=epsilon,
        )


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

    max_numel_per_chunk = max(1, _SCALAR_FOREACH_TEMP_BYTES_CAP // first_moments[0].element_size())

    for start, end in _iter_chunk_slices_by_numel(params, max_numel_per_chunk):
        _lion_update_foreach_chunk(
            params[start:end],
            grads[start:end],
            first_moments[start:end],
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            weight_decay=weight_decay,
        )
