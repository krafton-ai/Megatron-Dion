"""Dion kernel helpers that do not own Megatron runtime state."""

import math
from typing import Dict, List

import torch
from torch import Tensor

from .ortho import (
    _dion_math_precision_context,
)
from .types import DionParamConfig


def scaled_lr_for_shape(
    *,
    lr: float,
    m_global: int,
    n_global: int,
    scale_mode: str,
    rank_fraction: float,
    extra_scale_factor: float = 0.2,
) -> float:
    """Return the canonical 2D Dion learning-rate scaling."""
    if m_global <= 0 or n_global <= 0:
        raise RuntimeError(
            "[DION_INVALID_SCALE_SHAPE] "
            f"m_global={m_global} n_global={n_global}"
        )
    if rank_fraction <= 0.0:
        raise RuntimeError(f"[DION_INVALID_RANK_FRACTION] rank_fraction={rank_fraction}")

    rank_scale = extra_scale_factor / math.sqrt(float(rank_fraction))
    if scale_mode == "spectral":
        return lr * rank_scale * math.sqrt(float(max(m_global, n_global)))
    if scale_mode == "unit_rms_norm":
        return lr * rank_scale * math.sqrt(float(m_global) / float(n_global))
    if scale_mode == "shape_scaling":
        return lr * rank_scale * math.sqrt(max(1.0, float(m_global) / float(n_global)))
    raise RuntimeError(f"[DION_INVALID_SCALE_MODE] scale_mode={scale_mode!r}")


@torch.compile(fullgraph=True)
def _apply_batched_matmul_regular(
    X: List[Tensor],
    A: Tensor,
    B: Tensor,
    *,
    alpha: float,
    beta: float,
) -> None:
    update = A @ B.mT
    update = update.unbind(dim=0)
    update = torch._foreach_mul(update, alpha)
    torch._foreach_mul_(X, beta)
    torch._foreach_add_(X, update)


@torch.compile(fullgraph=True)
def _apply_batched_matmul_transposed(
    X: List[Tensor],
    A: Tensor,
    B: Tensor,
    *,
    alpha: float,
    beta: float,
) -> None:
    update = B @ A.mT
    update = update.unbind(dim=0)
    update = torch._foreach_mul(update, alpha)
    torch._foreach_mul_(X, beta)
    torch._foreach_add_(X, update)


def apply_batched_matmul(
    X: List[Tensor],
    A: Tensor,
    B: Tensor,
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
    transpose: bool = False,
) -> None:
    """Batch matrix multiplication and in-place addition."""
    if A.size(0) != B.size(0) or A.size(0) != len(X):
        raise RuntimeError(
            "[DION_INVALID_BATCH_BADDMM] "
            f"A_batch={A.size(0)} B_batch={B.size(0)} X_batch={len(X)}"
        )

    with _dion_math_precision_context():
        if not transpose:
            _apply_batched_matmul_regular(X, A, B, alpha=alpha, beta=beta)
        else:
            _apply_batched_matmul_transposed(X, A, B, alpha=alpha, beta=beta)


def apply_error_feedback(
    momentums: List[Tensor],
    P_batch: Tensor,
    R_batch: Tensor,
    configs: List[DionParamConfig],
    groups: List[Dict],
    *,
    default_mu: float,
) -> None:
    """Apply Dion error feedback to batched momentum buffers."""
    mu = groups[0].get("mu", default_mu)

    is_transposed = configs[0].is_transposed
    if all(c.is_transposed == is_transposed for c in configs):
        apply_batched_matmul(
            momentums,
            P_batch,
            R_batch,
            alpha=-(1.0 - mu),
            beta=1.0,
            transpose=is_transposed,
        )
        return

    for i, momentum in enumerate(momentums):
        with _dion_math_precision_context():
            if configs[i].is_transposed:
                update = R_batch[i] @ P_batch[i].t()
            else:
                update = P_batch[i] @ R_batch[i].t()

        momentum.add_(update, alpha=-(1.0 - mu))
        del update


def fix_all_zero_or_nan(
    P_batch: Tensor,
    R_batch: Tensor,
    Q_batch: Tensor,
    M_batch: Tensor,
):
    """Reference-aligned NaN/zero fix for batched Dion intermediates."""
    is_all_zero = (M_batch == 0).all(dim=(-2, -1), keepdim=True)
    not_all_zero = ~is_all_zero

    fixed_p = P_batch.nan_to_num() * not_all_zero

    q_clean = Q_batch.nan_to_num()
    if q_clean.shape != R_batch.shape:
        raise RuntimeError(
            "[DION_BAD_BATCH_Q_SHAPE_MISMATCH] "
            f"Q_shape={tuple(q_clean.shape)} R_shape={tuple(R_batch.shape)}"
        )
    fixed_r = R_batch.nan_to_num() * not_all_zero + q_clean * is_all_zero

    return fixed_p, fixed_r


@torch.compiler.disable
def local_column_sum_sq(X: Tensor) -> Tensor:
    """Return float32 per-column squared sums for one local tensor batch."""
    return X.to(dtype=torch.float32).square().sum(dim=-2, keepdim=True)


@torch.compile(fullgraph=True)
def _compute_update_batch_regular(
    q_new_f32: Tensor,
    p_for_delta: Tensor,
) -> Tensor:
    return torch.bmm(p_for_delta, q_new_f32.transpose(1, 2))


@torch.compile(fullgraph=True)
def _compute_update_batch_transposed(
    q_new_f32: Tensor,
    p_for_delta: Tensor,
) -> Tensor:
    return torch.bmm(q_new_f32, p_for_delta.transpose(1, 2))


def compute_update_batch(
    Q_new_batch: Tensor,
    P_batch: Tensor,
    configs: List[DionParamConfig],
    *,
    real_batch_size: int,
    delta_shape,
) -> Tensor:
    """Compute the batched Dion low-rank update tensor before LR scaling."""
    p_for_delta = P_batch[:real_batch_size]
    q_new_for_delta = Q_new_batch[:real_batch_size].to(dtype=p_for_delta.dtype)
    is_transposed = configs[0].is_transposed
    with _dion_math_precision_context():
        if all(c.is_transposed == is_transposed for c in configs[:real_batch_size]):
            if is_transposed:
                return _compute_update_batch_transposed(q_new_for_delta, p_for_delta)
            return _compute_update_batch_regular(q_new_for_delta, p_for_delta)

        delta_batch = torch.empty(
            (real_batch_size, *delta_shape),
            dtype=q_new_for_delta.dtype,
            device=q_new_for_delta.device,
        )

        transposed_indices = [
            i for i, cfg in enumerate(configs[:real_batch_size]) if cfg.is_transposed
        ]
        regular_indices = [
            i for i, cfg in enumerate(configs[:real_batch_size]) if not cfg.is_transposed
        ]

        if regular_indices:
            regular_delta = torch.bmm(
                p_for_delta[regular_indices],
                q_new_for_delta[regular_indices].transpose(1, 2),
            )
            delta_batch[regular_indices].copy_(regular_delta)
            del regular_delta

        if transposed_indices:
            transposed_delta = torch.bmm(
                q_new_for_delta[transposed_indices],
                p_for_delta[transposed_indices].transpose(1, 2),
            )
            delta_batch[transposed_indices].copy_(transposed_delta)
            del transposed_delta

        return delta_batch


@torch.compiler.disable
def normalize_columns(
    R_batch: Tensor,
    col_sum_sq: Tensor,
    *,
    epsilon: float,
):
    """Return column-normalized Q_new and its local post-normalize squared sums."""
    original_dtype = R_batch.dtype
    safe_denominator = col_sum_sq.sqrt().add_(epsilon)
    q_new = R_batch.to(dtype=torch.float32) / safe_denominator
    q_new = q_new.to(dtype=original_dtype)
    post_col_sum_sq = q_new.to(torch.float32).square().sum(dim=1)
    return q_new, post_col_sum_sq
