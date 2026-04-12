"""Dion kernel helpers that do not own Megatron runtime state."""

import math
from typing import Dict, List

import torch
from torch import Tensor

from .types import DionParamConfig


def scaled_lr_for_shape(
    *,
    lr: float,
    m_for_lr: int,
    n_for_lr: int,
    rule: str,
    rank_fraction: float,
) -> float:
    """Return the canonical 2D Dion learning-rate scaling."""
    if rule == "moonlight":
        base_scale = 0.2 / (rank_fraction ** 0.5)
        return base_scale * (max(m_for_lr, n_for_lr) ** 0.5) * lr
    if rule == "dion":
        if m_for_lr <= 0 or n_for_lr <= 0:
            raise RuntimeError(
                "[DION_INVALID_LR_SCALING_SHAPE] "
                f"m_for_lr={m_for_lr} n_for_lr={n_for_lr}"
            )
        return lr * math.sqrt(float(n_for_lr) / float(m_for_lr))
    raise RuntimeError(f"[DION_INVALID_LR_SCALING_RULE] rule={rule!r}")


def apply_batched_matmul_(
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

    if not transpose:
        update = A @ B.mT
    else:
        update = B @ A.mT

    update = update.unbind(dim=0)
    update = torch._foreach_mul(update, alpha)
    torch._foreach_mul_(X, beta)
    torch._foreach_add_(X, update)

    del update


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
        apply_batched_matmul_(
            momentums,
            P_batch,
            R_batch,
            alpha=-(1.0 - mu),
            beta=1.0,
            transpose=is_transposed,
        )
        return

    for i, momentum in enumerate(momentums):
        if configs[i].is_transposed:
            update = R_batch[i] @ P_batch[i].t()
        else:
            update = P_batch[i] @ R_batch[i].t()

        momentum.add_(update, alpha=-(1.0 - mu))
        del update


def sanitize_dion_intermediate_batch(
    P_batch: Tensor,
    R_batch: Tensor,
    Q_batch: Tensor,
    M_batch: Tensor,
):
    """Apply reference-aligned NaN/zero sanitization for batched Dion intermediates."""
    is_all_zero = (M_batch == 0).all(dim=(-2, -1), keepdim=True)
    has_nan = torch.isnan(P_batch).any(dim=(-2, -1), keepdim=True) | torch.isnan(R_batch).any(
        dim=(-2, -1), keepdim=True
    )
    unexpected_nan = has_nan & (~is_all_zero)
    not_all_zero = ~is_all_zero

    fixed_p = P_batch.nan_to_num() * not_all_zero

    q_clean = Q_batch.nan_to_num()
    if q_clean.shape != R_batch.shape:
        raise RuntimeError(
            "[DION_BAD_BATCH_Q_SHAPE_MISMATCH] "
            f"Q_shape={tuple(q_clean.shape)} R_shape={tuple(R_batch.shape)}"
        )
    fixed_r = R_batch.nan_to_num() * not_all_zero + q_clean * is_all_zero

    return fixed_p, fixed_r, unexpected_nan, is_all_zero


def local_column_sum_sq(X: Tensor) -> Tensor:
    """Return float32 per-column squared sums for one local tensor batch."""
    return X.to(dtype=torch.float32).square().sum(dim=-2, keepdim=True)


def compute_update_batch(
    Q_new_batch: Tensor,
    P_batch: Tensor,
    configs: List[DionParamConfig],
    *,
    real_batch_size: int,
    delta_shape,
) -> Tensor:
    """Compute the batched Dion low-rank update tensor before LR scaling."""
    q_new_f32 = Q_new_batch[:real_batch_size].float()
    p_for_delta = P_batch[:real_batch_size]
    is_transposed = configs[0].is_transposed
    if all(c.is_transposed == is_transposed for c in configs[:real_batch_size]):
        if is_transposed:
            delta_batch = torch.bmm(q_new_f32, p_for_delta.transpose(1, 2))
        else:
            delta_batch = torch.bmm(p_for_delta, q_new_f32.transpose(1, 2))
        return delta_batch

    delta_batch = torch.empty(
        (real_batch_size, *delta_shape),
        dtype=q_new_f32.dtype,
        device=q_new_f32.device,
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
            q_new_f32[regular_indices].transpose(1, 2),
        )
        delta_batch[regular_indices].copy_(regular_delta)
        del regular_delta

    if transposed_indices:
        transposed_delta = torch.bmm(
            q_new_f32[transposed_indices],
            p_for_delta[transposed_indices].transpose(1, 2),
        )
        delta_batch[transposed_indices].copy_(transposed_delta)
        del transposed_delta

    return delta_batch


def normalize_columns(
    R_batch: Tensor,
    col_sum_sq: Tensor,
    *,
    epsilon: float,
):
    """Return column-normalized Q_new and its local post-normalize squared sums."""
    col_norms = col_sum_sq.sqrt()
    q_new = R_batch / (col_norms + epsilon)
    post_col_sum_sq = q_new.to(torch.float32).square().sum(dim=1)
    return q_new, post_col_sum_sq
