"""Dion kernel helpers that do not own Megatron runtime state."""

import math
from typing import Dict, Generator, List, Optional

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
from torch import Tensor

from .ortho import (
    _dion_math_precision_context,
    orthogonalize_local_matrix_batch,
    orthogonalize_local_slice,
)
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


def sanitize_zero_or_nan_dion_batch(
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


@torch.compile(fullgraph=True)
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
    q_new_f32 = Q_new_batch[:real_batch_size].float()
    p_for_delta = P_batch[:real_batch_size]
    is_transposed = configs[0].is_transposed
    with _dion_math_precision_context():
        if all(c.is_transposed == is_transposed for c in configs[:real_batch_size]):
            if is_transposed:
                return _compute_update_batch_transposed(q_new_f32, p_for_delta)
            return _compute_update_batch_regular(q_new_f32, p_for_delta)

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


@torch.compile(fullgraph=True)
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


def orthogonalize_fs_only_batch(
    optimizer,
    P_batch: torch.Tensor,
    dist_metas: List,
    *,
    batch_collectives=None,
    replicate_group: Optional[torch.distributed.ProcessGroup] = None,
) -> Generator[torch.Tensor, None, None]:
    """Match `dion_reference.dion_update_fsdp()` for fs-only batches."""
    from .runtime import replicate_reduce_op

    fs_orthogonalize = (
        batch_collectives.fs_orthogonalize if batch_collectives is not None else None
    )
    if fs_orthogonalize is None:
        raise RuntimeError(
            "[DION_MISSING_FS_ONLY_ORTHOGONALIZE_GROUP] "
            f"step={optimizer._step_count} rank={optimizer._global_rank}"
        )
    fs_group = fs_orthogonalize.process_group
    fs_world_size = fs_orthogonalize.world_size
    fs_rank = fs_orthogonalize.rank

    if fs_group is None or fs_world_size <= 1:
        yield
        return orthogonalize_local_slice(
            optimizer,
            P_batch,
            dist_metas=dist_metas,
            tag="fs_only_local",
            sketch_tag="logical_local",
        )

    batch_size = P_batch.size(0)
    if batch_size != fs_world_size:
        raise RuntimeError(
            "[DION_FSONLY_BATCH_SIZE_MISMATCH] "
            f"batch_size={batch_size} fs_world_size={fs_world_size}"
        )

    if len(fs_orthogonalize.indices) != batch_size:
        raise RuntimeError(
            "[DION_FSONLY_ORTHOGONALIZE_GROUP_SIZE_MISMATCH] "
            f"batch_size={batch_size} group_size={len(fs_orthogonalize.indices)}"
        )
    if fs_rank >= len(fs_orthogonalize.indices):
        raise RuntimeError(
            "[DION_FSONLY_ORTHOGONALIZE_RANK_MISMATCH] "
            f"fs_rank={fs_rank} group_size={len(fs_orthogonalize.indices)}"
        )
    local_slot = fs_orthogonalize.indices[fs_rank]
    fs_world = dist.get_world_size(fs_group)

    local_dist_metas = [dist_metas[local_slot]] if dist_metas and local_slot < len(dist_metas) else None
    P_single = funcol.reduce_scatter_tensor(
        P_batch.contiguous(),
        reduceOp="sum",
        scatter_dim=0,
        group=fs_group,
    )
    yield

    if replicate_group is not None and dist.get_world_size(replicate_group) > 1:
        dist.all_reduce(P_single, op=replicate_reduce_op(optimizer), group=replicate_group)
        yield

    P_single_ortho = orthogonalize_local_slice(
        optimizer,
        P_single,
        dist_metas=local_dist_metas,
        tag="fs_only_local",
        sketch_tag="logical_local",
    ).contiguous().to(P_single.dtype)

    P_ortho = funcol.all_gather_tensor(
        P_single_ortho.contiguous(),
        gather_dim=0,
        group=fs_group,
    )
    yield
    canonical_slots = tuple(int(idx) for idx in fs_orthogonalize.indices)
    if sorted(canonical_slots) != list(range(batch_size)):
        raise RuntimeError(
            "[DION_FSONLY_GATHER_PERMUTATION_INVALID] "
            f"batch_size={batch_size} indices={canonical_slots}"
        )
    if canonical_slots != tuple(range(batch_size)):
        gathered_p_ortho = P_ortho
        P_ortho = torch.empty_like(gathered_p_ortho)
        for owner_rank, batch_slot in enumerate(canonical_slots):
            P_ortho[batch_slot].copy_(gathered_p_ortho[owner_rank])
    return P_ortho


def orthogonalize_dense_replicate_batch_async(
    optimizer,
    P_batch: torch.Tensor,
    dist_metas: List,
    *,
    comm_group: torch.distributed.ProcessGroup,
    cache_scope: Optional[int] = None,
) -> Generator[torch.Tensor, None, None]:
    """Match `dion_reference.py::dion_update_ddp()` for dense replicate orthogonalization."""
    comm_world_size = dist.get_world_size(comm_group)
    if comm_world_size <= 1:
        yield
        return orthogonalize_local_matrix_batch(
            optimizer,
            P_batch,
            dist_metas=dist_metas,
            tag="dense_replicate_local",
        )

    batch_size = P_batch.size(0)
    original_batch_size = batch_size
    scope = "global" if cache_scope is None else str(int(cache_scope))
    if batch_size % comm_world_size != 0:
        pad = comm_world_size - (batch_size % comm_world_size)
        padded_batch_size = batch_size + pad
        P_padded = optimizer._cached_buffer(
            f"dense_replicate_p_padded_{scope}",
            (padded_batch_size, P_batch.size(1), P_batch.size(2)),
            P_batch.dtype,
            P_batch.device,
            zero=True,
        )
        P_padded[:batch_size].copy_(P_batch)
        P_batch = P_padded
        dist_metas = list(dist_metas) + [None] * pad
        batch_size = P_batch.size(0)

    comm_rank = dist.get_rank(comm_group)
    P_ortho_full = optimizer._cached_buffer(
        f"dense_replicate_p_ortho_full_{scope}",
        P_batch.shape,
        P_batch.dtype,
        P_batch.device,
    )

    for chunk_start in range(0, batch_size, comm_world_size):
        chunk_end = chunk_start + comm_world_size
        P_chunk = P_batch[chunk_start:chunk_end].contiguous()
        P_single = P_chunk[comm_rank : comm_rank + 1]
        local_index = chunk_start + comm_rank
        local_dist_metas = dist_metas[local_index : local_index + 1] if dist_metas else None
        P_single_ortho = orthogonalize_local_matrix_batch(
            optimizer,
            P_single,
            dist_metas=local_dist_metas,
            tag="dense_replicate_local",
        )
        P_ortho_chunk = funcol.all_gather_tensor(
            P_single_ortho.contiguous(),
            gather_dim=0,
            group=comm_group,
        )
        yield
        P_ortho_full[chunk_start:chunk_end].copy_(P_ortho_chunk)

    return P_ortho_full[:original_batch_size]
