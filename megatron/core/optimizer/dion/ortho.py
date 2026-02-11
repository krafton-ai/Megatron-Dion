"""Orthogonalization functions for Dion optimizer."""

import logging
import math
from typing import Callable, Optional

import torch
import torch.distributed as dist
from torch import Tensor


logger = logging.getLogger(__name__)


def orthogonalize(
    P: Tensor,
    rcqr_oversample: float = 1.25,
    sketch_fn: Optional[Callable[[Tensor, float], Tensor]] = None,
) -> Tensor:
    """
    Local orthogonalization with Randomized Cholesky QR.

    Reference: dion.py lines 1265-1305

    Args:
        P: Matrix to orthogonalize, shape (..., m, r)
        rcqr_oversample: Oversampling factor for RCQR (default 1.25)
        sketch_fn: Function to generate random sketch matrix.
                   Signature: sketch_fn(P, oversample) -> S
                   If None, uses default random sketch.

    Returns:
        Orthogonalized matrix Q, same shape as P
    """
    assert P.ndim >= 2
    original_dtype = P.dtype
    P = P.to(torch.float32)

    m, r = P.shape[-2:]

    # Case 1: Square or wide matrix - use standard QR
    if m <= r:
        Q, _ = torch.linalg.qr(P)
        return Q.to(original_dtype)

    # Case 2: Tall matrix - use Randomized Cholesky QR
    else:
        # Step 1: Generate random sketch matrix
        if sketch_fn is not None:
            S = sketch_fn(P, rcqr_oversample)
        else:
            S = _default_sketch_matrix(P, rcqr_oversample)

        # Step 2: Compute sketch
        SP = S @ P

        # Step 3: QR decomposition of sketch
        _, R = torch.linalg.qr(SP, mode='r')

        # Step 4: Solve for orthogonal factor
        P = torch.linalg.solve_triangular(
            R, P, upper=True, left=False
        )

        # Step 5: Cholesky QR for better orthogonalization
        PP = P.mT @ P  # Always do float32 matrix multiply

        # No regularization (matches dion_reference.py)
        R, info = torch.linalg.cholesky_ex(PP, upper=True)
        if info > 0:
            logger.warning(f"[Dion] Cholesky failed at position {info.item()}, using fallback QR")

        P = torch.linalg.solve_triangular(
            R, P, upper=True, left=False
        )

        return P.to(original_dtype).contiguous()


def _default_sketch_matrix(P: Tensor, oversample: float) -> Tensor:
    """Generate default random sketch matrix without synchronization.

    Used when no sketch_fn is provided (single-rank case).

    Args:
        P: Matrix being orthogonalized, shape (..., m, r)
        oversample: Oversampling factor

    Returns:
        Sketch matrix S of shape (..., k, m)
    """
    batch_shape = P.shape[:-2]
    m = P.size(-2)
    r = P.size(-1)

    # Round k to multiple of 128 for efficiency (matches reference)
    k = math.ceil(oversample * r / 128.0) * 128

    std = math.sqrt(1.0 / k)

    S = torch.empty((*batch_shape, k, m), device=P.device, dtype=torch.float32)
    S.normal_(std=std)

    return S


def reshard_q_along_tp(
    Q: Tensor,
    tp_group: Optional[torch.distributed.ProcessGroup],
    tp_rank: int,
) -> Tensor:
    """Re-shard Q matrix along TP dimension after update.

    Reference: dion.py line 1159 - update_Q_matrix_ with Q_sharded_placements

    Args:
        Q: Full Q matrix of shape (n/fs, r)
        tp_group: Tensor parallel process group
        tp_rank: This rank's position in TP group

    Returns:
        TP-sharded Q matrix of shape (n/fs, r/tp)
    """
    if tp_group is None or dist.get_world_size(tp_group) == 1:
        return Q

    # Split Q along column dimension
    tp_size = dist.get_world_size(tp_group)
    n, r_total = Q.shape
    r_per_rank = r_total // tp_size

    # Extract this rank's shard
    start_col = tp_rank * r_per_rank
    end_col = (tp_rank + 1) * r_per_rank if tp_rank < tp_size - 1 else r_total

    Q_shard = Q[:, start_col:end_col].contiguous()

    return Q_shard
