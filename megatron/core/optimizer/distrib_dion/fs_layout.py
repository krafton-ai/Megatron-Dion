"""FS layout helpers for Dion distributed optimizer.

These helpers implement the orthogonal TP x FS sharding conventions used by the
Dion distributed optimizer wrapper.
"""

from __future__ import annotations

from typing import Tuple

import torch


def compute_fs_shard_range(global_size: int, fs_size: int, fs_rank: int) -> Tuple[int, int]:
    """Compute (start_idx, end_idx) for FS sharding along one dimension.

    Uneven division is handled by distributing the remainder to the lowest ranks.
    """
    if fs_size <= 0:
        raise ValueError(f"fs_size must be positive, got {fs_size}")
    if fs_rank < 0 or fs_rank >= fs_size:
        raise ValueError(f"fs_rank must be in [0, {fs_size}), got {fs_rank}")

    size_per_rank = global_size // fs_size
    remainder = global_size % fs_size

    if fs_rank < remainder:
        start_idx = fs_rank * (size_per_rank + 1)
        end_idx = start_idx + size_per_rank + 1
    else:
        start_idx = remainder * (size_per_rank + 1) + (fs_rank - remainder) * size_per_rank
        end_idx = start_idx + size_per_rank

    return start_idx, end_idx


def get_fs_split_dim(tp_split_dim: int) -> int:
    """Return FS split dim orthogonal to TP split dim.

    - tp_split_dim=0 (ColumnParallel, TP splits rows) -> FS splits cols (dim=1)
    - tp_split_dim=1 (RowParallel, TP splits cols) -> FS splits rows (dim=0)
    - tp_split_dim=-1 (no TP) -> default row split (dim=0)
    """
    if tp_split_dim == 0:
        return 1
    if tp_split_dim == 1:
        return 0
    return 0


def compute_local_shape(m: int, n: int, start_idx: int, end_idx: int, fs_split_dim: int) -> Tuple[int, int]:
    """Return local (m, n) shape after FS sharding."""
    local_split_size = end_idx - start_idx
    if fs_split_dim == 0:
        return (local_split_size, n)
    return (m, local_split_size)


def slice_fs_shard_2d(tensor2d: torch.Tensor, fs_split_dim: int, start_idx: int, end_idx: int) -> torch.Tensor:
    """Return a view for the FS shard slice of a 2D tensor."""
    if fs_split_dim == 0:
        return tensor2d[start_idx:end_idx, :]
    return tensor2d[:, start_idx:end_idx]


def write_fs_shard_2d_(
    dst2d: torch.Tensor, fs_split_dim: int, start_idx: int, end_idx: int, src2d: torch.Tensor
) -> None:
    """In-place write `src2d` into `dst2d` at the FS shard slice."""
    if fs_split_dim == 0:
        dst2d[start_idx:end_idx, :].copy_(src2d)
    else:
        dst2d[:, start_idx:end_idx].copy_(src2d)

