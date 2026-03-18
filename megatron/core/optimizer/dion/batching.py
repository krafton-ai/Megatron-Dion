"""Dion optimizer batching helpers aligned with the reference implementation."""

from __future__ import annotations

from typing import Iterable, Iterator, List, Sequence

import torch
from torch import Tensor


def _normalize_shard_dim(dim, has_axis: bool) -> int:
    if dim is None and not has_axis:
        return -1
    if dim is None:
        raise RuntimeError(
            "[Dion] missing shard tensor dim for active sharded axis in batch key construction"
        )
    return int(dim)


def build_batch_key(shape, cfg, dtype: torch.dtype) -> tuple:
    """Build the canonical Dion batch key used by local/distributed runtimes."""
    if len(shape) == 2:
        resolved_shape = (int(shape[0]), int(shape[1]))
    else:
        resolved_shape = tuple(int(dim) for dim in shape)
    return (
        resolved_shape,
        bool(cfg.has_fs_axis),
        bool(cfg.has_tp_axis),
        bool(cfg.is_transposed),
        bool(cfg.compressed_all_reduce),
        _normalize_shard_dim(cfg.inner_shard_tensor_dim, cfg.has_tp_axis),
        _normalize_shard_dim(cfg.outer_shard_tensor_dim, cfg.has_fs_axis),
        dtype,
    )


class BatchProcessor:
    """Handles batch processing for Dion optimizer to ensure mathematical equivalence."""

    def __init__(self, max_batch_size: int | None = None):
        if max_batch_size is not None and int(max_batch_size) <= 0:
            raise ValueError(f"Invalid max_batch_size={max_batch_size}")
        self.max_batch_size = max_batch_size

    def group_items(self, items: Sequence, batch_keys: Sequence[tuple]) -> list[tuple[tuple, list]]:
        """Group items by canonical batch key while preserving first-seen key order."""
        if len(items) != len(batch_keys):
            raise RuntimeError(
                f"[Dion] item/key length mismatch for batching: items={len(items)} keys={len(batch_keys)}"
            )
        grouped = {}
        for item, batch_key in zip(items, batch_keys):
            grouped.setdefault(batch_key, []).append(item)
        return list(grouped.items())

    def create_batches(
        self,
        items: Sequence,
        batch_keys: Sequence[tuple],
        batch_size: int | None = None,
    ) -> Iterator[list]:
        """Yield canonical batches, preserving first-seen group order."""
        limit = self.max_batch_size if batch_size is None else batch_size
        if limit is not None and int(limit) <= 0:
            raise ValueError(f"Invalid batch_size={limit}")
        for _, group_items in self.group_items(items, batch_keys):
            effective_limit = len(group_items) if limit is None else int(limit)
            for i in range(0, len(group_items), effective_limit):
                batch = group_items[i : i + effective_limit]
                if batch:
                    yield batch


def pad_batch(batch: List[Tensor], batch_size: int) -> List[Tensor]:
    """Pad with inert zero tensors so partial distributed batches remain numerically stable."""
    assert len(batch) > 0
    assert len(batch) <= batch_size
    while len(batch) < batch_size:
        batch.append(torch.zeros_like(batch[0]))
    return batch
