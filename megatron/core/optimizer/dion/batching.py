"""
Dion optimizer batch processing utilities.
"""

from typing import List

import torch
from torch import Tensor

from .constants import DEFAULT_MAX_BATCH_SIZE


class BatchProcessor:
    """Handles batch processing for Dion optimizer to ensure mathematical equivalence."""
    def __init__(self, max_batch_size: int = DEFAULT_MAX_BATCH_SIZE):
        self.max_batch_size = max_batch_size

    def create_batches(self, params, configs):
        """Group parameters into reference-aligned batches while preserving input order."""
        # Match reference batching semantics:
        # - preserve first-seen parameter order
        # - separate by sharding configuration, local shape, and dtype
        shape_groups = {}
        for (p, group), cfg in zip(params, configs):
            shard_key = (
                cfg.has_fs_axis,
                cfg.has_tp_axis,
                cfg.is_transposed,
                cfg.outer_shard_tensor_dim,
                cfg.inner_shard_tensor_dim,
            )
            shape_key = tuple(p.shape) if len(p.shape) == 2 else (p.numel(),)
            key = (shard_key, shape_key, p.dtype)

            if key not in shape_groups:
                shape_groups[key] = []
            shape_groups[key].append((p, group, cfg))

        # Dict insertion order preserves first occurrence order, matching reference.
        for key in shape_groups.keys():
            param_group = shape_groups[key]
            for i in range(0, len(param_group), self.max_batch_size):
                batch = param_group[i:i+self.max_batch_size]
                if batch:
                    yield batch


def pad_batch(batch: List[Tensor], batch_size: int) -> List[Tensor]:
    """Match dion_reference.pad_batch(): use empty_like dummy tensors."""
    assert len(batch) > 0
    assert len(batch) <= batch_size
    while len(batch) < batch_size:
        batch.append(torch.empty_like(batch[0]))
    return batch
