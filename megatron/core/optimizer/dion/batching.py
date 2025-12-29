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
        """Group parameters into batches by sharding configuration and shape for efficient processing."""
        # Group by sharding configuration AND shape
        shape_groups = {}
        for (p, group), cfg in zip(params, configs):
            # Create key based on sharding config and parameter shape
            # Include shard dims to ensure all params in batch have same sharding axis
            shard_key = (
                cfg.has_fs_axis,
                cfg.has_tp_axis,
                cfg.is_transposed,
                cfg.outer_shard_tensor_dim,  # FS sharding dimension (0=rows, 1=cols, None)
                cfg.inner_shard_tensor_dim,  # TP sharding dimension (0=rows, 1=cols, None)
            )
            shape_key = tuple(p.shape) if len(p.shape) == 2 else (p.numel(),)
            key = (shard_key, shape_key)

            if key not in shape_groups:
                shape_groups[key] = []
            shape_groups[key].append((p, group, cfg))

        # Create batches from each shape group
        # Sort keys for deterministic order across all ranks
        for key in sorted(shape_groups.keys()):
            param_group = shape_groups[key]
            for i in range(0, len(param_group), self.max_batch_size):
                batch = param_group[i:i+self.max_batch_size]
                if batch:
                    yield batch


def pad_batch(batch: List[Tensor], batch_size: int) -> List[Tensor]:
    """Pad batch with zeros_like dummies to reach batch_size (empty_like causes NaN)."""
    assert len(batch) > 0
    assert len(batch) <= batch_size
    while len(batch) < batch_size:
        batch.append(torch.zeros_like(batch[0]))
    return batch
