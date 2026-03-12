"""Canonical runtime metadata for Dion FS shards."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class DionShardInfo:
    """Unified shard information for a Dion parameter.

    This is stored in `DistributedOptimizerForDion._dion_shard_info` keyed by the
    model parameter.
    """

    data_shard: torch.Tensor
    opt_shard: torch.Tensor
    local_shape: Tuple[int, int]
    global_shape: Tuple[int, int]
    start_idx: int
    end_idx: int
    fs_split_dim: int
    gbuf_index: int
    bucket_index: int
    param_range_info: dict
    rs_start: int
    rs_end: int
    stock_param_start: int
    stock_param_end: int
    per_expert_global_shape: Optional[Tuple[int, int]] = None
