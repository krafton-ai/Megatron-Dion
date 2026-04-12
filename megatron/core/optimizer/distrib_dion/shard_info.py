"""Typed Dion metadata surfaces used by the Megatron-Core wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass(frozen=True)
class DionShardLayout:
    """Minimal Dion shard layout for one logical model parameter."""

    local_shape: Tuple[int, int]
    global_shape: Tuple[int, int]
    fs_shard_dim: int
    start_idx: int
    end_idx: int
    per_expert_global_shape: Optional[Tuple[int, int]] = None

    @property
    def local_numel(self) -> int:
        return int(self.local_shape[0]) * int(self.local_shape[1])
