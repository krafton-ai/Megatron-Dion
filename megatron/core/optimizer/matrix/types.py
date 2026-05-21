"""Backend-neutral data contracts for matrix-aware optimizers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, FrozenSet, Optional, Tuple

import torch


@dataclass
class ScalarStepParam:
    """One scalar-update step item routed by the distributed adapter."""

    param: torch.Tensor | None = None
    grad: torch.Tensor | None = None
    optimizer_state: dict | None = None
    optim_group: dict | None = None


@dataclass
class MatrixStepParam:
    """One matrix-update step item routed by the distributed adapter."""

    param: torch.Tensor | None = None
    grad: torch.Tensor | None = None
    optimizer_state: dict | None = None
    optim_group: dict | None = None
    config: Any = None
    dist_meta: Any = None
    commit_update: Optional[Callable[[torch.Tensor, torch.Tensor], None]] = None


@dataclass
class MatrixDistMeta:
    """Minimal distributed metadata consumed by matrix-aware adapters."""

    shape: Tuple[int, ...] | None = None
    global_shape: Tuple[int, int] | None = None
    fs_start_idx: int = -1
    fs_end_idx: int = -1
    tp_shard_dim: int = -1
    fs_shard_dim: int = -1
    is_transposed: bool = False
    param_uid: Tuple | None = None
    is_matrix_param: bool = False
    param_name: str = ""
    fs_group: Optional[torch.distributed.ProcessGroup] = None
    fs_world_size: int = 1
    fs_rank: int = -1
    tp_group: Optional[torch.distributed.ProcessGroup] = None
    tp_world_size: int = 1
    tp_rank: int = -1
    per_expert_global_shape: Optional[Tuple[int, int]] = None
    local_shape: Optional[Tuple[int, int]] = None
    tensor_row_shard_sizes: Optional[Tuple[int, ...]] = None
    row_shard_start_idx: int = -1
    row_shard_end_idx: int = -1
    row_shard_sizes: Optional[Tuple[int, ...]] = None
    expert_axis: int = -1
    num_local_experts: int = 1
    local_expert_index: int = -1
    parent_param_uid: Tuple | None = None
    parent_param_name: str = ""


@dataclass(frozen=True)
class MatrixShardLayout:
    """Minimal shard layout for one 2D model parameter."""

    local_shape: Tuple[int, int]
    global_shape: Tuple[int, int]
    fs_shard_dim: int
    start_idx: int
    end_idx: int
    per_expert_global_shape: Optional[Tuple[int, int]] = None

    @property
    def local_numel(self) -> int:
        return int(self.local_shape[0]) * int(self.local_shape[1])


@dataclass(frozen=True)
class MatrixShardEntry:
    """Bucket-local shard layout for one parameter."""

    param: torch.nn.Parameter
    shard_layout: MatrixShardLayout
    size_per_rank: int
    shard_capacity: int
    shard_offset: int
    canonical_bucket_start: int
    canonical_bucket_end: int
    canonical_rank_flat_segments: Tuple[Tuple[Tuple[int, int], ...], ...]
    grad_rank_flat_segments: Tuple[Tuple[Tuple[int, int], ...], ...]
    rank_split_ranges: Tuple[Tuple[int, int], ...]

    @property
    def local_shape(self) -> Tuple[int, int]:
        return self.shard_layout.local_shape

    @property
    def global_shape(self) -> Tuple[int, int]:
        return self.shard_layout.global_shape

    @property
    def fs_shard_dim(self) -> int:
        return int(self.shard_layout.fs_shard_dim)

    @property
    def start_idx(self) -> int:
        return int(self.shard_layout.start_idx)

    @property
    def end_idx(self) -> int:
        return int(self.shard_layout.end_idx)

    @property
    def local_numel(self) -> int:
        return int(self.shard_layout.local_numel)


@dataclass(frozen=True)
class MatrixBucketLayout:
    """Bucket-local transport layout under stock distributed-optimizer control."""

    entries: Tuple[MatrixShardEntry, ...]
    shard_size: int
    gathered_numel: int
    grad_gathered_numel: int
    param_ids: FrozenSet[int]

    @property
    def has_params(self) -> bool:
        return bool(self.entries)

    @property
    def entry_count(self) -> int:
        return len(self.entries)

    @property
    def max_shard_capacity(self) -> int:
        return max((int(entry.shard_capacity) for entry in self.entries), default=0)


@dataclass(frozen=True)
class MatrixStandardGatherRoute:
    """Bucket all-gather layout for scalar params inside a mixed bucket."""

    group_size: int
    group_rank: int
    standard_shard_size: int
    standard_numel: int
    rank_segments: Tuple[Tuple[Tuple[int, int, int], ...], ...]


__all__ = [
    "MatrixBucketLayout",
    "MatrixDistMeta",
    "MatrixShardEntry",
    "MatrixShardLayout",
    "MatrixStandardGatherRoute",
    "MatrixStepParam",
    "ScalarStepParam",
]
