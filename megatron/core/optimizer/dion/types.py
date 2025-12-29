"""
Dion optimizer type definitions and dataclasses.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class DionMixedPrecisionConfig:
    """Configuration for mixed precision in Dion optimizer."""
    momentum_dtype: Optional[torch.dtype] = None  # Momentum state dtype
    Q_dtype: Optional[torch.dtype] = None  # Q matrix dtype
    variance_dtype: Optional[torch.dtype] = None  # For Adam variance (unused in pure Dion)


@dataclass
class DionParamConfig:
    """Per-parameter configuration for Dion optimizer."""
    outer_shard_tensor_dim: Optional[int] = None  # FS axis dimension (row sharding)
    inner_shard_tensor_dim: Optional[int] = None  # TP axis dimension (column sharding)
    outer_shard_mesh_dim: Optional[int] = None
    inner_shard_mesh_dim: Optional[int] = None
    has_fs_axis: bool = False
    has_tp_axis: bool = False
    is_transposed: bool = False
    compressed_all_reduce: bool = False


@dataclass
class MegatronDionDistMeta:
    """Metadata for distributed Dion optimizer (2D parallelism: RP × FS)."""
    buffer_idx: int = 0
    bucket_idx: int = 0
    shape: torch.Size = None
    global_shape: torch.Size = None
    global_range: Tuple[int, int] = None
    tp_split_dim: int = -1
    fs_split_dim: int = -1  # FS sharding dimension (0=row, 1=col), -1 if no FS
    local_range: Tuple[int, int] = None
    rank_fraction: float = 0.25
    is_transposed: bool = False
    param_uid: Tuple = None  # (buffer_idx, bucket_idx, start) for unique identification
    is_dion_param: bool = False

    # Clear 2D parallelism groups
    replica_group: Optional[torch.distributed.ProcessGroup] = None  # RP group (across replicas)
    replica_group_world_size: int = 1  # Number of replicas
    replica_group_rank: int = -1  # This rank's replica ID

    shard_group: Optional[torch.distributed.ProcessGroup] = None  # FS group (within replica)
    shard_group_world_size: int = 1  # Number of shards per replica
    shard_group_rank: int = -1  # This rank's shard ID within replica

    # Stable replica group ID for deterministic batching (don't use id(group))
    replica_group_id: int = 0

    def __init__(self, buffer_idx: int = 0, bucket_idx: int = 0, shape: torch.Size = None,
                 global_range: Tuple[int, int] = None, tp_split_dim: int = -1,
                 rank_fraction: float = 0.25, global_shape: torch.Size = None,
                 param_uid: Tuple = None, is_dion_param: bool = False,
                 fs_split_dim: int = -1):
        self.buffer_idx = buffer_idx
        self.bucket_idx = bucket_idx
        self.shape = shape
        self.global_range = global_range
        self.tp_split_dim = tp_split_dim
        self.fs_split_dim = fs_split_dim
        self.rank_fraction = rank_fraction
        self.param_uid = param_uid
        self.is_dion_param = is_dion_param

        if global_shape is None:
            self.global_shape = shape if tp_split_dim == -1 else shape
        else:
            self.global_shape = global_shape

        if self.global_shape and len(self.global_shape) == 2:
            m, n = self.global_shape
            self.is_transposed = (tp_split_dim == 1) or (tp_split_dim == -1 and m < n)
        else:
            self.is_transposed = False
