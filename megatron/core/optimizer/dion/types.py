"""Dion optimizer type definitions."""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class DionMixedPrecisionConfig:
    """Configuration for mixed precision in Dion optimizer."""

    momentum_dtype: Optional[torch.dtype] = None
    Q_dtype: Optional[torch.dtype] = None
    variance_dtype: Optional[torch.dtype] = None


@dataclass
class DionParamConfig:
    """Per-parameter configuration for Dion optimizer."""

    outer_shard_tensor_dim: Optional[int] = None
    inner_shard_tensor_dim: Optional[int] = None
    outer_shard_mesh_dim: Optional[int] = None
    inner_shard_mesh_dim: Optional[int] = None
    has_fs_axis: bool = False
    active_fs_axis: bool = False
    has_tp_axis: bool = False
    is_transposed: bool = False
    compressed_all_reduce: bool = False


@dataclass
class MegatronDionDistMeta:
    """Minimal distributed metadata consumed by the Dion algorithm."""

    shape: Tuple[int, ...] | None = None
    global_shape: Tuple[int, int] | None = None
    tp_split_dim: int = -1
    fs_split_dim: int = -1
    rank_fraction: float = 0.25
    is_transposed: bool = False
    param_uid: Tuple | None = None
    is_dion_param: bool = False
    param_name: str = ""
    shard_group: Optional[torch.distributed.ProcessGroup] = None
    shard_group_world_size: int = 1
    shard_group_rank: int = -1
    per_expert_global_shape: Optional[Tuple[int, int]] = None
