"""Dion optimizer type definitions."""

from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import torch


@dataclass
class DionMixedPrecisionConfig:
    """Configuration for mixed precision in Dion optimizer."""

    momentum_dtype: Optional[torch.dtype] = None
    q_dtype: Optional[torch.dtype] = None
    variance_dtype: Optional[torch.dtype] = None


@dataclass
class DionParamConfig:
    """Per-parameter configuration for Dion optimizer."""

    fs_shard_dim: Optional[int] = None
    tp_shard_dim: Optional[int] = None
    has_fs_shard: bool = False
    use_fs_shard: bool = False
    has_tp_shard: bool = False
    use_tp_shard: bool = False
    is_transposed: bool = False
    compressed_all_reduce: bool = False


@dataclass
class ScalarStepParam:
    """One scalar/non-Dion step item routed by the distributed adapter."""

    param: torch.Tensor | None = None
    grad: torch.Tensor | None = None
    optimizer_state: dict | None = None
    optim_group: dict | None = None


@dataclass
class DionStepParam:
    """One Dion step item routed by the distributed adapter."""

    param: torch.Tensor | None = None
    grad: torch.Tensor | None = None
    optimizer_state: dict | None = None
    optim_group: dict | None = None
    config: Optional[DionParamConfig] = None
    dist_meta: Any = None


@dataclass
class DionBatchGroup:
    """One grouped Dion batch carrier before fixed-size batch assembly."""

    params: list[torch.Tensor] | None = None
    grads: list[torch.Tensor] | None = None
    optimizer_states: list[dict] | None = None
    optim_groups: list[dict] | None = None
    configs: list[DionParamConfig] | None = None
    dist_metas: list[Any] | None = None


@dataclass
class DionDistMeta:
    """Minimal distributed metadata consumed by the Dion algorithm."""

    shape: Tuple[int, ...] | None = None
    global_shape: Tuple[int, int] | None = None
    tp_shard_dim: int = -1
    fs_shard_dim: int = -1
    rank_fraction: float = 0.25
    is_transposed: bool = False
    param_uid: Tuple | None = None
    is_dion_param: bool = False
    param_name: str = ""
    outer_shard_group: Optional[torch.distributed.ProcessGroup] = None
    fs_world_size: int = 1
    fs_rank: int = -1
    tp_group: Optional[torch.distributed.ProcessGroup] = None
    tp_world_size: int = 1
    tp_rank: int = -1
    per_expert_global_shape: Optional[Tuple[int, int]] = None
    param_config: Optional[DionParamConfig] = None


@dataclass
class DionBatchRoute:
    """Adapter-published batch schedule contract for one canonical batch key."""

    sync_groups: Tuple[torch.distributed.ProcessGroup, ...] = ()
    kernel_kind: str = "ddp"
    replicate_group: Optional[torch.distributed.ProcessGroup] = None
    replicate_subset_ranks: Optional[Tuple[int, ...]] = None
    ortho_group: Optional[torch.distributed.ProcessGroup] = None
    q_norm_group: Optional[torch.distributed.ProcessGroup] = None
    compressed_replicate_group: Optional[torch.distributed.ProcessGroup] = None
    compressed_replicate_ranks: Optional[Tuple[int, ...]] = None
    batch_world_size: int = 1


@dataclass
class DionAxisCollective:
    """One grouped TP/FS collective or writeback route over one shared axis."""

    indices: Tuple[int, ...] = ()
    process_group: Optional[torch.distributed.ProcessGroup] = None
    world_size: int = 1
    rank: int = 0


@dataclass
class DionBatchCollectives:
    """Adapter-published TP/FS collectives for one concrete Dion batch."""

    tp_q_gathers: Tuple[DionAxisCollective, ...] = ()
    fs_p_collectives: Tuple[DionAxisCollective, ...] = ()
    tp_r_collectives: Tuple[DionAxisCollective, ...] = ()
    tp_q_reshards: Tuple[DionAxisCollective, ...] = ()
    fs_orthogonalize: Optional[DionAxisCollective] = None
    orthogonalize_mesh: Any = None


@dataclass
class DionBatch:
    """Adapter-published ready-to-execute batch for one Dion kernel call."""

    batch_key: tuple = ()
    params: list[torch.Tensor] | None = None
    grads: list[torch.Tensor] | None = None
    momentums: list[torch.Tensor] | None = None
    q_tensors: list[torch.Tensor] | None = None
    configs: list[DionParamConfig] | None = None
    dist_metas: list[Any] | None = None
    optim_groups: list[dict] | None = None
    optimizer_states: list[dict] | None = None
    param_shapes: Tuple[Tuple[int, int], ...] = ()
    real_batch_size: int = 0
    global_param_offset: int = 0
    batch_route: Optional["DionBatchRoute"] = None
    batch_collectives: Optional[DionBatchCollectives] = None


@dataclass
class DionQLayout:
    """Adapter-published Q-state layout contract for one logical Dion parameter."""

    q_global_shape: Tuple[int, int] | None = None
    q_local_shape: Tuple[int, int] | None = None
    q_gathered_shape: Tuple[int, int] | None = None
    q_base_global: int = 0
    q_base_local: int = 0
    r_global: int = 0
    r_local: int = 0
    q_local_layout: Tuple[str, ...] = ()
    q_gathered_layout: Tuple[str, ...] = ()


@dataclass
class DionQInit:
    """Adapter-published state-init contract for one logical Dion parameter."""

    tp_world_size: int = 1
    tp_rank: int = 0
    q_needs_tp_unshard: bool = False
    q_init_seed: Optional[int] = None
    q_layout: Optional[DionQLayout] = None
    broadcast_q_fn: Optional[Callable[[torch.Tensor], None]] = None
