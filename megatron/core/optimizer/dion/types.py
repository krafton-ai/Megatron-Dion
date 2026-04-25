"""Dion optimizer type definitions."""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Tuple

import torch


@dataclass
class DionMixedPrecisionConfig:
    """Configuration for mixed precision in Dion optimizer.

    None uses the same dtype as the optimizer parameter.
    """

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
    commit_update: Optional[Callable[[torch.Tensor, torch.Tensor], None]] = None


@dataclass
class DionBatchEntry:
    """One typed per-parameter entry used to assemble a Dion batch."""

    param: torch.Tensor | None = None
    grad: torch.Tensor | None = None
    optimizer_state: dict | None = None
    optim_group: dict | None = None
    config: Optional[DionParamConfig] = None
    dist_meta: Any = None
    momentum: torch.Tensor | None = None
    q_tensor: torch.Tensor | None = None
    param_shape: Tuple[int, int] = ()
    commit_update: Optional[Callable[[torch.Tensor, torch.Tensor], None]] = None


@dataclass
class DionBatchGroup:
    """One grouped Dion batch before fixed-size batch assembly."""

    params: list[torch.Tensor] | None = None
    grads: list[torch.Tensor] | None = None
    optimizer_states: list[dict] | None = None
    optim_groups: list[dict] | None = None
    configs: list[DionParamConfig] | None = None
    dist_metas: list[Any] | None = None
    commit_updates: list[Callable[[torch.Tensor, torch.Tensor], None] | None] | None = None
    sync_groups: Tuple[torch.distributed.ProcessGroup, ...] = ()
    kernel_kind: str = "ddp"
    replicate_group: Optional[torch.distributed.ProcessGroup] = None
    ortho_group: Optional[torch.distributed.ProcessGroup] = None
    q_norm_group: Optional[torch.distributed.ProcessGroup] = None
    compressed_replicate_group: Optional[torch.distributed.ProcessGroup] = None
    batch_world_size: int = 1


@dataclass
class DionDistMeta:
    """Minimal distributed metadata consumed by the Dion algorithm."""

    shape: Tuple[int, ...] | None = None
    global_shape: Tuple[int, int] | None = None
    fs_start_idx: int = -1
    fs_end_idx: int = -1
    tp_shard_dim: int = -1
    fs_shard_dim: int = -1
    rank_fraction: float = 0.25
    is_transposed: bool = False
    param_uid: Tuple | None = None
    is_dion_param: bool = False
    param_name: str = ""
    fs_group: Optional[torch.distributed.ProcessGroup] = None
    fs_world_size: int = 1
    fs_rank: int = -1
    tp_group: Optional[torch.distributed.ProcessGroup] = None
    tp_world_size: int = 1
    tp_rank: int = -1
    per_expert_global_shape: Optional[Tuple[int, int]] = None
    local_shape: Optional[Tuple[int, int]] = None
    expert_axis: int = -1
    num_local_experts: int = 1
    local_expert_index: int = -1
    param_config: Optional[DionParamConfig] = None
    parent_param_uid: Tuple | None = None
    parent_param_name: str = ""
    is_qkv_child: bool = False
    qkv_child_kind: str = ""
    qkv_split_shapes: Optional[Tuple[int, int, int]] = None
    linear_split_rows: Optional[Tuple[int, int]] = None
    is_linear_child: bool = False
    linear_child_kind: str = ""


@dataclass
class DionAxisCollective:
    """One grouped TP/FS collective or reshard collective over one shared axis."""

    indices: Tuple[int, ...] = ()
    process_group: Optional[torch.distributed.ProcessGroup] = None
    world_size: int = 1
    rank: int = 0


@dataclass
class DionBatchCollectives:
    """TP/FS collective state for one concrete Dion batch."""

    tp_q_gathers: Tuple[DionAxisCollective, ...] = ()
    fs_p_collectives: Tuple[DionAxisCollective, ...] = ()
    tp_r_collectives: Tuple[DionAxisCollective, ...] = ()
    tp_q_reshards: Tuple[DionAxisCollective, ...] = ()
    fs_collective: Optional[DionAxisCollective] = None


@dataclass
class DionBatch:
    """Ready-to-execute batch for one Dion kernel call."""

    batch_key: tuple = ()
    entries: Tuple[DionBatchEntry, ...] = ()
    real_batch_size: int = 0
    global_param_offset: int = 0
    batch_group: Optional["DionBatchGroup"] = None
    batch_collectives: Optional[DionBatchCollectives] = None
    _params: Tuple[torch.Tensor | None, ...] = field(init=False, repr=False)
    _grads: Tuple[torch.Tensor | None, ...] = field(init=False, repr=False)
    _momentums: Tuple[torch.Tensor | None, ...] = field(init=False, repr=False)
    _q_tensors: Tuple[torch.Tensor | None, ...] = field(init=False, repr=False)
    _configs: Tuple[DionParamConfig | None, ...] = field(init=False, repr=False)
    _dist_metas: Tuple[Any, ...] = field(init=False, repr=False)
    _optim_groups: Tuple[dict | None, ...] = field(init=False, repr=False)
    _optimizer_states: Tuple[dict | None, ...] = field(init=False, repr=False)
    _param_shapes: Tuple[Tuple[int, int], ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._params = tuple(entry.param for entry in self.entries)
        self._grads = tuple(entry.grad for entry in self.entries)
        self._momentums = tuple(entry.momentum for entry in self.entries)
        self._q_tensors = tuple(entry.q_tensor for entry in self.entries)
        self._configs = tuple(entry.config for entry in self.entries)
        self._dist_metas = tuple(entry.dist_meta for entry in self.entries)
        self._optim_groups = tuple(entry.optim_group for entry in self.entries)
        self._optimizer_states = tuple(entry.optimizer_state for entry in self.entries)
        self._param_shapes = tuple(entry.param_shape for entry in self.entries)

    @property
    def params(self) -> Tuple[torch.Tensor | None, ...]:
        return self._params

    @property
    def grads(self) -> Tuple[torch.Tensor | None, ...]:
        return self._grads

    @property
    def momentums(self) -> Tuple[torch.Tensor | None, ...]:
        return self._momentums

    @property
    def q_tensors(self) -> Tuple[torch.Tensor | None, ...]:
        return self._q_tensors

    @property
    def configs(self) -> Tuple[DionParamConfig | None, ...]:
        return self._configs

    @property
    def dist_metas(self) -> Tuple[Any, ...]:
        return self._dist_metas

    @property
    def optim_groups(self) -> Tuple[dict | None, ...]:
        return self._optim_groups

    @property
    def optimizer_states(self) -> Tuple[dict | None, ...]:
        return self._optimizer_states

    @property
    def param_shapes(self) -> Tuple[Tuple[int, int], ...]:
        return self._param_shapes

@dataclass
class DionQLayout:
    """Q-state layout contract for one Dion parameter."""

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
    """State-init contract for one Dion parameter."""

    tp_world_size: int = 1
    tp_rank: int = 0
    use_q_unshard: bool = False
    q_seed: Optional[int] = None
    q_layout: Optional[DionQLayout] = None
    broadcast_q: Optional[Callable[[torch.Tensor], None]] = None
