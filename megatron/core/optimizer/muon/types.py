"""Muon optimizer type definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Tuple

import torch

from ..matrix.types import MatrixDistMeta, MatrixStepParam


@dataclass
class MuonMixedPrecisionConfig:
    """Optional state dtype overrides for Muon."""

    momentum_dtype: Optional[torch.dtype | str] = None
    scalar_momentum_dtype: Optional[torch.dtype | str] = None
    scalar_variance_dtype: Optional[torch.dtype | str] = None


@dataclass
class MuonParamConfig:
    """Per-parameter Muon topology and math configuration."""

    fs_shard_dim: Optional[int] = None
    tp_shard_dim: Optional[int] = None
    has_fs_shard: bool = False
    use_fs_shard: bool = False
    has_tp_shard: bool = False
    use_tp_shard: bool = False
    is_transposed: bool = False
    momentum_beta: float = 0.95
    use_nesterov: bool = False
    ns_backend: str = "standard"
    coefficient_type: str = "quintic"
    num_ns_steps: int = 5
    ns_epsilon: float = 1e-7
    gram_restart_iterations: Tuple[int, ...] = (2,)
    gram_kernel_policy: str = "torch"
    gram_dtype: Optional[torch.dtype | str] = None
    scale_mode: str = "spectral"
    extra_scale_factor: float = 1.0
    scale_factor: float = 1.0
    fs_mode: str = "blockwise"
    tp_mode: str = "blockwise"
    split_qkv: bool = False
    split_qkvg: bool = False
    split_linear: bool = False

    def __post_init__(self) -> None:
        if self.ns_backend not in ("standard", "gram"):
            raise ValueError(f"Invalid Muon ns_backend: {self.ns_backend!r}")
        if self.coefficient_type not in ("simple", "quintic", "polar_express", "aol", "custom"):
            raise ValueError(f"Invalid Muon coefficient_type: {self.coefficient_type!r}")
        if int(self.num_ns_steps) < 1:
            raise ValueError(f"num_ns_steps must be at least 1, got {self.num_ns_steps}")
        if self.scale_mode not in ("spectral", "unit_rms_norm", "shape_scaling"):
            raise ValueError(f"Invalid Muon scale_mode: {self.scale_mode!r}")
        if self.fs_mode not in ("blockwise", "distributed", "duplicated_debug"):
            raise ValueError(f"Invalid Muon fs_mode: {self.fs_mode!r}")
        if self.tp_mode not in ("blockwise", "distributed", "duplicated", "duplicated_debug"):
            raise ValueError(f"Invalid Muon tp_mode: {self.tp_mode!r}")
        if self.gram_kernel_policy not in (
            "torch",
            "auto",
            "dao",
            "quack",
            "compile",
            "disabled",
            "eager",
        ):
            raise ValueError(
                "Invalid Gram Newton-Schulz kernel policy, "
                f"got {self.gram_kernel_policy!r}"
            )
        self.gram_restart_iterations = tuple(int(i) for i in self.gram_restart_iterations)


@dataclass
class MuonStepParam(MatrixStepParam):
    """One Muon matrix step item routed by the distributed adapter."""

    config: Optional[MuonParamConfig] = None
    dist_meta: Any = None


@dataclass
class MuonBatchEntry:
    """One typed per-parameter entry used to assemble a Muon batch."""

    param: torch.Tensor | None = None
    grad: torch.Tensor | None = None
    optimizer_state: dict | None = None
    optim_group: dict | None = None
    config: Optional[MuonParamConfig] = None
    dist_meta: Any = None
    momentum: torch.Tensor | None = None
    param_shape: Tuple[int, int] = ()
    global_shape: Tuple[int, int] = ()
    commit_update: Optional[Callable[[torch.Tensor, torch.Tensor], None]] = None


@dataclass
class MuonBatchGroup:
    """One grouped Muon batch before concrete execution."""

    entries: list[MuonBatchEntry] | None = None
    sync_groups: Tuple[torch.distributed.ProcessGroup, ...] = ()
    kernel_kind: str = "local"
    fs_group: Optional[torch.distributed.ProcessGroup] = None
    tp_group: Optional[torch.distributed.ProcessGroup] = None
    batch_world_size: int = 1


@dataclass
class MuonDistMeta(MatrixDistMeta):
    """Muon-specific metadata layered on the matrix distributed contract."""

    is_muon_param: bool = False
    param_config: Optional[MuonParamConfig] = None
    is_qkv_child: bool = False
    qkv_child_kind: str = ""
    qkv_split_shapes: Optional[Tuple[int, int, int]] = None
    is_qkvg_child: bool = False
    qkvg_child_kind: str = ""
    qkvg_split_shapes: Optional[Tuple[int, int, int, int]] = None
    is_linear_child: bool = False
    linear_child_kind: str = ""
    linear_split_rows: Optional[Tuple[int, int]] = None
    linear_partition_stride: int = 1

    def __post_init__(self) -> None:
        if self.is_muon_param and not self.is_matrix_param:
            self.is_matrix_param = True


@dataclass
class MuonBatch:
    """Ready-to-execute Muon batch."""

    batch_key: tuple = ()
    entries: Tuple[MuonBatchEntry, ...] = ()
    real_batch_size: int = 0
    batch_cache_key: int = 0
    fs_mode: str = "blockwise"
    tp_mode: str = "blockwise"
    ns_backend: str = "standard"
    batch_group: Optional[MuonBatchGroup] = None
    _params: Tuple[torch.Tensor | None, ...] = field(init=False, repr=False)
    _grads: Tuple[torch.Tensor | None, ...] = field(init=False, repr=False)
    _momentums: Tuple[torch.Tensor | None, ...] = field(init=False, repr=False)
    _configs: Tuple[MuonParamConfig | None, ...] = field(init=False, repr=False)
    _dist_metas: Tuple[Any, ...] = field(init=False, repr=False)
    _optim_groups: Tuple[dict | None, ...] = field(init=False, repr=False)
    _optimizer_states: Tuple[dict | None, ...] = field(init=False, repr=False)
    _param_shapes: Tuple[Tuple[int, int], ...] = field(init=False, repr=False)
    _global_shapes: Tuple[Tuple[int, int], ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.entries = tuple(self.entries)
        if self.real_batch_size == 0 and self.entries:
            self.real_batch_size = len(self.entries)
        self._params = tuple(entry.param for entry in self.entries)
        self._grads = tuple(entry.grad for entry in self.entries)
        self._momentums = tuple(entry.momentum for entry in self.entries)
        self._configs = tuple(entry.config for entry in self.entries)
        self._dist_metas = tuple(entry.dist_meta for entry in self.entries)
        self._optim_groups = tuple(entry.optim_group for entry in self.entries)
        self._optimizer_states = tuple(entry.optimizer_state for entry in self.entries)
        self._param_shapes = tuple(entry.param_shape for entry in self.entries)
        self._global_shapes = tuple(entry.global_shape for entry in self.entries)

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
    def configs(self) -> Tuple[MuonParamConfig | None, ...]:
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

    @property
    def global_shapes(self) -> Tuple[Tuple[int, int], ...]:
        return self._global_shapes


__all__ = [
    "MuonBatch",
    "MuonBatchEntry",
    "MuonBatchGroup",
    "MuonDistMeta",
    "MuonMixedPrecisionConfig",
    "MuonParamConfig",
    "MuonStepParam",
]
