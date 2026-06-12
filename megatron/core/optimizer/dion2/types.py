"""Dion2 optimizer type definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Tuple

import torch

from ..matrix.types import MatrixDistMeta, MatrixStepParam


@dataclass
class Dion2MixedPrecisionConfig:
    """Optional state dtype overrides for Dion2."""

    momentum_dtype: Optional[torch.dtype | str] = None
    scalar_momentum_dtype: Optional[torch.dtype | str] = None
    scalar_variance_dtype: Optional[torch.dtype | str] = None


@dataclass
class Dion2ParamConfig:
    """Per-parameter Dion2 topology and math configuration."""

    fs_shard_dim: Optional[int] = None
    tp_shard_dim: Optional[int] = None
    has_fs_shard: bool = False
    use_fs_shard: bool = False
    has_tp_shard: bool = False
    use_tp_shard: bool = False
    is_transposed: bool = False
    fraction: float = 0.25
    ef_decay: float = 0.95
    adjust_lr: Optional[str] = "spectral_norm"
    select_dim: str | int = "auto"
    selection_policy: str = "local_shard"
    ns_backend: str = "standard"
    coefficient_type: str = "dion2_polar_express"
    num_ns_steps: int = 5
    ns_epsilon: float = 1e-7
    gram_restart_iterations: Tuple[int, ...] = (2,)
    gram_kernel_policy: str = "torch"
    gram_dtype: Optional[torch.dtype | str] = None
    fp32_matmul_prec: str = "medium"
    fs_mode: str = "distributed"
    tp_mode: str = "distributed"
    split_parameters: bool = True

    def __post_init__(self) -> None:
        if not (0.0 < float(self.fraction) <= 1.0):
            raise ValueError(f"Dion2 fraction must be in (0, 1], got {self.fraction}")
        if float(self.ef_decay) < 0.0:
            raise ValueError(f"Dion2 ef_decay must be non-negative, got {self.ef_decay}")
        if self.adjust_lr not in ("spectral_norm", "rms_norm", None):
            raise ValueError(f"Invalid Dion2 adjust_lr: {self.adjust_lr!r}")
        if self.select_dim not in ("auto", "row", "rows", "col", "cols", "column", "columns", 0, 1, -2, -1):
            raise ValueError(f"Invalid Dion2 select_dim: {self.select_dim!r}")
        if self.selection_policy not in ("local_shard",):
            raise ValueError(f"Invalid Dion2 selection_policy: {self.selection_policy!r}")
        if self.ns_backend not in ("standard", "gram"):
            raise ValueError(f"Invalid Dion2 ns_backend: {self.ns_backend!r}")
        if self.coefficient_type not in (
            "simple",
            "quintic",
            "polar_express",
            "dion2_polar_express",
            "aol",
            "custom",
        ):
            raise ValueError(f"Invalid Dion2 coefficient_type: {self.coefficient_type!r}")
        if int(self.num_ns_steps) < 1:
            raise ValueError(f"Dion2 num_ns_steps must be at least 1, got {self.num_ns_steps}")
        if self.fs_mode == "duplicated_debug":
            self.fs_mode = "duplicated"
        if self.tp_mode == "duplicated_debug":
            self.tp_mode = "duplicated"
        if self.fs_mode not in ("blockwise", "duplicated", "distributed"):
            raise ValueError(f"Invalid Dion2 fs_mode: {self.fs_mode!r}")
        if self.tp_mode not in ("blockwise", "duplicated", "distributed"):
            raise ValueError(f"Invalid Dion2 tp_mode: {self.tp_mode!r}")
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
                "Invalid Dion2 Gram Newton-Schulz kernel policy, "
                f"got {self.gram_kernel_policy!r}"
            )
        self.gram_restart_iterations = tuple(int(i) for i in self.gram_restart_iterations)


@dataclass
class Dion2StepParam(MatrixStepParam):
    """One Dion2 matrix step item routed by the distributed adapter."""

    config: Optional[Dion2ParamConfig] = None
    dist_meta: Any = None


@dataclass
class Dion2BatchEntry:
    """One typed per-parameter entry used to assemble a Dion2 batch."""

    param: torch.Tensor | None = None
    grad: torch.Tensor | None = None
    optimizer_state: dict | None = None
    optim_group: dict | None = None
    config: Optional[Dion2ParamConfig] = None
    dist_meta: Any = None
    momentum: torch.Tensor | None = None
    param_shape: Tuple[int, int] = ()
    global_shape: Tuple[int, int] = ()
    commit_update: Optional[Callable[[torch.Tensor, torch.Tensor], None]] = None


@dataclass
class Dion2DistMeta(MatrixDistMeta):
    """Dion2-specific metadata layered on the Matrix distributed invariant."""

    is_dion2_param: bool = False
    param_config: Optional[Dion2ParamConfig] = None
    is_qkv_child: bool = False
    qkv_child_kind: str = ""
    qkv_split_shapes: Optional[Tuple[int, int, int]] = None
    qkv_split_axis: int = 0
    is_qkvg_child: bool = False
    qkvg_child_kind: str = ""
    qkvg_split_shapes: Optional[Tuple[int, int, int, int]] = None
    qkvg_split_axis: int = 0
    is_gdn_child: bool = False
    gdn_child_kind: str = ""
    gdn_split_shapes: Optional[Tuple[int, int, int, int, int, int]] = None
    gdn_split_axis: int = 0
    is_linear_child: bool = False
    linear_child_kind: str = ""
    linear_split_rows: Optional[Tuple[int, int]] = None
    linear_child_kinds: Tuple[str, str] = ("gate", "up")
    linear_split_axis: int = 0
    linear_partition_stride: int = 1

    def __post_init__(self) -> None:
        if self.is_dion2_param and not self.is_matrix_param:
            self.is_matrix_param = True


@dataclass
class Dion2Batch:
    """Ready-to-execute Dion2 batch."""

    batch_key: tuple = ()
    entries: Tuple[Dion2BatchEntry, ...] = ()
    real_batch_size: int = 0
    _params: Tuple[torch.Tensor | None, ...] = field(init=False, repr=False)
    _grads: Tuple[torch.Tensor | None, ...] = field(init=False, repr=False)
    _momentums: Tuple[torch.Tensor | None, ...] = field(init=False, repr=False)
    _configs: Tuple[Dion2ParamConfig | None, ...] = field(init=False, repr=False)
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
    def configs(self) -> Tuple[Dion2ParamConfig | None, ...]:
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
    "Dion2Batch",
    "Dion2BatchEntry",
    "Dion2DistMeta",
    "Dion2MixedPrecisionConfig",
    "Dion2ParamConfig",
    "Dion2StepParam",
]
