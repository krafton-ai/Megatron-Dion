"""ARO optimizer type definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Tuple

import torch

from ..matrix.types import MatrixDistMeta, MatrixStepParam


@dataclass
class AroMixedPrecisionConfig:
    """Optional state dtype overrides for ARO."""

    momentum_dtype: Optional[torch.dtype | str] = None
    rotation_dtype: Optional[torch.dtype | str] = None
    scalar_momentum_dtype: Optional[torch.dtype | str] = None
    scalar_variance_dtype: Optional[torch.dtype | str] = None


@dataclass
class AroParamConfig:
    """Per-parameter ARO topology and math configuration."""

    momentum: float = 0.95
    base_optimizer: str = "sinkhorn"
    sinkhorn_iters: int = 5
    qr_backend: str = "scqr"
    scqr_eps: float = 1e-6
    update_rms_scale: float = 0.2
    scalar_optimizer: str = "adam"
    scalar_lr_scale: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95
    scalar_eps: float = 1e-8
    split_qkv: bool = False
    split_qkvg: bool = False
    split_linear: bool = False
    orientation: str = "normal"
    fs_shard_dim: int = -1
    tp_shard_dim: int = -1
    has_fs_shard: bool = False
    has_tp_shard: bool = False

    def __post_init__(self) -> None:
        if self.base_optimizer != "sinkhorn":
            raise ValueError(f"Invalid ARO base optimizer: {self.base_optimizer!r}")
        if int(self.sinkhorn_iters) < 1:
            raise ValueError(f"aro_sinkhorn_iters must be at least 1, got {self.sinkhorn_iters}")
        if self.qr_backend not in ("scqr", "qr"):
            raise ValueError(f"Invalid ARO qr backend: {self.qr_backend!r}")
        if self.scalar_optimizer not in ("adam", "adamw", "lion"):
            raise ValueError(f"Invalid ARO scalar optimizer: {self.scalar_optimizer!r}")
        if self.orientation not in ("normal", "transpose"):
            raise ValueError(f"Invalid ARO orientation: {self.orientation!r}")
        if float(self.scalar_lr_scale) < 0.0:
            raise ValueError(f"Invalid ARO scalar_lr_scale: {self.scalar_lr_scale}")
        if float(self.update_rms_scale) < 0.0:
            raise ValueError(f"Invalid ARO update_rms_scale: {self.update_rms_scale}")


@dataclass
class AroStepParam(MatrixStepParam):
    """One ARO matrix step item routed by the distributed adapter."""

    config: Optional[AroParamConfig] = None
    dist_meta: Any = None


@dataclass
class AroBatchEntry:
    """One typed per-parameter entry used to execute an ARO batch."""

    param: torch.Tensor | None = None
    grad: torch.Tensor | None = None
    optimizer_state: dict | None = None
    optim_group: dict | None = None
    config: Optional[AroParamConfig] = None
    dist_meta: Any = None
    momentum: torch.Tensor | None = None
    rotation: torch.Tensor | None = None
    param_shape: Tuple[int, int] = ()
    global_shape: Tuple[int, int] = ()
    orientation: str = "normal"
    commit_update: Optional[Callable[[torch.Tensor, torch.Tensor], None]] = None


@dataclass
class AroDistMeta(MatrixDistMeta):
    """ARO-specific metadata layered on the matrix distributed invariant."""

    is_aro_param: bool = False
    param_config: Optional[AroParamConfig] = None
    orientation: str = "normal"
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
        if self.is_aro_param and not self.is_matrix_param:
            self.is_matrix_param = True
        if self.orientation not in ("normal", "transpose"):
            raise ValueError(f"Invalid ARO orientation: {self.orientation!r}")


@dataclass
class AroBatch:
    """Ready-to-execute ARO batch."""

    batch_key: tuple = ()
    entries: Tuple[AroBatchEntry, ...] = ()
    real_batch_size: int = 0
    _params: Tuple[torch.Tensor | None, ...] = field(init=False, repr=False)
    _grads: Tuple[torch.Tensor | None, ...] = field(init=False, repr=False)
    _momentums: Tuple[torch.Tensor | None, ...] = field(init=False, repr=False)
    _rotations: Tuple[torch.Tensor | None, ...] = field(init=False, repr=False)
    _configs: Tuple[AroParamConfig | None, ...] = field(init=False, repr=False)
    _dist_metas: Tuple[Any, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.entries = tuple(self.entries)
        if self.real_batch_size == 0 and self.entries:
            self.real_batch_size = len(self.entries)
        self._params = tuple(entry.param for entry in self.entries)
        self._grads = tuple(entry.grad for entry in self.entries)
        self._momentums = tuple(entry.momentum for entry in self.entries)
        self._rotations = tuple(entry.rotation for entry in self.entries)
        self._configs = tuple(entry.config for entry in self.entries)
        self._dist_metas = tuple(entry.dist_meta for entry in self.entries)

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
    def rotations(self) -> Tuple[torch.Tensor | None, ...]:
        return self._rotations

    @property
    def configs(self) -> Tuple[AroParamConfig | None, ...]:
        return self._configs

    @property
    def dist_metas(self) -> Tuple[Any, ...]:
        return self._dist_metas


__all__ = [
    "AroBatch",
    "AroBatchEntry",
    "AroDistMeta",
    "AroMixedPrecisionConfig",
    "AroParamConfig",
    "AroStepParam",
]
