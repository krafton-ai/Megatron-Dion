"""Dion2 optimizer package."""

from .algorithm import (
    MegatronDion2,
    TensorParallelDion2,
    build_dion2_optimizer,
    get_megatron_dion2_optimizer,
    init_dion2_state,
)
from .backend import Dion2Backend
from .distributed import (
    DistributedDion2Optimizer,
    build_dion2_distributed_optimizer,
    get_dion2_param_override,
)
from .state import is_dion2_matrix_param, prepare_dion2_params
from .types import Dion2DistMeta, Dion2MixedPrecisionConfig, Dion2ParamConfig

__all__ = [
    "Dion2Backend",
    "Dion2DistMeta",
    "Dion2MixedPrecisionConfig",
    "Dion2ParamConfig",
    "DistributedDion2Optimizer",
    "MegatronDion2",
    "TensorParallelDion2",
    "build_dion2_distributed_optimizer",
    "build_dion2_optimizer",
    "get_dion2_param_override",
    "get_megatron_dion2_optimizer",
    "init_dion2_state",
    "is_dion2_matrix_param",
    "prepare_dion2_params",
]
