"""Matrix-aware distributed optimizer substrate."""

from .backend import MatrixBackend, MatrixStateSpec
from .distrib_optimizer import DistributedMatrixOptimizer
from .parameter import is_matrix_param, prepare_matrix_params
from .types import MatrixDistMeta, MatrixStepParam

__all__ = [
    "DistributedMatrixOptimizer",
    "MatrixBackend",
    "MatrixDistMeta",
    "MatrixStateSpec",
    "MatrixStepParam",
    "is_matrix_param",
    "prepare_matrix_params",
]
