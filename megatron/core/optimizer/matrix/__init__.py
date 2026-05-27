"""Matrix-aware distributed optimizer substrate."""

from .backend import MatrixBackend, MatrixStateSpec
from .distrib_optimizer import DistributedMatrixOptimizer
from .types import MatrixDistMeta, MatrixStepParam

__all__ = [
    "DistributedMatrixOptimizer",
    "MatrixBackend",
    "MatrixDistMeta",
    "MatrixStateSpec",
    "MatrixStepParam",
]
