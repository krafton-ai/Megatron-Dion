"""Distributed MCore Muon optimizer integration."""

from .integration import build_muon_distributed_optimizer
from .optimizer import DistributedMuonOptimizer
from ..backend import MuonBackend
from ..state import is_muon_matrix_param, mark_muon_candidates
from ..types import MuonBatch, MuonBatchEntry, MuonDistMeta, MuonParamConfig

__all__ = [
    "DistributedMuonOptimizer",
    "MuonBackend",
    "MuonBatch",
    "MuonBatchEntry",
    "MuonDistMeta",
    "MuonParamConfig",
    "build_muon_distributed_optimizer",
    "is_muon_matrix_param",
    "mark_muon_candidates",
]
