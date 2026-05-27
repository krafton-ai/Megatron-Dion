"""MCore-native Muon optimizer package."""

from .algorithm import (
    MegatronMuon,
    TensorParallelMuon,
    build_muon_optimizer,
    get_megatron_muon_optimizer,
    init_muon_state,
)
from .backend import MuonBackend
from .distributed import DistributedMuonOptimizer, build_muon_distributed_optimizer
from .state import is_muon_matrix_param, mark_muon_candidates


def get_muon_param_override(config, param, param_override, name):
    """Return Muon-specific param override.

    Muon uses the standard Megatron weight-decay and LR override contract for
    now; scalar-vs-matrix routing is decided inside the optimizer.
    """
    del config, param, param_override, name
    return None

__all__ = [
    "DistributedMuonOptimizer",
    "MegatronMuon",
    "MuonBackend",
    "TensorParallelMuon",
    "build_muon_distributed_optimizer",
    "build_muon_optimizer",
    "get_megatron_muon_optimizer",
    "get_muon_param_override",
    "init_muon_state",
    "is_muon_matrix_param",
    "mark_muon_candidates",
]
