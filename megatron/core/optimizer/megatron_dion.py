"""
Dion (Distributed OrthoNormalized Updates) optimizer implementation for Megatron-LM.

NOTE: This module is a backward-compatibility shim.
The implementation has been refactored into the `dion/` package.
Please import directly from `megatron.core.optimizer.dion` for new code.

Features:
- partition_dim-aware FS sharding: Different FS split dimensions based on TP partition_dim
  - partition_dim=0 (ColumnParallelLinear): TP splits dim 0 (rows) -> FS splits dim 1 (cols) -> (m/tp, n/fs)
  - partition_dim=1 (RowParallelLinear): TP splits dim 1 (cols) -> FS splits dim 0 (rows) -> (m/fs, n/tp)
- Orthogonal TP x FS sharding for optimal memory and communication efficiency
- FS shard ranges computed per partition_dim for correct parameter decomposition
- Unified handling across all parameter types (linear layers with partition_dim 0 or 1)

This module provides an implementation of the Dion optimizer with:
- Low-rank approximation for memory-efficient training
- 2D parallelism support (RP x FS) with orthogonal tensor parallelism (TP)
- partition_dim-aware parameter sharding for heterogeneous layer configurations
- Asynchronous communication patterns for better performance
- Randomized Cholesky QR for numerical stability
- Mixed precision support for memory optimization
- Compressed communication with reduce-scatter/all-gather patterns

The implementation follows the reference from microsoft/dion
while being compatible with Megatron-LM's DistributedOptimizer infrastructure.

Key terminology:
- RP (Replicate Process): Ranks with same FS shard across replicas (gradient averaging)
- FS (Fully Shard): Different param shards within same replica (reduce-scatter/all-gather)
- TP (Tensor Parallel): Sharding with partition_dim indicating which dimension is split
  - partition_dim=0: Row-wise sharding (ColumnParallelLinear)
  - partition_dim=1: Column-wise sharding (RowParallelLinear)
- DP (Data Parallel): Full parallelism = RP x FS
"""

# Backward-compatibility shim: Re-export all public symbols from dion/ package
from .dion import (
    # Main optimizer
    MegatronDion,
    # Configuration types
    DionMixedPrecisionConfig,
    DionParamConfig,
    MegatronDionDistMeta,
    # Async runtime
    AsyncTask,
    AsyncRuntime,
    # Batch processing
    BatchProcessor,
    pad_batch,
)

__all__ = [
    "MegatronDion",
    "DionMixedPrecisionConfig",
    "DionParamConfig",
    "MegatronDionDistMeta",
    "AsyncTask",
    "AsyncRuntime",
    "BatchProcessor",
    "pad_batch",
]
