"""Dion optimizer package for Megatron-LM with 2D parallelism (RP × FS) support."""

from .async_runtime import AsyncRuntime, AsyncTask
from .batching import BatchProcessor, pad_batch
from .constants import (
    DEFAULT_EPSILON,
    DEFAULT_LR,
    DEFAULT_MAX_BATCH_SIZE,
    DEFAULT_MU,
    DEFAULT_RANK_FRACTION,
    DEFAULT_RCQR_OVERSAMPLE,
    DEFAULT_WEIGHT_DECAY,
    SCALAR_OPT_ADAMW,
    SCALAR_OPT_LION,
)
from .algorithm import MegatronDion
from .types import DionMixedPrecisionConfig, DionParamConfig, MegatronDionDistMeta

__all__ = [
    # Main optimizer
    "MegatronDion",
    # Configuration types
    "DionMixedPrecisionConfig",
    "DionParamConfig",
    "MegatronDionDistMeta",
    # Async runtime
    "AsyncTask",
    "AsyncRuntime",
    # Batch processing
    "BatchProcessor",
    "pad_batch",
    # Constants
    "DEFAULT_LR",
    "DEFAULT_MU",
    "DEFAULT_WEIGHT_DECAY",
    "DEFAULT_RANK_FRACTION",
    "DEFAULT_EPSILON",
    "DEFAULT_RCQR_OVERSAMPLE",
    "DEFAULT_MAX_BATCH_SIZE",
    "SCALAR_OPT_ADAMW",
    "SCALAR_OPT_LION",
]
