"""
Dion optimizer constants and default hyperparameters.
"""

# Default hyperparameters
DEFAULT_LR = 0.01
DEFAULT_MU = 0.95
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_RANK_FRACTION = 1.0
DEFAULT_RANK_MULTIPLE_OF = 1
DEFAULT_EPSILON = 1e-8
DEFAULT_RCQR_OVERSAMPLE = 1.25
DEFAULT_BETAS = (0.9, 0.95)
DEFAULT_EPS = 1e-8

# Batch processing
DEFAULT_MAX_BATCH_SIZE = 8
DEFAULT_MAX_CONCURRENT_TASKS = 3

# Scalar optimizer options
SCALAR_OPT_ADAMW = "adamw"
SCALAR_OPT_LION = "lion"
