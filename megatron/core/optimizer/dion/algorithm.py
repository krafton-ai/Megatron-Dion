"""Dion optimizer main class for Megatron-LM."""
from typing import Dict, Generator, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch import Tensor
from torch.optim.optimizer import Optimizer

from .async_runtime import AsyncRuntime
from .batch_runtime import (
    apply_error_feedback_to_momentum as apply_error_feedback_to_momentum_,
    apply_batch_updates as apply_batch_updates_,
    batch_dion_update_async as batch_dion_update_async_,
    orthogonalize_dense_replicate_batch_async as orthogonalize_dense_replicate_batch_async_,
    orthogonalize_p_batch as orthogonalize_p_batch_,
    sanitize_dion_batch_for_update as sanitize_dion_batch_for_update_,
    run_compressed_comm_async as run_compressed_comm_async_,
)
from .constants import (
    DEFAULT_LR,
    DEFAULT_MU,
    DEFAULT_WEIGHT_DECAY,
)
from .ortho import (
    orthogonalize,
    reshard_q_along_tp,
)
from .ortho_runtime import (
    distributed_orthogonalize as distributed_orthogonalize_,
    make_local_sketch_for_update as make_local_sketch_for_update_,
    make_seeded_sketch_for_update as make_seeded_sketch_for_update_,
    orthogonalize_local_matrix_batch as orthogonalize_local_matrix_batch_,
    orthogonalize_local_slice as orthogonalize_local_slice_,
    sketch_keys_for_update as sketch_keys_for_update_,
)
from .reference_kernels import orthogonalize_fs_only_batch_
from .runtime import (
    collapse_batch_across_replicas as collapse_batch_across_replicas_,
    collapse_batch_over_replicate_subset as collapse_batch_over_replicate_subset_,
    collapse_grads_across_replicas as collapse_grads_across_replicas_,
    enable_distributed_mode as enable_distributed_mode_,
    normalize_cols_async as normalize_cols_async_,
    iter_dist_tasks as iter_dist_tasks_,
    replicate_reduce_op as replicate_reduce_op_,
    reduce_p_across_fs_groups as reduce_p_across_fs_groups_,
    reduce_r_across_tp as reduce_r_across_tp_,
    resolve_async_task_limit as resolve_async_task_limit_,
    unshard_q_batch as unshard_q_batch_,
)
from .scalar import apply_scalar_update
from .state import (
    has_fs_shard as has_fs_shard_,
    has_tp_shard as has_tp_shard_,
    is_fs_only_config as is_fs_only_config_,
    q_needs_tp_unshard as q_needs_tp_unshard_,
)
from .types import (
    DionMixedPrecisionConfig,
    DionParamConfig,
    ScalarStepParam,
)

class MegatronDion(Optimizer):
    """
    Dion optimizer with batch processing and compressed communication.

    Implements the DIstributed OrthoNormalized updates (Dion) optimizer with low-rank approximation
    for efficient distributed training. Supports 2D parallelism model:
    - DP (Data Parallel) = RP × FS
    - RP (Replicate Process): Multiple replicas with the same parameter shard (gradient averaging)
    - FS (Fully Shard): Different parameter shards within the same replica (reduce-scatter/all-gather)
    - TP (Tensor Parallel): Column-wise sharding of tensors

    Maintains mathematical equivalence with the reference implementation on top
    of the DistributedOptimizer runtime contract.
    """

    def __init__(
        self,
        params,
        lr: float = DEFAULT_LR,
        mu: float = DEFAULT_MU,
        weight_decay: float = DEFAULT_WEIGHT_DECAY,
        rank_fraction: float = 1.0,  # Reference implementation default: full rank
        rank_multiple_of: int = 1,
        epsilon: float = 1e-8,
        rcqr_oversample: float = 1.25,
        betas: tuple = (0.9, 0.95),
        eps: float = 1e-8,
        # Configuration flags
        rp_average_in_collective: bool = True,
        use_fs_collectives: bool = True,  # Enable FS collectives in Distributed mode
        mixed_precision_config: Optional[DionMixedPrecisionConfig] = None,
        enable_async: bool = True,  # Enable async execution where possible
        use_compressed_comm: bool = True,  # Enable compressed communication
        scalar_optimizer: str = "adamw",  # Scalar optimizer for non-Dion params ("adamw" or "lion")
        lr_scaling_rule: str = "moonlight",  # 2D Dion LR scaling rule ("moonlight" or "dion")
        max_concurrent_tasks: Optional[int] = None,  # Optional async concurrency hint
    ):
        defaults = dict(
            lr=lr,
            mu=mu,
            weight_decay=weight_decay,
            rank_fraction=rank_fraction,
            rank_multiple_of=rank_multiple_of,
            epsilon=epsilon,
            rcqr_oversample=rcqr_oversample,
            betas=betas,
            eps=eps,
            rp_average_in_collective=rp_average_in_collective,
            use_fs_collectives=use_fs_collectives,
            enable_async=enable_async,
            use_compressed_comm=use_compressed_comm,
            scalar_optimizer=scalar_optimizer,  # "adamw" or "lion"
            lr_scaling_rule=lr_scaling_rule,
            # Reference implementation uses only RCQR and standard scaling
            algorithm="dion",  # Default algorithm, same as original dion.py
            step=0,  # Per-group step counter
        )
        if lr_scaling_rule not in ("moonlight", "dion"):
            raise RuntimeError(
                "[DION_INVALID_LR_SCALING_RULE] "
                f"expected one of ('moonlight', 'dion'), got {lr_scaling_rule!r}"
            )
        super().__init__(params, defaults)

        # Cache global rank for logging (avoid repeated dist.get_rank() calls)
        self._global_rank = dist.get_rank() if dist.is_initialized() else 0

        # Configuration storage
        self._param_config: Dict[Tensor, DionParamConfig] = {}
        self.is_distributed_mode = False
        self.use_fs_collectives = use_fs_collectives

        self._route_step_params_fn = None
        self.dist_metas = {}

        # Compressed communication support
        self.use_compressed_comm = use_compressed_comm

        # Async collectives
        self.enable_async = enable_async
        self.max_concurrent_tasks = max_concurrent_tasks

        # Mixed precision configuration
        if mixed_precision_config is None:
            mixed_precision_config = DionMixedPrecisionConfig()
        self._mixed_precision_config = mixed_precision_config

        if mixed_precision_config.momentum_dtype is None or mixed_precision_config.q_dtype is None:
            raise RuntimeError(
                "[Dion] momentum_dtype and q_dtype must be explicit. "
                f"got momentum={mixed_precision_config.momentum_dtype} "
                f"Q={mixed_precision_config.q_dtype}"
            )

        # Update counters
        self._dion_update_count = 0
        self._adamw_update_count = 0
        self._step_count = 0

        self._buffer_cache: Dict[str, torch.Tensor] = {}
        self._ortho_sanity_enabled = False
        self._ortho_sanity_every = 1
        self._ortho_sanity_p_tol = 5e-3
        self._ortho_sanity_q_tol = 5e-3
        self._ortho_sanity_mode = "fail"
        self._ortho_sanity_log = False
        self._ortho_sanity_trace = False

    @torch.no_grad()
    def step(self, closure=None):
        """Perform optimization step with async communication."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Reset counters at the beginning of each step
        self._dion_update_count = 0
        self._adamw_update_count = 0

        # Increment per-group step counters
        for optim_group in self.param_groups:
            optim_group['step'] = optim_group.get('step', 0) + 1

        self._step_count += 1
        if not self.is_distributed_mode:
            raise RuntimeError(
                "[DION_STEP_REQUIRES_DISTRIBUTED_MODE] "
                f"step={self._step_count} rank={self._global_rank}"
            )
        dion_tasks = iter_dist_tasks_(self)

        # Execute all tasks with the explicit runtime width from config.
        task_count = 0
        if dion_tasks:
            dion_tasks_list = list(dion_tasks)
            task_count = len(dion_tasks_list)
            if task_count > 0:
                max_tasks = resolve_async_task_limit_(
                    max_concurrent_tasks=self.max_concurrent_tasks,
                    task_count=task_count,
                )
                runtime = AsyncRuntime((t for t in dion_tasks_list), max_concurrent_tasks=max_tasks)
                runtime.run()

                del runtime
            # Always delete dion_tasks_list if it was created
            del dion_tasks_list

        return loss

    def _cached_buffer(
        self,
        name: str,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
        *,
        zero: bool = False,
    ) -> torch.Tensor:
        """Return a reusable cached buffer for Dion batch-update temporaries."""
        shape = tuple(shape)
        buffer = self._buffer_cache.get(name)
        if (
            buffer is None
            or tuple(buffer.shape) != shape
            or buffer.dtype != dtype
            or buffer.device != device
        ):
            buffer = torch.empty(shape, dtype=dtype, device=device)
            self._buffer_cache[name] = buffer
        if zero:
            buffer.zero_()
        return buffer

    def _run_scalar_bucket_async(
        self,
        scalar_params: List[ScalarStepParam],
    ) -> Generator[None, None, None]:
        """Process only non-Dion params while preserving standard DO bucket ownership."""
        scalar_opt = self.defaults.get('scalar_optimizer', 'adamw')
        default_betas = self.defaults.get('betas', (0.9, 0.95))
        default_eps = self.defaults.get('eps', 1e-8)
        for scalar_param in scalar_params:
            p = scalar_param.param
            grad = scalar_param.grad
            state = scalar_param.optimizer_state
            optim_group = scalar_param.optim_group
            lr = optim_group.get('lr', self.defaults['lr'])
            if p.ndim == 1:
                weight_decay = 0.0
            else:
                wd_mult = optim_group.get('wd_mult', 1.0)
                weight_decay = optim_group.get(
                    'weight_decay',
                    self.defaults['weight_decay'] * wd_mult,
                )

            apply_scalar_update(
                optimizer_name=scalar_opt,
                p=p,
                grad=grad,
                state=state,
                optim_group=optim_group,
                default_betas=default_betas,
                default_eps=default_eps,
                lr=lr,
                weight_decay=weight_decay,
            )
        if False:
            yield

    enable_distributed_mode = enable_distributed_mode_
    _replicate_reduce_op = replicate_reduce_op_
    _collapse_grads_across_replicas = collapse_grads_across_replicas_
    _collapse_batch_across_replicas = collapse_batch_across_replicas_
    _collapse_batch_over_replicate_subset = collapse_batch_over_replicate_subset_
    _unshard_q_batch = unshard_q_batch_
    _reduce_p_across_fs_groups = reduce_p_across_fs_groups_
    _reduce_r_across_tp = reduce_r_across_tp_
    _normalize_cols_async = normalize_cols_async_
    _orthogonalize = staticmethod(orthogonalize)
    _reshard_q_along_tp = staticmethod(reshard_q_along_tp)

    _has_tp_shard = staticmethod(has_tp_shard_)
    _has_fs_shard = staticmethod(has_fs_shard_)
    _q_needs_tp_unshard = staticmethod(q_needs_tp_unshard_)
    _is_fs_only_config = staticmethod(is_fs_only_config_)
    _make_seeded_sketch = make_seeded_sketch_for_update_
    _sketch_keys = sketch_keys_for_update_
    _make_local_sketch = make_local_sketch_for_update_
    _distributed_orthogonalize = distributed_orthogonalize_
    _orthogonalize_local_slice = orthogonalize_local_slice_
    _orthogonalize_local_matrix_batch = orthogonalize_local_matrix_batch_
    _orthogonalize_p_batch = orthogonalize_p_batch_
    _orthogonalize_dense_replicate_batch_async = orthogonalize_dense_replicate_batch_async_
    _orthogonalize_fs_only_batch = orthogonalize_fs_only_batch_
    _apply_batch_updates = apply_batch_updates_
    _sanitize_dion_batch = sanitize_dion_batch_for_update_
    _apply_error_feedback = apply_error_feedback_to_momentum_
    _batch_dion_update_async = batch_dion_update_async_
    _run_compressed_comm_async = run_compressed_comm_async_
