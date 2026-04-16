"""Dion optimizer main class for Megatron-LM."""
import math
from typing import Dict, Generator, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch import Tensor
from torch.optim.optimizer import Optimizer

from .kernels import orthogonalize_fs_only_batch
from .ortho import (
    distributed_orthogonalize,
    make_local_sketch_for_update,
    make_seeded_sketch_for_update,
    orthogonalize,
    orthogonalize_local_matrix_batch,
    orthogonalize_local_slice,
    reshard_q_along_tp,
    sketch_keys_for_update,
)
from .runtime import (
    apply_error_feedback_to_momentum,
    apply_batch_updates,
    batch_dion_update_async,
    collapse_batch_across_replicas,
    collapse_grads_across_replicas,
    enable_distributed_mode,
    normalize_cols_async,
    iter_dist_tasks,
    orthogonalize_dense_replicate_batch_async,
    orthogonalize_p_batch,
    replicate_reduce_op,
    reduce_p_across_fs_groups,
    reduce_r_across_tp,
    run_compressed_comm_async,
    sanitize_dion_batch_for_update,
    resolve_async_task_limit,
    AsyncRuntime,
    unshard_q_batch,
)
from .state import (
    has_fs_shard,
    has_tp_shard,
    is_fs_only_config,
    use_q_unshard,
)
from .types import (
    DionMixedPrecisionConfig,
    DionParamConfig,
    ScalarStepParam,
)

DEFAULT_LR = 0.01
DEFAULT_MU = 0.95
DEFAULT_WEIGHT_DECAY = 0.01


def _adam_update(
    p: Tensor,
    grad: Tensor,
    state: dict,
    betas: tuple,
    eps: float,
    lr: float,
    weight_decay: float,
    *,
    decoupled_weight_decay: bool = True,
    step_override: int | None = None,
    store_step_in_state: bool = True,
) -> None:
    """Adam/AdamW update."""
    beta1, beta2 = betas

    if 'exp_avg' not in state:
        state['exp_avg'] = torch.zeros_like(grad, dtype=torch.float32)
        state['exp_avg_sq'] = torch.zeros_like(grad, dtype=torch.float32)
        if store_step_in_state:
            state['step'] = 0

    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

    if step_override is None:
        if 'step' not in state:
            state['step'] = 0
        state['step'] += 1
        step = int(state['step'])
    else:
        step = int(step_override)
        if step <= 0:
            raise RuntimeError(
                f"[DION_INVALID_ADAM_STEP] step_override must be positive, got {step}"
            )
        if store_step_in_state:
            state['step'] = step
        elif 'step' in state:
            state.pop('step', None)

    grad_fp32 = grad.float() if grad.dtype != torch.float32 else grad
    if not decoupled_weight_decay and weight_decay != 0.0:
        grad_fp32 = grad_fp32.add(p.detach().float(), alpha=weight_decay)

    exp_avg.mul_(beta1).add_(grad_fp32, alpha=1 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(grad_fp32, grad_fp32, value=1 - beta2)

    bias_correction1 = 1 - beta1 ** step
    bias_correction2 = 1 - beta2 ** step

    if decoupled_weight_decay and weight_decay != 0.0:
        p.mul_(1 - lr * weight_decay)

    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
    step_size = lr / bias_correction1
    p.data.addcdiv_(exp_avg, denom, value=-step_size)


def _adamw_update(
    p: Tensor,
    grad: Tensor,
    state: dict,
    betas: tuple,
    eps: float,
    lr: float,
    weight_decay: float,
) -> None:
    """AdamW update for non-Dion parameters."""
    _adam_update(
        p,
        grad,
        state,
        betas,
        eps,
        lr,
        weight_decay,
        decoupled_weight_decay=True,
    )


def _lion_update(
    p: Tensor,
    grad: Tensor,
    state: dict,
    betas: tuple,
    lr: float,
    weight_decay: float,
) -> None:
    """Lion optimizer update (sign-based, momentum-only)."""
    beta1, beta2 = betas

    if 'momentum' not in state:
        state['momentum'] = torch.zeros_like(grad, dtype=torch.float32)

    if 'step' not in state:
        state['step'] = 0

    momentum = state['momentum']
    state['step'] += 1
    grad_fp32 = grad.float() if grad.dtype != torch.float32 else grad
    update = momentum.mul(beta1).add_(grad_fp32, alpha=1 - beta1)
    update_sign = update.sign()
    momentum.mul_(beta2).add_(grad_fp32, alpha=1 - beta2)
    p.mul_(1 - lr * weight_decay)
    p.add_(update_sign.to(p.dtype), alpha=-lr)


def _apply_scalar_update(
    *,
    optimizer_name: str,
    p: Tensor,
    grad: Tensor,
    state: dict,
    optim_group: dict,
    default_betas: tuple,
    default_eps: float,
    lr: float,
    weight_decay: float,
) -> None:
    """Apply the configured scalar optimizer without Megatron runtime branching."""
    if optimizer_name == "lion":
        _lion_update(
            p,
            grad,
            state,
            default_betas,
            lr,
            weight_decay,
        )
        return

    _adamw_update(
        p,
        grad,
        state,
        default_betas,
        optim_group.get("eps", default_eps),
        lr,
        weight_decay,
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
        lr_scaling_rule: str = "dion",  # 2D Dion LR scaling rule ("moonlight" or "dion")
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
        dion_tasks = iter_dist_tasks(self)

        # Execute all tasks with the explicit runtime width from config.
        task_count = 0
        if dion_tasks:
            dion_tasks_list = list(dion_tasks)
            task_count = len(dion_tasks_list)
            if task_count > 0:
                max_tasks = resolve_async_task_limit(
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

            _apply_scalar_update(
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

    enable_distributed_mode = enable_distributed_mode
    _replicate_reduce_op = replicate_reduce_op
    _collapse_grads_across_replicas = collapse_grads_across_replicas
    _collapse_batch_across_replicas = collapse_batch_across_replicas
    _unshard_q_batch = unshard_q_batch
    _reduce_p_across_fs_groups = reduce_p_across_fs_groups
    _reduce_r_across_tp = reduce_r_across_tp
    _normalize_cols_async = normalize_cols_async
    _orthogonalize = staticmethod(orthogonalize)
    _reshard_q_along_tp = staticmethod(reshard_q_along_tp)

    _has_tp_shard = staticmethod(has_tp_shard)
    _has_fs_shard = staticmethod(has_fs_shard)
    _use_q_unshard = staticmethod(use_q_unshard)
    _is_fs_only_config = staticmethod(is_fs_only_config)
    _make_seeded_sketch = make_seeded_sketch_for_update
    _sketch_keys = sketch_keys_for_update
    _make_local_sketch = make_local_sketch_for_update
    _distributed_orthogonalize = distributed_orthogonalize
    _orthogonalize_local_slice = orthogonalize_local_slice
    _orthogonalize_local_matrix_batch = orthogonalize_local_matrix_batch
    _orthogonalize_p_batch = orthogonalize_p_batch
    _orthogonalize_dense_replicate_batch_async = orthogonalize_dense_replicate_batch_async
    _orthogonalize_fs_only_batch = orthogonalize_fs_only_batch
    _apply_batch_updates = apply_batch_updates
    _sanitize_dion_batch = sanitize_dion_batch_for_update
    _apply_error_feedback = apply_error_feedback_to_momentum
    _batch_dion_update_async = batch_dion_update_async
    _run_compressed_comm_async = run_compressed_comm_async
