"""Dion optimizer main class for Megatron-LM."""
from collections import OrderedDict
import os
import time
from typing import Dict, Generator, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.optim.optimizer import Optimizer

from .runtime import (
    enable_distributed_mode,
    iter_dist_tasks,
    resolve_async_task_limit,
    AsyncRuntime,
)
from .scalar_opts import adamw_update_foreach, lion_update_foreach
from .types import (
    DionMixedPrecisionConfig,
    DionParamConfig,
)
from ..matrix.types import ScalarStepParam

DEFAULT_LR = 0.01
DEFAULT_MU = 0.95
DEFAULT_WEIGHT_DECAY = 0.01


class MegatronDion(Optimizer):
    """
    Dion optimizer with batch processing and low-rank synchronization.

    Implements the DIstributed OrthoNormalized updates (Dion) optimizer with low-rank approximation
    for efficient distributed training. Supports CP-excluded Dion communication domains:
    - RP (Replicate Process): ranks with the same Dion FS shard
    - FS (Fully Shard): ranks covering different Dion shards
    - TP (Tensor Parallel): Column-wise sharding of tensors

    When context parallelism is active, Megatron's standard distributed-optimizer
    data-parallel group can include CP ranks. Dion RP/FS groups are therefore
    resolved from MCore's CP-excluded optimizer topology, not inferred from the
    standard optimizer data-parallel group size.

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
        scalar_eps: float = 1e-8,
        # Configuration flags
        rp_average_in_collective: bool = True,
        use_fs_collectives: bool = True,  # Enable FS collectives in Distributed mode
        mixed_precision_config: Optional[DionMixedPrecisionConfig] = None,
        enable_async: bool = True,  # Enable async execution where possible
        use_low_rank_sync: bool = True,  # Enable low-rank P/R replica sync
        scalar_optimizer: str = "adam",
        scalar_lr_scale: float = 1.0,
        scale_mode: str = "spectral",  # 2D Dion scale mode
        extra_scale_factor: float = 0.2,
        split_qkv: bool = False,
        split_linear: bool = False,
        max_concurrent_tasks: Optional[int] = None,  # Optional async concurrency hint
    ):
        if isinstance(params, (list, tuple)):
            for param_group in params:
                if isinstance(param_group, dict) and "wd_mult" in param_group:
                    base_weight_decay = float(param_group.get("weight_decay", weight_decay))
                    param_group["weight_decay"] = base_weight_decay * float(
                        param_group.get("wd_mult", 1.0)
                    )

        defaults = dict(
            lr=lr,
            mu=mu,
            weight_decay=weight_decay,
            rank_fraction=rank_fraction,
            rank_multiple_of=rank_multiple_of,
            epsilon=epsilon,
            rcqr_oversample=rcqr_oversample,
            betas=betas,
            scalar_eps=scalar_eps,
            rp_average_in_collective=rp_average_in_collective,
            use_fs_collectives=use_fs_collectives,
            enable_async=enable_async,
            use_low_rank_sync=use_low_rank_sync,
            scalar_optimizer=scalar_optimizer,
            scalar_lr_scale=scalar_lr_scale,
            scale_mode=scale_mode,
            extra_scale_factor=extra_scale_factor,
            split_qkv=bool(split_qkv),
            split_linear=bool(split_linear),
            # Reference implementation uses only RCQR and standard scaling
            algorithm="dion",  # Default algorithm, same as original dion.py
            step=0,  # Per-group step counter
        )
        if scale_mode not in ("spectral", "unit_rms_norm", "shape_scaling"):
            raise RuntimeError(
                "[DION_INVALID_SCALE_MODE] "
                f"expected one of ('spectral', 'unit_rms_norm', 'shape_scaling'), "
                f"got {scale_mode!r}"
            )
        if float(scalar_lr_scale) < 0.0:
            raise RuntimeError(
                "[DION_INVALID_SCALAR_LR_SCALE] "
                f"scalar_lr_scale={scalar_lr_scale}"
            )
        super().__init__(params, defaults)

        # Cache global rank for logging (avoid repeated dist.get_rank() calls)
        self._global_rank = dist.get_rank() if dist.is_initialized() else 0

        # Configuration storage
        self._param_config: Dict[torch.Tensor, DionParamConfig] = {}
        self.is_distributed_mode = False
        self.use_fs_collectives = use_fs_collectives

        self._route_step_params = None
        self.dist_metas = {}

        self.use_low_rank_sync = use_low_rank_sync

        # Async collectives
        self.enable_async = enable_async
        self.max_concurrent_tasks = max_concurrent_tasks

        # Mixed precision configuration
        if mixed_precision_config is None:
            mixed_precision_config = DionMixedPrecisionConfig()
        self._mixed_precision_config = mixed_precision_config

        # Update counters
        self._dion_update_count = 0
        self._scalar_update_count = 0
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
        self._scalar_update_count = 0

        # Increment per-group step counters
        for optim_group in self.param_groups:
            optim_group['step'] = optim_group.get('step', 0) + 1

        self._step_count += 1
        if not self.is_distributed_mode:
            raise RuntimeError(
                "[DION_STEP_REQUIRES_DISTRIBUTED_MODE] "
                f"step={self._step_count} rank={self._global_rank}"
            )
        profile_enabled = os.environ.get("DION_PROFILE_SPLIT", "").lower() in (
            "1",
            "true",
            "yes",
        )
        self._profile_enabled = profile_enabled
        if profile_enabled:
            self._profile_records = []
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            profile_step_start = time.perf_counter()
        dion_tasks = iter_dist_tasks(self)

        # Execute all tasks with the explicit runtime width from config.
        max_tasks = resolve_async_task_limit(
            max_concurrent_tasks=self.max_concurrent_tasks,
            task_count=None,
        )
        runtime = AsyncRuntime(dion_tasks, max_concurrent_tasks=max_tasks)
        runtime.run()
        del runtime
        if profile_enabled:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - profile_step_start
            if self._global_rank == 0:
                records = getattr(self, "_profile_records", [])
                by_label = OrderedDict()
                by_desc = {}
                for label, seconds, desc in records:
                    by_label[label] = by_label.get(label, 0.0) + float(seconds)
                    if desc:
                        by_desc[desc] = by_desc.get(desc, 0.0) + float(seconds)
                label_summary = ", ".join(
                    f"{label}={seconds:.3f}s" for label, seconds in by_label.items()
                )
                print(
                    f"[DION_PROFILE] step={self._step_count} total={elapsed:.3f}s "
                    f"{label_summary}",
                    flush=True,
                )
                for desc, seconds in sorted(
                    by_desc.items(), key=lambda item: item[1], reverse=True
                )[:12]:
                    print(
                        f"[DION_PROFILE_BATCH] step={self._step_count} "
                        f"time={seconds:.3f}s {desc}",
                        flush=True,
                    )
        self._buffer_cache.clear()

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

    def _apply_scalar_batches(
        self,
        scalar_params: List[ScalarStepParam],
    ) -> Generator[None, None, None]:
        """Process standard params with grouped foreach-style scalar updates."""
        default_scalar_opt = self.defaults.get('scalar_optimizer', 'adam')
        default_scalar_lr_scale = float(self.defaults.get('scalar_lr_scale', 1.0))
        default_betas = self.defaults.get('betas', (0.9, 0.95))
        default_eps = self.defaults.get('scalar_eps', 1e-8)
        scalar_contract_order: list[tuple] = []
        scalar_batches = OrderedDict()

        def _effective_weight_decay(param, optim_group) -> float:
            wd_mult = optim_group.get('wd_mult', 1.0)
            return float(
                optim_group.get(
                    'weight_decay',
                    self.defaults['weight_decay'] * wd_mult,
                )
            )

        def _resolve_scalar_algorithm(optim_group) -> str:
            group_algorithm = optim_group.get('algorithm', None)
            if group_algorithm is None or group_algorithm == "dion":
                scalar_opt = optim_group.get('scalar_optimizer', default_scalar_opt)
            else:
                scalar_opt = group_algorithm
            if scalar_opt == "adam":
                return "adamw"
            if scalar_opt not in {"adamw", "lion"}:
                raise RuntimeError(
                    "[DION_INVALID_SCALAR_OPT] "
                    f"scalar_optimizer={scalar_opt}"
                )
            return str(scalar_opt)

        def _effective_betas(optim_group) -> tuple[float, float]:
            if 'betas' in optim_group:
                beta1, beta2 = optim_group['betas']
                return float(beta1), float(beta2)
            if 'beta1' in optim_group or 'beta2' in optim_group:
                return (
                    float(optim_group.get('beta1', default_betas[0])),
                    float(optim_group.get('beta2', default_betas[1])),
                )
            return float(default_betas[0]), float(default_betas[1])

        def _state_tensor_dtype(state_tensor, default_dtype: torch.dtype) -> torch.dtype:
            return state_tensor.dtype if state_tensor is not None else default_dtype

        def _ensure_scalar_first_moment(param, state) -> torch.Tensor:
            if 'first_moment' in state and 'exp_avg' in state:
                raise RuntimeError(
                    "[DION_SCALAR_STATE_LAYOUT_CONFLICT] found both first_moment and exp_avg"
                )
            if 'first_moment' not in state:
                if 'exp_avg' in state:
                    state['first_moment'] = state.pop('exp_avg')
                elif 'momentum' in state:
                    state['first_moment'] = state.pop('momentum')
                else:
                    momentum_dtype = (
                        self._mixed_precision_config.momentum_dtype
                        if self._mixed_precision_config.momentum_dtype is not None
                        else param.dtype
                    )
                    state['first_moment'] = torch.zeros_like(param, dtype=momentum_dtype)
            return state['first_moment']

        def _ensure_scalar_second_moment(param, state) -> torch.Tensor:
            if 'second_moment' in state and 'exp_avg_sq' in state:
                raise RuntimeError(
                    "[DION_SCALAR_STATE_LAYOUT_CONFLICT] found both second_moment and exp_avg_sq"
                )
            if 'second_moment' not in state:
                if 'exp_avg_sq' in state:
                    state['second_moment'] = state.pop('exp_avg_sq')
                elif 'variance' in state:
                    state['second_moment'] = state.pop('variance')
                else:
                    variance_dtype = (
                        self._mixed_precision_config.variance_dtype
                        if self._mixed_precision_config.variance_dtype is not None
                        else param.dtype
                    )
                    state['second_moment'] = torch.zeros_like(param, dtype=variance_dtype)
            return state['second_moment']

        for scalar_param in scalar_params:
            p = scalar_param.param
            grad = scalar_param.grad
            state = scalar_param.optimizer_state
            optim_group = scalar_param.optim_group
            scalar_opt = _resolve_scalar_algorithm(optim_group)
            lr_scale = float(optim_group.get('scalar_lr_scale', default_scalar_lr_scale))
            if lr_scale < 0.0:
                raise RuntimeError(
                    "[DION_INVALID_SCALAR_LR_SCALE] "
                    f"scalar_lr_scale={lr_scale}"
                )
            lr = float(optim_group.get('lr', self.defaults['lr'])) * lr_scale
            weight_decay = _effective_weight_decay(p, optim_group)
            step = int(optim_group.get('step', 0))
            eps = float(
                optim_group.get(
                    'scalar_eps',
                    optim_group.get('eps', optim_group.get('epsilon', default_eps)),
                )
            )
            beta1, beta2 = _effective_betas(optim_group)
            if step <= 0:
                raise RuntimeError(f"[DION_INVALID_SCALAR_STEP] step={step}")

            first_moment = _ensure_scalar_first_moment(p, state)
            state['step'] = step
            second_moment = None
            if scalar_opt == "adamw":
                second_moment = _ensure_scalar_second_moment(p, state)

            scalar_contract_key = (
                scalar_opt,
                id(optim_group),
                str(p.device),
                str(p.dtype),
                str(first_moment.dtype),
                str(_state_tensor_dtype(second_moment, torch.float32)),
                float(lr),
                float(weight_decay),
                float(eps),
                float(beta1),
                float(beta2),
                int(step),
            )
            if scalar_contract_key not in scalar_batches:
                scalar_batches[scalar_contract_key] = {
                    "scalar_optimizer": scalar_opt,
                    "params": [],
                    "grads": [],
                    "first_moments": [],
                    "second_moments": [],
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "eps": eps,
                    "step": step,
                    "beta1": beta1,
                    "beta2": beta2,
                }
                scalar_contract_order.append(scalar_contract_key)
            scalar_batch = scalar_batches[scalar_contract_key]
            scalar_batch["params"].append(p)
            scalar_batch["grads"].append(grad)
            scalar_batch["first_moments"].append(first_moment)
            if second_moment is not None:
                scalar_batch["second_moments"].append(second_moment)

        for scalar_contract_key in scalar_contract_order:
            scalar_batch = scalar_batches[scalar_contract_key]
            if scalar_batch["scalar_optimizer"] == "lion":
                lion_update_foreach(
                    scalar_batch["params"],
                    scalar_batch["grads"],
                    scalar_batch["first_moments"],
                    lr=scalar_batch["lr"],
                    beta1=scalar_batch["beta1"],
                    beta2=scalar_batch["beta2"],
                    weight_decay=scalar_batch["weight_decay"],
                )
                continue

            adamw_update_foreach(
                scalar_batch["params"],
                scalar_batch["grads"],
                scalar_batch["first_moments"],
                scalar_batch["second_moments"],
                lr=scalar_batch["lr"],
                beta1=scalar_batch["beta1"],
                beta2=scalar_batch["beta2"],
                weight_decay=scalar_batch["weight_decay"],
                step=scalar_batch["step"],
                epsilon=scalar_batch["eps"],
            )
        if False:
            yield

    enable_distributed_mode = enable_distributed_mode
