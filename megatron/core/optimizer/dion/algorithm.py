"""Dion optimizer main class for Megatron-LM."""

import logging
import os
from typing import Any, Dict, Generator, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
from torch import Tensor
from torch.optim.optimizer import Optimizer

from megatron.core import parallel_state

from .async_runtime import AsyncRuntime, AsyncTask
from .batching import BatchProcessor, pad_batch
from .constants import DEFAULT_LR, DEFAULT_MU, DEFAULT_WEIGHT_DECAY
from .ortho import orthogonalize, reshard_q_along_tp
from .scalar_opt import adamw_update, lion_update
from .types import DionMixedPrecisionConfig, DionParamConfig, MegatronDionDistMeta
from .utils import get_global_shape, str_to_dtype, infer_local_2d_shape


logger = logging.getLogger(__name__)


class MegatronDion(Optimizer):
    """
    Dion optimizer with batch processing and compressed communication.

    Implements the DIstributed OrthoNormalized updates (Dion) optimizer with low-rank approximation
    for efficient distributed training. Supports 2D parallelism model:
    - DP (Data Parallel) = RP × FS
    - RP (Replicate Process): Multiple replicas with the same parameter shard (gradient averaging)
    - FS (Fully Shard): Different parameter shards within the same replica (reduce-scatter/all-gather)
    - TP (Tensor Parallel): Column-wise sharding of tensors

    Maintains mathematical equivalence with reference implementation while
    being compatible with DistributedOptimizer.
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
        # Process groups for 2D parallelism (DP = RP × FS)
        tp_group=None,  # Tensor Parallel group (column-wise sharding)
        rp_group=None,  # Replicate Process group: replicas with same shard (gradient averaging)
        fs_group=None,  # Fully Shard group: different shards within same replica (row-wise sharding)
        # Configuration flags
        rp_average_in_collective: bool = True,  # Average gradients across RP group
        use_fs_collectives: bool = True,  # Enable FS collectives in Distributed mode
        mixed_precision_config: Optional[DionMixedPrecisionConfig] = None,
        enable_async: bool = True,  # Enable async execution where possible
        use_compressed_comm: bool = True,  # Enable compressed communication
        scalar_optimizer: str = "adamw",  # Scalar optimizer for non-Dion params ("adamw" or "lion")
        lr_scaling: str = "dion",  # LR scaling: "dion_ref", "dion", or "moonlight"
        is_ep_mode: bool = False,  # EP mode: use sync execution to avoid collective desync
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
            lr_scaling=lr_scaling,  # "dion_ref", "dion", or "moonlight"
            # Reference implementation uses only RCQR and standard scaling
            algorithm="dion",  # Default algorithm, same as original dion.py
            step=0,  # Per-group step counter
        )
        super().__init__(params, defaults)

        # Store lr_scaling type
        self._lr_scaling = lr_scaling

        # Cache global rank for logging (avoid repeated dist.get_rank() calls)
        self._global_rank = dist.get_rank() if dist.is_initialized() else 0

        global_rank = self._global_rank

        # EP mode: use sync execution to avoid collective desync across ranks
        # When EP > 1, different optimizer instances (Dense vs Expert) have different
        # fs_group compositions, and AsyncRuntime can cause ranks to reach different
        # yield points, leading to collective hangs.
        self.is_ep_mode = is_ep_mode

        # Store the correct process groups
        self.tp_group = tp_group  # Tensor Parallel
        self.rp_group = rp_group  # RP: replicas with same shard
        self.fs_group = fs_group  # FS: shards within same replica

        # Compute world sizes and ranks
        self.tp_world_size = dist.get_world_size(tp_group) if tp_group else 1
        self.rp_world_size = dist.get_world_size(rp_group) if rp_group else 1
        self.fs_size = dist.get_world_size(fs_group) if fs_group else -1  # Use -1 to indicate not set

        self.tp_rank = dist.get_rank(tp_group) if tp_group else 0
        self.rp_rank = dist.get_rank(rp_group) if rp_group else 0
        self.fs_rank = dist.get_rank(fs_group) if fs_group else 0

        # Configuration storage
        self._param_config: Dict[Tensor, DionParamConfig] = {}
        self.distributed_mode = False
        self.dist_metas = {}
        self.dist_group = None
        self.buffer_indices = {}
        self.is_distributed_mode = False
        self.use_fs_collectives = use_fs_collectives

        # UID → param mapping for state lookup across offload/reload cycles
        self._uid_to_param: Dict[Tuple, Tensor] = {}

        # UID → dist_meta for reverse lookup when param identity changes
        self._dist_meta_by_uid: Dict[Tuple, any] = {}

        # Compressed communication support
        self.use_compressed_comm = use_compressed_comm

        # Batch processor for improved performance
        self.batch_processor = BatchProcessor()

        # Async collectives
        self.enable_async = enable_async

        # Mixed precision configuration
        if mixed_precision_config is None:
            mixed_precision_config = DionMixedPrecisionConfig()
        self._mixed_precision_config = mixed_precision_config

        # Warn if half precision dtypes may cause numerical instability
        if mixed_precision_config.momentum_dtype is None or mixed_precision_config.Q_dtype is None:
            logger.warning(
                f"[Dion] momentum_dtype/Q_dtype not set. For fp16/bf16 models, "
                f"set momentum_dtype=torch.float32 and Q_dtype=torch.float32 to avoid instability. "
                f"Current: momentum={mixed_precision_config.momentum_dtype}, Q={mixed_precision_config.Q_dtype}"
            )

        # Update counters
        self._dion_update_count = 0
        self._adamw_update_count = 0
        self._step_count = 0

        # RNG generator for orthogonalization (re-seeded each step)
        self._ortho_generator: Optional[torch.Generator] = None

    @torch.no_grad()
    def step(self, closure=None):
        """Perform optimization step with async communication."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Test mode: Apply Adam update using Dion gradients (DION_TEST_ADAM=1)
        if os.getenv("DION_TEST_ADAM", "0") == "1":
            return self._step_test_adam(loss)

        # Reset counters at the beginning of each step
        self._dion_update_count = 0
        self._adamw_update_count = 0

        # Increment per-group step counters
        for group in self.param_groups:
            group['step'] = group.get('step', 0) + 1

        self._step_count += 1

        # Initialize orthogonalization RNG at step start
        if not hasattr(self, '_cached_device'):
            self._cached_device = None
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is not None or hasattr(p, 'main_grad'):
                        self._cached_device = p.device
                        break
                if self._cached_device is not None:
                    break
        if self._cached_device is not None:
            self._init_ortho_generator(self._cached_device)

        # Create async tasks for optimization
        if self.is_distributed_mode:
            dion_tasks = self._create_dion_tasks_distributed()
        else:
            dion_tasks = self._create_dion_tasks_local()

        # Execute all tasks asynchronously
        task_count = 0
        if dion_tasks:
            dion_tasks_list = list(dion_tasks)
            task_count = len(dion_tasks_list)
            if task_count > 0:
                # EP mode: sequential execution to avoid collective desync
                # Non-EP mode: parallel execution for better performance
                max_tasks = 1 if self.is_ep_mode else 3
                runtime = AsyncRuntime((t for t in dion_tasks_list), max_concurrent_tasks=max_tasks)
                runtime.run()

                del runtime
            # Always delete dion_tasks_list if it was created
            del dion_tasks_list

        return loss

    def offload_to_cpu(self):
        """
        Release Dion-specific GPU buffers during offload.

        Q buffers auto-recreate on device change via lazy allocation.

        The Standard DO pattern handles offload via:
        1. Buffer level: param_and_grad_buffer.offload_to_cpu() - handles grad/param data
        2. Optimizer state: move_optimizer("cpu") - moves optimizer state tensors to CPU
        3. Q buffers: Auto-cleared when device changes (see _gather_q_across_tp)
        """
        # Release Q buffers for immediate memory reclaim
        for group in self.param_groups:
            for p in group['params']:
                if p in self.state:
                    state = self.state[p]
                    if '_q_full_buffer' in state:
                        state['_q_full_buffer'] = None
                    if '_q_gather_buffer' in state:
                        state['_q_gather_buffer'] = None

        # Clear batch Q caches
        if hasattr(self, '_q_full_buffers'):
            self._q_full_buffers.clear()
        if hasattr(self, '_q_gather_buffers'):
            self._q_gather_buffers.clear()
        if hasattr(self, '_q_buffer_device'):
            del self._q_buffer_device

        torch.cuda.empty_cache()

    def _create_dion_tasks_distributed(self) -> Generator[AsyncTask, None, None]:
        """Create async tasks for Distributed mode optimization."""
        # Group parameters by buffer/bucket
        buffer_buckets = {}
        unbuffered_params = []  # For 1D params not in buffer_indices

        for group in self.param_groups:
            for p in group['params']:
                grad = self._get_gradient(p)
                if grad is None:
                    continue

                # Initialize state if needed (lazy initialization)
                state = self._get_or_initialize_state(p, group)

                # Group by buffer/bucket
                if p in self.buffer_indices:
                    buf_idx, bucket_idx = self.buffer_indices[p]
                    key = (buf_idx, bucket_idx)
                    if key not in buffer_buckets:
                        buffer_buckets[key] = []
                    buffer_buckets[key].append((p, group))
                else:
                    # 1D params (bias, layernorm) not in buffer_indices
                    unbuffered_params.append((p, group))

        # Process each buffer/bucket with batching
        for (buf_idx, bucket_idx), params in sorted(buffer_buckets.items()):
            yield AsyncTask(self._process_bucket_batch_async(params, buf_idx, bucket_idx))

        # Process unbuffered params (1D params like bias, layernorm)
        if unbuffered_params:
            # Use a fake buffer index for unbuffered params
            yield AsyncTask(self._process_bucket_batch_async(unbuffered_params, -1, -1))

    def _create_dion_tasks_local(self) -> Generator[AsyncTask, None, None]:
        """Create async tasks for regular mode optimization."""
        for group in self.param_groups:
            params_with_grad = []
            for p in group['params']:
                grad = self._get_gradient(p)
                if grad is not None:
                    params_with_grad.append(p)

            if not params_with_grad:
                continue

            # Initialize states
            for p in params_with_grad:
                if p not in self.state:
                    self.state[p] = {}
                if len(self.state[p]) == 0:
                    self._init_state(p, self.state[p], group)

            # Group by configuration
            configs = [self._get_param_config(p) for p in params_with_grad]

            # Create batches
            batches = self.batch_processor.create_batches(
                [(p, group) for p in params_with_grad],
                configs
            )

            # Process each batch
            for batch in batches:
                batch_params = [item[0] for item in batch]
                batch_configs = [item[2] for item in batch]

                yield AsyncTask(self._process_param_batch_regular_async(batch_params, group, batch_configs))

    def _process_bucket_batch_async(self, params: List[Tuple[Tensor, dict]],
                                   buf_idx: int, bucket_idx: int) -> Generator[None, None, None]:
        """Process bucket with async batch operations for mathematical equivalence."""
        # Sort for deterministic ordering
        params_sorted = sorted(params, key=lambda it: (
            getattr(self.dist_metas.get(it[0], None), 'param_uid', (0, 0, 0)) or (0, 0, 0)
        ))

        # Classify parameters: Dion (2D with Q) vs AdamW (1D or no Q)
        dion_params = []
        adamw_params = []

        for p, group in params_sorted:
            grad = self._get_gradient(p)
            if grad is None:
                continue

            # Lookup state - MUST EXIST (initialized in _create_dion_tasks_distributed)
            state = self.state[p]
            config = self._get_param_config(p)
            meta = self.dist_metas.get(p, None)

            # Classification: 2D + is_dion_param + algorithm=='dion'
            # - Q initialized: 'Q' in state (Q is only created for 2D Dion params)
            algorithm = group.get('algorithm', 'dion')
            is_dion_marked = meta.is_dion_param if meta else False

            # Use GLOBAL shape to determine if 2D (same logic as state initialization)
            is_2d_global = False
            if meta and meta.global_shape and len(meta.global_shape) == 2:
                is_2d_global = True
            elif p.ndim == 2:
                is_2d_global = True

            use_dion = (algorithm == 'dion' and is_dion_marked and
                       is_2d_global and 'Q' in state)

            if use_dion:
                dion_params.append((p, grad, state, group, config, meta))
                self._dion_update_count += 1
            else:
                adamw_params.append((p, grad, state, group))
                self._adamw_update_count += 1

        # Process Dion parameters in batches with async
        if dion_params:
            yield from self._process_dion_batch_async(dion_params)

        # Process non-Dion parameters (AdamW or Lion)
        scalar_opt = self.defaults.get('scalar_optimizer', 'adamw')

        for p, grad, state, group in adamw_params:
            lr = group.get('lr', self.defaults['lr'])
            # Apply weight decay same as Adam:
            # - 1D params (bias, norm scale): no weight decay
            # - 2D+ params (embedding, lm_head): apply weight decay with wd_mult
            if p.ndim == 1:
                weight_decay = 0.0
            else:
                wd_mult = group.get('wd_mult', 1.0)
                weight_decay = self.defaults['weight_decay'] * wd_mult

            # Choose scalar optimizer
            if scalar_opt == 'lion':
                self._lion_update(p, grad, state, group, lr, weight_decay)
            else:  # 'adamw' or 'adam' (treat as adamw)
                self._adamw_update(p, grad, state, group, lr, weight_decay)

        # Clear lists
        params_sorted.clear()
        dion_params.clear()
        adamw_params.clear()

    def _process_param_batch_regular_async(self, params: List[Tensor], group: dict,
                                          configs: List[DionParamConfig]) -> Generator[None, None, None]:
        """Process parameter batch in regular mode with async."""
        dion_data = []
        adamw_data = []

        for p, config in zip(params, configs):
            grad = self._get_gradient(p)
            if grad is None:
                continue

            state = self.state[p]
            algorithm = group.get('algorithm', 'dion')
            use_dion = (p.ndim == 2 and 'Q' in state and algorithm == 'dion')

            if use_dion:
                dion_data.append((p, grad, state, config))
            else:
                adamw_data.append((p, grad, state))

        # Process Dion parameters as batch with async
        if dion_data:
            yield from self._process_regular_dion_batch_async(dion_data, group)

        # Process non-Dion parameters (AdamW or Lion)
        scalar_opt = self.defaults.get('scalar_optimizer', 'adamw')

        for p, grad, state in adamw_data:
            lr = group.get('lr', self.defaults['lr'])
            # Apply weight decay same as Adam:
            # - 1D params (bias, norm scale): no weight decay
            # - 2D+ params (embedding, lm_head): apply weight decay with wd_mult
            if p.ndim == 1:
                weight_decay = 0.0
            else:
                wd_mult = group.get('wd_mult', 1.0)
                weight_decay = self.defaults['weight_decay'] * wd_mult

            # Choose scalar optimizer
            if scalar_opt == 'lion':
                self._lion_update(p, grad, state, group, lr, weight_decay)
            else:  # 'adamw' or 'adam'
                self._adamw_update(p, grad, state, group, lr, weight_decay)

        # Clear lists
        dion_data.clear()
        adamw_data.clear()

    def _process_dion_batch_async(self, dion_params: List[Tuple]) -> Generator[None, None, None]:
        """Process batch of Dion parameters with async operations."""
        if not dion_params:
            return

        # Group parameters by shape to enable batching
        shape_groups = {}

        for p, grad, state, group, config, meta in dion_params:
            # Use stored local_shape from state
            local_shape = state.get('local_shape', None)
            if local_shape is None:
                # Fallback to inference for backward compatibility
                local_shape = self._infer_local_2d_shape(p, meta)
                if local_shape is None:
                    continue
                # Store for future use
                state['local_shape'] = local_shape

            # Create key based on shape and sharding configuration
            shape_key = (local_shape, config.has_fs_axis, config.has_tp_axis, config.is_transposed,
                        config.compressed_all_reduce,
                        config.inner_shard_tensor_dim, config.outer_shard_tensor_dim)

            if shape_key not in shape_groups:
                shape_groups[shape_key] = {
                    'params': [], 'grads': [], 'states': [], 'groups': [],
                    'configs': [], 'metas': []
                }

            shape_groups[shape_key]['params'].append(p)
            shape_groups[shape_key]['grads'].append(grad)
            shape_groups[shape_key]['states'].append(state)
            shape_groups[shape_key]['groups'].append(group)
            shape_groups[shape_key]['configs'].append(config)
            shape_groups[shape_key]['metas'].append(meta)

        # Process each shape group separately
        # Sort keys for deterministic order across all ranks
        global_param_offset = 0
        ortho_completed_count = 0

        # Sync shape_keys across TP group to ensure all TP ranks process same batches
        # This prevents collective desync when Distributed Optimizer shards params differently across TP ranks
        local_shape_keys = sorted(shape_groups.keys())
        if self.tp_group is not None and dist.get_world_size(self.tp_group) > 1:
            all_shape_keys = self._sync_shape_keys_across_tp_group(local_shape_keys)
        else:
            all_shape_keys = local_shape_keys

        for shape_key in all_shape_keys:
            # Check if this rank has params for this shape_key
            if shape_key not in shape_groups:
                # This rank doesn't have params for this shape_key
                # If has_tp_axis=True, we still need to participate in TP collectives with dummy data
                has_tp_axis = shape_key[2]
                if has_tp_axis and self.tp_group is not None and dist.get_world_size(self.tp_group) > 1:
                    local_num_params = 0
                    num_params_tensor = torch.tensor([local_num_params], device=torch.cuda.current_device(), dtype=torch.int64)
                    dist.all_reduce(num_params_tensor, op=dist.ReduceOp.MAX, group=self.tp_group)
                    max_num_params = int(num_params_tensor.item())

                    if max_num_params > 0:
                        shard_world_size = dist.get_world_size(self.tp_group)
                        batch_size = shard_world_size
                        num_batches = (max_num_params + batch_size - 1) // batch_size
                        for _ in range(num_batches):
                            yield from self._batch_dion_dummy_collective(shape_key, batch_size)
                continue

            group_data = shape_groups[shape_key]
            local_shape = shape_key[0]
            m, n = local_shape

            has_tp_axis = shape_key[2]
            has_fs_axis = shape_key[1]

            if has_tp_axis and self.tp_group:
                shard_world_size = dist.get_world_size(self.tp_group)
            elif has_fs_axis and self.fs_group:
                shard_world_size = dist.get_world_size(self.fs_group)
            else:
                shard_world_size = dist.get_world_size(self.rp_group) if self.rp_group else 1

            batch_size = shard_world_size

            # Sync number of params for this shape_key across TP group
            local_num_params = len(group_data['params'])
            if has_tp_axis and self.tp_group is not None and dist.get_world_size(self.tp_group) > 1:
                num_params_tensor = torch.tensor([local_num_params], device=torch.cuda.current_device(), dtype=torch.int64)
                dist.all_reduce(num_params_tensor, op=dist.ReduceOp.MAX, group=self.tp_group)
                max_num_params = int(num_params_tensor.item())
            else:
                max_num_params = local_num_params

            for i in range(0, max_num_params, batch_size):
                # Check if this rank has params for this batch iteration
                if i >= len(group_data['params']):
                    if has_tp_axis and self.tp_group is not None and dist.get_world_size(self.tp_group) > 1:
                        yield from self._batch_dion_dummy_collective(shape_key, batch_size)
                    continue
                batch_end = min(i + batch_size, len(group_data['params']))

                # Extract batch components - Store ORIGINAL params directly (not views)
                params = []
                param_shapes = []
                momentums = []
                Qs = []
                configs = []
                metas = []
                groups = []
                grads_to_process = []
                states = []

                for j in range(i, batch_end):
                    p = group_data['params'][j]
                    grad = group_data['grads'][j]
                    state = group_data['states'][j]
                    group = group_data['groups'][j]
                    config = group_data['configs'][j]
                    meta = group_data['metas'][j]

                    # Store ORIGINAL params and their 2D shapes
                    params.append(p)
                    param_shapes.append((m, n))

                    # Gradients need 2D view for M @ Q computation
                    grad_2d = grad.view(m, n)
                    grads_to_process.append(grad_2d)

                    # Momentum needs 2D view for M @ Q computation
                    momentum_2d = state['momentum'].view(m, n)
                    momentums.append(momentum_2d)

                    Qs.append(state['Q'])
                    configs.append(config)
                    metas.append(meta)
                    groups.append(group)
                    states.append(state)

                # Track real_batch_size before padding
                real_batch_size = len(params)

                # Match Reference pad_batch() exactly
                params = pad_batch(params, batch_size)
                grads_to_process = pad_batch(grads_to_process, batch_size)
                momentums = pad_batch(momentums, batch_size)
                Qs = pad_batch(Qs, batch_size)

                # Pad metadata lists to match (reuse first entry)
                while len(param_shapes) < batch_size:
                    param_shapes.append(param_shapes[0])
                while len(configs) < batch_size:
                    configs.append(configs[0])
                while len(metas) < batch_size:
                    metas.append(metas[0])
                while len(groups) < batch_size:
                    groups.append(groups[0])
                # NOTE: states is NOT padded - only real entries exist

                if params:
                    # Pass original params directly - updates apply to them
                    yield from self._batch_dion_update_async(params, momentums, Qs, configs, metas, groups, grads_to_process, states, param_shapes, real_batch_size, global_param_offset)

                    # Update global offset for next batch
                    global_param_offset += real_batch_size

                    del params, param_shapes, momentums, Qs, configs, metas, groups, grads_to_process, states

            group_data.clear()

        shape_groups.clear()

    def _process_regular_dion_batch_async(self, dion_data: List[Tuple], group: dict) -> Generator[None, None, None]:
        """Process Dion parameters in regular mode with async batching."""
        params = []
        momentums = []
        Qs = []
        configs = []
        states = []

        for p, grad, state, config in dion_data:
            # Update momentum: M <- M + g
            momentum = state['momentum']
            if momentum.dtype == torch.float32 and grad.dtype != torch.float32:
                momentum.add_(grad.to(torch.float32))
            else:
                momentum.add_(grad)

            params.append(p)
            momentums.append(momentum)
            Qs.append(state['Q'])
            configs.append(config)
            states.append(state)

        # Empty metas for regular mode
        metas = [None] * len(params)
        groups = [group] * len(params)

        # Use async batch update
        yield from self._batch_dion_update_async(params, momentums, Qs, configs, metas, groups, None, states)

        del params, momentums, Qs, configs, metas, groups, states

    def _get_gradient(self, param: torch.nn.Parameter) -> Optional[torch.Tensor]:
        """Get gradient from parameter, handling different storage modes.

        Important: Check value is not None, not just hasattr.
        This ensures all ranks use same gradient source for collective operations.
        """
        # Priority 1 - main_grad (DO + FS sharding mode)
        mg = getattr(param, 'main_grad', None)
        if mg is not None:
            return mg

        # Priority 2: decoupled_grad (precision-aware optimizer)
        dg = getattr(param, 'decoupled_grad', None)
        if dg is not None:
            return dg

        # Priority 3: grad (fallback only - regular mode OR non-sharded params)
        try:
            if param.grad is not None:
                return param.grad
        except RuntimeError:
            # Non-leaf tensor grad access failed
            pass

        return None

    def enable_distributed_mode(self, global_buffer_sizes, dist_group, tp_group,
                               dist_metas: Dict[torch.nn.Parameter, MegatronDionDistMeta],
                               rp_group=None,
                               fs_group=None):
        """Enable distributed mode for Megatron-Core backend.

        Args:
            global_buffer_sizes: Buffer sizes for gradients
            dist_group: Full data parallel group (RP × FS)
            tp_group: Tensor parallel process group
            dist_metas: Metadata for distributed parameters
            rp_group: Optional explicit RP group (replicas with same shard)
            fs_group: Optional explicit FS group (shards within same replica)
        """
        global_rank = self._global_rank

        self.global_buffer_sizes = global_buffer_sizes
        self.dist_group = dist_group
        self.dist_metas = dist_metas
        self.is_distributed_mode = True


        # Collective voting to detect inconsistent rp_group/fs_group provision
        if dist.is_initialized() and dist_group is not None:
            have_rp_arg = torch.tensor([1 if rp_group is not None else 0],
                                       device=torch.cuda.current_device(), dtype=torch.int64)
            have_fs_arg = torch.tensor([1 if fs_group is not None else 0],
                                       device=torch.cuda.current_device(), dtype=torch.int64)
            dist.all_reduce(have_rp_arg, op=dist.ReduceOp.MIN, group=dist_group)
            dist.all_reduce(have_fs_arg, op=dist.ReduceOp.MIN, group=dist_group)

            # Check if all ranks received same arguments
            if int(have_rp_arg.item()) != 1 and rp_group is not None:
                raise RuntimeError(
                    f"Global rank {global_rank}: Inconsistent rp_group provision! "
                    f"This rank received rp_group, but some DP ranks did not (MIN=0). "
                    f"Ensure rp_group is provided uniformly to all ranks to prevent fallback mismatch."
                )
            if int(have_fs_arg.item()) != 1 and fs_group is not None:
                raise RuntimeError(
                    f"Global rank {global_rank}: Inconsistent fs_group provision! "
                    f"This rank received fs_group, but some DP ranks did not (MIN=0). "
                    f"Ensure fs_group is provided uniformly to all ranks."
                )

        # Update 2D parallelism groups if provided

        if fs_group is not None:
            try:
                self.fs_group = fs_group
                self.fs_size = dist.get_world_size(fs_group)
                self.fs_rank = dist.get_rank(fs_group)
            except Exception as e:
                logger.error(f"[Dion] Global rank {global_rank}: Failed updating FS group: {e}")
                raise

        # Map parameters to buffer/bucket locations
        for param, meta in self.dist_metas.items():
            if hasattr(meta, 'buffer_idx') and hasattr(meta, 'bucket_idx'):
                self.buffer_indices[param] = (meta.buffer_idx, meta.bucket_idx)
            # Build UID -> meta cache for state migration when dist_metas.get(param) fails
            if meta.param_uid is not None:
                self._dist_meta_by_uid[meta.param_uid] = meta

        # Update process groups if needed
        if not self.tp_group:
            self.tp_group = tp_group
            self.tp_world_size = dist.get_world_size(tp_group) if tp_group else 1
            self.tp_rank = dist.get_rank(tp_group) if tp_group else 0

        self.distributed_mode = True

        # Verify consistent configuration across all ranks
        if dist.is_initialized() and dist_group is not None:
            world_size = dist.get_world_size(dist_group)
            if world_size > 1:
                rp_size_local = dist.get_world_size(self.rp_group) if self.rp_group else 0
                fs_size_local = dist.get_world_size(self.fs_group) if self.fs_group else 0

                # Gather config values AND group sizes from all ranks
                local_config = {
                    'use_compressed_comm': self.use_compressed_comm,
                    'use_fs_collectives': self.use_fs_collectives,
                    'rp_group_size': rp_size_local,
                    'fs_group_size': fs_size_local,
                }
                gathered_configs = [None] * world_size
                dist.all_gather_object(gathered_configs, local_config, group=dist_group)

                # Verify all ranks have same config
                for rank_idx, rank_config in enumerate(gathered_configs):
                    if rank_config != gathered_configs[0]:
                        logger.error(
                            f"CRITICAL: Dion config mismatch across ranks! "
                            f"Rank 0 config: {gathered_configs[0]}, "
                            f"Rank {rank_idx} config: {rank_config}. "
                            f"Difference: {set(rank_config.items()) ^ set(gathered_configs[0].items())}"
                        )
                        raise ValueError(
                            f"Dion config mismatch! "
                            f"All ranks must have identical use_compressed_comm, "
                            f"use_fs_collectives, AND same rp_group_size/fs_group_size. "
                            f"This ensures no fallback inconsistency."
                        )

        # Log Dion parameter statistics
        dion_param_count = sum(1 for meta in self.dist_metas.values() if meta.is_dion_param)
        total_param_count = len(self.dist_metas)
        param_2d_count = sum(1 for meta in self.dist_metas.values()
                            if meta.shape and len(meta.shape) == 2)
        dion_param_elements = sum(
            meta.shape[0] * meta.shape[1] if meta.shape and len(meta.shape) == 2 else 0
            for meta in self.dist_metas.values() if meta.is_dion_param
        )

    # Test mode: Adam update using Dion gradients (DION_TEST_ADAM=1)
    @torch.no_grad()
    def _step_test_adam(self, loss=None):
        """Test mode: Apply Adam update using Dion gradients."""
        self._step_count += 1
        global_rank = self._global_rank

        total_updated = 0
        total_zero_grad = 0

        for group in self.param_groups:
            lr = group.get('lr', self.defaults['lr'])
            # Apply per-group weight decay multiplier (wd_mult)
            wd_mult = group.get('wd_mult', 1.0)
            weight_decay = self.defaults['weight_decay'] * wd_mult
            beta1 = 0.9
            beta2 = 0.999
            eps = 1e-8

            for p in group['params']:
                if not getattr(p, 'is_dion_param', False):
                    continue

                # Get gradient (same path as Dion update)
                grad = self._get_gradient(p)
                if grad is None:
                    total_zero_grad += 1
                    continue

                # Ensure gradient is float32 for numerical stability
                if grad.dtype != torch.float32:
                    grad = grad.float()

                # Check for zero gradient
                grad_norm = grad.norm().item()
                if grad_norm < 1e-12:
                    total_zero_grad += 1
                    continue

                # Get or initialize Adam states
                state = self.state[p]
                if 'test_exp_avg' not in state:
                    state['test_exp_avg'] = torch.zeros_like(grad, dtype=torch.float32)
                    state['test_exp_avg_sq'] = torch.zeros_like(grad, dtype=torch.float32)
                    state['test_step'] = 0

                state['test_step'] += 1
                step = state['test_step']

                exp_avg = state['test_exp_avg']
                exp_avg_sq = state['test_exp_avg_sq']

                # Adam update
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                step_size = lr / bias_correction1
                denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(eps)

                # Weight decay (decoupled, AdamW style)
                if weight_decay > 0:
                    p.data.add_(p.data, alpha=-lr * weight_decay)

                # Apply update
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                total_updated += 1

        return loss

    # Scalar optimizer methods (AdamW, Lion)

    def _adamw_update(self, p: Tensor, grad: Tensor, state: Dict, group: Dict,
                     lr: float, weight_decay: float):
        """AdamW update for non-Dion parameters."""
        beta1, beta2 = self.defaults.get('betas', (0.9, 0.95))
        eps = group.get('eps', self.defaults.get('eps', 1e-8))

        # Call pure function
        adamw_update(p, grad, state, (beta1, beta2), eps, lr, weight_decay)

    def _lion_update(self, p: Tensor, grad: Tensor, state: Dict, group: Dict,
                    lr: float, weight_decay: float):
        """Lion optimizer update for non-Dion parameters."""
        beta1, beta2 = self.defaults.get('betas', (0.9, 0.95))

        # Call pure function
        lion_update(p, grad, state, (beta1, beta2), lr, weight_decay)

    # State management

    def _get_global_shape(self, meta: Optional[MegatronDionDistMeta],
                         local_m: int, local_n: int) -> Tuple[int, int]:
        """Get fully global shape (FS and TP restored) for rank calculation."""
        return get_global_shape(meta, local_m, local_n)

    def _str_to_dtype(self, dtype_val):
        """Convert string dtype to torch.dtype if needed."""
        return str_to_dtype(dtype_val)

    def _infer_local_2d_shape(self, p: Tensor, meta: MegatronDionDistMeta):
        """Infer local 2D shape from flattened parameter in Distributed mode."""
        return infer_local_2d_shape(p, meta)

    def _get_param_config(self, param: Tensor) -> DionParamConfig:
        """Get or create parameter configuration with reference-aligned axis selection."""
        if param in self._param_config:
            return self._param_config[param]

        config = DionParamConfig()
        config.has_fs_axis = False
        config.has_tp_axis = False

        meta = self.dist_metas.get(param, None) if hasattr(self, 'dist_metas') else None

        if self.is_distributed_mode and meta and meta.global_shape and len(meta.global_shape) == 2 and meta.is_dion_param:
            gm, gn = meta.global_shape
            if param.ndim == 2:
                lm, ln = param.shape

                # Detect TP sharding (inner axis)
                tp_split_dim = getattr(meta, 'tp_split_dim', -1) if meta else -1
                if (tp_split_dim in (0, 1)) and self.tp_world_size > 1:
                    config.has_tp_axis = True
                    config.inner_shard_tensor_dim = tp_split_dim

                # Detect FS sharding (outer axis)
                fs_split_dim = getattr(meta, 'fs_split_dim', -1) if meta else -1
                if fs_split_dim in (0, 1) and self.fs_size > 1:
                    config.has_fs_axis = True
                    config.outer_shard_tensor_dim = fs_split_dim

                # Determine is_transposed
                inner = config.inner_shard_tensor_dim
                outer = config.outer_shard_tensor_dim
                config.is_transposed = (lm < ln)
                if inner == 0 or outer == 1:
                    config.is_transposed = False
                if outer == 0 or inner == 1:
                    config.is_transposed = True

                # Auto-decide compressed_all_reduce
                if self.use_compressed_comm:
                    r_state = self.state.get(param, {}).get('r', None)
                    if r_state is not None:
                        r_global = int(r_state)
                    else:
                        if meta.global_shape is not None:
                            mg, ng = meta.global_shape
                            if config.has_tp_axis and self.tp_world_size > 1:
                                if config.inner_shard_tensor_dim == 0:
                                    mg_true = mg * self.tp_world_size
                                    ng_true = ng
                                elif config.inner_shard_tensor_dim == 1:
                                    mg_true = mg
                                    ng_true = ng * self.tp_world_size
                                else:
                                    mg_true, ng_true = mg, ng
                            else:
                                mg_true, ng_true = mg, ng
                            r_global = meta.rank_fraction * min(mg_true, ng_true)
                            r_global = max(1, int(r_global))
                        else:
                            r_global = meta.rank_fraction * min(lm, ln)
                            r_global = max(1, int(r_global))

                    if meta.global_shape is not None:
                        m_global, n_global = meta.global_shape
                        if config.has_tp_axis and self.tp_world_size > 1:
                            if config.inner_shard_tensor_dim == 0:
                                m_true_global = m_global * self.tp_world_size
                                n_true_global = n_global
                            elif config.inner_shard_tensor_dim == 1:
                                m_true_global = m_global
                                n_true_global = n_global * self.tp_world_size
                            else:
                                m_true_global, n_true_global = m_global, n_global
                        else:
                            m_true_global, n_true_global = m_global, n_global
                        config.compressed_all_reduce = ((m_true_global + n_true_global) * r_global < m_true_global * n_true_global)
                    else:
                        config.compressed_all_reduce = ((lm + ln) * r_global < lm * ln)
                else:
                    config.compressed_all_reduce = False

        elif param.ndim == 2:
            m, n = param.shape
            config.is_transposed = (m < n)
            if self.use_compressed_comm:
                r_state = self.state.get(param, {}).get('r', None)
                if r_state is not None:
                    r = int(r_state)
                else:
                    rank_fraction = self.defaults.get('rank_fraction', 1.0)
                    r = rank_fraction * min(m, n)
                    r = max(1, int(r))
                config.compressed_all_reduce = ((m + n) * r < m * n)
            else:
                config.compressed_all_reduce = False

        self._param_config[param] = config
        return config

    def _get_or_initialize_state(self, param: Tensor, group: dict) -> dict:
        """Get existing state or lazy-initialize it."""
        import hashlib

        param_uid = getattr(param, "_dion_param_uid", None)
        meta = self.dist_metas.get(param, None)
        if param_uid is None and meta is not None:
            param_uid = meta.param_uid

        if param not in self.state:
            if param_uid is not None:
                old_param = self._uid_to_param.get(param_uid)
                if old_param is not None and old_param is not param and old_param in self.state:
                    self.state[param] = self.state.pop(old_param)
                else:
                    self.state[param] = {}
                self._uid_to_param[param_uid] = param

                if param not in self.dist_metas and param_uid in self._dist_meta_by_uid:
                    self.dist_metas[param] = self._dist_meta_by_uid[param_uid]
            else:
                self.state[param] = {}

        state = self.state[param]
        if len(state) == 0:
            self._init_state(param, state, group)

        return state

    def _init_state(self, param: Tensor, state: Dict[str, Any], group: dict):
        """Initialize optimizer state with proper configuration."""
        import hashlib
        import math

        # Initialize momentum
        momentum_dtype = self._str_to_dtype(self._mixed_precision_config.momentum_dtype)
        if momentum_dtype is None:
            momentum_dtype = param.dtype
        state['momentum'] = torch.zeros_like(param, dtype=momentum_dtype)

        config = self._get_param_config(param)
        meta = self.dist_metas.get(param, None) if self.is_distributed_mode else None

        # Check if 2D parameter eligible for Dion
        if self.is_distributed_mode and meta:
            use_dion_flag = bool(meta and meta.is_dion_param)
            is_2d = use_dion_flag
            local_shape = self._infer_local_2d_shape(param, meta) if is_2d else None
            if local_shape:
                m, n = local_shape
            else:
                m, n = None, None
        else:
            is_2d = (param.ndim == 2)
            if is_2d:
                m, n = param.shape
            else:
                m, n = None, None

        # Initialize Dion state for 2D parameters
        algorithm = group.get('algorithm', 'dion')
        if algorithm == 'dion':
            if not is_2d or m is None or n is None:
                return

            rank_fraction = group.get('rank_fraction', self.defaults['rank_fraction'])
            rank_multiple_of = group.get('rank_multiple_of', self.defaults['rank_multiple_of'])

            m_global, n_global = self._get_global_shape(meta, m, n)

            # NOTE: meta.global_shape is already TP restored in distrib_optimizer_for_dion.py
            # (_build_gbuf_range_map applies: global_n = n * tp_world_size for tp_split_dim=1)
            # Do NOT restore TP again here - it was causing double application bug!
            m_true_global, n_true_global = m_global, n_global

            r_global = rank_fraction * min(m_true_global, n_true_global)
            r_global = rank_multiple_of * math.ceil(r_global / rank_multiple_of)
            r_global = min(r_global, m_true_global, n_true_global)
            r_global = max(1, int(r_global))

            q_base_local = m if config.is_transposed else n

            if config.has_tp_axis and self.tp_group is not None and self.tp_world_size > 1:
                # Ensure r_global is divisible by tp_world_size to avoid r_local=0
                if r_global < self.tp_world_size:
                    r_global = self.tp_world_size
                elif r_global % self.tp_world_size != 0:
                    r_global = self.tp_world_size * math.ceil(r_global / self.tp_world_size)
                r_local = r_global // self.tp_world_size
            else:
                r_local = r_global

            Q_shape = (q_base_local, r_local)
            r = r_global

            Q_dtype = self._str_to_dtype(self._mixed_precision_config.Q_dtype) or param.dtype

            if self.is_distributed_mode and meta:
                q_base_global = m_true_global if config.is_transposed else n_true_global
                fs_rank = self.fs_rank
                tp_rank = self.tp_rank

                param_uid = meta.param_uid if meta.param_uid else f"{meta.buffer_idx}_{meta.bucket_idx}"
                seed_str = f"{m_true_global}_{n_true_global}_{r_global}_{param_uid}_Q_init"
                seed = int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (2**32)
                gen = torch.Generator(device=param.device)
                gen.manual_seed(seed)

                Q_global_full = torch.randn((q_base_global, r_global), device=param.device, dtype=Q_dtype, generator=gen)

                fs_start = fs_rank * q_base_local
                fs_end = fs_start + q_base_local
                tp_start = tp_rank * r_local
                tp_end = tp_start + r_local
                Q = Q_global_full[fs_start:fs_end, tp_start:tp_end].contiguous()

                del Q_global_full

                if self.rp_group is not None and dist.get_world_size(self.rp_group) > 1:
                    rp_group_ranks = dist.get_process_group_ranks(self.rp_group)
                    dist.broadcast(Q, src=rp_group_ranks[0], group=self.rp_group)
            else:
                seed_str = f"{param.shape}_{param.dtype}_dion_Q"
                seed = int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (2**32)
                gen = torch.Generator(device=param.device)
                gen.manual_seed(seed)
                Q = torch.randn(Q_shape, device=param.device, dtype=Q_dtype, generator=gen)

            state['Q'] = Q
            state['r'] = r
            state['local_shape'] = (m, n)
            state['true_global_shape'] = (m_true_global, n_true_global)

    # Orthogonalization methods

    def _init_ortho_generator(self, device: torch.device):
        """Initialize orthogonalization RNG generator for this step."""
        if self._ortho_generator is None:
            self._ortho_generator = torch.Generator(device=device)

        seed = 42 + self._step_count
        self._ortho_generator.manual_seed(seed)

    def _generate_random_sketch_matrix(self, P: torch.Tensor, oversample: float = 1.25) -> torch.Tensor:
        """Generate random sketch matrix for Randomized Cholesky QR."""
        import hashlib
        import math

        assert P.ndim >= 2
        batch_shape = P.shape[:-2]
        m = P.size(-2)
        r = P.size(-1)

        k = math.ceil(oversample * r / 128.0) * 128
        std = math.sqrt(1.0 / k)

        if self.tp_group and dist.get_world_size(self.tp_group) > 1:
            if self.tp_rank == 0:
                seed_str = f"{tuple(P.shape)}_{tuple(batch_shape)}_{k}_{m}_{self._step_count}_sketch"
                seed = int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (2**31)
            else:
                seed = 0

            seed_tensor = torch.tensor([seed], device=P.device, dtype=torch.int64)
            tp_group_ranks = dist.get_process_group_ranks(self.tp_group)
            dist.broadcast(seed_tensor, src=tp_group_ranks[0], group=self.tp_group)
            seed = int(seed_tensor.item())

            generator = torch.Generator(device=P.device).manual_seed(seed)
            S = torch.empty((*batch_shape, k, m), device=P.device, dtype=torch.float32)
            S.normal_(std=std, generator=generator)
        else:
            S = torch.empty((*batch_shape, k, m), device=P.device, dtype=torch.float32)
            S.normal_(std=std)

        return S

    # TP Sync methods for Option C

    def _sync_shape_keys_across_tp_group(self, local_shape_keys: List[Tuple]) -> List[Tuple]:
        """
        Sync shape_keys across TP group to ensure all TP ranks process same batches.

        Results are cached since param structure doesn't change during training.
        """
        if self.tp_group is None:
            return local_shape_keys

        tp_world_size = dist.get_world_size(self.tp_group)
        if tp_world_size <= 1:
            return local_shape_keys

        # Cache: param structure doesn't change during training
        cache_key = tuple(local_shape_keys)
        if not hasattr(self, '_shape_key_cache'):
            self._shape_key_cache = {}
        if cache_key in self._shape_key_cache:
            return self._shape_key_cache[cache_key]

        local_keys_serialized = []
        for sk in local_shape_keys:
            local_shape = tuple(sk[0]) if isinstance(sk[0], (tuple, list)) else sk[0]
            key_data = (
                local_shape,
                bool(sk[1]),
                bool(sk[2]),
                bool(sk[3]),
                bool(sk[4]),
                int(sk[5]) if sk[5] is not None else -1,
                int(sk[6]) if sk[6] is not None else -1,
            )
            local_keys_serialized.append(key_data)

        gathered_keys_list = [None] * tp_world_size
        dist.all_gather_object(gathered_keys_list, local_keys_serialized, group=self.tp_group)

        all_keys_set = set()
        for rank_keys in gathered_keys_list:
            for key_data in rank_keys:
                all_keys_set.add(key_data)

        all_keys_sorted = sorted(all_keys_set)
        self._shape_key_cache[cache_key] = all_keys_sorted
        return all_keys_sorted

    def _batch_dion_dummy_collective(self, shape_key: Tuple, batch_size: int) -> Generator[None, None, None]:
        """
        Participate in TP collectives with dummy tensors when this rank has no params for a batch.

        When Distributed Optimizer shards params differently across TP ranks,
        some ranks may not have params for certain shape_keys. This function ensures those
        ranks still participate in TP collectives to prevent hangs.
        """
        if self.tp_group is None:
            return

        tp_world_size = dist.get_world_size(self.tp_group)
        if tp_world_size <= 1:
            return

        local_shape = shape_key[0]
        m, n = local_shape
        has_tp_axis = shape_key[2]
        is_transposed = shape_key[3]
        inner_shard_dim = shape_key[5] if shape_key[5] != -1 else None

        rank_fraction = self.defaults.get('rank_fraction', 1.0)
        rank_multiple_of = self.defaults.get('rank_multiple_of', 1)
        q_base_local = m if is_transposed else n

        # Compute r_global and r_local (matching _init_state logic)
        if inner_shard_dim == 0:
            m_true_global = m * tp_world_size
            n_true_global = n
        elif inner_shard_dim == 1:
            m_true_global = m
            n_true_global = n * tp_world_size
        else:
            m_true_global = m
            n_true_global = n

        r_global = int(rank_fraction * min(m_true_global, n_true_global))
        r_global = rank_multiple_of * math.ceil(r_global / rank_multiple_of)
        r_global = min(r_global, m_true_global, n_true_global)
        r_global = max(1, r_global)

        if r_global < tp_world_size:
            r_global = tp_world_size
        elif r_global % tp_world_size != 0:
            r_global = tp_world_size * math.ceil(r_global / tp_world_size)

        r_local = r_global // tp_world_size
        r_local = max(1, r_local)
        r_full = r_global

        device = torch.cuda.current_device()
        dtype = torch.float32

        # STEP 2: Q unshard via all_gather
        q_local = torch.zeros(q_base_local, r_local, device=device, dtype=dtype)
        gathered_list = [torch.zeros(q_base_local, r_local, device=device, dtype=dtype) for _ in range(tp_world_size)]

        for _ in range(batch_size):
            handle = dist.all_gather(gathered_list, q_local.contiguous(), group=self.tp_group, async_op=True)
            handle.wait()

        yield

        # STEP 4: _distributed_orthogonalize collectives
        p_dim = m if not is_transposed else n
        oversample = self.defaults.get('rcqr_oversample', 1.25)
        k = math.ceil(oversample * r_full / 128) * 128
        if k == 0:
            k = 128
        m_global = p_dim * tp_world_size

        S_full = torch.zeros(batch_size, k, m_global, device=device, dtype=dtype)
        tp_group_ranks = dist.get_process_group_ranks(self.tp_group)
        broadcast_src = tp_group_ranks[0]
        dist.broadcast(S_full, src=broadcast_src, group=self.tp_group)

        SP_batch = torch.zeros(batch_size, k, r_full, device=device, dtype=dtype)
        SP_slice = funcol.reduce_scatter_tensor(SP_batch, reduceOp="sum", scatter_dim=0, group=self.tp_group)
        slice_batch_size = SP_slice.size(0)

        R_slice = torch.zeros(slice_batch_size, r_full, r_full, device=device, dtype=dtype)
        R_full = funcol.all_gather_tensor(R_slice.contiguous(), gather_dim=0, group=self.tp_group)

        PP_local = torch.zeros(batch_size, r_full, r_full, device=device, dtype=dtype)
        PP_slice = funcol.reduce_scatter_tensor(PP_local, reduceOp="sum", scatter_dim=0, group=self.tp_group)

        R_slice2 = torch.zeros(PP_slice.size(0), r_full, r_full, device=device, dtype=dtype)
        R_full2 = funcol.all_gather_tensor(R_slice2.contiguous(), gather_dim=0, group=self.tp_group)

        yield

        # STEP 5: R all_reduce (conditional)
        R_batch = torch.zeros(batch_size, r_full, r_full, device=device, dtype=dtype)
        need_tp_R = has_tp_axis and (
            (not is_transposed and inner_shard_dim == 0) or
            (is_transposed and inner_shard_dim == 1)
        )
        if need_tp_R:
            tensors = [R_batch[i] for i in range(batch_size)]
            reduced = funcol.all_reduce_coalesced(tensors, reduceOp="sum", group=self.tp_group)
            yield

        del q_local, gathered_list, S_full, SP_batch, SP_slice
        del R_slice, R_full, PP_local, PP_slice, R_slice2, R_full2, R_batch

    def _orthogonalize(self, P: torch.Tensor, rcqr_oversample: float = 1.25) -> torch.Tensor:
        """Local orthogonalization with Randomized Cholesky QR."""
        # Use pure function with sketch_fn callback
        def sketch_fn(P_inner, oversample):
            return self._generate_random_sketch_matrix(P_inner, oversample)

        return orthogonalize(P, rcqr_oversample, sketch_fn)

    def _reshard_q_along_tp(self, Q: torch.Tensor, tp_group, tp_rank: int) -> torch.Tensor:
        """Re-shard Q matrix along TP dimension after update."""
        return reshard_q_along_tp(Q, tp_group, tp_rank)

    def _distributed_orthogonalize(
        self,
        P_batch: torch.Tensor,
        *,
        shard_group: Optional[torch.distributed.ProcessGroup],
        oversample: float = 1.25,
    ) -> torch.Tensor:
        """Distributed orthogonalization matching Algorithm 2."""
        import math

        batch_size = P_batch.size(0)
        m_shard_local = P_batch.size(1)
        r = P_batch.size(2)
        device = P_batch.device
        original_dtype = P_batch.dtype

        if shard_group is not None:
            shard_world_size = dist.get_world_size(shard_group)
            shard_rank = dist.get_rank(shard_group)
        else:
            shard_world_size = 1
            shard_rank = 0

        if shard_group is None or shard_world_size <= 1:
            result = torch.empty_like(P_batch)
            for i in range(batch_size):
                result[i] = self._orthogonalize(P_batch[i], rcqr_oversample=oversample)
            return result

        P_batch_fp32 = P_batch.float()

        k = math.ceil(oversample * r / 128) * 128
        if k == 0:
            # Fallback: minimum k=128 when r is very small
            k = 128
        std = math.sqrt(1.0 / k)

        if self._ortho_generator is not None and self._ortho_generator.device == device:
            gen = self._ortho_generator
        else:
            gen = torch.Generator(device=device)
            seed = 42 + self._step_count
            gen.manual_seed(seed)

        m_global = m_shard_local * shard_world_size
        S_full = torch.empty(batch_size, k, m_global, device=device, dtype=torch.float32)

        # PP-safe: use shard_group for broadcast, not default process group
        shard_group_ranks = dist.get_process_group_ranks(shard_group)
        broadcast_src = shard_group_ranks[0]
        if dist.get_rank() == broadcast_src:
            S_full.normal_(mean=0.0, std=std, generator=gen)
        dist.broadcast(S_full, src=broadcast_src, group=shard_group)

        m_start = shard_rank * m_shard_local
        m_end = m_start + m_shard_local
        S_batch = S_full[:, :, m_start:m_end].contiguous()
        del S_full

        SP_batch = S_batch @ P_batch_fp32
        del S_batch

        SP_slice = funcol.reduce_scatter_tensor(
            SP_batch, reduceOp="sum", scatter_dim=0, group=shard_group
        )
        del SP_batch

        _, R_slice = torch.linalg.qr(SP_slice, mode='r')
        del SP_slice

        R_full = funcol.all_gather_tensor(
            R_slice.contiguous(), gather_dim=0, group=shard_group
        )
        del R_slice

        P_batch_fp32 = torch.linalg.solve_triangular(
            R_full, P_batch_fp32, upper=True, left=False
        )

        PP_local = P_batch_fp32.mT @ P_batch_fp32

        PP_slice = funcol.reduce_scatter_tensor(
            PP_local, reduceOp="sum", scatter_dim=0, group=shard_group
        )
        del PP_local

        R_slice, info = torch.linalg.cholesky_ex(PP_slice, upper=True)
        del PP_slice, info

        R_full = funcol.all_gather_tensor(
            R_slice.contiguous(), gather_dim=0, group=shard_group
        )
        del R_slice

        P_batch_fp32 = torch.linalg.solve_triangular(
            R_full, P_batch_fp32, upper=True, left=False
        )
        del R_full

        return P_batch_fp32.to(original_dtype).contiguous()

    # Batch update methods

    def _batch_dion_update_async(self, params: List[Tensor], momentums: List[Tensor],
                                Qs: List[Tensor], configs: List[DionParamConfig],
                                metas: List, groups: List[dict],
                                grads: Optional[List[Tensor]] = None,
                                states: Optional[List[dict]] = None,
                                param_shapes: Optional[List[Tuple[int, int]]] = None,
                                real_batch_size: Optional[int] = None,
                                global_param_offset: int = 0) -> Generator[None, None, None]:
        """Perform batched Dion update with async communication."""
        batch_size = len(params)
        if real_batch_size is None:
            real_batch_size = batch_size

        use_compressed = self.use_compressed_comm and any(c.compressed_all_reduce for c in configs)

        # STEP 1: M <- M + G (no decay - decay is applied in error feedback)
        if grads:
            if momentums[0].dtype == grads[0].dtype:
                torch._foreach_add_(momentums, grads)
            else:
                for m, g in zip(momentums, grads):
                    m.add_(g.to(m.dtype))

        # STEP 2: Q Unshard
        Q_for_matmul = [None] * batch_size
        tp_unshard_indices = []
        for i in range(batch_size):
            if configs[i].has_tp_axis and self.tp_group is not None and self.tp_world_size > 1:
                tp_unshard_indices.append(i)
            else:
                Q_for_matmul[i] = Qs[i]

        if tp_unshard_indices:
            tp_size = self.tp_world_size
            async_handles = []

            for i in tp_unshard_indices:
                q = Qs[i]
                n, r_local = q.shape
                r_full = r_local * tp_size

                Q_full = torch.empty(n, r_full, dtype=q.dtype, device=q.device)
                gathered_list = [torch.empty(n, r_local, dtype=q.dtype, device=q.device)
                                for _ in range(tp_size)]

                handle = dist.all_gather(gathered_list, q.contiguous(), group=self.tp_group, async_op=True)
                async_handles.append((i, handle, gathered_list, Q_full, r_local))

            yield

            for i, handle, gathered_list, Q_full, r_local in async_handles:
                handle.wait()
                for j in range(tp_size):
                    Q_full[:, j*r_local:(j+1)*r_local].copy_(gathered_list[j])
                Q_for_matmul[i] = Q_full

        # STEP 3: P = M @ Q
        M_for_matmul = []
        for i, m in enumerate(momentums):
            M_for_matmul.append(m.mT.float() if configs[i].is_transposed else m.float())
        M_batch = torch.stack(M_for_matmul)
        Q_batch = torch.stack([q.float() for q in Q_for_matmul])
        del M_for_matmul, Q_for_matmul

        P_batch = M_batch @ Q_batch

        # STEP 3.5: All-Reduce for P (only if FS collectives are enabled)
        # NOTE: Expert params use different shard_group (EP-internal), so we group by shard_group
        if self.use_fs_collectives:
            need_fs = [i for i, c in enumerate(configs) if c.has_fs_axis and (
                (not c.is_transposed and getattr(c, 'outer_shard_tensor_dim', None) == 1) or
                (c.is_transposed and getattr(c, 'outer_shard_tensor_dim', None) == 0))]

            if need_fs:
                # Group params by their shard_group (expert vs dense use different groups)
                group_to_indices = {}
                for i in need_fs:
                    meta = metas[i] if i < len(metas) else None
                    shard_group = getattr(meta, 'shard_group', None) if meta else None
                    if shard_group is None:
                        shard_group = self.fs_group
                    group_id = id(shard_group) if shard_group else 0
                    if group_id not in group_to_indices:
                        group_to_indices[group_id] = (shard_group, [])
                    group_to_indices[group_id][1].append(i)

                # All-reduce for each group separately
                did_yield = False
                for group_id, (group, indices) in group_to_indices.items():
                    if group and dist.get_world_size(group) > 1:
                        tensors = [P_batch[idx] for idx in indices]
                        reduced = funcol.all_reduce_coalesced(tensors, reduceOp="sum", group=group)
                        if not did_yield:
                            yield
                            did_yield = True
                        for idx, t in zip(indices, reduced):
                            P_batch[idx].copy_(t)
                        del tensors, reduced
                if not did_yield and need_fs:
                    yield  # Ensure we yield at least once if there were params to process

        # STEP 4: Orthogonalize P
        if use_compressed:
            P_batch, R_batch = yield from self._compressed_communication_async(
                P_batch, M_batch, Q_batch, configs, metas
            )
        else:
            if self.tp_group and dist.get_world_size(self.tp_group) > 1:
                P_batch = self._distributed_orthogonalize(P_batch, shard_group=self.tp_group,
                                                         oversample=self.defaults['rcqr_oversample'])
            else:
                for i in range(batch_size):
                    P_batch[i] = self._orthogonalize(P_batch[i], rcqr_oversample=self.defaults['rcqr_oversample'])
            yield

            # STEP 5: R = M.T @ P
            R_batch = M_batch.mT @ P_batch

            if self.tp_group and dist.get_world_size(self.tp_group) > 1:
                need_tp_R = [i for i, c in enumerate(configs) if c.has_tp_axis and (
                    (not c.is_transposed and c.inner_shard_tensor_dim == 0) or
                    (c.is_transposed and c.inner_shard_tensor_dim == 1))]
                if need_tp_R:
                    tensors = [R_batch[i] for i in need_tp_R]
                    reduced = funcol.all_reduce_coalesced(tensors, reduceOp="sum", group=self.tp_group)
                    yield
                    for i, t in zip(need_tp_R, reduced):
                        R_batch[i].copy_(t)
                    del tensors, reduced

        # STEP 6: Fix NaN/zero
        P_batch, R_batch = self._fix_all_zero_or_nan_batch(P_batch, R_batch, Q_batch, M_batch)

        # STEP 7: Error feedback
        self._batch_error_feedback(momentums, P_batch, R_batch, configs, groups)

        # STEP 8: Column normalize R -> Q_new
        Q_new_batch = yield from self._batch_column_normalize_async(R_batch, configs, metas, real_batch_size, global_param_offset)

        # STEP 9: Apply weight updates and Q reshard
        # Batch compute delta = P @ Q_new^T using bmm (replaces N individual matmuls with 1)
        Q_new_f32 = Q_new_batch[:real_batch_size].float()
        P_for_delta = P_batch[:real_batch_size]
        is_transposed = configs[0].is_transposed
        if all(c.is_transposed == is_transposed for c in configs[:real_batch_size]):
            if is_transposed:
                delta_batch = torch.bmm(Q_new_f32, P_for_delta.transpose(1, 2))
            else:
                delta_batch = torch.bmm(P_for_delta, Q_new_f32.transpose(1, 2))
        else:
            delta_list = []
            for i in range(real_batch_size):
                if configs[i].is_transposed:
                    delta_list.append(torch.mm(Q_new_f32[i], P_for_delta[i].t()))
                else:
                    delta_list.append(torch.mm(P_for_delta[i], Q_new_f32[i].t()))
            delta_batch = torch.stack(delta_list)
            del delta_list
        del Q_new_f32, P_for_delta

        # Pre-compute scaled_lr (same global shape within a batch)
        state0 = states[0]
        if 'true_global_shape' in state0:
            m_global, n_global = state0['true_global_shape']
        else:
            m_global, n_global = self._get_global_shape(metas[0], param_shapes[0][0], param_shapes[0][1])
        lr = groups[0].get('lr', self.defaults['lr'])

        if self._lr_scaling == "dion_ref":
            scaled_lr = ((m_global / n_global) ** 0.5) * lr
        elif self._lr_scaling == "dion":
            rank_fraction = self.defaults.get('rank_fraction', 0.25)
            base_scale = 0.2 / (rank_fraction ** 0.5)
            scaled_lr = base_scale * (max(m_global, n_global) ** 0.5) * lr
        else:  # "moonlight"
            scaled_lr = 0.2 * (max(m_global, n_global) ** 0.5) * lr

        wd_mult = groups[0].get('wd_mult', 1.0)
        weight_decay = self.defaults['weight_decay'] * wd_mult
        has_tp = configs[0].has_tp_axis and self.tp_group is not None

        for i in range(real_batch_size):
            p = params[i]
            delta = delta_batch[i]
            if delta.shape != p.shape:
                delta = delta.contiguous().view(p.shape)

            if weight_decay > 0:
                p.mul_(1 - lr * weight_decay)
            p.add_(delta.to(p.dtype), alpha=-scaled_lr)

            Q_new = Q_new_batch[i].to(Qs[i].dtype)
            if has_tp:
                Q_new = self._reshard_q_along_tp(Q_new, self.tp_group, self.tp_rank)
            Qs[i].copy_(Q_new)

        del M_batch, Q_batch, P_batch, R_batch, Q_new_batch, delta_batch

    def _compressed_communication_async(
        self,
        P_batch: torch.Tensor,
        M_batch: torch.Tensor,
        Q_batch: torch.Tensor,
        configs: List[DionParamConfig],
        metas: List,
    ) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
        """Compressed communication for Dion optimizer."""
        comm_group = self.rp_group
        comm_world_size = dist.get_world_size(comm_group) if comm_group else 1

        if comm_group is None or comm_world_size <= 1:
            if self.tp_group and dist.get_world_size(self.tp_group) > 1:
                P_ortho = self._distributed_orthogonalize(P_batch, shard_group=self.tp_group,
                                                         oversample=self.defaults['rcqr_oversample'])
            else:
                P_ortho = P_batch.clone()
                for i in range(P_batch.size(0)):
                    P_ortho[i] = self._orthogonalize(P_batch[i], rcqr_oversample=self.defaults['rcqr_oversample'])
            yield

            R_list = []
            for i in range(P_ortho.size(0)):
                R = M_batch[i].mT @ P_ortho[i]
                R_list.append(R)
            R_batch = torch.stack(R_list)
            del R_list

            if self.tp_group and dist.get_world_size(self.tp_group) > 1:
                need_idx = [
                    i for i, c in enumerate(configs)
                    if c.has_tp_axis and (
                        (not c.is_transposed and c.inner_shard_tensor_dim == 0) or
                        (c.is_transposed and c.inner_shard_tensor_dim == 1)
                    )
                ]

                if need_idx:
                    tensors = [R_batch[i] for i in need_idx]
                    reduced = funcol.all_reduce_coalesced(tensors, reduceOp="sum", group=self.tp_group)
                    yield
                    for i, t in zip(need_idx, reduced):
                        R_batch[i].copy_(t)
                    del tensors, reduced

            return P_ortho, R_batch

        # Rest of compressed communication for RP > 1
        comm_rank = dist.get_rank(comm_group)
        batch_size = P_batch.size(0)

        batch_size_tensor = torch.tensor([batch_size], device=P_batch.device, dtype=torch.int64)
        all_batch_sizes = [torch.zeros_like(batch_size_tensor) for _ in range(comm_world_size)]
        dist.all_gather(all_batch_sizes, batch_size_tensor, group=comm_group)

        batch_sizes = [int(t.item()) for t in all_batch_sizes]
        del all_batch_sizes, batch_size_tensor

        original_batch_size = batch_size

        if batch_size % comm_world_size != 0:
            pad = comm_world_size - (batch_size % comm_world_size)
            m, r = P_batch.size(1), P_batch.size(2)

            P_pad = torch.zeros((pad, m, r), device=P_batch.device, dtype=P_batch.dtype)
            M_pad = torch.zeros((pad, M_batch.size(1), M_batch.size(2)), device=M_batch.device, dtype=M_batch.dtype)
            Q_pad = torch.zeros((pad, Q_batch.size(1), Q_batch.size(2)), device=Q_batch.device, dtype=Q_batch.dtype)

            P_batch = torch.cat([P_batch, P_pad], dim=0)
            M_batch = torch.cat([M_batch, M_pad], dim=0)
            Q_batch = torch.cat([Q_batch, Q_pad], dim=0)

            del P_pad, M_pad, Q_pad

            if isinstance(configs, list):
                dummy_cfg = DionParamConfig()
                configs = list(configs) + [dummy_cfg] * pad
            if isinstance(metas, list):
                metas = list(metas) + [None] * pad
            batch_size = P_batch.size(0)

        P_replicated = P_batch
        P_slice = P_replicated
        slice_configs = configs
        slice_size = P_slice.size(0)

        cfg = slice_configs[0] if slice_configs else DionParamConfig()

        p_is_tp_sharded = cfg.has_tp_axis and (
            (not cfg.is_transposed and cfg.inner_shard_tensor_dim == 0) or
            (cfg.is_transposed and cfg.inner_shard_tensor_dim == 1)
        )
        p_is_fs_sharded = (not p_is_tp_sharded) and cfg.has_fs_axis and (
            (not cfg.is_transposed and cfg.outer_shard_tensor_dim == 0) or
            (cfg.is_transposed and cfg.outer_shard_tensor_dim == 1)
        )

        if p_is_tp_sharded and self.tp_group and dist.get_world_size(self.tp_group) > 1:
            tp_world_size = dist.get_world_size(self.tp_group)
            P_ortho_list = []
            for chunk_start in range(0, slice_size, tp_world_size):
                chunk_end = min(chunk_start + tp_world_size, slice_size)
                chunk_size = chunk_end - chunk_start
                P_chunk = P_slice[chunk_start:chunk_end]

                original_chunk_size = chunk_size
                if chunk_size < tp_world_size:
                    pad_size = tp_world_size - chunk_size
                    m_local = P_chunk.size(1)
                    r_val = P_chunk.size(2)
                    P_pad = torch.empty(pad_size, m_local, r_val, device=P_chunk.device, dtype=P_chunk.dtype)
                    for i in range(pad_size):
                        torch.nn.init.orthogonal_(P_pad[i])
                        P_pad[i] *= 1e-6
                    P_chunk = torch.cat([P_chunk, P_pad], dim=0)
                    del P_pad

                P_chunk_ortho = self._distributed_orthogonalize(
                    P_chunk, shard_group=self.tp_group, oversample=self.defaults['rcqr_oversample']
                )
                P_ortho_list.append(P_chunk_ortho[:original_chunk_size])
                del P_chunk

            P_ortho_slice = torch.cat(P_ortho_list, dim=0) if P_ortho_list else P_slice
            del P_ortho_list

        elif p_is_fs_sharded and self.fs_group and dist.get_world_size(self.fs_group) > 1:
            fs_size = self.fs_size
            P_ortho_list = []
            for chunk_start in range(0, slice_size, fs_size):
                chunk_end = min(chunk_start + fs_size, slice_size)
                chunk_size = chunk_end - chunk_start
                P_chunk = P_slice[chunk_start:chunk_end]

                original_chunk_size = chunk_size
                if chunk_size < fs_size:
                    pad_size = fs_size - chunk_size
                    m_local = P_chunk.size(1)
                    r_val = P_chunk.size(2)
                    P_pad = torch.empty(pad_size, m_local, r_val, device=P_chunk.device, dtype=P_chunk.dtype)
                    for i in range(pad_size):
                        torch.nn.init.orthogonal_(P_pad[i])
                        P_pad[i] *= 1e-6
                    P_chunk = torch.cat([P_chunk, P_pad], dim=0)
                    del P_pad

                P_chunk_ortho = self._distributed_orthogonalize(
                    P_chunk, shard_group=self.fs_group, oversample=self.defaults['rcqr_oversample']
                )
                P_ortho_list.append(P_chunk_ortho[:original_chunk_size])
                del P_chunk

            P_ortho_slice = torch.cat(P_ortho_list, dim=0) if P_ortho_list else P_slice
            del P_ortho_list

        elif slice_size > 0:
            P_ortho_slice = torch.empty_like(P_slice, dtype=torch.float32)
            for i in range(slice_size):
                P_ortho_slice[i] = self._orthogonalize(P_slice[i], rcqr_oversample=self.defaults['rcqr_oversample'])
            P_ortho_slice = P_ortho_slice.to(P_slice.dtype)
        else:
            P_ortho_slice = P_slice

        P_ortho_full = P_ortho_slice

        R_list = []
        for i in range(batch_size):
            R = M_batch[i].mT @ P_ortho_full[i]
            R_list.append(R)
        R_batch = torch.stack(R_list)
        del R_list

        if self.tp_group and dist.get_world_size(self.tp_group) > 1:
            need_idx = [
                i for i, c in enumerate(configs)
                if c.has_tp_axis and (
                    (not c.is_transposed and c.inner_shard_tensor_dim == 0) or
                    (c.is_transposed and c.inner_shard_tensor_dim == 1)
                )
            ]

            if need_idx:
                tensors = [R_batch[i] for i in need_idx]
                reduced = funcol.all_reduce_coalesced(tensors, reduceOp="sum", group=self.tp_group)
                yield
                for i, t in zip(need_idx, reduced):
                    R_batch[i].copy_(t)
                del tensors, reduced

        P_ortho_full = P_ortho_full[:original_batch_size]
        R_batch = R_batch[:original_batch_size]

        return P_ortho_full, R_batch

    def _fit_Q_to_R_batch(self, Q_batch: Tensor, R_batch: Tensor) -> Tensor:
        """Fit Q to match R's rank dimension for adaptive rank recovery."""
        batch_size = Q_batch.size(0)
        base_dim = Q_batch.size(1)
        r_Q = Q_batch.size(2)
        r_R = R_batch.size(2)

        if r_R == r_Q:
            return Q_batch

        Q_fitted = torch.zeros(batch_size, base_dim, r_R, device=Q_batch.device, dtype=Q_batch.dtype)
        base = min(r_R, r_Q)
        Q_fitted[..., :base].copy_(Q_batch[..., :base])

        if r_R > base:
            add_cols = r_R - base
            add = torch.empty(batch_size, base_dim, add_cols, device=Q_batch.device, dtype=torch.float32)

            gen = torch.Generator(device=Q_batch.device)
            gen.manual_seed(0xD10F33D)
            add.normal_(mean=0.0, std=1.0, generator=gen)

            Q_cat = torch.cat([Q_fitted.to(torch.float32), add], dim=-1)
            Q_ortho, _ = torch.linalg.qr(Q_cat, mode='reduced')
            Q_fitted = Q_ortho.to(Q_batch.dtype).contiguous()

            del add, Q_cat, Q_ortho

        return Q_fitted

    def _fix_all_zero_or_nan_batch(self, P_batch: Tensor, R_batch: Tensor,
                                   Q_batch: Tensor, M_batch: Tensor) -> Tuple[Tensor, Tensor]:
        """Batch NaN/zero handling for numerical stability."""
        is_all_zero = (M_batch == 0).all(dim=(-2, -1), keepdim=True)
        not_all_zero = ~is_all_zero

        P_batch = P_batch.nan_to_num() * not_all_zero

        Q_clean = Q_batch.nan_to_num()
        Q_fitted = self._fit_Q_to_R_batch(Q_clean, R_batch)
        R_batch = R_batch.nan_to_num() * not_all_zero + Q_fitted * is_all_zero

        return P_batch, R_batch

    def _foreach_baddbmm_(self, X: List[Tensor], A: Tensor, B: Tensor,
                         alpha: float = 1.0, beta: float = 1.0,
                         transpose: bool = False):
        """Batch matrix multiplication and in-place addition."""
        assert A.size(0) == B.size(0) == len(X)

        if not transpose:
            update = A @ B.mT
        else:
            update = B @ A.mT

        update = update.unbind(dim=0)
        update = torch._foreach_mul(update, alpha)
        torch._foreach_mul_(X, beta)
        torch._foreach_add_(X, update)

        del update

    def _batch_error_feedback(self, momentums: List[Tensor], P_batch: Tensor,
                             R_batch: Tensor, configs: List[DionParamConfig],
                             groups: List[dict]):
        """Apply error feedback to momentum: M = M - (1-mu) * (P @ R^T)."""
        # Get mu value (same for all params in this batch)
        mu = groups[0].get('mu', self.defaults['mu'])

        is_transposed = configs[0].is_transposed
        if all(c.is_transposed == is_transposed for c in configs):
            self._foreach_baddbmm_(
                momentums, P_batch, R_batch,
                alpha=-(1.0 - mu), beta=1.0,
                transpose=is_transposed
            )
        else:
            for i, momentum in enumerate(momentums):
                if configs[i].is_transposed:
                    update = R_batch[i] @ P_batch[i].t()
                else:
                    update = P_batch[i] @ R_batch[i].t()

                momentum.add_(update, alpha=-(1.0 - mu))
                del update

    def _batch_column_normalize_async(self, R_batch: Tensor, configs: List[DionParamConfig],
                                     metas: List, real_batch_size: int = None,
                                     global_param_offset: int = 0) -> Generator[Tensor, None, None]:
        """Async batch column normalization with yields for communication."""
        batch_size = R_batch.shape[0]
        if real_batch_size is None:
            real_batch_size = batch_size

        col_sum_sq_list = []
        to_reduce_indices = []
        to_reduce_tensors = []

        for i in range(batch_size):
            R = R_batch[i]
            config = configs[i]

            col_sum_sq = (R * R).sum(dim=0, keepdim=True)

            fs_world_ok = (self.use_fs_collectives and self.fs_group is not None and dist.get_world_size(self.fs_group) > 1)
            need_allreduce = (config.has_fs_axis and fs_world_ok)

            if need_allreduce:
                to_reduce_indices.append(i)
                to_reduce_tensors.append(col_sum_sq)

            col_sum_sq_list.append(col_sum_sq)

        if self.use_fs_collectives and self.fs_group is not None and dist.get_world_size(self.fs_group) > 1:
            num_to_reduce = torch.tensor([len(to_reduce_tensors)], device=R_batch.device, dtype=torch.int64)
            dist.all_reduce(num_to_reduce, op=dist.ReduceOp.MAX, group=self.fs_group)
            max_to_reduce = int(num_to_reduce.item())
            del num_to_reduce

            while len(to_reduce_tensors) < max_to_reduce:
                to_reduce_tensors.append(torch.zeros_like(to_reduce_tensors[0] if to_reduce_tensors else col_sum_sq_list[0]))

            if to_reduce_tensors:
                reduced = funcol.all_reduce_coalesced(to_reduce_tensors, reduceOp="sum", group=self.fs_group)
                yield

                for j, i in enumerate(to_reduce_indices):
                    col_sum_sq_list[i] = reduced[j]

                del to_reduce_tensors, reduced

        Q_new_list = []
        epsilon = self.defaults['epsilon']
        for i in range(batch_size):
            col_sum_sq = col_sum_sq_list[i]
            R = R_batch[i]
            col_norms = col_sum_sq.sqrt()
            Q_new = R / (col_norms + epsilon)

            Q_new_list.append(Q_new)
            del col_norms

        result = torch.stack(Q_new_list)
        del Q_new_list, col_sum_sq_list

        return result
