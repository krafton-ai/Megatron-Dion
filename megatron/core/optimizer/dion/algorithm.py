"""Dion optimizer main class for Megatron-LM."""

import logging
import hashlib
import math
import os
from collections import Counter
from typing import Any, Dict, Generator, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
from torch import Tensor
from torch.optim.optimizer import Optimizer

from megatron.core import parallel_state

# Diagnostic logging for debugging parallelism numerical issues.
# Enable with: DION_DIAG=1  (logs at first 3 steps only to avoid overhead)
_DION_DIAG = os.environ.get("DION_DIAG", "0") == "1"
_DION_DIAG_MAX_STEPS = int(os.environ.get("DION_DIAG_STEPS", "3"))
_DION_EXPERT_ADAMW = os.environ.get("DION_EXPERT_ADAMW", "0") == "1"

from .async_runtime import AsyncRuntime, AsyncTask
from .batching import BatchProcessor, pad_batch
from .constants import (
    DEFAULT_LR,
    DEFAULT_MAX_CONCURRENT_TASKS,
    DEFAULT_MU,
    DEFAULT_WEIGHT_DECAY,
)
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
        state_replica_group=None,  # Optimizer-state replicas for the same local shard
        # Configuration flags
        rp_average_in_collective: bool = True,
        use_fs_collectives: bool = True,  # Enable FS collectives in Distributed mode
        mixed_precision_config: Optional[DionMixedPrecisionConfig] = None,
        enable_async: bool = True,  # Enable async execution where possible
        use_compressed_comm: bool = True,  # Enable compressed communication
        scalar_optimizer: str = "adamw",  # Scalar optimizer for non-Dion params ("adamw" or "lion")
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
            # Reference implementation uses only RCQR and standard scaling
            algorithm="dion",  # Default algorithm, same as original dion.py
            step=0,  # Per-group step counter
        )
        super().__init__(params, defaults)

        # Cache global rank for logging (avoid repeated dist.get_rank() calls)
        self._global_rank = dist.get_rank() if dist.is_initialized() else 0

        global_rank = self._global_rank

        # Store the correct process groups
        self.tp_group = tp_group  # Tensor Parallel
        self.rp_group = rp_group  # RP: replicas with same shard
        self.fs_group = fs_group  # FS: shards within same replica
        self.state_replica_group = state_replica_group

        # Compute world sizes and ranks
        self.tp_world_size = dist.get_world_size(tp_group) if tp_group else 1
        self.rp_world_size = dist.get_world_size(rp_group) if rp_group else 1
        self.fs_size = dist.get_world_size(fs_group) if fs_group else -1  # Use -1 to indicate not set
        self.state_replica_world_size = (
            dist.get_world_size(state_replica_group) if state_replica_group else 1
        )

        self.tp_rank = dist.get_rank(tp_group) if tp_group else 0
        self.rp_rank = dist.get_rank(rp_group) if rp_group else 0
        self.fs_rank = dist.get_rank(fs_group) if fs_group else 0
        self.state_replica_rank = dist.get_rank(state_replica_group) if state_replica_group else 0

        # Configuration storage
        self._param_config: Dict[Tensor, DionParamConfig] = {}
        self.dist_metas = {}
        self.full_data_parallel_group = None
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

        # EP mode: use sync execution to avoid collective desync across ranks
        # When EP > 1, different optimizer instances (Dense vs Expert) have different
        # fs_group compositions, and AsyncRuntime can cause ranks to reach different
        # yield points, leading to collective hangs.
        self.is_ep_mode = is_ep_mode

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

        self._scratch_buffers: Dict[str, torch.Tensor] = {}

    def _compressed_replicate_group(self):
        """Return the true Dion compressed replicate group.

        In the Megatron-Core backend, CP is part of the standard gradient-construction
        contract, not the Dion compressed replicate mesh. Compressed P/R collapse should
        therefore run only over explicit RP replicas.
        """
        if self.rp_group is None:
            return None
        return self.rp_group if dist.get_world_size(self.rp_group) > 1 else None

    def _broadcast_replicate_domain_(self, tensor: Tensor) -> None:
        """Broadcast optimizer state across the true Dion replicate domain.

        Optimizer-state replicas arising from stock partial distributed optimizer
        instances cannot be treated as a global batch/state sync group. The same
        inter-instance group can carry different parameter families, so only true
        Dion RP replicas participate here.
        """
        if self.rp_group is None or dist.get_world_size(self.rp_group) <= 1:
            return

        group_ranks = dist.get_process_group_ranks(self.rp_group)
        dist.broadcast(tensor, src=group_ranks[0], group=self.rp_group)

    def _broadcast_state_replicas_(self, tensor: Tensor) -> None:
        """Broadcast optimizer state across stock DO state replicas for one local shard."""
        if self.state_replica_group is None or self.state_replica_world_size <= 1:
            return

        group_ranks = dist.get_process_group_ranks(self.state_replica_group)
        dist.broadcast(tensor, src=group_ranks[0], group=self.state_replica_group)

    def _seeded_sketch_fn(
        self,
        *,
        metas: Optional[List],
        tag: str,
    ):
        """Build a topology-independent sketch generator for one logical Dion update batch.

        The sketch for one logical parameter must not depend on which other parameters
        were batched with it. FS/TP topology changes alter batch composition, and a
        batch-global RNG seed would therefore make the same logical parameter see a
        different sketch across topologies.
        """
        if not metas:
            return None

        logical_ids = []
        for meta in metas:
            if meta is None:
                logical_ids.append(None)
                continue
            logical_ids.append(
                (
                    getattr(meta, "param_uid", None),
                    getattr(meta, "param_name", ""),
                )
            )

        def _make_sketch(P: Tensor, oversample: float) -> Tensor:
            batch_shape = P.shape[:-2]
            if len(batch_shape) == 0:
                batch = 1
            elif len(batch_shape) == 1:
                batch = batch_shape[0]
            else:
                raise RuntimeError(
                    "[DION_INVALID_SKETCH_BATCH] "
                    f"tag={tag} expected batched 3D tensor, got shape={tuple(P.shape)}"
                )
            if batch != len(logical_ids):
                raise RuntimeError(
                    "[DION_SKETCH_META_MISMATCH] "
                    f"tag={tag} batch={batch} logical_ids={len(logical_ids)}"
                )
            m = P.size(-2)
            r = P.size(-1)
            k = math.ceil(oversample * r / 128.0) * 128
            if k <= 0:
                raise RuntimeError(
                    f"[DION_INVALID_SKETCH_RANK] tag={tag} r={r} oversample={oversample} k={k}"
                )
            std = math.sqrt(1.0 / k)
            if batch == 1 and len(batch_shape) == 0:
                logical_id = logical_ids[0]
                seed_key = repr(
                    (
                        tag,
                        self._step_count,
                        logical_id,
                    )
                ).encode("utf-8")
                seed = int.from_bytes(
                    hashlib.blake2b(seed_key, digest_size=8).digest(), "little"
                ) & ((1 << 63) - 1)
                gen = torch.Generator(device=P.device)
                gen.manual_seed(seed)
                sketch = torch.empty((k, m), device=P.device, dtype=P.dtype)
                sketch.normal_(mean=0.0, std=std, generator=gen)
                return sketch
            sketch = torch.empty((batch, k, m), device=P.device, dtype=P.dtype)
            for idx, logical_id in enumerate(logical_ids):
                seed_key = repr(
                    (
                        tag,
                        self._step_count,
                        logical_id,
                    )
                ).encode("utf-8")
                seed = int.from_bytes(
                    hashlib.blake2b(seed_key, digest_size=8).digest(), "little"
                ) & ((1 << 63) - 1)
                gen = torch.Generator(device=P.device)
                gen.manual_seed(seed)
                sketch[idx].normal_(
                    mean=0.0,
                    std=std,
                    generator=gen,
                )
            return sketch

        return _make_sketch

    def _sync_state_replica_q_init_(self, dion_params: List[Tuple]) -> None:
        """Synchronize newly initialized Q state in canonical param_uid order.

        Lazy state init happens while scanning optimizer param groups, whose local order is
        not a valid cross-rank contract for stock DO state replicas. After we sort Dion
        params by param_uid, the same local shard appears in the same order on all state
        replicas; only here is it safe to synchronize Q.
        """
        if self.state_replica_group is None or self.state_replica_world_size <= 1:
            return

        for p, grad, state, group, config, meta in dion_params:
            if not state.get("_needs_state_replica_q_sync", False):
                continue
            Q = state.get("Q", None)
            if Q is None:
                raise RuntimeError(
                    "[DION_STATE_REPLICA_Q_SYNC_MISSING_Q] "
                    f"param={getattr(meta, 'param_name', '') if meta is not None else ''}"
                )
            self._broadcast_state_replicas_(Q)
            state["_needs_state_replica_q_sync"] = False

    def _replicate_reduce_op(self):
        """Replicate-domain collapse must match `dion_reference.py` semantics."""
        return dist.ReduceOp.AVG

    def _collapse_grads_across_cp(
        self,
        grads: List[Tensor],
    ) -> Generator[None, None, None]:
        """Collapse true Dion replicate replicas on G for non-compressed batches."""
        replicate_group = self._compressed_replicate_group()
        if replicate_group is None or not grads:
            return

        op = self._replicate_reduce_op()
        for grad in grads:
            dist.all_reduce(grad, op=op, group=replicate_group)
        yield

    def _collapse_batch_across_cp(
        self,
        batch: Tensor,
    ) -> Generator[None, None, None]:
        """Collapse true Dion replicate replicas on a batched tensor."""
        replicate_group = self._compressed_replicate_group()
        if replicate_group is None:
            return

        dist.all_reduce(batch, op=self._replicate_reduce_op(), group=replicate_group)
        yield

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
        for group in self.param_groups:
            group['step'] = group.get('step', 0) + 1

        self._step_count += 1
        pp_world_size = parallel_state.get_pipeline_model_parallel_world_size()
        rank = dist.get_rank() if dist.is_initialized() else 0
        if pp_world_size > 1:
            logger.info("[DION_PP_DEBUG] rank=%s inner_step enter step=%s", rank, self._step_count)

        # Create async tasks for optimization
        if self.is_distributed_mode:
            dion_tasks = self._iter_dist_tasks()
        else:
            dion_tasks = self._iter_local_tasks()

        # Execute all tasks
        # EP mode uses max_concurrent_tasks=1 to ensure all ranks execute collectives
        # in the same order, preventing async desync hangs.
        task_count = 0
        if dion_tasks:
            dion_tasks_list = list(dion_tasks)
            task_count = len(dion_tasks_list)
            if pp_world_size > 1:
                logger.info("[DION_PP_DEBUG] rank=%s inner_step tasks=%s", rank, task_count)
            if task_count > 0:
                max_tasks = 1 if self.is_ep_mode else DEFAULT_MAX_CONCURRENT_TASKS
                if pp_world_size > 1:
                    logger.info(
                        "[DION_PP_DEBUG] rank=%s inner_step before runtime_run max_tasks=%s",
                        rank,
                        max_tasks,
                    )
                runtime = AsyncRuntime((t for t in dion_tasks_list), max_concurrent_tasks=max_tasks)
                runtime.run()
                if pp_world_size > 1:
                    logger.info("[DION_PP_DEBUG] rank=%s inner_step after runtime_run", rank)

                del runtime
            # Always delete dion_tasks_list if it was created
            del dion_tasks_list

        if pp_world_size > 1:
            logger.info("[DION_PP_DEBUG] rank=%s inner_step exit", rank)
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
        self._scratch_buffers.clear()

        torch.cuda.empty_cache()

    def _scratch_tensor(
        self,
        name: str,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
        *,
        zero: bool = False,
    ) -> torch.Tensor:
        """Return a reusable scratch tensor for Dion batch-update workspace."""
        shape = tuple(shape)
        scratch = self._scratch_buffers.get(name)
        if (
            scratch is None
            or tuple(scratch.shape) != shape
            or scratch.dtype != dtype
            or scratch.device != device
        ):
            scratch = torch.empty(shape, dtype=dtype, device=device)
            self._scratch_buffers[name] = scratch
        if zero:
            scratch.zero_()
        return scratch

    def _iter_dist_tasks(self) -> Generator[AsyncTask, None, None]:
        """Create async tasks for Distributed mode optimization."""
        # Scalar/non-Dion params continue to use standard DO bucket ownership.
        #
        # Dion params must use a canonical local-shard order in distributed mode.
        # With multiple distributed-optimizer instances, the same local shard can
        # exist in several optimizer-state replicas. Those replicas must consume
        # identical Dion batch/RNG order. Optimizer param-group order is not a
        # valid cross-rank contract here; use stock DO local-shard identity
        # (`param_uid`) instead.
        scalar_buckets = {}
        unbuffered_scalar_params = []
        dion_params = []

        for group in self.param_groups:
            for p in group['params']:
                grad = self._get_param_grad(p)
                if grad is None:
                    continue

                # Initialize state if needed (lazy initialization)
                state = self._get_or_initialize_state(p, group)
                config = self._get_param_config(p)
                meta = self.dist_metas.get(p, None)

                if self._use_dion_update(p, state, group, meta):
                    dion_params.append((p, grad, state, group, config, meta))
                    self._dion_update_count += 1
                    continue

                self._adamw_update_count += 1
                if p in self.buffer_indices:
                    buf_idx, bucket_idx = self.buffer_indices[p]
                    key = (buf_idx, bucket_idx)
                    if key not in scalar_buckets:
                        scalar_buckets[key] = []
                    scalar_buckets[key].append((p, grad, state, group))
                else:
                    unbuffered_scalar_params.append((p, grad, state, group))

        if dion_params:
            ordered_dion_params = []
            for item in dion_params:
                p, grad, state, group, config, meta = item
                param_uid = getattr(meta, "param_uid", None) if meta is not None else None
                if param_uid is None:
                    raise RuntimeError(
                        "[DION_MISSING_PARAM_UID] distributed Dion param is missing param_uid: "
                        f"name={getattr(meta, 'param_name', '') if meta is not None else ''} "
                        f"shape={tuple(p.shape)}"
                    )
                ordered_dion_params.append((param_uid, item))
            ordered_dion_params.sort(key=lambda entry: entry[0])
            dion_params = [item for _, item in ordered_dion_params]
            self._sync_state_replica_q_init_(dion_params)
            self._diag_state_replica_task_order_(dion_params)
            self._diag_state_replica_q_by_uid_(dion_params)

        if dion_params:
            yield AsyncTask(self._run_dion_batch_async(dion_params))

        for (buf_idx, bucket_idx), params in sorted(scalar_buckets.items()):
            yield AsyncTask(self._run_scalar_bucket_async(params, buf_idx, bucket_idx))

        if unbuffered_scalar_params:
            yield AsyncTask(self._run_scalar_bucket_async(unbuffered_scalar_params, -1, -1))

    def _diag_state_replica_task_order_(self, dion_params: List[Tuple]) -> None:
        """Prove whether inter-instance state replicas see the same ordered Dion task list."""
        if os.getenv("DION_STATE_REPLICA_TASK_DIAG", "0") != "1":
            return
        if self.state_replica_group is None or self.state_replica_world_size <= 1:
            return

        local_ids = []
        for p, grad, state, group, config, meta in dion_params:
            local_ids.append(
                (
                    getattr(meta, "param_name", "") if meta is not None else "",
                    getattr(meta, "param_uid", None) if meta is not None else None,
                )
            )

        gathered_ids = [None] * self.state_replica_world_size
        dist.all_gather_object(gathered_ids, local_ids, group=self.state_replica_group)

        canonical_ids = tuple(gathered_ids[0])
        mismatch_ranks = [
            idx for idx, rank_ids in enumerate(gathered_ids) if tuple(rank_ids) != canonical_ids
        ]
        try:
            group_ranks = dist.get_process_group_ranks(self.state_replica_group)
        except Exception:
            group_ranks = []

        if mismatch_ranks:
            logger.error(
                "[DION_STATE_REPLICA_TASK_ORDER_MISMATCH] rank=%s group_ranks=%s "
                "mismatch_local_ranks=%s local_ids=%s gathered_ids=%s",
                dist.get_rank() if dist.is_initialized() else 0,
                group_ranks,
                mismatch_ranks,
                local_ids,
                gathered_ids,
            )
            raise RuntimeError(
                "[DION_STATE_REPLICA_TASK_ORDER_MISMATCH] "
                f"group_ranks={group_ranks} mismatch_local_ranks={mismatch_ranks}"
            )

        logger.info(
            "[DION_STATE_REPLICA_TASK_ORDER_OK] rank=%s group_ranks=%s num_params=%s",
            dist.get_rank() if dist.is_initialized() else 0,
            group_ranks,
            len(local_ids),
        )

    def _diag_state_replica_q_by_uid_(self, dion_params: List[Tuple]) -> None:
        """Compare initialized Q state by param_uid across optimizer-state replicas."""
        if os.getenv("DION_STATE_REPLICA_UID_DIAG", "0") != "1":
            return
        if self.state_replica_group is None or self.state_replica_world_size <= 1:
            return

        limit = int(os.getenv("DION_STATE_REPLICA_UID_DIAG_LIMIT", "32"))
        local_entries = []
        for p, grad, state, group, config, meta in dion_params:
            if len(local_entries) >= limit:
                break
            q = state.get("Q", None)
            if q is None:
                continue
            local_entries.append(
                (
                    getattr(meta, "param_uid", None) if meta is not None else None,
                    getattr(meta, "param_name", "") if meta is not None else "",
                    tuple(q.shape),
                    float(q.float().sum().item()),
                    float(q.float().abs().sum().item()),
                    float((q.float() ** 2).sum().item()),
                    float(q.float().abs().max().item()),
                )
            )

        if not local_entries:
            return

        gathered = [None] * self.state_replica_world_size
        dist.all_gather_object(gathered, local_entries, group=self.state_replica_group)
        canonical = tuple(gathered[0])
        mismatch_ranks = [idx for idx, entries in enumerate(gathered) if tuple(entries) != canonical]
        try:
            group_ranks = dist.get_process_group_ranks(self.state_replica_group)
        except Exception:
            group_ranks = []
        if mismatch_ranks:
            logger.error(
                "[DION_STATE_REPLICA_Q_UID_MISMATCH] rank=%s group_ranks=%s mismatch_local_ranks=%s "
                "local_entries=%s gathered=%s",
                dist.get_rank() if dist.is_initialized() else 0,
                group_ranks,
                mismatch_ranks,
                local_entries,
                gathered,
            )
            raise RuntimeError(
                "[DION_STATE_REPLICA_Q_UID_MISMATCH] "
                f"group_ranks={group_ranks} mismatch_local_ranks={mismatch_ranks}"
            )

        logger.info(
            "[DION_STATE_REPLICA_Q_UID_OK] rank=%s group_ranks=%s num_entries=%s",
            dist.get_rank() if dist.is_initialized() else 0,
            group_ranks,
            len(local_entries),
        )

    def _iter_local_tasks(self) -> Generator[AsyncTask, None, None]:
        """Create async tasks for regular mode optimization."""
        for group in self.param_groups:
            params_with_grad = []
            for p in group['params']:
                grad = self._get_param_grad(p)
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

                yield AsyncTask(self._run_local_param_batch_async(batch_params, group, batch_configs))

    def _run_bucket_batch_async(self, params: List[Tuple[Tensor, dict]],
                                   buf_idx: int, bucket_idx: int) -> Generator[None, None, None]:
        """Process bucket with async batch operations for mathematical equivalence."""
        pp_world_size = parallel_state.get_pipeline_model_parallel_world_size()
        rank = dist.get_rank() if dist.is_initialized() else 0
        if pp_world_size > 1:
            logger.info(
                "[DION_PP_DEBUG] rank=%s bucket_task enter buf=%s bucket=%s params=%s",
                rank,
                buf_idx,
                bucket_idx,
                len(params),
            )
        # Preserve incoming bucket order to match reference param/RNG order.
        params_sorted = params

        # Classify parameters: Dion (2D with Q) vs AdamW (1D or no Q)
        dion_params = []
        adamw_params = []

        for p, group in params_sorted:
            grad = self._get_param_grad(p)
            if grad is None:
                continue

            # Lookup state - MUST EXIST (initialized in _iter_dist_tasks)
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

            # Debug: DION_EXPERT_ADAMW=1 forces expert params to use AdamW
            if use_dion and _DION_EXPERT_ADAMW and meta and getattr(meta, 'is_expert', False):
                use_dion = False

            if (
                pp_world_size > 1
                and meta is not None
                and is_dion_marked
                and is_2d_global
                and 'Q' not in state
            ):
                logger.error(
                    "[DION_PP_DEBUG] rank=%s missing_Q name=%s uid=%s shape=%s bucket=(%s,%s) "
                    "algo=%s state_keys=%s",
                    rank,
                    getattr(meta, "param_name", "") or f"id_{id(p)}",
                    getattr(meta, "param_uid", None),
                    tuple(meta.global_shape) if getattr(meta, "global_shape", None) is not None else tuple(p.shape),
                    getattr(meta, "buffer_idx", None),
                    getattr(meta, "bucket_idx", None),
                    algorithm,
                    sorted(state.keys()),
                )

            if use_dion:
                dion_params.append((p, grad, state, group, config, meta))
                self._dion_update_count += 1
            else:
                adamw_params.append((p, grad, state, group))
                self._adamw_update_count += 1

        if pp_world_size > 1:
            logger.info(
                "[DION_PP_DEBUG] rank=%s bucket_task classify buf=%s bucket=%s dion=%s adamw=%s",
                rank,
                buf_idx,
                bucket_idx,
                len(dion_params),
                len(adamw_params),
            )
            if buf_idx == 0 and bucket_idx == 0 and rank < 4:
                adamw_2d = []
                for p, grad, state, group in adamw_params:
                    meta = self.dist_metas.get(p, None)
                    if p.ndim != 2:
                        continue
                    adamw_2d.append(
                        (
                            getattr(meta, "param_name", "") or f"id_{id(p)}",
                            tuple(meta.global_shape) if getattr(meta, "global_shape", None) is not None else tuple(p.shape),
                            bool(getattr(meta, "is_dion_param", False)) if meta is not None else False,
                            'Q' in state,
                        )
                    )
                logger.info(
                    "[DION_PP_DEBUG] rank=%s bucket0 adamw_2d=%s",
                    rank,
                    adamw_2d,
                )

        # Process Dion parameters in batches with async
        if dion_params:
            yield from self._run_dion_batch_async(dion_params)

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
        if pp_world_size > 1:
            logger.info(
                "[DION_PP_DEBUG] rank=%s bucket_task exit buf=%s bucket=%s",
                rank,
                buf_idx,
                bucket_idx,
            )

    def _use_dion_update(
        self,
        p: Tensor,
        state: Dict,
        group: Dict,
        meta,
    ) -> bool:
        """Return whether a param should follow the Dion update path."""
        algorithm = group.get('algorithm', 'dion')
        is_dion_marked = meta.is_dion_param if meta else False

        is_2d_global = False
        if meta and meta.global_shape and len(meta.global_shape) == 2:
            is_2d_global = True
        elif p.ndim == 2:
            is_2d_global = True

        use_dion = (
            algorithm == 'dion'
            and is_dion_marked
            and is_2d_global
            and 'Q' in state
        )

        if use_dion and _DION_EXPERT_ADAMW and meta and getattr(meta, 'is_expert', False):
            use_dion = False
        return use_dion

    def _run_scalar_bucket_async(
        self,
        scalar_params: List[Tuple[Tensor, Tensor, Dict, Dict]],
        buf_idx: int,
        bucket_idx: int,
    ) -> Generator[None, None, None]:
        """Process only non-Dion params while preserving standard DO bucket ownership."""
        pp_world_size = parallel_state.get_pipeline_model_parallel_world_size()
        rank = dist.get_rank() if dist.is_initialized() else 0
        if pp_world_size > 1:
            logger.info(
                "[DION_PP_DEBUG] rank=%s scalar_bucket enter buf=%s bucket=%s params=%s",
                rank,
                buf_idx,
                bucket_idx,
                len(scalar_params),
            )

        scalar_opt = self.defaults.get('scalar_optimizer', 'adamw')
        for p, grad, state, group in scalar_params:
            lr = group.get('lr', self.defaults['lr'])
            if p.ndim == 1:
                weight_decay = 0.0
            else:
                wd_mult = group.get('wd_mult', 1.0)
                weight_decay = self.defaults['weight_decay'] * wd_mult

            if scalar_opt == 'lion':
                self._lion_update(p, grad, state, group, lr, weight_decay)
            else:
                self._adamw_update(p, grad, state, group, lr, weight_decay)

        if pp_world_size > 1:
            logger.info(
                "[DION_PP_DEBUG] rank=%s scalar_bucket exit buf=%s bucket=%s",
                rank,
                buf_idx,
                bucket_idx,
            )
        if False:
            yield

    def _run_local_param_batch_async(self, params: List[Tensor], group: dict,
                                          configs: List[DionParamConfig]) -> Generator[None, None, None]:
        """Process parameter batch in regular mode with async."""
        dion_data = []
        adamw_data = []

        for p, config in zip(params, configs):
            grad = self._get_param_grad(p)
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
            yield from self._run_local_dion_batch_async(dion_data, group)

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

    def _run_dion_batch_async(self, dion_params: List[Tuple]) -> Generator[None, None, None]:
        """Process batch of Dion parameters with async operations."""
        if not dion_params:
            return

        pp_world_size = parallel_state.get_pipeline_model_parallel_world_size()
        rank = dist.get_rank() if dist.is_initialized() else 0

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
            # _sync_tp_shape_keys serializes None→-1 for sorting.
            # We must normalize inner/outer dim=None → -1 when the axis is absent
            # so that local shape_groups keys match synced keys.
            inner_dim = config.inner_shard_tensor_dim
            if inner_dim is None and not config.has_tp_axis:
                inner_dim = -1
            outer_dim = config.outer_shard_tensor_dim
            if outer_dim is None and not config.has_fs_axis:
                outer_dim = -1
            shape_key = (local_shape, config.has_fs_axis, config.has_tp_axis, config.is_transposed,
                        config.compressed_all_reduce,
                        inner_dim, outer_dim)

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

        # Process each shape group separately while preserving first-seen order.
        global_param_offset = 0
        ortho_completed_count = 0

        # Sync shape_keys across every group that must keep the same batch schedule.
        local_shape_keys = list(shape_groups.keys())
        grouped_shape_keys = {}
        for shape_key in local_shape_keys:
            sync_groups = self._shape_sync_groups(shape_key)
            if not sync_groups:
                grouped_shape_keys.setdefault(None, (None, []))[1].append(shape_key)
                continue
            for sync_group in sync_groups:
                grouped_shape_keys.setdefault(id(sync_group), (sync_group, []))[1].append(shape_key)

        all_shape_keys = []
        for sync_group, group_keys in grouped_shape_keys.values():
            if sync_group is not None:
                all_shape_keys.extend(self._sync_shape_keys(group_keys, sync_group))
            else:
                all_shape_keys.extend(group_keys)
        all_shape_keys = list(dict.fromkeys(all_shape_keys))

        for shape_key in all_shape_keys:
            local_shape = shape_key[0]
            m, n = local_shape

            has_tp_axis = shape_key[2]
            has_fs_axis = shape_key[1]
            sync_groups = self._shape_sync_groups(shape_key)

            if has_tp_axis and self.tp_group:
                batch_size = dist.get_world_size(self.tp_group)
            elif has_fs_axis and self.fs_group:
                batch_size = dist.get_world_size(self.fs_group)
            else:
                batch_size = dist.get_world_size(self.rp_group) if self.rp_group else 1

            if shape_key not in shape_groups:
                raise RuntimeError(
                    "[DION_MISSING_LOCAL_SHARD] "
                    f"shape_key={shape_key} rank={rank} sync_groups={sync_groups}"
                )

            group_data = shape_groups[shape_key]
            self._diag_state_replica_batch_ids(shape_key, group_data)
            for sync_group in sync_groups:
                self._align_group_data_order(group_data, sync_group)
            if pp_world_size > 1:
                logger.info(
                    "[DION_PP_DEBUG] rank=%s dion_shape enter shape=%s local_params=%s",
                    rank,
                    shape_key,
                    len(group_data['params']),
                )

            local_num_params = len(group_data['params'])
            for sync_group in sync_groups:
                world_size = dist.get_world_size(sync_group)
                gathered_counts = [None] * world_size
                dist.all_gather_object(gathered_counts, local_num_params, group=sync_group)
                mismatch_ranks = [
                    idx for idx, rank_count in enumerate(gathered_counts)
                    if int(rank_count) != int(local_num_params)
                ]
                if mismatch_ranks:
                    try:
                        group_ranks = dist.get_process_group_ranks(sync_group)
                    except Exception:
                        group_ranks = []
                    raise RuntimeError(
                        "[DION_PARAM_COUNT_MISMATCH] "
                        f"shape_key={shape_key} group_ranks={group_ranks} "
                        f"mismatch_local_ranks={mismatch_ranks} gathered_counts={gathered_counts}"
                    )

            for i in range(0, local_num_params, batch_size):

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

                local_batch_ids = []
                for meta in metas:
                    param_name = getattr(meta, "param_name", "") if meta is not None else ""
                    param_uid = getattr(meta, "param_uid", None) if meta is not None else None
                    local_batch_ids.append((param_name, param_uid))
                for sync_group in sync_groups:
                    world_size = dist.get_world_size(sync_group)
                    gathered_batch_ids = [None] * world_size
                    dist.all_gather_object(gathered_batch_ids, local_batch_ids, group=sync_group)
                    canonical_batch_ids = tuple(gathered_batch_ids[0])
                    mismatch_ranks = [
                        idx for idx, rank_ids in enumerate(gathered_batch_ids)
                        if tuple(rank_ids) != canonical_batch_ids
                    ]
                    if mismatch_ranks:
                        try:
                            group_ranks = dist.get_process_group_ranks(sync_group)
                        except Exception:
                            group_ranks = []
                        raise RuntimeError(
                            "[DION_BATCH_ID_MISMATCH] "
                            f"shape_key={shape_key} batch_start={i} batch_end={batch_end} "
                            f"group_ranks={group_ranks} mismatch_local_ranks={mismatch_ranks} "
                            f"local_batch_ids={local_batch_ids} gathered_batch_ids={gathered_batch_ids}"
                        )

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
            if pp_world_size > 1:
                logger.info(
                    "[DION_PP_DEBUG] rank=%s dion_shape exit shape=%s",
                    rank,
                    shape_key,
                )

        shape_groups.clear()

    def _run_local_dion_batch_async(self, dion_data: List[Tuple], group: dict) -> Generator[None, None, None]:
        """Process Dion parameters in regular mode with async batching."""
        params = []
        momentums = []
        Qs = []
        configs = []
        states = []

        for p, grad, state, config in dion_data:
            # Update momentum: M <- M + g (no decay here, it's in error feedback)
            # M <- M + G (decay is applied in error feedback, not here)
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

    def _get_param_grad(self, param: torch.nn.Parameter) -> Optional[torch.Tensor]:
        """Get gradient from parameter, handling different storage modes.

        Important: Check value is not None, not just hasattr.
        This ensures all ranks use same gradient source for collective operations.
        """
        # Standard DO optimizer-side contract is `.grad` / `.decoupled_grad`.
        # Do not fall back to `main_grad` in steady state; if optimizer-side grad
        # wiring failed, that is a real runtime bug and should surface elsewhere.
        dg = getattr(param, 'decoupled_grad', None)
        if dg is not None:
            return dg

        # Priority 2: grad
        try:
            if param.grad is not None:
                return param.grad
        except RuntimeError:
            # Non-leaf tensor grad access failed
            pass

        return None

    def enable_distributed_mode(
        self,
        global_buffer_sizes,
        full_data_parallel_group,
        tp_group,
                               dist_metas: Dict[torch.nn.Parameter, MegatronDionDistMeta],
                               rp_group=None,
                               fs_group=None,
                               state_replica_group=None,
    ):
        """Enable distributed mode for Megatron-Core backend.

        Args:
            global_buffer_sizes: Buffer sizes for gradients
            full_data_parallel_group: Full data parallel group (RP × FS)
            tp_group: Tensor parallel process group
            dist_metas: Metadata for distributed parameters
            rp_group: Optional explicit RP group (replicas with same shard)
            fs_group: Optional explicit FS group (shards within same replica)
            state_replica_group: Optional group of optimizer-state replicas for the same local shard
        """
        global_rank = self._global_rank

        self.global_buffer_sizes = global_buffer_sizes
        self.full_data_parallel_group = full_data_parallel_group
        self.dist_metas = dist_metas
        self.is_distributed_mode = True


        # Collective voting to detect inconsistent group provision
        if dist.is_initialized() and full_data_parallel_group is not None:
            have_rp_arg = torch.tensor([1 if rp_group is not None else 0],
                                       device=torch.cuda.current_device(), dtype=torch.int64)
            have_fs_arg = torch.tensor([1 if fs_group is not None else 0],
                                       device=torch.cuda.current_device(), dtype=torch.int64)
            have_state_replica_arg = torch.tensor(
                [1 if state_replica_group is not None else 0],
                device=torch.cuda.current_device(),
                dtype=torch.int64,
            )
            dist.all_reduce(have_rp_arg, op=dist.ReduceOp.MIN, group=full_data_parallel_group)
            dist.all_reduce(have_fs_arg, op=dist.ReduceOp.MIN, group=full_data_parallel_group)
            dist.all_reduce(
                have_state_replica_arg,
                op=dist.ReduceOp.MIN,
                group=full_data_parallel_group,
            )

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
            if int(have_state_replica_arg.item()) != 1 and state_replica_group is not None:
                raise RuntimeError(
                    f"Global rank {global_rank}: Inconsistent state_replica_group provision! "
                    f"This rank received state_replica_group, but some DP ranks did not (MIN=0)."
                )

        # Update 2D parallelism groups if provided

        if fs_group is not None:
            try:
                self.fs_group = fs_group
                self.fs_size = dist.get_world_size(fs_group)
                self.fs_rank = dist.get_rank(fs_group)
                fs_group_ranks = dist.get_process_group_ranks(fs_group)
                if len(fs_group_ranks) != self.fs_size or dist.get_rank() not in fs_group_ranks:
                    raise RuntimeError(
                        f"Global rank {global_rank}: invalid fs_group membership: "
                        f"size={self.fs_size} ranks={fs_group_ranks}"
                    )
            except Exception as e:
                logger.error(f"[Dion] Global rank {global_rank}: Failed updating FS group: {e}")
                raise

        if self.rp_group is not None:
            rp_group_ranks = dist.get_process_group_ranks(self.rp_group)
            rp_size = dist.get_world_size(self.rp_group)
            if len(rp_group_ranks) != rp_size or dist.get_rank() not in rp_group_ranks:
                raise RuntimeError(
                    f"Global rank {global_rank}: invalid rp_group membership: "
                    f"size={rp_size} ranks={rp_group_ranks}"
                )

        if state_replica_group is not None:
            self.state_replica_group = state_replica_group
            self.state_replica_world_size = dist.get_world_size(state_replica_group)
            self.state_replica_rank = dist.get_rank(state_replica_group)
            state_group_ranks = dist.get_process_group_ranks(state_replica_group)
            if (
                len(state_group_ranks) != self.state_replica_world_size
                or dist.get_rank() not in state_group_ranks
            ):
                raise RuntimeError(
                    f"Global rank {global_rank}: invalid state_replica_group membership: "
                    f"size={self.state_replica_world_size} ranks={state_group_ranks}"
                )

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

        # Verify consistent configuration across all ranks
        if dist.is_initialized() and full_data_parallel_group is not None:
            world_size = dist.get_world_size(full_data_parallel_group)
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
                dist.all_gather_object(
                    gathered_configs,
                    local_config,
                    group=full_data_parallel_group,
                )

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

    @staticmethod
    def _shape_key_to_config(shape_key: Tuple) -> DionParamConfig:
        """Build a DionParamConfig from a synchronized shape_key tuple."""
        return DionParamConfig(
            has_fs_axis=bool(shape_key[1]),
            has_tp_axis=bool(shape_key[2]),
            is_transposed=bool(shape_key[3]),
            compressed_all_reduce=bool(shape_key[4]),
            inner_shard_tensor_dim=shape_key[5] if shape_key[5] != -1 else None,
            outer_shard_tensor_dim=shape_key[6] if shape_key[6] != -1 else None,
        )

    def _restore_tp_global_shape(
        self,
        local_m: int,
        local_n: int,
        config: DionParamConfig,
    ) -> Tuple[int, int]:
        """Restore TP-sharded global shape from a local 2D shard shape."""
        if not config.has_tp_axis or self.tp_world_size <= 1:
            return local_m, local_n
        if config.inner_shard_tensor_dim == 0:
            return local_m * self.tp_world_size, local_n
        if config.inner_shard_tensor_dim == 1:
            return local_m, local_n * self.tp_world_size
        return local_m, local_n

    def _resolve_q_layout(
        self,
        local_m: int,
        local_n: int,
        config: DionParamConfig,
        *,
        global_shape: Optional[Tuple[int, int]] = None,
        rank_fraction: Optional[float] = None,
        rank_multiple_of: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Resolve global/local Q layout and rank dimensions for a 2D Dion param."""
        if global_shape is None:
            m_global, n_global = self._restore_tp_global_shape(local_m, local_n, config)
        else:
            m_global, n_global = global_shape

        if rank_fraction is None:
            rank_fraction = self.defaults.get('rank_fraction', 1.0)
        if rank_multiple_of is None:
            rank_multiple_of = self.defaults.get('rank_multiple_of', 1)

        q_base_local = local_m if config.is_transposed else local_n
        q_base_global = m_global if config.is_transposed else n_global

        r_global = rank_fraction * min(m_global, n_global)
        r_global = rank_multiple_of * math.ceil(r_global / rank_multiple_of)
        r_global = min(r_global, m_global, n_global)
        r_global = max(1, int(r_global))

        if self._needs_tp_q_unshard(config):
            if r_global < self.tp_world_size:
                r_global = self.tp_world_size
            elif r_global % self.tp_world_size != 0:
                r_global = self.tp_world_size * math.ceil(r_global / self.tp_world_size)
            r_local = r_global // self.tp_world_size
        else:
            r_local = r_global

        return {
            'global_shape': (m_global, n_global),
            'q_base_local': q_base_local,
            'q_base_global': q_base_global,
            'r_global': r_global,
            'r_local': max(1, int(r_local)),
            'q_shape': (q_base_local, max(1, int(r_local))),
        }

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
                    layout = self._resolve_q_layout(
                        lm,
                        ln,
                        config,
                        global_shape=tuple(meta.global_shape) if meta.global_shape is not None else None,
                        rank_fraction=meta.rank_fraction,
                        rank_multiple_of=self.defaults.get('rank_multiple_of', 1),
                    )
                    r_state = self.state.get(param, {}).get('r', None)
                    if r_state is not None:
                        r_global = int(r_state)
                    else:
                        r_global = layout['r_global']

                    m_true_global, n_true_global = layout['global_shape']
                    config.compressed_all_reduce = (
                        (m_true_global + n_true_global) * r_global < m_true_global * n_true_global
                    )
                else:
                    config.compressed_all_reduce = False

        elif param.ndim == 2:
            m, n = param.shape
            config.is_transposed = (m < n)
            if self.use_compressed_comm:
                layout = self._resolve_q_layout(
                    m,
                    n,
                    config,
                    global_shape=(m, n),
                    rank_fraction=self.defaults.get('rank_fraction', 1.0),
                    rank_multiple_of=self.defaults.get('rank_multiple_of', 1),
                )
                r_state = self.state.get(param, {}).get('r', None)
                if r_state is not None:
                    r = int(r_state)
                else:
                    r = layout['r_global']
                m_global, n_global = layout['global_shape']
                config.compressed_all_reduce = ((m_global + n_global) * r < m_global * n_global)
            else:
                config.compressed_all_reduce = False

        self._param_config[param] = config
        return config

    def _get_or_initialize_state(self, param: Tensor, group: dict) -> dict:
        """Get existing state or lazy-initialize it."""


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
            # See docs/dion/expert_partition_dim_analysis.md "Issue 2: TP Restoration Double Application Bug"
            m_true_global, n_true_global = m_global, n_global

            layout = self._resolve_q_layout(
                m,
                n,
                config,
                global_shape=(m_true_global, n_true_global),
                rank_fraction=rank_fraction,
                rank_multiple_of=rank_multiple_of,
            )
            r_global = layout['r_global']
            r_local = layout['r_local']
            q_base_local = layout['q_base_local']
            q_base_global = layout['q_base_global']
            Q_shape = layout['q_shape']
            r = r_global

            Q_dtype = self._str_to_dtype(self._mixed_precision_config.Q_dtype) or param.dtype

            if self.is_distributed_mode and meta:
                param_key = getattr(meta, "param_name", "") or ""
                if not param_key:
                    param_key = str(meta.param_uid) if meta.param_uid is not None else f"{meta.buffer_idx}_{meta.bucket_idx}"
                q_seed_key = repr(
                    (
                        getattr(meta, "param_uid", None),
                        q_base_global,
                        r_global,
                        str(Q_dtype),
                    )
                ).encode("utf-8")
                q_seed = int.from_bytes(
                    hashlib.blake2b(q_seed_key, digest_size=8).digest(), "little"
                ) & ((1 << 63) - 1)
                q_gen = torch.Generator(device=param.device)
                q_gen.manual_seed(q_seed)
                Q_global_full = torch.randn(
                    (q_base_global, r_global),
                    device=param.device,
                    dtype=Q_dtype,
                    generator=q_gen,
                )

                if config.has_fs_axis and getattr(meta, "shard_group_world_size", 1) > 1:
                    fs_rank = getattr(meta, "shard_group_rank", -1)
                    if fs_rank < 0:
                        logger.error(
                            "[DION_Q_INIT_GROUP_ISSUE] param=%s missing shard_group_rank "
                            "for FS-sharded Q init: local_shape=%s global_shape=%s fs_split_dim=%s",
                            param_key,
                            (m, n),
                            (m_true_global, n_true_global),
                            config.outer_shard_tensor_dim,
                        )
                        raise RuntimeError(
                            f"Dion Q init missing shard_group_rank for FS-sharded param {param_key}"
                        )
                else:
                    fs_rank = 0

                if config.has_tp_axis and self.tp_group is not None and self.tp_world_size > 1:
                    tp_rank = self.tp_rank
                else:
                    tp_rank = 0

                fs_start = fs_rank * q_base_local
                fs_end = fs_start + q_base_local
                tp_start = tp_rank * r_local
                tp_end = tp_start + r_local
                Q = Q_global_full[fs_start:fs_end, tp_start:tp_end].contiguous()

                del Q_global_full

                if Q.shape != Q_shape:
                    logger.error(
                        "[DION_Q_INIT_SHAPE_MISMATCH] param=%s expected=%s got=%s "
                        "local_shape=%s global_shape=%s has_fs=%s has_tp=%s "
                        "fs_rank=%s tp_rank=%s fs_range=[%s:%s] tp_range=[%s:%s]",
                        param_key,
                        Q_shape,
                        tuple(Q.shape),
                        (m, n),
                        (m_true_global, n_true_global),
                        config.has_fs_axis,
                        config.has_tp_axis,
                        fs_rank,
                        tp_rank,
                        fs_start,
                        fs_end,
                        tp_start,
                        tp_end,
                    )
                    raise RuntimeError(
                        f"Dion Q init shape mismatch for {param_key}: expected {Q_shape}, got {tuple(Q.shape)}"
                    )

                self._broadcast_replicate_domain_(Q)
                if (
                    os.getenv("DION_STATE_REPLICA_DIAG", "0") == "1"
                    and self.state_replica_group is not None
                    and self.state_replica_world_size > 1
                ):
                    seen = getattr(self, "_state_replica_q_diag_seen", None)
                    if seen is None:
                        seen = set()
                        self._state_replica_q_diag_seen = seen
                    diag_key = getattr(meta, "param_uid", None)
                    if diag_key not in seen and len(seen) < 16:
                        seen.add(diag_key)
                        q_fp = (
                            getattr(meta, "param_name", ""),
                            getattr(meta, "param_uid", None),
                            tuple(Q.shape),
                            float(Q.float().sum().item()),
                            float(Q.float().abs().sum().item()),
                            float((Q.float() ** 2).sum().item()),
                            float(Q.float().abs().max().item()),
                        )
                        gathered = [None] * self.state_replica_world_size
                        dist.all_gather_object(
                            gathered,
                            q_fp,
                            group=self.state_replica_group,
                        )
                        logger.info(
                            "[DION_STATE_REPLICA_Q_INIT] rank=%s group=%s gathered=%s",
                            self._global_rank,
                            dist.get_process_group_ranks(self.state_replica_group),
                            gathered,
                        )
            else:
                Q = torch.randn(Q_shape, device=param.device, dtype=Q_dtype)
                self._broadcast_replicate_domain_(Q)

            state['Q'] = Q
            state['_needs_state_replica_q_sync'] = True
            state['r'] = r
            state['local_shape'] = (m, n)
            state['true_global_shape'] = (m_true_global, n_true_global)
            # Store per-expert global shape for GroupedMLP LR scaling
            per_expert = getattr(meta, 'per_expert_global_shape', None)
            if per_expert is not None:
                state['per_expert_global_shape'] = per_expert

    # TP Sync methods

    def _sync_shape_keys(self, local_shape_keys: List[Tuple], group) -> List[Tuple]:
        """Align shape-key schedules to rank0 order when contents match exactly."""
        if group is None:
            return local_shape_keys

        world_size = dist.get_world_size(group)
        if world_size <= 1:
            return local_shape_keys

        cache_key = (id(group), tuple(local_shape_keys))
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

        gathered_keys_list = [None] * world_size
        dist.all_gather_object(gathered_keys_list, local_keys_serialized, group=group)

        canonical_keys = tuple(gathered_keys_list[0])
        canonical_counter = Counter(canonical_keys)
        mismatch_ranks = []
        for idx, rank_keys in enumerate(gathered_keys_list):
            if Counter(rank_keys) != canonical_counter:
                mismatch_ranks.append(idx)
        if mismatch_ranks:
            try:
                group_ranks = dist.get_process_group_ranks(group)
            except Exception:
                group_ranks = []
            raise RuntimeError(
                "[DION_SHAPE_KEY_MISMATCH] "
                f"group_ranks={group_ranks} mismatch_local_ranks={mismatch_ranks} "
                f"local_shape_keys={local_keys_serialized} gathered_shape_keys={gathered_keys_list}"
            )

        ordered_keys = list(canonical_keys)
        self._shape_key_cache[cache_key] = ordered_keys
        return ordered_keys

    def _align_group_data_order(self, group_data: Dict[str, List[Any]], sync_group) -> None:
        """Reorder one shape-group's param lists to the rank0 canonical batch-id order."""
        if sync_group is None or dist.get_world_size(sync_group) <= 1:
            return

        local_ids = []
        for meta in group_data['metas']:
            param_name = getattr(meta, "param_name", "") if meta is not None else ""
            param_uid = getattr(meta, "param_uid", None) if meta is not None else None
            local_ids.append((param_name, param_uid))

        world_size = dist.get_world_size(sync_group)
        gathered_ids = [None] * world_size
        dist.all_gather_object(gathered_ids, local_ids, group=sync_group)

        canonical_ids = tuple(gathered_ids[0])
        canonical_counter = Counter(canonical_ids)
        mismatch_ranks = []
        for idx, rank_ids in enumerate(gathered_ids):
            if Counter(rank_ids) != canonical_counter:
                mismatch_ranks.append(idx)
        if mismatch_ranks:
            try:
                group_ranks = dist.get_process_group_ranks(sync_group)
            except Exception:
                group_ranks = []
            raise RuntimeError(
                "[DION_PARAM_ID_CONTENT_MISMATCH] "
                f"group_ranks={group_ranks} mismatch_local_ranks={mismatch_ranks} "
                f"local_ids={local_ids} gathered_ids={gathered_ids}"
            )

        if tuple(local_ids) == canonical_ids:
            return

        index_by_id = {param_id: idx for idx, param_id in enumerate(local_ids)}
        reorder_indices = [index_by_id[param_id] for param_id in canonical_ids]
        for key in ('params', 'grads', 'states', 'groups', 'configs', 'metas'):
            group_data[key] = [group_data[key][idx] for idx in reorder_indices]

    def _diag_state_replica_batch_ids(self, shape_key: Tuple, group_data: Dict[str, List[Any]]) -> None:
        """Log state-replica content/order for one shape group to prove schedule mismatches."""
        if os.getenv("DION_STATE_REPLICA_BATCH_DIAG", "0") != "1":
            return
        if self.state_replica_group is None or self.state_replica_world_size <= 1:
            return

        local_ids = []
        for meta in group_data['metas']:
            param_name = getattr(meta, "param_name", "") if meta is not None else ""
            param_uid = getattr(meta, "param_uid", None) if meta is not None else None
            local_ids.append((param_name, param_uid))

        world_size = dist.get_world_size(self.state_replica_group)
        gathered_ids = [None] * world_size
        dist.all_gather_object(gathered_ids, local_ids, group=self.state_replica_group)

        canonical_ids = tuple(gathered_ids[0])
        mismatch_ranks = [
            idx for idx, rank_ids in enumerate(gathered_ids)
            if tuple(rank_ids) != canonical_ids
        ]
        try:
            group_ranks = dist.get_process_group_ranks(self.state_replica_group)
        except Exception:
            group_ranks = []
        if mismatch_ranks:
            logger.info(
                "[DION_STATE_REPLICA_BATCH_ID_MISMATCH] rank=%s shape_key=%s group_ranks=%s "
                "mismatch_local_ranks=%s local_ids=%s gathered_ids=%s",
                dist.get_rank() if dist.is_initialized() else 0,
                shape_key,
                group_ranks,
                mismatch_ranks,
                local_ids,
                gathered_ids,
            )
        else:
            logger.info(
                "[DION_STATE_REPLICA_BATCH_ID_OK] rank=%s shape_key=%s group_ranks=%s ids=%s",
                dist.get_rank() if dist.is_initialized() else 0,
                shape_key,
                group_ranks,
                local_ids,
            )

    def _diag_state_replica_values(
        self,
        stage: str,
        *,
        metas: List,
        grads: Optional[List[Tensor]] = None,
        momentums: Optional[List[Tensor]] = None,
        Qs: Optional[List[Tensor]] = None,
        params: Optional[List[Tensor]] = None,
        real_batch_size: Optional[int] = None,
    ) -> None:
        """Compare selected tensor fingerprints across optimizer-state replicas."""
        if os.getenv("DION_STATE_REPLICA_VALUE_DIAG", "0") != "1":
            return
        if self.state_replica_group is None or self.state_replica_world_size <= 1:
            return

        if real_batch_size is None:
            real_batch_size = len(metas)
        name_substr = os.getenv("DION_STATE_REPLICA_VALUE_NAME", "")
        tol = float(os.getenv("DION_STATE_REPLICA_VALUE_TOL", "1e-6"))

        def _fp(tensor: Optional[Tensor]):
            if tensor is None:
                return None
            tf = tensor.float()
            return (
                tuple(tf.shape),
                float(tf.norm().item()),
                float(tf.sum().item()),
                float(tf.abs().max().item()),
            )

        local = []
        for idx, meta in enumerate(metas[:real_batch_size]):
            param_name = getattr(meta, "param_name", "") if meta is not None else ""
            if name_substr and name_substr not in param_name:
                continue
            local.append(
                {
                    "slot": int(idx),
                    "name": param_name,
                    "uid": getattr(meta, "param_uid", None) if meta is not None else None,
                    "grad": _fp(grads[idx]) if grads is not None else None,
                    "momentum": _fp(momentums[idx]) if momentums is not None else None,
                    "q": _fp(Qs[idx]) if Qs is not None else None,
                    "param": _fp(params[idx]) if params is not None else None,
                }
            )

        if not local:
            return

        gathered = [None] * self.state_replica_world_size
        dist.all_gather_object(gathered, local, group=self.state_replica_group)
        canonical = gathered[0]

        def _same_fp(left, right) -> bool:
            if left is None or right is None:
                return left is None and right is None
            if left[0] != right[0]:
                return False
            for i in range(1, 4):
                if abs(float(left[i]) - float(right[i])) > tol:
                    return False
            return True

        mismatch_ranks = []
        mismatch_payload = []
        for rank_idx, rank_entries in enumerate(gathered[1:], start=1):
            same = len(rank_entries) == len(canonical)
            if same:
                for lhs, rhs in zip(canonical, rank_entries):
                    if (
                        lhs["slot"] != rhs["slot"]
                        or lhs["name"] != rhs["name"]
                        or lhs["uid"] != rhs["uid"]
                        or not _same_fp(lhs["grad"], rhs["grad"])
                        or not _same_fp(lhs["momentum"], rhs["momentum"])
                        or not _same_fp(lhs["q"], rhs["q"])
                        or not _same_fp(lhs["param"], rhs["param"])
                    ):
                        same = False
                        mismatch_payload.append({"ref": lhs, "rank": rhs})
                        break
            if not same:
                mismatch_ranks.append(rank_idx)

        group_ranks = dist.get_process_group_ranks(self.state_replica_group)
        if mismatch_ranks:
            logger.error(
                "[DION_STATE_REPLICA_VALUE_MISMATCH] rank=%s step=%s stage=%s group_ranks=%s mismatch_local_ranks=%s "
                "canonical=%s gathered=%s payload=%s",
                self._global_rank,
                self._step_count,
                stage,
                group_ranks,
                mismatch_ranks,
                canonical,
                gathered,
                mismatch_payload[:4],
            )
        else:
            logger.info(
                "[DION_STATE_REPLICA_VALUE_OK] rank=%s step=%s stage=%s group_ranks=%s entries=%s",
                self._global_rank,
                self._step_count,
                stage,
                group_ranks,
                canonical,
            )

    def _diag_global_state_fp(
        self,
        stage: str,
        *,
        metas: List,
        grads: Optional[List[Tensor]] = None,
        momentums: Optional[List[Tensor]] = None,
        Qs: Optional[List[Tensor]] = None,
        params: Optional[List[Tensor]] = None,
        real_batch_size: Optional[int] = None,
    ) -> None:
        """Aggregate global tensor fingerprints for selected params over full_data_parallel_group.

        This diagnostic is used to compare the logical global Dion state across
        different FS/CP configurations. It intentionally keeps only one
        representative per state-replica group so duplicate local shards do not
        get double-counted.
        """
        if os.getenv("DION_GLOBAL_STATE_FP_DIAG", "0") != "1":
            return
        if not self.is_distributed_mode or self.full_data_parallel_group is None:
            return

        steps_env = os.getenv("DION_GLOBAL_STATE_FP_STEPS", "1")
        wanted_steps = set()
        for item in steps_env.split(","):
            item = item.strip()
            if not item:
                continue
            try:
                wanted_steps.add(int(item))
            except ValueError:
                continue
        if wanted_steps and self._step_count not in wanted_steps:
            return

        if real_batch_size is None:
            real_batch_size = len(metas)
        name_substr = os.getenv("DION_GLOBAL_STATE_FP_NAME", "")

        def _tensor_stats(tensor: Optional[Tensor]):
            if tensor is None:
                return None
            tf = tensor.detach().float()
            return {
                "shape": tuple(tf.shape),
                "sum": float(tf.sum().item()),
                "abs_sum": float(tf.abs().sum().item()),
                "sum_sq": float((tf * tf).sum().item()),
                "max_abs": float(tf.abs().max().item()),
            }

        local_entries = []
        for idx, meta in enumerate(metas[:real_batch_size]):
            param_name = getattr(meta, "param_name", "") if meta is not None else ""
            if name_substr and name_substr not in param_name:
                continue
            local_entries.append(
                {
                    "slot": int(idx),
                    "name": param_name,
                    "uid": getattr(meta, "param_uid", None) if meta is not None else None,
                    "contribute": bool(
                        self.state_replica_group is None
                        or self.state_replica_world_size <= 1
                        or self.state_replica_rank == 0
                    ),
                    "rank": int(self._global_rank),
                    "state_replica_rank": int(self.state_replica_rank),
                    "grad": _tensor_stats(grads[idx]) if grads is not None else None,
                    "momentum": _tensor_stats(momentums[idx]) if momentums is not None else None,
                    "q": _tensor_stats(Qs[idx]) if Qs is not None else None,
                    "param": _tensor_stats(params[idx]) if params is not None else None,
                    "global_shape": getattr(meta, "global_shape", None) if meta is not None else None,
                }
            )

        if not local_entries:
            return

        world_size = dist.get_world_size(self.full_data_parallel_group)
        gathered = [None] * world_size
        dist.all_gather_object(gathered, local_entries, group=self.full_data_parallel_group)

        by_uid = {}
        for rank_entries in gathered:
            if not rank_entries:
                continue
            for entry in rank_entries:
                by_uid.setdefault(entry["uid"], []).append(entry)

        for uid, entries in by_uid.items():
            name = entries[0]["name"]
            aggregates = {}
            for tensor_key in ("grad", "momentum", "q", "param"):
                present = [e for e in entries if e["contribute"] and e.get(tensor_key) is not None]
                if not present:
                    aggregates[tensor_key] = None
                    continue
                stats0 = present[0][tensor_key]
                aggregates[tensor_key] = {
                    "shape0": stats0["shape"],
                    "contributors": len(present),
                    "sum": sum(e[tensor_key]["sum"] for e in present),
                    "abs_sum": sum(e[tensor_key]["abs_sum"] for e in present),
                    "sum_sq": sum(e[tensor_key]["sum_sq"] for e in present),
                    "max_abs": max(e[tensor_key]["max_abs"] for e in present),
                }
            logger.info(
                "[DION_GLOBAL_STATE_FP] rank=%s step=%s stage=%s uid=%s name=%s global_shape=%s aggregates=%s entries=%s",
                self._global_rank,
                self._step_count,
                stage,
                uid,
                name,
                entries[0].get("global_shape"),
                aggregates,
                entries,
            )

    def _diag_global_batch_fp(
        self,
        stage: str,
        *,
        metas: List,
        batch: Optional[Tensor],
        real_batch_size: Optional[int] = None,
    ) -> None:
        """Aggregate global fingerprints for one logical batched intermediate tensor.

        This is used to prove where topology-dependent divergence first appears
        inside the Dion update. Like `_diag_global_state_fp`, it keeps only one
        representative per state-replica group.
        """
        if os.getenv("DION_GLOBAL_BATCH_FP_DIAG", "0") != "1":
            return
        if batch is None or not self.is_distributed_mode or self.full_data_parallel_group is None:
            return

        steps_env = os.getenv("DION_GLOBAL_BATCH_FP_STEPS", "1")
        wanted_steps = set()
        for item in steps_env.split(","):
            item = item.strip()
            if not item:
                continue
            try:
                wanted_steps.add(int(item))
            except ValueError:
                continue
        if wanted_steps and self._step_count not in wanted_steps:
            return

        if real_batch_size is None:
            real_batch_size = len(metas)
        name_substr = os.getenv("DION_GLOBAL_BATCH_FP_NAME", "")

        def _tensor_stats(tensor: Tensor):
            tf = tensor.detach().float()
            return {
                "shape": tuple(tf.shape),
                "sum": float(tf.sum().item()),
                "abs_sum": float(tf.abs().sum().item()),
                "sum_sq": float((tf * tf).sum().item()),
                "max_abs": float(tf.abs().max().item()),
            }

        local_entries = []
        for idx, meta in enumerate(metas[:real_batch_size]):
            param_name = getattr(meta, "param_name", "") if meta is not None else ""
            if name_substr and name_substr not in param_name:
                continue
            local_entries.append(
                {
                    "slot": int(idx),
                    "name": param_name,
                    "uid": getattr(meta, "param_uid", None) if meta is not None else None,
                    "contribute": bool(
                        self.state_replica_group is None
                        or self.state_replica_world_size <= 1
                        or self.state_replica_rank == 0
                    ),
                    "rank": int(self._global_rank),
                    "state_replica_rank": int(self.state_replica_rank),
                    "tensor": _tensor_stats(batch[idx]),
                    "global_shape": getattr(meta, "global_shape", None) if meta is not None else None,
                }
            )

        if not local_entries:
            return

        world_size = dist.get_world_size(self.full_data_parallel_group)
        gathered = [None] * world_size
        dist.all_gather_object(gathered, local_entries, group=self.full_data_parallel_group)

        by_uid = {}
        for rank_entries in gathered:
            if not rank_entries:
                continue
            for entry in rank_entries:
                by_uid.setdefault(entry["uid"], []).append(entry)

        for uid, entries in by_uid.items():
            name = entries[0]["name"]
            present = [e for e in entries if e["contribute"] and e.get("tensor") is not None]
            if not present:
                aggregates = None
            else:
                stats0 = present[0]["tensor"]
                aggregates = {
                    "shape0": stats0["shape"],
                    "contributors": len(present),
                    "sum": sum(e["tensor"]["sum"] for e in present),
                    "abs_sum": sum(e["tensor"]["abs_sum"] for e in present),
                    "sum_sq": sum(e["tensor"]["sum_sq"] for e in present),
                    "max_abs": max(e["tensor"]["max_abs"] for e in present),
                }
            logger.info(
                "[DION_GLOBAL_BATCH_FP] rank=%s step=%s stage=%s uid=%s name=%s global_shape=%s aggregates=%s entries=%s",
                self._global_rank,
                self._step_count,
                stage,
                uid,
                name,
                entries[0].get("global_shape"),
                aggregates,
                entries,
            )

    def _sync_tp_shape_keys(self, local_shape_keys: List[Tuple]) -> List[Tuple]:
        """Backward-compatible TP wrapper around _sync_shape_keys()."""
        return self._sync_shape_keys(local_shape_keys, self.tp_group)

    def _shape_sync_groups(self, shape_key: Tuple):
        """Return all process groups that must see the same batch schedule for this shape."""
        cfg = self._shape_key_to_config(shape_key)
        groups = []
        if (
            self.state_replica_group is not None
            and self.state_replica_world_size > 1
        ):
            groups.append(self.state_replica_group)
        if (
            cfg.has_tp_axis
            and self.tp_group is not None
            and dist.get_world_size(self.tp_group) > 1
        ):
            groups.append(self.tp_group)
        if (
            cfg.has_fs_axis
            and self.use_fs_collectives
            and self.fs_group is not None
            and dist.get_world_size(self.fs_group) > 1
        ):
            if all(id(self.fs_group) != id(existing) for existing in groups):
                groups.append(self.fs_group)

        return groups

    # Orthogonalization methods

    def _orthogonalize(
        self,
        P: torch.Tensor,
        rcqr_oversample: float = 1.25,
        sketch_fn=None,
    ) -> torch.Tensor:
        """Local orthogonalization matching dion_reference RNG semantics."""
        return orthogonalize(P, rcqr_oversample, sketch_fn)

    def _reshard_q_along_tp(self, Q: torch.Tensor, tp_group, tp_rank: int) -> torch.Tensor:
        """Re-shard Q matrix along TP dimension after update."""
        return reshard_q_along_tp(Q, tp_group, tp_rank)

    @staticmethod
    def _needs_fs_p_reduce(config: DionParamConfig) -> bool:
        """Return whether STEP 3.5 must reduce P across FS shards."""
        return config.has_fs_axis and (
            (not config.is_transposed and config.outer_shard_tensor_dim == 1)
            or (config.is_transposed and config.outer_shard_tensor_dim == 0)
        )

    def _needs_tp_q_unshard(self, config: DionParamConfig) -> bool:
        """Return whether STEP 2 must all-gather Q across TP ranks."""
        return config.has_tp_axis and self.tp_group is not None and self.tp_world_size > 1

    def _needs_tp_r_reduce(self, config: DionParamConfig) -> bool:
        """Return whether STEP 5 must all-reduce R across TP ranks."""
        return self._p_is_tp_sharded(config) and self.tp_group is not None and self.tp_world_size > 1

    def _fs_group_for_meta(self, meta=None):
        """Return the FS shard group for a param, including expert-local groups."""
        shard_group = getattr(meta, 'shard_group', None) if meta is not None else None
        if shard_group is None:
            shard_group = self.fs_group
        return shard_group

    @staticmethod
    def _p_is_tp_sharded(config: DionParamConfig) -> bool:
        """Return whether P is sharded along TP for this config."""
        return config.has_tp_axis and (
            (not config.is_transposed and config.inner_shard_tensor_dim == 0)
            or (config.is_transposed and config.inner_shard_tensor_dim == 1)
        )

    @staticmethod
    def _p_is_fs_sharded(config: DionParamConfig) -> bool:
        """Return whether P is sharded along FS for this config."""
        return config.has_fs_axis and (
            (not config.is_transposed and config.outer_shard_tensor_dim == 0)
            or (config.is_transposed and config.outer_shard_tensor_dim == 1)
        )

    def _ortho_group_for_config(self, config: DionParamConfig, meta=None):
        """Return the process group that must participate in P orthogonalization."""
        if self._p_is_tp_sharded(config) and self.tp_group and dist.get_world_size(self.tp_group) > 1:
            return self.tp_group

        if (
            not self._p_is_tp_sharded(config)
            and self._p_is_fs_sharded(config)
            and self.use_fs_collectives
        ):
            shard_group = self._fs_group_for_meta(meta)
            if shard_group and dist.get_world_size(shard_group) > 1:
                return shard_group

        return None

    def _distributed_orthogonalize(
        self,
        P_batch: torch.Tensor,
        *,
        shard_group: Optional[torch.distributed.ProcessGroup],
        oversample: float = 1.25,
        metas: Optional[List] = None,
    ) -> torch.Tensor:
        """Distributed orthogonalization matching Algorithm 2."""

        def _log_nonfinite(stage: str, tensor: torch.Tensor, extra: str = ""):
            if not (_DION_DIAG and self._step_count <= _DION_DIAG_MAX_STEPS):
                return
            if torch.isfinite(tensor).all():
                return
            warn_count = getattr(self, "_ortho_nonfinite_warn_count", 0)
            if warn_count >= 16:
                return
            setattr(self, "_ortho_nonfinite_warn_count", warn_count + 1)
            logger.warning(
                "[DION_ORTHO_NONFINITE] rank=%s step=%s stage=%s shape=%s %s",
                self._global_rank,
                self._step_count,
                stage,
                tuple(tensor.shape),
                extra,
            )

        def _log_group_mismatch(stage: str, tensor: torch.Tensor):
            if not (_DION_DIAG and self._step_count <= _DION_DIAG_MAX_STEPS):
                return
            if shard_group is None or dist.get_world_size(shard_group) <= 1:
                return
            warn_count = getattr(self, "_group_mismatch_warn_count", 0)
            if warn_count >= 16:
                return
            group_ranks = dist.get_process_group_ranks(shard_group)
            broadcast_src = group_ranks[0]
            ref = tensor.detach().clone()
            dist.broadcast(ref, src=broadcast_src, group=shard_group)
            tensor_f = tensor.float()
            ref_f = ref.float()
            finite = torch.isfinite(tensor_f) & torch.isfinite(ref_f)
            if finite.any():
                max_diff = (tensor_f[finite] - ref_f[finite]).abs().max().item()
            else:
                max_diff = float("inf")
            if max_diff <= 1e-6:
                return
            setattr(self, "_group_mismatch_warn_count", warn_count + 1)
            logger.warning(
                "[DION_GROUP_MISMATCH] rank=%s step=%s stage=%s shape=%s max_diff=%.6e src=%s",
                self._global_rank,
                self._step_count,
                stage,
                tuple(tensor.shape),
                max_diff,
                broadcast_src,
            )

        def _log_triangular_state(stage: str, tensor: torch.Tensor):
            if not (_DION_DIAG and self._step_count <= _DION_DIAG_MAX_STEPS):
                return
            if tensor.dim() < 3 or tensor.size(-1) == 0:
                return
            warn_count = getattr(self, "_triangular_state_log_count", 0)
            if warn_count >= 16:
                return
            setattr(self, "_triangular_state_log_count", warn_count + 1)
            tensor_f = tensor.float()
            lower = tensor_f.tril(-1).abs().amax().item()
            diag = torch.diagonal(tensor_f, dim1=-2, dim2=-1).abs()
            finite_diag = diag[torch.isfinite(diag)]
            diag_abs_min = finite_diag.amin().item() if finite_diag.numel() > 0 else float("nan")
            logger.warning(
                "[DION_TRIANGULAR_STATE] rank=%s step=%s stage=%s shape=%s lower_max=%.6e diag_abs_min=%.6e",
                self._global_rank,
                self._step_count,
                stage,
                tuple(tensor.shape),
                lower,
                diag_abs_min,
            )


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

        # Match dion_reference.distributed_orthogonalize(): square/wide matrices use
        # batch-sharded all-to-all + local QR, not randomized Cholesky QR.
        if m_shard_local <= r:
            if batch_size != shard_world_size:
                raise RuntimeError(
                    "[DION_QR_BATCH_SIZE_MISMATCH] "
                    f"batch_size={batch_size} shard_world_size={shard_world_size}"
                )

            send_list = [P_batch[i].contiguous() for i in range(batch_size)]
            recv_list = [torch.empty_like(send_list[0]) for _ in range(shard_world_size)]
            dist.all_to_all(recv_list, send_list, group=shard_group)

            P_single = torch.cat(recv_list, dim=0)
            Q_single, _ = torch.linalg.qr(P_single.to(dtype=torch.float32), mode='reduced')

            m_global = Q_single.size(0)
            if m_global % shard_world_size != 0:
                raise RuntimeError(
                    "[DION_QR_ROW_SPLIT_MISMATCH] "
                    f"m_global={m_global} shard_world_size={shard_world_size}"
                )

            row_chunk = m_global // shard_world_size
            send_back = [Q_single[j * row_chunk:(j + 1) * row_chunk].contiguous() for j in range(shard_world_size)]
            recv_back = [torch.empty_like(send_back[0]) for _ in range(batch_size)]
            dist.all_to_all(recv_back, send_back, group=shard_group)

            result = torch.stack(recv_back, dim=0)
            return result.to(original_dtype).contiguous()

        m_global = m_shard_local * shard_world_size
        sketch_fn = self._seeded_sketch_fn(metas=metas, tag="distributed_ortho")
        if sketch_fn is not None:
            S_full = sketch_fn(
                torch.empty(batch_size, m_global, r, device=device, dtype=original_dtype),
                oversample,
            )
        else:
            k = math.ceil(oversample * r / 128) * 128
            if k <= 0:
                raise RuntimeError(
                    f"[DION_INVALID_SKETCH_RANK] r={r} oversample={oversample} k={k}"
                )
            std = math.sqrt(1.0 / k)
            S_full = torch.empty(batch_size, k, m_global, device=device, dtype=original_dtype)

            # PP-safe: use shard_group for broadcast, not default process group
            shard_group_ranks = dist.get_process_group_ranks(shard_group)
            broadcast_src = shard_group_ranks[0]
            if dist.get_rank() == broadcast_src:
                S_full.normal_(mean=0.0, std=std)
            dist.broadcast(S_full, src=broadcast_src, group=shard_group)

        m_start = shard_rank * m_shard_local
        m_end = m_start + m_shard_local
        S_batch = S_full[:, :, m_start:m_end].contiguous()
        del S_full

        SP_batch = S_batch @ P_batch
        _log_nonfinite("sp_batch", SP_batch, f"m_local={m_shard_local} r={r} shard_world={shard_world_size}")
        del S_batch

        SP_slice = funcol.reduce_scatter_tensor(
            SP_batch, reduceOp="sum", scatter_dim=0, group=shard_group
        )
        _log_nonfinite("sp_slice", SP_slice, f"m_local={m_shard_local} r={r} shard_world={shard_world_size}")
        del SP_batch

        SP_slice_f32 = SP_slice.to(dtype=torch.float32)
        r_slices = []
        for i in range(SP_slice_f32.size(0)):
            _, r_local = torch.linalg.qr(SP_slice_f32[i], mode='r')
            r_slices.append(r_local)
        R_slice = torch.stack(r_slices, dim=0)
        _log_nonfinite("r_slice_qr", R_slice, f"m_local={m_shard_local} r={r} shard_world={shard_world_size}")
        del SP_slice, SP_slice_f32, r_slices

        R_full = funcol.all_gather_tensor(
            R_slice.contiguous(), gather_dim=0, group=shard_group
        )
        _log_nonfinite("r_full_qr", R_full, f"m_local={m_shard_local} r={r} shard_world={shard_world_size}")
        _log_group_mismatch("r_full_qr", R_full)
        _log_triangular_state("r_full_qr", R_full)
        del R_slice

        P_batch_f32 = P_batch.to(dtype=torch.float32)
        p_solved = []
        for i in range(P_batch_f32.size(0)):
            p_solved.append(
                torch.linalg.solve_triangular(
                    R_full[i], P_batch_f32[i], upper=True, left=False
                )
            )
        P_batch_fp32 = torch.stack(p_solved, dim=0)
        _log_nonfinite("p_after_qr_solve", P_batch_fp32, f"m_local={m_shard_local} r={r} shard_world={shard_world_size}")
        del P_batch_f32, p_solved

        PP_local = P_batch_fp32.mT @ P_batch_fp32
        _log_nonfinite("pp_local", PP_local, f"m_local={m_shard_local} r={r} shard_world={shard_world_size}")

        PP_slice = funcol.reduce_scatter_tensor(
            PP_local, reduceOp="sum", scatter_dim=0, group=shard_group
        )
        _log_nonfinite("pp_slice", PP_slice, f"m_local={m_shard_local} r={r} shard_world={shard_world_size}")
        del PP_local

        # `PP = P^T P` is symmetric positive semidefinite by construction. In practice,
        # TP-heavy shards can pick up tiny negative eigenvalues from float32 roundoff,
        # which makes Cholesky fail even though the reference math is still the same.
        PP_slice = 0.5 * (PP_slice + PP_slice.mT)
        diag_max = torch.diagonal(PP_slice, dim1=-2, dim2=-1).amax(dim=-1).clamp_min(1.0)
        jitter = torch.finfo(PP_slice.dtype).eps * diag_max
        eye = torch.eye(PP_slice.size(-1), device=PP_slice.device, dtype=PP_slice.dtype)
        PP_slice = PP_slice + jitter.view(-1, 1, 1) * eye.unsqueeze(0)

        r_slices = []
        info_list = []
        for i in range(PP_slice.size(0)):
            r_local, info_local = torch.linalg.cholesky_ex(PP_slice[i], upper=True)
            r_slices.append(r_local)
            info_list.append(info_local)
        R_slice = torch.stack(r_slices, dim=0)
        info = torch.stack(info_list, dim=0)
        _log_nonfinite(
            "r_slice_cholesky",
            R_slice,
            f"m_local={m_shard_local} r={r} shard_world={shard_world_size} info={info.tolist()}",
        )
        if (info > 0).any():
            info_list_cpu = info.detach().cpu().tolist()
            del PP_slice, info, R_slice, r_slices, info_list
            raise RuntimeError(
                "[DION_DISTRIBUTED_ORTHO_CHOLESKY_FAILED] distributed orthogonalization "
                "encountered a non-positive-definite slice and refuses local QR fallback "
                f"(m_local={m_shard_local}, r={r}, shard_world_size={shard_world_size}, "
                f"info={info_list_cpu})"
            )
        del PP_slice, info, r_slices, info_list

        R_full = funcol.all_gather_tensor(
            R_slice.contiguous(), gather_dim=0, group=shard_group
        )
        _log_nonfinite("r_full_cholesky", R_full, f"m_local={m_shard_local} r={r} shard_world={shard_world_size}")
        _log_group_mismatch("r_full_cholesky", R_full)
        _log_triangular_state("r_full_cholesky", R_full)
        del R_slice

        p_solved = []
        for i in range(P_batch_fp32.size(0)):
            p_solved.append(
                torch.linalg.solve_triangular(
                    R_full[i], P_batch_fp32[i], upper=True, left=False
                )
            )
        P_batch_fp32 = torch.stack(p_solved, dim=0)
        _log_nonfinite("p_final", P_batch_fp32, f"m_local={m_shard_local} r={r} shard_world={shard_world_size}")
        del R_full, p_solved

        return P_batch_fp32.to(original_dtype).contiguous()

    # Batch update methods

    def _unshard_q_batch(
        self,
        Qs: List[Tensor],
        configs: List[DionParamConfig],
    ) -> Generator[List[Tensor], None, None]:
        """All-gather TP-sharded Q blocks needed for local matmul."""
        batch_size = len(Qs)
        Q_for_matmul: List[Optional[Tensor]] = [None] * batch_size
        tp_unshard_indices = [i for i, cfg in enumerate(configs) if self._needs_tp_q_unshard(cfg)]

        for i in range(batch_size):
            if i not in tp_unshard_indices:
                Q_for_matmul[i] = Qs[i]

        if tp_unshard_indices:
            tp_size = self.tp_world_size
            grouped_indices: Dict[Tuple[int, int, torch.dtype, torch.device], List[int]] = {}
            for i in tp_unshard_indices:
                q_local = Qs[i]
                key = (q_local.size(0), q_local.size(1), q_local.dtype, q_local.device)
                grouped_indices.setdefault(key, []).append(i)

            pending = []
            for group_seq, ((n, r_local, dtype, device), indices) in enumerate(grouped_indices.items()):
                local_batch = self._scratch_tensor(
                    f"q_local_batch_{group_seq}",
                    (len(indices), n, r_local),
                    dtype,
                    device,
                )
                for slot, idx in enumerate(indices):
                    local_batch[slot].copy_(Qs[idx])

                gathered_batch = self._scratch_tensor(
                    f"q_gather_batch_{group_seq}",
                    (tp_size, len(indices), n, r_local),
                    dtype,
                    device,
                )
                handle = dist.all_gather_into_tensor(
                    gathered_batch.view(-1),
                    local_batch.view(-1),
                    group=self.tp_group,
                    async_op=True,
                )
                pending.append((indices, n, r_local, local_batch, gathered_batch, handle))

            yield

            for indices, n, r_local, local_batch, gathered_batch, handle in pending:
                handle.wait()
                q_full_batch = (
                    gathered_batch.permute(1, 2, 0, 3)
                    .contiguous()
                    .view(len(indices), n, tp_size * r_local)
                )
                if _DION_DIAG and self._step_count <= _DION_DIAG_MAX_STEPS:
                    warn_count = getattr(self, "_q_tp_mismatch_warn_count", 0)
                    if warn_count < 16:
                        tp_group_ranks = dist.get_process_group_ranks(self.tp_group)
                        broadcast_src = tp_group_ranks[0]
                        ref_batch = q_full_batch.detach().clone()
                        dist.broadcast(ref_batch, src=broadcast_src, group=self.tp_group)
                        q_full_f = q_full_batch.float()
                        ref_f = ref_batch.float()
                        finite = torch.isfinite(q_full_f) & torch.isfinite(ref_f)
                        if finite.any():
                            max_diff = (q_full_f[finite] - ref_f[finite]).abs().max().item()
                        else:
                            max_diff = float("inf")
                        if max_diff > 1e-6:
                            setattr(self, "_q_tp_mismatch_warn_count", warn_count + 1)
                            logger.warning(
                                "[DION_Q_TP_MISMATCH] rank=%s step=%s shape=%s max_diff=%.6e src=%s",
                                self._global_rank,
                                self._step_count,
                                tuple(q_full_batch.shape),
                                max_diff,
                                broadcast_src,
                            )
                for slot, idx in enumerate(indices):
                    Q_for_matmul[idx] = q_full_batch[slot]
                del local_batch

        return [q for q in Q_for_matmul]

    def _reduce_p_across_fs_groups(
        self,
        P_batch: torch.Tensor,
        configs: List[DionParamConfig],
        metas: List,
    ) -> Generator[None, None, None]:
        """Reduce STEP 3.5 P batches across the correct FS shard groups."""
        if not self.use_fs_collectives:
            return

        need_fs = [i for i, cfg in enumerate(configs) if self._needs_fs_p_reduce(cfg)]
        if not need_fs:
            return

        group_to_indices = {}
        for i in need_fs:
            meta = metas[i] if i < len(metas) else None
            shard_group = self._fs_group_for_meta(meta)
            group_id = id(shard_group) if shard_group else 0
            if group_id not in group_to_indices:
                group_to_indices[group_id] = (shard_group, [])
            group_to_indices[group_id][1].append(i)

        did_yield = False
        batch_size = len(configs)
        for group, indices in group_to_indices.values():
            if group and dist.get_world_size(group) > 1:
                cp_world_size = parallel_state.get_context_parallel_world_size()
                do_cp_debug = False
                if cp_world_size > 1:
                    debug_count = getattr(self, "_cp_p_reduce_debug_count", 0)
                    if debug_count < 8:
                        setattr(self, "_cp_p_reduce_debug_count", debug_count + 1)
                        do_cp_debug = True
                        try:
                            group_ranks = dist.get_process_group_ranks(group)
                        except Exception:
                            group_ranks = []
                        logger.info(
                            "[DION_CP_P_REDUCE] global_rank=%s seq=%s group=%s batch=%s indices=%s shape=%s dtype=%s contiguous=%s",
                            self._global_rank,
                            debug_count,
                            group_ranks,
                            batch_size,
                            indices,
                            tuple(P_batch.shape),
                            P_batch.dtype,
                            bool(P_batch.is_contiguous()),
                        )
                        torch.cuda.synchronize(P_batch.device)
                        logger.info(
                            "[DION_CP_P_REDUCE_READY] global_rank=%s group=%s shape=%s",
                            self._global_rank,
                            group_ranks,
                            tuple(P_batch.shape),
                        )
                if len(indices) == batch_size and len(group_to_indices) == 1:
                    dist.all_reduce(P_batch, op=dist.ReduceOp.SUM, group=group)
                else:
                    tensors = [P_batch[idx] for idx in indices]
                    reduced = funcol.all_reduce_coalesced(tensors, reduceOp="sum", group=group)
                    for idx, tensor in zip(indices, reduced):
                        P_batch[idx].copy_(tensor)
                    del tensors, reduced
                if do_cp_debug:
                    torch.cuda.synchronize(P_batch.device)
                    logger.info(
                        "[DION_CP_P_REDUCE_DONE] global_rank=%s group=%s shape=%s",
                        self._global_rank,
                        group_ranks,
                        tuple(P_batch.shape),
                    )
                if not did_yield:
                    yield
                    did_yield = True

        if not did_yield:
            yield

    def _reduce_r_across_tp(
        self,
        R_batch: torch.Tensor,
        configs: List[DionParamConfig],
    ) -> Generator[None, None, None]:
        """Reduce STEP 5 R batches across TP when required."""
        if not (self.tp_group and dist.get_world_size(self.tp_group) > 1):
            return

        need_tp_R = [i for i, cfg in enumerate(configs) if self._needs_tp_r_reduce(cfg)]
        if not need_tp_R:
            return

        if len(need_tp_R) == R_batch.size(0):
            dist.all_reduce(R_batch, op=dist.ReduceOp.SUM, group=self.tp_group)
        else:
            tensors = [R_batch[i] for i in need_tp_R]
            reduced = funcol.all_reduce_coalesced(tensors, reduceOp="sum", group=self.tp_group)
            for i, tensor in zip(need_tp_R, reduced):
                R_batch[i].copy_(tensor)
            del tensors, reduced

        yield

    def _orthogonalize_chunked_slice(
        self,
        P_slice: torch.Tensor,
        *,
        shard_group,
        chunk_world_size: int,
    ) -> torch.Tensor:
        """Run distributed orthogonalization in fixed-size chunks for TP/FS shard groups."""
        slice_size = P_slice.size(0)
        P_ortho = torch.empty_like(P_slice)

        for chunk_start in range(0, slice_size, chunk_world_size):
            chunk_end = min(chunk_start + chunk_world_size, slice_size)
            chunk_size = chunk_end - chunk_start
            P_chunk = P_slice[chunk_start:chunk_end]

            original_chunk_size = chunk_size
            if chunk_size < chunk_world_size:
                pad_size = chunk_world_size - chunk_size
                m_local = P_chunk.size(1)
                r_val = P_chunk.size(2)
                padded_chunk = self._scratch_tensor(
                    "p_chunk_pad",
                    (chunk_world_size, m_local, r_val),
                    P_chunk.dtype,
                    P_chunk.device,
                )
                padded_chunk[:chunk_size].copy_(P_chunk)
                for i in range(pad_size):
                    torch.nn.init.orthogonal_(padded_chunk[chunk_size + i])
                    padded_chunk[chunk_size + i] *= 1e-6
                P_chunk = padded_chunk

            P_chunk_ortho = self._distributed_orthogonalize(
                P_chunk,
                shard_group=shard_group,
                oversample=self.defaults['rcqr_oversample'],
            )
            P_ortho[chunk_start:chunk_end].copy_(P_chunk_ortho[:original_chunk_size])
            del P_chunk

        return P_ortho if slice_size > 0 else P_slice

    def _orthogonalize_local_slice(
        self,
        P_slice: torch.Tensor,
        metas: Optional[List] = None,
        *,
        tag: str = "local",
    ) -> torch.Tensor:
        """Local fallback orthogonalization for a batched slice."""
        if P_slice.size(0) == 0:
            return P_slice

        P_ortho_slice = torch.empty_like(P_slice, dtype=torch.float32)
        for i in range(P_slice.size(0)):
            sketch_fn = None
            if metas is not None and i < len(metas):
                sketch_fn = self._seeded_sketch_fn(metas=[metas[i]], tag=tag)
            P_ortho_slice[i] = self._orthogonalize(
                P_slice[i],
                rcqr_oversample=self.defaults['rcqr_oversample'],
                sketch_fn=sketch_fn,
            )
        return P_ortho_slice.to(P_slice.dtype)

    @staticmethod
    def _is_fs_only_config(config: DionParamConfig) -> bool:
        """Return whether the config follows dion_reference.py fs-only semantics."""
        return config.has_fs_axis and not config.has_tp_axis

    def _orthogonalize_fs_only_batch(
        self,
        P_batch: torch.Tensor,
        metas: List,
    ) -> Generator[torch.Tensor, None, None]:
        """Match `dion_reference.dion_update_fsdp()` for fs-only batches.

        Reference contract:
        - batch size equals outer shard mesh size
        - collapse partial P by reduce-scattering along batch dim
        - each rank orthogonalizes one full matrix in the batch
        - all-gather the orthogonalized batch back before computing R
        """
        meta0 = metas[0] if metas else None
        shard_group = self._fs_group_for_meta(meta0)
        if shard_group is None or dist.get_world_size(shard_group) <= 1:
            yield
            return self._orthogonalize_local_slice(P_batch, metas=metas, tag="fs_only_local")

        shard_world = dist.get_world_size(shard_group)
        batch_size = P_batch.size(0)
        if batch_size != shard_world:
            raise RuntimeError(
                "[DION_FSONLY_BATCH_SIZE_MISMATCH] "
                f"batch_size={batch_size} shard_world_size={shard_world}"
            )

        # Reference FSDP path: collapse partial P onto one full matrix per rank.
        P_single = funcol.reduce_scatter_tensor(
            P_batch.contiguous(),
            reduceOp="sum",
            scatter_dim=0,
            group=shard_group,
        )
        yield

        shard_rank = dist.get_rank(shard_group)
        local_meta = [metas[shard_rank]] if metas and shard_rank < len(metas) else None
        P_single_ortho = self._orthogonalize_local_slice(
            P_single,
            metas=local_meta,
            tag="fs_only",
        )

        # Gather the orthogonalized batch back, matching dion_update_fsdp().
        P_ortho = funcol.all_gather_tensor(
            P_single_ortho.contiguous(),
            gather_dim=0,
            group=shard_group,
        )
        yield
        return P_ortho

    def _orthogonalize_p_batch(
        self,
        P_batch: torch.Tensor,
        configs: List[DionParamConfig],
        metas: List,
    ) -> Generator[torch.Tensor, None, None]:
        """Orthogonalize a P batch using the correct local or distributed path."""
        if configs and all(self._is_fs_only_config(cfg) for cfg in configs):
            P_batch = yield from self._orthogonalize_fs_only_batch(P_batch, metas)
            return P_batch

        ortho_group = self._ortho_group_for_config(configs[0], metas[0] if metas else None)
        if ortho_group is not None:
            P_batch = self._distributed_orthogonalize(
                P_batch,
                shard_group=ortho_group,
                oversample=self.defaults['rcqr_oversample'],
                metas=metas,
            )
        else:
            for i in range(P_batch.size(0)):
                P_batch[i] = self._orthogonalize(
                    P_batch[i], rcqr_oversample=self.defaults['rcqr_oversample']
                )
        yield
        return P_batch

    def _apply_batch_updates(
        self,
        params: List[Tensor],
        momentums: List[Tensor],
        Qs: List[Tensor],
        Q_new_batch: torch.Tensor,
        P_batch: torch.Tensor,
        configs: List[DionParamConfig],
        groups: List[dict],
        states: List[dict],
        metas: List,
        param_shapes: List[Tuple[int, int]],
        real_batch_size: int,
        *,
        diag_log=None,
        diag_rank=None,
        diag_key=None,
    ) -> None:
        """Apply weight decay, Dion delta update, and TP re-sharding for Q."""
        Q_new_f32 = Q_new_batch[:real_batch_size].float()
        P_for_delta = P_batch[:real_batch_size]
        is_transposed = configs[0].is_transposed
        if all(c.is_transposed == is_transposed for c in configs[:real_batch_size]):
            if is_transposed:
                delta_batch = torch.bmm(Q_new_f32, P_for_delta.transpose(1, 2))
            else:
                delta_batch = torch.bmm(P_for_delta, Q_new_f32.transpose(1, 2))
        else:
            delta_shape = params[0].shape
            delta_batch = torch.empty(
                (real_batch_size, *delta_shape),
                dtype=Q_new_f32.dtype,
                device=Q_new_f32.device,
            )

            transposed_indices = [
                i for i, cfg in enumerate(configs[:real_batch_size]) if cfg.is_transposed
            ]
            regular_indices = [
                i for i, cfg in enumerate(configs[:real_batch_size]) if not cfg.is_transposed
            ]

            if regular_indices:
                regular_delta = torch.bmm(
                    P_for_delta[regular_indices],
                    Q_new_f32[regular_indices].transpose(1, 2),
                )
                delta_batch[regular_indices].copy_(regular_delta)
                del regular_delta

            if transposed_indices:
                transposed_delta = torch.bmm(
                    Q_new_f32[transposed_indices],
                    P_for_delta[transposed_indices].transpose(1, 2),
                )
                delta_batch[transposed_indices].copy_(transposed_delta)
                del transposed_delta
        del Q_new_f32, P_for_delta

        state0 = states[0]
        if 'true_global_shape' in state0:
            m_global, n_global = state0['true_global_shape']
        else:
            m_global, n_global = self._get_global_shape(metas[0], param_shapes[0][0], param_shapes[0][1])

        m_for_lr, n_for_lr = m_global, n_global
        if 'per_expert_global_shape' in state0:
            m_for_lr, n_for_lr = state0['per_expert_global_shape']

        lr = groups[0].get('lr', self.defaults['lr'])

        rank_fraction = self.defaults.get('rank_fraction', 0.25)
        base_scale = 0.2 / (rank_fraction ** 0.5)
        scaled_lr = base_scale * (max(m_for_lr, n_for_lr) ** 0.5) * lr

        wd_mult = groups[0].get('wd_mult', 1.0)
        weight_decay = self.defaults['weight_decay'] * wd_mult
        has_tp = configs[0].has_tp_axis and self.tp_group is not None

        if diag_log is not None:
            diag_log("S8_Q_new", Q_new_batch)
            diag_log("S9_delta", delta_batch)
            print(
                f"[DION_DIAG] rank={diag_rank} {diag_key} LR: global=({m_global},{n_global}) "
                f"scaled_lr={scaled_lr:.6e} lr={lr:.6e} wd={weight_decay:.6e}",
                flush=True,
            )

        for i in range(real_batch_size):
            param = params[i]
            delta = delta_batch[i]
            if delta.shape != param.shape:
                delta = delta.contiguous().view(param.shape)

            if weight_decay > 0:
                param.mul_(1 - lr * weight_decay)
            param.add_(delta.to(param.dtype), alpha=-scaled_lr)

            Q_new = Q_new_batch[i].to(Qs[i].dtype)
            if has_tp:
                Q_new = self._reshard_q_along_tp(Q_new, self.tp_group, self.tp_rank)
            Qs[i].copy_(Q_new)

        if diag_log is not None:
            diag_log("S9_params_after", params)

        del delta_batch

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

        # Diagnostic logging helper
        _diag = _DION_DIAG and self._step_count <= _DION_DIAG_MAX_STEPS
        if _diag:
            _rank = self._global_rank
            _step = self._step_count
            _key = f"step{_step}_off{global_param_offset}"
            _cfg0 = configs[0]
            _p_tp0 = self._p_is_tp_sharded(_cfg0)
            _p_fs0 = self._p_is_fs_sharded(_cfg0)
            _need_fs_p0 = self._needs_fs_p_reduce(_cfg0)
            _need_tp_r0 = self._needs_tp_r_reduce(_cfg0)
            _batch_ids = []
            for meta in metas[:real_batch_size]:
                if meta is None:
                    _batch_ids.append(("local", None))
                else:
                    _batch_ids.append(
                        (
                            getattr(meta, "param_name", "") or f"id_{id(meta)}",
                            getattr(meta, "param_uid", None),
                        )
                    )
            def _log(tag, tensor, extra=""):
                if isinstance(tensor, torch.Tensor):
                    view = tensor
                    if tensor.dim() > 0 and tensor.size(0) == batch_size:
                        view = tensor[:real_batch_size]
                    _shape = tuple(view.shape)
                    if view.dim() > 0 and view.size(0) == real_batch_size:
                        flat = view.float().reshape(real_batch_size, -1)
                        norms = [row.norm().item() for row in flat]
                        maxs = [row.abs().max().item() for row in flat]
                        print(
                            f"[DION_DIAG] rank={_rank} {_key} {tag}: "
                            f"real_norms={[f'{n:.6e}' for n in norms]} "
                            f"real_maxs={[f'{m:.6e}' for m in maxs]} "
                            f"shape={_shape} {extra}",
                            flush=True,
                        )
                    else:
                        _norm = view.float().norm().item()
                        _max = view.float().abs().max().item()
                        print(
                            f"[DION_DIAG] rank={_rank} {_key} {tag}: "
                            f"norm={_norm:.6e} max={_max:.6e} shape={_shape} {extra}",
                            flush=True,
                        )
                elif isinstance(tensor, list) and len(tensor) > 0 and isinstance(tensor[0], torch.Tensor):
                    norms = [t.float().norm().item() for t in tensor[:real_batch_size]]
                    print(f"[DION_DIAG] rank={_rank} {_key} {tag}: norms={[f'{n:.6e}' for n in norms]} {extra}", flush=True)
            _log("config", params[0], f"has_fs={_cfg0.has_fs_axis} has_tp={_cfg0.has_tp_axis} is_tr={_cfg0.is_transposed} inner={_cfg0.inner_shard_tensor_dim} outer={_cfg0.outer_shard_tensor_dim} p_tp={_p_tp0} p_fs={_p_fs0} need_fs_p={_need_fs_p0} need_tp_r={_need_tp_r0} bs={batch_size} rbs={real_batch_size}")
            print(
                f"[DION_DIAG] rank={_rank} {_key} batch_ids={_batch_ids}",
                flush=True,
            )

        use_compressed = self.use_compressed_comm and any(c.compressed_all_reduce for c in configs)

        if not use_compressed:
            yield from self._collapse_grads_across_cp(grads)

        # STEP 1: M <- M + G (no decay - decay is applied in error feedback)
        if grads:
            # All momentums in a batch have the same dtype (set by mixed_precision_config)
            if momentums[0].dtype == grads[0].dtype:
                torch._foreach_add_(momentums, grads)
            else:
                for m, g in zip(momentums, grads):
                    m.add_(g.to(m.dtype))

        if _diag:
            _log("S1_grad", grads)
            _log("S1_M_after", momentums)
        self._diag_state_replica_values(
            "post_s1",
            metas=metas,
            grads=grads,
            momentums=momentums,
            Qs=Qs,
            params=params,
            real_batch_size=real_batch_size,
        )
        self._diag_global_state_fp(
            "post_s1",
            metas=metas,
            grads=grads,
            momentums=momentums,
            Qs=Qs,
            params=params,
            real_batch_size=real_batch_size,
        )

        # STEP 2: Q Unshard
        Q_for_matmul = yield from self._unshard_q_batch(Qs, configs)
        cp_world_size = parallel_state.get_context_parallel_world_size()
        cp_debug_count = getattr(self, "_cp_batch_debug_count", 0)
        do_cp_batch_debug = cp_world_size > 1 and cp_debug_count < 6
        if do_cp_batch_debug:
            setattr(self, "_cp_batch_debug_count", cp_debug_count + 1)
            torch.cuda.synchronize(momentums[0].device)
            logger.info(
                "[DION_CP_BATCH_Q_READY] global_rank=%s seq=%s real_batch=%s q0_shape=%s",
                self._global_rank,
                cp_debug_count,
                real_batch_size,
                tuple(Q_for_matmul[0].shape),
            )

        # STEP 3: P = M @ Q
        M_batch = torch.stack(
            [
                (momentum.mT if config.is_transposed else momentum).to(torch.float32)
                for momentum, config in zip(momentums, configs)
            ],
            dim=0,
        )
        Q_batch = torch.stack([q.to(torch.float32) for q in Q_for_matmul], dim=0)
        del Q_for_matmul

        P_batch = M_batch @ Q_batch
        if do_cp_batch_debug:
            torch.cuda.synchronize(P_batch.device)
            logger.info(
                "[DION_CP_BATCH_P_READY] global_rank=%s seq=%s p_shape=%s",
                self._global_rank,
                cp_debug_count,
                tuple(P_batch.shape),
            )

        if _diag:
            _log("S3_M_batch", M_batch)
            _log("S3_Q_batch", Q_batch)
            _log("S3_P_pre_ar", P_batch)
        self._diag_global_batch_fp(
            "s3_q_batch",
            metas=metas,
            batch=Q_batch,
            real_batch_size=real_batch_size,
        )
        self._diag_global_batch_fp(
            "s3_p_pre_ar",
            metas=metas,
            batch=P_batch,
            real_batch_size=real_batch_size,
        )

        # STEP 3.5: All-Reduce for P (only if FS collectives are enabled)
        # NOTE: Expert params use different shard_group (EP-internal), so we group by shard_group
        if not (configs and all(self._is_fs_only_config(cfg) for cfg in configs)):
            yield from self._reduce_p_across_fs_groups(P_batch, configs, metas)

        # STEP 4: Orthogonalize P
        if use_compressed:
            P_batch, R_batch = yield from self._run_compressed_comm_async(
                P_batch, M_batch, Q_batch, configs, metas
            )
        else:
            P_batch = yield from self._orthogonalize_p_batch(P_batch, configs, metas)
            # STEP 5: R = M.T @ P
            R_batch = M_batch.mT @ P_batch
            yield from self._reduce_r_across_tp(R_batch, configs)

        if _diag:
            _log("S4_P_ortho", P_batch)
            _log("S5_R_batch", R_batch)
        self._diag_global_batch_fp(
            "s4_p_ortho",
            metas=metas,
            batch=P_batch,
            real_batch_size=real_batch_size,
        )
        self._diag_global_batch_fp(
            "s5_r_batch",
            metas=metas,
            batch=R_batch,
            real_batch_size=real_batch_size,
        )

        # STEP 6: Fix NaN/zero
        P_batch, R_batch = self._fix_bad_batch(
            P_batch,
            R_batch,
            Q_batch,
            M_batch,
            real_batch_size=real_batch_size,
            global_param_offset=global_param_offset,
            configs=configs,
        )

        # STEP 7: Error feedback
        self._batch_error_feedback(momentums, P_batch, R_batch, configs, groups)

        # STEP 8: Column normalize R -> Q_new
        Q_new_batch = yield from self._normalize_cols_async(R_batch, configs, metas, real_batch_size, global_param_offset)
        self._diag_global_batch_fp(
            "s8_q_new",
            metas=metas,
            batch=Q_new_batch,
            real_batch_size=real_batch_size,
        )

        if _diag:
            _log("S7_M_after_ef", momentums)
        self._apply_batch_updates(
            params,
            momentums,
            Qs,
            Q_new_batch,
            P_batch,
            configs,
            groups,
            states,
            metas,
            param_shapes,
            real_batch_size,
            diag_log=_log if _diag else None,
            diag_rank=_rank if _diag else None,
            diag_key=_key if _diag else None,
        )
        self._diag_state_replica_values(
            "post_update",
            metas=metas,
            grads=grads,
            momentums=momentums,
            Qs=Qs,
            params=params,
            real_batch_size=real_batch_size,
        )
        self._diag_global_state_fp(
            "post_update",
            metas=metas,
            grads=grads,
            momentums=momentums,
            Qs=Qs,
            params=params,
            real_batch_size=real_batch_size,
        )

        del M_batch, Q_batch, P_batch, R_batch, Q_new_batch

    def _run_compressed_comm_async(
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
        cfg0 = configs[0] if configs else DionParamConfig()
        meta0 = metas[0] if metas else None
        ortho_group = self._ortho_group_for_config(cfg0, meta0)

        if comm_group is None or comm_world_size <= 1:
            yield from self._collapse_batch_across_cp(P_batch)
            # RP=1: no P all-reduce needed, but orthogonalization is always required
            if ortho_group is not None:
                P_ortho = self._distributed_orthogonalize(
                    P_batch,
                    shard_group=ortho_group,
                    oversample=self.defaults['rcqr_oversample'],
                )
            else:
                P_ortho = P_batch.clone()
                for i in range(P_batch.size(0)):
                    P_ortho[i] = self._orthogonalize(
                        P_batch[i], rcqr_oversample=self.defaults['rcqr_oversample']
                    )

            yield

            R_batch = M_batch.mT @ P_ortho
            yield from self._reduce_r_across_tp(R_batch, configs)
            yield from self._collapse_batch_across_cp(R_batch)

            return P_ortho, R_batch

        # Rest of compressed communication for RP > 1
        batch_size = P_batch.size(0)

        original_batch_size = batch_size

        if batch_size % comm_world_size != 0:
            pad = comm_world_size - (batch_size % comm_world_size)
            padded_batch_size = batch_size + pad
            P_padded = self._scratch_tensor(
                "compressed_p_padded",
                (padded_batch_size, P_batch.size(1), P_batch.size(2)),
                P_batch.dtype,
                P_batch.device,
                zero=True,
            )
            M_padded = self._scratch_tensor(
                "compressed_m_padded",
                (padded_batch_size, M_batch.size(1), M_batch.size(2)),
                M_batch.dtype,
                M_batch.device,
                zero=True,
            )
            Q_padded = self._scratch_tensor(
                "compressed_q_padded",
                (padded_batch_size, Q_batch.size(1), Q_batch.size(2)),
                Q_batch.dtype,
                Q_batch.device,
                zero=True,
            )
            P_padded[:batch_size].copy_(P_batch)
            M_padded[:batch_size].copy_(M_batch)
            Q_padded[:batch_size].copy_(Q_batch)

            P_batch = P_padded
            M_batch = M_padded
            Q_batch = Q_padded

            if isinstance(configs, list):
                dummy_cfg = DionParamConfig()
                configs = list(configs) + [dummy_cfg] * pad
            if isinstance(metas, list):
                metas = list(metas) + [None] * pad
            batch_size = P_batch.size(0)

        P_slice = P_batch
        cfg = configs[0] if configs else DionParamConfig()

        p_is_tp_sharded = self._p_is_tp_sharded(cfg)
        p_is_fs_sharded = (not p_is_tp_sharded) and self._p_is_fs_sharded(cfg)
        fs_group = self._fs_group_for_meta(metas[0] if metas else None)

        if p_is_tp_sharded and self.tp_group and dist.get_world_size(self.tp_group) > 1:
            P_ortho_slice = self._orthogonalize_chunked_slice(
                P_slice,
                shard_group=self.tp_group,
                chunk_world_size=dist.get_world_size(self.tp_group),
            )

        elif p_is_fs_sharded and self.use_fs_collectives and fs_group and dist.get_world_size(fs_group) > 1:
            P_ortho_slice = self._orthogonalize_chunked_slice(
                P_slice,
                shard_group=fs_group,
                chunk_world_size=dist.get_world_size(fs_group),
            )

        else:
            P_ortho_slice = self._orthogonalize_local_slice(
                P_slice,
                metas=metas,
                tag="compressed_local",
            )

        P_ortho_full = P_ortho_slice

        R_batch = M_batch.mT @ P_ortho_full
        yield from self._reduce_r_across_tp(R_batch, configs)
        yield from self._collapse_batch_across_cp(R_batch)

        P_ortho_full = P_ortho_full[:original_batch_size]
        R_batch = R_batch[:original_batch_size]

        return P_ortho_full, R_batch

    def _fit_q_to_r_batch(self, Q_batch: Tensor, R_batch: Tensor) -> Tensor:
        """Fit Q to match R's rank dimension for adaptive rank recovery."""
        batch_size = Q_batch.size(0)
        base_dim = Q_batch.size(1)
        r_Q = Q_batch.size(2)
        r_R = R_batch.size(2)

        if r_R == r_Q:
            return Q_batch

        Q_fitted = self._scratch_tensor(
            "q_fit_fitted",
            (batch_size, base_dim, r_R),
            torch.float32,
            Q_batch.device,
            zero=True,
        )
        base = min(r_R, r_Q)
        Q_fitted[..., :base].copy_(Q_batch[..., :base].to(torch.float32))

        if r_R > base:
            add_cols = r_R - base
            add = self._scratch_tensor(
                "q_fit_add",
                (batch_size, base_dim, add_cols),
                torch.float32,
                Q_batch.device,
            )

            gen = torch.Generator(device=Q_batch.device)
            gen.manual_seed(0xD10F33D)
            add.normal_(mean=0.0, std=1.0, generator=gen)

            Q_cat = self._scratch_tensor(
                "q_fit_cat",
                (batch_size, base_dim, r_R + add_cols),
                torch.float32,
                Q_batch.device,
            )
            Q_cat[..., :r_R].copy_(Q_fitted)
            Q_cat[..., r_R:].copy_(add)
            Q_ortho, _ = torch.linalg.qr(Q_cat, mode='reduced')
            return Q_ortho.to(Q_batch.dtype).contiguous()

        return Q_fitted.to(Q_batch.dtype)

    def _fix_bad_batch(
        self,
        P_batch: Tensor,
        R_batch: Tensor,
        Q_batch: Tensor,
        M_batch: Tensor,
        *,
        real_batch_size: Optional[int] = None,
        global_param_offset: int = 0,
        configs: Optional[List[DionParamConfig]] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Batch NaN/zero handling aligned with dion_reference.fix_all_zero_or_nan()."""
        is_all_zero = (M_batch == 0).all(dim=(-2, -1), keepdim=True)
        has_nan = torch.isnan(P_batch).any(dim=(-2, -1), keepdim=True) | \
                  torch.isnan(R_batch).any(dim=(-2, -1), keepdim=True)
        unexpected_nan = has_nan & (~is_all_zero)
        real_slot_mask = None
        if real_batch_size is not None:
            real_slot_mask = torch.zeros(
                unexpected_nan.size(0), dtype=torch.bool, device=unexpected_nan.device
            )
            real_slot_mask[:real_batch_size] = True
        warn_mask = unexpected_nan.view(unexpected_nan.size(0), -1).any(dim=1)
        if real_slot_mask is not None:
            warn_mask = warn_mask & real_slot_mask
        if warn_mask.any():
            warn_count = getattr(self, "_unexpected_nan_warn_count", 0)
            if warn_count < 8:
                setattr(self, "_unexpected_nan_warn_count", warn_count + 1)
                bad_slots = []
                for idx in torch.nonzero(warn_mask, as_tuple=False).view(-1).tolist():
                    cfg = configs[idx] if (configs is not None and idx < len(configs)) else None
                    bad_slots.append(
                        {
                            "slot": int(idx),
                            "global_offset": int(global_param_offset + idx),
                            "is_real": bool(real_batch_size is None or idx < real_batch_size),
                            "shape": tuple(P_batch[idx].shape),
                            "is_tr": bool(cfg.is_transposed) if cfg is not None else None,
                            "inner": getattr(cfg, "inner_shard_tensor_dim", None) if cfg is not None else None,
                            "outer": getattr(cfg, "outer_shard_tensor_dim", None) if cfg is not None else None,
                        }
                    )
                logger.warning(
                    "[DION_UNEXPECTED_NAN] rank=%s step=%s count=%s real_batch_size=%s bad_slots=%s",
                    self._global_rank,
                    self._step_count,
                    int(unexpected_nan.sum().item()),
                    int(real_batch_size) if real_batch_size is not None else -1,
                    bad_slots,
                )
        not_all_zero = ~is_all_zero

        P_batch = P_batch.nan_to_num() * not_all_zero

        Q_clean = Q_batch.nan_to_num()
        Q_fitted = self._fit_q_to_r_batch(Q_clean, R_batch)
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
        """Apply error feedback to momentum."""
        # FIXED: According to Algorithm 4 in dion.pdf and dion_reference.py:
        # Error feedback should be: M = M - (1-mu) * (P @ R^T)
        # Get mu value (same for all params in this batch)
        mu = groups[0].get('mu', self.defaults['mu'])

        is_transposed = configs[0].is_transposed
        if all(c.is_transposed == is_transposed for c in configs):
            self._foreach_baddbmm_(
                momentums, P_batch, R_batch,
                alpha=-(1.0 - mu), beta=1.0,  # FIXED: Use -(1-mu) instead of -1.0
                transpose=is_transposed
            )
        else:
            for i, momentum in enumerate(momentums):
                if configs[i].is_transposed:
                    update = R_batch[i] @ P_batch[i].t()
                else:
                    update = P_batch[i] @ R_batch[i].t()

                momentum.add_(update, alpha=-(1.0 - mu))  # FIXED: Use -(1-mu) instead of -1.0
                del update

    def _normalize_cols_async(self, R_batch: Tensor, configs: List[DionParamConfig],
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

        result = torch.empty_like(R_batch)
        epsilon = self.defaults['epsilon']
        for i in range(batch_size):
            col_sum_sq = col_sum_sq_list[i]
            R = R_batch[i]
            col_norms = col_sum_sq.sqrt()
            result[i].copy_(R / (col_norms + epsilon))
            del col_norms

        del col_sum_sq_list

        return result
