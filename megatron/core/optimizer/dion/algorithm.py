"""Dion optimizer main class for Megatron-LM."""

import logging
import hashlib
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
from torch import Tensor
from torch.optim.optimizer import Optimizer
from torch.distributed.tensor import DeviceMesh, DTensor, Shard

from megatron.core import parallel_state

from ..distrib_dion.param_selection import is_dion_param
from .async_runtime import AsyncRuntime, AsyncTask
from .batching import BatchProcessor, build_batch_key, pad_batch
from .constants import (
    DEFAULT_LR,
    DEFAULT_MU,
    DEFAULT_WEIGHT_DECAY,
)
from .ortho import (
    _dion_ortho_precision_context,
    distributed_orthogonalize_dtensor_exact,
    orthogonalize,
    orthogonalize_dtensor_exact,
    reshard_q_along_tp,
)
from .scalar_opt import adamw_update, lion_update
from .types import DionMixedPrecisionConfig, DionParamConfig, MegatronDionDistMeta
from .utils import get_global_shape, str_to_dtype


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
        lr_scaling_rule: str = "moonlight",  # 2D Dion LR scaling rule ("moonlight" or "dion")
        local_batch_size: Optional[int] = None,  # Optional local-mode batching hint
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
        self._device_mesh_cache: Dict[int, DeviceMesh] = {}

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
        self.is_distributed_mode = False
        self.use_fs_collectives = use_fs_collectives

        # UID → param mapping for state lookup across offload/reload cycles
        self._uid_to_param: Dict[Tuple, Tensor] = {}

        # UID → dist_meta for deterministic distributed metadata binding
        self._dist_meta_by_uid: Dict[Tuple, any] = {}

        # Compressed communication support
        self.use_compressed_comm = use_compressed_comm

        # Batch processor for improved performance
        self.batch_processor = BatchProcessor(max_batch_size=local_batch_size)
        self.local_batch_size = local_batch_size

        # Async collectives
        self.enable_async = enable_async
        self.max_concurrent_tasks = max_concurrent_tasks

        # Mixed precision configuration
        if mixed_precision_config is None:
            mixed_precision_config = DionMixedPrecisionConfig()
        self._mixed_precision_config = mixed_precision_config

        if mixed_precision_config.momentum_dtype is None or mixed_precision_config.Q_dtype is None:
            raise RuntimeError(
                "[Dion] momentum_dtype and Q_dtype must be explicit. "
                f"got momentum={mixed_precision_config.momentum_dtype} "
                f"Q={mixed_precision_config.Q_dtype}"
            )

        # Update counters
        self._dion_update_count = 0
        self._adamw_update_count = 0
        self._step_count = 0

        self._buffer_cache: Dict[str, torch.Tensor] = {}
        self._ortho_sanity_enabled = os.getenv("DION_ORTHO_SANITY", "0") == "1"
        self._ortho_sanity_every = self._parse_positive_int_env("DION_ORTHO_SANITY_EVERY", 1)
        self._ortho_sanity_p_tol = self._parse_positive_float_env("DION_ORTHO_SANITY_P_TOL", 5e-3)
        self._ortho_sanity_q_tol = self._parse_positive_float_env("DION_ORTHO_SANITY_Q_TOL", 5e-3)
        self._ortho_sanity_mode = self._parse_choice_env(
            "DION_ORTHO_SANITY_MODE",
            "fail",
            ("fail", "warn"),
        )
        self._ortho_sanity_log = os.getenv("DION_ORTHO_SANITY_LOG", "0") == "1"
        self._ortho_sanity_trace = os.getenv("DION_ORTHO_SANITY_TRACE", "0") == "1"
        self._debug_every = self._parse_positive_int_env("DION_DEBUG_EVERY", 1)
        self._debug_fs_only = os.getenv("DION_DEBUG_FS_ONLY", "0") == "1"
        self._debug_fs_only_batch_trace = os.getenv("DION_DEBUG_FS_ONLY_BATCH_TRACE", "0") == "1"
        self._debug_batch_order = os.getenv("DION_DEBUG_BATCH_ORDER", "0") == "1"
        self._debug_large_r = os.getenv("DION_DEBUG_LARGE_R", "0") == "1"
        self._debug_large_r_min_r = self._parse_positive_int_env(
            "DION_DEBUG_LARGE_R_MIN_R", 512
        )
        self._debug_post_fix = os.getenv("DION_DEBUG_POST_FIX", "0") == "1"
        self._debug_compare_sketch = os.getenv("DION_DEBUG_COMPARE_SKETCH", "0") == "1"
        self._debug_verify_fs_route = os.getenv("DION_DEBUG_VERIFY_FS_ROUTE", "0") == "1"
        self._debug_dump_bad_ortho_input = os.getenv("DION_DEBUG_DUMP_BAD_ORTHO_INPUT", "0") == "1"
        self._debug_dump_bad_ortho_fro = self._parse_positive_float_env(
            "DION_DEBUG_DUMP_BAD_ORTHO_FRO", 1e-2
        )
        self._debug_dump_bad_ortho_once = False
        self._debug_lr_scaling = os.getenv("DION_DEBUG_LR_SCALING", "0") == "1"
        self._debug_lr_scaling_steps = self._parse_positive_int_env(
            "DION_DEBUG_LR_SCALING_STEPS", 1
        )
        self._debug_lr_scaling_logged_params: Set[str] = set()
        self._debug_trace_params = {
            name for name in os.getenv("DION_DEBUG_TRACE_PARAMS", "").split(",") if name
        }
        self._debug_trace_steps = self._parse_positive_int_env(
            "DION_DEBUG_TRACE_STEPS", 3
        )
        self._debug_dump_target_trace = os.getenv("DION_DEBUG_DUMP_TARGET_TRACE", "0") == "1"
        self._debug_dump_target_tags = {
            tag for tag in os.getenv("DION_DEBUG_DUMP_TARGET_TAGS", "").split(",") if tag
        }
        self._debug_dump_unsharded_q = os.getenv("DION_DEBUG_DUMP_UNSHARDED_Q", "0") == "1"
        self._debug_ortho_batch_trace = os.getenv("DION_DEBUG_ORTHO_BATCH_TRACE", "0") == "1"
        self._ortho_batch_trace_counter = 0

    @staticmethod
    def _parse_positive_int_env(name: str, default: int) -> int:
        value = os.getenv(name, str(default))
        try:
            parsed = int(value)
        except ValueError as exc:
            raise RuntimeError(f"[DION_INVALID_ENV] {name}={value!r} is not an int") from exc
        if parsed <= 0:
            raise RuntimeError(f"[DION_INVALID_ENV] {name} must be > 0, got {parsed}")
        return parsed

    @staticmethod
    def _parse_positive_float_env(name: str, default: float) -> float:
        value = os.getenv(name, str(default))
        try:
            parsed = float(value)
        except ValueError as exc:
            raise RuntimeError(f"[DION_INVALID_ENV] {name}={value!r} is not a float") from exc
        if parsed <= 0:
            raise RuntimeError(f"[DION_INVALID_ENV] {name} must be > 0, got {parsed}")
        return parsed

    @staticmethod
    def _parse_choice_env(name: str, default: str, allowed: Tuple[str, ...]) -> str:
        value = os.getenv(name, default)
        if value not in allowed:
            raise RuntimeError(
                f"[DION_INVALID_ENV] {name}={value!r} must be one of {allowed}"
            )
        return value

    def _should_run_debug_probe(self, enabled: bool) -> bool:
        return enabled and (self._step_count % self._debug_every == 0)

    @staticmethod
    def _format_meta_id(meta) -> Dict[str, Any]:
        if meta is None:
            return {"param_uid": None, "param_name": ""}
        return {
            "param_uid": getattr(meta, "param_uid", None),
            "param_name": getattr(meta, "param_name", ""),
        }

    def _log_debug_batch_order(
        self,
        *,
        tag: str,
        sync_group,
        batch_key=None,
        local_ids=None,
        canonical_ids=None,
        reorder_indices=None,
        local_keys=None,
        canonical_keys=None,
    ) -> None:
        if not self._should_run_debug_probe(self._debug_batch_order):
            return
        try:
            group_ranks = dist.get_process_group_ranks(sync_group) if sync_group is not None else []
        except Exception:
            group_ranks = []
        logger.info(
            "[%s] step=%d rank=%d group_ranks=%s batch_key=%s local_ids=%s canonical_ids=%s reorder_indices=%s local_keys=%s canonical_keys=%s",
            tag,
            self._step_count,
            self._global_rank,
            group_ranks,
            batch_key,
            local_ids,
            canonical_ids,
            reorder_indices,
            local_keys,
            canonical_keys,
        )

    def _log_large_r_batch(
        self,
        *,
        tag: str,
        configs: List["DionParamConfig"],
        metas: List,
        param_shapes: Optional[List[Tuple[int, int]]],
        Q_batch: torch.Tensor,
        real_batch_size: int,
    ) -> None:
        if not self._should_run_debug_probe(self._debug_large_r):
            return

        rows = []
        oversample = self.defaults["rcqr_oversample"]
        for idx in range(real_batch_size):
            r_local = int(Q_batch[idx].size(-1))
            if r_local < self._debug_large_r_min_r:
                continue
            local_shape = (
                tuple(param_shapes[idx])
                if param_shapes is not None and idx < len(param_shapes)
                else None
            )
            cfg = configs[idx]
            meta = metas[idx] if idx < len(metas) else None
            k = math.ceil(oversample * r_local / 128.0) * 128
            rows.append(
                {
                    "slot": idx,
                    "param_uid": getattr(meta, "param_uid", None) if meta is not None else None,
                    "param_name": getattr(meta, "param_name", "") if meta is not None else "",
                    "local_shape": local_shape,
                    "r_local": r_local,
                    "k": int(k),
                    "is_transposed": bool(cfg.is_transposed),
                    "has_fs": bool(cfg.has_fs_axis),
                    "has_tp": bool(cfg.has_tp_axis),
                    "outer": cfg.outer_shard_tensor_dim,
                    "inner": cfg.inner_shard_tensor_dim,
                }
            )

        if rows:
            logger.info(
                "[%s] step=%d rank=%d rows=%s",
                tag,
                self._step_count,
                self._global_rank,
                rows,
            )

    @staticmethod
    def _tensor_stats(tensor: Optional[Tensor]) -> dict:
        if tensor is None:
            return {"norm": None, "sum": None, "amax": None}
        flat = tensor.detach().float().reshape(-1)
        if flat.numel() == 0:
            return {"norm": 0.0, "sum": 0.0, "amax": 0.0}
        stats = {
            "norm": float(flat.norm().item()),
            "sum": float(flat.sum().item()),
            "amax": float(flat.abs().max().item()),
        }
        if os.getenv("DION_DEBUG_TRACE_HASH", "0") == "1":
            tensor_bytes = (
                tensor.detach().to(torch.float32).contiguous().cpu().numpy().tobytes()
            )
            stats["hash"] = hashlib.blake2b(tensor_bytes, digest_size=8).hexdigest()
        return stats

    def _maybe_log_target_trace(
        self,
        *,
        tag: str,
        metas: Optional[List],
        grads: Optional[List[Tensor]] = None,
        momentums: Optional[List[Tensor]] = None,
        qs: Optional[List[Tensor]] = None,
        p_batch: Optional[Tensor] = None,
        r_batch: Optional[Tensor] = None,
        q_new_batch: Optional[Tensor] = None,
        deltas: Optional[Tensor] = None,
        params: Optional[List[Tensor]] = None,
        real_batch_size: Optional[int] = None,
        extra: Optional[dict] = None,
    ) -> None:
        if not self._debug_trace_params:
            return
        if self._step_count > self._debug_trace_steps:
            return
        if metas is None:
            return
        limit = len(metas) if real_batch_size is None else min(real_batch_size, len(metas))
        rows = []
        for idx in range(limit):
            meta = metas[idx]
            param_name = getattr(meta, "param_name", "") if meta is not None else ""
            if "*" not in self._debug_trace_params and param_name not in self._debug_trace_params:
                continue
            row = {
                "slot": int(idx),
                "param_uid": getattr(meta, "param_uid", None) if meta is not None else None,
                "param_name": param_name,
            }
            if grads is not None and idx < len(grads):
                row["grad"] = self._tensor_stats(grads[idx])
            if momentums is not None and idx < len(momentums):
                row["momentum"] = self._tensor_stats(momentums[idx])
            if qs is not None and idx < len(qs):
                row["Q"] = self._tensor_stats(qs[idx])
            if p_batch is not None and idx < p_batch.size(0):
                row["P"] = self._tensor_stats(p_batch[idx])
            if r_batch is not None and idx < r_batch.size(0):
                row["R"] = self._tensor_stats(r_batch[idx])
            if q_new_batch is not None and idx < q_new_batch.size(0):
                row["Q_new"] = self._tensor_stats(q_new_batch[idx])
            if deltas is not None and idx < deltas.size(0):
                row["delta"] = self._tensor_stats(deltas[idx])
            if params is not None and idx < len(params):
                row["param"] = self._tensor_stats(params[idx])
            rows.append(row)
        if rows:
            logger.warning(
                "[DION_TARGET_TRACE] step=%d rank=%d tag=%s rows=%s extra=%s",
                self._step_count,
                self._global_rank,
                tag,
                rows,
                extra or {},
            )
            self._maybe_dump_target_trace_tensors(
                tag=tag,
                metas=metas,
                grads=grads,
                momentums=momentums,
                qs=qs,
                p_batch=p_batch,
                r_batch=r_batch,
                q_new_batch=q_new_batch,
                deltas=deltas,
                params=params,
                real_batch_size=real_batch_size,
                extra=extra,
            )

    def _maybe_dump_target_trace_tensors(
        self,
        *,
        tag: str,
        metas: List,
        grads: Optional[List[Tensor]],
        momentums: Optional[List[Tensor]],
        qs: Optional[List[Tensor]],
        p_batch: Optional[Tensor],
        r_batch: Optional[Tensor],
        q_new_batch: Optional[Tensor],
        deltas: Optional[Tensor],
        params: Optional[List[Tensor]],
        real_batch_size: Optional[int],
        extra: Optional[dict],
    ) -> None:
        if not self._debug_dump_target_trace:
            return
        if self._debug_dump_target_tags and tag not in self._debug_dump_target_tags:
            return

        limit = len(metas) if real_batch_size is None else min(real_batch_size, len(metas))
        dump_dir = self._debug_trace_dump_dir()
        dump_dir.mkdir(parents=True, exist_ok=True)

        rp_rank = None
        rp_group_ranks = None
        if self.rp_group is not None:
            rp_rank = dist.get_rank(self.rp_group)
            rp_group_ranks = list(dist.get_process_group_ranks(self.rp_group))
        rp_world_size = len(rp_group_ranks) if rp_group_ranks is not None else 1

        for idx in range(limit):
            meta = metas[idx]
            param_name = getattr(meta, "param_name", "") if meta is not None else ""
            if "*" not in self._debug_trace_params and param_name not in self._debug_trace_params:
                continue

            safe_name = param_name.replace(".", "_").replace("/", "_") or "unknown"
            dump_path = dump_dir / (
                f"dion_target_trace_rpw{rp_world_size}_step{self._step_count}_rank{self._global_rank}_"
                f"tag_{tag}_{safe_name}.pt"
            )
            payload = {
                "step": self._step_count,
                "rank": self._global_rank,
                "tag": tag,
                "slot": int(idx),
                "meta": self._format_meta_id(meta),
                "extra": dict(extra or {}),
                "rp_rank": rp_rank,
                "rp_group_ranks": rp_group_ranks,
                "grad": grads[idx].detach().cpu() if grads is not None and idx < len(grads) else None,
                "momentum": (
                    momentums[idx].detach().cpu()
                    if momentums is not None and idx < len(momentums)
                    else None
                ),
                "Q": qs[idx].detach().cpu() if qs is not None and idx < len(qs) else None,
                "P": p_batch[idx].detach().cpu() if p_batch is not None and idx < p_batch.size(0) else None,
                "R": r_batch[idx].detach().cpu() if r_batch is not None and idx < r_batch.size(0) else None,
                "Q_new": (
                    q_new_batch[idx].detach().cpu()
                    if q_new_batch is not None and idx < q_new_batch.size(0)
                    else None
                ),
                "delta": (
                    deltas[idx].detach().cpu()
                    if deltas is not None and idx < deltas.size(0)
                    else None
                ),
                "param": params[idx].detach().cpu() if params is not None and idx < len(params) else None,
            }
            torch.save(payload, dump_path)
            logger.warning(
                "[DION_TARGET_TRACE_DUMP] step=%d rank=%d tag=%s slot=%d path=%s meta=%s",
                self._step_count,
                self._global_rank,
                tag,
                idx,
                str(dump_path),
                payload["meta"],
            )

    def _compressed_replicate_group(self):
        """Return the true Dion compressed replicate group.

        In the Megatron-Core backend, CP is part of the standard gradient-construction
        contract, not the Dion compressed replicate mesh. Compressed P/R collapse should
        therefore run only over explicit RP replicas.
        """
        if self.rp_group is None:
            return None
        return self.rp_group if dist.get_world_size(self.rp_group) > 1 else None

    def _pure_dp_replicate_group(self):
        """Return the pure-DP group that excludes CP."""
        pure_dp_group = parallel_state.get_data_parallel_group(
            with_context_parallel=False,
            partial_data_parallel=False,
        )
        if pure_dp_group is None:
            raise RuntimeError(
                "[DION_COMPRESSED_MISSING_PURE_DP_GROUP] "
                f"step={self._step_count} rank={self._global_rank}"
            )
        return pure_dp_group if dist.get_world_size(pure_dp_group) > 1 else None

    def _reference_replicate_group_for_config(self, config: "DionParamConfig"):
        """Return the lifted reference replicate mesh for one Dion batch config.

        `dion_reference.py` models only TP / FS / RP. Under Megatron-Core, CP is
        the gradient-construction axis, not part of the Dion replicate mesh. The
        lifted replicate mesh must therefore remain the explicit Dion RP group for
        all dense / TP / FS routes; CP may change how the logical gradient is
        constructed, but it must not redefine the optimizer's replicate domain.
        """
        del config
        return self._compressed_replicate_group()

    def _fs_only_compressed_replicate_group(self):
        """Return the fs-only compressed replicate mesh.

        `dion_reference.py` models TP / FS / RP only. When CP>1 under Megatron-Core,
        CP is not part of Dion's replicate mesh; it is only the gradient-construction
        axis. For fs-only compressed batches, match the reference semantics by using:
        - explicit RP replicas when CP==1
        - pure DP replicas (without CP) when CP>1
        """
        replicate_group = self._compressed_replicate_group()
        if replicate_group is None:
            return None

        cp_group = parallel_state.get_context_parallel_group(check_initialized=False)
        cp_world_size = dist.get_world_size(cp_group) if cp_group is not None else 1
        if cp_world_size <= 1:
            return replicate_group

        pure_dp_group = parallel_state.get_data_parallel_group(
            with_context_parallel=False,
            partial_data_parallel=False,
        )
        if pure_dp_group is None:
            raise RuntimeError(
                "[DION_FSONLY_COMPRESSED_MISSING_PURE_DP_GROUP] "
                f"step={self._step_count} rank={self._global_rank} cp_world_size={cp_world_size}"
            )
        pure_dp_world_size = dist.get_world_size(pure_dp_group)
        if pure_dp_world_size <= 1:
            raise RuntimeError(
                "[DION_FSONLY_COMPRESSED_INVALID_PURE_DP_GROUP] "
                f"step={self._step_count} rank={self._global_rank} "
                f"cp_world_size={cp_world_size} pure_dp_world_size={pure_dp_world_size}"
            )
        return pure_dp_group

    def _fs_only_compressed_replicate_spec(
        self,
    ) -> Tuple[
        Optional[torch.distributed.ProcessGroup],
        Optional[List[int]],
    ]:
        """Resolve the fs-only compressed replicate collapse surface.

        Reference fs-only compressed semantics require averaging only across ranks
        that represent the same logical FS slot after reduce-scatter. Under
        Megatron-Core with CP>1, using the full pure-DP group can mix different
        FS slot positions (for example FS=8 on a 16-rank DP×CP domain). The
        correct lifted surface is therefore:
        - keep the standard RP inter-instance group as the primary gather domain
        - restrict averaging to ranks that also share the same CP index
        """
        replicate_group = self._compressed_replicate_group()
        if replicate_group is None:
            return None, None

        cp_group = parallel_state.get_context_parallel_group(check_initialized=False)
        cp_world_size = dist.get_world_size(cp_group) if cp_group is not None else 1
        if cp_world_size <= 1:
            return replicate_group, None

        pure_dp_group = parallel_state.get_data_parallel_group(
            with_context_parallel=False,
            partial_data_parallel=False,
        )
        if pure_dp_group is None:
            raise RuntimeError(
                "[DION_FSONLY_COMPRESSED_MISSING_PURE_DP_GROUP] "
                f"step={self._step_count} rank={self._global_rank} cp_world_size={cp_world_size}"
            )

        replicate_ranks = dist.get_process_group_ranks(replicate_group)
        pure_dp_ranks = set(dist.get_process_group_ranks(pure_dp_group))
        subset_ranks = [rank for rank in replicate_ranks if rank in pure_dp_ranks]
        if not subset_ranks:
            raise RuntimeError(
                "[DION_FSONLY_COMPRESSED_EMPTY_REPLICA_SUBSET] "
                f"step={self._step_count} rank={self._global_rank} "
                f"replicate_ranks={replicate_ranks} pure_dp_ranks={sorted(pure_dp_ranks)}"
            )
        if self._global_rank not in subset_ranks:
            raise RuntimeError(
                "[DION_FSONLY_COMPRESSED_RANK_NOT_IN_REPLICA_SUBSET] "
                f"step={self._step_count} rank={self._global_rank} "
                f"replicate_ranks={replicate_ranks} subset_ranks={subset_ranks}"
            )
        if len(subset_ranks) == len(replicate_ranks):
            return replicate_group, None
        return replicate_group, subset_ranks

    def _maybe_dump_unsharded_q_batch(
        self,
        *,
        metas: Optional[List],
        q_for_matmul: List[Tensor],
    ) -> None:
        if not self._debug_dump_unsharded_q or not metas:
            return

        dump_dir = self._debug_trace_dump_dir()
        dump_dir.mkdir(parents=True, exist_ok=True)

        for idx, meta in enumerate(metas):
            param_name = getattr(meta, "param_name", "") if meta is not None else ""
            if "*" not in self._debug_trace_params and param_name not in self._debug_trace_params:
                continue
            if idx >= len(q_for_matmul):
                continue

            safe_name = param_name.replace(".", "_").replace("/", "_") or "unknown"
            dump_path = dump_dir / (
                f"dion_unsharded_q_step{self._step_count}_rank{self._global_rank}_{safe_name}.pt"
            )
            payload = {
                "step": self._step_count,
                "rank": self._global_rank,
                "slot": int(idx),
                "meta": self._format_meta_id(meta),
                "Q_full": q_for_matmul[idx].detach().cpu(),
            }
            torch.save(payload, dump_path)
            logger.warning(
                "[DION_UNSHARDED_Q_DUMP] step=%d rank=%d slot=%d path=%s meta=%s",
                self._step_count,
                self._global_rank,
                idx,
                str(dump_path),
                payload["meta"],
            )

    def _broadcast_replicate_domain_(self, tensor: Tensor) -> None:
        """Broadcast optimizer state across the true Dion replicate domain.

        Optimizer-state replicas arising from standard partial distributed optimizer
        instances cannot be treated as a global batch/state sync group. The same
        inter-instance group can carry different parameter families, so only true
        Dion RP replicas participate here.
        """
        if self.rp_group is None or dist.get_world_size(self.rp_group) <= 1:
            return

        group_ranks = dist.get_process_group_ranks(self.rp_group)
        dist.broadcast(tensor, src=group_ranks[0], group=self.rp_group)

    def _device_mesh_for_group(self, group, mesh_dim_name: str) -> DeviceMesh:
        """Create and cache a 1D DeviceMesh for an existing process group."""
        if group is None:
            raise RuntimeError("[DION_DEVICE_MESH_MISSING_GROUP] process group is required")
        key = id(group)
        mesh = self._device_mesh_cache.get(key)
        if mesh is not None:
            return mesh
        group_ranks = dist.get_process_group_ranks(group)
        mesh = DeviceMesh.from_group(
            group=group,
            device_type="cuda",
            mesh=torch.tensor(group_ranks, dtype=torch.int64),
            mesh_dim_names=(mesh_dim_name,),
        )
        self._device_mesh_cache[key] = mesh
        return mesh

    def _broadcast_state_replicas_(self, tensor: Tensor) -> None:
        """Broadcast optimizer state across standard DO state replicas for one local shard."""
        if self.state_replica_group is None or self.state_replica_world_size <= 1:
            return

        group_ranks = dist.get_process_group_ranks(self.state_replica_group)
        dist.broadcast(tensor, src=group_ranks[0], group=self.state_replica_group)

    def _next_q_init_seed(
        self,
        *,
        config: "DionParamConfig",
        meta,
        q_global_shape: Tuple[int, int],
    ) -> int:
        """Resolve a topology-invariant RNG seed for one logical Q initialization.

        Reference DTensor init creates one logical global Q and shards it across FS/TP.
        PP/CP/SP are not Dion optimizer axes, so the logical Q seed must not depend on
        local parameter scan order or pipeline partitioning. Tie the seed to the base
        training seed and the logical parameter identity instead.
        """
        param_uid = getattr(meta, "param_uid", None) if meta is not None else None
        param_name = getattr(meta, "param_name", "") if meta is not None else ""
        if param_uid is None and not param_name:
            raise RuntimeError(
                "[DION_Q_INIT_SEED_ID_MISSING] logical Dion Q init requires param_uid or param_name"
            )

        try:
            from megatron.training.global_vars import get_args
            args = get_args()
            base_seed = int(args.seed)
        except Exception:
            # Training init seeds torch with `seed + 100 * pp_rank`; remove PP so the
            # logical Q seed matches across pipeline partitions even before args exist.
            pp_rank = parallel_state.get_pipeline_model_parallel_rank()
            base_seed = (int(torch.initial_seed()) - 100 * int(pp_rank)) % (2**63 - 1)

        seed_key = repr(
            (
                "dion_q_init",
                int(base_seed),
                param_uid if param_uid is not None else param_name,
                tuple(int(dim) for dim in q_global_shape),
                bool(config.is_transposed),
            )
        ).encode("utf-8")
        return int.from_bytes(
            hashlib.blake2b(seed_key, digest_size=8).digest(),
            "little",
        ) & ((1 << 63) - 1)

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

        logical_ids = self._logical_sketch_seed_keys(metas=metas, tag=tag)

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
                seed_key = repr(logical_id).encode("utf-8")
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
                seed_key = repr(logical_id).encode("utf-8")
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

        setattr(_make_sketch, "_dion_sketch_tag", tag)
        setattr(_make_sketch, "_dion_logical_seed_keys", tuple(logical_ids))
        setattr(
            _make_sketch,
            "_dion_batch_meta_ids",
            tuple(self._format_meta_id(meta) for meta in metas),
        )

        return _make_sketch

    def _logical_sketch_seed_keys(
        self,
        *,
        metas: Optional[List],
        tag: str,
    ) -> Optional[List[object]]:
        """Return topology-invariant logical sketch ids for one Dion update batch.

        Sketch generation must be keyed only by the logical parameter identity and
        optimizer step, not by ambient RNG state, node count, or batch composition.
        The returned objects are passed by value into the sketch generator so both
        local and distributed RCQR routes consume the same logical sketch contract.
        """
        if not metas:
            return None

        logical_ids: List[object] = []
        for meta in metas:
            if meta is None:
                logical_ids.append((tag, self._step_count, None))
                continue
            logical_ids.append(
                (
                    tag,
                    self._step_count,
                    getattr(meta, "param_uid", None),
                    getattr(meta, "param_name", ""),
                )
            )
        return logical_ids

    def _logical_local_sketch_fn(self, *, metas: Optional[List]):
        """Return the topology-invariant local sketch contract.

        A logical Dion update can reach local orthogonalization through multiple
        Megatron backend routes (dense replicate, fs-only, compressed local
        chunking). Those routes must not change the sketch seen by the logical
        parameter when the pre-orthogonalization P input is the same.
        """
        return self._seeded_sketch_fn(metas=metas, tag="logical_local")

    def _reference_fs_only_sketch_fn(
        self,
        *,
        shard_group: torch.distributed.ProcessGroup,
        shard_rank: int,
    ):
        """Reproduce reference DTensor sketch RNG for fs-only local orthogonalization.

        In `dion_reference.dion_update_fsdp()`, `P_single` is a batch-sharded DTensor
        with one local batch slot per shard-group rank. `orthogonalize(P_single)`
        therefore draws one logical global sketch tensor of shape
        `(shard_world_size, k, m)` and each rank receives its batch-shard slice.

        This helper matches that contract without materializing the full global
        sketch locally: all ranks share one seed, then advance the generator by
        `shard_rank` batch slices before drawing the local `(1, k, m)` shard.
        """

        shard_group_ranks = dist.get_process_group_ranks(shard_group)
        broadcast_src = shard_group_ranks[0]

        def _make_sketch(P: Tensor, oversample: float) -> Tensor:
            if P.ndim != 3:
                raise RuntimeError(
                    "[DION_FSONLY_SKETCH_INVALID_RANK] "
                    f"expected batched 3D tensor, got shape={tuple(P.shape)}"
                )
            if P.size(0) != 1:
                raise RuntimeError(
                    "[DION_FSONLY_SKETCH_INVALID_BATCH] "
                    f"expected local fs-only batch size 1, got shape={tuple(P.shape)}"
                )

            m = P.size(-2)
            r = P.size(-1)
            k = math.ceil(oversample * r / 128.0) * 128
            if k <= 0:
                raise RuntimeError(
                    f"[DION_INVALID_SKETCH_RANK] r={r} oversample={oversample} k={k}"
                )

            std = math.sqrt(1.0 / k)
            seed_tensor = torch.zeros((), device=P.device, dtype=torch.int64)
            if dist.get_rank() == broadcast_src:
                seed_tensor.random_()
            dist.broadcast(seed_tensor, src=broadcast_src, group=shard_group)
            self._broadcast_replicate_domain_(seed_tensor)

            gen = torch.Generator(device=P.device)
            gen.manual_seed(int(seed_tensor.item()))

            # Match drawing the global batch-sharded DTensor sketch and taking this
            # rank's contiguous local batch shard.
            for _ in range(shard_rank):
                torch.empty((1, k, m), device=P.device, dtype=P.dtype).normal_(
                    mean=0.0,
                    std=std,
                    generator=gen,
                )

            local_sketch = torch.empty((1, k, m), device=P.device, dtype=P.dtype)
            local_sketch.normal_(mean=0.0, std=std, generator=gen)
            return local_sketch

        setattr(_make_sketch, "_dion_sketch_tag", "reference_fs_only")

        return _make_sketch

    def _sync_state_replica_q_init_(self, dion_params: List[Tuple]) -> None:
        """Synchronize newly initialized Q state in canonical param_uid order.

        Lazy state init happens while scanning optimizer param groups, whose local order is
        not a valid cross-rank contract for standard DO state replicas. After we sort Dion
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
        *,
        replicate_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> Generator[None, None, None]:
        """Collapse true Dion replicate replicas on G for non-compressed batches."""
        if replicate_group is None:
            replicate_group = self._compressed_replicate_group()
        if replicate_group is None or not grads:
            return

        op = self._replicate_reduce_op()
        for grad in grads:
            if grad.is_contiguous():
                dist.all_reduce(grad, op=op, group=replicate_group)
                continue

            reduced_grad = grad.contiguous()
            dist.all_reduce(reduced_grad, op=op, group=replicate_group)
            grad.copy_(reduced_grad)
        yield

    def _collapse_batch_across_cp(
        self,
        batch: Tensor,
        *,
        replicate_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> Generator[None, None, None]:
        """Collapse true Dion replicate replicas on a batched tensor."""
        if replicate_group is None:
            replicate_group = self._compressed_replicate_group()
        if replicate_group is None:
            return

        dist.all_reduce(batch, op=self._replicate_reduce_op(), group=replicate_group)
        yield

    def _collapse_batch_over_rank_subset(
        self,
        batch: Tensor,
        *,
        primary_group: torch.distributed.ProcessGroup,
        subset_ranks: List[int],
    ) -> Generator[None, None, None]:
        """Average `batch` across a deterministic subset of an existing group."""
        if primary_group is None:
            raise RuntimeError("[DION_INVALID_PRIMARY_GROUP_FOR_SUBSET_COLLAPSE]")
        if not subset_ranks:
            raise RuntimeError("[DION_EMPTY_SUBSET_FOR_SUBSET_COLLAPSE]")

        primary_ranks = dist.get_process_group_ranks(primary_group)
        rank_to_index = {rank: idx for idx, rank in enumerate(primary_ranks)}
        subset_indices = [rank_to_index[rank] for rank in subset_ranks]

        gathered = funcol.all_gather_tensor(
            batch.contiguous(),
            gather_dim=0,
            group=primary_group,
        )
        yield

        gathered = gathered.view(len(primary_ranks), *batch.shape)
        index_tensor = torch.tensor(
            subset_indices,
            device=batch.device,
            dtype=torch.long,
        )
        reduced = gathered.index_select(0, index_tensor).mean(dim=0)
        batch.copy_(reduced)

    def _resolve_async_task_limit(self, task_count: int) -> int:
        """Resolve the async runtime width without relying on hidden global defaults."""
        if task_count <= 0:
            raise RuntimeError(f"[Dion] invalid async task_count={task_count}")
        if self.max_concurrent_tasks is None:
            return task_count
        if int(self.max_concurrent_tasks) <= 0:
            raise RuntimeError(
                f"[Dion] invalid max_concurrent_tasks={self.max_concurrent_tasks}"
            )
        return min(task_count, int(self.max_concurrent_tasks))

    def _should_run_ortho_sanity(self) -> bool:
        return self._ortho_sanity_enabled and (self._step_count % self._ortho_sanity_every == 0)

    def _format_sanity_targets(
        self,
        metas: Optional[List],
        per_slot_err: torch.Tensor,
        *,
        per_slot_fro: Optional[torch.Tensor] = None,
        limit: int = 4,
    ) -> List[dict]:
        if per_slot_err.numel() == 0:
            return []
        top_k = min(limit, per_slot_err.numel())
        top_vals, top_idx = torch.topk(per_slot_err, k=top_k)
        targets = []
        for idx, err in zip(top_idx.tolist(), top_vals.tolist()):
            meta = metas[idx] if metas is not None and idx < len(metas) else None
            targets.append(
                {
                    "slot": int(idx),
                    "param_uid": getattr(meta, "param_uid", None) if meta is not None else None,
                    "param_name": getattr(meta, "param_name", "") if meta is not None else "",
                    "err": float(err),
                }
            )
            if per_slot_fro is not None:
                targets[-1]["fro_norm"] = float(per_slot_fro[idx].item())
        return targets

    def _handle_ortho_sanity_result(
        self,
        *,
        tag: str,
        tol: float,
        max_err: float,
        max_fro_norm: float,
        metas: Optional[List],
        per_slot_err: torch.Tensor,
        per_slot_fro: torch.Tensor,
    ) -> None:
        targets = self._format_sanity_targets(metas, per_slot_err, per_slot_fro=per_slot_fro)
        if self._ortho_sanity_log:
            logger.info(
                "[%s] step=%d tol=%.6e max_err=%.6e max_fro_norm=%.6e targets=%s",
                tag,
                self._step_count,
                tol,
                max_err,
                max_fro_norm,
                targets,
            )
        if max_err <= tol:
            return
        message = (
            f"[{tag}_FAILED] "
            f"step={self._step_count} tol={tol} max_err={max_err:.6e} "
            f"max_fro_norm={max_fro_norm:.6e} targets={targets}"
        )
        if self._ortho_sanity_mode == "warn":
            logger.warning(message)
            return
        raise RuntimeError(message)

    def _log_p_orthogonality_snapshot(
        self,
        *,
        tag: str,
        P_batch: torch.Tensor,
        metas: Optional[List],
        real_batch_size: int,
        extra: Optional[dict] = None,
        force: bool = False,
    ) -> None:
        if force:
            if not self._should_run_debug_probe(self._debug_post_fix or self._debug_fs_only):
                return
        elif not self._ortho_sanity_trace or not self._should_run_ortho_sanity():
            return
        if real_batch_size <= 0:
            raise RuntimeError(f"[DION_ORTHO_TRACE_INVALID_BATCH] real_batch_size={real_batch_size}")

        P_real = P_batch[:real_batch_size].to(torch.float32)
        with _dion_ortho_precision_context():
            gram = torch.bmm(P_real.transpose(1, 2), P_real)
        if not torch.isfinite(gram).all():
            raise RuntimeError(
                "[DION_ORTHO_TRACE_NONFINITE_P] "
                f"tag={tag} step={self._step_count} real_batch_size={real_batch_size}"
            )

        r = gram.size(-1)
        eye = torch.eye(r, device=gram.device, dtype=gram.dtype).unsqueeze(0)
        diff = gram - eye
        diag_abs = diff.diagonal(dim1=-2, dim2=-1).abs()
        offdiag = diff.masked_fill(
            torch.eye(r, device=gram.device, dtype=torch.bool).unsqueeze(0),
            0.0,
        )
        offdiag_abs = offdiag.abs()
        per_slot_err = torch.maximum(diag_abs.max(dim=-1).values, offdiag_abs.amax(dim=(-2, -1)))
        per_slot_fro = torch.linalg.matrix_norm(diff, ord="fro", dim=(-2, -1))
        max_err = float(per_slot_err.max().item())
        max_fro_norm = float(per_slot_fro.max().item())
        logger.info(
            "[%s] step=%d rank=%d max_err=%.6e max_fro_norm=%.6e targets=%s extra=%s",
            tag,
            self._step_count,
            self._global_rank,
            max_err,
            max_fro_norm,
            self._format_sanity_targets(metas, per_slot_err, per_slot_fro=per_slot_fro),
            extra or {},
        )

    @staticmethod
    def _p_orthogonality_metrics(P: torch.Tensor) -> tuple[bool, float, float]:
        """Return finiteness, max-abs error, and Fro norm for P^T P - I."""
        with _dion_ortho_precision_context():
            P_fp32 = P.to(torch.float32)
            gram = P_fp32.mT @ P_fp32
        if not torch.isfinite(gram).all():
            return False, float("inf"), float("inf")
        r = gram.size(-1)
        diff = gram - torch.eye(r, device=gram.device, dtype=gram.dtype)
        diag_abs = diff.diagonal(dim1=-2, dim2=-1).abs()
        offdiag = diff.masked_fill(torch.eye(r, device=gram.device, dtype=torch.bool), 0.0)
        max_err = torch.maximum(diag_abs.max(), offdiag.abs().max())
        fro_norm = torch.linalg.matrix_norm(diff, ord="fro")
        return True, float(max_err.item()), float(fro_norm.item())

    @staticmethod
    def _repo_test_logs_dir() -> Path:
        return Path(__file__).resolve().parents[4] / "test_logs"

    @staticmethod
    def _debug_trace_dump_dir() -> Path:
        dump_dir_env = os.getenv("DION_DEBUG_TRACE_DUMP_DIR", "").strip()
        if dump_dir_env:
            return Path(dump_dir_env)
        return MegatronDion._repo_test_logs_dir()

    def _maybe_dump_bad_ortho_input(
        self,
        *,
        P_in: torch.Tensor,
        P_exact: Optional[torch.Tensor],
        P_local_ref: Optional[torch.Tensor],
        meta,
        tag: str,
        extra: Optional[dict] = None,
        force: bool = False,
    ) -> None:
        if not self._debug_dump_bad_ortho_input:
            return
        if self._debug_dump_bad_ortho_once and not force:
            return

        if P_exact is not None:
            _, exact_max_err, exact_fro = self._p_orthogonality_metrics(P_exact)
        else:
            exact_max_err, exact_fro = float("nan"), float("inf")
        if P_local_ref is not None:
            _, ref_max_err, ref_fro = self._p_orthogonality_metrics(P_local_ref)
        else:
            ref_max_err, ref_fro = float("nan"), float("nan")
        trigger_fro = exact_fro if math.isfinite(exact_fro) else ref_fro
        if not force and trigger_fro <= self._debug_dump_bad_ortho_fro:
            return

        dump_dir = self._repo_test_logs_dir()
        dump_dir.mkdir(parents=True, exist_ok=True)
        meta_id = self._format_meta_id(meta)
        param_name = meta_id.get("param_name", "") if meta_id is not None else ""
        safe_name = param_name.replace(".", "_").replace("/", "_") or "unknown"
        dump_path = dump_dir / (
            f"dion_bad_ortho_input_step{self._step_count}_rank{self._global_rank}_{tag}_{safe_name}.pt"
        )
        torch.save(
            {
                "step": self._step_count,
                "rank": self._global_rank,
                "tag": tag,
                "meta": meta_id,
                "extra": extra or {},
                "exact_max_err": exact_max_err,
                "exact_fro_norm": exact_fro,
                "local_ref_max_err": ref_max_err,
                "local_ref_fro_norm": ref_fro,
                "P_in": P_in.detach().cpu(),
                "P_exact": P_exact.detach().cpu() if P_exact is not None else None,
                "P_local_ref": P_local_ref.detach().cpu() if P_local_ref is not None else None,
            },
            dump_path,
        )
        self._debug_dump_bad_ortho_once = True
        logger.warning(
            "[DION_DEBUG_BAD_ORTHO_INPUT_DUMP] step=%d rank=%d tag=%s path=%s meta=%s "
            "exact_fro=%.6e local_ref_fro=%.6e",
            self._step_count,
            self._global_rank,
            tag,
            str(dump_path),
            meta_id,
            exact_fro,
            ref_fro,
        )

    def _maybe_compare_fs_only_exact(
        self,
        *,
        P_in: torch.Tensor,
        P_exact: torch.Tensor,
        meta,
        shard_group: torch.distributed.ProcessGroup,
        shard_rank: int,
        tag: str,
    ) -> None:
        if not self._debug_compare_sketch:
            return
        if not self._should_run_debug_probe(self._debug_compare_sketch):
            return
        if P_in.ndim != 3 or P_in.size(0) != 1:
            return
        if P_in.size(-1) < self._debug_large_r_min_r:
            return
        param_name = getattr(meta, "param_name", "") if meta is not None else ""
        if self._debug_trace_params and "*" not in self._debug_trace_params:
            if param_name not in self._debug_trace_params:
                return

        ortho_mesh = self._device_mesh_for_group(shard_group, "fs_only_exact_compare")
        p_dtensor = DTensor.from_local(
            P_in.contiguous(),
            device_mesh=ortho_mesh,
            placements=(Shard(0),),
        )
        local_ref = orthogonalize_dtensor_exact(
            p_dtensor,
            oversample=self.defaults['rcqr_oversample'],
            logical_seed_keys=self._logical_sketch_seed_keys(
                metas=[meta] if meta is not None else None,
                tag="logical_local",
            ),
            batch_meta_ids=[self._format_meta_id(meta)] if meta is not None else None,
        ).to_local().to(P_in.dtype)
        out_diff = (P_exact[0] - local_ref[0]).to(torch.float32)
        out_diff_finite = bool(torch.isfinite(out_diff).all().item())
        out_diff_max_abs = (
            float(out_diff.abs().max().item()) if out_diff_finite else float("inf")
        )
        out_diff_rel = (
            float(out_diff.norm().item())
            / max(
                float(P_exact[0].to(torch.float32).norm().item()),
                float(local_ref[0].to(torch.float32).norm().item()),
                1e-12,
            )
            if out_diff_finite
            else float("inf")
        )
        exact_finite, exact_max_err, exact_fro = self._p_orthogonality_metrics(P_exact[0])
        ref_finite, ref_max_err, ref_fro = self._p_orthogonality_metrics(local_ref[0])
        logger.info(
            "[DION_DEBUG_FS_ONLY_EXACT_COMPARE] step=%d rank=%d tag=%s meta=%s "
            "exact_finite=%s exact_max_err=%.6e exact_fro=%.6e "
            "local_ref_finite=%s local_ref_max_err=%.6e local_ref_fro=%.6e "
            "out_diff_finite=%s out_diff_rel=%.6e out_diff_max_abs=%.6e",
            self._step_count,
            self._global_rank,
            tag,
            self._format_meta_id(meta) if meta is not None else None,
            exact_finite,
            exact_max_err,
            exact_fro,
            ref_finite,
            ref_max_err,
            ref_fro,
            out_diff_finite,
            out_diff_rel,
            out_diff_max_abs,
        )
        self._maybe_dump_bad_ortho_input(
            P_in=P_in[0],
            P_exact=P_exact[0],
            P_local_ref=local_ref[0],
            meta=meta,
            tag=tag,
            extra={"shard_rank": shard_rank},
        )

    def _maybe_compare_local_logical(
        self,
        *,
        P_in: torch.Tensor,
        P_runtime: torch.Tensor,
        meta,
        tag: str,
    ) -> None:
        if not self._debug_compare_sketch:
            return
        if not self._should_run_debug_probe(self._debug_compare_sketch):
            return
        if P_in.ndim != 3 or P_in.size(0) != 1:
            return
        if P_in.size(-1) < self._debug_large_r_min_r:
            return
        param_name = getattr(meta, "param_name", "") if meta is not None else ""
        if self._debug_trace_params and "*" not in self._debug_trace_params:
            if param_name not in self._debug_trace_params:
                return

        logical_ref = orthogonalize(
            P_in.clone(),
            rcqr_oversample=self.defaults['rcqr_oversample'],
            sketch_fn=self._logical_local_sketch_fn(metas=[meta] if meta is not None else None),
        ).to(P_in.dtype)
        out_diff = (P_runtime[0] - logical_ref[0]).to(torch.float32)
        out_diff_finite = bool(torch.isfinite(out_diff).all().item())
        out_diff_max_abs = (
            float(out_diff.abs().max().item()) if out_diff_finite else float("inf")
        )
        out_diff_rel = (
            float(out_diff.norm().item())
            / max(
                float(P_runtime[0].to(torch.float32).norm().item()),
                float(logical_ref[0].to(torch.float32).norm().item()),
                1e-12,
            )
            if out_diff_finite
            else float("inf")
        )
        runtime_finite, runtime_max_err, runtime_fro = self._p_orthogonality_metrics(P_runtime[0])
        logical_finite, logical_max_err, logical_fro = self._p_orthogonality_metrics(logical_ref[0])
        logger.warning(
            "[DION_DEBUG_LOCAL_LOGICAL_COMPARE] step=%d rank=%d tag=%s meta=%s "
            "runtime_finite=%s runtime_max_err=%.6e runtime_fro=%.6e "
            "logical_finite=%s logical_max_err=%.6e logical_fro=%.6e "
            "out_diff_finite=%s out_diff_rel=%.6e out_diff_max_abs=%.6e",
            self._step_count,
            self._global_rank,
            tag,
            self._format_meta_id(meta) if meta is not None else None,
            runtime_finite,
            runtime_max_err,
            runtime_fro,
            logical_finite,
            logical_max_err,
            logical_fro,
            out_diff_finite,
            out_diff_rel,
            out_diff_max_abs,
        )
        self._maybe_dump_bad_ortho_input(
            P_in=P_in[0],
            P_exact=P_runtime[0],
            P_local_ref=logical_ref[0],
            meta=meta,
            tag=tag,
            extra={"mode": "local_logical_compare"},
        )

    def _maybe_log_sketch_comparison(
        self,
        *,
        P_in: torch.Tensor,
        P_out: torch.Tensor,
        meta,
        tag: str,
    ) -> None:
        """Compare seeded-sketch ortho against reference/default sketch on the same input."""
        if not self._debug_compare_sketch:
            return
        if not self._should_run_debug_probe(self._debug_compare_sketch):
            return
        if P_in.ndim != 2:
            return
        if P_in.size(-1) < self._debug_large_r_min_r:
            return

        current_finite, current_max_err, current_fro = self._p_orthogonality_metrics(P_out)
        try:
            reference_out = self._orthogonalize(
                P_in,
                rcqr_oversample=self.defaults['rcqr_oversample'],
                sketch_fn=None,
            )
            reference_finite, reference_max_err, reference_fro = self._p_orthogonality_metrics(reference_out)
        except Exception as exc:
            logger.warning(
                "[DION_DEBUG_SKETCH_COMPARE] step=%d rank=%d tag=%s meta=%s "
                "current_finite=%s current_max_err=%.6e current_fro=%.6e reference_exc=%r",
                self._step_count,
                self._global_rank,
                tag,
                self._format_meta_id(meta) if meta is not None else None,
                current_finite,
                current_max_err,
                current_fro,
                exc,
            )
            return

        logger.info(
            "[DION_DEBUG_SKETCH_COMPARE] step=%d rank=%d tag=%s meta=%s "
            "current_finite=%s current_max_err=%.6e current_fro=%.6e "
            "reference_finite=%s reference_max_err=%.6e reference_fro=%.6e",
            self._step_count,
            self._global_rank,
            tag,
            self._format_meta_id(meta) if meta is not None else None,
            current_finite,
            current_max_err,
            current_fro,
            reference_finite,
            reference_max_err,
            reference_fro,
        )

    def _log_q_norm_snapshot(
        self,
        *,
        tag: str,
        col_sum_sq: torch.Tensor,
        metas: Optional[List],
        real_batch_size: int,
        extra: Optional[dict] = None,
    ) -> None:
        if not self._ortho_sanity_trace or not self._should_run_ortho_sanity():
            return
        if real_batch_size <= 0:
            raise RuntimeError(f"[DION_ORTHO_TRACE_INVALID_BATCH] real_batch_size={real_batch_size}")

        col_sum_sq_real = col_sum_sq[:real_batch_size].to(torch.float32)
        if col_sum_sq_real.ndim == 3 and col_sum_sq_real.size(1) == 1:
            col_sum_sq_real = col_sum_sq_real.squeeze(1)
        if col_sum_sq_real.ndim != 2:
            raise RuntimeError(
                "[DION_ORTHO_TRACE_INVALID_Q_SHAPE] "
                f"tag={tag} step={self._step_count} shape={tuple(col_sum_sq_real.shape)}"
            )
        if not torch.isfinite(col_sum_sq_real).all():
            raise RuntimeError(
                "[DION_ORTHO_TRACE_NONFINITE_Q] "
                f"tag={tag} step={self._step_count} real_batch_size={real_batch_size}"
            )

        col_norms = col_sum_sq_real.sqrt()
        per_slot_err = (col_norms - 1.0).abs().max(dim=-1).values
        per_slot_fro = torch.linalg.vector_norm(col_norms - 1.0, ord=2, dim=-1)
        max_err = float(per_slot_err.max().item())
        max_fro_norm = float(per_slot_fro.max().item())
        logger.info(
            "[%s] step=%d rank=%d max_err=%.6e max_fro_norm=%.6e targets=%s extra=%s",
            tag,
            self._step_count,
            self._global_rank,
            max_err,
            max_fro_norm,
            self._format_sanity_targets(metas, per_slot_err, per_slot_fro=per_slot_fro),
            extra or {},
        )

    def _check_p_orthogonality(
        self,
        P_batch: torch.Tensor,
        configs: List[DionParamConfig],
        metas: List,
        real_batch_size: int,
    ) -> None:
        if not self._should_run_ortho_sanity():
            return
        if real_batch_size <= 0:
            raise RuntimeError(f"[DION_ORTHO_SANITY_INVALID_BATCH] real_batch_size={real_batch_size}")

        cfg0 = configs[0]
        meta0 = metas[0] if metas else None
        fs_only_batch = all(self._is_fs_only_config(cfg) for cfg in configs[:real_batch_size])
        ortho_group = None if fs_only_batch else self._ortho_group_for_config(cfg0, meta0)
        base_group_id = id(ortho_group) if ortho_group is not None else 0
        for idx in range(real_batch_size):
            group = (
                None
                if fs_only_batch
                else self._ortho_group_for_config(configs[idx], metas[idx] if metas else None)
            )
            group_id = id(group) if group is not None else 0
            if group_id != base_group_id:
                raise RuntimeError(
                    "[DION_ORTHO_SANITY_GROUP_MISMATCH] "
                    f"step={self._step_count} slot={idx} expected_group_id={base_group_id} "
                    f"got_group_id={group_id}"
                )

        P_real = P_batch[:real_batch_size].to(torch.float32)
        with _dion_ortho_precision_context():
            gram = torch.bmm(P_real.transpose(1, 2), P_real)
        if ortho_group is not None and dist.get_world_size(ortho_group) > 1:
            dist.all_reduce(gram, op=dist.ReduceOp.SUM, group=ortho_group)

        if not torch.isfinite(gram).all():
            raise RuntimeError(
                "[DION_ORTHO_SANITY_NONFINITE_P] "
                f"step={self._step_count} real_batch_size={real_batch_size}"
            )

        r = gram.size(-1)
        eye = torch.eye(r, device=gram.device, dtype=gram.dtype).unsqueeze(0)
        diff = gram - eye
        diag_abs = diff.diagonal(dim1=-2, dim2=-1).abs()
        offdiag = diff.masked_fill(torch.eye(r, device=gram.device, dtype=torch.bool).unsqueeze(0), 0.0)
        offdiag_abs = offdiag.abs()
        per_slot_err = torch.maximum(diag_abs.max(dim=-1).values, offdiag_abs.amax(dim=(-2, -1)))
        per_slot_fro = torch.linalg.matrix_norm(diff, ord="fro", dim=(-2, -1))
        max_err = float(per_slot_err.max().item())
        max_fro_norm = float(per_slot_fro.max().item())
        self._handle_ortho_sanity_result(
            tag="DION_ORTHO_SANITY_P",
            tol=self._ortho_sanity_p_tol,
            max_err=max_err,
            max_fro_norm=max_fro_norm,
            metas=metas,
            per_slot_err=per_slot_err,
            per_slot_fro=per_slot_fro,
        )

    def _check_q_column_norms(
        self,
        Q_batch: torch.Tensor,
        configs: List[DionParamConfig],
        metas: List,
        real_batch_size: int,
    ) -> None:
        if not self._should_run_ortho_sanity():
            return
        if real_batch_size <= 0:
            raise RuntimeError(f"[DION_ORTHO_SANITY_INVALID_BATCH] real_batch_size={real_batch_size}")

        Q_real = Q_batch[:real_batch_size].to(torch.float32)
        col_sum_sq = (Q_real * Q_real).sum(dim=1)

        fs_world_ok = (
            self.use_fs_collectives
            and self.fs_group is not None
            and dist.get_world_size(self.fs_group) > 1
        )
        need_reduce = [idx for idx, cfg in enumerate(configs[:real_batch_size]) if cfg.has_fs_axis and fs_world_ok]
        if need_reduce:
            reduced = col_sum_sq[need_reduce].contiguous()
            dist.all_reduce(reduced, op=dist.ReduceOp.SUM, group=self.fs_group)
            col_sum_sq[need_reduce] = reduced

        if not torch.isfinite(col_sum_sq).all():
            raise RuntimeError(
                "[DION_ORTHO_SANITY_NONFINITE_Q] "
                f"step={self._step_count} real_batch_size={real_batch_size}"
            )

        col_norms = col_sum_sq.sqrt()
        per_slot_err = (col_norms - 1.0).abs().max(dim=-1).values
        per_slot_fro = torch.linalg.vector_norm(col_norms - 1.0, ord=2, dim=-1)
        max_err = float(per_slot_err.max().item())
        max_fro_norm = float(per_slot_fro.max().item())
        self._handle_ortho_sanity_result(
            tag="DION_ORTHO_SANITY_Q",
            tol=self._ortho_sanity_q_tol,
            max_err=max_err,
            max_fro_norm=max_fro_norm,
            metas=metas,
            per_slot_err=per_slot_err,
            per_slot_fro=per_slot_fro,
        )

    def _scaled_lr_for_2d_param(
        self,
        *,
        lr: float,
        m_for_lr: int,
        n_for_lr: int,
    ) -> float:
        rule = self.defaults.get("lr_scaling_rule", "moonlight")
        if rule == "moonlight":
            rank_fraction = self.defaults.get("rank_fraction", 0.25)
            base_scale = 0.2 / (rank_fraction ** 0.5)
            return base_scale * (max(m_for_lr, n_for_lr) ** 0.5) * lr
        if rule == "dion":
            if m_for_lr <= 0 or n_for_lr <= 0:
                raise RuntimeError(
                    "[DION_INVALID_LR_SCALING_SHAPE] "
                    f"m_for_lr={m_for_lr} n_for_lr={n_for_lr}"
                )
            return lr * math.sqrt(float(n_for_lr) / float(m_for_lr))
        raise RuntimeError(f"[DION_INVALID_LR_SCALING_RULE] rule={rule!r}")

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
        # Create async tasks for optimization
        if self.is_distributed_mode:
            dion_tasks = self._iter_dist_tasks()
        else:
            dion_tasks = self._iter_local_tasks()

        # Execute all tasks with the explicit runtime width from config.
        task_count = 0
        if dion_tasks:
            dion_tasks_list = list(dion_tasks)
            task_count = len(dion_tasks_list)
            if task_count > 0:
                max_tasks = self._resolve_async_task_limit(task_count)
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
        self._buffer_cache.clear()

        torch.cuda.empty_cache()

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

    def _iter_dist_tasks(self) -> Generator[AsyncTask, None, None]:
        """Create async tasks for Distributed mode optimization."""
        # Scalar/non-Dion params continue to use standard DO bucket ownership.
        #
        # Dion params must use a canonical local-shard order in distributed mode.
        # With multiple distributed-optimizer instances, the same local shard can
        # exist in several optimizer-state replicas. Those replicas must consume
        # identical Dion batch/RNG order. Optimizer param-group order is not a
        # valid cross-rank contract here; use standard DO local-shard identity
        # (`param_uid`) instead.
        scalar_params = []
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
                scalar_params.append((p, grad, state, group))

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

        if dion_params:
            yield AsyncTask(self._run_dion_batch_async(dion_params))

        if scalar_params:
            yield AsyncTask(self._run_scalar_bucket_async(scalar_params))

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
            batch_keys = [
                build_batch_key(tuple(p.shape), cfg, p.dtype)
                for p, cfg in zip(params_with_grad, configs)
            ]

            # Create batches
            batches = self.batch_processor.create_batches(
                [(p, group, cfg) for p, cfg in zip(params_with_grad, configs)],
                batch_keys=batch_keys,
                batch_size=self.local_batch_size,
            )

            # Process each batch
            for batch in batches:
                batch_params = [item[0] for item in batch]
                batch_configs = [item[2] for item in batch]

                yield AsyncTask(self._run_local_param_batch_async(batch_params, group, batch_configs))

    def _use_dion_update(
        self,
        p: Tensor,
        state: Dict,
        group: Dict,
        meta,
    ) -> bool:
        """Return whether a param should follow the Dion update path."""
        algorithm = group.get('algorithm', 'dion')
        is_dion_marked = (
            bool(meta.is_dion_param)
            if meta is not None
            else is_dion_param(p, getattr(p, "_param_name", None))
        )

        is_2d_global = False
        if meta and meta.global_shape and len(meta.global_shape) == 2:
            is_2d_global = True
        elif is_dion_marked and p.ndim == 2:
            is_2d_global = True

        use_dion = (
            algorithm == 'dion'
            and is_dion_marked
            and is_2d_global
            and 'Q' in state
        )

        return use_dion

    def _run_scalar_bucket_async(
        self,
        scalar_params: List[Tuple[Tensor, Tensor, Dict, Dict]],
    ) -> Generator[None, None, None]:
        """Process only non-Dion params while preserving standard DO bucket ownership."""
        scalar_opt = self.defaults.get('scalar_optimizer', 'adamw')
        for p, grad, state, group in scalar_params:
            lr = group.get('lr', self.defaults['lr'])
            if p.ndim == 1:
                weight_decay = 0.0
            else:
                wd_mult = group.get('wd_mult', 1.0)
                weight_decay = group.get('weight_decay', self.defaults['weight_decay'] * wd_mult)

            if scalar_opt == 'lion':
                self._lion_update(p, grad, state, group, lr, weight_decay)
            else:
                self._adamw_update(p, grad, state, group, lr, weight_decay)
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
            use_dion = self._use_dion_update(p, state, group, None)

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
                weight_decay = group.get('weight_decay', self.defaults['weight_decay'] * wd_mult)

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

        batch_items = []
        batch_keys = []

        for p, grad, state, group, config, meta in dion_params:
            # Use stored local_shape from state
            local_shape = state.get('local_shape', None)
            if local_shape is None:
                local_shape = self._require_local_2d_shape(p, meta)
                state['local_shape'] = local_shape

            batch_items.append((p, grad, state, group, config, meta))
            batch_keys.append(build_batch_key(local_shape, config, grad.dtype))

        batch_groups = {}
        for batch_key, grouped_items in self.batch_processor.group_items(batch_items, batch_keys):
            batch_groups[batch_key] = {
                'params': [item[0] for item in grouped_items],
                'grads': [item[1] for item in grouped_items],
                'states': [item[2] for item in grouped_items],
                'groups': [item[3] for item in grouped_items],
                'configs': [item[4] for item in grouped_items],
                'metas': [item[5] for item in grouped_items],
            }

        # Process each shape group separately while preserving first-seen order.
        global_param_offset = 0
        ortho_completed_count = 0

        # Sync batch_keys across every group that must keep the same batch schedule.
        local_batch_keys = list(batch_groups.keys())
        grouped_batch_keys = {}
        for batch_key in local_batch_keys:
            sync_groups = self._batch_sync_groups(batch_key)
            if not sync_groups:
                grouped_batch_keys.setdefault(None, (None, []))[1].append(batch_key)
                continue
            for sync_group in sync_groups:
                grouped_batch_keys.setdefault(id(sync_group), (sync_group, []))[1].append(batch_key)

        all_batch_keys = []
        for sync_group, group_keys in grouped_batch_keys.values():
            if sync_group is not None:
                all_batch_keys.extend(self._sync_batch_keys(group_keys, sync_group))
            else:
                all_batch_keys.extend(group_keys)
        all_batch_keys = list(dict.fromkeys(all_batch_keys))

        for batch_key in all_batch_keys:
            local_shape = batch_key[0]
            m, n = local_shape

            cfg = self._batch_key_to_config(batch_key)
            has_tp_axis = self._has_active_tp_axis(cfg)
            has_fs_axis = self._has_active_fs_axis(cfg)
            sync_groups = self._batch_sync_groups(batch_key)
            replicate_group = self._reference_replicate_group_for_config(cfg)

            if has_tp_axis and self.tp_group:
                batch_size = dist.get_world_size(self.tp_group)
            elif has_fs_axis and self.fs_group:
                batch_size = dist.get_world_size(self.fs_group)
            elif cfg.compressed_all_reduce and replicate_group is not None:
                batch_size = dist.get_world_size(replicate_group)
            else:
                batch_size = dist.get_world_size(replicate_group) if replicate_group else 1

            if os.getenv("DION_DEBUG_FORCE_BATCH1", "0") == "1":
                batch_size = 1

            if batch_key not in batch_groups:
                raise RuntimeError(
                    "[DION_MISSING_LOCAL_SHARD] "
                    f"batch_key={batch_key} rank={rank} sync_groups={sync_groups}"
                )

            group_data = batch_groups[batch_key]
            for sync_group in sync_groups:
                self._align_group_data_order(group_data, sync_group)
            if self._should_run_debug_probe(self._debug_batch_order):
                logger.info(
                    "[DION_DEBUG_BATCH_GROUP] step=%d rank=%d batch_key=%s sync_groups=%s local_ids=%s",
                    self._step_count,
                    rank,
                    batch_key,
                    [
                        dist.get_process_group_ranks(sync_group)
                        for sync_group in sync_groups
                    ],
                    [
                        self._format_meta_id(meta)
                        for meta in group_data['metas']
                    ],
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
                        f"batch_key={batch_key} group_ranks={group_ranks} "
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
                    self._log_debug_batch_order(
                        tag="DION_DEBUG_BATCH_IDS_WINDOW",
                        sync_group=sync_group,
                        batch_key=batch_key,
                        local_ids=local_batch_ids,
                        canonical_ids=canonical_batch_ids,
                    )
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
                            f"batch_key={batch_key} batch_start={i} batch_end={batch_end} "
                            f"group_ranks={group_ranks} mismatch_local_ranks={mismatch_ranks} "
                            f"local_batch_ids={local_batch_ids} gathered_batch_ids={gathered_batch_ids}"
                        )

                # Track real_batch_size before padding
                real_batch_size = len(params)
                if (
                    os.getenv("DION_DEBUG_PARTIAL_BATCHES", "0") == "1"
                    and real_batch_size < batch_size
                ):
                    partial_warn_count = getattr(self, "_partial_batch_warn_count", 0)
                    if partial_warn_count < 12:
                        setattr(self, "_partial_batch_warn_count", partial_warn_count + 1)
                        logger.warning(
                            "[DION_PARTIAL_BATCH] rank=%s step=%s batch_key=%s real_batch_size=%s batch_size=%s names=%s",
                            rank,
                            self._step_count,
                            batch_key,
                            real_batch_size,
                            batch_size,
                            [
                                getattr(meta, "param_name", "") if meta is not None else ""
                                for meta in metas[:real_batch_size]
                            ][:8],
                        )

                # Partial distributed batches still need fixed batch width, but dummy slots
                # must remain inert because orthogonalization runs before bad-batch repair.
                params = pad_batch(params, batch_size)
                grads_to_process = pad_batch(grads_to_process, batch_size)
                momentums = pad_batch(momentums, batch_size)
                Qs = pad_batch(Qs, batch_size)

                # Pad metadata lists to match.
                #
                # Tensor padding uses inert zero tensors. The matching metadata must
                # stay anonymous as well; reusing a real param's metadata makes dummy
                # slots participate in seeded sketch / logical-ID paths as if they were
                # copies of that real parameter.
                while len(param_shapes) < batch_size:
                    param_shapes.append(param_shapes[0])
                while len(configs) < batch_size:
                    configs.append(configs[0])
                while len(metas) < batch_size:
                    metas.append(None)
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

        batch_groups.clear()

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
                    f"Ensure rp_group is provided uniformly to all ranks."
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

        for param, meta in self.dist_metas.items():
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
                            f"This ensures identical distributed Dion execution."
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

    def _require_local_2d_shape(self, p: Tensor, meta: MegatronDionDistMeta) -> Tuple[int, int]:
        """Return the exact local 2D shard shape from distributed metadata."""
        if meta is None or meta.shape is None or len(meta.shape) != 2:
            raise RuntimeError(
                "[Dion] distributed param is missing exact local 2D shape metadata "
                f"param_uid={getattr(meta, 'param_uid', None)} "
                f"param_name={getattr(meta, 'param_name', '')} "
                f"meta_shape={getattr(meta, 'shape', None)}"
            )
        local_shape = tuple(int(dim) for dim in meta.shape)
        if local_shape[0] <= 0 or local_shape[1] <= 0:
            raise RuntimeError(
                "[Dion] invalid empty local 2D shape metadata "
                f"param_uid={getattr(meta, 'param_uid', None)} "
                f"param_name={getattr(meta, 'param_name', '')} "
                f"local_shape={local_shape}"
            )
        if int(p.numel()) != local_shape[0] * local_shape[1]:
            raise RuntimeError(
                "[Dion] local 2D shape metadata does not match shard numel "
                f"param_uid={getattr(meta, 'param_uid', None)} "
                f"param_name={getattr(meta, 'param_name', '')} "
                f"local_shape={local_shape} numel={int(p.numel())}"
            )
        return local_shape

    @staticmethod
    def _batch_key_to_config(batch_key: Tuple) -> DionParamConfig:
        """Build a DionParamConfig from a synchronized Dion batch key."""
        return DionParamConfig(
            has_fs_axis=bool(batch_key[1]),
            active_fs_axis=bool(batch_key[2]),
            has_tp_axis=bool(batch_key[3]),
            is_transposed=bool(batch_key[4]),
            compressed_all_reduce=bool(batch_key[5]),
            inner_shard_tensor_dim=batch_key[6] if batch_key[6] != -1 else None,
            outer_shard_tensor_dim=batch_key[7] if batch_key[7] != -1 else None,
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
        config.active_fs_axis = False
        config.has_tp_axis = False

        meta = self.dist_metas.get(param, None) if hasattr(self, 'dist_metas') else None

        if self.is_distributed_mode and meta and meta.global_shape and len(meta.global_shape) == 2 and meta.is_dion_param:
            gm, gn = meta.global_shape
            if param.ndim == 2:
                lm, ln = param.shape

                # Preserve the logical 2D layout from distributed-optimizer metadata
                # even when the active shard world size for a given axis is 1.
                #
                # Dion math and orientation (`is_transposed`) must be topology-invariant:
                # the same logical parameter should follow the same inner/outer-shard
                # route regardless of whether that logical shard axis degenerates to a
                # single rank in the current valid topology.
                #
                # Actual communication still gates on the real process-group world size
                # at the call sites (`_needs_tp_q_unshard`, `_ortho_group_for_config`,
                # `_batch_sync_groups`, Q-init rank selection, etc.). This block is
                # responsible only for recovering the logical layout contract from the
                # standard Megatron distributed-optimizer metadata.
                #
                # Without this separation, square matrices such as attention output
                # projections can flip between transposed/non-transposed Dion routes
                # across FS/RP factorizations even though their logical shard layout is
                # unchanged, which violates topology invariance.
                # Detect TP sharding (inner axis)
                tp_split_dim = getattr(meta, 'tp_split_dim', -1) if meta else -1
                if tp_split_dim in (0, 1):
                    config.has_tp_axis = True
                    config.inner_shard_tensor_dim = tp_split_dim

                # Detect FS sharding (outer axis)
                fs_split_dim = getattr(meta, 'fs_split_dim', -1) if meta else -1
                if fs_split_dim in (0, 1):
                    config.has_fs_axis = True
                    config.active_fs_axis = getattr(meta, "shard_group_world_size", 1) > 1
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
            local_shape = self._require_local_2d_shape(param, meta) if is_2d else None
            if local_shape:
                m, n = local_shape
            else:
                m, n = None, None
        else:
            is_2d = is_dion_param(param, getattr(param, "_param_name", None))
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
                    param_key = str(meta.param_uid) if meta.param_uid is not None else ""
                q_seed = self._next_q_init_seed(
                    config=config,
                    meta=meta,
                    q_global_shape=(q_base_global, r_global),
                )
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

    def _sync_batch_keys(self, local_batch_keys: List[Tuple], group) -> List[Tuple]:
        """Align batch-key schedules to rank0 order when contents match exactly."""
        if group is None:
            return local_batch_keys

        world_size = dist.get_world_size(group)
        if world_size <= 1:
            return local_batch_keys

        cache_key = (id(group), tuple(local_batch_keys))
        if not hasattr(self, '_batch_key_cache'):
            self._batch_key_cache = {}
        if cache_key in self._batch_key_cache:
            return self._batch_key_cache[cache_key]

        local_keys_serialized = []
        for sk in local_batch_keys:
            local_shape = tuple(sk[0]) if isinstance(sk[0], (tuple, list)) else sk[0]
            key_data = (
                local_shape,
                bool(sk[1]),
                bool(sk[2]),
                bool(sk[3]),
                bool(sk[4]),
                bool(sk[5]),
                int(sk[6]) if sk[6] is not None else -1,
                int(sk[7]) if sk[7] is not None else -1,
                str(sk[8]),
            )
            local_keys_serialized.append(key_data)

        gathered_keys_list = [None] * world_size
        dist.all_gather_object(gathered_keys_list, local_keys_serialized, group=group)

        canonical_keys = tuple(gathered_keys_list[0])
        self._log_debug_batch_order(
            tag="DION_DEBUG_BATCH_KEYS",
            sync_group=group,
            local_keys=local_keys_serialized,
            canonical_keys=canonical_keys,
        )
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
                "[DION_BATCH_KEY_MISMATCH] "
                f"group_ranks={group_ranks} mismatch_local_ranks={mismatch_ranks} "
                f"local_batch_keys={local_keys_serialized} gathered_batch_keys={gathered_keys_list}"
            )

        ordered_keys = [
            (
                key_data[0],
                key_data[1],
                key_data[2],
                key_data[3],
                key_data[4],
                key_data[5],
                key_data[6],
                key_data[7],
                str_to_dtype(key_data[8]),
            )
            for key_data in canonical_keys
        ]
        self._batch_key_cache[cache_key] = ordered_keys
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
            self._log_debug_batch_order(
                tag="DION_DEBUG_BATCH_IDS_STABLE",
                sync_group=sync_group,
                local_ids=local_ids,
                canonical_ids=canonical_ids,
            )
            return

        index_by_id = {param_id: idx for idx, param_id in enumerate(local_ids)}
        reorder_indices = [index_by_id[param_id] for param_id in canonical_ids]
        self._log_debug_batch_order(
            tag="DION_DEBUG_BATCH_IDS_REORDER",
            sync_group=sync_group,
            local_ids=local_ids,
            canonical_ids=canonical_ids,
            reorder_indices=reorder_indices,
        )
        for key in ('params', 'grads', 'states', 'groups', 'configs', 'metas'):
            group_data[key] = [group_data[key][idx] for idx in reorder_indices]

    def _sync_tp_batch_keys(self, local_batch_keys: List[Tuple]) -> List[Tuple]:
        """TP wrapper around `_sync_batch_keys()`."""
        return self._sync_batch_keys(local_batch_keys, self.tp_group)

    def _batch_sync_groups(self, batch_key: Tuple):
        """Return all process groups that must see the same batch schedule for this batch key."""
        cfg = self._batch_key_to_config(batch_key)
        groups = []
        compressed_replicate_group = self._reference_replicate_group_for_config(cfg)
        if (
            cfg.compressed_all_reduce
            and compressed_replicate_group is not None
            and dist.get_world_size(compressed_replicate_group) > 1
        ):
            groups.append(compressed_replicate_group)
        if (
            self.state_replica_group is not None
            and self.state_replica_world_size > 1
        ):
            if all(id(self.state_replica_group) != id(existing) for existing in groups):
                groups.append(self.state_replica_group)
        if (
            self._has_active_tp_axis(cfg)
            and self.tp_group is not None
            and dist.get_world_size(self.tp_group) > 1
        ):
            if all(id(self.tp_group) != id(existing) for existing in groups):
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
        return config.active_fs_axis and (
            (not config.is_transposed and config.outer_shard_tensor_dim == 1)
            or (config.is_transposed and config.outer_shard_tensor_dim == 0)
        )

    def _has_active_tp_axis(self, config: DionParamConfig) -> bool:
        """Return whether TP sharding is logically present and physically active."""
        return config.has_tp_axis and self.tp_group is not None and self.tp_world_size > 1

    def _has_active_fs_axis(self, config: DionParamConfig, meta=None) -> bool:
        """Return whether FS sharding is logically present and physically active."""
        if not config.has_fs_axis:
            return False
        if getattr(config, "active_fs_axis", False):
            return True
        shard_group = self._fs_group_for_meta(meta)
        if shard_group is None:
            return False
        return dist.get_world_size(shard_group) > 1

    def _needs_tp_q_unshard(self, config: DionParamConfig) -> bool:
        """Return whether STEP 2 must all-gather Q across TP ranks."""
        return self._has_active_tp_axis(config)

    def _needs_tp_r_reduce(self, config: DionParamConfig) -> bool:
        """Return whether STEP 5 must all-reduce R across TP ranks."""
        return self._p_is_tp_sharded(config)

    def _fs_group_for_meta(self, meta=None):
        """Return the FS shard group for a param, including expert-local groups."""
        shard_group = getattr(meta, 'shard_group', None) if meta is not None else None
        if shard_group is None:
            shard_group = self.fs_group
        return shard_group

    def _p_is_tp_sharded(self, config: DionParamConfig) -> bool:
        """Return whether P is sharded along TP for this config."""
        return self._has_active_tp_axis(config) and (
            (not config.is_transposed and config.inner_shard_tensor_dim == 0)
            or (config.is_transposed and config.inner_shard_tensor_dim == 1)
        )

    @staticmethod
    def _p_is_fs_sharded(config: DionParamConfig) -> bool:
        """Return whether P is sharded along FS for this config."""
        return config.active_fs_axis and (
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
        real_batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Distributed orthogonalization matching `dion_reference.py::distributed_orthogonalize()`."""
        batch_size = P_batch.size(0)
        m_shard_local = P_batch.size(1)
        r = P_batch.size(2)
        original_dtype = P_batch.dtype
        self._ortho_batch_trace_counter += 1

        if shard_group is not None:
            shard_world_size = dist.get_world_size(shard_group)
        else:
            shard_world_size = 1

        if shard_group is None or shard_world_size <= 1:
            result = torch.empty_like(P_batch)
            for i in range(batch_size):
                result[i] = self._orthogonalize(P_batch[i], rcqr_oversample=oversample)
            return result

        if self._debug_ortho_batch_trace:
            logger.info(
                "[DION_DEBUG_ORTHO_BATCH_TRACE] step=%d rank=%d counter=%d shard_world=%d batch_size=%d m_local=%d r=%d metas=%s",
                self._step_count,
                self._global_rank,
                self._ortho_batch_trace_counter,
                shard_world_size,
                batch_size,
                m_shard_local,
                r,
                [self._format_meta_id(meta) for meta in metas] if metas is not None else [],
            )
        if os.getenv("DION_DEBUG_ORTHO_GROUP_TRACE", "0") == "1":
            logger.info(
                "[DION_DEBUG_ORTHO_GROUP_TRACE] step=%d rank=%d counter=%d shard_world=%d group_ranks=%s mesh=%s metas=%s",
                self._step_count,
                self._global_rank,
                self._ortho_batch_trace_counter,
                shard_world_size,
                list(dist.get_process_group_ranks(shard_group)),
                self._device_mesh_for_group(shard_group, "ortho").mesh.detach().cpu().tolist(),
                [self._format_meta_id(meta) for meta in metas] if metas is not None else [],
            )

        ortho_mesh = self._device_mesh_for_group(shard_group, "ortho")
        p_dtensor = DTensor.from_local(
            P_batch,
            device_mesh=ortho_mesh,
            placements=(Shard(P_batch.ndim - 2),),
        )
        batch_meta_ids = (
            [self._format_meta_id(meta) for meta in metas[:batch_size]]
            if metas is not None
            else None
        )
        p_out = distributed_orthogonalize_dtensor_exact(
            p_dtensor,
            oversample=oversample,
            shard_mesh_dim=0,
            logical_seed_keys=self._logical_sketch_seed_keys(
                metas=metas[:batch_size] if metas is not None else None,
                tag="logical_local",
            ),
            batch_meta_ids=batch_meta_ids,
        ).to_local().contiguous()
        return p_out.to(original_dtype)

    # Batch update methods

    def _unshard_q_batch(
        self,
        Qs: List[Tensor],
        configs: List[DionParamConfig],
        metas: Optional[List] = None,
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
                local_batch = self._cached_buffer(
                    f"q_local_batch_{group_seq}",
                    (len(indices), n, r_local),
                    dtype,
                    device,
                )
                for slot, idx in enumerate(indices):
                    local_batch[slot].copy_(Qs[idx])

                gathered_batch = self._cached_buffer(
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
                for slot, idx in enumerate(indices):
                    Q_for_matmul[idx] = q_full_batch[slot]
                del local_batch

        q_result = [q for q in Q_for_matmul]
        self._maybe_dump_unsharded_q_batch(metas=metas, q_for_matmul=q_result)
        return q_result

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
                if len(indices) == batch_size and len(group_to_indices) == 1:
                    dist.all_reduce(P_batch, op=dist.ReduceOp.SUM, group=group)
                else:
                    tensors = [P_batch[idx] for idx in indices]
                    reduced = funcol.all_reduce_coalesced(tensors, reduceOp="sum", group=group)
                    for idx, tensor in zip(indices, reduced):
                        P_batch[idx].copy_(tensor)
                    del tensors, reduced
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
                padded_chunk = self._cached_buffer(
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
        sketch_tag: Optional[str] = None,
        sketch_fn=None,
        use_seeded_sketch: bool = True,
    ) -> torch.Tensor:
        """Local orthogonalization path for a batched slice."""
        if P_slice.size(0) == 0:
            return P_slice

        # Keep the local-batched path invariant to batch composition.
        #
        # PP/EP/local batching can change which logical params are grouped together on
        # one rank. If we let `orthogonalize()` consume one batch-global RNG stream,
        # the sketch for one logical parameter depends on which other parameters were
        # present in the same local batch. That changes Dion updates even when the
        # underlying logical parameter and its P input are identical.
        #
        # The intended Megatron backend contract is that local batching must not alter
        # the logical Dion update. Use per-param seeded sketches by default for this
        # local-batched path unless a caller explicitly overrides the sketch function
        # for focused validation.
        if use_seeded_sketch and sketch_fn is None and metas is not None:
            sketch_fn = self._seeded_sketch_fn(
                metas=metas,
                tag=sketch_tag if sketch_tag is not None else tag,
            )
        if metas is not None and self._debug_trace_params:
            target_names = {
                getattr(meta, "param_name", "")
                for meta in metas
                if meta is not None
            }
            if "*" in self._debug_trace_params or target_names.intersection(self._debug_trace_params):
                logger.warning(
                    "[DION_DEBUG_LOCAL_SKETCH_ROUTE] step=%d rank=%d tag=%s sketch_tag=%s "
                    "use_seeded_sketch=%s sketch_fn_is_none=%s metas=%s",
                    self._step_count,
                    self._global_rank,
                    tag,
                    sketch_tag,
                    use_seeded_sketch,
                    sketch_fn is None,
                    [self._format_meta_id(meta) for meta in metas],
                )

        P_ortho_slice = orthogonalize(
            P_slice,
            rcqr_oversample=self.defaults['rcqr_oversample'],
            sketch_fn=sketch_fn,
        ).to(torch.float32)
        if metas is not None:
            for i in range(P_slice.size(0)):
                meta = metas[i] if i < len(metas) else None
                self._maybe_log_sketch_comparison(
                    P_in=P_slice[i],
                    P_out=P_ortho_slice[i],
                    meta=meta,
                    tag=tag,
                )
        return P_ortho_slice.to(P_slice.dtype)

    def _orthogonalize_reference_local_slice(
        self,
        P_slice: torch.Tensor,
        metas: Optional[List] = None,
        *,
        tag: str = "reference_local",
    ) -> torch.Tensor:
        """Regular-Tensor local orthogonalization for one logical full matrix.

        `dion_reference.py::orthogonalize()` is the correct mathematical kernel
        for DDP / unsharded and fs-only single-matrix orthogonalization routes.
        The sketch itself, however, must be a function of the logical parameter
        identity / step rather than ambient RNG state so the same logical Dion
        update is preserved when the topology is re-embedded across different
        replica counts or node counts.
        """
        return self._orthogonalize_local_slice(
            P_slice,
            metas=metas,
            tag=tag,
            sketch_tag="logical_local",
        )

    def _is_fs_only_config(self, config: DionParamConfig) -> bool:
        """Return whether the config follows dion_reference.py fs-only semantics."""
        return self._has_active_fs_axis(config) and not self._has_active_tp_axis(config)

    def _orthogonalize_fs_only_batch(
        self,
        P_batch: torch.Tensor,
        metas: List,
        *,
        replicate_group: Optional[torch.distributed.ProcessGroup] = None,
        replicate_ranks: Optional[List[int]] = None,
    ) -> Generator[torch.Tensor, None, None]:
        """Match `dion_reference.dion_update_fsdp()` for fs-only batches.

        Reference contract:
        - batch size equals outer shard mesh size
        - collapse partial P by reduce-scattering along batch dim
        - if compressed RP is enabled, average P_single over the replicate mesh
        - each rank orthogonalizes one full matrix in the batch
        - all-gather the orthogonalized batch back before computing R
        """
        meta0 = metas[0] if metas else None
        shard_group = self._fs_group_for_meta(meta0)
        if shard_group is None or dist.get_world_size(shard_group) <= 1:
            yield
            return self._orthogonalize_local_slice(
                P_batch,
                metas=metas,
                tag="fs_only_local",
                sketch_tag="logical_local",
            )

        shard_world = dist.get_world_size(shard_group)
        batch_size = P_batch.size(0)
        if batch_size != shard_world:
            raise RuntimeError(
                "[DION_FSONLY_BATCH_SIZE_MISMATCH] "
                f"batch_size={batch_size} shard_world_size={shard_world}"
            )
        if self._should_run_debug_probe(self._debug_fs_only):
            logger.info(
                "[DION_DEBUG_FS_ONLY_BATCH] step=%d rank=%d shard_group_ranks=%s batch_size=%d metas=%s",
                self._step_count,
                self._global_rank,
                dist.get_process_group_ranks(shard_group),
                batch_size,
                [self._format_meta_id(meta) for meta in metas],
            )

        shard_rank = dist.get_rank(shard_group)
        local_meta = [metas[shard_rank]] if metas and shard_rank < len(metas) else None
        # Reference FSDP path: Partial() -> Shard(0), i.e. reduce-scatter along the
        # batch dimension so each rank receives the reduced slot at its shard index.
        P_single = funcol.reduce_scatter_tensor(
            P_batch.contiguous(),
            reduceOp="sum",
            scatter_dim=0,
            group=shard_group,
        )
        yield

        self._maybe_log_target_trace(
            tag="fs_only_single_pre_replicate",
            metas=local_meta,
            p_batch=P_single,
            real_batch_size=1,
            extra={
                "shard_rank": int(shard_rank),
                "shard_world": int(shard_world),
            },
        )

        if replicate_group is not None and dist.get_world_size(replicate_group) > 1:
            if replicate_ranks is not None:
                yield from self._collapse_batch_over_rank_subset(
                    P_single,
                    primary_group=replicate_group,
                    subset_ranks=replicate_ranks,
                )
            else:
                dist.all_reduce(
                    P_single,
                    op=self._replicate_reduce_op(),
                    group=replicate_group,
                )
                yield

        self._maybe_log_target_trace(
            tag="fs_only_single_post_replicate",
            metas=local_meta,
            p_batch=P_single,
            real_batch_size=1,
            extra={
                "shard_rank": int(shard_rank),
                "shard_world": int(shard_world),
                "replicate_world": int(dist.get_world_size(replicate_group))
                if replicate_group is not None
                else 1,
            },
        )

        if self._debug_verify_fs_route and self._should_run_debug_probe(self._debug_verify_fs_route):
            expected_batch = P_batch.clone()
            dist.all_reduce(expected_batch, op=dist.ReduceOp.SUM, group=shard_group)
            expected_single = expected_batch[shard_rank : shard_rank + 1]
            route_diff = (expected_single - P_single).to(torch.float32)
            route_finite = bool(torch.isfinite(route_diff).all().item())
            route_max_abs = float(route_diff.abs().max().item()) if route_finite else float("inf")
            route_fro = (
                float(torch.linalg.matrix_norm(route_diff, ord="fro").item())
                if route_finite
                else float("inf")
            )
            logger.info(
                "[DION_DEBUG_VERIFY_FS_ROUTE] step=%d rank=%d shard_rank=%d finite=%s max_abs=%.6e fro_norm=%.6e meta=%s",
                self._step_count,
                self._global_rank,
                shard_rank,
                route_finite,
                route_max_abs,
                route_fro,
                self._format_meta_id(metas[shard_rank]) if metas and shard_rank < len(metas) else None,
            )
        if self._should_run_debug_probe(self._debug_fs_only_batch_trace):
            pure_dp_group = parallel_state.get_data_parallel_group(
                with_context_parallel=False,
                partial_data_parallel=False,
            )
            cp_group = parallel_state.get_context_parallel_group(check_initialized=False)
            pure_dp_ranks = (
                dist.get_process_group_ranks(pure_dp_group) if pure_dp_group is not None else None
            )
            cp_group_ranks = (
                dist.get_process_group_ranks(cp_group) if cp_group is not None else None
            )
            candidate_replicate_ranks = None
            if replicate_group is not None and pure_dp_ranks is not None:
                candidate_replicate_ranks = [
                    rank
                    for rank in dist.get_process_group_ranks(replicate_group)
                    if rank in pure_dp_ranks
                ]
            logger.warning(
                "[DION_DEBUG_FS_ONLY_SINGLE] step=%d rank=%d shard_rank=%d shard_group_ranks=%s "
                "rp_group_ranks=%s pure_dp_ranks=%s cp_group_ranks=%s candidate_replicate_ranks=%s "
                "meta_order=%s local_meta=%s p_single_sum=%.6e p_single_norm=%.6e",
                self._step_count,
                self._global_rank,
                shard_rank,
                dist.get_process_group_ranks(shard_group),
                dist.get_process_group_ranks(replicate_group) if replicate_group is not None else None,
                pure_dp_ranks,
                cp_group_ranks,
                candidate_replicate_ranks,
                [self._format_meta_id(meta) for meta in metas],
                [self._format_meta_id(meta) for meta in local_meta] if local_meta is not None else [],
                float(P_single.detach().to(torch.float32).sum().item()),
                float(P_single.detach().to(torch.float32).norm().item()),
            )
        if self._should_run_debug_probe(self._debug_fs_only):
            logger.info(
                "[DION_DEBUG_FS_ONLY_ROUTE] step=%d rank=%d shard_rank=%d local_meta=%s",
                self._step_count,
                self._global_rank,
                shard_rank,
                [self._format_meta_id(meta) for meta in local_meta] if local_meta is not None else [],
            )
        self._log_p_orthogonality_snapshot(
            tag="DION_ORTHO_TRACE_FS_SINGLE_PRE",
            P_batch=P_single,
            metas=local_meta,
            real_batch_size=1,
            extra={
                "fs_group_rank": shard_rank,
                "fs_group_world": shard_world,
                "tag": "fs_only_pre",
            },
            force=self._debug_fs_only,
        )
        # Match `dion_reference.py::dion_update_fsdp()` literally: after
        # reduce-scatter over the outer-shard mesh (and optional replicate
        # averaging), each rank holds one batch-sharded full matrix as a DTensor
        # and applies `orthogonalize()` on that DTensor. This path must therefore
        # follow the exact DTensor orthogonalize contract rather than the regular
        # local-Tensor orthogonalizer.
        fs_only_mesh = self._device_mesh_for_group(shard_group, "fs_only_ortho")
        p_single_dtensor = DTensor.from_local(
            P_single.contiguous(),
            device_mesh=fs_only_mesh,
            placements=(Shard(0),),
        )
        batch_meta_ids = (
            [self._format_meta_id(meta) for meta in local_meta]
            if local_meta is not None
            else None
        )
        P_single_ortho = orthogonalize_dtensor_exact(
            p_single_dtensor,
            oversample=self.defaults['rcqr_oversample'],
            logical_seed_keys=self._logical_sketch_seed_keys(
                metas=local_meta,
                tag="logical_local",
            ),
            batch_meta_ids=batch_meta_ids,
        ).to_local().contiguous().to(P_single.dtype)
        self._maybe_log_target_trace(
            tag="fs_only_single_post_ortho",
            metas=local_meta,
            p_batch=P_single_ortho,
            real_batch_size=1,
            extra={
                "shard_rank": int(shard_rank),
                "shard_world": int(shard_world),
            },
        )
        self._maybe_compare_fs_only_exact(
            P_in=P_single,
            P_exact=P_single_ortho,
            meta=local_meta[0] if local_meta else None,
            shard_group=shard_group,
            shard_rank=shard_rank,
            tag="fs_only_exact",
        )
        self._log_p_orthogonality_snapshot(
            tag="DION_ORTHO_TRACE_FS_SINGLE_POST",
            P_batch=P_single_ortho,
            metas=local_meta,
            real_batch_size=1,
            extra={
                "fs_group_rank": shard_rank,
                "fs_group_world": shard_world,
                "tag": "fs_only_post",
            },
            force=self._debug_fs_only,
        )

        # Gather the orthogonalized batch back, matching dion_update_fsdp().
        P_ortho = funcol.all_gather_tensor(
            P_single_ortho.contiguous(),
            gather_dim=0,
            group=shard_group,
        )
        yield
        if self._should_run_debug_probe(self._debug_fs_only_batch_trace):
            pure_dp_group = parallel_state.get_data_parallel_group(
                with_context_parallel=False,
                partial_data_parallel=False,
            )
            cp_group = parallel_state.get_context_parallel_group(check_initialized=False)
            pure_dp_ranks = (
                dist.get_process_group_ranks(pure_dp_group) if pure_dp_group is not None else None
            )
            cp_group_ranks = (
                dist.get_process_group_ranks(cp_group) if cp_group is not None else None
            )
            candidate_replicate_ranks = None
            if replicate_group is not None and pure_dp_ranks is not None:
                candidate_replicate_ranks = [
                    rank
                    for rank in dist.get_process_group_ranks(replicate_group)
                    if rank in pure_dp_ranks
                ]
            slot_sums = [float(P_ortho[i].detach().to(torch.float32).sum().item()) for i in range(P_ortho.size(0))]
            slot_norms = [float(P_ortho[i].detach().to(torch.float32).norm().item()) for i in range(P_ortho.size(0))]
            logger.warning(
                "[DION_DEBUG_FS_ONLY_GATHER] step=%d rank=%d shard_rank=%d shard_group_ranks=%s "
                "rp_group_ranks=%s pure_dp_ranks=%s cp_group_ranks=%s candidate_replicate_ranks=%s "
                "meta_order=%s slot_sums=%s slot_norms=%s",
                self._step_count,
                self._global_rank,
                shard_rank,
                dist.get_process_group_ranks(shard_group),
                dist.get_process_group_ranks(replicate_group) if replicate_group is not None else None,
                pure_dp_ranks,
                cp_group_ranks,
                candidate_replicate_ranks,
                [self._format_meta_id(meta) for meta in metas],
                slot_sums,
                slot_norms,
            )
        self._log_p_orthogonality_snapshot(
            tag="DION_ORTHO_TRACE_FS_BATCH_POST_GATHER",
            P_batch=P_ortho,
            metas=metas,
            real_batch_size=batch_size,
            extra={
                "fs_group_rank": shard_rank,
                "fs_group_world": shard_world,
                "tag": "fs_only_batch_post_gather",
            },
            force=self._debug_fs_only,
        )
        return P_ortho

    def _orthogonalize_p_batch(
        self,
        P_batch: torch.Tensor,
        configs: List[DionParamConfig],
        metas: List,
        real_batch_size: Optional[int] = None,
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
                real_batch_size=real_batch_size,
            )
        else:
            P_batch = self._orthogonalize_reference_local_slice(
                P_batch,
                metas=metas,
                tag="dense_replicate_reference_local",
            )
        yield
        return P_batch

    def _orthogonalize_dense_replicate_batch_async(
        self,
        P_batch: torch.Tensor,
        metas: List,
        *,
        comm_group: torch.distributed.ProcessGroup,
    ) -> Generator[torch.Tensor, None, None]:
        """Match `dion_reference.py::dion_update_ddp()` when dense grads are already synced.

        Reference DDP contract for `compressed_all_reduce=False` still batch-shards P
        across the replicate mesh: each rank orthogonalizes only its local batch slot
        (or chunk in the padded generalization) and the orthogonalized batch is then
        gathered back. Local orthogonalization on every rank for every slot is not the
        reference contract.
        """
        comm_world_size = dist.get_world_size(comm_group)
        if comm_world_size <= 1:
            yield
            return self._orthogonalize_reference_local_slice(
                P_batch,
                metas=metas,
                tag="dense_replicate_reference_local",
            )

        batch_size = P_batch.size(0)
        original_batch_size = batch_size
        if batch_size % comm_world_size != 0:
            pad = comm_world_size - (batch_size % comm_world_size)
            padded_batch_size = batch_size + pad
            P_padded = self._cached_buffer(
                "dense_replicate_p_padded",
                (padded_batch_size, P_batch.size(1), P_batch.size(2)),
                P_batch.dtype,
                P_batch.device,
                zero=True,
            )
            P_padded[:batch_size].copy_(P_batch)
            P_batch = P_padded
            metas = list(metas) + [None] * pad
            batch_size = P_batch.size(0)

        comm_rank = dist.get_rank(comm_group)
        P_ortho_full = self._cached_buffer(
            "dense_replicate_p_ortho_full",
            P_batch.shape,
            P_batch.dtype,
            P_batch.device,
        )

        # Reference dion_update_ddp() orthogonalizes exactly one full matrix per
        # replicate rank. If the Megatron backend batches more logical parameters
        # than the replicate world size, replay that same contract chunk-wise
        # instead of orthogonalizing a rank-local batch with one shared RNG stream.
        for chunk_start in range(0, batch_size, comm_world_size):
            chunk_end = chunk_start + comm_world_size
            P_chunk = P_batch[chunk_start:chunk_end].contiguous()
            P_single = P_chunk[comm_rank : comm_rank + 1]
            local_index = chunk_start + comm_rank
            local_metas = metas[local_index : local_index + 1] if metas else None
            trace_extra = {
                "comm_rank": int(comm_rank),
                "comm_world_size": int(comm_world_size),
                "chunk_start": int(chunk_start),
                "chunk_end": int(chunk_end),
            }
            self._maybe_log_target_trace(
                tag="dense_replicate_pre_ortho",
                metas=local_metas,
                p_batch=P_single,
                real_batch_size=1,
                extra=trace_extra,
            )
            P_single_ortho = self._orthogonalize_reference_local_slice(
                P_single,
                metas=local_metas,
                tag="dense_replicate_reference_local",
            )
            self._maybe_log_target_trace(
                tag="dense_replicate_post_ortho",
                metas=local_metas,
                p_batch=P_single_ortho,
                real_batch_size=1,
                extra=trace_extra,
            )
            P_ortho_chunk = funcol.all_gather_tensor(
                P_single_ortho.contiguous(),
                gather_dim=0,
                group=comm_group,
            )
            yield
            P_ortho_full[chunk_start:chunk_end].copy_(P_ortho_chunk)

        return P_ortho_full[:original_batch_size]

    def _apply_batch_updates(
        self,
        params: List[Tensor],
        momentums: List[Tensor],
        Qs: List[Tensor],
        Q_new_batch: torch.Tensor,
        Q_state_batch: torch.Tensor,
        P_batch: torch.Tensor,
        configs: List[DionParamConfig],
        groups: List[dict],
        states: List[dict],
        metas: List,
        param_shapes: List[Tuple[int, int]],
        real_batch_size: int,
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
        scaled_lr = self._scaled_lr_for_2d_param(
            lr=lr,
            m_for_lr=m_for_lr,
            n_for_lr=n_for_lr,
        )

        wd_mult = groups[0].get('wd_mult', 1.0)
        weight_decay = groups[0].get('weight_decay', self.defaults['weight_decay'] * wd_mult)
        has_tp = self._has_active_tp_axis(configs[0])
        self._maybe_log_target_trace(
            tag="pre_param_update",
            metas=metas,
            qs=Qs,
            p_batch=P_batch,
            q_new_batch=Q_new_batch,
            deltas=delta_batch,
            params=params,
            real_batch_size=real_batch_size,
            extra={
                "scaled_lr": float(scaled_lr),
                "lr": float(lr),
                "weight_decay": float(weight_decay),
            },
        )

        for i in range(real_batch_size):
            param = params[i]
            delta = delta_batch[i]
            if delta.shape != param.shape:
                delta = delta.contiguous().view(param.shape)

            if self._debug_lr_scaling and self._step_count <= self._debug_lr_scaling_steps:
                meta = metas[i]
                param_name = getattr(meta, "param_name", None)
                if not param_name:
                    param_name = f"param_uid={getattr(meta, 'param_uid', 'unknown')}"
                if param_name not in self._debug_lr_scaling_logged_params:
                    logger.info(
                        "[DION_LR_SCALING] step=%s rule=%s param=%s local_shape=%s "
                        "global_shape=(%s,%s) m_for_lr=%s n_for_lr=%s lr=%.10e scaled_lr=%.10e",
                        self._step_count,
                        self.defaults.get("lr_scaling_rule", "moonlight"),
                        param_name,
                        tuple(param.shape),
                        m_global,
                        n_global,
                        m_for_lr,
                        n_for_lr,
                        lr,
                        scaled_lr,
                    )
                    self._debug_lr_scaling_logged_params.add(param_name)

            if weight_decay > 0:
                param.mul_(1 - lr * weight_decay)
            param.add_(delta.to(param.dtype), alpha=-scaled_lr)

            Q_state = Q_state_batch[i].to(Qs[i].dtype)
            if has_tp:
                Q_state = self._reshard_q_along_tp(Q_state, self.tp_group, self.tp_rank)
            Qs[i].copy_(Q_state)

        self._maybe_log_target_trace(
            tag="post_param_update",
            metas=metas,
            qs=Qs,
            params=params,
            real_batch_size=real_batch_size,
            extra={
                "scaled_lr": float(scaled_lr),
                "lr": float(lr),
                "weight_decay": float(weight_decay),
            },
        )

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

        cfg0 = configs[0] if configs else DionParamConfig()
        replicate_group = self._reference_replicate_group_for_config(cfg0)
        replicate_world_size = (
            dist.get_world_size(replicate_group) if replicate_group is not None else 1
        )
        use_compressed = (
            self.use_compressed_comm
            and replicate_world_size > 1
            and any(c.compressed_all_reduce for c in configs)
        )

        if not use_compressed:
            yield from self._collapse_grads_across_cp(grads, replicate_group=replicate_group)
        self._maybe_log_target_trace(
            tag="pre_momentum_add",
            metas=metas,
            grads=grads,
            momentums=momentums,
            qs=Qs,
            params=params,
            real_batch_size=real_batch_size,
            extra={
                "use_compressed": bool(use_compressed),
                "replicate_world_size": int(replicate_world_size),
            },
        )

        # STEP 1: M <- M + G (no decay - decay is applied in error feedback)
        if grads:
            # All momentums in a batch have the same dtype (set by mixed_precision_config)
            if momentums[0].dtype == grads[0].dtype:
                torch._foreach_add_(momentums, grads)
            else:
                for m, g in zip(momentums, grads):
                    m.add_(g.to(m.dtype))
        self._maybe_log_target_trace(
            tag="post_momentum_add",
            metas=metas,
            grads=grads,
            momentums=momentums,
            qs=Qs,
            params=params,
            real_batch_size=real_batch_size,
            extra={
                "use_compressed": bool(use_compressed),
                "replicate_world_size": int(replicate_world_size),
            },
        )

        # STEP 2: Q Unshard
        Q_for_matmul = yield from self._unshard_q_batch(Qs, configs, metas)
        self._maybe_log_target_trace(
            tag="post_q_unshard",
            metas=metas,
            momentums=momentums,
            qs=Q_for_matmul,
            params=params,
            real_batch_size=real_batch_size,
            extra={
                "use_compressed": bool(use_compressed),
                "replicate_world_size": int(replicate_world_size),
            },
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
        self._log_large_r_batch(
            tag="DION_DEBUG_LARGE_R_PRE_ORTHO",
            configs=configs,
            metas=metas,
            param_shapes=param_shapes,
            Q_batch=Q_batch,
            real_batch_size=real_batch_size,
        )

        P_batch = M_batch @ Q_batch
        self._maybe_log_target_trace(
            tag="post_matmul_pre_fs_reduce",
            metas=metas,
            momentums=momentums,
            qs=Qs,
            p_batch=P_batch,
            params=params,
            real_batch_size=real_batch_size,
            extra={
                "use_compressed": bool(use_compressed),
                "replicate_world_size": int(replicate_world_size),
            },
        )

        # STEP 3.5: All-Reduce for P (only if FS collectives are enabled)
        # NOTE: Expert params use different shard_group (EP-internal), so we group by shard_group
        if not (configs and all(self._is_fs_only_config(cfg) for cfg in configs)):
            yield from self._reduce_p_across_fs_groups(P_batch, configs, metas)
        self._maybe_log_target_trace(
            tag="post_fs_reduce_pre_ortho",
            metas=metas,
            momentums=momentums,
            qs=Qs,
            p_batch=P_batch,
            params=params,
            real_batch_size=real_batch_size,
            extra={
                "use_compressed": bool(use_compressed),
                "replicate_world_size": int(replicate_world_size),
            },
        )

        # STEP 4: Orthogonalize P
        if use_compressed:
            P_batch, R_batch = yield from self._run_compressed_comm_async(
                P_batch, M_batch, Q_batch, configs, metas
            )
        else:
            ortho_group = self._ortho_group_for_config(configs[0], metas[0] if metas else None)
            use_dense_replicate_batch = (
                replicate_world_size > 1
                and ortho_group is None
                and not (configs and all(self._is_fs_only_config(cfg) for cfg in configs))
            )
            if use_dense_replicate_batch:
                P_batch = yield from self._orthogonalize_dense_replicate_batch_async(
                    P_batch,
                    metas,
                    comm_group=replicate_group,
                )
            else:
                P_batch = yield from self._orthogonalize_p_batch(
                    P_batch,
                    configs,
                    metas,
                    real_batch_size=real_batch_size,
                )
            # STEP 5: R = M.T @ P
            R_batch = M_batch.mT @ P_batch
            yield from self._reduce_r_across_tp(R_batch, configs)
        self._maybe_log_target_trace(
            tag="post_pr",
            metas=metas,
            momentums=momentums,
            qs=Qs,
            p_batch=P_batch,
            r_batch=R_batch,
            params=params,
            real_batch_size=real_batch_size,
            extra={
                "use_compressed": bool(use_compressed),
            },
        )

        # STEP 6: Fix NaN/zero
        self._log_p_orthogonality_snapshot(
            tag="DION_DEBUG_P_PRE_FIX",
            P_batch=P_batch,
            metas=metas,
            real_batch_size=real_batch_size,
            extra={"tag": "pre_fix"},
            force=self._debug_post_fix,
        )
        P_batch, R_batch = self._fix_bad_batch(
            P_batch,
            R_batch,
            Q_batch,
            M_batch,
            real_batch_size=real_batch_size,
            global_param_offset=global_param_offset,
            configs=configs,
        )
        self._log_p_orthogonality_snapshot(
            tag="DION_DEBUG_P_POST_FIX",
            P_batch=P_batch,
            metas=metas,
            real_batch_size=real_batch_size,
            extra={"tag": "post_fix"},
            force=self._debug_post_fix,
        )
        self._check_p_orthogonality(P_batch, configs, metas, real_batch_size)

        # STEP 7: Error feedback
        self._batch_error_feedback(momentums, P_batch, R_batch, configs, groups)

        # STEP 8: Column normalize R -> Q_new
        Q_new_batch = yield from self._normalize_cols_async(R_batch, configs, metas, real_batch_size, global_param_offset)
        self._check_q_column_norms(Q_new_batch, configs, metas, real_batch_size)
        self._maybe_log_target_trace(
            tag="post_q_norm",
            metas=metas,
            momentums=momentums,
            qs=Qs,
            p_batch=P_batch,
            r_batch=R_batch,
            q_new_batch=Q_new_batch,
            params=params,
            real_batch_size=real_batch_size,
        )
        # `dion_reference.py` persists the column-normalized `Q_new` directly.
        # Do not apply any extra basis alignment or right-rotation here: those
        # operations are outside the reference Dion contract and can make the
        # persisted optimizer state topology-dependent even when the current-step
        # update `delta = P @ Q_new^T` is still close.
        Q_state_batch = Q_new_batch

        self._apply_batch_updates(
            params,
            momentums,
            Qs,
            Q_new_batch,
            Q_state_batch,
            P_batch,
            configs,
            groups,
            states,
            metas,
            param_shapes,
            real_batch_size,
        )

        del M_batch, Q_batch, P_batch, R_batch, Q_new_batch, Q_state_batch

    def _run_compressed_comm_async(
        self,
        P_batch: torch.Tensor,
        M_batch: torch.Tensor,
        Q_batch: torch.Tensor,
        configs: List[DionParamConfig],
        metas: List,
    ) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
        """Compressed communication for Dion optimizer."""
        cfg0 = configs[0] if configs else DionParamConfig()
        meta0 = metas[0] if metas else None
        comm_group = self._reference_replicate_group_for_config(cfg0)
        comm_world_size = dist.get_world_size(comm_group) if comm_group else 1
        ortho_group = self._ortho_group_for_config(cfg0, meta0)
        batch_size = P_batch.size(0)
        cfg = configs[0] if configs else DionParamConfig()
        p_is_tp_sharded = self._p_is_tp_sharded(cfg)
        all_fs_only = bool(configs) and all(self._is_fs_only_config(config) for config in configs)
        fs_group = self._fs_group_for_meta(metas[0] if metas else None)
        fs_only_replicate_group = None
        fs_only_replicate_ranks = None

        if all_fs_only and self.use_fs_collectives and fs_group and dist.get_world_size(fs_group) > 1:
            # Reference `dion_update_fsdp()` semantics do not depend on replicate mesh size.
            # The fs-only kernel always:
            # 1. reduce-scatters P across the outer-shard mesh,
            # 2. orthogonalizes one full matrix per shard rank,
            # 3. all-gathers P_ortho back before forming R,
            # and only additionally averages P/R over the replicate mesh when that mesh
            # actually has size > 1. RP=1 must therefore still take the fs-only kernel.
            if comm_group is not None and comm_world_size > 1:
                fs_only_replicate_group, fs_only_replicate_ranks = (
                    self._fs_only_compressed_replicate_spec()
                )
            P_ortho_full = yield from self._orthogonalize_fs_only_batch(
                P_batch,
                metas,
                replicate_group=fs_only_replicate_group,
                replicate_ranks=fs_only_replicate_ranks,
            )
            R_batch = M_batch.mT @ P_ortho_full
            yield from self._reduce_r_across_tp(R_batch, configs)
            if fs_only_replicate_group is not None:
                if fs_only_replicate_ranks is not None:
                    yield from self._collapse_batch_over_rank_subset(
                        R_batch,
                        primary_group=fs_only_replicate_group,
                        subset_ranks=fs_only_replicate_ranks,
                    )
                else:
                    yield from self._collapse_batch_across_cp(
                        R_batch,
                        replicate_group=fs_only_replicate_group,
                    )
            return P_ortho_full, R_batch

        if comm_group is None or comm_world_size <= 1:
            yield from self._collapse_batch_across_cp(P_batch, replicate_group=comm_group)
            # RP=1: no P all-reduce needed, but orthogonalization is always required
            if ortho_group is not None:
                P_ortho = self._distributed_orthogonalize(
                    P_batch,
                    shard_group=ortho_group,
                    oversample=self.defaults['rcqr_oversample'],
                    metas=metas,
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
            yield from self._collapse_batch_across_cp(R_batch, replicate_group=comm_group)

            return P_ortho, R_batch

        # Rest of compressed communication for RP > 1
        original_batch_size = batch_size

        if p_is_tp_sharded and self.tp_group and dist.get_world_size(self.tp_group) > 1:
            # Match dion_reference FSDP+TP compressed contract:
            # average replicated P shards before distributed orthogonalization.
            yield from self._collapse_batch_across_cp(P_batch, replicate_group=comm_group)
            P_ortho_full = self._distributed_orthogonalize(
                P_batch,
                shard_group=self.tp_group,
                oversample=self.defaults['rcqr_oversample'],
                metas=metas,
            )
        else:
            # Match dion_reference.dion_update_ddp() compressed contract:
            # batch the dense params in comm_world_size-sized chunks, reduce-scatter
            # each chunk along the batch dim, orthogonalize exactly one matrix per
            # rank, then all-gather the orthogonalized chunk back.
            if batch_size % comm_world_size != 0:
                pad = comm_world_size - (batch_size % comm_world_size)
                padded_batch_size = batch_size + pad
                P_padded = self._cached_buffer(
                    "compressed_p_padded",
                    (padded_batch_size, P_batch.size(1), P_batch.size(2)),
                    P_batch.dtype,
                    P_batch.device,
                    zero=True,
                )
                M_padded = self._cached_buffer(
                    "compressed_m_padded",
                    (padded_batch_size, M_batch.size(1), M_batch.size(2)),
                    M_batch.dtype,
                    M_batch.device,
                    zero=True,
                )
                Q_padded = self._cached_buffer(
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

            comm_rank = dist.get_rank(comm_group)
            P_ortho_full = self._cached_buffer(
                "compressed_p_ortho_full",
                P_batch.shape,
                P_batch.dtype,
                P_batch.device,
            )

            for chunk_start in range(0, batch_size, comm_world_size):
                chunk_end = chunk_start + comm_world_size
                P_chunk = P_batch[chunk_start:chunk_end].contiguous()
                P_single = funcol.reduce_scatter_tensor(
                    P_chunk,
                    reduceOp="avg",
                    scatter_dim=0,
                    group=comm_group,
                )
                yield

                if P_single.size(0) != 1:
                    raise RuntimeError(
                        "[DION_INVALID_DENSE_COMPRESSED_LOCAL_CHUNK] "
                        f"step={self._step_count} comm_world_size={comm_world_size} "
                        f"local_chunk={P_single.size(0)} chunk_start={chunk_start}"
                    )

                local_index = chunk_start + comm_rank
                local_metas = metas[local_index : local_index + 1] if metas else None
                trace_extra = {
                    "comm_rank": int(comm_rank),
                    "comm_world_size": int(comm_world_size),
                    "chunk_start": int(chunk_start),
                    "chunk_end": int(chunk_end),
                }
                self._maybe_log_target_trace(
                    tag="compressed_pre_ortho",
                    metas=local_metas,
                    p_batch=P_single,
                    real_batch_size=1,
                    extra=trace_extra,
                )
                P_ortho_single = self._orthogonalize_reference_local_slice(
                    P_single,
                    metas=local_metas,
                    tag="compressed_reference_local",
                )
                if local_metas and len(local_metas) == 1:
                    self._maybe_compare_local_logical(
                        P_in=P_single,
                        P_runtime=P_ortho_single,
                        meta=local_metas[0],
                        tag="compressed_local",
                    )
                self._maybe_log_target_trace(
                    tag="compressed_post_ortho",
                    metas=local_metas,
                    p_batch=P_ortho_single,
                    real_batch_size=1,
                    extra=trace_extra,
                )
                P_ortho_chunk = funcol.all_gather_tensor(
                    P_ortho_single.contiguous(),
                    gather_dim=0,
                    group=comm_group,
                )
                yield
                P_ortho_full[chunk_start:chunk_end].copy_(P_ortho_chunk)

        R_batch = M_batch.mT @ P_ortho_full
        yield from self._reduce_r_across_tp(R_batch, configs)
        if all_fs_only and self.use_fs_collectives and fs_group and dist.get_world_size(fs_group) > 1:
            if fs_only_replicate_ranks is not None:
                yield from self._collapse_batch_over_rank_subset(
                    R_batch,
                    primary_group=fs_only_replicate_group,
                    subset_ranks=fs_only_replicate_ranks,
                )
            else:
                yield from self._collapse_batch_across_cp(
                    R_batch,
                    replicate_group=fs_only_replicate_group,
                )
        else:
            yield from self._collapse_batch_across_cp(R_batch, replicate_group=comm_group)

        P_ortho_full = P_ortho_full[:original_batch_size]
        R_batch = R_batch[:original_batch_size]

        return P_ortho_full, R_batch

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
        if Q_clean.shape != R_batch.shape:
            raise RuntimeError(
                "[DION_BAD_BATCH_Q_SHAPE_MISMATCH] "
                f"Q_shape={tuple(Q_clean.shape)} R_shape={tuple(R_batch.shape)}"
            )
        R_batch = R_batch.nan_to_num() * not_all_zero + Q_clean * is_all_zero

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

        result = torch.empty_like(R_batch)
        epsilon = self.defaults['epsilon']
        if real_batch_size <= 0:
            return result

        R_real = R_batch[:real_batch_size].to(torch.float32)
        col_sum_sq_real = R_real.square().sum(dim=1, keepdim=True)
        self._log_q_norm_snapshot(
            tag="DION_ORTHO_TRACE_Q_LOCAL",
            col_sum_sq=col_sum_sq_real,
            metas=metas,
            real_batch_size=real_batch_size,
            extra={"phase": "local_before_reduce"},
        )

        fs_groups = []
        for i in range(real_batch_size):
            cfg = configs[i]
            meta = metas[i] if metas is not None and i < len(metas) else None
            shard_group = self._fs_group_for_meta(meta)
            need_reduce = (
                cfg.has_fs_axis
                and self.use_fs_collectives
                and shard_group is not None
                and dist.get_world_size(shard_group) > 1
            )
            fs_groups.append(shard_group if need_reduce else None)

        first_group = fs_groups[0] if fs_groups else None
        for idx, group in enumerate(fs_groups):
            if group is not first_group:
                raise RuntimeError(
                    "[DION_Q_NORM_GROUP_MISMATCH] "
                    f"step={self._step_count} slot={idx} expected_group={id(first_group) if first_group is not None else 0} "
                    f"got_group={id(group) if group is not None else 0}"
                )

        if first_group is not None:
            dist.all_reduce(col_sum_sq_real, op=dist.ReduceOp.SUM, group=first_group)
            yield
        self._log_q_norm_snapshot(
            tag="DION_ORTHO_TRACE_Q_GLOBAL",
            col_sum_sq=col_sum_sq_real,
            metas=metas,
            real_batch_size=real_batch_size,
            extra={
                "phase": "global_after_reduce",
                "reduced": bool(first_group is not None),
            },
        )

        col_norms_real = col_sum_sq_real.sqrt()
        result[:real_batch_size].copy_(R_batch[:real_batch_size] / (col_norms_real + epsilon))
        post_col_sum_sq_real = result[:real_batch_size].to(torch.float32).square().sum(dim=1)
        if first_group is not None:
            dist.all_reduce(post_col_sum_sq_real, op=dist.ReduceOp.SUM, group=first_group)
            yield
        self._log_q_norm_snapshot(
            tag="DION_ORTHO_TRACE_Q_POST",
            col_sum_sq=post_col_sum_sq_real,
            metas=metas,
            real_batch_size=real_batch_size,
            extra={
                "phase": "post_normalize_global",
                "reduced": bool(first_group is not None),
            },
        )

        if real_batch_size < batch_size:
            result[real_batch_size:].copy_(R_batch[real_batch_size:])

        return result
