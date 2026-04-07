"""
Distributed optimizer wrapper for Dion optimizer in Megatron-LM.

Supports orthogonal TP × FS sharding:
- tp_split_dim=0 (ColumnParallel): FS shards cols
- tp_split_dim=1 (RowParallel): FS shards rows
"""

import logging
import os
import traceback
import torch
import torch.distributed as dist
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from .distrib_optimizer import (
    DistributedOptimizer,
    Range,
)
from .megatron_dion import MegatronDion, MegatronDionDistMeta
from .distrib_dion.bucket_layout import (
    DionBucketEntry,
    DionBucketLayout,
    build_dion_entries_,
    bucket_rank_range_,
    param_rank_range_,
    select_non_dion_bucket_params_,
)
from .distrib_dion.checkpoint_io import (
    apply_non_dion_shards_,
    apply_group_shards_to_model_params_,
    build_persistent_dion_param_state,
    copy_group_params_to_main_shards_,
    restore_persistent_dion_param_state_,
)
from .distrib_dion.grad_wiring import (
    copy_dion_grads_,
    copy_non_dion_grads_,
)
from .distrib_dion.fs_layout import (
    get_fs_split_dim,
    slice_fs_shard_2d,
)
from .distrib_dion.overlap_sync import (
    finish_bucket_group_grad_sync_,
    release_rs_buffers_,
)
from .distrib_dion.param_selection import is_dion_param, is_moe_expert_param
from .distrib_dion.param_update import check_shard_identity_
from .distrib_dion.param_utils import get_tp_split_dim, is_tp_enabled
from .distrib_dion.shard_info import DionShardLayout, DionShardBinding
from .. import parallel_state, tensor_parallel
from ..distributed.param_and_grad_buffer import shard_buffer
from ..fp8_utils import is_float8tensor
from ..transformer.fsdp_dtensor_checkpoint import get_global_unique_param_name
from ..utils import get_data_parallel_group_if_dtensor, to_local_if_dtensor

logger = logging.getLogger(__name__)
_PP_GRADNORM_PROBE_CALL_IDX = 0
_GRAD_NORM_TOPK_CALL_IDX = 0


class _DeferredHandle:
    """Waitable handle that runs a local callback after the collective completes."""

    def __init__(self, work, callback=None):
        self._work = work
        self._callback = callback

    def wait(self):
        if self._work is not None:
            self._work.wait()
        if self._callback is not None:
            self._callback()
        self._work = None
        self._callback = None


class _HandleGroup:
    """Aggregate multiple waitable handles behind one `.wait()` call."""

    def __init__(self, handles):
        self._handles = [handle for handle in handles if handle is not None]

    def wait(self):
        for handle in self._handles:
            handle.wait()
        self._handles = []


@dataclass
class _PreparedDionGatherEntry:
    """Prepacked Dion local shard input for one canonical bucket gather."""

    entry: DionBucketEntry
    full_view_2d: torch.Tensor
    shard_buffer: torch.Tensor


@dataclass
class _PreparedDionBucketGather:
    """Bucket-local Dion gather state for one bucket-wise all-gather."""

    prepared_buffer: torch.Tensor
    entries: List[_PreparedDionGatherEntry]
class DistributedOptimizerForDion(DistributedOptimizer):
    """
    Distributed optimizer for MegatronDion with true parameter sharding.

    Architecture:
    - True parameter sharding: Only local shards stored on GPU
    - Buffer sizes reduced to shard size (not full size)
    - Bucket-wise all-gather/reduce-scatter via standard Megatron-Core DO
    - FSDP-style memory efficiency: Full params only during forward/backward

    Extends DistributedOptimizer to support Dion's 2D parallelism model:
    - RP (Replicate Process): Gradient averaging across replicas with same FS shard
    - FS (Fully Shard): True row-wise sharding for 2D params (GPU memory saved)
    - TP (Tensor Parallel): Column-wise tensor sharding

    Provides automatic configuration of 2D process groups and FS-aware annotation.
    """

    @staticmethod
    def _group_size(group) -> int:
        return group.size() if hasattr(group, "size") else dist.get_world_size(group)

    @staticmethod
    def _group_rank(group) -> int:
        return group.rank() if hasattr(group, "rank") else dist.get_rank(group)

    @classmethod
    def _bucket_shard_group(cls, param_and_grad_buffer, bucket):
        """Return the standard local-shard group that owns this bucket."""
        expert_flags = [not getattr(param, "allreduce", True) for param in bucket.params]
        if any(expert_flags):
            if not all(expert_flags):
                raise RuntimeError(
                    f"[Dion][EP] mixed dense/expert bucket is invalid: bucket_id={bucket.bucket_id}"
                )
            group = parallel_state.get_expert_data_parallel_group(
                check_initialized=False,
                partial_expert_data_parallel=True,
            )
            if group is None:
                raise RuntimeError(
                    f"[Dion][EP] missing expert local-shard group for bucket {bucket.bucket_id}"
                )
            return group

        group = getattr(param_and_grad_buffer, "data_parallel_group", None)
        if group is None:
            raise RuntimeError(
                f"[Dion] missing standard local-shard group for bucket {bucket.bucket_id}"
            )
        return group

    @classmethod
    def _bucket_shard_topology(cls, param_and_grad_buffer, bucket) -> Tuple[object, int, int]:
        """Return the standard bucket shard group and this rank's position in it."""
        shard_group = cls._bucket_shard_group(param_and_grad_buffer, bucket)
        shard_size = cls._group_size(shard_group)
        shard_rank = cls._group_rank(shard_group)
        if shard_size <= 0:
            raise RuntimeError(
                f"[Dion] invalid shard group size for bucket {bucket.bucket_id}: {shard_size}"
            )
        return shard_group, shard_size, shard_rank

    @classmethod
    def _build_bucket_param_map(
        cls,
        parent_result,
        ordered_params,
        dp_group,
        dp_rank,
        bucket_index,
        param_index_map,
        bucket_offset: int,
        bucket_size: int,
        bucket_param_to_index=None,
        param_to_name=None,
    ):
        """Rebuild the parent DO param_map in canonical bucket-param order."""
        from collections import OrderedDict
        from .distrib_optimizer import DistributedOptimizer

        parent_param_map = parent_result["param_map"]
        dp_world_size = dp_group.size()
        if bucket_size % dp_world_size != 0:
            raise RuntimeError(
                f"[Dion] bucket_size must be divisible by dp_size for canonical param_map "
                f"(bucket={bucket_index}, bucket_size={bucket_size}, dp_size={dp_world_size})"
            )

        max_gbuf_range_size = bucket_size // dp_world_size
        gbuf_world_start = dp_rank * max_gbuf_range_size
        gbuf_world_end = min(bucket_size, gbuf_world_start + max_gbuf_range_size)
        gbuf_world_range = Range(
            gbuf_world_start + bucket_offset,
            gbuf_world_end + bucket_offset,
        )
        reconstructed_param_map = DistributedOptimizer._build_model_gbuf_param_range_map(
            param_index_map,
            gbuf_world_range,
            bucket_offset,
        )

        canonical_param_map = OrderedDict()
        for param in ordered_params:
            parent_info = parent_param_map.get(param)
            reconstructed_info = reconstructed_param_map.get(
                param,
                {
                    "param": Range(0, 0),
                    "gbuf_world": Range(0, 0),
                    "gbuf_local": Range(0, 0),
                    "gbuf_world_in_bucket": Range(0, 0),
                },
            )
            chosen_info = parent_info or reconstructed_info
            canonical_param_map[param] = chosen_info

        parent_result["param_map"] = canonical_param_map
        return canonical_param_map

    @classmethod
    def _mark_bucket_dion_params(cls, param_map, param_to_name, fs_size):
        """Classify bucket params and build static Dion metadata once."""
        from ..parallel_state import get_tensor_model_parallel_world_size

        dion_param_count = 0
        dion_static_info_by_param = {}

        for param in param_map.keys():
            param_name = None
            if param_to_name is not None and param in param_to_name:
                param_name = param_to_name[param]
            if param_name:
                param._param_name = param_name

            param.is_dion_param = is_dion_param(param, param_name)

            is_expert = is_moe_expert_param(param, param_name)
            tp_split_dim = get_tp_split_dim(param)
            has_tp = is_tp_enabled(param)
            if is_expert and has_tp:
                from megatron.core import parallel_state

                tp_world_size = parallel_state.get_expert_tensor_parallel_world_size()
            else:
                tp_world_size = get_tensor_model_parallel_world_size() if has_tp else 1

            if not param.is_dion_param:
                continue

            dion_param_count += 1
            m, n = param.shape
            fs_split_dim = get_fs_split_dim(tp_split_dim)

            if tp_split_dim == 0:
                global_m = m * tp_world_size
                global_n = n
            elif tp_split_dim == 1:
                global_m = m
                global_n = n * tp_world_size
            else:
                global_m = m
                global_n = n

            num_local_experts = getattr(param, "num_local_experts", None)
            if num_local_experts is not None and num_local_experts > 1:
                if tp_split_dim == 0:
                    per_expert_global_shape = (global_m // num_local_experts, global_n)
                elif tp_split_dim == 1:
                    per_expert_global_shape = (global_m, global_n // num_local_experts)
                else:
                    per_expert_global_shape = (global_m, global_n)
            else:
                per_expert_global_shape = None

            dion_static_info_by_param[param] = {
                "is_dion": True,
                "global_shape": (global_m, global_n),
                "fs_split_dim": fs_split_dim,
                "tp_split_dim": tp_split_dim,
                "per_expert_global_shape": per_expert_global_shape,
            }

        return dion_param_count, dion_static_info_by_param

    def _init_runtime_groups(self) -> None:
        """Resolve the standard runtime groups that Dion math consumes."""
        optimizer_fs_group = getattr(self.optimizer, "fs_group", None) if hasattr(self, "optimizer") else None
        if optimizer_fs_group is not None:
            self.fs_group = optimizer_fs_group
            self.fs_size = self._group_size(optimizer_fs_group)
            self.fs_rank = self._group_rank(optimizer_fs_group)
        else:
            self.fs_group = self.data_parallel_group
            self.fs_size = self._group_size(self.data_parallel_group)
            self.fs_rank = self._group_rank(self.data_parallel_group)

        configured_fs_size = int(self._fs_size) if self._fs_size is not None else 1
        if self.fs_size != configured_fs_size:
            raise RuntimeError(
                f"[Dion] Global rank {self._global_rank}: "
                f"FS size mismatch configured={configured_fs_size} actual={self.fs_size}"
            )

        configured_rp_size = int(self._rp_size) if self._rp_size is not None else 1
        optimizer_rp_group = getattr(self.optimizer, "rp_group", None) if hasattr(self, "optimizer") else None
        if configured_rp_size <= 1:
            self.rp_group = None
        else:
            if optimizer_rp_group is None:
                raise RuntimeError(
                    "[Dion] RP>1 requires optimizer.rp_group to be initialized "
                    f"(configured_rp_size={configured_rp_size})"
                )
            rp_size = self._group_size(optimizer_rp_group)
            if rp_size != configured_rp_size:
                group_ranks = dist.get_process_group_ranks(optimizer_rp_group)
                raise RuntimeError(
                    "[Dion] RP topology mismatch after parent init: "
                    f"configured_rp_size={configured_rp_size} actual_rp_group_size={rp_size} "
                    f"rp_group_ranks={group_ranks}"
                )
            self.rp_group = optimizer_rp_group

        try:
            self.state_replica_group = parallel_state.get_inter_distributed_optimizer_instance_group(
                check_initialized=False
            )
        except Exception:
            self.state_replica_group = None

    @staticmethod
    def _lookup_param_name(name_map, param) -> Optional[str]:
        """Best-effort name lookup for maps keyed by either param or id(param)."""
        if not name_map:
            return None
        try:
            name = name_map.get(id(param))
            if name is not None:
                return name
        except Exception:
            pass
        try:
            return name_map.get(param)
        except Exception:
            return None

    def _find_param_name(self, param) -> Optional[str]:
        """Slow-path runtime param-name lookup for current module objects."""
        if hasattr(self, "model_chunks"):
            for model in self.model_chunks:
                try:
                    for name, model_param in model.named_parameters():
                        if id(model_param) == id(param):
                            return name
                except Exception:
                    continue
        elif hasattr(self, "module") and isinstance(self.module, torch.nn.Module):
            for name, model_param in self.module.named_parameters():
                if id(model_param) == id(param):
                    return name
        return None

    def _canonical_param_name(self, param) -> Optional[str]:
        """Return the most direct stable name available for a model or shard param."""
        if param is None:
            return None

        name = self._lookup_param_name(getattr(self, "param_to_name", None), param)
        if name is not None:
            return name

        model_param = getattr(param, "_model_param", None)
        if model_param is not None:
            name = self._lookup_param_name(getattr(self, "param_to_name", None), model_param)
            if name is not None:
                return name
            name = getattr(model_param, "_param_name", None)
            if name is not None:
                return name

        name = getattr(param, "_param_name", None)
        if name is not None:
            return name

        return self._find_param_name(model_param if model_param is not None else param)

    def _logical_param_name(self, param) -> Optional[str]:
        """Return a PP/EP-invariant logical parameter name when available."""
        if param is None:
            return None

        model_param = getattr(param, "_model_param", None)
        if model_param is None:
            model_param = param

        if hasattr(self, "model_chunks"):
            try:
                return get_global_unique_param_name(self.model_chunks, model_param)
            except Exception:
                pass

        return self._canonical_param_name(param)

    def _shard_param_name(self, shard_param) -> str:
        """Return a deterministic checkpoint/log name for one optimizer shard param."""
        name = self._canonical_param_name(shard_param)
        if name is not None:
            return name
        return f"id_{id(shard_param)}"

    def _shard_param_uid(self, shard_param):
        """Return the logical checkpoint/state identity for one optimizer shard."""
        param_uid = getattr(shard_param, "_dion_param_uid", None)
        if param_uid is not None:
            return param_uid
        meta = getattr(self, "dist_metas", {}).get(shard_param)
        if meta is not None and getattr(meta, "param_uid", None) is not None:
            return meta.param_uid
        raise RuntimeError(
            "[Dion] missing param_uid for optimizer shard "
            f"name={self._canonical_param_name(shard_param) or f'id_{id(shard_param)}'} "
            f"shape={tuple(shard_param.shape)}"
        )

    @staticmethod
    def _bucket_param_view(bucket, param) -> Optional[torch.Tensor]:
        """Return the canonical full-param view for one bucket param."""
        if bucket is None or getattr(bucket, "param_data", None) is None:
            return None
        if not hasattr(bucket, "param_to_index") or param not in bucket.param_to_index:
            return None
        start, end = bucket.param_to_index[param]
        return bucket.param_data.view(-1)[start:end].view(param.data.shape)

    @staticmethod
    def _bucket_full_param_view_2d(bucket, param, entry) -> Optional[torch.Tensor]:
        """Return the canonical full 2D bucket view for one Dion param."""
        if bucket is None or getattr(bucket, "param_data", None) is None:
            return None
        if not hasattr(bucket, "param_to_index") or param not in bucket.param_to_index:
            return None
        start, end = bucket.param_to_index[param]
        full_flat = bucket.param_data.view(-1)[start:end]
        global_shape = entry.global_shape
        fs_split_dim = int(entry.fs_split_dim)
        if fs_split_dim == 0:
            full_shape = (int(global_shape[0]), int(param.shape[1]))
        else:
            full_shape = (int(param.shape[0]), int(global_shape[1]))
        expected_numel = int(full_shape[0]) * int(full_shape[1])
        if full_flat.numel() != expected_numel:
            raise RuntimeError(
                "[Dion] canonical full-param view size mismatch "
                f"for param={getattr(param, 'shape', None)} bucket={getattr(bucket, 'bucket_id', -1)} "
                f"slice_numel={int(full_flat.numel())} expected_numel={expected_numel} "
                f"full_shape={full_shape}"
            )
        return full_flat.view(full_shape)

    def _bind_bucket_param_views(
        self,
        bucket,
        *,
        copy_data: bool,
        params: Optional[list[torch.nn.Parameter]] = None,
    ) -> None:
        """Ensure selected params in `bucket` alias the bucket's canonical param buffer."""
        if bucket is None or getattr(bucket, "param_data", None) is None:
            return
        params_to_bind = list(bucket.params) if params is None else list(params)
        source_copies = {}
        if copy_data:
            for param in params_to_bind:
                expected_view = self._bucket_param_view(bucket, param)
                if expected_view is None:
                    continue
                if (
                    param.data.shape == expected_view.shape
                    and param.data.data_ptr() == expected_view.data_ptr()
                ):
                    continue
                if param.data.numel() != expected_view.numel():
                    continue
                source_copies[param] = param.data.view(expected_view.shape).clone()
        for param in params_to_bind:
            expected_view = self._bucket_param_view(bucket, param)
            if expected_view is None:
                continue
            if (
                param.data.shape == expected_view.shape
                and param.data.data_ptr() == expected_view.data_ptr()
            ):
                continue
            if copy_data and param in source_copies:
                source_view = source_copies[param]
                expected_view.copy_(source_view)
            param.data = expected_view

    def _check_bucket_param_views(
        self,
        bucket,
        *,
        context: str,
        params: Optional[list[torch.nn.Parameter]] = None,
    ) -> None:
        """Verify selected param views still alias the canonical bucket buffer."""
        if bucket is None or not getattr(bucket, "_dion_requires_param_sync_check", False):
            return
        if getattr(bucket, "param_data", None) is None:
            raise RuntimeError(
                f"[Dion] {context}: bucket {getattr(bucket, 'bucket_id', -1)} missing bucket.param_data"
            )
        params_to_check = list(bucket.params) if params is None else list(params)
        for param in params_to_check:
            expected_view = self._bucket_param_view(bucket, param)
            if expected_view is None:
                continue
            if (
                param.data.shape != expected_view.shape
                or param.data.data_ptr() != expected_view.data_ptr()
            ):
                param_name = self._lookup_param_name(getattr(self, "param_to_name", None), param)
                if param_name is None:
                    param_name = self._find_param_name(param) or f"id_{id(param)}"
                logger.error(
                    "[DION_PARAM_BUFFER_VIEW_MISMATCH] context=%s bucket_id=%s param=%s param_ptr=%s bucket_ptr=%s",
                    context,
                    getattr(bucket, "bucket_id", -1),
                    param_name,
                    param.data.data_ptr(),
                    expected_view.data_ptr(),
                )
                raise RuntimeError(
                    f"[Dion] {context}: param.data no longer aliases bucket.param_data for {param_name}"
                )

    def _mark_buckets_full_param_ready(self, ready: bool) -> None:
        """Track whether full param views are ready for forward."""
        if not hasattr(self, "buffers"):
            return
        for buffer in self.buffers:
            for bucket in buffer.buckets:
                if hasattr(bucket, "_dion_full_param_ready"):
                    bucket._dion_full_param_ready = ready

    def _init_bucket_comm(self, bucket, fs_group) -> None:
        """Attach the Dion shard group to a bucket."""
        bucket.dion_shard_group = fs_group

        expert_flags = [not getattr(p, "allreduce", True) for p in bucket.params]
        is_expert_bucket = any(expert_flags)
        if is_expert_bucket and not all(expert_flags):
            raise RuntimeError(
                f"[Dion][EP] mixed dense/expert bucket is invalid for Dion EP: "
                f"bucket_id={bucket.bucket_id} param_count={len(bucket.params)}"
            )
        if is_expert_bucket:
            expert_group = self._expected_expert_shard_group()
            self._assert_group_matches(
                label=f"expert bucket {bucket.bucket_id}",
                actual_group=fs_group,
                expected_group=expert_group,
                extra="Megatron-Core EP local-shard ownership must stay on intra_expt_dp_group.",
            )

    def _attach_dion_bucket_layout(
        self,
        *,
        gbuf_idx: int,
        buffer,
        bucket,
        dion_bucket_layout: DionBucketLayout,
    ) -> None:
        """Validate that planned Dion entries already target the current runtime bucket."""
        if not hasattr(bucket, "param_to_index") or bucket.param_to_index is None:
            raise RuntimeError(
                f"[Dion] buffer={gbuf_idx} bucket={bucket.bucket_id} missing bucket.param_to_index"
            )

        for entry in dion_bucket_layout.entries:
            entry_param = entry.param
            if entry_param not in bucket.param_to_index:
                buffer_name_map = getattr(buffer, "param_to_name", None)
                param_name = self._lookup_param_name(buffer_name_map, entry_param)
                raise RuntimeError(
                    f"[Dion] buffer={gbuf_idx} bucket={bucket.bucket_id} planned param "
                    f"{param_name or f'id_{id(entry_param)}'} "
                    "is not present in the current runtime bucket"
                )
        bucket.dion_layout = dion_bucket_layout
        for entry in dion_bucket_layout.entries:
            self._dion_buckets_by_param[entry.param] = bucket
            self._dion_entries_by_param[entry.param] = entry

    def _init_dion_bucket(
        self,
        *,
        gbuf_idx: int,
        buffer,
        bucket,
        dion_bucket_layout: DionBucketLayout,
        fs_group,
    ) -> None:
        """Configure one bucket that contains at least one Dion param."""
        self._attach_dion_bucket_layout(
            gbuf_idx=gbuf_idx,
            buffer=buffer,
            bucket=bucket,
            dion_bucket_layout=dion_bucket_layout,
        )
        self._init_bucket_comm(bucket, fs_group)

        bucket.dion_optimizer = self
        bucket._dion_requires_param_sync_check = True
        bucket._dion_full_param_ready = True
        self._bind_bucket_param_views(bucket, copy_data=True)

    def _init_non_dion_bucket(self, *, gbuf_idx: int, buffer, bucket, fs_group) -> None:
        """Configure one bucket that has no Dion layout."""
        has_dion = any(getattr(param, "is_dion_param", False) for param in bucket.params)
        if has_dion:
            name_map = getattr(self, "param_to_name", {}) or getattr(buffer, "param_to_name", {})
            dion_params = []
            for param in bucket.params:
                if getattr(param, "is_dion_param", False):
                    param_name = self._lookup_param_name(name_map, param)
                    dion_params.append((id(param), param_name or f"id_{id(param)}", tuple(param.shape)))
            logger.error(
                f"[Dion] Buffer {gbuf_idx} Bucket {bucket.bucket_id} has Dion params but no dion_bucket_layout. params={dion_params}"
            )
            raise RuntimeError(
                f"[Dion] Buffer {gbuf_idx} Bucket {bucket.bucket_id} has Dion params but no dion_bucket_layout"
            )

        bucket.dion_layout = None
        self._init_bucket_comm(bucket, fs_group)
        bucket.dion_optimizer = self
        bucket._dion_requires_param_sync_check = False
        bucket._dion_full_param_ready = True

    def _init_dion_buffers(self) -> None:
        """Configure buffer-level Dion bucket layouts after parent optimizer init."""
        if not hasattr(self, "gbuf_ranges") or not hasattr(self, "buffers"):
            return

        fs_group = self.fs_group

        for gbuf_idx, gbuf_range_maps in enumerate(self.gbuf_ranges):
            buffer = self.buffers[gbuf_idx]
            dtype_key = (buffer.param_dtype, buffer.grad_dtype)
            if dtype_key not in gbuf_range_maps:
                raise RuntimeError(
                    f"[Dion] missing gbuf_ranges entry for buffer={gbuf_idx} dtype={dtype_key}"
                )
            bucket_range_maps = gbuf_range_maps[dtype_key]

            for bucket in buffer.buckets:
                bucket_range_map = bucket_range_maps[bucket.bucket_id]
                dion_bucket_layout = bucket_range_map.pop("dion_bucket_layout", None)

                if dion_bucket_layout is not None and dion_bucket_layout.has_params:
                    self._init_dion_bucket(
                        gbuf_idx=gbuf_idx,
                        buffer=buffer,
                        bucket=bucket,
                        dion_bucket_layout=dion_bucket_layout,
                        fs_group=fs_group,
                    )
                else:
                    self._init_non_dion_bucket(
                        gbuf_idx=gbuf_idx,
                        buffer=buffer,
                        bucket=bucket,
                        fs_group=fs_group,
                    )

    def _log_unmapped_main_grads(self, model_groups, label: str) -> None:
        """Fail fast if any params are missing from Dion bookkeeping.

        Pure non-Dion buckets are handled by the parent DO path and should not be
        flagged as uncategorized.
        """
        missing_gbuf_map = 0
        missing_param_map = 0
        uncategorized = 0
        examples = []
        for group in model_groups:
            for model_param in group:
                if model_param not in self.model_param_gbuf_map:
                    missing_gbuf_map += 1
                    if len(examples) < 5:
                        examples.append(
                            f"missing_model_param_gbuf_map:{self._find_param_name(model_param) or getattr(model_param, '_param_name', f'id_{id(model_param)}')}"
                        )
                    continue

                gbuf_idx, dtype, bucket_idx = self.model_param_gbuf_map[model_param]
                buffer = self.buffers[gbuf_idx]
                gbuf_range_map = self.gbuf_ranges[gbuf_idx][dtype][bucket_idx]
                if model_param not in gbuf_range_map["param_map"]:
                    missing_param_map += 1
                    if len(examples) < 5:
                        examples.append(
                            f"missing_bucket_param_map:{self._find_param_name(model_param) or getattr(model_param, '_param_name', f'id_{id(model_param)}')}"
                        )
                    continue

                bucket = buffer.buckets[bucket_idx]
                is_dion_param = id(model_param) in getattr(bucket, "dion_param_ids", frozenset())
                is_non_dion_param = model_param in bucket.params and not is_dion_param
                if not (is_dion_param or is_non_dion_param):
                    uncategorized += 1
                    if len(examples) < 5:
                        examples.append(
                            f"uncategorized:{self._find_param_name(model_param) or getattr(model_param, '_param_name', f'id_{id(model_param)}')}"
                        )

        if missing_gbuf_map > 0 or missing_param_map > 0 or uncategorized > 0:
            raise RuntimeError(
                "[Dion] params missing from canonical bucket bookkeeping "
                f"(label={label} missing_model_param_gbuf_map={missing_gbuf_map} "
                f"missing_bucket_param_map={missing_param_map} uncategorized={uncategorized} "
                f"examples={examples})"
            )

    def _rebind_dion_shards(self) -> None:
        """Rebind runtime optimizer param objects back into the canonical Dion shard map."""
        for group in self.optimizer.param_groups:
            for shard_param in group["params"]:
                if not hasattr(shard_param, "_model_param"):
                    continue
                model_param = shard_param._model_param
                old_opt_shard = self._get_opt_shard(model_param)
                if old_opt_shard is None:
                    continue
                if old_opt_shard is not shard_param:
                    if hasattr(old_opt_shard, "_dion_param_uid"):
                        shard_param._dion_param_uid = old_opt_shard._dion_param_uid
                    self._update_opt_shard(model_param, shard_param)

        if hasattr(self, "shard_fp32_from_float16_groups"):
            for model_group, shard_group in zip(
                self.model_float16_groups, self.shard_fp32_from_float16_groups
            ):
                for model_param, shard_param in zip(model_group, shard_group):
                    if shard_param is not None:
                        self._update_opt_shard(model_param, shard_param)

        self._log_unmapped_main_grads(self.model_float16_groups, "model_float16_groups")
        self._log_unmapped_main_grads(self.model_fp32_groups, "model_fp32_groups")

    def _setup_dion_after_init(self) -> None:
        """Run Dion-specific setup after parent DistributedOptimizer init."""
        self._shard_bindings_by_param = {}
        self._dion_buckets_by_param = {}
        self._dion_entries_by_param = {}

        if hasattr(self, 'buffers'):
            self._init_dion_buffers()

        if hasattr(self, 'optimizer') and isinstance(self.optimizer, (MegatronDion,)):
            if hasattr(self, 'gbuf_ranges') and hasattr(self, 'buffers'):
                self._check_dion_params()
                if hasattr(self, 'opt_group_ranges'):
                    (
                        self.model_float16_groups,
                        self.model_fp32_groups,
                        self.shard_float16_groups,
                        self.shard_fp32_groups,
                        self.shard_fp32_from_float16_groups,
                    ) = self._build_param_groups(
                        self.gbuf_ranges,
                        self.model_param_gbuf_map,
                        self.opt_group_ranges,
                        self.config
                    )

                    self._refresh_param_groups()
                    self._rebind_dion_shards()
                    self._enable_overlap_grad_reduce = bool(getattr(self.ddp_config, "overlap_grad_reduce", False))
                    self._enable_overlap_param_gather = bool(getattr(self.ddp_config, "overlap_param_gather", False))
                    try:
                        self._enable_dion_mode()
                    except Exception as error:
                        logger.error(
                            f"[Dion] Global rank {self._global_rank}: Failed in _enable_dion_mode: {error}"
                        )
                        logger.error(traceback.format_exc())
                        raise

        if dist.is_initialized() and hasattr(self, 'data_parallel_group'):
            dist.barrier(group=self.data_parallel_group)

    @classmethod
    def _build_model_gbuf_range(cls, param_and_grad_buffer, bucket_index):
        """
        Build model grad buffer ranges using parent DO standard, then add Dion metadata.

        Architecture:
        - Uses parent DistributedOptimizer's standard bucket structure
        - Adds Dion metadata only for 2D weight matrices
        - Non-Dion params follow standard DO path entirely

        Returns: Parent DO structure + optional typed Dion shard layout per param
        """
        # STEP 1: Call parent to get DO standard bucket structure
        from .distrib_optimizer import DistributedOptimizer
        parent_result = DistributedOptimizer._build_model_gbuf_range(
            param_and_grad_buffer, bucket_index
        )

        # STEP 2: Verify parent DO param_map consistency
        bucket = param_and_grad_buffer.buckets[bucket_index]
        dp_group = param_and_grad_buffer.data_parallel_group
        dp_rank = dp_group.rank()
        dp_world_size = dp_group.size()
        ordered_params = tuple(bucket.params_list)

        param_map = cls._build_bucket_param_map(
            parent_result=parent_result,
            ordered_params=ordered_params,
            dp_group=dp_group,
            dp_rank=dp_rank,
            bucket_index=bucket_index,
            param_index_map=param_and_grad_buffer.param_index_map,
            bucket_offset=bucket.offset,
            bucket_size=bucket.grad_data.numel(),
            bucket_param_to_index=getattr(bucket, "param_to_index", None),
            param_to_name=getattr(param_and_grad_buffer, "param_to_name", None),
        )

        _, fs_size, fs_rank = cls._bucket_shard_topology(param_and_grad_buffer, bucket)
        dion_param_count, dion_static_info_by_param = cls._mark_bucket_dion_params(
            param_map=param_map,
            param_to_name=getattr(param_and_grad_buffer, "param_to_name", None),
            fs_size=fs_size,
        )
        canonical_param_map_snapshot = {}
        for param, range_info in param_map.items():
            canonical_param_map_snapshot[param] = {
                "param": (int(range_info["param"].start), int(range_info["param"].end)),
                "gbuf_local": (
                    int(range_info["gbuf_local"].start),
                    int(range_info["gbuf_local"].end),
                ),
                "gbuf_world": (
                    int(range_info["gbuf_world"].start),
                    int(range_info["gbuf_world"].end),
                ),
                "gbuf_world_in_bucket": (
                    int(range_info["gbuf_world_in_bucket"].start),
                    int(range_info["gbuf_world_in_bucket"].end),
                ),
            }

        # STEP 3: Recalculate buffer ranges for FS × TP hybrid sharding
        # Dion params use FS shard, non-Dion params use DP shard

        (
            dion_bucket_layout,
            dion_shard_layout_by_param,
            dion_param_count,
        ) = (
            build_dion_entries_(
                bucket=bucket,
                param_map=param_map,
                dion_static_info_by_param=dion_static_info_by_param,
                fs_size=fs_size,
                fs_rank=fs_rank,
            )
        )
        for param, snapshot in canonical_param_map_snapshot.items():
            range_info = param_map[param]
            current = {
                "param": (int(range_info["param"].start), int(range_info["param"].end)),
                "gbuf_local": (
                    int(range_info["gbuf_local"].start),
                    int(range_info["gbuf_local"].end),
                ),
                "gbuf_world": (
                    int(range_info["gbuf_world"].start),
                    int(range_info["gbuf_world"].end),
                ),
                "gbuf_world_in_bucket": (
                    int(range_info["gbuf_world_in_bucket"].start),
                    int(range_info["gbuf_world_in_bucket"].end),
                ),
            }
            if current != snapshot:
                param_name = ""
                if hasattr(param_and_grad_buffer, "param_to_name"):
                    param_name = param_and_grad_buffer.param_to_name.get(param, "")
                raise RuntimeError(
                    "[Dion] build_dion_entries_ mutated canonical standard DO param_map "
                    f"for param={param_name or f'id_{id(param)}'} before={snapshot} after={current}"
                )

        for param, range_info in param_map.items():
            range_info["dion_shard_layout"] = dion_shard_layout_by_param.get(param)
        parent_result["local_total"] = 0 if dion_bucket_layout is None else dion_bucket_layout.shard_size

        # Calculate param counts for summary
        total_params = len(param_map)
        non_dion_count = total_params - dion_param_count

        # Add Dion communication layout to the parent result.
        parent_result["dion_bucket_layout"] = dion_bucket_layout
        parent_result["non_dion_count"] = non_dion_count

        # Return hybrid sharding structure (parent ranges + Dion metadata)
        return parent_result

    @classmethod
    def _has_local_dion_shard(cls, range_info: Dict) -> bool:
        """Return whether this rank owns a real Dion local shard for the param."""
        shard_layout = range_info.get("dion_shard_layout", None)
        if shard_layout is None:
            return False

        for dim in shard_layout.local_shape:
            if int(dim) <= 0:
                raise RuntimeError(
                    "[Dion] invalid empty Dion local shard in optimizer ownership map "
                    f"shape={tuple(int(x) for x in shard_layout.local_shape)}"
                )
        return shard_layout.local_numel > 0

    @classmethod
    def _build_model_param_gbuf_map(
        cls, gbuf_ranges: List[Dict]
    ) -> Dict[torch.nn.Parameter, Tuple]:
        """Create the reverse mapping for locally owned optimizer params.

        Dion rebuilds each bucket param map in canonical bucket order and inserts
        zero-length placeholder entries for params that are not locally owned on this
        rank. Those placeholders are useful for stable bucket bookkeeping, but they
        must never participate in optimizer ownership.

        Ownership rule:
        - non-Dion params: standard DO local `param` range
        - Dion params: local FS shard described by canonical `dion_shard_layout`
        """
        param_gbuf_map = {}
        for gbuf_index, gbuf_range_map in enumerate(gbuf_ranges):
            for dtype, gbuf_range_map_for_all_buckets in gbuf_range_map.items():
                for bucket_index, bucket_range_map in enumerate(gbuf_range_map_for_all_buckets):
                    for param, range_info in bucket_range_map["param_map"].items():
                        param_range = range_info.get("param", None)
                        has_standard_local_range = (
                            param_range is not None and param_range.size > 0
                        )
                        has_local_dion_shard = cls._has_local_dion_shard(range_info)
                        if not has_standard_local_range and not has_local_dion_shard:
                            continue
                        assert param not in param_gbuf_map, (
                            "Param should not appear in model_param_gbuf_map more than once; "
                            "only locally owned optimizer params belong in the map."
                        )
                        param_gbuf_map[param] = (gbuf_index, dtype, bucket_index)
        return param_gbuf_map

    @classmethod
    def _build_optimizer_group_ranges(cls, param_groups: List[Dict], gbuf_ranges: List[Dict]):
        """Build optimizer groups from locally owned optimizer params only.

        Canonical zero-length placeholder entries exist only to stabilize bucket param
        ordering. They must not leak into optimizer param groups, otherwise the optimizer
        starts owning params whose local shard is empty and later violates the parent DO
        local-shard write-back contract.

        Ownership rule:
        - non-Dion params: standard DO local `param` range
        - Dion params: local FS shard described by canonical `dion_shard_layout`
        """
        world_param_group_map = {}
        for group_index, group in enumerate(param_groups):
            for param in group["params"]:
                assert param.requires_grad
                world_param_group_map[param] = group_index

        local_param_group_map = {}
        group_ranges = [{"params": []} for _ in param_groups]
        for gbuf_range_map in gbuf_ranges:
            for _, bucket_range_maps in gbuf_range_map.items():
                for bucket_range_map in bucket_range_maps:
                    for param, range_info in bucket_range_map["param_map"].items():
                        param_range = range_info.get("param", None)
                        has_standard_local_range = (
                            param_range is not None and param_range.size > 0
                        )
                        has_local_dion_shard = cls._has_local_dion_shard(range_info)
                        if not has_standard_local_range and not has_local_dion_shard:
                            continue
                        group_index = world_param_group_map[param]
                        group_range = group_ranges[group_index]
                        group_range["params"].append(param)
                        local_param_group_map[param] = (
                            group_index,
                            len(group_range["params"]) - 1,
                        )

        for group_index, group_range in enumerate(group_ranges):
            group_range["orig_group"] = param_groups[group_index]
            group_range["orig_group_idx"] = param_groups[group_index]

        return local_param_group_map, group_ranges

    def __init__(self, *args, **kwargs):
        """Initialize with improved Dion support."""
        # Initialize Dion param info before parent init
        self._shard_layouts_by_param = {}
        self._original_dp_group = kwargs.pop('full_data_parallel_group', None)

        # 2D parallelism configuration
        # RP = Replicate Process (replicas with same shard)
        # FS = Fully Shard (shards within same replica)
        self._rp_size = kwargs.pop('rp_size', None) or kwargs.pop('replica_model_parallel_size', None)
        self._fs_size = kwargs.pop('fs_size', None) or kwargs.pop('fully_shard_model_parallel_size', None)

        if self._rp_size is None:
            self._rp_size = 1

        # Call parent initialization with full DP group (RP × FS)
        # DistributedOptimizer will do uniform sharding across all DP ranks
        # Dion will handle 2D topology (RP × FS) at optimizer state level
        super().__init__(*args, **kwargs)

        # Cache global rank for logging (avoid repeated dist.get_rank() calls)
        self._global_rank = dist.get_rank() if dist.is_initialized() else 0

        self._init_runtime_groups()

        # Unified shard binding mapping.
        self._setup_dion_after_init()

    def _all_gather_bucket_params(self, bucket, async_op=False):
        """Gather all custom bucket params that do not follow the pure standard DO path."""
        handles = []
        dion_layout = getattr(bucket, "dion_layout", None)
        has_dion = dion_layout is not None and dion_layout.has_params
        has_mixed_non_dion = has_dion and bucket.has_non_dion_params

        prepared_dion_entries = None
        if has_mixed_non_dion:
            prepared_dion_entries = self._prepare_dion_bucket_gather(bucket)
            # Mixed non-Dion must restore canonical bucket.param_data before Dion writes
            # its canonical spans. Keep that logical ordering, but follow the stock
            # bucket-group async handle lifecycle when overlap-param-gather is enabled.
            mixed_non_dion_handle = self._all_gather_mixed_non_dion_bucket(
                bucket, async_op=async_op
            )
            if mixed_non_dion_handle is not None:
                handles.append(mixed_non_dion_handle)
            dion_handle = self._all_gather_dion_bucket(
                bucket,
                async_op=async_op,
                prepared_gather=prepared_dion_entries,
            )
            if dion_handle is not None:
                handles.append(dion_handle)
        else:
            dion_handle = self._all_gather_dion_bucket(bucket, async_op=async_op)
            if dion_handle is not None:
                handles.append(dion_handle)
            mixed_non_dion_handle = self._all_gather_mixed_non_dion_bucket(bucket, async_op=async_op)
            if mixed_non_dion_handle is not None:
                handles.append(mixed_non_dion_handle)
        if async_op and handles:
            return _HandleGroup(handles)
        return None

    def _bucket_dion_full_view(self, bucket, entry: DionBucketEntry) -> torch.Tensor:
        """Return the canonical full-param view for one Dion entry."""
        full_view_2d = self._bucket_full_param_view_2d(bucket, entry.param, entry)
        if full_view_2d is None:
            raise RuntimeError(
                "[Dion] canonical FS gather requires bucket.param_data view "
                f"for param={self._canonical_param_name(entry.param) or f'id_{id(entry.param)}'} "
                f"bucket={getattr(bucket, 'bucket_id', -1)}"
            )
        return full_view_2d

    def _fill_dion_shard_buffer(
        self,
        *,
        entry: DionBucketEntry,
        full_view_2d: torch.Tensor,
        shard_buffer: torch.Tensor,
    ) -> None:
        """Pack one Dion local shard into a gather input buffer."""
        canonical_local_source = slice_fs_shard_2d(
            full_view_2d,
            int(entry.fs_split_dim),
            int(entry.start_idx),
            int(entry.end_idx),
        )
        local_source = canonical_local_source
        bound_data_shard = self._get_data_shard(entry.param)
        if bound_data_shard is not None:
            if bound_data_shard.numel() != canonical_local_source.numel():
                raise RuntimeError(
                    "[Dion] canonical FS gather source size mismatch "
                    f"param={self._canonical_param_name(entry.param) or f'id_{id(entry.param)}'} "
                    f"bound={int(bound_data_shard.numel())} canonical={int(canonical_local_source.numel())}"
                )
            if bound_data_shard.data_ptr() != canonical_local_source.data_ptr():
                raise RuntimeError(
                    "[Dion] canonical FS gather source mismatch "
                    f"for param={self._canonical_param_name(entry.param) or f'id_{id(entry.param)}'}: "
                    "registered data_shard no longer aliases bucket.param_data canonical FS slice"
                )
        local_numel = int(entry.local_numel)
        if local_source.numel() != local_numel:
            raise RuntimeError(
                "[Dion] local gather source size mismatch "
                f"param={self._canonical_param_name(entry.param) or f'id_{id(entry.param)}'} "
                f"source={int(local_source.numel())} expected={local_numel}"
            )
        shard_buffer.zero_()
        shard_buffer[:local_numel].copy_(local_source.reshape(-1)[:local_numel])

    def _prepare_dion_bucket_gather(self, bucket) -> _PreparedDionBucketGather:
        """Prepack one bucket-local Dion shard buffer in canonical entry order."""
        dion_layout = getattr(bucket, "dion_layout", None)
        if dion_layout is None or not dion_layout.has_params:
            return _PreparedDionBucketGather(
                prepared_buffer=torch.empty(
                    0,
                    dtype=bucket.param_data.dtype,
                    device=torch.cuda.current_device(),
                ),
                entries=[],
            )

        device = torch.cuda.current_device()
        dtype = bucket.param_data.dtype
        prepared_buffer = torch.empty(
            int(dion_layout.shard_size),
            dtype=dtype,
            device=device,
        )
        prepared_entries = []
        for entry in dion_layout.entries:
            full_view_2d = self._bucket_dion_full_view(bucket, entry)
            shard_start = int(entry.shard_offset)
            shard_end = shard_start + int(entry.shard_capacity)
            shard_buffer = prepared_buffer[shard_start:shard_end]
            self._fill_dion_shard_buffer(
                entry=entry,
                full_view_2d=full_view_2d,
                shard_buffer=shard_buffer,
            )
            prepared_entries.append(
                _PreparedDionGatherEntry(
                    entry=entry,
                    full_view_2d=full_view_2d,
                    shard_buffer=shard_buffer,
                )
            )
        return _PreparedDionBucketGather(
            prepared_buffer=prepared_buffer,
            entries=prepared_entries,
        )

    @staticmethod
    def _serialize_dion_bucket_gather_layout(dion_layout: DionBucketLayout) -> tuple[int, ...]:
        """Serialize only bucket-global packing invariants for cross-rank checks."""
        payload: List[int] = [
            int(dion_layout.entry_count),
            int(dion_layout.shard_size),
            int(dion_layout.gathered_numel),
            int(dion_layout.max_shard_capacity),
        ]
        for entry in dion_layout.entries:
            payload.extend(
                [
                    int(entry.shard_offset),
                    int(entry.shard_capacity),
                    int(entry.canonical_bucket_start),
                    int(entry.canonical_bucket_end),
                    int(entry.fs_split_dim),
                    int(entry.size_per_rank),
                ]
            )
            payload.append(len(entry.canonical_rank_flat_segments))
            for rank_segments in entry.canonical_rank_flat_segments:
                payload.append(len(rank_segments))
                for seg_start, seg_end in rank_segments:
                    payload.extend([int(seg_start), int(seg_end)])
        return tuple(payload)

    def _restore_dion_bucket_from_gathered(
        self,
        *,
        bucket,
        prepared_gather: _PreparedDionBucketGather,
        gathered_buffer: torch.Tensor,
        shard_group_size: int,
    ) -> None:
        """Restore canonical bucket.param_data from one bucket-wise gathered Dion shard buffer."""
        if gathered_buffer.dim() != 2:
            raise RuntimeError(
                f"[Dion] gathered Dion bucket buffer must be 2D, got shape={tuple(gathered_buffer.shape)}"
            )
        if gathered_buffer.size(0) != shard_group_size:
            raise RuntimeError(
                "[Dion] gathered Dion bucket buffer group-size mismatch "
                f"bucket={getattr(bucket, 'bucket_id', -1)} "
                f"buffer_group={int(gathered_buffer.size(0))} expected_group={int(shard_group_size)}"
            )

        for prepared in prepared_gather.entries:
            entry = prepared.entry
            full_view_2d = prepared.full_view_2d
            full_flat = full_view_2d.view(-1)
            shard_start = int(entry.shard_offset)
            shard_end = shard_start + int(entry.shard_capacity)

            for rank_i, rank_segments in enumerate(entry.canonical_rank_flat_segments):
                rank_source = gathered_buffer[rank_i, shard_start:shard_end]
                source_offset = 0
                for seg_start, seg_end in rank_segments:
                    seg_len = int(seg_end) - int(seg_start)
                    if seg_len <= 0:
                        continue
                    local_start = int(seg_start) - int(entry.canonical_bucket_start)
                    local_end = int(seg_end) - int(entry.canonical_bucket_start)
                    full_flat[local_start:local_end].copy_(
                        rank_source[source_offset : source_offset + seg_len]
                    )
                    source_offset += seg_len

            local_target = slice_fs_shard_2d(
                full_view_2d,
                int(entry.fs_split_dim),
                int(entry.start_idx),
                int(entry.end_idx),
            )
            local_numel = int(entry.local_numel)
            if local_numel != local_target.numel():
                raise RuntimeError(
                    "[Dion] local restore shard size mismatch "
                    f"param={self._canonical_param_name(entry.param) or f'id_{id(entry.param)}'} "
                    f"source={local_numel} target={int(local_target.numel())}"
                )
            self._update_data_shard(entry.param, local_target)
            entry.param._fs_shard = local_target

    def _all_gather_dion_bucket(self, bucket, async_op=False, prepared_gather=None):
        """Gather one bucket-local Dion shard buffer back into canonical bucket.param_data."""
        dion_layout = getattr(bucket, "dion_layout", None)
        if dion_layout is None or not dion_layout.has_params:
            return None

        shard_group = getattr(bucket, "dion_shard_group", None)
        if shard_group is None:
            shard_group = self.fs_group
        shard_group_size = self._group_size(shard_group) if shard_group is not None else 1
        if prepared_gather is None:
            prepared_gather = self._prepare_dion_bucket_gather(bucket)

        if shard_group_size == 1:
            gathered_buffer = prepared_gather.prepared_buffer.view(1, -1)
            self._restore_dion_bucket_from_gathered(
                bucket=bucket,
                prepared_gather=prepared_gather,
                gathered_buffer=gathered_buffer,
                shard_group_size=1,
            )
            return None
        if shard_group is None:
            return None

        layout_len = int(dion_layout.entry_count)
        bucket_shard_size = int(dion_layout.shard_size)
        expected_gathered_numel = int(dion_layout.gathered_numel)
        if expected_gathered_numel != bucket_shard_size * shard_group_size:
            raise RuntimeError(
                "[Dion] FS gather shard size invariant mismatch "
                f"bucket={getattr(bucket, 'bucket_id', -1)} "
                f"shard_size={bucket_shard_size} fs_size={shard_group_size} "
                f"expected_total={expected_gathered_numel}"
            )

        if not getattr(bucket, "_dion_ag_invariants_verified", False):
            device = torch.cuda.current_device()
            local = torch.tensor([layout_len, bucket_shard_size], device=device, dtype=torch.int64)
            local_min = local.clone()
            local_max = local.clone()
            dist.all_reduce(local_min, op=dist.ReduceOp.MIN, group=shard_group)
            dist.all_reduce(local_max, op=dist.ReduceOp.MAX, group=shard_group)
            if not torch.equal(local_min, local_max):
                logger.error(
                    "[Dion] FS AG invariants mismatch (bucket_id=%s): local(layout_len=%s, shard_size=%s) "
                    "min=%s max=%s",
                    getattr(bucket, "bucket_id", -1),
                    layout_len,
                    bucket_shard_size,
                    tuple(int(x) for x in local_min.tolist()),
                    tuple(int(x) for x in local_max.tolist()),
                )
                raise RuntimeError("Dion FS all-gather invariants mismatch across FS ranks")

            bucket._dion_ag_invariants_verified = True
            bucket._dion_ag_verified_layout_len = layout_len
            bucket._dion_ag_verified_shard_size = bucket_shard_size
            local_layout_signature = self._serialize_dion_bucket_gather_layout(dion_layout)
            gathered_layout_signatures = [None for _ in range(shard_group_size)]
            dist.all_gather_object(gathered_layout_signatures, local_layout_signature, group=shard_group)
            reference_signature = gathered_layout_signatures[0]
            for rank_i, gathered_signature in enumerate(gathered_layout_signatures[1:], start=1):
                if gathered_signature != reference_signature:
                    raise RuntimeError(
                        "[Dion] FS all-gather packed layout mismatch across shard ranks "
                        f"bucket={getattr(bucket, 'bucket_id', -1)} mismatched_rank={rank_i}"
                    )

        max_shard_capacity = int(dion_layout.max_shard_capacity)
        if max_shard_capacity <= 0:
            raise RuntimeError(
                f"[Dion] invalid FS gather shard capacity for bucket={getattr(bucket, 'bucket_id', -1)}"
            )

        device = torch.cuda.current_device()
        gathered_buffer = torch.empty(
            expected_gathered_numel,
            dtype=bucket.param_data.dtype,
            device=device,
        )
        work = torch.distributed.all_gather_into_tensor(
            output_tensor=gathered_buffer,
            input_tensor=prepared_gather.prepared_buffer,
            group=shard_group,
            async_op=async_op,
        )

        def _restore_callback():
            self._restore_dion_bucket_from_gathered(
                bucket=bucket,
                prepared_gather=prepared_gather,
                gathered_buffer=gathered_buffer.view(shard_group_size, bucket_shard_size),
                shard_group_size=shard_group_size,
            )

        if async_op:
            return _DeferredHandle(work, _restore_callback)

        _restore_callback()
        return None

    def _gather_dion_entry_into_bucket(
        self,
        *,
        entry: DionBucketEntry,
        shard_buffer: torch.Tensor,
        full_view_2d: torch.Tensor,
        shard_group,
        shard_group_size: int,
        async_op: bool,
        dtype: torch.dtype,
        device,
    ):
        """Gather one Dion local shard into the canonical full bucket destination."""
        shard_capacity = int(entry.shard_capacity)
        full_flat = full_view_2d.view(-1)
        direct_flat = (
            entry.fs_split_dim == 0
            and len(entry.canonical_rank_flat_segments) == shard_group_size
            and len(entry.canonical_rank_flat_segments[0]) == 1
            and full_flat.numel() == shard_capacity * shard_group_size
        )
        if direct_flat:
            return torch.distributed.all_gather_into_tensor(
                output_tensor=full_flat,
                input_tensor=shard_buffer,
                group=shard_group,
                async_op=async_op,
            )

        direct_cols = entry.fs_split_dim == 1 and all(
            (int(rank_end) - int(rank_start)) == int(entry.size_per_rank)
            for rank_start, rank_end in entry.rank_split_ranges
        )
        if direct_cols:
            gather_tensor = shard_buffer.view(full_view_2d.shape[0], int(entry.size_per_rank))
            gather_outputs = [
                full_view_2d[:, int(rank_start) : int(rank_end)]
                for rank_start, rank_end in entry.rank_split_ranges
            ]
            return torch.distributed.all_gather(
                tensor_list=gather_outputs,
                tensor=gather_tensor,
                group=shard_group,
                async_op=async_op,
            )

        # Irregular FS layouts cannot all-gather directly into the canonical destination
        # without a padding mismatch. Use one rank-sized broadcast buffer and write each
        # rank's true canonical segments directly into bucket.param_data.
        group_rank = self._group_rank(shard_group)
        broadcast_buffer = None
        for rank_i, rank_segments in enumerate(entry.canonical_rank_flat_segments):
            if rank_i == group_rank:
                rank_source = shard_buffer
            else:
                if broadcast_buffer is None:
                    broadcast_buffer = torch.empty(
                        shard_capacity,
                        dtype=dtype,
                        device=device,
                    )
                rank_source = broadcast_buffer
            torch.distributed.broadcast(
                rank_source,
                src=torch.distributed.get_global_rank(shard_group, rank_i),
                group=shard_group,
                async_op=False,
            )
            source_offset = 0
            for seg_start, seg_end in rank_segments:
                seg_len = int(seg_end) - int(seg_start)
                if seg_len <= 0:
                    continue
                full_flat[
                    int(seg_start) - int(entry.canonical_bucket_start) :
                    int(seg_end) - int(entry.canonical_bucket_start)
                ].copy_(rank_source[source_offset : source_offset + seg_len])
                source_offset += seg_len
        return None

    def _bucket_standard_local_shard_group(self, bucket):
        """Resolve the stock DO local-shard group that owns one bucket."""
        dp_group = getattr(bucket, "intra_distributed_optimizer_instance_group", None)
        if dp_group is not None:
            return (
                dp_group,
                int(getattr(bucket, "intra_distributed_optimizer_instance_size", dp_group.size())),
                int(getattr(bucket, "intra_distributed_optimizer_instance_rank", dp_group.rank())),
            )

        if not hasattr(self, "per_model_bucket_groups"):
            return None, None, None

        for bucket_groups in self.per_model_bucket_groups.values():
            for bucket_group in bucket_groups:
                if bucket not in bucket_group.buckets:
                    continue
                dp_group = getattr(bucket_group, "intra_distributed_optimizer_instance_group", None)
                if dp_group is None:
                    return None, None, None
                return (
                    dp_group,
                    int(bucket_group.intra_distributed_optimizer_instance_size),
                    int(bucket_group.intra_distributed_optimizer_instance_rank),
                )

        return None, None, None

    def _all_gather_mixed_non_dion_bucket(self, bucket, async_op=False):
        """Gather mixed-bucket non-Dion params back into canonical bucket.param_data."""
        dion_layout = getattr(bucket, "dion_layout", None)
        if dion_layout is None or not bucket.has_non_dion_params:
            return None

        dp_group, dp_size, dp_rank = self._bucket_standard_local_shard_group(bucket)
        if dp_group is None:
            raise RuntimeError(
                "[Dion] mixed non-Dion all-gather requires the standard local-shard group."
            )
        if dp_size == 1:
            return None

        shard_size = bucket.param_data.numel() // dp_size
        if shard_size <= 0:
            return None

        shard_buffer = torch.zeros(
            shard_size,
            dtype=bucket.param_data.dtype,
            device=torch.cuda.current_device(),
        )

        for param in select_non_dion_bucket_params_(
            bucket_params=bucket.params_list,
            dion_layout=dion_layout,
        ):
            full_start, full_end = bucket.param_to_index[param]
            bucket_start, bucket_end = bucket_rank_range_(full_start, full_end, shard_size, dp_rank)
            param_start, param_end = param_rank_range_(full_start, full_end, shard_size, dp_rank)
            actual_size = max(0, int(param_end) - int(param_start))
            if actual_size <= 0:
                continue

            full_view = self._bucket_param_view(bucket, param)
            if full_view is None:
                param_name = self._canonical_param_name(param)
                raise RuntimeError(
                    "[Dion] mixed non-Dion all-gather requires canonical bucket.param_data view "
                    f"for param={param_name or f'id_{id(param)}'} "
                    f"bucket={getattr(bucket, 'bucket_id', -1)}"
                )
            param_flat = full_view.view(-1)
            local_shard = param_flat[int(param_start):int(param_end)]
            shard_buffer[int(bucket_start) : int(bucket_start) + actual_size].copy_(local_shard)

        return torch.distributed.all_gather_into_tensor(
            output_tensor=bucket.param_data,
            input_tensor=shard_buffer,
            group=dp_group,
            async_op=async_op,
        )

    def _check_dion_params(self):
        """
        Improved annotation with batch processing for efficiency.
        Uses original model parameters directly (independent of FS sharding).
        """
        # Use FS group for annotation (each FS group operates independently)
        # Parameters are sharded at annotation phase
        # Each FS group will split parameters uniformly across FS ranks
        # RP groups replicate the same FS sharding pattern

        # Get FS and RP info from optimizer
        fs_group = self.optimizer.fs_group if hasattr(self.optimizer, 'fs_group') else None

        if fs_group is not None:
            annotation_group = fs_group  # FS group (size=2)
            fs_rank = dist.get_rank(fs_group)
            fs_size = dist.get_world_size(fs_group)
        else:
            # Standard single-instance DO may not materialize a separate fs_group.
            annotation_group = self.data_parallel_group
            fs_rank = annotation_group.rank()
            fs_size = dist.get_world_size(annotation_group)

        # Validate metadata created by `_build_model_gbuf_range()` and build
        # cheap lookup tables used later in the optimizer.
        total_params = 0
        total_dion_params = 0
        unique_param_ids = set()
        unique_two_d_total = 0
        unique_two_d_dion = 0
        unique_two_d_fp8_skipped = 0
        unique_two_d_manual_disabled = 0
        unique_two_d_not_candidate = 0
        unique_two_d_role_excluded = 0
        unexpected_two_d_leftovers = []

        for gbuf_idx, gbuf_range_maps in enumerate(self.gbuf_ranges):
            buffer = self.buffers[gbuf_idx]

            for dtype in sorted(gbuf_range_maps.keys(), key=lambda dt: str(dt)):
                gbuf_range_map_list = gbuf_range_maps[dtype]

                for bucket_idx, gbuf_range_map in enumerate(gbuf_range_map_list):
                    # Validate per-bucket Dion metadata.
                    for param in gbuf_range_map["param_map"].keys():
                        total_params += 1
                        pri = gbuf_range_map["param_map"][param]

                        shard_layout = pri.get("dion_shard_layout", None)
                        is_dion = shard_layout is not None
                        flag_is_dion = bool(getattr(param, "is_dion_param", False))
                        if flag_is_dion != is_dion:
                            pname = ""
                            try:
                                pname = buffer.param_to_name.get(param, "")
                            except Exception:
                                pname = ""
                            raise RuntimeError(
                                "[Dion] dion flag mismatch between param annotation and range map "
                                f"(name={pname or f'id_{id(param)}'} "
                                f"is_dion_param={flag_is_dion} dion_shard_layout_present={is_dion} "
                                f"gbuf={gbuf_idx} bucket={bucket_idx})"
                            )

                        if not is_dion:
                            continue

                        total_dion_params += 1

                        if shard_layout.local_numel <= 0:
                            pname = ""
                            try:
                                pname = buffer.param_to_name.get(param, "")
                            except Exception:
                                pname = ""
                            raise RuntimeError(
                                "[Dion] invalid Dion local shard layout from _build_model_gbuf_range "
                                f"(name={pname or f'id_{id(param)}'} local_shape={shard_layout.local_shape} "
                                f"gbuf={gbuf_idx} bucket={bucket_idx})"
                            )

                        self._shard_layouts_by_param[param] = shard_layout

                        if id(param) in unique_param_ids:
                            continue

                        unique_param_ids.add(id(param))
                        if param.ndim != 2:
                            continue

                        unique_two_d_total += 1
                        unique_two_d_dion += 1

                    for param in gbuf_range_map["param_map"].keys():
                        if id(param) in unique_param_ids:
                            continue

                        unique_param_ids.add(id(param))
                        if param.ndim != 2:
                            continue

                        unique_two_d_total += 1
                        if getattr(param, "use_dion", None) is False:
                            unique_two_d_manual_disabled += 1
                            continue

                        if not getattr(param, "dion_candidate", False):
                            unique_two_d_not_candidate += 1
                            continue

                        if getattr(param, "is_embedding_or_output_parameter", False) or getattr(
                            param, "is_lm_head_parameter", False
                        ):
                            unique_two_d_role_excluded += 1
                            continue

                        if is_float8tensor(param):
                            unique_two_d_fp8_skipped += 1
                            continue

                        pname = ""
                        try:
                            pname = buffer.param_to_name.get(param, "")
                        except Exception:
                            pname = ""
                        unexpected_two_d_leftovers.append(pname or f"id_{id(param)}")

        expected_dion_two_d = (
            unique_two_d_total
            - unique_two_d_fp8_skipped
            - unique_two_d_manual_disabled
            - unique_two_d_not_candidate
            - unique_two_d_role_excluded
        )
        if unique_two_d_dion != expected_dion_two_d:
            raise RuntimeError(
                "[Dion] 2D Dion classification count mismatch "
                f"(two_d_total={unique_two_d_total} "
                f"dion_two_d={unique_two_d_dion} "
                f"fp8_skipped={unique_two_d_fp8_skipped} "
                f"manual_disabled={unique_two_d_manual_disabled} "
                f"not_candidate={unique_two_d_not_candidate} "
                f"role_excluded={unique_two_d_role_excluded} "
                f"expected_dion_two_d={expected_dion_two_d})"
            )

        if unexpected_two_d_leftovers:
            raise RuntimeError(
                "[Dion] unexpected non-Dion 2D parameters remain after classification: "
                + ", ".join(unexpected_two_d_leftovers[:32])
            )

        if self._global_rank == 0:
            logger.debug(
                "[DION_PARAM_CLASSIFY_SUMMARY] total_params=%s dion_params=%s two_d_total=%s "
                "dion_two_d=%s fp8_two_d_skipped=%s manual_disabled_two_d=%s "
                "not_candidate_two_d=%s role_excluded_two_d=%s unexpected_two_d=%s",
                total_params,
                total_dion_params,
                unique_two_d_total,
                unique_two_d_dion,
                unique_two_d_fp8_skipped,
                unique_two_d_manual_disabled,
                unique_two_d_not_candidate,
                unique_two_d_role_excluded,
                len(unexpected_two_d_leftovers),
            )

    def _build_param_groups(
        self,
        gbuf_ranges: List,
        param_gbuf_map: Dict,
        opt_group_ranges: List,
        config,
    ):
        """
        Build parameter groups with batch processing for 2D views.
        """
        use_precision_aware_optimizer = (
            config.use_precision_aware_optimizer_no_fp8_or_ds_fp8
        )
        # Initialize parameter groups
        model_fp16_groups = []
        model_fp32_groups = []
        shard_float16_groups = []
        shard_fp32_groups = []
        main_shard_groups = []

        # Process each optimizer group
        for group_range in opt_group_ranges:
            # Initialize group lists
            model_fp16_params = []
            model_fp32_params_this_group = []
            shard_float16_params_this_group = []
            shard_fp32_params_this_group = []
            main_shard_params = []

            # Add to main groups
            model_fp16_groups.append(model_fp16_params)
            model_fp32_groups.append(model_fp32_params_this_group)
            shard_float16_groups.append(shard_float16_params_this_group)
            shard_fp32_groups.append(shard_fp32_params_this_group)
            main_shard_groups.append(main_shard_params)

            try:
                self._process_param_group(
                    group_range["params"],
                    gbuf_ranges,
                    param_gbuf_map,
                    config,
                    model_fp16_params,
                    model_fp32_params_this_group,
                    shard_float16_params_this_group,
                    shard_fp32_params_this_group,
                    main_shard_params,
                )
            except Exception as e:
                global_rank = self._global_rank
                logger.error(
                    f"[Dion] Global rank {global_rank}: Failed in _process_param_group "
                    f"for group of {len(group_range['params'])} params: {e}"
                )
                for i, p in enumerate(group_range["params"]):
                    logger.error(f"  Param {i}: shape={p.shape}, ndim={p.ndim}, requires_grad={p.requires_grad}")
                import traceback
                logger.error(traceback.format_exc())
                raise

            # Update optimizer's params
            if not use_precision_aware_optimizer:
                group_range["orig_group"]["params"] = [
                    *shard_fp32_params_this_group,
                    *main_shard_params,
                ]
            else:
                group_range["orig_group"]["params"] = [
                    *shard_fp32_params_this_group,
                    *shard_float16_params_this_group,
                ]

        return (
            model_fp16_groups,
            model_fp32_groups,
            shard_float16_groups,
            shard_fp32_groups,
            main_shard_groups,
        )

    def _process_param_group(self, model_params, gbuf_ranges, param_gbuf_map, config,
                            model_fp16_params, model_fp32_params,
                            shard_float16_params, shard_fp32_params,
                            main_shard_params):
        """Process one optimizer param group into standard/Dion shard lists."""
        for model_param in model_params:
            assert model_param.requires_grad

            gbuf_index, dtype, bucket_index = param_gbuf_map[model_param]
            gbuf_range = gbuf_ranges[gbuf_index][dtype][bucket_index]
            param_range_info = gbuf_range["param_map"][model_param]
            param_range = param_range_info["param"]
            dion_shard_layout = param_range_info.get("dion_shard_layout", None)

            # Handle different parameter types
            if model_param.type() in ['torch.cuda.HalfTensor', 'torch.cuda.BFloat16Tensor']:
                self._process_float16_param(
                    model_param, param_range, dion_shard_layout,
                    config,
                    model_fp16_params, shard_float16_params,
                    main_shard_params
                )
            elif model_param.type() == 'torch.cuda.FloatTensor':
                self._process_float32_param(
                    model_param, param_range, dion_shard_layout,
                    config,
                    model_fp32_params, shard_fp32_params
                )
            else:
                raise TypeError(f'Unsupported parameter type: {model_param.type()}')

    def _create_fs_shard(self, model_param, shard_layout: DionShardLayout):
        """Create the local FS shard view from `model_param`.

        Args:
            model_param: The model parameter tensor (2D, TP-partitioned)
            shard_layout: Typed Dion shard layout

        Returns:
            shard: The local FS shard tensor view into `model_param.data`
        """
        start_idx = int(shard_layout.start_idx)
        end_idx = int(shard_layout.end_idx)
        fs_split_dim = int(shard_layout.fs_split_dim)

        # Keep the FP16 shard on the same storage graph as the forward-visible param.
        # This matches standard Megatron-Core more closely: local shard state is a view
        # into the full param buffer, not a clone-owned shadow tensor.
        shard = slice_fs_shard_2d(model_param.detach(), fs_split_dim, start_idx, end_idx)

        shard._model_param = model_param
        return shard

    def _prepare_fs_shard(self, model_param, shard):
        """Attach FS shard to model_param for optimizer state.

        `param.data` stays as the forward-visible full/TP-local tensor.
        `_fs_shard` is the local FS view into that same storage.

        Args:
            model_param: The model parameter tensor (FS-full, TP-sharded; Megatron-Core standard)
            shard: The FS shard tensor
        """
        shard_layout = self._param_shard_layout(model_param)
        if shard_layout is not None:
            expected_view = slice_fs_shard_2d(
                model_param.data,
                int(shard_layout.fs_split_dim),
                int(shard_layout.start_idx),
                int(shard_layout.end_idx),
            )
            if shard.shape != expected_view.shape or shard.data_ptr() != expected_view.data_ptr():
                param_name = getattr(model_param, '_param_name', f'id_{id(model_param)}')
                logger.error(
                    "[DION_FS_ALIAS_MISMATCH] param=%s shard_shape=%s expected_shape=%s shard_ptr=%s expected_ptr=%s",
                    param_name,
                    tuple(shard.shape),
                    tuple(expected_view.shape),
                    shard.data_ptr(),
                    expected_view.data_ptr(),
                )
        model_param._fs_shard = shard

    # Unified Shard Registration/Query Helpers

    def _register_dion_shard(
        self,
        model_param: torch.nn.Parameter,
        data_shard: torch.Tensor,
        opt_shard: torch.Tensor,
        shard_layout: DionShardLayout,
    ) -> None:
        """Register all shard info for a Dion parameter in one call.

        Args:
            model_param: The model parameter tensor
            data_shard: FP16 shard tensor (all-gather source)
            opt_shard: FP32 shard tensor (optimizer state)
            shard_layout: Typed Dion shard layout
        """
        shard_info = DionShardBinding(
            data_shard=data_shard,
            opt_shard=opt_shard,
        )
        self._shard_bindings_by_param[model_param] = shard_info
        self._shard_layouts_by_param[model_param] = shard_layout

    def _get_data_shard(self, model_param: torch.nn.Parameter) -> Optional[torch.Tensor]:
        """Get data shard (FP16) for a model parameter.

        Args:
            model_param: The model parameter tensor

        Returns:
            Data shard tensor or None if not found
        """
        info = self._shard_bindings_by_param.get(model_param)
        return info.data_shard if info else None

    def _get_opt_shard(self, model_param: torch.nn.Parameter) -> Optional[torch.Tensor]:
        """Get optimizer shard (FP32) for a model parameter.

        Args:
            model_param: The model parameter tensor

        Returns:
            Optimizer shard tensor or None if not found
        """
        info = self._shard_bindings_by_param.get(model_param)
        return info.opt_shard if info else None

    def _update_data_shard(self, model_param: torch.nn.Parameter, new_data_shard: torch.Tensor) -> None:
        """Update data shard for a model parameter.

        This is used when the data_shard tensor is replaced (e.g., during all-gather operations).

        Args:
            model_param: The model parameter tensor
            new_data_shard: The new data shard tensor
        """
        info = self._shard_bindings_by_param.get(model_param)
        if info is not None:
            info.data_shard = new_data_shard

    def _update_opt_shard(self, model_param: torch.nn.Parameter, new_opt_shard: torch.Tensor) -> None:
        """Update optimizer shard for a model parameter.

        This is used when the opt_shard tensor is replaced (e.g., during checkpoint restoration).

        Args:
            model_param: The model parameter tensor
            new_opt_shard: The new optimizer shard tensor
        """
        info = self._shard_bindings_by_param.get(model_param)
        if info is not None:
            info.opt_shard = new_opt_shard

    def _param_shard_layout(self, model_param: torch.nn.Parameter) -> Optional[DionShardLayout]:
        """Return typed Dion shard layout for one model param, if any."""
        return self._shard_layouts_by_param.get(model_param)

    def _check_is_dion(self, model_param, context=""):
        """Verify is_dion_param flag is set correctly.

        Args:
            model_param: The model parameter tensor
            context: Context string for logging (e.g., "FP16", "FP32")

        Raises:
            RuntimeError: If is_dion_param flag is missing or inconsistent
        """
        if not hasattr(model_param, 'is_dion_param'):
            raise RuntimeError(
                f"[Dion] {context} param missing is_dion_param flag! "
                f"Must be set in _build_model_gbuf_range. "
                f"param shape={model_param.shape}"
            )
        elif not model_param.is_dion_param:
            raise RuntimeError(
                f"[Dion] {context} param has is_dion_param=False but entered "
                f"Dion processing path. param shape={model_param.shape}"
            )

    def _process_float16_param(self, model_param, param_range, dion_shard_layout,
                              config,
                              model_fp16_params, shard_float16_params,
                              main_shard_params):
        """Process float16/bfloat16 parameters."""
        param_name = self._canonical_param_name(model_param)
        use_precision_aware_optimizer = (
            config.use_precision_aware_optimizer_no_fp8_or_ds_fp8
        )
        if dion_shard_layout is not None:
            try:
                # Create FS shard using helper
                shard_model_param = self._create_fs_shard(model_param, dion_shard_layout)

                # Prepare FS shard (attach to model_param)
                self._prepare_fs_shard(model_param, shard_model_param)

                # Verify is_dion_param flag
                self._check_is_dion(model_param, "FP16")
            except Exception as e:
                global_rank = self._global_rank
                logger.error(f"[Dion] Global rank {global_rank}: Failed for Dion param")
                logger.error(f"  model_param.shape: {model_param.shape}")
                logger.error(f"  dion_shard_layout: {dion_shard_layout}")
                import traceback
                logger.error(traceback.format_exc())
                raise

            # Copy metadata
            tensor_parallel.copy_tensor_model_parallel_attributes(
                shard_model_param, model_param
            )
            if hasattr(model_param, 'shared'):
                shard_model_param.shared = model_param.shared

            # Create fp32 main param
            if not use_precision_aware_optimizer:
                shard_main_param = shard_model_param.clone().float()
                shard_main_param._model_param = model_param
            else:
                shard_main_param = None

            # Register shard bindings/layout for Dion params.
            opt_shard = shard_main_param if shard_main_param is not None else shard_model_param
            self._register_dion_shard(
                model_param=model_param,
                data_shard=shard_model_param,
                opt_shard=opt_shard,
                shard_layout=dion_shard_layout,
            )
        else:
            # Standard 1D handling (non-Dion params)
            # Non-Dion params are ALWAYS DP-sharded via reduce-scatter.

            if is_float8tensor(model_param) and config.fp8_recipe != "delayed":
                shard_model_param = None
            else:
                # Always use DP shard for non-Dion params
                # param_range contains the DP shard range in the resized bucket layout
                shard_model_param = model_param.detach().view(-1)[
                    param_range.start : param_range.end
                ]

                shard_model_param._model_param = model_param
                tensor_parallel.copy_tensor_model_parallel_attributes(
                    shard_model_param, model_param
                )
                if hasattr(model_param, 'shared'):
                    shard_model_param.shared = model_param.shared

            # Generate main param
            if not use_precision_aware_optimizer:
                if is_float8tensor(model_param):
                    # Handle FP8 tensors - always use DP shard
                    if hasattr(model_param, 'get_high_precision_init_val'):
                        shard_main_param = (
                            model_param.get_high_precision_init_val()
                            .view(-1)[param_range.start : param_range.end]
                            .clone()
                            .to(model_param.device)
                            .float()
                        )
                        model_param.clear_high_precision_init_val()
                    else:
                        shard_main_param = model_param.float().view(-1)[
                            param_range.start : param_range.end
                        ]
                    shard_main_param._model_param = model_param
                else:
                    shard_main_param = shard_model_param.clone().float()
                    shard_main_param._model_param = model_param
            else:
                shard_main_param = None

        # Copy metadata to main param
        if shard_main_param is not None:
            tensor_parallel.copy_tensor_model_parallel_attributes(
                shard_main_param, model_param
            )
            if hasattr(model_param, 'shared'):
                shard_main_param.shared = model_param.shared

        # Store handle
        model_param.main_param = shard_main_param
        model_param.main_param_sharded = True

            # Note: Non-Dion params use standard DO path, not registered in shard bindings

        # Add to groups
        model_fp16_params.append(model_param)
        shard_float16_params.append(shard_model_param)
        main_shard_params.append(shard_main_param)

    def _process_float32_param(self, model_param, param_range, dion_shard_layout,
                              config,
                              model_fp32_params, shard_fp32_params):
        """Process float32 parameters."""
        if dion_shard_layout is not None:
            # Create FS shard using helper
            shard_model_param = self._create_fs_shard(model_param, dion_shard_layout)

            # Prepare FS shard (attach to model_param)
            self._prepare_fs_shard(model_param, shard_model_param)

            # Copy metadata
            tensor_parallel.copy_tensor_model_parallel_attributes(
                shard_model_param, model_param
            )
            if hasattr(model_param, 'shared'):
                shard_model_param.shared = model_param.shared

            # Add forward backlink for bucket-wise communication
            model_param.main_param = shard_model_param
            model_param.main_param_sharded = True

            # Verify is_dion_param flag
            self._check_is_dion(model_param, "FP32")

            # Register shard bindings/layout for Dion params.
            # FP32 params: data_shard == opt_shard == shard_model_param
            self._register_dion_shard(
                model_param=model_param,
                data_shard=shard_model_param,
                opt_shard=shard_model_param,
                shard_layout=dion_shard_layout,
            )
        else:
            # Standard 1D handling
            shard_model_param = model_param.view(-1)[param_range.start : param_range.end]
            shard_model_param._model_param = model_param

            tensor_parallel.copy_tensor_model_parallel_attributes(
                shard_model_param, model_param
            )
            if hasattr(model_param, 'shared'):
                shard_model_param.shared = model_param.shared

            # Note: Non-Dion params use standard DO path, not registered in shard bindings

        model_fp32_params.append(model_param)
        shard_fp32_params.append(shard_model_param)

    def _enable_dion_mode(self):
        """Enable distributed mode with improved batch processing."""
        global_rank = self._global_rank

        if not isinstance(self.optimizer, (MegatronDion,)):
            return

        from ..parallel_state import (
            get_tensor_model_parallel_group,
            get_tensor_model_parallel_world_size,
        )

        # PP is orthogonal to Dion math/communication. A PP-only run still needs
        # distributed Dion metadata so stage-local optimizer shards keep the same
        # logical parameter identity and 2D Dion state selection as non-PP runs.
        # Only a true single-rank job can skip distributed Dion setup entirely.
        if dist.is_initialized() and dist.get_world_size() == 1:
            return

        # Collect buffer sizes
        gbuf_sizes = [
            [(bucket.grad_data.numel(), bucket.offset) for bucket in buffer.buckets]
            for buffer in self.buffers
        ]

        # Create replica groups with improved batching (will be no-op if groups already set)
        self._validate_prebuilt_topology_groups()

        # Create true RP and FS groups for 2D parallelism (will be no-op if groups already set)
        self._init_topology_cache()

        # Create dist_metas with batch processing
        try:
            dist_metas_sharded = self._build_dist_metas()
        except Exception as e:
            logger.error(f"[Dion] Global rank {global_rank}: Failed in _build_dist_metas: {e}")
            logger.error(traceback.format_exc())
            raise

        # Enable distributed mode with 2D parallelism support
        # Use original DP group for full_data_parallel_group (not fs_group used for sharding)
        full_data_parallel_group = (
            self._original_dp_group
            if hasattr(self, '_original_dp_group') and self._original_dp_group
            else self.data_parallel_group
        )

        enable_args = {
            'global_buffer_sizes': gbuf_sizes,
            'full_data_parallel_group': full_data_parallel_group,  # Original full DP = RP × FS
            'tp_group': get_tensor_model_parallel_group(),
            'dist_metas': dist_metas_sharded,
            'rp_group': self.rp_group,
            'fs_group': self.fs_group,
            'state_replica_group': self.state_replica_group,
        }


        self.optimizer.enable_distributed_mode(**enable_args)

        # Register mixed bucket adapters (buffer_indices populated in enable_distributed_mode)
        if self._enable_overlap_param_gather or self._enable_overlap_grad_reduce:
            overlap_features = []
            if self._enable_overlap_grad_reduce:
                overlap_features.append("gradient RS")
            if self._enable_overlap_param_gather:
                overlap_features.append("parameter AG")
            overlap_msg = " + ".join(overlap_features)

        expected_rp_size = int(self._rp_size) if self._rp_size is not None else 1
        if expected_rp_size > 1:
            # Verify all ranks have rp_group before RP collectives. For RP=1, the
            # standard contract is that no rp_group exists and no RP collectives run.
            have_rp = torch.tensor(
                [1 if self.rp_group is not None else 0],
                device=torch.cuda.current_device(), dtype=torch.int64
            )
            dist.all_reduce(have_rp, op=dist.ReduceOp.MIN, group=self.data_parallel_group)
            all_have_rp = int(have_rp.item()) == 1
            if not all_have_rp:
                raise RuntimeError(
                    f"[Dion] Global rank {global_rank}: not all DP ranks have rp_group "
                    f"(MIN={have_rp.item()}); RP topology must be identical across the data-parallel group"
                )

            # All ranks should have rp_group when all_have_rp == True
            # Double-check to catch bugs
            if self.rp_group is None:
                raise RuntimeError(
                    f"Global rank {global_rank}: all_have_rp=True but rp_group is None! "
                    f"This indicates a bug in group creation or collective voting logic."
                )

            # Verify Dion eligibility consistency within RP group
            # All ranks in same RP group must have identical Dion param counts
            # Now ALL ranks will execute this block (no partial participation)
            my_dion_count = sum(1 for meta in dist_metas_sharded.values() if meta.is_dion_param)
            my_cnt_tensor = torch.tensor([my_dion_count], device=torch.cuda.current_device(), dtype=torch.int64)

            rp_world_size = dist.get_world_size(self.rp_group)
            gathered = [torch.zeros_like(my_cnt_tensor) for _ in range(rp_world_size)]
            dist.all_gather(gathered, my_cnt_tensor, group=self.rp_group)

            gathered_counts = [int(t.item()) for t in gathered]
            if not all(count == my_dion_count for count in gathered_counts):
                # Collect Dion param offsets for error diagnosis
                my_dion_offsets = sorted([
                    meta.param_uid for meta in dist_metas_sharded.values() if meta.is_dion_param
                ])
                gathered_offsets = [None] * rp_world_size
                dist.all_gather_object(gathered_offsets, my_dion_offsets, group=self.rp_group)

                # Find symmetric difference
                for rp_rank, offsets in enumerate(gathered_offsets):
                    if offsets != my_dion_offsets:
                        my_set = set(my_dion_offsets)
                        other_set = set(offsets)
                        only_mine = my_set - other_set
                        only_other = other_set - my_set
                        logger.error(f"[Dion] RP rank {rp_rank} differs from me: "
                                   f"Only in mine: {sorted(only_mine)[:5]}, "
                                   f"Only in theirs: {sorted(only_other)[:5]}")

                raise RuntimeError(
                    f"CRITICAL: Dion eligibility mismatch within RP group! "
                    f"My Dion count: {my_dion_count}, RP group counts: {gathered_counts}. "
                    f"This will cause collective operation hangs. "
                    f"DistributedOptimizer did uniform sharding across DP (RP×FS), "
                    f"so RP group members have different param chunks. "
                    f"Consider disabling DistributedOptimizer sharding or implementing custom FS-aware sharding."
                )

    def _validate_prebuilt_topology_groups(self):
        """
        DISABLED: No-op - RP/FS groups already created in __init__.py

        This function previously created additional subgroups based on owner_tuples,
        but those extra groups are not part of the current runtime contract.
        Creating unnecessary groups causes new_group() call count mismatch across ranks,
        leading to NCCL hang.

        Solution: Skip all new_group() calls. Use collective voting to ensure all ranks
        take the same path (critical for PyTorch new_group() synchronization).
        """
        global_rank = self._global_rank
        dp_group = self.data_parallel_group
        dp_world_size = dist.get_world_size(dp_group)

        # RP=1 is valid and represented as rp_group=None. What must be uniform is
        # the availability of fs_group, while rp_group may be absent on all ranks.
        have_fs = torch.tensor(
            [1 if hasattr(self, 'fs_group') and self.fs_group is not None else 0],
            device=torch.cuda.current_device(), dtype=torch.int64
        )
        dist.all_reduce(have_fs, op=dist.ReduceOp.MIN, group=dp_group)
        all_have_fs = int(have_fs.item()) == 1

        if all_have_fs:
            # All ranks have the required FS group. RP may legitimately be None
            # on every rank when the standard runtime has a single optimizer instance.
            self.owner_tuple_to_subgroup = {}
            dist.barrier(group=dp_group)
            return
        elif hasattr(self, 'fs_group') and self.fs_group is not None:
            # Only some ranks have fs_group - FATAL ERROR
            raise RuntimeError(
                f"Global rank {global_rank}: FS groups exist only on subset of ranks! "
                f"Ensure rp_group/fs_group are provided uniformly to prevent new_group() mismatch."
            )

        # If we reach here, no ranks have fs_group - this is a real topology bug.
        raise RuntimeError(
            f"Global rank {global_rank}: No FS group found! "
            f"FS group must be provided from the standard Megatron-Core optimizer topology."
        )

    def _init_topology_cache(self):
        """
        DISABLED: No-op - RP/FS groups already created in __init__.py

        This function previously created RP/FS groups, but they are already created
        in __init__.py with deterministic world-level synchronization.
        Attempting to create groups again here causes new_group() call count mismatch
        across ranks (especially with TP>1), leading to NCCL hang.

        Solution: Skip all new_group() calls. Use collective voting to ensure all ranks
        take the same path (critical for PyTorch new_group() synchronization).
        """
        global_rank = self._global_rank
        dp_group = self.data_parallel_group
        dp_world_size = dist.get_world_size(dp_group)


        # FS group is required. RP group may legitimately be absent on every rank
        # when the standard runtime has a single optimizer instance.
        have_fs = torch.tensor(
            [1 if hasattr(self, 'fs_group') and self.fs_group is not None else 0],
            device=torch.cuda.current_device(), dtype=torch.int64
        )
        dist.all_reduce(have_fs, op=dist.ReduceOp.MIN, group=dp_group)
        all_have_fs = int(have_fs.item()) == 1

        if all_have_fs:
            # All ranks have FS group. RP may be None on all ranks for RP=1.
            # Barrier to ensure all ranks finish together
            dist.barrier(group=dp_group)
            return

        elif hasattr(self, 'fs_group') and self.fs_group is not None:
            # Only some ranks have fs_group - FATAL ERROR
            raise RuntimeError(
                f"Global rank {global_rank}: FS groups exist only on subset of ranks! "
                f"Ensure rp_group/fs_group are provided uniformly to prevent new_group() mismatch."
            )

        # If we reach here, no ranks have fs_group - should never happen.
        raise RuntimeError(
            f"Global rank {global_rank}: No FS group found! "
            f"FS group must be provided from the standard Megatron-Core optimizer topology."
        )

    @staticmethod
    def _group_info(group) -> Tuple[int, int]:
        """Return (world_size, rank) for a process group, or (1, -1) if absent."""
        if group is None:
            return 1, -1
        return dist.get_world_size(group), dist.get_rank(group)

    @staticmethod
    def _group_ranks(group):
        if group is None:
            return None
        return tuple(dist.get_process_group_ranks(group))

    def _expected_expert_shard_group(self):
        group = parallel_state.get_expert_data_parallel_group(
            check_initialized=False,
            partial_expert_data_parallel=True,
        )
        if group is None:
            raise RuntimeError(
                "[Dion][EP] missing expert-local shard group "
                "(expected intra_expt_dp_group for expert param/bucket)."
            )
        return group

    def _assert_group_matches(self, *, label: str, actual_group, expected_group, extra: str = "") -> None:
        actual_ranks = self._group_ranks(actual_group)
        expected_ranks = self._group_ranks(expected_group)
        if actual_ranks != expected_ranks:
            raise RuntimeError(
                f"[Dion][EP] {label} group mismatch: "
                f"actual={actual_ranks} expected={expected_ranks}. {extra}".strip()
            )

    def _select_shard_group(self, model_param):
        """Return the FS shard group for a model param (dense vs expert-parallel)."""
        if not getattr(model_param, 'allreduce', True):
            expert_group = self._expected_expert_shard_group()
            self._assert_group_matches(
                label="expert param shard_group",
                actual_group=self.fs_group,
                expected_group=expert_group,
                extra=(
                    f"param={getattr(model_param, '_param_name', '') or id(model_param)} "
                    "Dion EP must keep expert params on the standard expert-local DO shard group."
                ),
            )
            return expert_group
        return self.fs_group

    @staticmethod
    def _make_param_uid(
        *,
        param_name: str,
        logical_global_shape=None,
        is_dion_param: bool,
    ):
        """Build a topology-independent logical param identity for optimizer state.

        The same logical parameter must keep the same state identity across
        different FS/TP/CP shard layouts. Runtime shard ranges are not part of the
        logical identity.
        """
        if not param_name:
            raise RuntimeError(
                "[Dion] Missing param_name while building logical param_uid"
            )
        return (
            str(param_name or ""),
            tuple(int(dim) for dim in logical_global_shape) if logical_global_shape is not None else (),
            bool(is_dion_param),
        )

    def _build_dist_meta(
        self,
        *,
        model_param,
        shard_param,
    ) -> MegatronDionDistMeta:
        """Build one MegatronDionDistMeta from cached shard info."""
        if is_tp_enabled(model_param):
            tp_split_dim = get_tp_split_dim(model_param)
        else:
            tp_split_dim = -1

        param_name = self._logical_param_name(model_param) or ""
        shard_group = self._select_shard_group(model_param)
        shard_group_world_size, shard_group_rank = self._group_info(shard_group)
        dion_shard_layout = self._shard_layouts_by_param.get(model_param)
        if dion_shard_layout is None:
            raise RuntimeError(
                "[Dion] missing dion_shard_layout while building distributed metadata "
                f"for param={param_name or id(model_param)}"
            )

        dist_meta = MegatronDionDistMeta(
            shape=shard_param.shape,
            global_shape=dion_shard_layout.global_shape,
            tp_split_dim=tp_split_dim,
            fs_split_dim=dion_shard_layout.fs_split_dim,
            rank_fraction=self.optimizer.defaults.get('rank_fraction', 0.25),
            is_dion_param=getattr(model_param, 'is_dion_param', True),
            is_transposed=bool(shard_param.shape[0] < shard_param.shape[1]),
            param_uid=self._make_param_uid(
                param_name=param_name,
                logical_global_shape=(
                    dion_shard_layout.per_expert_global_shape
                    if dion_shard_layout.per_expert_global_shape is not None
                    else dion_shard_layout.global_shape
                ),
                is_dion_param=True,
            ),
            param_name=param_name,
            shard_group=shard_group,
            shard_group_world_size=shard_group_world_size,
            shard_group_rank=shard_group_rank,
            per_expert_global_shape=dion_shard_layout.per_expert_global_shape,
        )

        return dist_meta

    def _build_dist_metas(self):
        """Create dist_metas with batch processing."""
        dist_metas_sharded = {}

        # Batch process Dion shard bindings.
        for model_param, shard_info in self._shard_bindings_by_param.items():
            shard_param = shard_info.opt_shard  # Key for dist_metas_sharded
            dist_meta = self._build_dist_meta(
                model_param=model_param,
                shard_param=shard_param,
            )
            dist_metas_sharded[shard_param] = dist_meta
            # Store param_uid directly on param for lookup after offload/reload
            # This allows _get_or_initialize_state to find UID even when dist_metas lookup fails
            shard_param._dion_param_uid = dist_meta.param_uid

        # Add non-Dion parameters
        self._add_non_dion_metas(dist_metas_sharded)

        uid_to_entries = {}
        for shard_param, meta in dist_metas_sharded.items():
            uid_to_entries.setdefault(meta.param_uid, []).append(
                {
                    "name": getattr(meta, "param_name", "") or self._canonical_param_name(shard_param),
                    "shape": tuple(meta.shape) if meta.shape is not None else tuple(shard_param.shape),
                    "is_dion": bool(getattr(meta, "is_dion_param", False)),
                }
            )
        duplicate_uids = {uid: entries for uid, entries in uid_to_entries.items() if len(entries) > 1}
        if duplicate_uids:
            for uid, entries in sorted(duplicate_uids.items(), key=lambda item: str(item[0])):
                logger.error("[Dion] Duplicate param_uid=%s entries=%s", uid, entries)
            raise RuntimeError(
                "[Dion] Duplicate param_uid detected in dist_metas; "
                "optimizer state identity would be ambiguous"
            )

        return dist_metas_sharded

    def _add_non_dion_metas(self, dist_metas_sharded):
        """Add dist_metas for non-Dion parameters."""
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p not in dist_metas_sharded:
                    model_param = getattr(p, '_model_param', None)
                    if model_param is None:
                        raise RuntimeError(
                            "[Dion] non-Dion optimizer shard is missing _model_param backlink; "
                            f"shape={tuple(p.shape)} id={id(p)}"
                        )

                    # Create basic dist_meta
                    param_name = (
                        self._logical_param_name(model_param)
                        or self._logical_param_name(p)
                        or self._canonical_param_name(model_param)
                        or self._canonical_param_name(p)
                        or ""
                    )
                    param_uid = self._make_param_uid(
                        param_name=param_name,
                        logical_global_shape=None,
                        is_dion_param=getattr(model_param, 'is_dion_param', False),
                    )
                    dist_metas_sharded[p] = MegatronDionDistMeta(
                        shape=p.shape,
                        global_shape=tuple(model_param.shape) if model_param.ndim == 2 else None,
                        tp_split_dim=-1,
                        rank_fraction=self.optimizer.defaults.get('rank_fraction', 0.25),
                        is_dion_param=getattr(model_param, 'is_dion_param', False),
                        is_transposed=bool(p.ndim == 2 and p.shape[0] < p.shape[1]),
                        param_uid=param_uid,
                        param_name=param_name,
                    )
                    # Store param_uid directly on param for lookup after offload/reload
                    p._dion_param_uid = param_uid

    def _build_grad_copy_context(self):
        """Build cached logging and group-pair context for grad wiring."""
        use_precision_aware_optimizer = bool(
            getattr(self.config, "use_precision_aware_optimizer_no_fp8_or_ds_fp8", False)
        )
        if use_precision_aware_optimizer:
            grad_group_pairs = (
                (self.model_float16_groups, self.shard_float16_groups),
                (self.model_fp32_groups, self.shard_fp32_groups),
            )
        else:
            grad_group_pairs = (
                (self.model_float16_groups, self.shard_fp32_from_float16_groups),
                (self.model_fp32_groups, self.shard_fp32_groups),
            )

        return {
            "use_precision_aware_optimizer": use_precision_aware_optimizer,
            "grad_group_pairs": grad_group_pairs,
            "log_grad_issue": self._log_grad_issue,
        }

    def _log_grad_issue(
        self,
        kind: str,
        model_param: torch.nn.Parameter,
        shard_param: Optional[torch.nn.Parameter] = None,
        **extra,
    ) -> None:
        """Emit one structured grad-binding error log."""
        payload = {
            "kind": kind,
            "param": self._find_param_name(model_param)
            or getattr(model_param, "_param_name", f"id_{id(model_param)}"),
            "model_shape": tuple(model_param.shape),
            "shard_shape": tuple(shard_param.shape) if shard_param is not None else None,
        }
        payload.update(extra)
        logger.error("[DION_GRAD_ISSUE] %s", payload)

    def _build_param_copy_context(self):
        """Build shared setup used by model<->main param copy paths."""
        return {
            "use_precision_aware_optimizer": bool(
                getattr(self.config, "use_precision_aware_optimizer_no_fp8_or_ds_fp8", False)
            ),
            "main_shard_groups": getattr(self, "shard_fp32_from_float16_groups", None),
        }

    def _rebind_model_params_to_canonical_bucket_storage(self, *, dion_only: bool = False) -> None:
        """Make selected model params alias canonical bucket.param_data before step writeback."""
        if not hasattr(self, "buffers"):
            return
        for buffer in self.buffers:
            for bucket in buffer.buckets:
                if getattr(bucket, "param_data", None) is None:
                    continue
                dion_layout = getattr(bucket, "dion_layout", None)
                if dion_only:
                    if dion_layout is None or not dion_layout.has_params:
                        continue
                    dion_params = []
                    seen_param_ids = set()
                    for entry in dion_layout.entries:
                        if entry.param is None or id(entry.param) in seen_param_ids:
                            continue
                        seen_param_ids.add(id(entry.param))
                        dion_params.append(entry.param)
                    if not dion_params:
                        continue
                    self._bind_bucket_param_views(bucket, copy_data=True, params=dion_params)
                else:
                    self._bind_bucket_param_views(bucket, copy_data=True)
                    if dion_layout is None or not dion_layout.has_params:
                        continue

                if dion_layout is None or not dion_layout.has_params:
                    continue
                for entry in dion_layout.entries:
                    full_view_2d = self._bucket_dion_full_view(bucket, entry)
                    canonical_local = slice_fs_shard_2d(
                        full_view_2d,
                        int(entry.fs_split_dim),
                        int(entry.start_idx),
                        int(entry.end_idx),
                    )
                    self._update_data_shard(entry.param, canonical_local)
                    entry.param._fs_shard = canonical_local
                self._check_bucket_param_views(
                    bucket,
                    context="copy_main_params_to_model_params",
                    params=dion_params if dion_only else None,
                )

    def _bucket_param_data(self, model_param: torch.nn.Parameter):
        """Return the canonical bucket.param_data buffer for a model param."""
        gbuf_index, _, bucket_index = self.model_param_gbuf_map[model_param]
        return self.buffers[gbuf_index].buckets[bucket_index].param_data

    def _check_main_shards(self, main_shard_groups) -> None:
        """Run the one-time shard identity check against optimizer param groups."""
        if not hasattr(self, '_identity_check_done'):
            self._identity_check_done = False

        if not self._identity_check_done and hasattr(self, 'optimizer'):
            self._identity_check_done = True
            check_shard_identity_(
                optimizer=self.optimizer,
                model_float16_groups=self.model_float16_groups,
                main_shard_groups=main_shard_groups,
            )

    def _copy_model_grads_to_main_grads(self):
        """Map canonical model-side grads onto optimizer-side shard grads.

        When RP=1 with DO overlap:
        - DO already finished reduced-grad production into canonical model-side buffers.
        - FS group = the current distributed-optimizer shard group.
        - This method should only bridge those canonical reduced grads onto optimizer
          shard params.

        Contract:
        - model side canonical grad: `model_param.main_grad`
        - optimizer side canonical grad: `shard_param.grad` or `shard_param.decoupled_grad`

        Dion parameters still need a custom bridge because the optimizer shard objects are
        separate FP32 tensors and Dion reads custom FS/TP-local grad slices. Non-Dion should
        match standard Megatron-Core DO as closely as possible.
        """
        # Match parent's logging pattern (distrib_optimizer.py:2392)

        if self.is_stub_optimizer:
            return
        if self.ddp_config.use_megatron_fsdp:
            return

        # Copy non-Dion gradients through the standard DO local-shard contract first.
        # Dion params use the custom logical local-shard bridge below.

        grad_copy_ctx = self._build_grad_copy_context()
        use_precision_aware_optimizer = grad_copy_ctx["use_precision_aware_optimizer"]
        log_grad_issue = grad_copy_ctx["log_grad_issue"]
        grad_group_pairs = grad_copy_ctx["grad_group_pairs"]

        for model_groups, shard_groups in grad_group_pairs:
            copy_non_dion_grads_(
                model_groups=model_groups,
                shard_groups=shard_groups,
                get_param_range_fn=self._get_model_param_range_map,
                log_grad_issue_fn=log_grad_issue,
                use_precision_aware_optimizer=use_precision_aware_optimizer,
            )
            copy_dion_grads_(
                model_groups=model_groups,
                shard_groups=shard_groups,
                get_dion_shard_layout_fn=self._param_shard_layout,
                log_grad_issue_fn=log_grad_issue,
                use_precision_aware_optimizer=use_precision_aware_optimizer,
            )

        self._release_rs_buffers()

    def _materialize_pending_dion_local_grads(self) -> None:
        """Compatibility fallback: materialize any unexpected pending Dion grads idempotently."""
        if not hasattr(self, 'per_model_bucket_groups'):
            return
        for bucket_groups in self.per_model_bucket_groups.values():
            for bucket_group in bucket_groups:
                materialize = getattr(bucket_group, "_materialize_dion_local_grads", None)
                if materialize is not None:
                    materialize()

    def get_main_grads_for_grad_norm(self) -> List[torch.Tensor]:
        """Return stock grad-norm inputs without Dion-specific side paths."""
        return super().get_main_grads_for_grad_norm()

    @torch.no_grad()
    def get_grad_norm(self):
        """Compute grad norm with exact FP32 accumulation over the current grad surface.

        Dion local shards are frequently represented as many aliased/view-backed tensors.
        The stock multi-tensor L2 path can drift on this surface. Use an explicit FP32 sum
        of squared norms so the returned total matches the canonical shard set exactly.
        """
        grads_for_norm = self.get_main_grads_for_grad_norm()
        total_sq = torch.zeros(1, dtype=torch.float64, device=torch.cuda.current_device())
        data_parallel_group = None
        for grad in grads_for_norm:
            data_parallel_group = get_data_parallel_group_if_dtensor(grad, data_parallel_group)
            local_grad = to_local_if_dtensor(grad).detach()
            total_sq += local_grad.float().pow(2).sum().to(dtype=torch.float64)

        if data_parallel_group is not None:
            torch.distributed.all_reduce(
                total_sq, op=torch.distributed.ReduceOp.SUM, group=data_parallel_group
            )
        torch.distributed.all_reduce(
            total_sq, op=torch.distributed.ReduceOp.SUM, group=self.get_grad_stats_parallel_group()
        )
        return float(total_sq.sqrt().item())

    def requires_individual_grad_norm_in_chain(self) -> bool:
        """Shared-group chained grad norm must respect Dion's exact local accumulation."""
        return True

    def prepare_grads(self) -> bool:
        """
        Match standard Megatron-Core prepare_grads semantics.

        The standard training loop already calls `finalize_model_grads()`, which in turn
        calls `model_chunk.finish_grad_sync()` before `optimizer.step()`. Repeating
        `finish_grad_sync()` here is incorrect for the multi-instance path because mixed
        non-Dion RS-local buffers have already been flushed and cleared by the first call.
        """
        return super().prepare_grads()

    def _copy_model_params_to_main_params(self, state_dict=None):
        """Copy model params to main params with FS sharding awareness.

        Handles FS-sharded parameters correctly when loading from checkpoint.
        """
        from .cpu_offloading import HybridDeviceOptimizer
        copy_ctx = self._build_param_copy_context()
        use_precision_aware_optimizer = copy_ctx["use_precision_aware_optimizer"]
        main_shard_groups = copy_ctx["main_shard_groups"]
        if isinstance(self.optimizer, HybridDeviceOptimizer):
            self.optimizer.update_fp32_param_by_new_param()
            return

        if self.ddp_config.use_megatron_fsdp:
            return

        # Precision-aware optimizer early return
        if use_precision_aware_optimizer:
            return

        if state_dict is not None:
            # Build mapping for state dict params
            model_param_to_state_dict_param_map = self._build_model_param_to_state_dict_param_map(
                state_dict
            )

        if state_dict is not None:
            get_source_param = lambda model_param: model_param_to_state_dict_param_map[model_param]
        else:
            get_source_param = lambda model_param: model_param
        get_param_range = lambda model_param: self._get_model_param_range_map(model_param)["param"]

        copy_group_params_to_main_shards_(
            model_groups=self.model_float16_groups,
            shard_main_groups=main_shard_groups,
            get_source_param_fn=get_source_param,
            get_param_range_fn=get_param_range,
            get_dion_shard_layout_fn=self._param_shard_layout,
        )
        copy_group_params_to_main_shards_(
            model_groups=self.model_fp32_groups,
            shard_main_groups=self.shard_fp32_groups,
            get_source_param_fn=get_source_param,
            get_param_range_fn=get_param_range,
            get_dion_shard_layout_fn=self._param_shard_layout,
        )

    def _copy_main_params_to_model_params(self):
        """Copy parameters with efficient batch flattening for 2D params.

        Copy updated optimizer shards into local model-param shards.

        Stock DO treats bucket.param_data as the canonical post-step restore / gather
        surface. Rebind forward-visible model params to that bucket storage before
        writing local shards so the subsequent param-gather path never depends on
        stale lingering aliases.
        """
        if self.is_stub_optimizer:
            return
        copy_ctx = self._build_param_copy_context()
        use_precision_aware_optimizer = copy_ctx["use_precision_aware_optimizer"]
        main_shard_groups = copy_ctx["main_shard_groups"]

        # FSDP early return
        if self.ddp_config.use_megatron_fsdp:
            for model_chunk in self.model_chunks:
                model_chunk.param_and_grad_buffer.copy_main_weights_to_model_weights()
            return

        # Precision-aware optimizer early return
        if use_precision_aware_optimizer:
            return

        self._mark_buckets_full_param_ready(False)

        # Copy updated optimizer shards (FP32) to model params (BF16)
        # Use model_float16_groups/shard_fp32_from_float16_groups directly
        # to avoid object identity mismatch in shard lookups

        zero_range_warned = getattr(self, '_zero_range_warned', 0)

        self._check_main_shards(main_shard_groups)
        apply_non_dion_shards_(
            model_groups=self.model_float16_groups,
            shard_groups=main_shard_groups,
            get_param_range_map_fn=self._get_model_param_range_map,
            get_bucket_param_data_fn=self._bucket_param_data,
        )
        _, zero_range_warned = apply_group_shards_to_model_params_(
            model_groups=self.model_float16_groups,
            shard_groups=main_shard_groups,
            shard16_groups=self.shard_float16_groups,
            get_data_shard_fn=self._get_data_shard,
            get_param_range_map_fn=self._get_model_param_range_map,
            get_dion_shard_layout_fn=self._param_shard_layout,
            get_bucket_param_data_fn=self._bucket_param_data,
            zero_range_warned=zero_range_warned,
        )

        self._zero_range_warned = zero_range_warned

        # Keep model_fp32_groups current as local shards as well when deferring
        # the expensive full-param restore path.
        apply_non_dion_shards_(
            model_groups=self.model_fp32_groups,
            shard_groups=self.shard_fp32_groups,
            get_param_range_map_fn=self._get_model_param_range_map,
            get_bucket_param_data_fn=self._bucket_param_data,
        )

        self._rebind_model_params_to_canonical_bucket_storage(dion_only=True)

        _, zero_range_warned = apply_group_shards_to_model_params_(
            model_groups=self.model_fp32_groups,
            shard_groups=self.shard_fp32_groups,
            shard16_groups=self.shard_fp32_groups,
            get_data_shard_fn=self._get_data_shard,
            get_param_range_map_fn=self._get_model_param_range_map,
            get_dion_shard_layout_fn=self._param_shard_layout,
            get_bucket_param_data_fn=self._bucket_param_data,
            zero_range_warned=zero_range_warned,
        )
        self._zero_range_warned = zero_range_warned

        return



    def _refresh_param_groups(self):
        """Update optimizer param groups after rebuilding."""
        from .cpu_offloading import HybridDeviceOptimizer

        if isinstance(self.optimizer, HybridDeviceOptimizer):
            self.optimizer = HybridDeviceOptimizer(
                params=[g["orig_group"] for g in self.opt_group_ranges],
                **self.optimizer.defaults
            )
        else:
            self.optimizer.param_groups = [g["orig_group"] for g in self.opt_group_ranges]
            self.optimizer.load_state_dict(self.optimizer.state_dict())

    # DO Overlap Integration

    def finish_grad_sync(self):
        """
        Wait for async gradient reduce-scatter (DO bucket groups + FS adapters).

        This is called before optimizer.step() to ensure all gradients
        are properly reduced and sharded.
        """
        # 1) Wait DO bucket groups (if exists)
        if not hasattr(self, 'per_model_bucket_groups'):
            return  # No bucket groups

        finish_bucket_group_grad_sync_(self.per_model_bucket_groups)

    def _release_rs_buffers(self):
        """
        Release RS buffers after reduce-scatter completes.
        Buffers will be lazily reallocated on next backward pass.
        Note: Using = None only (not storage().resize_(0)) because the buffer
        may have views that would cause illegal memory access if storage is resized.
        """
        if not hasattr(self, 'buffers'):
            return

        release_rs_buffers_(self.buffers)

    @torch.no_grad()
    def step_with_ready_grads(self):
        """Step optimizer after gradient synchronization completes."""

        timers = self.config.timers
        if timers is not None:
            timers('optimizer-inner-step', log_level=1).start(
                barrier=self.config.barrier_with_L1_time
            )
        if not self.is_stub_optimizer:
            self.optimizer.step()
        if timers is not None:
            timers('optimizer-inner-step').stop()

        if timers is not None:
            timers('optimizer-copy-main-to-model-params', log_level=1).start(
                barrier=self.config.barrier_with_L1_time
            )
        if not self.is_stub_optimizer:
            if self.config.reuse_grad_buf_for_mxfp8_param_ag:
                if not self.config.overlap_param_gather:
                    self._copy_main_params_to_param_buffer()
            else:
                self._copy_main_params_to_model_params()
        if timers is not None:
            timers('optimizer-copy-main-to-model-params').stop()
        return True

    def sharded_state_dict(
        self,
        model_sharded_state_dict=None,
        is_loading: bool = False,
        sharding_type=None,
        metadata=None,
    ):
        """Build optimizer checkpoint state with standard common-state structure.

        Legacy checkpoint IO remains on the inherited standard path.
        For distributed checkpoint, keep the standard common optimizer-state layout
        and store only Dion-specific per-param state as an extra payload keyed by
        logical `param_uid`.
        """
        from ..dist_checkpointing.mapping import ShardedObject

        if model_sharded_state_dict is None:
            model_sharded_state_dict = {}

        dp_rank = self.data_parallel_group.rank()
        base_key = f'optimizer.distributed.dp_group_idx_{self.data_parallel_group_idx}'
        replica_id = (self.distributed_optimizer_instance_id, 0, dp_rank)

        # Part 1: Common state (step, param_groups) - same format as parent
        common_state = self.state_dict()
        state_dict = {
            k: ShardedObject(f'{base_key}.{k}', v, (1,), (0,), replica_id=replica_id)
            for k, v in common_state.items()
        }

        # Part 2: Dion-only persistent parameter state. Keep it separate from the
        # standard distributed-optimizer param_state protocol because Dion carries
        # arbitrary-shape tensors such as Q that the standard sharding formats do not
        # represent.
        dion_param_state = build_persistent_dion_param_state(
            self.optimizer.param_groups,
            self.optimizer.state,
            self._shard_param_uid,
        )
        state_dict['dion_param_state'] = ShardedObject(
            f'{base_key}.dion_param_state', dion_param_state, (1,), (0,),
            replica_id=replica_id,
        )

        return state_dict

    def load_state_dict(self, state_dict):
        """Load optimizer checkpoint state with standard common-state outer protocol."""
        dion_param_state = state_dict.get('dion_param_state', None)
        if dion_param_state is None and state_dict.get('param_state_sharding_type') == 'dion_non_reshardable':
            dion_param_state = state_dict.get('param_state', None)

        if dion_param_state is None:
            super().load_state_dict(state_dict)
            return

        self._ensure_dion_state_initialized_for_load()

        common_state_dict = {
            k: v
            for k, v in state_dict.items()
            if k not in ('dion_param_state', 'param_state', 'param_state_sharding_type')
        }
        super().load_state_dict(common_state_dict)

        if not isinstance(dion_param_state, dict) or len(dion_param_state) == 0:
            raise RuntimeError("[Dion] distributed checkpoint missing Dion param state payload")

        restore_summary = restore_persistent_dion_param_state_(
            param_groups=self.optimizer.param_groups,
            optimizer_state=self.optimizer.state,
            get_param_key_fn=self._shard_param_uid,
            key_to_state=dion_param_state,
        )
        if restore_summary["unnamed"] > 0 or restore_summary["no_payload_entry"] > 0:
            raise RuntimeError(
                "[Dion] distributed checkpoint restore left unresolved param state entries "
                f"(restored={restore_summary['restored']} "
                f"no_payload_entry={restore_summary['no_payload_entry']} "
                f"unnamed={restore_summary['unnamed']})"
            )
        logger.info(
            '[Dion] Restored %s Dion param states from distributed checkpoint '
            '(no_payload_entry=%s, unnamed=%s)',
            restore_summary["restored"],
            restore_summary["no_payload_entry"],
            restore_summary["unnamed"],
        )

        # Sync fp32 main params from bf16 model params.
        # The fp32 copies still hold stale values from __init__ (cloned before checkpoint
        # loading). Without this, the first optimizer step would corrupt model params.
        # bf16->fp32 cast has ~1e-3 precision loss, acceptable for training resumption.
        self._copy_model_params_to_main_params()

    def _ensure_dion_state_initialized_for_load(self):
        """Initialize Dion optimizer state tensors before calling standard common-state load."""
        init_state_fn = getattr(self.optimizer, "_init_state", None)
        if init_state_fn is None:
            return
        for group in self.optimizer.param_groups:
            for param in group["params"]:
                state = self.optimizer.state.get(param)
                if state is None:
                    self.optimizer.state[param] = {}
                    state = self.optimizer.state[param]
                if len(state) == 0:
                    init_state_fn(param, state, group)

    def offload_to_cpu(self):
        """Clean up Dion-specific buffers during offload.

        Standard DO pattern handles most cleanup via:
        1. Buffer level: param_and_grad_buffer.offload_to_cpu()
        2. Optimizer state: move_optimizer("cpu")
        3. Q buffers: Auto-cleared when device changes
        """
        if hasattr(self, 'optimizer') and hasattr(self.optimizer, 'offload_to_cpu'):
            self.optimizer.offload_to_cpu()
