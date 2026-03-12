# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import functools
import logging
import math
import os
import warnings
from contextlib import nullcontext
from enum import Enum
from functools import partial
from typing import Dict, List, Optional

import torch
from torch.distributed import _coalescing_manager

import megatron.core.nccl_allocator as nccl_allocator
from megatron.core import parallel_state
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.rerun_state_machine import get_rerun_state_machine

from ..fp8_utils import (
    is_float8tensor,
    is_mxfp8tensor,
    modify_underlying_storage,
    post_all_gather_processing,
)
from ..utils import is_torch_min_version, log_on_each_pipeline_stage
from .distributed_data_parallel_config import DistributedDataParallelConfig
from .reduce_scatter_with_fp32_accumulation import reduce_scatter_with_fp32_accumulation

logger = logging.getLogger(__name__)

_DION_PACK_DEBUG_SEEN = set()

try:
    if is_torch_min_version("1.13.0"):
        dist_all_gather_func = torch.distributed.all_gather_into_tensor
        dist_reduce_scatter_func = torch.distributed.reduce_scatter_tensor
    else:
        dist_all_gather_func = torch.distributed._all_gather_base
        dist_reduce_scatter_func = torch.distributed._reduce_scatter_base
except:
    dist_all_gather_func = torch.distributed._all_gather_base
    dist_reduce_scatter_func = torch.distributed._reduce_scatter_base


class BufferType(Enum):
    """
    Enumeration for buffer type.
    """

    PARAM = 1
    GRAD = 2


def shard_buffer(buffer: torch.Tensor, data_parallel_world_size: int):
    """
    Shard buffer into data_parallel_world_size chunks of equal size.
    """
    assert buffer.numel() % data_parallel_world_size == 0
    shard_size = buffer.numel() // data_parallel_world_size
    sharded_buffer = [
        buffer[(r * shard_size) : ((r + 1) * shard_size)] for r in range(data_parallel_world_size)
    ]
    return sharded_buffer


def _dion_pack_debug_enabled(param_name: Optional[str]) -> bool:
    spec = os.getenv("DION_PARAM_GRAD_FP_NAMES", "").strip()
    if not spec or not param_name:
        return False
    if spec == "1":
        return True
    return any(token and token in param_name for token in spec.split(","))


def _log_dion_target_main_grad_phase_(bucket, *, phase: str, param_to_name=None) -> None:
    """Env-gated target-param phase trace for canonical model_param.main_grad."""
    targets_raw = os.getenv("DION_PARAM_GRAD_FP_NAMES", "").strip()
    if not targets_raw or bucket is None:
        return

    targets = [token.strip() for token in targets_raw.split(",") if token.strip()]
    if not targets:
        return

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    bucket_id = getattr(bucket, "bucket_id", None)

    for param in getattr(bucket, "params", []) or []:
        param_name = None
        if param_to_name:
            param_name = param_to_name.get(param)
        if not param_name:
            param_name = getattr(param, "_param_name", None)
        if not param_name or not any(target in param_name for target in targets):
            continue

        key = (phase, rank, bucket_id, param_name)
        if key in _DION_PACK_DEBUG_SEEN:
            continue
        _DION_PACK_DEBUG_SEEN.add(key)

        grad = getattr(param, "main_grad", None)
        if grad is None:
            logger.info(
                "[DION_MAIN_GRAD_PHASE] phase=%s rank=%s bucket=%s param=%s grad=None",
                phase,
                rank,
                bucket_id,
                param_name,
            )
            continue

        grad_fp32 = grad.detach().float()
        flat = grad_fp32.view(-1)
        logger.info(
            "[DION_MAIN_GRAD_PHASE] phase=%s rank=%s bucket=%s param=%s ptr=%s norm=%.6f sum=%.6f abs_sum=%.6f amax=%.6f sample=%s",
            phase,
            rank,
            bucket_id,
            param_name,
            int(grad_fp32.data_ptr()),
            float(grad_fp32.norm().item()),
            float(grad_fp32.sum().item()),
            float(grad_fp32.abs().sum().item()),
            float(grad_fp32.abs().max().item()) if flat.numel() > 0 else 0.0,
            flat[: min(8, flat.numel())].cpu().tolist(),
        )


def _log_dion_target_mixed_mask_phase_(bucket, *, phase: str, param_to_name=None) -> None:
    """Env-gated trace for mixed-bucket masking on target canonical main_grad tensors."""
    targets_raw = os.getenv("DION_PARAM_GRAD_FP_NAMES", "").strip()
    if not targets_raw or bucket is None:
        return

    targets = [token.strip() for token in targets_raw.split(",") if token.strip()]
    if not targets:
        return

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    bucket_id = getattr(bucket, "bucket_id", None)
    full_ranges = tuple(getattr(bucket, "mixed_dion_full_bucket_ranges", ()) or ())

    for param in getattr(bucket, "params", []) or []:
        param_name = None
        if param_to_name:
            param_name = param_to_name.get(param)
        if not param_name:
            param_name = getattr(param, "_param_name", None)
        if not param_name or not any(target in param_name for target in targets):
            continue

        key = ("mixed-mask", phase, rank, bucket_id, param_name)
        if key in _DION_PACK_DEBUG_SEEN:
            continue
        _DION_PACK_DEBUG_SEEN.add(key)

        grad = getattr(param, "main_grad", None)
        if grad is None:
            logger.info(
                "[DION_MIXED_MASK_PHASE] phase=%s rank=%s bucket=%s mixed=%s full_ranges=%s param=%s grad=None",
                phase,
                rank,
                bucket_id,
                bool(getattr(bucket, "has_non_dion_params", False)),
                full_ranges,
                param_name,
            )
            continue

        grad_fp32 = grad.detach().float()
        flat = grad_fp32.view(-1)
        logger.info(
            "[DION_MIXED_MASK_PHASE] phase=%s rank=%s bucket=%s mixed=%s full_ranges=%s param=%s ptr=%s norm=%.6f sum=%.6f abs_sum=%.6f amax=%.6f sample=%s",
            phase,
            rank,
            bucket_id,
            bool(getattr(bucket, "has_non_dion_params", False)),
            full_ranges,
            param_name,
            int(grad_fp32.data_ptr()),
            float(grad_fp32.norm().item()),
            float(grad_fp32.sum().item()),
            float(grad_fp32.abs().sum().item()),
            float(grad_fp32.abs().max().item()) if flat.numel() > 0 else 0.0,
            flat[: min(8, flat.numel())].cpu().tolist(),
        )


def lookup_mixed_non_dion_entry(bucket, param: torch.nn.Parameter, param_name: Optional[str] = None):
    """Return the canonical mixed non-Dion pack-plan entry for `param`."""
    if bucket is None:
        return None

    entry = None
    id_to_entry = getattr(bucket, "non_dion_param_id_to_entry", None)
    if id_to_entry:
        entry = id_to_entry.get(id(param))
        if entry is not None:
            return entry

    name_to_entry = getattr(bucket, "non_dion_param_name_to_entry", None)
    if name_to_entry:
        if param_name is None:
            param_name = getattr(param, "_param_name", None)
        if param_name:
            entry = name_to_entry.get(param_name)
            if entry is not None:
                return entry

    return None


def mask_mixed_dion_full_bucket_ranges_(bucket) -> None:
    """Zero Dion-owned full-bucket spans before stock mixed non-Dion RS.

    Mixed buckets should follow the stock DO local-shard contract for non-Dion params:
    - input is the canonical full `bucket.grad_data`
    - output is one local shard view of that same buffer

    Dion params are packed separately into `bucket.dion_grad_buffer`, so their full-bucket
    grad spans must be zeroed before the stock RS path runs for mixed non-Dion params.
    """
    full_ranges = getattr(bucket, "mixed_dion_full_bucket_ranges", None)
    if not full_ranges:
        return
    grad_data = getattr(bucket, "grad_data", None)
    if grad_data is None:
        raise RuntimeError("[Dion] mixed bucket is missing canonical bucket.grad_data.")
    _log_dion_target_mixed_mask_phase_(
        bucket,
        phase="before_mask_mixed_dion_ranges",
        param_to_name=getattr(bucket, "param_to_name", None),
    )
    grad_flat = grad_data.view(-1)
    for start, end in full_ranges:
        if end <= start:
            continue
        grad_flat[start:end].zero_()
    _log_dion_target_mixed_mask_phase_(
        bucket,
        phase="after_mask_mixed_dion_ranges",
        param_to_name=getattr(bucket, "param_to_name", None),
    )


def scale_mixed_non_dion_full_bucket_ranges_(bucket, scale: float) -> None:
    """Apply stock DO grad scaling to the non-Dion spans of a mixed bucket.

    Stock Megatron-Core scales the full canonical `bucket.grad_data` before reduce-scatter.
    Mixed Dion buckets split that buffer into:
    - Dion-owned spans: packed separately and already scaled during Dion packing
    - non-Dion spans: must still receive the stock `gradient_scaling_factor` on
      `bucket.grad_data` before the stock local-shard RS path runs
    """
    if scale == 1.0:
        return
    pack_plan = getattr(bucket, "non_dion_pack_plan", None)
    if not pack_plan:
        raise RuntimeError(
            f"[Dion] mixed bucket {getattr(bucket, 'bucket_id', -1)} missing canonical non-Dion pack plan."
        )
    grad_data = getattr(bucket, "grad_data", None)
    if grad_data is None:
        raise RuntimeError(
            f"[Dion] mixed bucket {getattr(bucket, 'bucket_id', -1)} missing canonical bucket.grad_data."
        )
    grad_flat = grad_data.view(-1)
    seen = set()
    for entry in pack_plan:
        full_start = int(entry.get("full_start", 0))
        full_end = int(entry.get("full_end", full_start))
        if full_end <= full_start:
            continue
        key = (full_start, full_end)
        if key in seen:
            continue
        seen.add(key)
        grad_flat[full_start:full_end].mul_(scale)


def build_mixed_non_dion_rs_input_(bucket, scale: float) -> torch.Tensor:
    """Build mixed non-Dion RS input without mutating canonical bucket.grad_data."""
    grad_data = getattr(bucket, "grad_data", None)
    if grad_data is None:
        raise RuntimeError(
            f"[Dion] mixed bucket {getattr(bucket, 'bucket_id', -1)} missing canonical bucket.grad_data."
        )

    mixed_input = grad_data.detach().clone()
    tmp_bucket = type(
        "_MixedNonDionTmpBucket",
        (),
        {
            "bucket_id": getattr(bucket, "bucket_id", -1),
            "grad_data": mixed_input,
            "non_dion_pack_plan": getattr(bucket, "non_dion_pack_plan", None),
            "mixed_dion_full_bucket_ranges": getattr(bucket, "mixed_dion_full_bucket_ranges", None),
        },
    )()
    scale_mixed_non_dion_full_bucket_ranges_(tmp_bucket, scale)
    full_ranges = getattr(tmp_bucket, "mixed_dion_full_bucket_ranges", None)
    if full_ranges:
        grad_flat = mixed_input.view(-1)
        for start, end in full_ranges:
            if end <= start:
                continue
            grad_flat[start:end].zero_()
    bucket.non_dion_grad_buffer = mixed_input
    return mixed_input


class _ParamAndGradBucket:
    """
    Bucket to keep track of a subset of the model's parameters and gradients.

    Args:
        params: List of parameters whose gradients are collated in this bucket.
        param_data: View in _ParamAndGradBuffer.param_data that this bucket is responsible for.
        grad_data: View in _ParamAndGradBuffer.grad_data that this bucket is responsible for.
        offset: Offset of this bucket's view in the larger _ParamAndGradBuffer.
        numel_unpadded: Number of unpadded elements in bucket.
        gradient_scaling_factor: This factor is utilized to scale gradients prior to their
            communication. Its application is twofold: it facilitates the averaging of gradients
            and the scaling of gradients in the context of the Mixture of Experts (MoE) model.
        bucket_id: Index of bucket in buffer.
    """

    def __init__(
        self,
        params: List[torch.nn.Parameter],
        param_data: Optional[torch.Tensor],
        grad_data: torch.Tensor,
        offset: int,
        numel_unpadded: int,
        gradient_scaling_factor: float,
        bucket_id: int,
    ):
        self.params_list = params
        self.params = set(params)
        # Make sure there are no duplicate params.
        assert len(self.params_list) == len(self.params)
        self.param_data = param_data
        self.grad_data = grad_data
        # The distributed optimizer needs to keep track of this bucket's offset
        # within the full grad_buffer.
        self.offset = offset
        self.numel_unpadded = numel_unpadded
        self.gradient_scaling_factor = gradient_scaling_factor
        self.bucket_id = bucket_id
        self.param_to_index = {}
        offset = 0
        for param in params:
            self.param_to_index[param] = (offset, offset + param.numel())
            offset += param.numel()

        # Mixed bucket support: track Dion and non-Dion params separately
        self.is_dion_bucket = False
        self.dion_comm_group = None  # FS communication group

        # Classify params within this bucket
        self.dion_params = []      # Dion params (2D, FS sharded)
        self.non_dion_params = []  # Non-Dion params (1D, DP sharded)
        for param in params:
            if getattr(param, 'is_dion_param', False):
                self.dion_params.append(param)
            else:
                self.non_dion_params.append(param)

        # Track param ranges within bucket
        self.dion_param_ranges = {}
        self.non_dion_param_ranges = {}

        # Grad buffer sections
        self.dion_grad_section_size = 0
        self.non_dion_grad_section_size = 0

        # Full grad buffers for accumulation
        self.dion_grad_buffer = None
        self.dion_grad_local_view = None
        self.non_dion_grad_buffer = None
        self.non_dion_grad_local_view = None
        self.dion_shard_size = 0
        self.non_dion_shard_size = 0
        self.dion_param_layout = []
        self.fs_full_grad_total = 0

        # Param shard range for FS sharding
        self.dion_param_shard_range = {}

        # All-gather callback function
        self.fs_all_gather_fn = None
        self.non_dion_all_gather_fn = None

        # Cached computed properties
        self._cached_dion_param_ids = None
        self.dion_param_ptr_to_entry = None
        self.dion_param_name_to_entry = None
        self._dion_param_layout_cache_len = None

    # Helper Properties

    @property
    def dion_param_ids(self) -> set:
        """Set of Dion param IDs from dion_param_layout."""
        if self._cached_dion_param_ids is None:
            if self.dion_param_layout:
                self._cached_dion_param_ids = {
                    id(entry['param']) for entry in self.dion_param_layout
                }
            else:
                self._cached_dion_param_ids = set()
        return self._cached_dion_param_ids

    @property
    def has_dion_params(self) -> bool:
        """Whether bucket has Dion params (requires both layout and buffer)."""
        return (
            bool(self.dion_param_layout)
            and len(self.dion_param_layout) > 0
            and self.dion_grad_buffer is not None
        )

    @property
    def has_non_dion_params(self) -> bool:
        """Whether bucket has non-Dion params."""
        return any(id(p) not in self.dion_param_ids for p in self.params)

    @property
    def is_mixed(self) -> bool:
        """Whether bucket is mixed (has both Dion and non-Dion params)."""
        return self.has_dion_params and self.has_non_dion_params

    @property
    def is_pure_dion(self) -> bool:
        """Whether bucket is pure Dion (only Dion params)."""
        return self.has_dion_params and not self.has_non_dion_params

    def invalidate_dion_cache(self):
        """Invalidate Dion param cache (call when layout changes)."""
        self._cached_dion_param_ids = None
        self.dion_param_ptr_to_entry = None
        self.dion_param_name_to_entry = None
        self._dion_param_layout_cache_len = None


class _ParamAndGradBucketGroup:
    """
    Put multiple buckets into a group so that their communications can be aggregated together.
    Provides functionality to register when params in the bucket group have grads ready to be
    synced; an asynchronous communication call is automatically launched when _all_ params in
    the bucket group have grads ready.

    Args:
        buckets: A list of buckets.
        ddp_config: DistributedDataParallel config object.
        collective_group: intra_distributed_optimizer_instance_group if using distributed
            optimizer, data_parallel_group if not.
        collective_group_size: World size using the intra data-parallel group.
    """

    def __init__(
        self,
        buckets: List[_ParamAndGradBucket],
        ddp_config: DistributedDataParallelConfig,
        collective_group: torch.distributed.ProcessGroup,
        collective_group_size: int,
    ):
        self.buckets = buckets
        self.ddp_config = ddp_config

        if self.ddp_config.use_distributed_optimizer:
            self.intra_distributed_optimizer_instance_group = collective_group
            self.intra_distributed_optimizer_instance_size = collective_group_size
            self.intra_distributed_optimizer_instance_rank = collective_group.rank()
        else:
            self.data_parallel_group = collective_group

        # State for bookkeeping: params is the set of parameters this bucket group is
        # responsible for, params_with_grad is the set of parameters with grads
        # available. When overlap_grad_reduce is True, communication (all-reduce
        # or reduce-scatter) is issued when params_with_grad equals params.
        self.param_to_bucket = {}
        self.params = set()
        for bucket in self.buckets:
            for param in bucket.params_list:
                self.param_to_bucket[param] = bucket
                self.params.add(param)

        self.next_param_gather_bucket_group = None

        if self.ddp_config.num_distributed_optimizer_instances > 1:
            self.inter_distributed_optimizer_instance_group = None
            self.communication_stream = None
            assert (
                not self.ddp_config.reduce_scatter_with_fp32_accumulation
            ), "RS w/ FP32 accumulation not supported with num_distributed_optimizer_instances > 1"

        global dist_reduce_scatter_func
        if self.ddp_config.reduce_scatter_with_fp32_accumulation:
            dist_reduce_scatter_func = reduce_scatter_with_fp32_accumulation

        self.reset()
        self.param_gather_handle = None
        self.param_gather_dispatched = False
        self.grad_reduce_handle = None

        # Each time a local shard is created from bucket.param_data or bucket.grad_data, it
        # introduces some CPU overheads. We use these two lists to cache the created local
        # shards to avoid unnecessary CPU operations. This does not increase GPU memory usage
        # because it only saves a slice view, which shares the same memory with bucket.param_data
        # or bucket.grad_data.
        self.cached_param_buffer_shard_list = [None] * len(self.buckets)
        self.cached_grad_buffer_shard_list = [None] * len(self.buckets)

    def _mark_dion_param_sync_ready(self, ready: bool):
        """Update forward-readiness state for custom Dion param-gather buckets."""
        for bucket in self.buckets:
            if hasattr(bucket, "_dion_full_param_ready"):
                bucket._dion_full_param_ready = ready

    def _check_dion_param_sync_ready(self):
        """Verify bucket.param_data remains the forward-visible canonical storage."""
        for bucket in self.buckets:
            if not getattr(bucket, "_dion_requires_param_sync_check", False):
                continue
            optimizer = getattr(bucket, "dion_optimizer", None)
            if optimizer is None:
                continue
            optimizer._check_bucket_param_views(bucket, context="finish_param_sync")
            bucket._dion_full_param_ready = True

    def _get_stock_local_grad_view(self, idx: int, bucket: _ParamAndGradBucket) -> torch.Tensor:
        """Return the canonical stock local optimizer shard for one bucket."""
        if self.cached_grad_buffer_shard_list[idx] is None:
            self.cached_grad_buffer_shard_list[idx] = shard_buffer(
                bucket.grad_data, self.intra_distributed_optimizer_instance_size
            )
        return self.cached_grad_buffer_shard_list[idx][
            self.intra_distributed_optimizer_instance_rank
        ]

    def _has_dion_runtime(self, bucket: _ParamAndGradBucket) -> bool:
        """Whether a bucket has active Dion grad runtime state for this step."""
        return (
            bool(getattr(bucket, "dion_param_layout", None))
            and getattr(bucket, "dion_grad_buffer", None) is not None
        )

    def _has_non_dion(self, bucket: _ParamAndGradBucket) -> bool:
        """Whether a bucket owns any non-Dion params."""
        return bucket.has_non_dion_params

    def _is_pure_non_dion(self, bucket: _ParamAndGradBucket) -> bool:
        """Whether a bucket should follow the stock DO RS path without Dion helpers."""
        return self._has_non_dion(bucket) and not bool(getattr(bucket, "dion_param_layout", None))

    def _start_dion_bucket_rs(
        self,
        *,
        bucket: _ParamAndGradBucket,
        reduce_op,
        async_op: bool,
        handles: list,
        communication_group,
    ) -> None:
        """Launch the Dion-only logical local-shard RS path for one bucket."""
        has_dion = self._has_dion_runtime(bucket)
        if not has_dion:
            bucket.dion_grad_local_view = None
            return

        dion_rs_group = getattr(bucket, "dion_comm_group", None) or communication_group
        dion_shard_size = int(getattr(bucket, "dion_shard_size", 0))
        dion_grad_buffer = bucket.dion_grad_buffer
        expected_input = dion_shard_size * dion_rs_group.size()
        if dion_grad_buffer is None or dion_grad_buffer.numel() != expected_input:
            raise RuntimeError(
                f"[Dion] Invalid Dion RS buffer size: bucket={bucket.bucket_id} "
                f"input={0 if dion_grad_buffer is None else dion_grad_buffer.numel()} "
                f"expected={expected_input} shard={dion_shard_size} group={dion_rs_group.size()}"
            )

        dion_shards = getattr(bucket, "cached_dion_grad_buffer_shard_list", None)
        if (
            dion_shards is None
            or len(dion_shards) != dion_rs_group.size()
            or dion_shards[0].numel() * dion_rs_group.size() != dion_grad_buffer.numel()
        ):
            dion_shards = shard_buffer(dion_grad_buffer, dion_rs_group.size())
            bucket.cached_dion_grad_buffer_shard_list = dion_shards
        dion_rs_output = dion_shards[dion_rs_group.rank()]
        bucket.dion_grad_local_view = dion_rs_output

        if dion_rs_group.size() <= 1:
            dion_rs_output.copy_(dion_grad_buffer[:dion_shard_size])
            return

        handle = dist_reduce_scatter_func(
            dion_rs_output,
            dion_grad_buffer,
            op=reduce_op,
            group=dion_rs_group,
            async_op=async_op,
        )
        if async_op and handle is not None:
            handles.append(handle)

    def _start_non_dion_bucket_rs(
        self,
        *,
        idx: int,
        bucket: _ParamAndGradBucket,
        reduce_op,
        async_op: bool,
        handles: list,
        communication_group,
    ) -> None:
        """Launch the canonical stock local-shard RS path for pure/mixed non-Dion grads."""
        has_non_dion = self._has_non_dion(bucket)
        has_dion = self._has_dion_runtime(bucket)
        stock_local_data_view = self._get_stock_local_grad_view(idx, bucket)

        if not has_non_dion:
            bucket.non_dion_grad_local_view = None
            if has_dion:
                stock_local_data_view.zero_()
            return

        bucket.non_dion_grad_local_view = stock_local_data_view

        if has_dion:
            non_dion_group = getattr(bucket, "non_dion_dp_group", None)
            if non_dion_group is None:
                raise RuntimeError(
                    f"[Dion] Mixed bucket {bucket.bucket_id} missing canonical non-Dion group."
                )
            non_dion_local_shard_size = int(getattr(bucket, "non_dion_pack_total", 0))
            if non_dion_local_shard_size != stock_local_data_view.numel():
                raise RuntimeError(
                    "[Dion] Mixed non-Dion local shard size mismatch "
                    f"bucket={bucket.bucket_id} local={non_dion_local_shard_size} "
                    f"stock_local={stock_local_data_view.numel()}"
                )
            # Follow the stock DO contract as closely as possible:
            # - bucket.grad_data remains the canonical full-bucket grad buffer
            # - mixed non-Dion takes the stock local-shard RS path on a dedicated
            #   stock-equivalent input buffer so canonical model_param.main_grad is preserved
            mixed_non_dion_input = build_mixed_non_dion_rs_input_(
                bucket,
                bucket.gradient_scaling_factor,
            )
            if non_dion_group.size() <= 1:
                stock_local_data_view.copy_(mixed_non_dion_input[:non_dion_local_shard_size])
                return
            handle = dist_reduce_scatter_func(
                stock_local_data_view,
                mixed_non_dion_input,
                op=reduce_op,
                group=non_dion_group,
                async_op=async_op,
            )
            if async_op and handle is not None:
                handles.append(handle)
            return

        expected_bucket_input = stock_local_data_view.numel() * communication_group.size()
        if bucket.grad_data.numel() != expected_bucket_input:
            raise RuntimeError(
                f"[Dion] Invalid stock RS buffer size: bucket={bucket.bucket_id} "
                f"input={bucket.grad_data.numel()} expected={expected_bucket_input} "
                f"local={stock_local_data_view.numel()} group={communication_group.size()}"
            )
        handle = dist_reduce_scatter_func(
            stock_local_data_view,
            bucket.grad_data,
            op=reduce_op,
            group=communication_group,
            async_op=async_op,
        )
        if async_op and handle is not None:
            handles.append(handle)

    def _all_reduce_inter_instance_local_views(
        self,
        *,
        idx: int,
        bucket: _ParamAndGradBucket,
        reduce_op,
        async_op: bool,
    ) -> None:
        """All-reduce the already-materialized optimizer local views across instances."""
        dion_local_view = getattr(bucket, "dion_grad_local_view", None)
        has_non_dion = self._has_non_dion(bucket)
        non_dion_local_view = getattr(bucket, "non_dion_grad_local_view", None)

        if dion_local_view is not None and dion_local_view.numel() > 0:
            torch.distributed.all_reduce(
                dion_local_view,
                op=reduce_op,
                group=self.inter_distributed_optimizer_instance_group,
                async_op=async_op,
            )

        if not has_non_dion:
            bucket.non_dion_grad_local_view = None
            return

        if non_dion_local_view is None:
            non_dion_local_view = self._get_stock_local_grad_view(idx, bucket)
            bucket.non_dion_grad_local_view = non_dion_local_view

        if non_dion_local_view is not None and non_dion_local_view.numel() > 0:
            torch.distributed.all_reduce(
                non_dion_local_view,
                op=reduce_op,
                group=self.inter_distributed_optimizer_instance_group,
                async_op=async_op,
            )

    def reset(self):
        """
        Reset metadata in bucket group in preparation for the next iteration of training.
        """
        self.params_with_grad = set()
        self.is_last_microbatch = True

    def check_grads(self, check_for_nan_or_inf, check_for_large):
        """
        Make sure norm of grads in bucket are not NaN prior to data-parallel
        all-reduce / reduce-scatter.
        """
        rerun_state_machine = get_rerun_state_machine()
        for i in range(len(self.buckets)):
            grad_norm = self.buckets[i].grad_data.norm(p=2)
            # check for NaN, Inf and unexpectedly large grads
            if check_for_nan_or_inf:
                rerun_state_machine.validate_result(
                    result=grad_norm,
                    rejection_func=torch.isnan,
                    message=f"found NaN in local grad norm for bucket #{i} "
                    f"in backward pass before data-parallel communication collective",
                    tolerance=0.001,  # 0.1% tolerance to account for non-deterministic FA backward
                    fatal=True,
                )
                rerun_state_machine.validate_result(
                    result=grad_norm,
                    rejection_func=torch.isinf,
                    message=f"found Inf in local grad norm for bucket #{i} "
                    f"in backward pass before data-parallel communication collective",
                    tolerance=0.001,  # 0.1% tolerance to account for non-deterministic FA backward
                    fatal=True,
                )
            if check_for_large:
                rerun_state_machine.validate_result(
                    result=grad_norm,
                    rejection_func=partial(
                        rerun_state_machine.is_unexpectedly_large, threshold=10, context="grads"
                    ),
                    message=f"found unexpected large grads in bucket #{i} "
                    f"in backward pass before data-parallel communication collective",
                    tolerance=0.001,  # 0.1% tolerance to account for non-deterministic FA backward
                    fatal=False,
                )

    def start_param_sync(self, force_sync: bool = False):
        """
        Initiates all necessary param all-gathers for this bucket.

        When ddp_config.overlap_param_gather is set to True, dispatches an asynchronous
        communication call (unless force_sync is True). When ddp_config.overlap_param_gather
        is set to False, makes synchronous call.

        Args:
            force_sync (bool, optional): force synchronous collective regardless of
                other settings if true.
        """
        assert self.ddp_config.use_distributed_optimizer

        # Check if any bucket has Dion params
        has_any_dion = any(
            hasattr(bucket, 'dion_param_layout') and bucket.dion_param_layout and len(bucket.dion_param_layout) > 0
            for bucket in self.buckets
        )

        if force_sync:
            if self.param_gather_handle is not None:
                self.param_gather_handle.wait()
                self.param_gather_handle = None
            self._mark_dion_param_sync_ready(False)

            # For Dion params: perform fresh all-gather after optimizer.step()
            if has_any_dion:
                pure_non_dion_indices = []
                for bucket in self.buckets:
                    has_dion = (
                        hasattr(bucket, 'dion_param_layout')
                        and bucket.dion_param_layout
                        and len(bucket.dion_param_layout) > 0
                    )
                    has_non_dion = bucket.has_non_dion_params

                    if has_dion and hasattr(bucket, 'fs_all_gather_fn') and bucket.fs_all_gather_fn is not None:
                        bucket.fs_all_gather_fn(async_op=False)
                        if has_non_dion and hasattr(bucket, 'non_dion_all_gather_fn') and bucket.non_dion_all_gather_fn is not None:
                            bucket.non_dion_all_gather_fn(async_op=False)
                    elif has_non_dion:
                        pure_non_dion_indices.append(bucket)

                if pure_non_dion_indices:
                    with _coalescing_manager(
                        self.intra_distributed_optimizer_instance_group, async_ops=False
                    ):
                        for idx, bucket in enumerate(self.buckets):
                            if bucket not in pure_non_dion_indices:
                                continue
                            if self.cached_param_buffer_shard_list[idx] is None:
                                self.cached_param_buffer_shard_list[idx] = shard_buffer(
                                    bucket.param_data, self.intra_distributed_optimizer_instance_size
                                )
                            local_data_view = self.cached_param_buffer_shard_list[idx][
                                self.intra_distributed_optimizer_instance_rank
                            ]
                            dist_all_gather_func(
                                bucket.param_data,
                                local_data_view,
                                group=self.intra_distributed_optimizer_instance_group,
                                async_op=False,
                            )
                self._check_dion_param_sync_ready()
            return
        else:
            assert self.param_gather_handle is None

        async_op = self.ddp_config.overlap_param_gather and not force_sync
        self._mark_dion_param_sync_ready(False)

        if has_any_dion:
            # Keep pure non-Dion buckets on the stock DO coalesced all-gather path even when
            # other buckets in the same bucket-group contain Dion params.
            pure_non_dion_indices = []
            custom_handles = []

            for idx, bucket in enumerate(self.buckets):
                has_dion = (
                    hasattr(bucket, 'dion_param_layout')
                    and bucket.dion_param_layout
                    and len(bucket.dion_param_layout) > 0
                )
                has_non_dion = bucket.has_non_dion_params

                if has_dion and hasattr(bucket, 'fs_all_gather_fn') and bucket.fs_all_gather_fn is not None:
                    fs_handles = bucket.fs_all_gather_fn(async_op=async_op)
                    if fs_handles:
                        bucket.fs_param_gather_handles = fs_handles

                if not has_non_dion:
                    continue

                if has_dion:
                    if hasattr(bucket, 'non_dion_all_gather_fn') and bucket.non_dion_all_gather_fn is not None:
                        non_dion_handles = bucket.non_dion_all_gather_fn(async_op=async_op)
                        if non_dion_handles:
                            bucket.non_dion_param_gather_handles = non_dion_handles
                else:
                    pure_non_dion_indices.append(idx)

            stock_cm = None
            if pure_non_dion_indices:
                with _coalescing_manager(
                    self.intra_distributed_optimizer_instance_group, async_ops=async_op
                ) as stock_cm:
                    for idx in pure_non_dion_indices:
                        bucket = self.buckets[idx]
                        if self.cached_param_buffer_shard_list[idx] is None:
                            self.cached_param_buffer_shard_list[idx] = shard_buffer(
                                bucket.param_data, self.intra_distributed_optimizer_instance_size
                            )
                        local_data_view = self.cached_param_buffer_shard_list[idx][
                            self.intra_distributed_optimizer_instance_rank
                        ]
                        dist_all_gather_func(
                            bucket.param_data,
                            local_data_view,
                            group=self.intra_distributed_optimizer_instance_group,
                            async_op=async_op,
                        )

            if async_op:
                class _HandleList:
                    def __init__(self, hs):
                        self.hs = [h for h in hs if h is not None]

                    def wait(self):
                        for h in self.hs:
                            if hasattr(h, "wait"):
                                h.wait()

                stock_handle = stock_cm if pure_non_dion_indices else None
                self.param_gather_handle = _HandleList([stock_handle, *custom_handles])
            else:
                self.param_gather_handle = None
                self._check_dion_param_sync_ready()
            self.param_gather_dispatched = True
            return

        # Original path for non-Dion
        # Coalesce communication kernels across buckets in the bucket group.
        with _coalescing_manager(
            self.intra_distributed_optimizer_instance_group, async_ops=async_op
        ) as cm:
            for idx, bucket in enumerate(self.buckets):
                if self.cached_param_buffer_shard_list[idx] is None:
                    self.cached_param_buffer_shard_list[idx] = shard_buffer(
                        bucket.param_data, self.intra_distributed_optimizer_instance_size
                    )
                local_data_view = self.cached_param_buffer_shard_list[idx][
                    self.intra_distributed_optimizer_instance_rank
                ]
                dist_all_gather_func(
                    bucket.param_data,
                    local_data_view,
                    group=self.intra_distributed_optimizer_instance_group,
                    async_op=async_op,
                )
        if async_op:
            self.param_gather_handle = cm
        else:
            self.param_gather_handle = None
            self._check_dion_param_sync_ready()
        self.param_gather_dispatched = True

    def finish_param_sync(self, skip_next_bucket_dispatch: bool = False):
        """
        Finishes param sync communication operation for this bucket. Dispatches
        next bucket's param sync if available, unless skip_next_bucket_dispatch
        is True.

        When ddp_config.overlap_param_gather is set to True, waits for asynchronous
        communication call to complete (and dispatches one if one is not already
        outstanding). Throws assertion error if ddp_config.overlap_param_gather is set to
        False.

        Args:
            skip_next_bucket_dispatch (bool, optional): if true, dispatch next
                bucket's communication if available.
        """
        assert self.ddp_config.use_distributed_optimizer
        assert self.ddp_config.overlap_param_gather

        # If current bucket's param AG has not been dispatched, dispatch it now (e.g., first
        # AG bucket in first model chunk if ddp_config.align_param_gather is False).
        if not self.param_gather_dispatched:
            self.start_param_sync()

        # Check if any bucket has Dion handles that need to be waited/unpacked
        has_any_dion_handles = any(
            (hasattr(bucket, 'fs_param_gather_handles') and bucket.fs_param_gather_handles is not None)
            or (hasattr(bucket, 'non_dion_param_gather_handles') and bucket.non_dion_param_gather_handles is not None)
            for bucket in self.buckets
        )

        if has_any_dion_handles:
            # Wait for and unpack Dion async handles

            # Wait for standard handle (for pure non-Dion buckets)
            if self.param_gather_handle is not None:
                self.param_gather_handle.wait()
                self.param_gather_handle = None

            # Wait and unpack Dion handles for each bucket
            for bucket in self.buckets:
                # --- Dion (FS) params: wait and unpack ---
                if hasattr(bucket, 'fs_param_gather_handles') and bucket.fs_param_gather_handles is not None:
                    handle_info = bucket.fs_param_gather_handles
                    if 'handle' in handle_info and handle_info['handle'] is not None:
                        handle_info['handle'].wait()
                    # Unpack gathered params to model params
                    if 'optimizer' in handle_info and handle_info['optimizer'] is not None:
                        handle_info['optimizer']._unpack_all_gathered_params(
                            gathered_buffer=handle_info['gathered_buffer'],
                            dion_param_layout=handle_info['dion_param_layout'],
                            pack_total=handle_info['pack_total'],
                            fs_size=handle_info['fs_size'],
                            buffer=handle_info['buffer'],
                        )
                    bucket.fs_param_gather_handles = None

                # --- Non-Dion params in mixed bucket: wait and unpack ---
                if hasattr(bucket, 'non_dion_param_gather_handles') and bucket.non_dion_param_gather_handles is not None:
                    handle_info = bucket.non_dion_param_gather_handles
                    if 'handle' in handle_info and handle_info['handle'] is not None:
                        handle_info['handle'].wait()
                    if 'optimizer' in handle_info and handle_info['optimizer'] is not None:
                        if hasattr(handle_info['optimizer'], '_unpack_non_dion_params'):
                            handle_info['optimizer']._unpack_non_dion_params(
                                handle_info.get('gathered_buffer'),
                                handle_info.get('pack_plan'),
                                handle_info.get('pack_total'),
                                handle_info.get('dp_size'),
                                handle_info.get('buffer'),
                            )
                    bucket.non_dion_param_gather_handles = None

            self._check_dion_param_sync_ready()

            # Dispatch next bucket's asynchronous param AG only if it has not been dispatched yet.
            if self.next_param_gather_bucket_group is not None and not skip_next_bucket_dispatch:
                if self.next_param_gather_bucket_group.param_gather_dispatched:
                    warnings.warn(
                        "The next bucket's parameter all-gather operation has already been "
                        "dispatched. This may be caused by a mismatch between the order of "
                        "parameter registration and forward pass execution, which will "
                        "hurt the communication-computation overlap performance."
                    )
                else:
                    self.next_param_gather_bucket_group.start_param_sync()

        else:
            # No Dion handles - use original logic
            if self.param_gather_handle is not None:
                self.param_gather_handle.wait()
                self.param_gather_handle = None
                self._check_dion_param_sync_ready()
                # Dispatch next bucket's asynchronous param AG only if it has not been dispatched yet.
                if self.next_param_gather_bucket_group is not None and not skip_next_bucket_dispatch:
                    if self.next_param_gather_bucket_group.param_gather_dispatched:
                        warnings.warn(
                            "The next bucket's parameter all-gather operation has already been "
                            "dispatched. This may be caused by a mismatch between the order of "
                            "parameter registration and forward pass execution, which will "
                            "hurt the communication-computation overlap performance."
                        )
                    else:
                        self.next_param_gather_bucket_group.start_param_sync()

                # For the mxfp8_param with "reuse_grad_buf_for_mxfp8_param_ag=True",
                # we need to copy the param_data from the shared_param/grad_buffer to param.data
                # after the param all-gather.
                if (
                    self.ddp_config.reuse_grad_buf_for_mxfp8_param_ag
                    and self.ddp_config.overlap_param_gather
                ):
                    for bucket in self.buckets:
                        for param in bucket.params:
                            param_start, param_end = bucket.param_to_index[param]
                            param_slice = bucket.param_data.view(-1)[param_start:param_end]
                            param.data.copy_(param_slice.view(param.data.shape))
                        # All-gathered params are not needed after being copied to param.data.
                        # Zero out the param buffer (shared with grad buffer) for gradient accumulation.
                        # We cannot zero out the entire grad buffer because one grad buffer may
                        # correspond to multiple param buffers. If we zero out the entire grad buffer,
                        # it would clear the data of those param buffers that have not yet completed AG.
                        bucket.param_data.zero_()

    def start_grad_sync(self):
        """
        Initiates grad sync (all-reduce or reduce-scatter) communication operations
        for all buckets in the bucket group.

        When ddp_config.overlap_grad_reduce is set to True, dispatches an asynchronous
        communication call. When ddp_config.overlap_grad_reduce is set to False, makes
        synchronous call.
        """
        assert (
            self.grad_reduce_handle is None
        ), "Should not have multiple communication calls outstanding at once"

        if self.ddp_config.check_for_nan_in_grad or self.ddp_config.check_for_large_grads:
            self.check_grads(
                check_for_nan_or_inf=self.ddp_config.check_for_nan_in_grad,
                check_for_large=self.ddp_config.check_for_large_grads,
            )

        # Pack Dion gradients with row-split/col-split separation
        for bucket in self.buckets:
            has_dion_layout = (hasattr(bucket, 'dion_param_layout') and bucket.dion_param_layout and
                              len(bucket.dion_param_layout) > 0)
            if not has_dion_layout:
                continue

            fs_group = getattr(bucket, 'dion_comm_group', None)
            fs_size = fs_group.size() if fs_group else 1
            pack_total = getattr(bucket, 'fs_pack_total', 0)
            dion_shard_size = getattr(bucket, 'dion_shard_size', 0)

            if dion_shard_size <= 0 or pack_total <= 0:
                continue

            packing_size = fs_size

            all_entries = []
            missing_main_grad = 0
            for entry in bucket.dion_param_layout:
                entry_param = entry.get("param")
                if entry_param is None:
                    continue
                if not hasattr(entry_param, "main_grad") or entry_param.main_grad is None:
                    missing_main_grad += 1
                    continue
                entry["_bucket_param"] = entry_param
                all_entries.append(entry)

            if missing_main_grad > 0:
                logger.warning(
                    "[Dion] bucket %s skipped %s Dion layout entries with missing main_grad during RS packing",
                    bucket.bucket_id,
                    missing_main_grad,
                )

            total_buffer_size = pack_total * packing_size
            # IMPORTANT: Always allocate separate buffer.
            # Reusing grad_data causes overlap between RS input and output
            # (dion_grad_buffer = grad_data[:N*pack_total], RS output = grad_data[:pack_total]),
            # which corrupts NCCL reduce_scatter results.
            dion_grad_buffer = torch.zeros(
                total_buffer_size,
                dtype=bucket.grad_data.dtype,
                device=bucket.grad_data.device,
            )
            bucket._memory_reused = False
            bucket.dion_grad_buffer = dion_grad_buffer
            bucket.dion_rs_local_buffer = None
            bucket.dion_grad_local_view = None
            bucket.dion_pack_debug_inputs = {}
            bucket.dion_pack_debug_is_avg = bool(self.ddp_config.average_in_collective)
            bucket.dion_pack_debug_group = fs_group

            for entry in all_entries:
                bucket_param = entry['_bucket_param']
                param_name = getattr(bucket_param, "_param_name", None)
                grad = bucket_param.main_grad.data
                pack_offset = entry['pack_offset']
                size_per_rank = entry.get('size_per_rank', 0)
                fs_split_dim = entry.get('fs_split_dim', 0)
                global_shape = entry.get('global_shape', grad.shape)
                local_shape = entry.get('local_shape', grad.shape)
                local_start_idx = int(entry.get('start_idx', 0))
                local_end_idx = int(entry.get('end_idx', 0))

                if fs_split_dim == 0:
                    m = global_shape[0]
                    n = local_shape[1] if len(local_shape) > 1 else grad.numel() // m
                else:
                    m = local_shape[0] if len(local_shape) > 0 else global_shape[0]
                    n = global_shape[1]

                if m * n != grad.numel():
                    if grad.ndim == 2:
                        m, n = grad.shape
                    else:
                        m, n = grad.numel(), 1

                grad_2d = grad.view(m, n)
                if os.environ.get("DION_DIAG_FS_PACK_COMPARE"):
                    try:
                        fs_group_rank = int(torch.distributed.get_rank(group=fs_group)) if fs_group else 0
                        pack_start_idx = fs_group_rank * size_per_rank
                        pack_end_idx = min(
                            pack_start_idx + size_per_rank,
                            m if fs_split_dim == 0 else n,
                        )
                        if fs_split_dim == 0:
                            pack_local = grad_2d[pack_start_idx:pack_end_idx, :]
                            direct_local = grad_2d[local_start_idx:local_end_idx, :]
                        else:
                            pack_local = grad_2d[:, pack_start_idx:pack_end_idx]
                            direct_local = grad_2d[:, local_start_idx:local_end_idx]
                        pack_local_max = (
                            float(pack_local.abs().amax()) if pack_local.numel() > 0 else 0.0
                        )
                        direct_local_max = (
                            float(direct_local.abs().amax()) if direct_local.numel() > 0 else 0.0
                        )
                        if (
                            pack_start_idx != local_start_idx
                            or pack_end_idx != local_end_idx
                            or (pack_local_max == 0.0) != (direct_local_max == 0.0)
                        ):
                            param_name = getattr(bucket_param, "_param_name", f"id_{id(bucket_param)}")
                            logger.error(
                                "[DION_FS_PACK_COMPARE] param=%s bucket=%s fs_rank=%s fs_split_dim=%s "
                                "pack=[%s:%s] direct=[%s:%s] grad_shape=%s local_shape=%s global_shape=%s "
                                "pack_local_max=%.3e direct_local_max=%.3e",
                                param_name,
                                getattr(bucket, "bucket_id", None),
                                fs_group_rank,
                                fs_split_dim,
                                pack_start_idx,
                                pack_end_idx,
                                local_start_idx,
                                local_end_idx,
                                tuple(grad_2d.shape),
                                tuple(local_shape),
                                tuple(global_shape),
                                pack_local_max,
                                direct_local_max,
                            )
                    except Exception as error:
                        logger.error("[DION_FS_PACK_COMPARE_FAILED] err=%r", error)
                # Scale for Dion packing before reduce-scatter.
                is_expert_bucket = not getattr(bucket_param, 'allreduce', True)
                if is_expert_bucket:
                    scale = bucket.gradient_scaling_factor
                else:
                    if self.ddp_config.average_in_collective:
                        scale = 1.0
                    else:
                        scale = bucket.gradient_scaling_factor

                fs_group_rank = int(torch.distributed.get_rank(group=fs_group)) if fs_group else 0
                if _dion_pack_debug_enabled(param_name):
                    all_pack_scaled = []
                    for target_rank in range(packing_size):
                        target_start_idx = target_rank * size_per_rank
                        target_end_idx = min(
                            target_start_idx + size_per_rank,
                            m if fs_split_dim == 0 else n,
                        )
                        if fs_split_dim == 0:
                            target_pack_seg = grad_2d[target_start_idx:target_end_idx, :]
                        else:
                            target_pack_seg = grad_2d[:, target_start_idx:target_end_idx]
                        target_pack_scaled = target_pack_seg.detach().float().clone()
                        if scale != 1.0:
                            target_pack_scaled.mul_(scale)
                        all_pack_scaled.append(target_pack_scaled.cpu())
                    bucket.dion_pack_debug_inputs[id(bucket_param)] = {
                        "param_name": param_name,
                        "buffer_idx": getattr(bucket, "global_buffer_idx", None),
                        "bucket_id": getattr(bucket, "bucket_id", None),
                        "pack_offset": int(pack_offset),
                        "local_shape": tuple(int(x) for x in local_shape),
                        "fs_split_dim": int(fs_split_dim),
                        "size_per_rank": int(size_per_rank),
                        "start_idx": int(local_start_idx),
                        "end_idx": int(local_end_idx),
                        "scale": float(scale),
                        "all_scaled": all_pack_scaled,
                    }

                for rank_j in range(packing_size):
                    fs_pos = rank_j
                    start_idx = fs_pos * size_per_rank
                    end_idx = min(start_idx + size_per_rank, m if fs_split_dim == 0 else n)

                    if fs_split_dim == 0:
                        seg_2d = grad_2d[start_idx:end_idx, :]
                    else:
                        seg_2d = grad_2d[:, start_idx:end_idx]

                    if seg_2d.numel() == 0:
                        continue

                    buf_offset = rank_j * pack_total + pack_offset
                    out = dion_grad_buffer[buf_offset:buf_offset + seg_2d.numel()].view_as(seg_2d)
                    out.copy_(seg_2d)
                    if scale != 1.0:
                        out.mul_(scale)

                del entry['_bucket_param']

        # For mixed buckets, non-Dion params follow the stock Megatron-Core DO path:
        # TE/backward hook accumulates into canonical `param.main_grad`, and the bucket
        # participates in the standard bucket.grad_data RS/AR lifecycle below. No custom
        # mixed non-Dion packing is performed here.

        # gradient_scaling_factor already takes into account whether we are computing
        # an average or sum in the data-parallel collective.
        for bucket in self.buckets:
            if bucket.gradient_scaling_factor != 1.0:
                has_dion = bool(getattr(bucket, 'dion_param_layout', None))
                has_non_dion = self._has_non_dion(bucket)
                if has_dion and has_non_dion:
                    # Mixed bucket:
                    # - Dion spans are packed separately below and already pre-scaled there
                    # - non-Dion spans still need stock DO full-buffer scaling, but
                    #   mixed non-Dion RS now uses a dedicated input buffer so the
                    #   canonical bucket.grad_data / model_param.main_grad stay intact.
                    pass
                elif has_dion and not has_non_dion:
                    # Pure Dion bucket: skip, Dion packing already pre-scaled.
                    pass
                else:
                    # Pure non-Dion buckets follow stock bucket.grad_data scaling.
                    bucket.grad_data *= bucket.gradient_scaling_factor

        # Decide reduce_op.
        reduce_op = torch.distributed.ReduceOp.SUM
        if self.ddp_config.average_in_collective:
            reduce_op = torch.distributed.ReduceOp.AVG

        # We use the following stream synchronization for the gradient reduction
        # within and across DistOpt instances.

        # Compute Stream: -------------Gradient compute-------------------
        # Comm. Stream:   ------(wait for NCCL)-----(wait for NCCL)-------
        # NCCL Stream:          -------RS------     -------AR------

        # Use async communications only when overlap_grad_reduce is True.
        async_op = (
            self.ddp_config.overlap_grad_reduce
            and self.ddp_config.num_distributed_optimizer_instances == 1
        )
        if (
            self.ddp_config.num_distributed_optimizer_instances > 1
            and self.ddp_config.overlap_grad_reduce
        ):
            # Assign a communication stream if we have multiple DistOpt instances and we
            # need to overlap communication.
            stream_context = torch.cuda.stream(self.communication_stream)

            # The RS/AR communication stream needs to wait for the default stream
            # to complete its gradient computation before launching the next
            # gradient reduction collective.
            self.communication_stream.wait_stream(torch.cuda.default_stream())
        else:
            stream_context = nullcontext()

        # Default communication group (overridden per-bucket for Dion buckets)
        communication_group = (
            self.intra_distributed_optimizer_instance_group
            if self.ddp_config.use_distributed_optimizer
            else self.data_parallel_group
        )

        # Keep pure non-Dion buckets on the stock DO reduce-scatter path and use
        # custom helpers only for mixed/Dion buckets.
        handles = []
        pure_non_dion_indices = []

        with stream_context:
            for idx, bucket in enumerate(self.buckets):
                if self.ddp_config.use_distributed_optimizer:
                    if self._is_pure_non_dion(bucket):
                        pure_non_dion_indices.append(idx)
                        continue
                    self._start_dion_bucket_rs(
                        bucket=bucket,
                        reduce_op=reduce_op,
                        async_op=async_op,
                        handles=handles,
                        communication_group=communication_group,
                    )
                    self._start_non_dion_bucket_rs(
                        idx=idx,
                        bucket=bucket,
                        reduce_op=reduce_op,
                        async_op=async_op,
                        handles=handles,
                        communication_group=communication_group,
                    )
                    _log_dion_target_main_grad_phase_(
                        bucket,
                        phase="start_grad_sync_after_rs_launch",
                        param_to_name=getattr(self, "param_to_name", None),
                    )
                    continue

                handle = torch.distributed.all_reduce(
                    bucket.grad_data,
                    op=reduce_op,
                    group=communication_group,
                    async_op=async_op,
                )
                if async_op and handle is not None:
                    handles.append(handle)

        if pure_non_dion_indices:
            with stream_context, _coalescing_manager(
                communication_group, async_ops=async_op
            ) as pure_cm:
                for idx in pure_non_dion_indices:
                    bucket = self.buckets[idx]
                    local_data_view = self._get_stock_local_grad_view(idx, bucket)
                    bucket.non_dion_grad_local_view = local_data_view
                    expected_bucket_input = local_data_view.numel() * communication_group.size()
                    if bucket.grad_data.numel() != expected_bucket_input:
                        raise RuntimeError(
                            "[Dion] Invalid pure non-Dion stock RS buffer size: "
                            f"bucket={bucket.bucket_id} input={bucket.grad_data.numel()} "
                            f"expected={expected_bucket_input} local={local_data_view.numel()} "
                            f"group={communication_group.size()}"
                        )
                    dist_reduce_scatter_func(
                        local_data_view,
                        bucket.grad_data,
                        op=reduce_op,
                        group=communication_group,
                        async_op=async_op,
                    )
            if async_op:
                handles.append(pure_cm)

        # With multiple DistOpt instances, we need to all-reduce across instances.
        if (
            self.ddp_config.use_distributed_optimizer
            and self.ddp_config.num_distributed_optimizer_instances > 1
        ):
            assert self.inter_distributed_optimizer_instance_group is not None
            # Create a new coalescing manager for the inter-instance all-reduce.
            with (
                stream_context,
                _coalescing_manager(
                    self.inter_distributed_optimizer_instance_group, async_ops=async_op
                ) as cm,
            ):
                for idx, bucket in enumerate(self.buckets):
                    dion_local_view = getattr(bucket, "dion_grad_local_view", None)
                    non_dion_local_view = getattr(bucket, "non_dion_grad_local_view", None)
                    self._all_reduce_inter_instance_local_views(
                        idx=idx,
                        bucket=bucket,
                        reduce_op=reduce_op,
                        async_op=async_op,
                    )

                    if (
                        not async_op
                        and os.getenv("DION_INTER_DIAG", "0") == "1"
                        and not getattr(bucket, "_dion_inter_diag_logged", False)
                    ):
                        fp = []
                        if dion_local_view is not None and dion_local_view.numel() > 0:
                            torch.cuda.synchronize(dion_local_view.device)
                            fp.append(
                                (
                                    "dion",
                                    tuple(dion_local_view.shape),
                                    float(dion_local_view.float().sum().item()),
                                    float(dion_local_view.float().abs().sum().item()),
                                    float((dion_local_view.float() ** 2).sum().item()),
                                    float(dion_local_view.float().abs().max().item()),
                                )
                            )
                        if non_dion_local_view is not None and non_dion_local_view.numel() > 0:
                            torch.cuda.synchronize(non_dion_local_view.device)
                            fp.append(
                                (
                                    "non_dion",
                                    tuple(non_dion_local_view.shape),
                                    float(non_dion_local_view.float().sum().item()),
                                    float(non_dion_local_view.float().abs().sum().item()),
                                    float((non_dion_local_view.float() ** 2).sum().item()),
                                    float(non_dion_local_view.float().abs().max().item()),
                                )
                            )
                        gathered = [None] * self.inter_distributed_optimizer_instance_group.size()
                        torch.distributed.all_gather_object(
                            gathered, fp, group=self.inter_distributed_optimizer_instance_group
                        )
                        logger.info(
                            "[DION_INTER_INSTANCE_DIAG] bucket=%s group=%s gathered=%s",
                            getattr(bucket, "bucket_id", -1),
                            torch.distributed.get_process_group_ranks(
                                self.inter_distributed_optimizer_instance_group
                            ),
                            gathered,
                        )
                        bucket._dion_inter_diag_logged = True
            # Add coalescing manager handle to the list
            if async_op and cm is not None:
                handles.append(cm)

        if async_op:
            # Use _HandleList to manage multiple async handles
            class _HandleList:
                def __init__(self, hs):
                    self.hs = [h for h in hs if h is not None]

                def wait(self):
                    for h in self.hs:
                        if hasattr(h, "wait"):
                            h.wait()

            self.grad_reduce_handle = _HandleList(handles)
        else:
            self.grad_reduce_handle = None

    def _flush_mixed_non_dion_rs_buffers(self):
        """Mixed non-Dion now follows stock DO local-shard contract; no flush is required."""
        return

    def _release_dion_grad_buffers(self):
        """
        Clean up Dion RS buffers after reduce-scatter completes.

        - dion_grad_buffer: Release reference (view of grad_data or separate allocation)
        - mixed non-Dion follows stock `bucket.grad_data` local-shard contract
        """
        for bucket in self.buckets:
            # Release dion_grad_buffer reference
            if hasattr(bucket, 'dion_grad_buffer'):
                bucket.dion_grad_buffer = None
            if hasattr(bucket, 'dion_rs_local_buffer'):
                bucket.dion_rs_local_buffer = None
            if hasattr(bucket, 'dion_grad_local_view'):
                bucket.dion_grad_local_view = None
            if hasattr(bucket, 'non_dion_grad_local_view'):
                bucket.non_dion_grad_local_view = None
            if hasattr(bucket, 'non_dion_grad_buffer'):
                bucket.non_dion_grad_buffer = None
            if hasattr(bucket, '_memory_reused'):
                bucket._memory_reused = False

    def finish_grad_sync(self):
        """
        Finishes grad sync (all-reduce or reduce-scatter) communication operations
        for all buckets in the bucket group.

        When ddp_config.overlap_grad_reduce is set to True, waits for asynchronous
        communication call to complete. When ddp_config.overlap_grad_reduce is set to False,
        makes synchronous call.
        """
        self.param_gather_dispatched = False
        pp_world_size = parallel_state.get_pipeline_model_parallel_world_size()
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if pp_world_size > 1:
            logger.info(
                "[DION_PP_DEBUG] rank=%s finish_grad_sync enter buckets=%s overlap=%s handle=%s",
                rank,
                [getattr(bucket, 'bucket_id', -1) for bucket in self.buckets],
                self.ddp_config.overlap_grad_reduce,
                self.grad_reduce_handle is not None,
            )
        # If overlap_grad_reduce is False, start (and finish) synchronous communication call here.
        if not self.ddp_config.overlap_grad_reduce:
            self.start_grad_sync()
            self._flush_mixed_non_dion_rs_buffers()
            for bucket in self.buckets:
                _log_dion_target_main_grad_phase_(
                    bucket,
                    phase="finish_grad_sync_sync_exit",
                    param_to_name=getattr(self, "param_to_name", None),
                )
            if pp_world_size > 1:
                logger.info("[DION_PP_DEBUG] rank=%s finish_grad_sync exit sync", rank)
            return
        # When using multiple DistOpt instances, we don't need to sync here as we launch
        # communications on a separate communication stream.
        if self.ddp_config.num_distributed_optimizer_instances > 1:
            if self.communication_stream is not None:
                torch.cuda.default_stream().wait_stream(self.communication_stream)
            if os.getenv("DION_POST_INTER_LOCAL_DIAG", "0") == "1":
                self._log_post_inter_local_views()
            self._flush_mixed_non_dion_rs_buffers()
            for bucket in self.buckets:
                _log_dion_target_main_grad_phase_(
                    bucket,
                    phase="finish_grad_sync_inter_exit",
                    param_to_name=getattr(self, "param_to_name", None),
                )
            if pp_world_size > 1:
                logger.info("[DION_PP_DEBUG] rank=%s finish_grad_sync exit inter-instance", rank)
            return
        assert self.grad_reduce_handle is not None, (
            f"Communication call has not been issued for this bucket "
            f"({len(self.params_with_grad)}/{len(self.params)} params have grad available)"
        )
        self.grad_reduce_handle.wait()
        self.grad_reduce_handle = None
        self._flush_mixed_non_dion_rs_buffers()
        for bucket in self.buckets:
            _log_dion_target_main_grad_phase_(
                bucket,
                phase="finish_grad_sync_async_exit",
                param_to_name=getattr(self, "param_to_name", None),
            )
        if pp_world_size > 1:
            logger.info("[DION_PP_DEBUG] rank=%s finish_grad_sync exit async", rank)

    def _log_post_inter_local_views(self):
        """Best-effort exact post-inter-instance local-view fingerprint check.

        This is a focused debug tool for multi-instance DO integration. It runs only
        after the default stream has waited on the communication stream, so the local
        optimizer shard views should already reflect the final stock-DO-equivalent RS/AR
        result for this step.
        """
        if not torch.distributed.is_initialized():
            return
        inter_group = getattr(self, "inter_distributed_optimizer_instance_group", None)
        if inter_group is None or inter_group.size() <= 1:
            return

        try:
            group_ranks = torch.distributed.get_process_group_ranks(inter_group)
        except Exception:
            group_ranks = None

        for bucket in self.buckets:
            dion_view = getattr(bucket, "dion_grad_local_view", None)
            if dion_view is not None and dion_view.numel() > 0:
                tensor = dion_view.detach().float()
                fp = (
                    "dion",
                    int(getattr(bucket, "bucket_id", -1)),
                    tuple(tensor.shape),
                    float(tensor.sum().item()),
                    float(tensor.abs().sum().item()),
                    float((tensor * tensor).sum().item()),
                    float(tensor.abs().max().item()),
                )
                gathered = [None] * inter_group.size()
                torch.distributed.all_gather_object(gathered, fp, group=inter_group)
                if any(item != fp for item in gathered):
                    logger.error(
                        "[DION_POST_INTER_LOCAL_MISMATCH] kind=dion bucket=%s group_ranks=%s gathered=%s",
                        getattr(bucket, "bucket_id", -1),
                        group_ranks,
                        gathered,
                    )
            non_dion_view = getattr(bucket, "non_dion_grad_local_view", None)
            if non_dion_view is not None and non_dion_view.numel() > 0:
                tensor = non_dion_view.detach().float()
                fp = (
                    "non_dion",
                    int(getattr(bucket, "bucket_id", -1)),
                    tuple(tensor.shape),
                    float(tensor.sum().item()),
                    float(tensor.abs().sum().item()),
                    float((tensor * tensor).sum().item()),
                    float(tensor.abs().max().item()),
                )
                gathered = [None] * inter_group.size()
                torch.distributed.all_gather_object(gathered, fp, group=inter_group)
                if any(item != fp for item in gathered):
                    logger.error(
                        "[DION_POST_INTER_LOCAL_MISMATCH] kind=non_dion bucket=%s group_ranks=%s gathered=%s",
                        getattr(bucket, "bucket_id", -1),
                        group_ranks,
                        gathered,
                    )

    def register_grad_ready(self, param: torch.nn.Parameter):
        """
        Registers grads for the passed-in param to be "ready" for grad sync.

        When the number of microbatches is greater than 1, we only want to register
        grads as ready when processing the last microbatch and ddp_config.overlap_grad_reduce
        is True.
        """
        assert (
            self.ddp_config.overlap_grad_reduce
        ), "register_grad_ready() should only be called when overlap_grad_reduce is True"
        if self.is_last_microbatch:
            assert param in self.param_to_bucket, "Param is not in the bucket group"
            assert param not in self.params_with_grad, "Cannot set grad twice"
            self.params_with_grad.add(param)
            # If all params in bucket group have grads available, issue communication call.
            if len(self.params_with_grad) == len(self.params):
                self.start_grad_sync()


class _ParamAndGradBuffer:
    """
    Groups parameters and gradients into a contiguous buffer, and then breaks the buffer into
    buckets with roughly `bucket_size` parameters each.

    Args:
        ddp_config: DistributedDataParallel config object.
        param_dtype: Type of param tensor.
        grad_dtype: Type of grad tensor.
        params: List of parameters whose parameters and gradients are collated in the underlying
            tensor.
        data_parallel_group: Data-parallel process group.
        bucket_size: The rough size of each bucket in terms of number of parameters.
        param_to_name: Mapping from `torch.nn.Parameter` to name (for logging purposes).
        gradient_scaling_factor: This factor is utilized to scale gradients prior to their
            communication. Its application is twofold: it facilitates the averaging of gradients
            and the scaling of gradients in the context of the Mixture of Experts (MoE) model.
        param_indices: The index of each param among the params with same dtype, if a param is fp8,
            use its "fake" high precision dtype to determine which params have same dtype with it.
            These indices are needed when loading a non-native-fp8 checkpoint in native-fp8 mode.
    """

    def __init__(
        self,
        ddp_config: DistributedDataParallelConfig,
        param_dtype: torch.dtype,
        grad_dtype: torch.dtype,
        params: List[torch.nn.Parameter],
        data_parallel_group: torch.distributed.ProcessGroup,
        bucket_size: int,
        param_to_name: Dict[torch.nn.Parameter, str],
        gradient_scaling_factor: float,
        param_indices: List[int],
        nccl_ub: bool,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):

        if pg_collection is None:
            self.dp_cp_group = parallel_state.get_data_and_context_parallel_group(
                with_context_parallel=True
            )
            self.tp_group = parallel_state.get_tensor_model_parallel_group()
        else:
            assert hasattr(pg_collection, 'tp') and hasattr(pg_collection, 'dp_cp')
            self.dp_cp_group = pg_collection.dp_cp
            self.tp_group = pg_collection.tp

        self.ddp_config = ddp_config
        self.params = params
        self.param_indices = param_indices

        # Check that params are unique.
        unique_params = set()
        for param in params:
            assert param not in unique_params
            unique_params.add(param)
        del unique_params

        # Store attributes that will be needed later.
        self.param_dtype = param_dtype
        self.grad_dtype = grad_dtype
        self.data_parallel_group = data_parallel_group
        self.data_parallel_world_size = self.data_parallel_group.size()
        self.gradient_scaling_factor = gradient_scaling_factor
        self.nccl_ub = nccl_ub
        self.param_to_name = param_to_name  # Store for Dion optimizer's _build_model_gbuf_range

        # Data structures to store underlying buckets and relevant indexing data.
        self.buckets = []
        self.param_to_bucket = {}  # Param -> bucket mapping.
        self.param_index_map = {}  # Param -> location in buffer mapping (used in dist. optimizer).
        self.param_data_cpu = None  # for offloading params to cpu

        def _pad(number_to_be_padded: int, divisor: int) -> int:
            return int(math.ceil(number_to_be_padded / divisor) * divisor)

        def _pad_end_of_bucket_if_needed(bucket_end_index: int) -> int:
            """
            Pads end index of bucket if using distributed optimizer (to ensure uniform sharding).
            """
            if self.ddp_config.use_distributed_optimizer:
                # Workaround for TE bug causing cuBLAS to pick an incompatible algorithm.
                # This also helps cuBLAS pick more efficient algorithms for GEMMs.
                # We now ensure that all buckets start at a memory address that is 256-byte
                # aligned (128 values since params and grads use >= 16-bit precision).
                if self.ddp_config.pad_buckets_for_high_nccl_busbw:
                    # Make sure the bucket size is divisible by a large power of 2 (2^16) to
                    # ensure NCCL collectives have high bus bandwidth at large DP counts,
                    # since NCCL message size (which for ring algorithms is bucket_size /
                    # dp_size) apparently needs to be divisible by a power of 2 for high busbw.
                    bucket_size_divisor = math.lcm(self.data_parallel_world_size, 128, 2**16)
                else:
                    bucket_size_divisor = math.lcm(self.data_parallel_world_size, 128)
                return _pad(bucket_end_index, bucket_size_divisor)
            return bucket_end_index

        def _pad_start_of_param_if_needed(param_start_index: int) -> int:
            """
            Pads start index of param if using distributed optimizer (to ensure "good" alignment).
            """
            if self.ddp_config.use_distributed_optimizer:
                # Ensure that params start at 128-byte aligned addresses (64 values
                # since params are >= 16-bit precision).
                return _pad(param_start_index, 64)
            return param_start_index

        # First, figure out how many elements should be in the underlying buffer storage.
        # Note that if we need to split the buffer into smaller buckets, each of these
        # might need to be padded as well (if using the distributed optimizer).
        param_start_index = 0
        bucket_start_index = param_start_index
        bucket_params = set()
        self.bucket_indices = []
        per_bucket_numel_unpadded = []
        bucket_id = 0

        def _update_bucket_metadata(param_end_index: int) -> int:
            """
            Record metadata for the bucket starting at bucket_start_index and ending with the
            passed-in param_end_index. Returns the bucket's end_index.
            """
            nonlocal bucket_start_index, bucket_params, bucket_id
            per_bucket_numel_unpadded.append(param_end_index - bucket_start_index)
            bucket_end_index = _pad_end_of_bucket_if_needed(param_end_index)

            # Record metadata of new bucket.
            self.bucket_indices.append((bucket_start_index, bucket_end_index))
            bucket_start_index = bucket_end_index

            # Prepare for next bucket.
            bucket_params = set()
            bucket_id += 1

            # Return the potentially padded bucket_end_index.
            return bucket_end_index

        def _does_param_require_new_bucket(param):
            """
            Split shared embedding parameters into separate bucket if using distributed
            optimizer that makes use of reduce-scatters instead of all-reduces.
            This ensures that the first and last pipeline stage partition optimizer state
            for the shared embedding parameters the same way across DP replicas, allowing
            the DP reduce-scatter to be before the embedding all-reduce.
            """
            return (
                getattr(param, "shared_embedding", False)
                and self.ddp_config.use_distributed_optimizer
            )

        for param in params[::-1]:
            # Iterate through parameters in reverse order to roughly follow backprop order.

            this_numel = param.data.nelement()
            param_start_index = _pad_start_of_param_if_needed(param_start_index)

            # Create bucket with collected parameters if current param needs its own bucket.
            if _does_param_require_new_bucket(param) and len(bucket_params) > 0:
                # Ensure this param accounts for the new padding introduced at end of
                # previous bucket.
                param_start_index = _update_bucket_metadata(param_start_index)

            param_end_index = param_start_index + this_numel
            self.param_index_map[param] = (param_start_index, param_end_index, bucket_id)
            bucket_params.add(param)

            # If we have enough elements already or the current param is part of the shared
            # embedding layer and needs a separate bucket, form a new bucket.
            if (
                bucket_size is not None and (param_end_index - bucket_start_index) >= bucket_size
            ) or _does_param_require_new_bucket(param):
                bucket_end_index = _update_bucket_metadata(param_end_index)
                param_start_index = bucket_end_index
            else:
                param_start_index = param_end_index

        # Add remaining params to a new bucket.
        if len(bucket_params) > 0:
            bucket_end_index = _update_bucket_metadata(param_end_index)

        # Next, create underlying storage for buffer (with numel elements that includes
        # padding as necessary).
        self.numel = bucket_end_index
        self.numel_unpadded = sum(per_bucket_numel_unpadded)
        assert self.numel_unpadded <= self.numel
        if self.ddp_config.use_distributed_optimizer:
            assert self.numel % self.data_parallel_world_size == 0
        else:
            assert self.numel == self.numel_unpadded

        self.param_data = None

        if self.nccl_ub:
            # If nccl_ub is True, use nccl_allocator to allocate memory for param_data/grad_data.
            nccl_allocator.init()
            pool = nccl_allocator.create_nccl_mem_pool(
                symmetric=not self.ddp_config.disable_symmetric_registration
            )
            mem_alloc_context = functools.partial(
                nccl_allocator.nccl_mem,
                pool,
                group=self.data_parallel_group,
                symmetric=not self.ddp_config.disable_symmetric_registration,
            )
        else:
            # If nccl_ub is False, mem_alloc_context is nullcontext.
            mem_alloc_context = nullcontext

        with mem_alloc_context():
            # For MXFP8 param: Create a shared buffer for param AG and grad RS for memory efficiency
            # The buffer is mapped to weight gradients whose dtype is either bf16 or FP32.
            # It can be temporarily reused by param AG.
            if self.ddp_config.use_distributed_optimizer and any(is_mxfp8tensor(p) for p in params):
                self.shared_buffer = torch.zeros(
                    self.numel,
                    dtype=self.grad_dtype,
                    device=torch.cuda.current_device(),
                    requires_grad=False,
                )
                # For FP32 weight grads, only half of the buffer is used to store params in bf16.
                if self.grad_dtype == torch.float32:
                    self.param_data = self.shared_buffer[: math.ceil(self.numel / 2)].view(
                        torch.bfloat16
                    )
                else:
                    self.param_data = self.shared_buffer
                self.grad_data = self.shared_buffer
            else:
                # Only re-map param tensors if using distributed optimizer.
                if self.ddp_config.use_distributed_optimizer:
                    self.param_data = torch.zeros(
                        self.numel,
                        dtype=self.param_dtype,
                        device=torch.cuda.current_device(),
                        requires_grad=False,
                    )
                self.grad_data = torch.zeros(
                    self.numel,
                    dtype=self.grad_dtype,
                    device=torch.cuda.current_device(),
                    requires_grad=False,
                )
        self.grad_data_size = 0

        # Finally, map param.data and param.main_grad fields to buffers.
        bucket_params = []
        bucket_start_index = 0
        cur_bucket_id = 0
        for param in params[::-1]:
            param_start_index, param_end_index, bucket_id = self.param_index_map[param]
            # For MXFP8 param: we only need to map weight gradients to the buffer.
            if not self.ddp_config.reuse_grad_buf_for_mxfp8_param_ag:
                # Assign param.data to appropriate segment of self.param_data.
                if self.param_data is not None:
                    new_param_data = self._get(
                        param.data.shape, param_start_index, buffer_type=BufferType.PARAM
                    )
                    if is_float8tensor(param):
                        modify_underlying_storage(param, new_param_data)
                    else:
                        old_param_data = param.data
                        param.data = new_param_data
                        assert old_param_data._base is None
                        # Copy tensor values (from initialization or checkpoint).
                        param.data.detach().copy_(old_param_data)
                        del old_param_data

            param.main_grad = self._get(
                param.data.shape, param_start_index, buffer_type=BufferType.GRAD
            )
            if bucket_id != cur_bucket_id:
                bucket_end_index = _pad_end_of_bucket_if_needed(param_start_index)
                self.buckets.append(
                    self._new_bucket(
                        bucket_params=bucket_params,
                        start_index=bucket_start_index,
                        end_index=bucket_end_index,
                        numel_unpadded=per_bucket_numel_unpadded[cur_bucket_id],
                        bucket_id=cur_bucket_id,
                    )
                )
                bucket_start_index = bucket_end_index
                bucket_params = []
                assert cur_bucket_id + 1 == len(self.buckets)
                assert bucket_id == cur_bucket_id + 1
                cur_bucket_id = bucket_id
            bucket_params.append(param)

        # Add remaining params to a new bucket.
        if len(bucket_params) > 0:
            bucket_end_index = _pad_end_of_bucket_if_needed(param_end_index)
            self.buckets.append(
                self._new_bucket(
                    bucket_params=bucket_params,
                    start_index=bucket_start_index,
                    end_index=bucket_end_index,
                    numel_unpadded=per_bucket_numel_unpadded[cur_bucket_id],
                    bucket_id=cur_bucket_id,
                )
            )

        # Log buckets for all PP stages.
        log_strs = []
        log_strs.append(
            f"Number of buckets for gradient all-reduce / reduce-scatter: {len(self.buckets)}"
        )
        for index, bucket in enumerate(self.buckets):
            numel = 0
            for param in bucket.params:
                numel += param.data.nelement()
            log_strs.append(
                f"Params for bucket {index + 1} ({numel} elements, "
                f"{bucket.grad_data.nelement()} padded size):"
            )
            for param in bucket.params:
                log_strs.append(f"\t{param_to_name[param]}")
        log_on_each_pipeline_stage(
            logger,
            logging.INFO,
            "\n".join(log_strs),
            tp_group=self.tp_group,
            dp_cp_group=self.dp_cp_group,
        )

    def scale_gradients(self, scaling_factor: float) -> None:
        """Scale the gradient data by `scaling_factor`."""
        self.grad_data *= scaling_factor
        for bucket in self.buckets:
            dion_local_view = getattr(bucket, "dion_grad_local_view", None)
            if dion_local_view is not None:
                dion_local_view.mul_(scaling_factor)

    def _get(self, shape: torch.Size, start_index: int, buffer_type: BufferType) -> torch.Tensor:
        """
        Return a tensor with the input `shape` as a view into the 1-D data starting at
        `start_index`.
        """
        end_index = start_index + shape.numel()
        assert end_index <= self.numel, "Requested tensor is out of buffer range"
        if buffer_type == BufferType.PARAM:
            assert self.param_data is not None
            buffer_tensor = self.param_data[start_index:end_index]
        elif buffer_type == BufferType.GRAD:
            buffer_tensor = self.grad_data[start_index:end_index]
        else:
            raise Exception("Illegal buffer type provided to GradBuffer._get() function")
        buffer_tensor = buffer_tensor.view(shape)
        return buffer_tensor

    def _new_bucket(
        self,
        bucket_params: List[torch.nn.Parameter],
        start_index: int,
        end_index: int,
        numel_unpadded: int,
        bucket_id: int,
    ) -> _ParamAndGradBucket:
        """
        Helper function that creates a new bucket. Also updates param->bucket mapping.
        """

        # Assert that indices are correctly padded (if needed), and that bucket
        # position is same as originally computed.
        if self.ddp_config.use_distributed_optimizer:
            assert start_index % self.data_parallel_world_size == 0
            assert end_index % self.data_parallel_world_size == 0
        assert (start_index, end_index) == self.bucket_indices[bucket_id]

        # Get appropriate view into global _ParamAndGradBuffer.
        bucketed_param_data = None
        if self.param_data is not None:
            bucketed_param_data = self._get(
                torch.Size([end_index - start_index]), start_index, buffer_type=BufferType.PARAM
            )
        bucketed_grad_data = self._get(
            torch.Size([end_index - start_index]), start_index, buffer_type=BufferType.GRAD
        )
        bucket = _ParamAndGradBucket(
            params=bucket_params,
            param_data=bucketed_param_data,
            grad_data=bucketed_grad_data,
            offset=start_index,
            numel_unpadded=numel_unpadded,
            gradient_scaling_factor=self.gradient_scaling_factor,
            bucket_id=bucket_id,
        )
        for bucket_param in bucket_params:
            assert bucket_param not in self.param_to_bucket
            self.param_to_bucket[bucket_param] = bucket

        return bucket

    def reset(self):
        """
        Zero out the underlying grad_buffer.
        """
        self.grad_data.zero_()

    def offload_to_cpu(self, move_params: bool = True, move_grads: bool = True):
        """
        Offload the buffers to CPU.
        """
        if move_grads and self.grad_data is not None and self.grad_data.storage().size() > 0:
            self.grad_data_size = self.grad_data.storage().size()
            self.grad_data.storage().resize_(0)
        if move_params and self.param_data is not None and self.param_data.storage().size() > 0:
            self.param_data_size = self.param_data.storage().size()
            if self.param_data_cpu is not None:
                self.param_data_cpu.copy_(self.param_data, non_blocking=True)
            else:
                self.param_data_cpu = self.param_data.cpu().pin_memory()
            self.param_data.storage().resize_(0)

    def reload_from_cpu(self, move_params: bool = True, move_grads: bool = True):
        """
        Reload the buffers from CPU.
        """
        if move_params and self.param_data is not None and self.param_data_cpu is not None and self.param_data.storage().size() == 0:
            self.param_data.storage().resize_(self.param_data_size)
            self.param_data.copy_(self.param_data_cpu, non_blocking=True)
        if move_grads and self.grad_data is not None and self.grad_data_size > 0:
            self.grad_data.storage().resize_(self.grad_data_size)
            self.grad_data.zero_()
            self.grad_data_size = 0


def partition_buckets(
    buffers: List[_ParamAndGradBuffer], force_single_bucket_group: bool = False
) -> List[_ParamAndGradBucketGroup]:
    """
    Automatically regroup the buckets of input buffers and return a list of bucket groups.

    In some scenarios, we need to put buckets from different buffers into a group so that their
    communication can be aggregated.

    For example, when there are both fp8 weights and bf16 biases in the model and virtual
    pipeline parallelism is enabled, each model chunk will have an fp8 bucket and a bf16 bucket,
    which doubles the number of communication kernels, and because of the use of
    CUDA_DEVICE_MAX_CONNECTIONS=1, having multiple back-to-back communications will prevent the
    overlap of communication kernels with computation kernels.

    The grouping strategy is:
    1. If force_single_bucket_group is True, put all buckets across all buffers into a single
       bucket group.
    2. If force_single_bucket_group is False, when there is no fp8 buffer in the input buffers,
       let each bucket group have only one bucket.
    3. If force_single_bucket_group is False, when using fp8 params, merge all non-fp8 buckets
       into the last fp8 bucket group.
       - Since the non-fp8 parameters (typically the biases of various layers) are relatively
         small, they are likely to be grouped into a single non-fp8 bucket.
       - The fp8 buckets start from the end of the model, i.e., the first bucket corresponds to
         the end of the model, while the last bucket corresponds to the beginning.
       - If we combine the non-fp8 bucket with the first fp8 bucket, we cannot initiate the
         reduce-scatter to synchronize gradients after the backward pass at the end of the model
         has completed. This is because we need to wait for the non-fp8 params from the beginning
         layers to obtain their gradients.
       - Combining the non-fp8 bucket with the last fp8 bucket can help avoid this issue.

    Args:
        buffers (list): list of input buffers.
        single_bucket_group_per_buffer (bool, optional): force group all buckets in each buffer
            into a single bucket group.
    """

    if len(buffers) == 0:
        return []

    dtype_to_buffer_map = {}
    for buffer in buffers:
        dtype = buffer.param_dtype
        # Make sure that the param_dtype of any two buffers is different.
        assert dtype not in dtype_to_buffer_map
        dtype_to_buffer_map[dtype] = buffer

    # Case 1: Put all buckets into a single bucket group if force_single_bucket_group is True.
    if force_single_bucket_group:
        buckets = []
        ddp_config = buffers[0].ddp_config
        data_parallel_group = buffers[0].data_parallel_group
        data_parallel_world_size = buffers[0].data_parallel_world_size
        for buffer in buffers:
            assert ddp_config == buffer.ddp_config
            assert data_parallel_group == buffer.data_parallel_group
            assert data_parallel_world_size == buffer.data_parallel_world_size
            buckets.extend(buffer.buckets)

        bucket_group = _ParamAndGradBucketGroup(
            buckets, ddp_config, data_parallel_group, data_parallel_world_size
        )
        return [bucket_group]

    if torch.uint8 not in dtype_to_buffer_map:
        # Case 2: When there is no fp8 buffer in the input buffers, let each bucket group have
        #         only one bucket.
        bucket_groups = []
        for buffer in buffers:
            for bucket in buffer.buckets:
                bucket_groups.append(
                    _ParamAndGradBucketGroup(
                        [bucket],
                        buffer.ddp_config,
                        buffer.data_parallel_group,
                        buffer.data_parallel_world_size,
                    )
                )
        return bucket_groups
    else:
        # Case 3: When using fp8 params, merge all non-fp8 buckets into the last fp8 bucket group.
        non_fp8_buckets = []
        for buffer in buffers:
            if buffer.param_dtype != torch.uint8:
                for bucket in buffer.buckets:
                    non_fp8_buckets.append(bucket)

        bucket_groups = []
        fp8_buffer = dtype_to_buffer_map[torch.uint8]
        for bucket in fp8_buffer.buckets:
            if len(bucket_groups) == len(fp8_buffer.buckets) - 1:
                # The last bucket group.
                group_buckets = [bucket] + non_fp8_buckets
            else:
                # The first N-1 bucket groups.
                group_buckets = [bucket]
            bucket_groups.append(
                _ParamAndGradBucketGroup(
                    group_buckets,
                    buffer.ddp_config,
                    buffer.data_parallel_group,
                    buffer.data_parallel_world_size,
                )
            )
        return bucket_groups
