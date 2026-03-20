# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import functools
import logging
import math
import os
import warnings
from contextlib import nullcontext
from dataclasses import dataclass
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
)
from ..utils import is_torch_min_version, log_on_each_pipeline_stage
from .distributed_data_parallel_config import DistributedDataParallelConfig
from .reduce_scatter_with_fp32_accumulation import reduce_scatter_with_fp32_accumulation

logger = logging.getLogger(__name__)

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


class _HandleGroup:
    """Aggregate multiple waitable handles behind a single `.wait()` surface."""

    def __init__(self, handles):
        self._handles = [handle for handle in handles if handle is not None]

    def wait(self):
        for handle in self._handles:
            if hasattr(handle, "wait"):
                handle.wait()
        self._handles = []


@dataclass
class DionGradState:
    """Ephemeral Dion grad transport state for one bucket in one step."""

    grad_buffer: torch.Tensor
    local_grad_view: Optional[torch.Tensor] = None
    shards: Optional[list[torch.Tensor]] = None


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
        param_to_index: Optional[Dict[torch.nn.Parameter, tuple[int, int]]] = None,
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
        if param_to_index is None:
            self.param_to_index = {}
            offset = 0
            for param in params:
                self.param_to_index[param] = (offset, offset + param.numel())
                offset += param.numel()
        else:
            self.param_to_index = param_to_index

        # Dion transport metadata.
        self.dion_shard_group = None
        self.dion_layout = None
        self.dion_state = None

    # Helper Properties

    @property
    def dion_param_ids(self):
        """Set of Dion param IDs from the bucket layout."""
        if self.dion_layout is None:
            return frozenset()
        return self.dion_layout.param_ids

    @property
    def has_dion_params(self) -> bool:
        """Whether bucket carries any Dion params."""
        return self.dion_layout is not None and self.dion_layout.has_params

    @property
    def has_non_dion_params(self) -> bool:
        """Whether bucket has non-Dion params."""
        dion_param_ids = self.dion_param_ids
        return any(id(param) not in dion_param_ids for param in self.params)

    @property
    def is_mixed(self) -> bool:
        """Whether bucket is mixed (has both Dion and non-Dion params)."""
        return self.has_dion_params and self.has_non_dion_params

    @property
    def is_pure_dion(self) -> bool:
        """Whether bucket is pure Dion (only Dion params)."""
        return self.has_dion_params and not self.has_non_dion_params


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
            self._debug_param_sync_state(bucket)

    def _debug_param_sync_state(self, bucket) -> None:
        """Log canonical full-param state after param sync for selected params."""
        if os.getenv("DION_DEBUG_SYNC_PARAMS", "0") != "1":
            return
        target_names = os.getenv("DION_DEBUG_HOOK_PARAMS", "")
        if not target_names:
            return
        targets = set(target_names.split(","))
        optimizer = getattr(bucket, "dion_optimizer", None)
        optimizer_name_map = getattr(optimizer, "param_to_name", None)
        selected_params = []
        for param in bucket.params_list:
            if optimizer_name_map is not None:
                param_name = optimizer_name_map.get(param, "")
            else:
                param_name = getattr(param, "_param_name", "")
            if param_name in targets:
                selected_params.append((param, param_name))
        if not selected_params:
            return

        log_cap = int(os.getenv("DION_DEBUG_SYNC_MAX_LOGS", "32"))
        log_count = getattr(self, "_dion_sync_debug_count", 0)
        if log_count >= log_cap:
            return
        self._dion_sync_debug_count = log_count + 1

        bucket_flat = bucket.param_data.detach().float().view(-1)
        logger.warning(
            "[DION_PARAM_SYNC_BUCKET] bucket_id=%s norm=%s sum=%s amax=%s numel=%s",
            int(getattr(bucket, "bucket_id", -1)),
            float(bucket_flat.norm().item()) if bucket_flat.numel() > 0 else 0.0,
            float(bucket_flat.sum().item()) if bucket_flat.numel() > 0 else 0.0,
            float(bucket_flat.abs().max().item()) if bucket_flat.numel() > 0 else 0.0,
            int(bucket_flat.numel()),
        )
        for param, param_name in selected_params:
            logical_name = ""
            if optimizer is not None and hasattr(optimizer, "_logical_param_name"):
                logical_name = optimizer._logical_param_name(param) or ""
            start, end = bucket.param_to_index[param]
            bucket_view = bucket.param_data.view(-1)[start:end].view(param.shape)
            param_flat = param.data.detach().float().view(-1)
            bucket_view_flat = bucket_view.detach().float().view(-1)
            logger.warning(
                "[DION_PARAM_SYNC_PARAM] param=%s logical_param=%s is_dion=%s data_norm=%s data_sum=%s "
                "data_amax=%s data_ptr=%s bucket_view_norm=%s bucket_view_sum=%s "
                "bucket_view_amax=%s bucket_view_ptr=%s",
                param_name,
                logical_name,
                bool(getattr(param, "is_dion_param", False)),
                float(param_flat.norm().item()) if param_flat.numel() > 0 else 0.0,
                float(param_flat.sum().item()) if param_flat.numel() > 0 else 0.0,
                float(param_flat.abs().max().item()) if param_flat.numel() > 0 else 0.0,
                int(param.data.data_ptr()),
                float(bucket_view_flat.norm().item()) if bucket_view_flat.numel() > 0 else 0.0,
                float(bucket_view_flat.sum().item()) if bucket_view_flat.numel() > 0 else 0.0,
                float(bucket_view_flat.abs().max().item()) if bucket_view_flat.numel() > 0 else 0.0,
                int(bucket_view.data_ptr()),
            )

    def _debug_finished_param_sync(self) -> None:
        """Log selected bucket/param state after a param-sync wait completes."""
        if os.getenv("DION_DEBUG_SYNC_PARAMS", "0") != "1":
            return
        for bucket in self.buckets:
            self._debug_param_sync_state(bucket)

    def _get_standard_local_grad_view(self, idx: int, bucket: _ParamAndGradBucket) -> torch.Tensor:
        """Return the canonical standard local optimizer shard for one bucket."""
        if self.cached_grad_buffer_shard_list[idx] is None:
            self.cached_grad_buffer_shard_list[idx] = shard_buffer(
                bucket.grad_data, self.intra_distributed_optimizer_instance_size
            )
        return self.cached_grad_buffer_shard_list[idx][
            self.intra_distributed_optimizer_instance_rank
        ]

    def _has_dion_state(self, bucket: _ParamAndGradBucket) -> bool:
        """Whether a bucket has active Dion grad state for this step."""
        dion_state = getattr(bucket, "dion_state", None)
        return bucket.has_dion_params and dion_state is not None

    def _has_dion_layout(self, bucket: _ParamAndGradBucket) -> bool:
        """Whether a bucket has Dion param-gather layout entries."""
        return bucket.has_dion_params

    def _group_has_dion_params(self, bucket_group) -> bool:
        """Whether a bucket-group contains Dion params that require the custom AG path."""
        if bucket_group is None:
            return False
        return any(bucket.has_dion_params for bucket in bucket_group.buckets)

    def _has_non_dion(self, bucket: _ParamAndGradBucket) -> bool:
        """Whether a bucket owns any non-Dion params."""
        return bucket.has_non_dion_params

    def _is_pure_non_dion(self, bucket: _ParamAndGradBucket) -> bool:
        """Whether a bucket should follow the standard DO RS path without Dion helpers."""
        return self._has_non_dion(bucket) and not bucket.has_dion_params

    def _collect_param_gather_launches(self, async_op: bool):
        """Collect custom Dion launches and standard-DO bucket indices for one dispatch."""
        pure_non_dion_indices = []
        custom_handles = []

        for idx, bucket in enumerate(self.buckets):
            has_dion = self._has_dion_layout(bucket)
            has_non_dion = self._has_non_dion(bucket)

            if has_dion:
                optimizer = getattr(bucket, "dion_optimizer", None)
                if optimizer is None:
                    raise RuntimeError(
                        f"[Dion] bucket {bucket.bucket_id} has Dion entries but no dion_optimizer"
                    )
                bucket_handle = optimizer._all_gather_bucket_params(
                    bucket,
                    async_op=async_op,
                )
                if bucket_handle is not None:
                    custom_handles.append(bucket_handle)

            if not has_non_dion:
                continue

            if has_dion:
                continue

            pure_non_dion_indices.append(idx)

        return pure_non_dion_indices, custom_handles

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
        has_dion = self._has_dion_state(bucket)
        if not has_dion:
            bucket.dion_state = None
            return

        dion_rs_group = getattr(bucket, "dion_shard_group", None) or communication_group
        dion_layout = bucket.dion_layout
        dion_shard_size = 0 if dion_layout is None else int(dion_layout.shard_size)
        dion_state = bucket.dion_state
        grad_buffer = dion_state.grad_buffer
        expected_input = dion_shard_size * dion_rs_group.size()
        if grad_buffer is None or grad_buffer.numel() != expected_input:
            raise RuntimeError(
                f"[Dion] Invalid Dion RS buffer size: bucket={bucket.bucket_id} "
                f"input={0 if grad_buffer is None else grad_buffer.numel()} "
                f"expected={expected_input} shard={dion_shard_size} group={dion_rs_group.size()}"
            )

        dion_shards = dion_state.shards
        if (
            dion_shards is None
            or len(dion_shards) != dion_rs_group.size()
            or dion_shards[0].numel() * dion_rs_group.size() != grad_buffer.numel()
        ):
            dion_shards = shard_buffer(grad_buffer, dion_rs_group.size())
            dion_state.shards = dion_shards
        dion_rs_output = dion_shards[dion_rs_group.rank()]
        dion_state.local_grad_view = dion_rs_output

        if dion_rs_group.size() <= 1:
            dion_rs_output.copy_(grad_buffer[:dion_shard_size])
            return

        handle = dist_reduce_scatter_func(
            dion_rs_output,
            grad_buffer,
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
        """Launch the canonical standard local-shard RS path for pure/mixed non-Dion grads."""
        has_non_dion = self._has_non_dion(bucket)
        standard_local_grad_view = self._get_standard_local_grad_view(idx, bucket)

        if not has_non_dion:
            if self._has_dion_state(bucket):
                standard_local_grad_view.zero_()
            return

        expected_bucket_input = standard_local_grad_view.numel() * communication_group.size()
        if bucket.grad_data.numel() != expected_bucket_input:
            raise RuntimeError(
                f"[Dion] Invalid standard RS buffer size: bucket={bucket.bucket_id} "
                f"input={bucket.grad_data.numel()} expected={expected_bucket_input} "
                f"local={standard_local_grad_view.numel()} group={communication_group.size()}"
            )
        handle = dist_reduce_scatter_func(
            standard_local_grad_view,
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
        dion_state = getattr(bucket, "dion_state", None)
        dion_local_view = None if dion_state is None else dion_state.local_grad_view
        has_non_dion = self._has_non_dion(bucket)
        non_dion_local_view = self._get_standard_local_grad_view(idx, bucket) if has_non_dion else None

        if dion_local_view is not None and dion_local_view.numel() > 0:
            torch.distributed.all_reduce(
                dion_local_view,
                op=reduce_op,
                group=self.inter_distributed_optimizer_instance_group,
                async_op=async_op,
            )

        if not has_non_dion:
            return

        if non_dion_local_view is not None and non_dion_local_view.numel() > 0:
            torch.distributed.all_reduce(
                non_dion_local_view,
                op=reduce_op,
                group=self.inter_distributed_optimizer_instance_group,
                async_op=async_op,
            )

    def _clear_dion_bucket_grad_spans(self, bucket: _ParamAndGradBucket) -> None:
        """Clear Dion-owned spans from canonical bucket.grad_data after transport staging.

        This preserves the stock DO non-Dion reduce-scatter path on bucket.grad_data
        without allocating a second full-bucket temporary. Dion local reduced shards
        are written back in `_materialize_dion_local_grads()`.
        """
        dion_layout = getattr(bucket, "dion_layout", None)
        if dion_layout is None or not dion_layout.entries:
            return

        bucket_flat = bucket.grad_data.view(-1)
        for entry in dion_layout.entries:
            start = int(entry.canonical_bucket_start)
            end = int(entry.canonical_bucket_end)
            if end <= start:
                continue
            bucket_flat[start:end].zero_()

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

        if force_sync:
            if self.param_gather_handle is not None:
                self.param_gather_handle.wait()
                self.param_gather_handle = None
            self._mark_dion_param_sync_ready(False)
            async_op = False
            pure_non_dion_indices, _ = self._collect_param_gather_launches(async_op=False)
            if pure_non_dion_indices:
                with _coalescing_manager(
                    self.intra_distributed_optimizer_instance_group, async_ops=False
                ):
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
                            async_op=False,
                        )
            self._check_dion_param_sync_ready()
            return
        else:
            assert self.param_gather_handle is None

        async_op = self.ddp_config.overlap_param_gather and not force_sync
        self._mark_dion_param_sync_ready(False)
        pure_non_dion_indices, custom_handles = self._collect_param_gather_launches(async_op=async_op)
        standard_handle = None
        if pure_non_dion_indices:
            with _coalescing_manager(
                self.intra_distributed_optimizer_instance_group, async_ops=async_op
            ) as cm:
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
            standard_handle = cm if async_op else None

        if async_op:
            self.param_gather_handle = _HandleGroup([standard_handle, *custom_handles])
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

        if self.param_gather_handle is not None:
            self.param_gather_handle.wait()
            self.param_gather_handle = None

        self._check_dion_param_sync_ready()
        self._debug_finished_param_sync()

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

        # Build Dion shard buffers with row-split/col-split separation
        for bucket in self.buckets:
            if not bucket.has_dion_params:
                continue

            fs_group = getattr(bucket, 'dion_shard_group', None)
            fs_size = fs_group.size() if fs_group else 1
            dion_layout = bucket.dion_layout
            dion_shard_size = 0 if dion_layout is None else int(dion_layout.shard_size)

            if dion_shard_size <= 0:
                continue

            shard_group_size = fs_size

            ready_entries = []
            missing_main_grad = 0
            for entry in dion_layout.entries:
                entry_param = entry.param
                if entry_param is None:
                    continue
                if not hasattr(entry_param, "main_grad") or entry_param.main_grad is None:
                    missing_main_grad += 1
                    continue
                ready_entries.append((entry, entry_param))

            if missing_main_grad > 0:
                raise RuntimeError(
                    "[Dion] bucket has Dion entries with missing main_grad during RS shard buffering "
                    f"bucket_id={bucket.bucket_id} missing_main_grad={missing_main_grad}"
                )

            total_buffer_size = dion_shard_size * shard_group_size
            # IMPORTANT: Always allocate separate buffer.
            # Reusing grad_data causes overlap between RS input and output
            # (`grad_buffer` = grad_data[:N*dion_shard_size], RS output = grad_data[:dion_shard_size]),
            # which corrupts NCCL reduce_scatter results.
            grad_buffer = torch.zeros(
                total_buffer_size,
                dtype=bucket.grad_data.dtype,
                device=bucket.grad_data.device,
            )
            bucket.dion_state = DionGradState(grad_buffer=grad_buffer)

            if os.getenv("DION_SYNC_BEFORE_RS_COPY", "0") == "1":
                torch.cuda.synchronize()

            for entry, bucket_param in ready_entries:
                grad = bucket_param.main_grad.data
                shard_offset = entry.shard_offset
                size_per_rank = entry.size_per_rank
                fs_split_dim = entry.fs_split_dim
                global_shape = entry.global_shape
                local_shape = entry.local_shape
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
                optimizer = getattr(bucket, "dion_optimizer", None)
                logical_name = ""
                if optimizer is not None and hasattr(optimizer, "_logical_param_name"):
                    logical_name = optimizer._logical_param_name(bucket_param) or ""
                param_name = getattr(bucket_param, "_param_name", "")
                debug_targets = {
                    item.strip()
                    for item in os.getenv("DION_DEBUG_HOOK_PARAMS", "").split(",")
                    if item.strip()
                }
                should_debug = bool(debug_targets) and (
                    param_name in debug_targets
                    or logical_name in debug_targets
                    or any(logical_name.endswith(target) for target in debug_targets)
                )
                if should_debug:
                    logger.warning(
                        "[DION_PACK_MAIN_GRAD] bucket=%s param=%s logical_param=%s grad_sum=%s grad_norm=%s fs_split_dim=%s size_per_rank=%s global_shape=%s local_shape=%s shard_offset=%s",
                        bucket.bucket_id,
                        param_name,
                        logical_name,
                        float(grad_2d.detach().float().sum().item()),
                        float(grad_2d.detach().float().norm().item()),
                        int(fs_split_dim),
                        int(size_per_rank),
                        tuple(int(x) for x in global_shape),
                        tuple(int(x) for x in local_shape),
                        int(shard_offset),
                    )
                # Scale the buffered Dion shard before reduce-scatter.
                is_expert_bucket = not getattr(bucket_param, 'allreduce', True)
                if is_expert_bucket:
                    scale = bucket.gradient_scaling_factor
                else:
                    if self.ddp_config.average_in_collective:
                        scale = 1.0
                    else:
                        scale = bucket.gradient_scaling_factor

                for rank_j in range(shard_group_size):
                    fs_pos = rank_j
                    start_idx = fs_pos * size_per_rank
                    end_idx = min(start_idx + size_per_rank, m if fs_split_dim == 0 else n)

                    if fs_split_dim == 0:
                        seg_2d = grad_2d[start_idx:end_idx, :]
                    else:
                        seg_2d = grad_2d[:, start_idx:end_idx]

                    if seg_2d.numel() == 0:
                        continue

                    buf_offset = rank_j * dion_shard_size + shard_offset
                    out = grad_buffer[buf_offset:buf_offset + seg_2d.numel()].view_as(seg_2d)
                    out.copy_(seg_2d)
                    if should_debug:
                        logger.warning(
                            "[DION_PACK_MAIN_GRAD_SEG] bucket=%s param=%s logical_param=%s rank_j=%s seg_sum=%s seg_norm=%s start_idx=%s end_idx=%s buf_offset=%s",
                            bucket.bucket_id,
                            param_name,
                            logical_name,
                            int(rank_j),
                            float(seg_2d.detach().float().sum().item()),
                            float(seg_2d.detach().float().norm().item()),
                            int(start_idx),
                            int(end_idx),
                            int(buf_offset),
                        )
                    if scale != 1.0:
                        out.mul_(scale)

            self._clear_dion_bucket_grad_spans(bucket)

        # For mixed buckets, non-Dion params follow the standard Megatron-Core DO path:
        # TE/backward hook accumulates into canonical `param.main_grad`, and the bucket
        # participates in the standard bucket.grad_data RS/AR lifecycle below. No custom
        # mixed non-Dion shard buffering is performed here.

        # gradient_scaling_factor already takes into account whether we are computing
        # an average or sum in the data-parallel collective.
        for bucket in self.buckets:
            if bucket.gradient_scaling_factor != 1.0:
                has_dion = bucket.has_dion_params
                has_non_dion = self._has_non_dion(bucket)
                if has_dion and not has_non_dion:
                    # Pure Dion bucket: skip, Dion shard buffering already pre-scaled.
                    pass
                else:
                    # Pure non-Dion buckets and mixed non-Dion spans follow the standard
                    # bucket.grad_data scaling path directly. Mixed buckets already have
                    # their Dion-owned spans cleared above.
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

        # Keep pure non-Dion buckets on the standard DO reduce-scatter path and use
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
                    local_data_view = self._get_standard_local_grad_view(idx, bucket)
                    expected_bucket_input = local_data_view.numel() * communication_group.size()
                    if bucket.grad_data.numel() != expected_bucket_input:
                        raise RuntimeError(
                            "[Dion] Invalid pure non-Dion standard RS buffer size: "
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
                    self._all_reduce_inter_instance_local_views(
                        idx=idx,
                        bucket=bucket,
                        reduce_op=reduce_op,
                        async_op=async_op,
                    )
            # Add coalescing manager handle to the list
            if async_op and cm is not None:
                handles.append(cm)

        if async_op:
            # Use _HandleList to manage multiple async handles
            self.grad_reduce_handle = _HandleGroup(handles)
        else:
            self.grad_reduce_handle = None

    def _materialize_dion_local_grads(self):
        """Write reduced Dion local shards back into canonical model-side main_grad."""
        for bucket in self.buckets:
            dion_state = getattr(bucket, "dion_state", None)
            dion_layout = getattr(bucket, "dion_layout", None)
            if dion_state is None or dion_layout is None or dion_state.local_grad_view is None:
                continue

            local_grad_view = dion_state.local_grad_view
            for entry in dion_layout.entries:
                model_param = entry.param
                model_grad = getattr(model_param, "main_grad", None)
                if model_grad is None:
                    raise RuntimeError(
                        "[Dion] missing canonical main_grad while materializing Dion local shard "
                        f"bucket={bucket.bucket_id} param={getattr(model_param, '_param_name', id(model_param))}"
                    )

                local_numel = int(entry.local_numel)
                shard_offset = int(entry.shard_offset)
                if shard_offset + local_numel > local_grad_view.numel():
                    raise RuntimeError(
                        "[Dion] Dion local grad slice exceeded reduced shard buffer "
                        f"bucket={bucket.bucket_id} shard_offset={shard_offset} "
                        f"local_numel={local_numel} local_buffer_numel={int(local_grad_view.numel())}"
                    )

                local_shard = local_grad_view[shard_offset : shard_offset + local_numel].view(
                    tuple(int(dim) for dim in entry.local_shape)
                )
                model_grad_2d = model_grad.view(model_param.shape)
                optimizer = getattr(bucket, "dion_optimizer", None)
                logical_name = ""
                if optimizer is not None and hasattr(optimizer, "_logical_param_name"):
                    logical_name = optimizer._logical_param_name(model_param) or ""
                param_name = getattr(model_param, "_param_name", "")
                debug_targets = {
                    item.strip()
                    for item in os.getenv("DION_DEBUG_HOOK_PARAMS", "").split(",")
                    if item.strip()
                }
                should_debug = bool(debug_targets) and (
                    param_name in debug_targets
                    or logical_name in debug_targets
                    or any(logical_name.endswith(target) for target in debug_targets)
                )
                if should_debug:
                    model_grad_float = model_grad_2d.detach().float()
                    local_shard_float = local_shard.detach().float()
                    logger.warning(
                        "[DION_MATERIALIZE_LOCAL_GRAD] phase=before bucket=%s param=%s logical_param=%s model_grad_sum=%s local_shard_sum=%s fs_split_dim=%s range=[%s:%s]",
                        bucket.bucket_id,
                        param_name,
                        logical_name,
                        float(model_grad_float.sum().item()),
                        float(local_shard_float.sum().item()),
                        int(entry.fs_split_dim),
                        int(entry.start_idx),
                        int(entry.end_idx),
                    )
                if entry.fs_split_dim == 0:
                    model_grad_2d[entry.start_idx : entry.end_idx, :].copy_(local_shard)
                else:
                    model_grad_2d[:, entry.start_idx : entry.end_idx].copy_(local_shard)
                if should_debug:
                    logger.warning(
                        "[DION_MATERIALIZE_LOCAL_GRAD] phase=after bucket=%s param=%s logical_param=%s model_grad_sum=%s local_shard_sum=%s fs_split_dim=%s range=[%s:%s]",
                        bucket.bucket_id,
                        param_name,
                        logical_name,
                        float(model_grad_2d.detach().float().sum().item()),
                        float(local_shard.detach().float().sum().item()),
                        int(entry.fs_split_dim),
                        int(entry.start_idx),
                        int(entry.end_idx),
                    )

            bucket.dion_state = None

    def finish_grad_sync(self):
        """
        Finishes grad sync (all-reduce or reduce-scatter) communication operations
        for all buckets in the bucket group.

        When ddp_config.overlap_grad_reduce is set to True, waits for asynchronous
        communication call to complete. When ddp_config.overlap_grad_reduce is set to False,
        makes synchronous call.
        """
        self.param_gather_dispatched = False
        # If overlap_grad_reduce is False, start (and finish) synchronous communication call here.
        if not self.ddp_config.overlap_grad_reduce:
            self.start_grad_sync()
            return
        # When using multiple DistOpt instances, we don't need to sync here as we launch
        # communications on a separate communication stream.
        if self.ddp_config.num_distributed_optimizer_instances > 1:
            if self.communication_stream is not None:
                torch.cuda.default_stream().wait_stream(self.communication_stream)
            return
        assert self.grad_reduce_handle is not None, (
            f"Communication call has not been issued for this bucket "
            f"({len(self.params_with_grad)}/{len(self.params)} params have grad available)"
        )
        self.grad_reduce_handle.wait()
        self.grad_reduce_handle = None

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
            dion_state = getattr(bucket, "dion_state", None)
            dion_local_view = None if dion_state is None else dion_state.local_grad_view
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
        bucket_param_to_index = {}
        for bucket_param in bucket_params:
            param_start_index, param_end_index, _ = self.param_index_map[bucket_param]
            bucket_param_to_index[bucket_param] = (
                param_start_index - start_index,
                param_end_index - start_index,
            )
        bucket = _ParamAndGradBucket(
            params=bucket_params,
            param_data=bucketed_param_data,
            grad_data=bucketed_grad_data,
            offset=start_index,
            numel_unpadded=numel_unpadded,
            gradient_scaling_factor=self.gradient_scaling_factor,
            bucket_id=bucket_id,
            param_to_index=bucket_param_to_index,
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
