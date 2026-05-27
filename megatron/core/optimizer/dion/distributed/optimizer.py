"""
Distributed optimizer wrapper for Dion optimizer in Megatron-LM.

Supports orthogonal TP x FS sharding:
- tp_shard_dim=0 (ColumnParallel): FS shards cols
- tp_shard_dim=1 (RowParallel): FS shards rows
"""

import logging
from dataclasses import replace
import torch
import torch.distributed as dist
from typing import Dict, List, Tuple, Optional

from ...matrix.distrib_optimizer import DistributedMatrixOptimizer
from ..algorithm import MegatronDion
from ..backend import DionBackend
from ..types import (
    DionBatch,
    DionDistMeta,
    DionParamConfig,
    DionStepParam,
)
from ...matrix.types import MatrixBucketLayout
from ..state import (
    build_param_config,
    init_q_state,
    init_param_state,
    is_p_fs_sharded,
    is_p_tp_sharded,
    require_2d_local_shape,
)
from ..params import mark_dion_bucket_params
from ...matrix.splits.linear import (
    iter_linear_child_kinds,
    linear_child_global_shape,
    linear_child_has_local_overlap,
    linear_child_local_shape,
    linear_child_name,
    linear_child_param_uid,
    linear_state_key,
    read_linear_child,
    resolve_linear_split_rows,
    write_linear_child_,
)
from ...matrix.splits.qkv import (
    copy_qkv_split_metadata,
    extract_qkv_child,
    iter_qkv_child_kinds,
    qkv_child_global_shape,
    qkv_child_has_local_overlap,
    qkv_child_local_shape,
    qkv_child_name,
    qkv_child_param_uid,
    qkv_child_row_range,
    qkv_state_key,
    resolve_qkv_split_shapes,
    scatter_qkv_child_,
)
from ...matrix.splits.qkvg import (
    copy_qkvg_split_metadata,
    extract_qkvg_child,
    iter_qkvg_child_kinds,
    qkvg_child_global_shape,
    qkvg_child_has_local_overlap,
    qkvg_child_local_shape,
    qkvg_child_name,
    qkvg_child_param_uid,
    qkvg_child_row_range,
    qkvg_state_key,
    resolve_qkvg_split_shapes,
    scatter_qkvg_child_,
)
from ..utils import get_global_shape, get_local_shape, local_expert_tensor_view
from ...optimizer import param_group_identifier_keys
from ...matrix.parameter import (
    all_gather_bucket_params_,
    build_bucket_param_map,
    check_bucket_param_views,
    init_matrix_bucket,
    init_standard_bucket,
    build_matrix_shard_entries,
    assert_shard_aliased,
    collect_matrix_bucket_params,
    restore_matrix_shards_from_bucket_,
    resolve_grad_rank_to_fs_rank,
    set_bucket_param_views,
)
from .batches import build_dion_batches
from ...matrix.checkpoint_io import (
    copy_main_params_to_model_shards,
    build_matrix_checkpoint_metadata,
    build_distributed_checkpoint_state,
    copy_model_params_to_main_shards,
    resolve_matrix_checkpoint_sharding_type,
    restore_distributed_checkpoint_state,
    split_distributed_checkpoint_state,
    validate_matrix_checkpoint_metadata,
)
from ...matrix.gradients import finish_bucket_group_grad_sync, set_optimizer_shard_grads
from ...matrix.grad_norm import (
    compute_grad_norm,
    grad_norm_inputs,
)
from .row_child import resolve_row_child_layout
from .split_child import build_split_child_dist_meta
from .bootstrap import (
    build_q_init,
    enable_distributed_dion,
    ensure_optimizer_state,
    make_group_broadcast,
    resolve_base_training_seed,
    sync_q_replicas,
    should_use_distributed_dion_update,
    validate_enabled_rp_topology,
)
from .dist_meta import (
    add_standard_metas,
    assert_same_group_ranks,
    build_param_dist_meta,
    build_all_dist_metas,
    get_expected_expert_fs_group,
    validate_dist_meta_uids,
)
from ...matrix.sharding import (
    compute_fs_shard_range,
    create_fs_shard,
    attach_fs_shard_,
    register_matrix_shard,
    resolve_fs_group_from_meta,
    resolve_ortho_group,
    resolve_tp_group,
)
from .... import parallel_state, tensor_parallel
from ....fp8_utils import is_float8tensor, quantize_param_shard
from ....transformer.fsdp_dtensor_checkpoint import get_expert_index_from_key

logger = logging.getLogger(__name__)


def _clear_split_child_q_sync(parent_state, sync_key: str) -> None:
    parent_state[sync_key] = False


def _make_split_child_q_sync_hook(parent_state, sync_key: str):
    def hook() -> None:
        _clear_split_child_q_sync(parent_state, sync_key)

    return hook


def _expert_split_token(dist_meta: DionDistMeta) -> tuple[str, int]:
    if getattr(dist_meta, "per_expert_global_shape", None) is None:
        return ("expert", -1)
    param_name = getattr(dist_meta, "param_name", "") or ""
    try:
        expert_index = get_expert_index_from_key(param_name) if param_name else None
    except AssertionError:
        expert_index = None
    if expert_index is not None:
        return ("expert", int(expert_index))
    local_expert_index = int(getattr(dist_meta, "local_expert_index", -1))
    if local_expert_index < 0:
        raise RuntimeError(
            "[DION_EXPERT_SPLIT_MISSING_EXPERT_IDENTITY] "
            f"param_uid={getattr(dist_meta, 'param_uid', None)} "
            f"param_name={getattr(dist_meta, 'param_name', '')}"
        )
    return ("local_expert", local_expert_index)


def _with_expert_token(parent_uid, expert_token: tuple[str, int]):
    if parent_uid is None:
        raise RuntimeError("[DION_EXPERT_SPLIT_UID_REQUIRES_PARENT_UID]")
    if expert_token[1] < 0:
        return parent_uid
    if isinstance(parent_uid, tuple):
        return (*parent_uid, expert_token)
    return (parent_uid, expert_token)


def _has_active_expert_layout(dist_meta: DionDistMeta) -> bool:
    return (
        getattr(dist_meta, "per_expert_global_shape", None) is not None
        and int(getattr(dist_meta, "expert_axis", -1)) in (0, 1)
        and int(getattr(dist_meta, "num_local_experts", 1)) > 1
    )


_CHILD_GROUP_CACHE: dict[tuple[int, ...], Optional[torch.distributed.ProcessGroup]] = {}


def _ensure_child_group(
    ranks: tuple[int, ...],
    *,
    create_group: bool,
) -> Optional[torch.distributed.ProcessGroup]:
    """Return a cached child-specific subgroup for split optimizer-only children."""
    if len(ranks) <= 1:
        return None
    if ranks in _CHILD_GROUP_CACHE:
        return _CHILD_GROUP_CACHE[ranks]
    if not create_group:
        raise RuntimeError(
            "[DION_SPLIT_CHILD_GROUP_NOT_PREPARED] "
            f"ranks={tuple(int(rank) for rank in ranks)}"
        )
    group = parallel_state.create_group(
        list(ranks),
        use_local_synchronization=True,
        group_desc="DION_SPLIT_CHILD_GROUP",
    )
    _CHILD_GROUP_CACHE[ranks] = group
    return group


class DistributedDionOptimizer(DistributedMatrixOptimizer):
    """
    Dion-specific adapter on top of the generic distributed matrix optimizer.

    Architecture:
    - True parameter sharding: Only local shards stored on GPU
    - Buffer sizes reduced to shard size (not full size)
    - Bucket-wise all-gather/reduce-scatter via standard Megatron-Core DO
    - FSDP-style memory efficiency: Full params only during forward/backward

    Extends DistributedMatrixOptimizer to support Dion's 2D parallelism model:
    - RP (Replicate Process): Gradient averaging across replicas with same FS shard
    - FS (Fully Shard): True row-wise sharding for 2D params (GPU memory saved)
    - TP (Tensor Parallel): Column-wise tensor sharding

    Provides automatic configuration of 2D process groups and FS-aware annotation.
    """

    def _resolve_dion_tp_group(self):
        """Return the TP group used by Dion math and optimizer state."""
        group = getattr(self, "_dion_tp_group", None)
        if group is None:
            group = (
                parallel_state.get_expert_tensor_parallel_group(check_initialized=False)
                if getattr(self, "_is_expert_dion", False)
                else parallel_state.get_tensor_model_parallel_group(check_initialized=False)
            )
        if group is not None:
            self._assert_group_excludes_context_parallel(group, label="tp_group")
        return group

    def _dion_checkpoint_topology_signature(self) -> dict:
        tp_group = self._resolve_dion_tp_group()
        return self._matrix_checkpoint_topology_signature(
            fs_group=getattr(self, "fs_group", None),
            tp_group=tp_group,
            rp_group=getattr(self, "rp_group", None),
            state_replica_group=getattr(self, "state_replica_group", None),
        )

    @classmethod
    def _bucket_fs_get_group_size_rank(cls, param_and_grad_buffer, bucket) -> Tuple[object, int, int]:
        """Return the authoritative FS topology for Dion math on one bucket."""
        is_expert_bucket = any(not getattr(param, "allreduce", True) for param in bucket.params)
        if is_expert_bucket:
            fs_group, fs_size, fs_rank = cls._bucket_shard_get_group_size_rank(
                param_and_grad_buffer, bucket
            )
        else:
            fs_group = getattr(param_and_grad_buffer, "dion_fs_group", None)
            fs_size = getattr(param_and_grad_buffer, "dion_fs_size", None)
            fs_rank = getattr(param_and_grad_buffer, "dion_fs_rank", None)
            if fs_size is None or fs_rank is None:
                fs_group, fs_size, fs_rank = cls._bucket_shard_get_group_size_rank(
                    param_and_grad_buffer, bucket
                )
        if fs_size <= 0:
            raise RuntimeError(
                f"[Dion] invalid FS group size for bucket {bucket.bucket_id}: {fs_size}"
            )
        return fs_group, fs_size, fs_rank

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
        return build_bucket_param_map(
            cls,
            parent_result,
            ordered_params,
            dp_group,
            dp_rank,
            bucket_index,
            param_index_map,
            bucket_offset,
            bucket_size,
            bucket_param_to_index=bucket_param_to_index,
            param_to_name=param_to_name,
        )

    @classmethod
    def _mark_dion_bucket_params(cls, param_map, param_to_name, fs_size):
        """Classify bucket params and build static Dion metadata once."""
        return mark_dion_bucket_params(param_map, param_to_name, fs_size)

    def _init_groups(self) -> None:
        """Resolve the standard runtime groups that Dion math consumes."""
        self.validation_data_parallel_group = (
            self._pure_data_parallel_group
            if self._pure_data_parallel_group is not None
            else self.data_parallel_group
        )
        if self._dion_fs_group is not None:
            self.fs_group = self._dion_fs_group
        elif self._is_expert_dion:
            self.fs_group = self.data_parallel_group
        else:
            self.fs_group = None
        self.shard_group = self.data_parallel_group
        self.fs_size = 1 if self.fs_group is None else self._group_size(self.fs_group)
        self.fs_rank = 0 if self.fs_group is None else self._group_rank(self.fs_group)

        configured_fs_size = int(self._fs_size) if self._fs_size is not None else 1
        if self.fs_size != configured_fs_size:
            raise RuntimeError(
                f"[Dion] Global rank {self._global_rank}: "
                f"FS size mismatch configured={configured_fs_size} actual={self.fs_size}"
            )

        configured_rp_size = int(self._rp_size) if self._rp_size is not None else 1
        override_rp_group = self._replica_group
        if override_rp_group is not None and self._group_size(override_rp_group) <= 1:
            override_rp_group = None

        if override_rp_group is not None:
            override_rp_size = self._group_size(override_rp_group)
            if override_rp_size != configured_rp_size:
                rp_group_ranks = dist.get_process_group_ranks(override_rp_group)
                raise RuntimeError(
                    "[Dion] RP override topology mismatch before runtime init: "
                    f"configured_rp_size={configured_rp_size} actual_rp_group_size={override_rp_size} "
                    f"rp_group_ranks={rp_group_ranks}"
                )
            self.rp_group = override_rp_group
        else:
            if configured_rp_size <= 1:
                self.rp_group = None
            else:
                try:
                    runtime_rp_group = parallel_state.get_inter_distributed_optimizer_instance_group(
                        check_initialized=False
                    )
                except Exception:
                    runtime_rp_group = None
                if runtime_rp_group is None:
                    raise RuntimeError(
                        "[Dion] RP>1 requires inter distributed optimizer instance group "
                        f"(configured_rp_size={configured_rp_size})"
                    )
                rp_size = self._group_size(runtime_rp_group)
                if rp_size != configured_rp_size:
                    rp_group_ranks = dist.get_process_group_ranks(runtime_rp_group)
                    raise RuntimeError(
                        "[Dion] RP topology mismatch after parent init: "
                        f"configured_rp_size={configured_rp_size} actual_rp_group_size={rp_size} "
                        f"rp_group_ranks={rp_group_ranks}"
                    )
                self.rp_group = runtime_rp_group

        if self.rp_group is not None:
            self._assert_group_excludes_context_parallel(self.rp_group, label="rp_group")
        self.state_replica_group = self._resolve_state_replica_group()
        tp_group = self._resolve_dion_tp_group()
        tp_size = 1 if tp_group is None else self._group_size(tp_group)
        self._validate_matrix_topology(
            fs_size=int(self.fs_size),
            rp_size=int(configured_rp_size),
            tp_size=int(tp_size),
            is_expert=bool(self._is_expert_dion),
            split_qkv=bool(getattr(self.optimizer, "defaults", {}).get("split_qkv", False)),
            split_qkvg=bool(getattr(self.optimizer, "defaults", {}).get("split_qkv", False)),
            split_linear=bool(getattr(self.optimizer, "defaults", {}).get("split_linear", False)),
        )

    def _shard_param_uid(self, shard_param):
        """Return the checkpoint/state identity for one optimizer shard."""
        param_uid = getattr(shard_param, "_dion_param_uid", None)
        if param_uid is not None:
            return param_uid

        dist_metas = getattr(self, "dist_metas", {})
        dist_meta = dist_metas.get(shard_param)
        if dist_meta is not None and getattr(dist_meta, "param_uid", None) is not None:
            shard_param._dion_param_uid = dist_meta.param_uid
            return dist_meta.param_uid

        model_param = getattr(shard_param, "_model_param", None)
        if model_param is not None:
            for candidate in (
                self._get_opt_shard(model_param),
                self._get_data_shard(model_param),
                model_param,
            ):
                if candidate is None:
                    continue
                param_uid = getattr(candidate, "_dion_param_uid", None)
                if param_uid is not None:
                    shard_param._dion_param_uid = param_uid
                    candidate_meta = getattr(self, "_dion_dist_meta_by_uid", {}).get(param_uid)
                    if candidate_meta is not None:
                        dist_metas[shard_param] = candidate_meta
                    return param_uid

                candidate_meta = dist_metas.get(candidate)
                if candidate_meta is not None and getattr(candidate_meta, "param_uid", None) is not None:
                    shard_param._dion_param_uid = candidate_meta.param_uid
                    dist_metas[shard_param] = candidate_meta
                    return candidate_meta.param_uid

        raise RuntimeError(
            "[Dion] missing param_uid for optimizer shard "
            f"name={self._param_name(shard_param) or f'id_{id(shard_param)}'} "
            f"shape={tuple(shard_param.shape)}"
        )

    def _set_bucket_param_views(
        self,
        bucket,
        *,
        copy_data: bool,
        params: Optional[list[torch.nn.Parameter]] = None,
    ) -> None:
        """Ensure selected params in `bucket` alias the bucket's canonical param buffer."""
        return set_bucket_param_views(self, bucket, copy_data=copy_data, params=params)

    def _check_bucket_param_views(
        self,
        bucket,
        *,
        context: str,
        params: Optional[list[torch.nn.Parameter]] = None,
    ) -> None:
        """Verify selected param views still alias the canonical bucket buffer."""
        return check_bucket_param_views(self, bucket, context=context, params=params)

    def _mark_buckets_full_param_ready(self, ready: bool) -> None:
        """Track whether Dion full param views are ready for forward."""
        self._set_bucket_runtime_flag("_matrix_full_param_ready", ready)

    def _init_bucket_comm(self, bucket, fs_group) -> None:
        """Attach the Dion shard group to a bucket."""
        bucket.matrix_shard_group = fs_group

        expert_flags = [not getattr(p, "allreduce", True) for p in bucket.params]
        is_expert_bucket = any(expert_flags)
        bucket.dion_is_expert_bucket = is_expert_bucket
        if is_expert_bucket and not all(expert_flags):
            raise RuntimeError(
                f"[Dion][EP] mixed dense/expert bucket is invalid for Dion EP: "
                f"bucket_id={bucket.bucket_id} param_count={len(bucket.params)}"
            )
        if is_expert_bucket:
            expert_group = get_expected_expert_fs_group()
            assert_same_group_ranks(
                label=f"expert bucket {bucket.bucket_id}",
                actual_group=fs_group,
                expected_group=expert_group,
                extra="Megatron-Core EP local-shard group must stay on intra_expt_dp_group.",
            )

    def _init_standard_bucket(self, *, gbuf_idx: int, buffer, bucket, fs_group) -> None:
        """Configure one bucket that has no Dion layout."""
        return init_standard_bucket(self, gbuf_idx=gbuf_idx, buffer=buffer, bucket=bucket, fs_group=fs_group)

    def _init_dion_buckets(self) -> None:
        """Configure buffer-level Dion bucket layouts after parent optimizer init."""
        if not hasattr(self, "gbuf_ranges") or not hasattr(self, "buffers"):
            return

        shard_group = self.shard_group

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
                dion_layout = bucket_range_map.pop("matrix_bucket_layout", None)

                if dion_layout is not None and dion_layout.has_params:
                    fs_group, _, _ = self._bucket_fs_get_group_size_rank(buffer, bucket)
                    init_matrix_bucket(
                        self,
                        gbuf_idx=gbuf_idx,
                        buffer=buffer,
                        bucket=bucket,
                        matrix_layout=dion_layout,
                        fs_group=fs_group,
                    )
                else:
                    self._init_standard_bucket(
                        gbuf_idx=gbuf_idx,
                        buffer=buffer,
                        bucket=bucket,
                        fs_group=shard_group,
                    )

    def _refresh_dion_shards(self) -> None:
        """Refresh runtime optimizer param objects in the canonical Dion shard map."""
        for optim_group in self.optimizer.param_groups:
            for shard_param in optim_group["params"]:
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

    def _setup_dion_path(self) -> None:
        """Run Dion-specific setup after parent DistributedOptimizer init."""
        self._shards_by_param = {}
        self._matrix_buckets_by_param = {}
        self._matrix_entries_by_param = {}

        if hasattr(self, 'buffers'):
            self._init_dion_buckets()

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
                    self._refresh_dion_shards()
                    self._enable_overlap_grad_reduce = bool(getattr(self.ddp_config, "overlap_grad_reduce", False))
                    self._enable_overlap_param_gather = bool(getattr(self.ddp_config, "overlap_param_gather", False))
                    try:
                        self._enable_dion_runtime()
                    except Exception as error:
                        logger.exception(
                            "[Dion] Global rank %s: Failed in _enable_dion_runtime: %s",
                            self._global_rank,
                            error,
                        )
                        raise

        if dist.is_initialized() and hasattr(self, 'data_parallel_group'):
            dist.barrier(group=self.data_parallel_group)

    def _configure_dion_runtime_policy(self) -> None:
        """Apply MCore runtime policy bits that Dion math must mirror exactly."""
        if not hasattr(self, "optimizer") or not isinstance(self.optimizer, MegatronDion):
            return

        self.optimizer.defaults["rp_average_in_collective"] = bool(
            getattr(self.ddp_config, "average_in_collective", False)
        )

        if int(getattr(self, "fs_size", 1)) > 1 and not bool(
            getattr(self.optimizer, "use_fs_collectives", False)
        ):
            raise RuntimeError(
                "[Dion] --no-dion-use-fs-collectives is unsupported when FS>1. "
                f"fs_size={int(self.fs_size)}. Dion FS layouts require FS P/Q "
                "collectives; no correct fallback is implemented."
            )

    @classmethod
    def _build_model_gbuf_range(cls, param_and_grad_buffer, bucket_index):
        """
        Build model grad buffer ranges using parent DO standard, then add Dion metadata.

        Architecture:
        - Uses parent DistributedOptimizer's standard bucket structure
        - Adds Dion metadata only for 2D weight matrices
        - Standard params follow standard DO path entirely

        Returns: Parent DO structure + optional typed Dion shard layout per param
        """
        # STEP 1: Call parent to get DO standard bucket structure
        from ...distrib_optimizer import DistributedOptimizer
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

        fs_group, fs_size, fs_rank = cls._bucket_fs_get_group_size_rank(param_and_grad_buffer, bucket)
        grad_rank_to_fs_rank = resolve_grad_rank_to_fs_rank(
            grad_group=dp_group,
            fs_group=fs_group,
            fs_size=fs_size,
            bucket_id=bucket.bucket_id,
        )
        dion_param_count, dion_info_by_param = cls._mark_dion_bucket_params(
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

        # STEP 3: Recalculate buffer ranges for FS x TP hybrid sharding
        # Dion params use FS shard, standard params use DP shard

        (
            dion_layout,
            dion_shard_layout_by_param,
            dion_param_count,
        ) = (
            build_matrix_shard_entries(
                bucket=bucket,
                param_map=param_map,
                matrix_info_by_param=dion_info_by_param,
                fs_size=fs_size,
                fs_rank=fs_rank,
                grad_shard_group_size=dp_world_size,
                grad_rank_to_fs_rank=grad_rank_to_fs_rank,
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
                    "[Dion] build_dion_shard_entries mutated canonical standard DO param_map "
                    f"for param={param_name or f'id_{id(param)}'} before={snapshot} after={current}"
                )

        for param, range_info in param_map.items():
            range_info["matrix_shard_layout"] = dion_shard_layout_by_param.get(param)
        parent_result["local_total"] = 0 if dion_layout is None else dion_layout.shard_size

        # Calculate param counts for summary
        total_params = len(param_map)
        standard_count = total_params - dion_param_count

        # Add Dion communication layout to the parent result.
        parent_result["matrix_bucket_layout"] = dion_layout
        parent_result["standard_count"] = standard_count

        # Return hybrid sharding structure (parent ranges + Dion metadata)
        return parent_result

    def __init__(self, *args, **kwargs):
        """Initialize Dion distributed optimizer state."""
        # Initialize Dion param info before parent init.
        self.set_matrix_backend(DionBackend())
        self._shard_layouts_by_param = {}
        self._replica_group = kwargs.pop("replica_group", kwargs.pop("replica_group_override", None))
        self._dion_fs_group = kwargs.pop("dion_fs_group", None)
        self._dion_tp_group = kwargs.pop("dion_tp_group", None)
        self._pure_data_parallel_group = kwargs.pop("pure_data_parallel_group", None)
        self._is_expert_dion = bool(kwargs.pop("is_expert_dion", False))
        self._dion_state_param_by_uid = {}
        self._dion_dist_meta_by_uid = {}
        self._dion_batch_key_cache = {}
        self._matrix_local_grad_by_param = {}

        # 2D parallelism configuration
        # RP = Replicate Process (replicas with same shard)
        # FS = Fully Shard (shards within same replica)
        self._rp_size = kwargs.pop('rp_size', None) or kwargs.pop('replica_model_parallel_size', None)
        self._fs_size = kwargs.pop('fs_size', None) or kwargs.pop('fully_shard_model_parallel_size', None)

        if self._rp_size is None:
            self._rp_size = 1
        if self._pure_data_parallel_group is None:
            raise RuntimeError(
                "DistributedDionOptimizer requires pure_data_parallel_group "
                "from dion.distributed.integration; direct construction without the canonical builder is unsupported."
            )

        per_model_buffers = kwargs.get("per_model_buffers", None)
        dion_fs_size = 1 if self._dion_fs_group is None else self._group_size(self._dion_fs_group)
        dion_fs_rank = 0 if self._dion_fs_group is None else self._group_rank(self._dion_fs_group)
        if per_model_buffers is not None:
            for buffers in per_model_buffers.values():
                for buffer in buffers:
                    buffer.dion_fs_group = self._dion_fs_group
                    buffer.dion_fs_size = int(dion_fs_size)
                    buffer.dion_fs_rank = int(dion_fs_rank)

        # The parent optimizer owns the standard DP/DPxCP shard layout. Dion
        # resolves its CP-excluded RP/FS domains separately from MCore groups.
        super().__init__(*args, **kwargs)

        self._global_rank = dist.get_rank() if dist.is_initialized() else 0

        self._init_groups()
        self._configure_dion_runtime_policy()

        self._setup_dion_path()

    def _all_gather_bucket_params_(self, bucket, async_op=False):
        """Gather all Dion bucket params that do not follow the pure standard DO path."""
        return all_gather_bucket_params_(
            self,
            bucket,
            async_op=async_op,
        )

    def _check_dion_params(self):
        """Validate Dion shard metadata and build optimizer lookup tables."""
        # Use the runtime FS group that owns Dion sharding. Single-instance DO
        # may not expose a separate FS group, so it falls back to the DP group.
        fs_group = self.fs_group

        if fs_group is not None:
            annotation_group = fs_group
            fs_rank = dist.get_rank(fs_group)
            fs_size = dist.get_world_size(fs_group)
        else:
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
        unique_two_d_tp_late_reduction_excluded = 0
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

                        shard_layout = pri.get("matrix_shard_layout", None)
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

                        if shard_layout.local_numel < 0:
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
                        pname = ""
                        try:
                            pname = buffer.param_to_name.get(param, "")
                        except Exception:
                            pname = ""
                        if getattr(param, "use_dion", None) is False:
                            unique_two_d_manual_disabled += 1
                            continue

                        if not getattr(
                            param,
                            "matrix_optimizer_candidate",
                            getattr(param, "dion_candidate", False),
                        ):
                            unique_two_d_not_candidate += 1
                            continue

                        if (
                            getattr(param, "is_embedding_or_output_parameter", False)
                            or getattr(param, "is_lm_head_parameter", False)
                        ):
                            unique_two_d_role_excluded += 1
                            continue

                        if (
                            getattr(param, "sequence_parallel", False)
                            or getattr(param, "average_gradients_across_tp_domain", False)
                        ):
                            unique_two_d_tp_late_reduction_excluded += 1
                            continue

                        if is_float8tensor(param):
                            unique_two_d_fp8_skipped += 1
                            continue

                        unexpected_two_d_leftovers.append(pname or f"id_{id(param)}")

        expected_dion_two_d = (
            unique_two_d_total
            - unique_two_d_fp8_skipped
            - unique_two_d_manual_disabled
            - unique_two_d_not_candidate
            - unique_two_d_role_excluded
            - unique_two_d_tp_late_reduction_excluded
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
                f"tp_late_reduction_excluded={unique_two_d_tp_late_reduction_excluded} "
                f"expected_dion_two_d={expected_dion_two_d})"
            )

        if unexpected_two_d_leftovers:
            raise RuntimeError(
                "[Dion] unexpected standard 2D parameters remain after classification: "
                + ", ".join(unexpected_two_d_leftovers[:32])
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
        model_fp16_groups = []
        model_fp32_groups = []
        shard_float16_groups = []
        shard_fp32_groups = []
        main_shard_groups = []

        for group_range in opt_group_ranges:
            model_fp16_params = []
            model_fp32_params_this_group = []
            shard_float16_params_this_group = []
            shard_fp32_params_this_group = []
            main_shard_params = []

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
                logger.exception(
                    "[Dion] Global rank %s: Failed in _process_param_group "
                    "for group of %s params: %s",
                    global_rank,
                    len(group_range["params"]),
                    e,
                )
                for i, p in enumerate(group_range["params"]):
                    logger.error(f"  Param {i}: shape={p.shape}, ndim={p.ndim}, requires_grad={p.requires_grad}")
                raise

            if not use_precision_aware_optimizer:
                group_range["orig_group"]["params"] = [
                    *shard_fp32_params_this_group,
                    *main_shard_params,
                ]
            else:
                float16_optimizer_params = [
                    shard_main_param if shard_main_param is not None else shard_model_param
                    for shard_model_param, shard_main_param in zip(
                        shard_float16_params_this_group,
                        main_shard_params,
                    )
                    if shard_main_param is not None or shard_model_param is not None
                ]
                group_range["orig_group"]["params"] = [
                    *shard_fp32_params_this_group,
                    *float16_optimizer_params,
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
            dion_shard_layout = param_range_info.get("matrix_shard_layout", None)

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

    def _check_dion_param(self, model_param, context=""):
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
        param_name = self._param_name(model_param)
        use_precision_aware_optimizer = (
            config.use_precision_aware_optimizer_no_fp8_or_ds_fp8
        )
        if dion_shard_layout is not None:
            try:
                shard_model_param = create_fs_shard(self, model_param, dion_shard_layout)
                attach_fs_shard_(self, model_param, shard_model_param)
                self._check_dion_param(model_param, "FP16")
            except Exception as e:
                global_rank = self._global_rank
                logger.exception(
                    "[Dion] Global rank %s: Failed for Dion param "
                    "shape=%s dion_shard_layout=%s: %s",
                    global_rank,
                    tuple(model_param.shape),
                    dion_shard_layout,
                    e,
                )
                raise

            tensor_parallel.copy_tensor_model_parallel_attributes(
                shard_model_param, model_param
            )
            copy_qkv_split_metadata(shard_model_param, model_param)
            copy_qkvg_split_metadata(shard_model_param, model_param)
            if hasattr(model_param, 'shared'):
                shard_model_param.shared = model_param.shared

            # Dion matrix updates use an fp32 optimizer shard even when standard params use
            # the precision-aware optimizer surface.
            shard_main_param = shard_model_param.clone().float()
            shard_main_param._model_param = model_param

            # Register shard state/layout for Dion params.
            opt_shard = shard_main_param if shard_main_param is not None else shard_model_param
            register_matrix_shard(
                self,
                model_param=model_param,
                data_shard=shard_model_param,
                opt_shard=opt_shard,
                shard_layout=dion_shard_layout,
            )
        else:
            # Standard params are always DP-sharded via reduce-scatter.
            if is_float8tensor(model_param) and config.fp8_recipe != "delayed":
                shard_model_param = None
            else:
                shard_model_param = model_param.detach().view(-1)[
                    param_range.start : param_range.end
                ]

                shard_model_param._model_param = model_param
                tensor_parallel.copy_tensor_model_parallel_attributes(
                    shard_model_param, model_param
                )
                copy_qkv_split_metadata(shard_model_param, model_param)
                copy_qkvg_split_metadata(shard_model_param, model_param)
                if hasattr(model_param, 'shared'):
                    shard_model_param.shared = model_param.shared

            if not use_precision_aware_optimizer:
                if is_float8tensor(model_param):
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

        if shard_main_param is not None:
            tensor_parallel.copy_tensor_model_parallel_attributes(
                shard_main_param, model_param
            )
            copy_qkv_split_metadata(shard_main_param, model_param)
            copy_qkvg_split_metadata(shard_main_param, model_param)
            if hasattr(model_param, 'shared'):
                shard_main_param.shared = model_param.shared

        model_param.main_param = shard_main_param
        model_param.main_param_sharded = True

        model_fp16_params.append(model_param)
        if use_precision_aware_optimizer and dion_shard_layout is not None:
            shard_float16_params.append(shard_main_param)
        else:
            shard_float16_params.append(shard_model_param)
        main_shard_params.append(shard_main_param)

    def _process_float32_param(self, model_param, param_range, dion_shard_layout,
                              config,
                              model_fp32_params, shard_fp32_params):
        """Process float32 parameters."""
        if dion_shard_layout is not None:
            shard_model_param = create_fs_shard(self, model_param, dion_shard_layout)
            attach_fs_shard_(self, model_param, shard_model_param)

            tensor_parallel.copy_tensor_model_parallel_attributes(
                shard_model_param, model_param
            )
            copy_qkv_split_metadata(shard_model_param, model_param)
            copy_qkvg_split_metadata(shard_model_param, model_param)
            if hasattr(model_param, 'shared'):
                shard_model_param.shared = model_param.shared

            # Expose the optimizer shard through MCore's main_param convention.
            model_param.main_param = shard_model_param
            model_param.main_param_sharded = True

            self._check_dion_param(model_param, "FP32")

            # Register shard state/layout for Dion params.
            # FP32 params: data_shard == opt_shard == shard_model_param
            register_matrix_shard(
                self,
                model_param=model_param,
                data_shard=shard_model_param,
                opt_shard=shard_model_param,
                shard_layout=dion_shard_layout,
            )
        else:
            shard_model_param = model_param.view(-1)[param_range.start : param_range.end]
            shard_model_param._model_param = model_param

            tensor_parallel.copy_tensor_model_parallel_attributes(
                shard_model_param, model_param
            )
            copy_qkv_split_metadata(shard_model_param, model_param)
            copy_qkvg_split_metadata(shard_model_param, model_param)
            if hasattr(model_param, 'shared'):
                shard_model_param.shared = model_param.shared

            # Note: standard params use standard DO path, not registered in shard state

        model_fp32_params.append(model_param)
        shard_fp32_params.append(shard_model_param)

    def _enable_dion_runtime(self):
        """Enable the distributed Dion runtime."""
        if not isinstance(self.optimizer, (MegatronDion,)):
            return
        runtime_tp_group = self._resolve_dion_tp_group()
        dist_metas_sharded = enable_distributed_dion(
            optimizer=self.optimizer,
            global_rank=self._global_rank,
            replica_group=self._get_replicate_group(),
            data_parallel_group=self.validation_data_parallel_group,
            tp_group=runtime_tp_group,
            rp_group=self.rp_group,
            fs_group=self.fs_group,
            state_replica_group=self.state_replica_group,
            expected_rp_size=int(self._rp_size) if self._rp_size is not None else 1,
            build_all_dist_metas=self._build_all_dist_metas,
            route_step_params=self._route_step_params,
            group_size=self._group_size,
            group_rank=self._group_rank,
            validate_enabled_rp_topology=validate_enabled_rp_topology,
            log_error=logger.error,
        )
        if dist_metas_sharded is None:
            return
        self.dist_metas = dist_metas_sharded
        self.optimizer.dist_metas = dist_metas_sharded
        self._dion_dist_meta_by_uid = {
            dist_meta.param_uid: dist_meta
            for dist_meta in dist_metas_sharded.values()
            if getattr(dist_meta, "param_uid", None) is not None
        }
        self._prepare_split_child_fs_groups_for_known_metas()

    def _prepare_split_child_fs_groups_for_known_metas(self) -> None:
        """Create split-child FS groups from known metadata before state routing."""
        if not hasattr(self, "optimizer") or not hasattr(self.optimizer, "dist_metas"):
            return
        split_qkv_enabled = bool(self.optimizer.defaults.get("split_qkv", False))
        split_linear_enabled = bool(self.optimizer.defaults.get("split_linear", False))
        if not split_qkv_enabled and not split_linear_enabled:
            return
        dist_metas = sorted(
            self.optimizer.dist_metas.values(),
            key=lambda dist_meta: repr(
                (
                    getattr(dist_meta, "param_uid", None),
                    getattr(dist_meta, "param_name", ""),
                )
            ),
        )
        for dist_meta in dist_metas:
            if dist_meta is None or not bool(getattr(dist_meta, "is_dion_param", False)):
                continue
            if split_linear_enabled:
                split_rows = resolve_linear_split_rows(
                    optimizer_state=None,
                    dist_meta=dist_meta,
                )
                if split_rows is not None:
                    self._init_linear_child_groups(
                        dist_meta=dist_meta,
                        split_rows=split_rows,
                    )
                    continue
            if split_qkv_enabled:
                split_shapes = resolve_qkvg_split_shapes(
                    param=None,
                    optimizer_state=None,
                    dist_meta=dist_meta,
                )
                if split_shapes is not None:
                    self._init_qkvg_child_groups(
                        dist_meta=dist_meta,
                        split_shapes=split_shapes,
                    )
                    continue
                split_shapes = resolve_qkv_split_shapes(
                    param=None,
                    optimizer_state=None,
                    dist_meta=dist_meta,
                )
                if split_shapes is not None:
                    self._init_qkv_child_groups(
                        dist_meta=dist_meta,
                        split_shapes=split_shapes,
                    )

    def _build_all_dist_metas(self):
        """Create dist_metas with batch processing."""
        return build_all_dist_metas(
            shard_pairs_by_param=self._shards_by_param,
            build_param_dist_meta=lambda model_param, shard_param: build_param_dist_meta(
                model_param=model_param,
                shard_param=shard_param,
                fs_group=self.fs_group,
                shard_layouts_by_param=self._shard_layouts_by_param,
                get_param_name=self._global_param_name,
                tp_group=self._resolve_dion_tp_group(),
                use_low_rank_sync=bool(getattr(self.optimizer, "use_low_rank_sync", False)),
                rank_fraction_default=self.optimizer.defaults.get("rank_fraction", 0.25),
                rank_multiple_of_default=self.optimizer.defaults.get("rank_multiple_of", 1),
            ),
            add_standard_metas=lambda dist_metas_sharded: add_standard_metas(
                param_groups=self.optimizer.param_groups,
                dist_metas_sharded=dist_metas_sharded,
                get_param_name=self._global_param_name,
                get_direct_param_name=self._param_name,
                rank_fraction_default=self.optimizer.defaults.get("rank_fraction", 0.25),
                rank_multiple_of_default=self.optimizer.defaults.get("rank_multiple_of", 1),
                use_low_rank_sync=bool(getattr(self.optimizer, "use_low_rank_sync", False)),
            ),
            validate_dist_meta_uids=lambda dist_metas_sharded: validate_dist_meta_uids(
                dist_metas_sharded=dist_metas_sharded,
                get_param_name=self._param_name,
                log_error=logger.error,
            ),
        )

    def _get_replicate_group(self):
        """Return the active Dion replicate group from runtime state."""
        if self._replica_group is not None:
            return self._replica_group if self._group_size(self._replica_group) > 1 else None
        if self.rp_group is None:
            return None
        return self.rp_group if self._group_size(self.rp_group) > 1 else None

    def _build_dion_batches(self, dion_params: List[Tuple]) -> List[DionBatch]:
        """Build ready-to-execute Dion batches for distributed Dion."""
        return build_dion_batches(
            dion_params=dion_params,
            use_fs_collectives=self.optimizer.use_fs_collectives,
            state_replica_group=self.state_replica_group,
            replica_validation_group=self.validation_data_parallel_group,
            batch_key_cache=self._dion_batch_key_cache,
            global_rank=self._global_rank,
            group_size=self._group_size,
            get_replicate_group=self._get_replicate_group,
            resolve_ortho_group=lambda config, dist_meta: resolve_ortho_group(
                config,
                dist_meta,
                use_fs_collectives=self.optimizer.use_fs_collectives,
                resolve_tp_group=lambda meta, expect_group: resolve_tp_group(
                    meta,
                    expect_group=expect_group,
                    group_size=self._group_size,
                ),
                resolve_fs_group_from_meta=lambda meta, expect_group: resolve_fs_group_from_meta(
                    meta,
                    expect_group=expect_group,
                    group_size=self._group_size,
                ),
                is_tp_sharded=is_p_tp_sharded,
                is_fs_sharded=is_p_fs_sharded,
            ),
            resolve_tp_group=lambda meta, expect_group: resolve_tp_group(
                meta,
                expect_group=expect_group,
                group_size=self._group_size,
            ),
            resolve_fs_group_from_meta=lambda meta, expect_group: resolve_fs_group_from_meta(
                meta,
                expect_group=expect_group,
                group_size=self._group_size,
            ),
        )

    def _sync_dion_state(self, dion_params: List[Tuple]) -> None:
        """Synchronize Dion-owned state before batch construction."""
        sync_q_replicas(
            dion_params=dion_params,
            state_replica_group=self.state_replica_group,
            group_size=self._group_size,
            make_group_broadcast=lambda process_group: make_group_broadcast(
                process_group,
                group_size=self._group_size,
            ),
        )

    @staticmethod
    def _require_param_config(param, dist_meta) -> DionParamConfig:
        """Return Dion config metadata for one parameter."""
        if dist_meta is None:
            raise RuntimeError(
                "[DION_MISSING_DIST_META_FOR_PARAM_CONFIG] "
                f"param_name={getattr(param, '_param_name', '')} "
                f"param_shape={tuple(param.shape)}"
            )
        config = getattr(dist_meta, "param_config", None)
        if config is None:
            raise RuntimeError(
                "[DION_MISSING_DIST_PARAM_CONFIG] "
                f"param_uid={getattr(dist_meta, 'param_uid', None)} "
                f"param_name={getattr(dist_meta, 'param_name', '')}"
            )
        return config

    def _refresh_dion_step_metadata(self, *, param, optimizer_state, optim_group, dist_meta) -> None:
        """Refresh step-local Dion metadata from the current optimizer group and state.

        This is intentionally metadata-only.  It handles restored optimizer state, where
        Q/r may already exist and state initialization will not call build_q_init().
        """
        if dist_meta is None:
            return
        if optim_group.get("algorithm", "dion") != "dion":
            return
        if not bool(getattr(dist_meta, "is_dion_param", False)):
            return
        if getattr(dist_meta, "global_shape", None) is None or len(dist_meta.global_shape) != 2:
            return

        local_shape = get_local_shape(
            dist_meta,
            *require_2d_local_shape(param, dist_meta),
        )
        self._refresh_dion_param_config(
            dist_meta=dist_meta,
            optim_group=optim_group,
            local_shape=local_shape,
            r_global_override=optimizer_state.get("r", None),
        )

    def _refresh_dion_param_config(
        self,
        *,
        dist_meta,
        optim_group,
        local_shape,
        r_global_override=None,
    ) -> None:
        """Rebuild a Dion config using current optimizer-group metadata and known Q rank."""
        rank_fraction = float(
            optim_group.get("rank_fraction", self.optimizer.defaults["rank_fraction"])
        )
        rank_multiple_of = int(
            optim_group.get("rank_multiple_of", self.optimizer.defaults["rank_multiple_of"])
        )
        dist_meta.rank_fraction = rank_fraction

        dist_meta.param_config = build_param_config(
            param_ndim=2,
            local_shape=tuple(int(dim) for dim in local_shape),
            dist_meta=dist_meta,
            use_low_rank_sync=bool(getattr(self.optimizer, "use_low_rank_sync", False)),
            r_global_override=(
                int(r_global_override) if r_global_override is not None else None
            ),
            rank_fraction_default=rank_fraction,
            rank_multiple_of_default=rank_multiple_of,
            tp_world_size=int(getattr(dist_meta, "tp_world_size", 1)),
            tp_active=bool(
                getattr(dist_meta, "tp_shard_dim", -1) in (0, 1)
                and int(getattr(dist_meta, "tp_world_size", 1)) > 1
            ),
        )

    def _init_optimizer_state(self, param, state, optim_group) -> None:
        """Build optimizer state at the adapter boundary."""
        optimizer = self.optimizer
        dist_meta = optimizer.dist_metas.get(param, None)
        config = self._require_param_config(param, dist_meta)
        is_dion_eligible = bool(getattr(dist_meta, "is_dion_param", False))
        local_shape = (
            get_local_shape(
                dist_meta,
                *require_2d_local_shape(param, dist_meta),
            )
            if is_dion_eligible
            else None
        )
        q_init = None
        split_qkv_enabled = bool(self.optimizer.defaults.get("split_qkv", False))
        split_linear_enabled = bool(self.optimizer.defaults.get("split_linear", False))
        split_qkvg_shapes = (
            resolve_qkvg_split_shapes(param=param, optimizer_state=state, dist_meta=dist_meta)
            if split_qkv_enabled
            else None
        )
        split_qkv_shapes = (
            resolve_qkv_split_shapes(param=param, optimizer_state=state, dist_meta=dist_meta)
            if split_qkv_enabled
            else None
        )
        split_linear_rows = (
            resolve_linear_split_rows(optimizer_state=state, dist_meta=dist_meta)
            if split_linear_enabled
            else None
        )
        if is_dion_eligible and split_linear_rows is not None and dist_meta is not None:
            self._init_linear_child_groups(
                dist_meta=dist_meta,
                split_rows=split_linear_rows,
            )
        if is_dion_eligible and split_qkvg_shapes is not None and dist_meta is not None:
            self._init_qkvg_child_groups(
                dist_meta=dist_meta,
                split_shapes=split_qkvg_shapes,
            )
        if is_dion_eligible and split_qkv_shapes is not None and dist_meta is not None:
            self._init_qkv_child_groups(
                dist_meta=dist_meta,
                split_shapes=split_qkv_shapes,
            )
        if (
            is_dion_eligible
            and split_qkvg_shapes is None
            and split_qkv_shapes is None
            and split_linear_rows is None
        ):
            q_init = build_q_init(
                param=param,
                optim_group=optim_group,
                dist_meta=dist_meta,
                rank_fraction_default=self.optimizer.defaults["rank_fraction"],
                rank_multiple_of_default=self.optimizer.defaults["rank_multiple_of"],
                use_low_rank_sync=bool(getattr(self.optimizer, "use_low_rank_sync", False)),
                base_training_seed=resolve_base_training_seed(),
                get_replicate_group=self._get_replicate_group,
                make_group_broadcast=lambda process_group: make_group_broadcast(
                    process_group,
                    group_size=self._group_size,
                ),
            )
        init_param_state(
            param=param,
            state=state,
            optim_group=optim_group,
            mixed_precision_config=optimizer._mixed_precision_config,
            config=config,
            dist_meta=dist_meta,
            is_dion_eligible=is_dion_eligible,
            local_shape=local_shape,
            rank_fraction_default=optimizer.defaults['rank_fraction'],
            rank_multiple_of_default=optimizer.defaults['rank_multiple_of'],
            split_qkv_default=split_qkv_enabled,
            split_linear_default=split_linear_enabled,
            q_init=q_init,
        )

    def _route_step_params(self):
        """Route one distributed optimizer step into Dion batches and scalar params."""
        return self._route_matrix_step_params(
            param_groups=self.optimizer.param_groups,
            dist_metas=self.optimizer.dist_metas,
            get_step_param_grad=self._get_step_param_grad,
            ensure_optimizer_state=self._ensure_optimizer_state,
            require_param_config=self._require_param_config,
        )

    def _get_step_param_grad(self, param):
        """Return the step grad tensor chosen at the adapter boundary."""
        model_param = getattr(param, "_model_param", None)
        if model_param is not None and getattr(
            model_param,
            "is_matrix_param",
            getattr(model_param, "is_dion_param", False),
        ):
            return self._get_local_grad(model_param, param)
        dg = getattr(param, 'decoupled_grad', None)
        if dg is not None:
            return dg

        try:
            if param.grad is not None:
                return param.grad
        except RuntimeError:
            pass

        return None

    def _should_use_distributed_dion_update(self, param, state, optim_group, dist_meta) -> bool:
        """Return the distributed Dion route selected by the adapter."""
        return should_use_distributed_dion_update(
            param=param,
            state=state,
            optim_group=optim_group,
            dist_meta=dist_meta,
            global_rank=self._global_rank,
        )

    def _split_parent_dist_meta(self, parent_dist_meta: DionDistMeta) -> DionDistMeta:
        """Return metadata for the single expert matrix that split children consume."""
        if not _has_active_expert_layout(parent_dist_meta):
            return parent_dist_meta

        per_expert_global_shape = tuple(
            int(dim) for dim in parent_dist_meta.per_expert_global_shape
        )
        expert_local_shape = getattr(parent_dist_meta, "local_shape", None)
        if expert_local_shape is None:
            raise RuntimeError(
                "[DION_EXPERT_SPLIT_MISSING_LOCAL_SHAPE] "
                f"param_uid={getattr(parent_dist_meta, 'param_uid', None)} "
                f"param_name={getattr(parent_dist_meta, 'param_name', '')}"
            )
        expert_local_shape = tuple(int(dim) for dim in expert_local_shape)
        expert_token = _expert_split_token(parent_dist_meta)
        parent_name = getattr(parent_dist_meta, "param_name", "")
        expert_name = (
            f"{parent_name}::{expert_token[0]}{expert_token[1]}"
            if expert_token[1] >= 0
            else parent_name
        )
        fs_start_idx = int(getattr(parent_dist_meta, "fs_start_idx", -1))
        fs_end_idx = int(getattr(parent_dist_meta, "fs_end_idx", -1))
        fs_shard_dim = int(getattr(parent_dist_meta, "fs_shard_dim", -1))
        fs_world_size = int(getattr(parent_dist_meta, "fs_world_size", 1))
        fs_rank = int(getattr(parent_dist_meta, "fs_rank", -1))
        if fs_shard_dim in (0, 1) and fs_world_size > 1 and fs_rank >= 0:
            fs_start_idx, fs_end_idx = compute_fs_shard_range(
                int(per_expert_global_shape[fs_shard_dim]),
                fs_world_size,
                fs_rank,
            )
        return replace(
            parent_dist_meta,
            shape=expert_local_shape,
            global_shape=per_expert_global_shape,
            fs_start_idx=int(fs_start_idx),
            fs_end_idx=int(fs_end_idx),
            per_expert_global_shape=None,
            local_shape=expert_local_shape,
            param_uid=_with_expert_token(parent_dist_meta.param_uid, expert_token),
            param_name=expert_name,
            expert_axis=-1,
            num_local_experts=1,
            local_expert_index=-1,
        )

    def _split_parent_views(
        self,
        *,
        param,
        grad,
        momentum,
        dist_meta: DionDistMeta,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int], DionDistMeta]:
        """Return the single-expert parent tensor views used by split-child routing."""
        physical_shape = tuple(int(dim) for dim in require_2d_local_shape(param, dist_meta))
        param_2d = param.view(*physical_shape)
        grad_2d = grad.reshape(*physical_shape)
        momentum_2d = momentum.reshape(*physical_shape)
        split_dist_meta = self._split_parent_dist_meta(dist_meta)

        if not _has_active_expert_layout(dist_meta):
            return param_2d, grad_2d, momentum_2d, physical_shape, split_dist_meta

        expert_axis = int(getattr(dist_meta, "expert_axis", -1))
        num_local_experts = int(getattr(dist_meta, "num_local_experts", 1))
        local_expert_index = int(getattr(dist_meta, "local_expert_index", -1))
        expert_local_shape = tuple(int(dim) for dim in split_dist_meta.local_shape)
        context = (
            f"param_uid={getattr(dist_meta, 'param_uid', None)} "
            f"param_name={getattr(dist_meta, 'param_name', '')}"
        )
        return (
            local_expert_tensor_view(
                param_2d,
                axis=expert_axis,
                num_local_experts=num_local_experts,
                local_expert_index=local_expert_index,
                local_shape=expert_local_shape,
                context=f"param:{context}",
            ),
            local_expert_tensor_view(
                grad_2d,
                axis=expert_axis,
                num_local_experts=num_local_experts,
                local_expert_index=local_expert_index,
                local_shape=expert_local_shape,
                context=f"grad:{context}",
            ),
            local_expert_tensor_view(
                momentum_2d,
                axis=expert_axis,
                num_local_experts=num_local_experts,
                local_expert_index=local_expert_index,
                local_shape=expert_local_shape,
                context=f"momentum:{context}",
            ),
            expert_local_shape,
            split_dist_meta,
        )

    @staticmethod
    def _normalize_linear_split_rows_for_meta(
        split_rows: tuple[int, int],
        dist_meta: DionDistMeta,
    ) -> tuple[int, int]:
        """Return split-linear row sizes in the active single-expert coordinate space."""
        split_rows = tuple(int(dim) for dim in split_rows)
        per_expert_shape = getattr(dist_meta, "per_expert_global_shape", None)
        if per_expert_shape is None:
            return split_rows
        per_expert_rows = int(per_expert_shape[0])
        if sum(split_rows) == per_expert_rows:
            return split_rows
        parent_global_shape = getattr(dist_meta, "global_shape", None)
        parent_rows = int(parent_global_shape[0]) if parent_global_shape is not None else 0
        expert_axis = int(getattr(dist_meta, "expert_axis", -1))
        num_local_experts = int(getattr(dist_meta, "num_local_experts", 1))
        if (
            expert_axis == 0
            and num_local_experts > 1
            and sum(split_rows) == parent_rows
            and all(int(row) % num_local_experts == 0 for row in split_rows)
        ):
            converted = tuple(int(row) // num_local_experts for row in split_rows)
            if sum(converted) == per_expert_rows:
                return converted
        raise RuntimeError(
            "[DION_EXPERT_LINEAR_SPLIT_ROWS_MISMATCH] "
            f"param_uid={getattr(dist_meta, 'param_uid', None)} "
            f"param_name={getattr(dist_meta, 'param_name', '')} "
            f"split_rows={split_rows} per_expert_global_shape={per_expert_shape} "
            f"global_shape={getattr(dist_meta, 'global_shape', None)}"
        )

    def _build_qkvg_child_dist_meta(
        self,
        *,
        parent_dist_meta: DionDistMeta,
        child_kind: str,
        split_shapes: tuple[int, int, int, int],
        child_local_shape: tuple[int, int],
        child_global_shape: tuple[int, int],
    ) -> DionDistMeta:
        """Build optimizer-only child metadata for one fused QKVG child."""
        child_uid = qkvg_child_param_uid(parent_dist_meta.param_uid, child_kind)
        child_name = qkvg_child_name(parent_dist_meta.param_name, child_kind)
        (
            child_fs_group,
            child_fs_world_size,
            child_fs_rank,
            child_fs_start_idx,
            child_fs_end_idx,
            fs_row_shard_sizes,
        ) = self._resolve_qkvg_child_fs_shard_layout(
            parent_dist_meta=parent_dist_meta,
            split_shapes=split_shapes,
            child_kind=child_kind,
            create_group=False,
        )
        (
            child_tp_group,
            child_tp_world_size,
            child_tp_rank,
            tp_row_shard_start_idx,
            tp_row_shard_end_idx,
            tp_row_shard_sizes,
        ) = self._resolve_qkvg_child_tp_shard_layout(
            parent_dist_meta=parent_dist_meta,
            split_shapes=split_shapes,
            child_kind=child_kind,
            create_group=False,
        )
        return build_split_child_dist_meta(
            parent_dist_meta=parent_dist_meta,
            child_uid=child_uid,
            child_name=child_name,
            child_local_shape=child_local_shape,
            child_global_shape=child_global_shape,
            fs_layout=(
                child_fs_group,
                child_fs_world_size,
                child_fs_rank,
                child_fs_start_idx,
                child_fs_end_idx,
                fs_row_shard_sizes,
            ),
            tp_layout=(
                child_tp_group,
                child_tp_world_size,
                child_tp_rank,
                tp_row_shard_start_idx,
                tp_row_shard_end_idx,
                tp_row_shard_sizes,
            ),
            child_fields={
                "is_transposed": False,
                "is_qkvg_child": True,
                "qkvg_child_kind": child_kind,
                "qkvg_split_shapes": tuple(int(dim) for dim in split_shapes),
            },
            error_prefix="QKVG_CHILD",
            use_low_rank_sync=bool(self.optimizer.use_low_rank_sync),
            rank_fraction_default=self.optimizer.defaults["rank_fraction"],
            rank_multiple_of_default=self.optimizer.defaults["rank_multiple_of"],
        )

    def _resolve_qkvg_child_row_group_layout(
        self,
        *,
        parent_dist_meta: DionDistMeta,
        split_shapes: tuple[int, int, int, int],
        child_kind: str,
        parent_group,
        parent_world_size: int,
        parent_rank: int,
        label: str,
        create_group: bool,
    ) -> tuple[
        Optional[torch.distributed.ProcessGroup],
        int,
        int,
        int,
        int,
        tuple[int, ...],
    ]:
        """Return child-local ownership for a parent row-sharded QKVG group."""
        child_index = {"q": 0, "gate": 1, "k": 2, "v": 3}.get(child_kind, None)
        if child_index is None:
            raise RuntimeError(
                "[DION_INVALID_QKVG_CHILD_KIND] "
                f"param_uid={parent_dist_meta.param_uid} param_name={parent_dist_meta.param_name} "
                f"child_kind={child_kind!r}"
            )

        parent_global_shape = tuple(int(dim) for dim in parent_dist_meta.global_shape)
        parent_global_rows = int(parent_global_shape[0])
        total_per_group = int(sum(int(dim) for dim in split_shapes))
        if total_per_group <= 0 or parent_global_rows % total_per_group != 0:
            raise RuntimeError(
                "[DION_QKVG_CHILD_INVALID_PARENT_GROUP_LAYOUT] "
                f"param_uid={parent_dist_meta.param_uid} param_name={parent_dist_meta.param_name} "
                f"parent_global_rows={parent_global_rows} split_shapes={split_shapes}"
            )
        child_global_rows = qkvg_child_global_shape(
            parent_global_shape,
            split_shapes,
            child_kind,
        )[0]

        child_ranges = []
        for rank_idx in range(parent_world_size):
            parent_rank_start, parent_rank_end = compute_fs_shard_range(
                parent_global_rows,
                parent_world_size,
                rank_idx,
            )
            child_range = qkvg_child_row_range(
                parent_row_start=parent_rank_start,
                parent_row_end=parent_rank_end,
                split_shapes=split_shapes,
                child_kind=child_kind,
            )
            child_ranges.append(child_range)

        return resolve_row_child_layout(
            parent_group=parent_group,
            parent_world_size=parent_world_size,
            parent_rank=parent_rank,
            child_rows=child_global_rows,
            child_ranges=tuple(child_ranges),
            label=label,
            detail=(
                f"param_uid={parent_dist_meta.param_uid} "
                f"param_name={parent_dist_meta.param_name} "
                f"child_kind={child_kind} split_shapes={split_shapes}"
            ),
            error_prefix="QKVG_CHILD",
            create_group=create_group,
            make_group=_ensure_child_group,
        ).as_tuple()

    def _resolve_qkvg_child_fs_shard_layout(
        self,
        *,
        parent_dist_meta: DionDistMeta,
        split_shapes: tuple[int, int, int, int],
        child_kind: str,
        create_group: bool,
    ) -> tuple[
        Optional[torch.distributed.ProcessGroup],
        int,
        int,
        int,
        int,
        Optional[tuple[int, ...]],
    ]:
        """Return child-local FS ownership for one split QKVG child."""
        parent_fs_world_size = int(getattr(parent_dist_meta, "fs_world_size", 1))
        parent_fs_rank = int(getattr(parent_dist_meta, "fs_rank", -1))
        parent_fs_shard_dim = int(getattr(parent_dist_meta, "fs_shard_dim", -1))
        parent_fs_group = getattr(parent_dist_meta, "fs_group", None)

        if parent_fs_shard_dim != 0 or parent_fs_world_size <= 1:
            return (
                parent_fs_group,
                parent_fs_world_size,
                parent_fs_rank,
                int(parent_dist_meta.fs_start_idx),
                int(parent_dist_meta.fs_end_idx),
                None,
            )

        return self._resolve_qkvg_child_row_group_layout(
            parent_dist_meta=parent_dist_meta,
            split_shapes=split_shapes,
            child_kind=child_kind,
            parent_group=parent_fs_group,
            parent_world_size=parent_fs_world_size,
            parent_rank=parent_fs_rank,
            label="FS",
            create_group=create_group,
        )

    def _resolve_qkvg_child_tp_shard_layout(
        self,
        *,
        parent_dist_meta: DionDistMeta,
        split_shapes: tuple[int, int, int, int],
        child_kind: str,
        create_group: bool,
    ) -> tuple[
        Optional[torch.distributed.ProcessGroup],
        int,
        int,
        int,
        int,
        Optional[tuple[int, ...]],
    ]:
        """Return child-local TP ownership for one split QKVG child."""
        parent_tp_world_size = int(getattr(parent_dist_meta, "tp_world_size", 1))
        parent_tp_rank = int(getattr(parent_dist_meta, "tp_rank", -1))
        parent_tp_shard_dim = int(getattr(parent_dist_meta, "tp_shard_dim", -1))
        parent_tp_group = getattr(parent_dist_meta, "tp_group", None)

        if parent_tp_shard_dim != 0 or parent_tp_world_size <= 1:
            return (
                parent_tp_group,
                parent_tp_world_size,
                parent_tp_rank,
                -1,
                -1,
                None,
            )

        return self._resolve_qkvg_child_row_group_layout(
            parent_dist_meta=parent_dist_meta,
            split_shapes=split_shapes,
            child_kind=child_kind,
            parent_group=parent_tp_group,
            parent_world_size=parent_tp_world_size,
            parent_rank=parent_tp_rank,
            label="TP",
            create_group=create_group,
        )

    def _ensure_qkvg_child_state_(
        self,
        *,
        parent_param,
        parent_state: dict,
        optim_group: dict,
        child_dist_meta: DionDistMeta,
    ) -> None:
        """Initialize child-specific Q state for one fused QKVG child when needed."""
        child_kind = child_dist_meta.qkvg_child_kind
        q_key = qkvg_state_key("Q", child_kind)
        if q_key in parent_state:
            return

        child_q_init = build_q_init(
            param=parent_param,
            optim_group=optim_group,
            dist_meta=child_dist_meta,
            rank_fraction_default=self.optimizer.defaults["rank_fraction"],
            rank_multiple_of_default=self.optimizer.defaults["rank_multiple_of"],
            use_low_rank_sync=bool(getattr(self.optimizer, "use_low_rank_sync", False)),
            base_training_seed=resolve_base_training_seed(),
            get_replicate_group=self._get_replicate_group,
            make_group_broadcast=lambda process_group: make_group_broadcast(
                process_group,
                group_size=self._group_size,
            ),
        )
        q_layout = child_q_init.q_layout
        if q_layout is None:
            raise RuntimeError(
                "[DION_MISSING_QKVG_CHILD_Q_LAYOUT] "
                f"param_uid={child_dist_meta.param_uid} param_name={child_dist_meta.param_name}"
            )
        q_state = init_q_state(
            param=parent_param,
            mixed_precision_config=self.optimizer._mixed_precision_config,
            config=child_dist_meta.param_config,
            dist_meta=child_dist_meta,
            q_layout=q_layout,
            q_seed=child_q_init.q_seed,
            tp_world_size=int(child_q_init.tp_world_size),
            tp_rank=int(child_q_init.tp_rank),
            use_q_unshard=bool(child_q_init.use_q_unshard),
        )
        broadcast_q = child_q_init.broadcast_q
        if broadcast_q is None:
            raise RuntimeError(
                "[DION_MISSING_QKVG_CHILD_BROADCAST] "
                f"param_uid={child_dist_meta.param_uid} param_name={child_dist_meta.param_name}"
            )
        broadcast_q(q_state)

        parent_state[q_key] = q_state
        parent_state[qkvg_state_key("r", child_kind)] = int(q_layout.r_global)
        parent_state[qkvg_state_key("local_shape", child_kind)] = tuple(
            int(dim) for dim in child_dist_meta.local_shape
        )
        parent_state[qkvg_state_key("global_shape", child_kind)] = tuple(
            int(dim) for dim in child_dist_meta.global_shape
        )
        parent_state[f"_qkvg_{child_kind}_needs_state_replica_q_sync"] = True

    def _build_qkv_child_dist_meta(
        self,
        *,
        parent_dist_meta: DionDistMeta,
        child_kind: str,
        split_shapes: tuple[int, int, int],
        child_local_shape: tuple[int, int],
        child_global_shape: tuple[int, int],
    ) -> DionDistMeta:
        """Build optimizer-only child metadata for one fused QKV child."""
        child_uid = qkv_child_param_uid(parent_dist_meta.param_uid, child_kind)
        child_name = qkv_child_name(parent_dist_meta.param_name, child_kind)
        (
            child_fs_group,
            child_fs_world_size,
            child_fs_rank,
            child_fs_start_idx,
            child_fs_end_idx,
            fs_row_shard_sizes,
        ) = self._resolve_qkv_child_fs_shard_layout(
            parent_dist_meta=parent_dist_meta,
            split_shapes=split_shapes,
            child_kind=child_kind,
            create_group=False,
        )
        (
            child_tp_group,
            child_tp_world_size,
            child_tp_rank,
            tp_row_shard_start_idx,
            tp_row_shard_end_idx,
            tp_row_shard_sizes,
        ) = self._resolve_qkv_child_tp_shard_layout(
            parent_dist_meta=parent_dist_meta,
            split_shapes=split_shapes,
            child_kind=child_kind,
            create_group=False,
        )
        return build_split_child_dist_meta(
            parent_dist_meta=parent_dist_meta,
            child_uid=child_uid,
            child_name=child_name,
            child_local_shape=child_local_shape,
            child_global_shape=child_global_shape,
            fs_layout=(
                child_fs_group,
                child_fs_world_size,
                child_fs_rank,
                child_fs_start_idx,
                child_fs_end_idx,
                fs_row_shard_sizes,
            ),
            tp_layout=(
                child_tp_group,
                child_tp_world_size,
                child_tp_rank,
                tp_row_shard_start_idx,
                tp_row_shard_end_idx,
                tp_row_shard_sizes,
            ),
            child_fields={
                "is_transposed": False,
                "is_qkv_child": True,
                "qkv_child_kind": child_kind,
                "qkv_split_shapes": tuple(int(dim) for dim in split_shapes),
            },
            error_prefix="QKV_CHILD",
            use_low_rank_sync=bool(self.optimizer.use_low_rank_sync),
            rank_fraction_default=self.optimizer.defaults["rank_fraction"],
            rank_multiple_of_default=self.optimizer.defaults["rank_multiple_of"],
        )

    def _resolve_qkv_child_row_group_layout(
        self,
        *,
        parent_dist_meta: DionDistMeta,
        split_shapes: tuple[int, int, int],
        child_kind: str,
        parent_group,
        parent_world_size: int,
        parent_rank: int,
        label: str,
        create_group: bool,
    ) -> tuple[
        Optional[torch.distributed.ProcessGroup],
        int,
        int,
        int,
        int,
        tuple[int, ...],
    ]:
        """Return child-local ownership for a parent row-sharded QKV group."""
        child_index = {"q": 0, "k": 1, "v": 2}.get(child_kind, None)
        if child_index is None:
            raise RuntimeError(
                "[DION_INVALID_QKV_CHILD_KIND] "
                f"param_uid={parent_dist_meta.param_uid} param_name={parent_dist_meta.param_name} "
                f"child_kind={child_kind!r}"
            )

        parent_global_shape = tuple(int(dim) for dim in parent_dist_meta.global_shape)
        parent_global_rows = int(parent_global_shape[0])
        total_per_group = int(sum(int(dim) for dim in split_shapes))
        if total_per_group <= 0 or parent_global_rows % total_per_group != 0:
            raise RuntimeError(
                "[DION_QKV_CHILD_INVALID_PARENT_GROUP_LAYOUT] "
                f"param_uid={parent_dist_meta.param_uid} param_name={parent_dist_meta.param_name} "
                f"parent_global_rows={parent_global_rows} split_shapes={split_shapes}"
            )
        child_global_rows = qkv_child_global_shape(
            parent_global_shape,
            split_shapes,
            child_kind,
        )[0]

        child_ranges = []
        for rank_idx in range(parent_world_size):
            parent_rank_start, parent_rank_end = compute_fs_shard_range(
                parent_global_rows,
                parent_world_size,
                rank_idx,
            )
            child_range = qkv_child_row_range(
                parent_row_start=parent_rank_start,
                parent_row_end=parent_rank_end,
                split_shapes=split_shapes,
                child_kind=child_kind,
            )
            child_ranges.append(child_range)

        return resolve_row_child_layout(
            parent_group=parent_group,
            parent_world_size=parent_world_size,
            parent_rank=parent_rank,
            child_rows=child_global_rows,
            child_ranges=tuple(child_ranges),
            label=label,
            detail=(
                f"param_uid={parent_dist_meta.param_uid} "
                f"param_name={parent_dist_meta.param_name} "
                f"child_kind={child_kind} split_shapes={split_shapes}"
            ),
            error_prefix="QKV_CHILD",
            create_group=create_group,
            make_group=_ensure_child_group,
        ).as_tuple()

    def _resolve_qkv_child_fs_shard_layout(
        self,
        *,
        parent_dist_meta: DionDistMeta,
        split_shapes: tuple[int, int, int],
        child_kind: str,
        create_group: bool,
    ) -> tuple[
        Optional[torch.distributed.ProcessGroup],
        int,
        int,
        int,
        int,
        Optional[tuple[int, ...]],
    ]:
        """Return child-local FS ownership for one split QKV child."""
        parent_fs_world_size = int(getattr(parent_dist_meta, "fs_world_size", 1))
        parent_fs_rank = int(getattr(parent_dist_meta, "fs_rank", -1))
        parent_fs_shard_dim = int(getattr(parent_dist_meta, "fs_shard_dim", -1))
        parent_fs_group = getattr(parent_dist_meta, "fs_group", None)

        if parent_fs_shard_dim != 0 or parent_fs_world_size <= 1:
            return (
                parent_fs_group,
                parent_fs_world_size,
                parent_fs_rank,
                int(parent_dist_meta.fs_start_idx),
                int(parent_dist_meta.fs_end_idx),
                None,
            )

        return self._resolve_qkv_child_row_group_layout(
            parent_dist_meta=parent_dist_meta,
            split_shapes=split_shapes,
            child_kind=child_kind,
            parent_group=parent_fs_group,
            parent_world_size=parent_fs_world_size,
            parent_rank=parent_fs_rank,
            label="FS",
            create_group=create_group,
        )

    def _resolve_qkv_child_tp_shard_layout(
        self,
        *,
        parent_dist_meta: DionDistMeta,
        split_shapes: tuple[int, int, int],
        child_kind: str,
        create_group: bool,
    ) -> tuple[
        Optional[torch.distributed.ProcessGroup],
        int,
        int,
        int,
        int,
        Optional[tuple[int, ...]],
    ]:
        """Return child-local TP ownership for one split QKV child."""
        parent_tp_world_size = int(getattr(parent_dist_meta, "tp_world_size", 1))
        parent_tp_rank = int(getattr(parent_dist_meta, "tp_rank", -1))
        parent_tp_shard_dim = int(getattr(parent_dist_meta, "tp_shard_dim", -1))
        parent_tp_group = getattr(parent_dist_meta, "tp_group", None)

        if parent_tp_shard_dim != 0 or parent_tp_world_size <= 1:
            return (
                parent_tp_group,
                parent_tp_world_size,
                parent_tp_rank,
                -1,
                -1,
                None,
            )

        return self._resolve_qkv_child_row_group_layout(
            parent_dist_meta=parent_dist_meta,
            split_shapes=split_shapes,
            child_kind=child_kind,
            parent_group=parent_tp_group,
            parent_world_size=parent_tp_world_size,
            parent_rank=parent_tp_rank,
            label="TP",
            create_group=create_group,
        )

    def _ensure_qkv_child_state_(
        self,
        *,
        parent_param,
        parent_state: dict,
        optim_group: dict,
        child_dist_meta: DionDistMeta,
    ) -> None:
        """Initialize child-specific Q state for one fused QKV child when needed."""
        child_kind = child_dist_meta.qkv_child_kind
        q_key = qkv_state_key("Q", child_kind)
        if q_key in parent_state:
            return

        child_q_init = build_q_init(
            param=parent_param,
            optim_group=optim_group,
            dist_meta=child_dist_meta,
            rank_fraction_default=self.optimizer.defaults["rank_fraction"],
            rank_multiple_of_default=self.optimizer.defaults["rank_multiple_of"],
            use_low_rank_sync=bool(getattr(self.optimizer, "use_low_rank_sync", False)),
            base_training_seed=resolve_base_training_seed(),
            get_replicate_group=self._get_replicate_group,
            make_group_broadcast=lambda process_group: make_group_broadcast(
                process_group,
                group_size=self._group_size,
            ),
        )
        q_layout = child_q_init.q_layout
        if q_layout is None:
            raise RuntimeError(
                "[DION_MISSING_QKV_CHILD_Q_LAYOUT] "
                f"param_uid={child_dist_meta.param_uid} param_name={child_dist_meta.param_name}"
            )
        q_state = init_q_state(
            param=parent_param,
            mixed_precision_config=self.optimizer._mixed_precision_config,
            config=child_dist_meta.param_config,
            dist_meta=child_dist_meta,
            q_layout=q_layout,
            q_seed=child_q_init.q_seed,
            tp_world_size=int(child_q_init.tp_world_size),
            tp_rank=int(child_q_init.tp_rank),
            use_q_unshard=bool(child_q_init.use_q_unshard),
        )
        broadcast_q = child_q_init.broadcast_q
        if broadcast_q is None:
            raise RuntimeError(
                "[DION_MISSING_QKV_CHILD_BROADCAST] "
                f"param_uid={child_dist_meta.param_uid} param_name={child_dist_meta.param_name}"
            )
        broadcast_q(q_state)

        parent_state[q_key] = q_state
        parent_state[qkv_state_key("r", child_kind)] = int(q_layout.r_global)
        parent_state[qkv_state_key("local_shape", child_kind)] = tuple(
            int(dim) for dim in child_dist_meta.local_shape
        )
        parent_state[qkv_state_key("global_shape", child_kind)] = tuple(
            int(dim) for dim in child_dist_meta.global_shape
        )
        parent_state[f"_qkv_{child_kind}_needs_state_replica_q_sync"] = True

    def _build_linear_child_dist_meta(
        self,
        *,
        parent_dist_meta: DionDistMeta,
        child_kind: str,
        split_rows: tuple[int, int],
        child_local_shape: tuple[int, int],
        child_global_shape: tuple[int, int],
    ) -> DionDistMeta:
        """Build optimizer-only child metadata for one fused linear_fc1 child."""
        child_uid = linear_child_param_uid(parent_dist_meta.param_uid, child_kind)
        child_name = linear_child_name(parent_dist_meta.param_name, child_kind)
        (
            child_fs_group,
            child_fs_world_size,
            child_fs_rank,
            child_fs_start_idx,
            child_fs_end_idx,
            fs_row_shard_sizes,
        ) = self._resolve_linear_child_fs_shard_layout(
            parent_dist_meta=parent_dist_meta,
            split_rows=split_rows,
            child_kind=child_kind,
            create_group=False,
        )
        (
            child_tp_group,
            child_tp_world_size,
            child_tp_rank,
            tp_row_shard_start_idx,
            tp_row_shard_end_idx,
            tp_row_shard_sizes,
        ) = self._resolve_linear_child_tp_shard_layout(
            parent_dist_meta=parent_dist_meta,
            split_rows=split_rows,
            child_kind=child_kind,
            create_group=False,
        )
        return build_split_child_dist_meta(
            parent_dist_meta=parent_dist_meta,
            child_uid=child_uid,
            child_name=child_name,
            child_local_shape=child_local_shape,
            child_global_shape=child_global_shape,
            fs_layout=(
                child_fs_group,
                child_fs_world_size,
                child_fs_rank,
                child_fs_start_idx,
                child_fs_end_idx,
                fs_row_shard_sizes,
            ),
            tp_layout=(
                child_tp_group,
                child_tp_world_size,
                child_tp_rank,
                tp_row_shard_start_idx,
                tp_row_shard_end_idx,
                tp_row_shard_sizes,
            ),
            child_fields={
                "linear_split_rows": tuple(int(dim) for dim in split_rows),
                "linear_partition_stride": 1,
                "is_linear_child": True,
                "linear_child_kind": child_kind,
            },
            error_prefix="LINEAR_CHILD",
            use_low_rank_sync=bool(self.optimizer.use_low_rank_sync),
            rank_fraction_default=self.optimizer.defaults["rank_fraction"],
            rank_multiple_of_default=self.optimizer.defaults["rank_multiple_of"],
        )

    def _resolve_linear_child_row_group_layout(
        self,
        *,
        parent_dist_meta: DionDistMeta,
        split_rows: tuple[int, int],
        child_kind: str,
        parent_group,
        parent_world_size: int,
        parent_rank: int,
        label: str,
        create_group: bool,
    ) -> tuple[
        Optional[torch.distributed.ProcessGroup],
        int,
        int,
        int,
        int,
        tuple[int, ...],
    ]:
        """Return child-local ownership for a parent row-sharded split-linear group."""
        parent_global_shape = tuple(int(dim) for dim in parent_dist_meta.global_shape)
        parent_global_rows = int(parent_global_shape[0])
        child_index = 0 if child_kind == "gate" else 1
        child_row_start = 0 if child_index == 0 else int(split_rows[0])
        child_row_end = child_row_start + int(split_rows[child_index])
        child_global_rows = child_row_end - child_row_start
        child_ranges = []
        for rank_idx in range(parent_world_size):
            parent_rank_start, parent_rank_end = compute_fs_shard_range(
                parent_global_rows,
                parent_world_size,
                rank_idx,
            )
            overlap_start = max(int(parent_rank_start), child_row_start)
            overlap_end = min(int(parent_rank_end), child_row_end)
            if overlap_end <= overlap_start:
                child_ranges.append(None)
                continue
            child_ranges.append(
                (int(overlap_start - child_row_start), int(overlap_end - child_row_start))
            )

        return resolve_row_child_layout(
            parent_group=parent_group,
            parent_world_size=parent_world_size,
            parent_rank=parent_rank,
            child_rows=child_global_rows,
            child_ranges=tuple(child_ranges),
            label=label,
            detail=(
                f"param_uid={parent_dist_meta.param_uid} "
                f"param_name={parent_dist_meta.param_name} "
                f"child_kind={child_kind} split_rows={split_rows}"
            ),
            error_prefix="LINEAR_CHILD",
            create_group=create_group,
            make_group=_ensure_child_group,
        ).as_tuple()

    def _resolve_linear_child_fs_shard_layout(
        self,
        *,
        parent_dist_meta: DionDistMeta,
        split_rows: tuple[int, int],
        child_kind: str,
        create_group: bool,
    ) -> tuple[
        Optional[torch.distributed.ProcessGroup],
        int,
        int,
        int,
        int,
        Optional[tuple[int, ...]],
    ]:
        """Return child-local FS ownership for one split linear child."""
        parent_fs_world_size = int(getattr(parent_dist_meta, "fs_world_size", 1))
        parent_fs_rank = int(getattr(parent_dist_meta, "fs_rank", -1))
        parent_fs_shard_dim = int(getattr(parent_dist_meta, "fs_shard_dim", -1))
        parent_fs_group = getattr(parent_dist_meta, "fs_group", None)

        if parent_fs_shard_dim != 0 or parent_fs_world_size <= 1:
            return (
                parent_fs_group,
                parent_fs_world_size,
                parent_fs_rank,
                int(parent_dist_meta.fs_start_idx),
                int(parent_dist_meta.fs_end_idx),
                None,
            )

        return self._resolve_linear_child_row_group_layout(
            parent_dist_meta=parent_dist_meta,
            split_rows=split_rows,
            child_kind=child_kind,
            parent_group=parent_fs_group,
            parent_world_size=parent_fs_world_size,
            parent_rank=parent_fs_rank,
            label="FS",
            create_group=create_group,
        )

    def _resolve_linear_child_tp_shard_layout(
        self,
        *,
        parent_dist_meta: DionDistMeta,
        split_rows: tuple[int, int],
        child_kind: str,
        create_group: bool,
    ) -> tuple[
        Optional[torch.distributed.ProcessGroup],
        int,
        int,
        int,
        int,
        Optional[tuple[int, ...]],
    ]:
        """Return child-local TP ownership for one split linear child."""
        parent_tp_world_size = int(getattr(parent_dist_meta, "tp_world_size", 1))
        parent_tp_rank = int(getattr(parent_dist_meta, "tp_rank", -1))
        parent_tp_shard_dim = int(getattr(parent_dist_meta, "tp_shard_dim", -1))
        parent_tp_group = getattr(parent_dist_meta, "tp_group", None)
        partition_stride = int(getattr(parent_dist_meta, "linear_partition_stride", 1))

        if parent_tp_shard_dim != 0 or parent_tp_world_size <= 1:
            return (
                parent_tp_group,
                parent_tp_world_size,
                parent_tp_rank,
                -1,
                -1,
                None,
            )

        if partition_stride == len(split_rows):
            child_index = 0 if child_kind == "gate" else 1
            child_global_rows = int(split_rows[child_index])
            child_ranges = []
            for rank_idx in range(parent_tp_world_size):
                child_start, child_end = compute_fs_shard_range(
                    child_global_rows,
                    parent_tp_world_size,
                    rank_idx,
                )
                if child_end <= child_start:
                    child_ranges.append(None)
                    continue
                child_ranges.append((int(child_start), int(child_end)))
            return resolve_row_child_layout(
                parent_group=parent_tp_group,
                parent_world_size=parent_tp_world_size,
                parent_rank=parent_tp_rank,
                child_rows=child_global_rows,
                child_ranges=tuple(child_ranges),
                label="TP",
                detail=(
                    f"param_uid={parent_dist_meta.param_uid} "
                    f"param_name={parent_dist_meta.param_name} "
                    f"child_kind={child_kind} split_rows={split_rows}"
                ),
                error_prefix="LINEAR_CHILD",
                create_group=create_group,
                make_group=_ensure_child_group,
            ).as_tuple()

        if partition_stride != 1:
            raise RuntimeError(
                "[DION_LINEAR_UNSUPPORTED_TP_PARTITION_STRIDE] "
                f"param_uid={parent_dist_meta.param_uid} param_name={parent_dist_meta.param_name} "
                f"partition_stride={partition_stride} split_rows={split_rows}"
            )

        return self._resolve_linear_child_row_group_layout(
            parent_dist_meta=parent_dist_meta,
            split_rows=split_rows,
            child_kind=child_kind,
            parent_group=parent_tp_group,
            parent_world_size=parent_tp_world_size,
            parent_rank=parent_tp_rank,
            label="TP",
            create_group=create_group,
        )

    def _init_linear_child_groups(
        self,
        *,
        dist_meta: DionDistMeta,
        split_rows: tuple[int, int],
    ) -> None:
        """Create deterministic split-child subgroups on all ranks before step-time routing."""
        split_rows = self._normalize_linear_split_rows_for_meta(split_rows, dist_meta)
        dist_meta = self._split_parent_dist_meta(dist_meta)
        for child_kind in iter_linear_child_kinds():
            self._resolve_linear_child_fs_shard_layout(
                parent_dist_meta=dist_meta,
                split_rows=split_rows,
                child_kind=child_kind,
                create_group=True,
            )
            self._resolve_linear_child_tp_shard_layout(
                parent_dist_meta=dist_meta,
                split_rows=split_rows,
                child_kind=child_kind,
                create_group=True,
            )

    def _init_qkvg_child_groups(
        self,
        *,
        dist_meta: DionDistMeta,
        split_shapes: tuple[int, int, int, int],
    ) -> None:
        """Create deterministic split-child subgroups on all ranks before step-time routing."""
        dist_meta = self._split_parent_dist_meta(dist_meta)
        for child_kind in iter_qkvg_child_kinds():
            self._resolve_qkvg_child_fs_shard_layout(
                parent_dist_meta=dist_meta,
                split_shapes=split_shapes,
                child_kind=child_kind,
                create_group=True,
            )
            self._resolve_qkvg_child_tp_shard_layout(
                parent_dist_meta=dist_meta,
                split_shapes=split_shapes,
                child_kind=child_kind,
                create_group=True,
            )

    def _init_qkv_child_groups(
        self,
        *,
        dist_meta: DionDistMeta,
        split_shapes: tuple[int, int, int],
    ) -> None:
        """Create deterministic split-child subgroups on all ranks before step-time routing."""
        dist_meta = self._split_parent_dist_meta(dist_meta)
        for child_kind in iter_qkv_child_kinds():
            self._resolve_qkv_child_fs_shard_layout(
                parent_dist_meta=dist_meta,
                split_shapes=split_shapes,
                child_kind=child_kind,
                create_group=True,
            )
            self._resolve_qkv_child_tp_shard_layout(
                parent_dist_meta=dist_meta,
                split_shapes=split_shapes,
                child_kind=child_kind,
                create_group=True,
            )

    def _ensure_linear_child_state_(
        self,
        *,
        parent_param,
        parent_state: dict,
        optim_group: dict,
        child_dist_meta: DionDistMeta,
    ) -> None:
        """Initialize child-specific Q state for one fused linear_fc1 child when needed."""
        child_kind = child_dist_meta.linear_child_kind
        q_key = linear_state_key("Q", child_kind)
        if q_key in parent_state:
            return

        child_q_init = build_q_init(
            param=parent_param,
            optim_group=optim_group,
            dist_meta=child_dist_meta,
            rank_fraction_default=self.optimizer.defaults["rank_fraction"],
            rank_multiple_of_default=self.optimizer.defaults["rank_multiple_of"],
            use_low_rank_sync=bool(getattr(self.optimizer, "use_low_rank_sync", False)),
            base_training_seed=resolve_base_training_seed(),
            get_replicate_group=self._get_replicate_group,
            make_group_broadcast=lambda process_group: make_group_broadcast(
                process_group,
                group_size=self._group_size,
            ),
        )
        q_layout = child_q_init.q_layout
        if q_layout is None:
            raise RuntimeError(
                "[DION_MISSING_LINEAR_CHILD_Q_LAYOUT] "
                f"param_uid={child_dist_meta.param_uid} param_name={child_dist_meta.param_name}"
            )
        q_state = init_q_state(
            param=parent_param,
            mixed_precision_config=self.optimizer._mixed_precision_config,
            config=child_dist_meta.param_config,
            dist_meta=child_dist_meta,
            q_layout=q_layout,
            q_seed=child_q_init.q_seed,
            tp_world_size=int(child_q_init.tp_world_size),
            tp_rank=int(child_q_init.tp_rank),
            use_q_unshard=bool(child_q_init.use_q_unshard),
        )
        broadcast_q = child_q_init.broadcast_q
        if broadcast_q is None:
            raise RuntimeError(
                "[DION_MISSING_LINEAR_CHILD_BROADCAST] "
                f"param_uid={child_dist_meta.param_uid} param_name={child_dist_meta.param_name}"
            )
        broadcast_q(q_state)

        parent_state[q_key] = q_state
        parent_state[linear_state_key("r", child_kind)] = int(q_layout.r_global)
        parent_state[linear_state_key("local_shape", child_kind)] = tuple(
            int(dim) for dim in child_dist_meta.local_shape
        )
        parent_state[linear_state_key("global_shape", child_kind)] = tuple(
            int(dim) for dim in child_dist_meta.global_shape
        )
        parent_state[f"_linear_{child_kind}_needs_state_replica_q_sync"] = True

    def _expand_split_qkvg_params(
        self,
        *,
        param,
        grad,
        optimizer_state,
        optim_group,
        config,
        dist_meta,
    ):
        """Expand one fused QKVG parent into optimizer-only child Dion step params."""
        if not bool(self.optimizer.defaults.get("split_qkv", False)):
            return None
        if optim_group.get("algorithm", "dion") != "dion":
            return None
        if dist_meta is None or not bool(getattr(dist_meta, "is_dion_param", False)):
            return None
        split_shapes = resolve_qkvg_split_shapes(
            param=param,
            optimizer_state=optimizer_state,
            dist_meta=dist_meta,
        )
        if split_shapes is None:
            return None

        momentum = optimizer_state.get("momentum", None)
        if momentum is None:
            raise RuntimeError(
                "[DION_QKVG_SPLIT_MISSING_MOMENTUM] "
                f"param_uid={getattr(dist_meta, 'param_uid', None)} "
                f"param_name={getattr(dist_meta, 'param_name', '')}"
            )
        (
            parent_param_view,
            parent_grad_view,
            parent_momentum_view,
            parent_local_shape,
            split_parent_dist_meta,
        ) = self._split_parent_views(
            param=param,
            grad=grad,
            momentum=momentum,
            dist_meta=dist_meta,
        )

        child_step_params: list[DionStepParam] = []
        for child_kind in iter_qkvg_child_kinds():
            if not qkvg_child_has_local_overlap(split_shapes, split_parent_dist_meta, child_kind):
                continue
            child_local_shape = qkvg_child_local_shape(
                parent_local_shape,
                split_shapes,
                child_kind,
                dist_meta=split_parent_dist_meta,
            )
            child_global_shape = qkvg_child_global_shape(
                tuple(int(dim) for dim in split_parent_dist_meta.global_shape),
                split_shapes,
                child_kind,
            )
            child_dist_meta = self._build_qkvg_child_dist_meta(
                parent_dist_meta=split_parent_dist_meta,
                child_kind=child_kind,
                split_shapes=split_shapes,
                child_local_shape=child_local_shape,
                child_global_shape=child_global_shape,
            )
            self._ensure_qkvg_child_state_(
                parent_param=param,
                parent_state=optimizer_state,
                optim_group=optim_group,
                child_dist_meta=child_dist_meta,
            )

            child_param = extract_qkvg_child(
                parent_param_view,
                split_shapes,
                child_kind,
                dist_meta=split_parent_dist_meta,
            )
            child_grad = extract_qkvg_child(
                parent_grad_view,
                split_shapes,
                child_kind,
                dist_meta=split_parent_dist_meta,
            )
            child_momentum = extract_qkvg_child(
                parent_momentum_view,
                split_shapes,
                child_kind,
                dist_meta=split_parent_dist_meta,
            )
            q_key = qkvg_state_key("Q", child_kind)
            r_key = qkvg_state_key("r", child_kind)
            local_shape_key = qkvg_state_key("local_shape", child_kind)
            global_shape_key = qkvg_state_key("global_shape", child_kind)
            sync_key = f"_qkvg_{child_kind}_needs_state_replica_q_sync"
            self._refresh_dion_param_config(
                dist_meta=child_dist_meta,
                optim_group=optim_group,
                local_shape=tuple(int(dim) for dim in optimizer_state[local_shape_key]),
                r_global_override=optimizer_state[r_key],
            )

            child_state = {
                "momentum": child_momentum,
                "Q": optimizer_state[q_key],
                "r": int(optimizer_state[r_key]),
                "local_shape": tuple(int(dim) for dim in optimizer_state[local_shape_key]),
                "global_shape": tuple(
                    int(dim) for dim in optimizer_state[global_shape_key]
                ),
                "_needs_state_replica_q_sync": bool(optimizer_state.get(sync_key, False)),
            }

            def _commit_update(
                updated_param,
                updated_momentum,
                *,
                parent_dist_meta=split_parent_dist_meta,
                split_shapes=split_shapes,
                child_kind=child_kind,
            ):
                scatter_qkvg_child_(
                    parent_param_view,
                    updated_param,
                    split_shapes,
                    child_kind,
                    dist_meta=parent_dist_meta,
                )
                scatter_qkvg_child_(
                    parent_momentum_view,
                    updated_momentum,
                    split_shapes,
                    child_kind,
                    dist_meta=parent_dist_meta,
                )

            child_step_params.append(
                DionStepParam(
                    param=child_param,
                    grad=child_grad,
                    optimizer_state=child_state,
                    optim_group=optim_group,
                    config=child_dist_meta.param_config,
                    dist_meta=child_dist_meta,
                    post_q_sync=_make_split_child_q_sync_hook(
                        optimizer_state,
                        sync_key,
                    ),
                    commit_update=_commit_update,
                )
            )

        if not child_step_params:
            raise RuntimeError(
                "[DION_QKVG_SPLIT_NO_LOCAL_CHILDREN] "
                f"param_uid={getattr(dist_meta, 'param_uid', None)} "
                f"param_name={getattr(dist_meta, 'param_name', '')} "
                f"split_shapes={split_shapes} "
                f"parent_local_shape={parent_local_shape}"
            )

        return child_step_params

    def _expand_split_qkv_params(
        self,
        *,
        param,
        grad,
        optimizer_state,
        optim_group,
        config,
        dist_meta,
    ):
        """Expand one fused QKV parent into optimizer-only child Dion step params."""
        if not bool(self.optimizer.defaults.get("split_qkv", False)):
            return None
        if optim_group.get("algorithm", "dion") != "dion":
            return None
        if dist_meta is None or not bool(getattr(dist_meta, "is_dion_param", False)):
            return None
        split_shapes = resolve_qkv_split_shapes(
            param=param,
            optimizer_state=optimizer_state,
            dist_meta=dist_meta,
        )
        if split_shapes is None:
            return None

        momentum = optimizer_state.get("momentum", None)
        if momentum is None:
            raise RuntimeError(
                "[DION_QKV_SPLIT_MISSING_MOMENTUM] "
                f"param_uid={getattr(dist_meta, 'param_uid', None)} "
                f"param_name={getattr(dist_meta, 'param_name', '')}"
            )
        (
            parent_param_view,
            parent_grad_view,
            parent_momentum_view,
            parent_local_shape,
            split_parent_dist_meta,
        ) = self._split_parent_views(
            param=param,
            grad=grad,
            momentum=momentum,
            dist_meta=dist_meta,
        )

        child_step_params: list[DionStepParam] = []
        for child_kind in iter_qkv_child_kinds():
            if not qkv_child_has_local_overlap(split_shapes, split_parent_dist_meta, child_kind):
                continue
            child_local_shape = qkv_child_local_shape(
                parent_local_shape,
                split_shapes,
                child_kind,
                dist_meta=split_parent_dist_meta,
            )
            child_global_shape = qkv_child_global_shape(
                tuple(int(dim) for dim in split_parent_dist_meta.global_shape),
                split_shapes,
                child_kind,
            )
            child_dist_meta = self._build_qkv_child_dist_meta(
                parent_dist_meta=split_parent_dist_meta,
                child_kind=child_kind,
                split_shapes=split_shapes,
                child_local_shape=child_local_shape,
                child_global_shape=child_global_shape,
            )
            self._ensure_qkv_child_state_(
                parent_param=param,
                parent_state=optimizer_state,
                optim_group=optim_group,
                child_dist_meta=child_dist_meta,
            )

            child_param = extract_qkv_child(
                parent_param_view,
                split_shapes,
                child_kind,
                dist_meta=split_parent_dist_meta,
            )
            child_grad = extract_qkv_child(
                parent_grad_view,
                split_shapes,
                child_kind,
                dist_meta=split_parent_dist_meta,
            )
            child_momentum = extract_qkv_child(
                parent_momentum_view,
                split_shapes,
                child_kind,
                dist_meta=split_parent_dist_meta,
            )
            q_key = qkv_state_key("Q", child_kind)
            r_key = qkv_state_key("r", child_kind)
            local_shape_key = qkv_state_key("local_shape", child_kind)
            global_shape_key = qkv_state_key("global_shape", child_kind)
            sync_key = f"_qkv_{child_kind}_needs_state_replica_q_sync"
            self._refresh_dion_param_config(
                dist_meta=child_dist_meta,
                optim_group=optim_group,
                local_shape=tuple(int(dim) for dim in optimizer_state[local_shape_key]),
                r_global_override=optimizer_state[r_key],
            )

            child_state = {
                "momentum": child_momentum,
                "Q": optimizer_state[q_key],
                "r": int(optimizer_state[r_key]),
                "local_shape": tuple(int(dim) for dim in optimizer_state[local_shape_key]),
                "global_shape": tuple(
                    int(dim) for dim in optimizer_state[global_shape_key]
                ),
                "_needs_state_replica_q_sync": bool(optimizer_state.get(sync_key, False)),
            }

            def _commit_update(
                updated_param,
                updated_momentum,
                *,
                parent_dist_meta=split_parent_dist_meta,
                split_shapes=split_shapes,
                child_kind=child_kind,
            ):
                scatter_qkv_child_(
                    parent_param_view,
                    updated_param,
                    split_shapes,
                    child_kind,
                    dist_meta=parent_dist_meta,
                )
                scatter_qkv_child_(
                    parent_momentum_view,
                    updated_momentum,
                    split_shapes,
                    child_kind,
                    dist_meta=parent_dist_meta,
                )

            child_step_params.append(
                DionStepParam(
                    param=child_param,
                    grad=child_grad,
                    optimizer_state=child_state,
                    optim_group=optim_group,
                    config=child_dist_meta.param_config,
                    dist_meta=child_dist_meta,
                    post_q_sync=_make_split_child_q_sync_hook(
                        optimizer_state,
                        sync_key,
                    ),
                    commit_update=_commit_update,
                )
            )

        if not child_step_params:
            raise RuntimeError(
                "[DION_QKV_SPLIT_NO_LOCAL_CHILDREN] "
                f"param_uid={getattr(dist_meta, 'param_uid', None)} "
                f"param_name={getattr(dist_meta, 'param_name', '')} "
                f"split_shapes={split_shapes} "
                f"parent_local_shape={parent_local_shape}"
            )

        return child_step_params

    def _expand_split_linear_params(
        self,
        *,
        param,
        grad,
        optimizer_state,
        optim_group,
        config,
        dist_meta,
    ):
        """Expand one fused linear_fc1 parent into optimizer-only gate/up child Dion step params."""
        del config
        if not bool(self.optimizer.defaults.get("split_linear", False)):
            return None
        if optim_group.get("algorithm", "dion") != "dion":
            return None
        if dist_meta is None or not bool(getattr(dist_meta, "is_dion_param", False)):
            return None
        split_rows = resolve_linear_split_rows(
            optimizer_state=optimizer_state,
            dist_meta=dist_meta,
        )
        if split_rows is None:
            return None
        split_rows = self._normalize_linear_split_rows_for_meta(split_rows, dist_meta)

        momentum = optimizer_state.get("momentum", None)
        if momentum is None:
            raise RuntimeError(
                "[DION_LINEAR_SPLIT_MISSING_MOMENTUM] "
                f"param_uid={getattr(dist_meta, 'param_uid', None)} "
                f"param_name={getattr(dist_meta, 'param_name', '')}"
            )
        (
            parent_param_view,
            parent_grad_view,
            parent_momentum_view,
            parent_local_shape,
            split_parent_dist_meta,
        ) = self._split_parent_views(
            param=param,
            grad=grad,
            momentum=momentum,
            dist_meta=dist_meta,
        )
        m_local, n_local = parent_local_shape
        parent_global_shape = get_global_shape(
            split_parent_dist_meta,
            m_local,
            n_local,
        )

        child_step_params: list[DionStepParam] = []
        for child_kind in iter_linear_child_kinds():
            if not linear_child_has_local_overlap(split_rows, split_parent_dist_meta, child_kind):
                continue
            child_local_shape = linear_child_local_shape(
                parent_local_shape,
                split_rows,
                split_parent_dist_meta,
                child_kind,
            )
            child_global_shape = linear_child_global_shape(
                tuple(int(dim) for dim in parent_global_shape),
                split_rows,
                child_kind,
            )
            child_dist_meta = self._build_linear_child_dist_meta(
                parent_dist_meta=split_parent_dist_meta,
                child_kind=child_kind,
                split_rows=split_rows,
                child_local_shape=child_local_shape,
                child_global_shape=child_global_shape,
            )
            self._ensure_linear_child_state_(
                parent_param=param,
                parent_state=optimizer_state,
                optim_group=optim_group,
                child_dist_meta=child_dist_meta,
            )

            child_param = read_linear_child(
                parent_param_view,
                split_rows,
                split_parent_dist_meta,
                child_kind,
            )
            child_grad = read_linear_child(
                parent_grad_view,
                split_rows,
                split_parent_dist_meta,
                child_kind,
            )
            child_momentum = read_linear_child(
                parent_momentum_view,
                split_rows,
                split_parent_dist_meta,
                child_kind,
            )
            q_key = linear_state_key("Q", child_kind)
            r_key = linear_state_key("r", child_kind)
            local_shape_key = linear_state_key("local_shape", child_kind)
            global_shape_key = linear_state_key("global_shape", child_kind)
            sync_key = f"_linear_{child_kind}_needs_state_replica_q_sync"
            self._refresh_dion_param_config(
                dist_meta=child_dist_meta,
                optim_group=optim_group,
                local_shape=tuple(int(dim) for dim in optimizer_state[local_shape_key]),
                r_global_override=optimizer_state[r_key],
            )

            child_state = {
                "momentum": child_momentum,
                "Q": optimizer_state[q_key],
                "r": int(optimizer_state[r_key]),
                "local_shape": tuple(int(dim) for dim in optimizer_state[local_shape_key]),
                "global_shape": tuple(
                    int(dim) for dim in optimizer_state[global_shape_key]
                ),
                "_needs_state_replica_q_sync": bool(optimizer_state.get(sync_key, False)),
            }

            def _commit_update(
                updated_param,
                updated_momentum,
                *,
                parent_dist_meta=split_parent_dist_meta,
                split_rows=split_rows,
                child_kind=child_kind,
            ):
                write_linear_child_(
                    parent_param_view,
                    updated_param,
                    split_rows,
                    parent_dist_meta,
                    child_kind,
                )
                write_linear_child_(
                    parent_momentum_view,
                    updated_momentum,
                    split_rows,
                    parent_dist_meta,
                    child_kind,
                )

            child_step_params.append(
                DionStepParam(
                    param=child_param,
                    grad=child_grad,
                    optimizer_state=child_state,
                    optim_group=optim_group,
                    config=child_dist_meta.param_config,
                    dist_meta=child_dist_meta,
                    post_q_sync=_make_split_child_q_sync_hook(
                        optimizer_state,
                        sync_key,
                    ),
                    commit_update=_commit_update,
                )
            )

        if not child_step_params:
            raise RuntimeError(
                "[DION_LINEAR_SPLIT_NO_LOCAL_CHILDREN] "
                f"param_uid={getattr(dist_meta, 'param_uid', None)} "
                f"param_name={getattr(dist_meta, 'param_name', '')} "
                f"split_rows={split_rows} "
                f"parent_local_shape={parent_local_shape}"
            )

        return child_step_params

    def _expand_split_dion_params(
        self,
        *,
        param,
        grad,
        optimizer_state,
        optim_group,
        config,
        dist_meta,
    ):
        """Expand optimizer-only Dion children when one parent carries a fused update surface."""
        qkvg_children = self._expand_split_qkvg_params(
            param=param,
            grad=grad,
            optimizer_state=optimizer_state,
            optim_group=optim_group,
            config=config,
            dist_meta=dist_meta,
        )
        if qkvg_children is not None:
            return qkvg_children
        qkv_children = self._expand_split_qkv_params(
            param=param,
            grad=grad,
            optimizer_state=optimizer_state,
            optim_group=optim_group,
            config=config,
            dist_meta=dist_meta,
        )
        if qkv_children is not None:
            return qkv_children
        return self._expand_split_linear_params(
            param=param,
            grad=grad,
            optimizer_state=optimizer_state,
            optim_group=optim_group,
            config=config,
            dist_meta=dist_meta,
        )

    def _ensure_optimizer_state(self, param, optim_group):
        """Own distributed state remap and metadata recovery at the adapter boundary."""
        return ensure_optimizer_state(
            optimizer=self.optimizer,
            param=param,
            optim_group=optim_group,
            dion_state_param_by_uid=self._dion_state_param_by_uid,
            dion_dist_meta_by_uid=self._dion_dist_meta_by_uid,
            init_optimizer_state=self._init_optimizer_state,
        )

    def _ensure_dion_checkpoint_state(self) -> None:
        """Initialize Dion optimizer state before distributed checkpoint load templates."""
        for optim_group in self.optimizer.param_groups:
            for param in optim_group["params"]:
                state = self._ensure_optimizer_state(param, optim_group)
                dist_meta = self.optimizer.dist_metas.get(param, None)
                if dist_meta is None or not bool(getattr(dist_meta, "is_dion_param", False)):
                    self._ensure_scalar_checkpoint_state(param, state, optim_group)
                else:
                    self._ensure_split_checkpoint_state(param, state, optim_group, dist_meta)

    def _checkpoint_split_parent_layout(self, dist_meta):
        """Return parent split metadata and shapes for checkpoint templates."""
        split_parent_dist_meta = self._split_parent_dist_meta(dist_meta)
        parent_local_shape = getattr(split_parent_dist_meta, "local_shape", None)
        if parent_local_shape is None:
            parent_local_shape = getattr(split_parent_dist_meta, "shape", None)
        parent_global_shape = getattr(split_parent_dist_meta, "global_shape", None)
        if parent_local_shape is None or parent_global_shape is None:
            return None
        return (
            split_parent_dist_meta,
            tuple(int(dim) for dim in parent_local_shape),
            tuple(int(dim) for dim in parent_global_shape),
        )

    def _iter_split_child_dist_metas(self, *, param, state, dist_meta):
        """Yield split child metadata needed to initialize checkpoint load templates."""
        parent_layout = self._checkpoint_split_parent_layout(dist_meta)
        if parent_layout is None:
            return
        split_parent_dist_meta, parent_local_shape, parent_global_shape = parent_layout

        if bool(self.optimizer.defaults.get("split_qkv", False)):
            qkvg_shapes = resolve_qkvg_split_shapes(
                param=param,
                optimizer_state=state,
                dist_meta=dist_meta,
            )
            if qkvg_shapes is not None:
                for child_kind in iter_qkvg_child_kinds():
                    if not qkvg_child_has_local_overlap(
                        qkvg_shapes, split_parent_dist_meta, child_kind
                    ):
                        continue
                    child_local_shape = qkvg_child_local_shape(
                        parent_local_shape,
                        qkvg_shapes,
                        child_kind,
                        dist_meta=split_parent_dist_meta,
                    )
                    child_global_shape = qkvg_child_global_shape(
                        parent_global_shape,
                        qkvg_shapes,
                        child_kind,
                    )
                    yield (
                        "qkvg",
                        self._build_qkvg_child_dist_meta(
                            parent_dist_meta=split_parent_dist_meta,
                            child_kind=child_kind,
                            split_shapes=qkvg_shapes,
                            child_local_shape=child_local_shape,
                            child_global_shape=child_global_shape,
                        ),
                    )
                return

            qkv_shapes = resolve_qkv_split_shapes(
                param=param,
                optimizer_state=state,
                dist_meta=dist_meta,
            )
            if qkv_shapes is not None:
                for child_kind in iter_qkv_child_kinds():
                    if not qkv_child_has_local_overlap(
                        qkv_shapes, split_parent_dist_meta, child_kind
                    ):
                        continue
                    child_local_shape = qkv_child_local_shape(
                        parent_local_shape,
                        qkv_shapes,
                        child_kind,
                        dist_meta=split_parent_dist_meta,
                    )
                    child_global_shape = qkv_child_global_shape(
                        parent_global_shape,
                        qkv_shapes,
                        child_kind,
                    )
                    yield (
                        "qkv",
                        self._build_qkv_child_dist_meta(
                            parent_dist_meta=split_parent_dist_meta,
                            child_kind=child_kind,
                            split_shapes=qkv_shapes,
                            child_local_shape=child_local_shape,
                            child_global_shape=child_global_shape,
                        ),
                    )
                return

        if not bool(self.optimizer.defaults.get("split_linear", False)):
            return
        split_rows = resolve_linear_split_rows(optimizer_state=state, dist_meta=dist_meta)
        if split_rows is None:
            return
        for child_kind in iter_linear_child_kinds():
            if not linear_child_has_local_overlap(
                split_rows,
                split_parent_dist_meta,
                child_kind,
            ):
                continue
            child_local_shape = linear_child_local_shape(
                parent_local_shape,
                split_rows,
                dist_meta=split_parent_dist_meta,
                child_kind=child_kind,
            )
            child_global_shape = linear_child_global_shape(
                parent_global_shape,
                split_rows,
                child_kind,
            )
            yield (
                "linear",
                self._build_linear_child_dist_meta(
                    parent_dist_meta=split_parent_dist_meta,
                    child_kind=child_kind,
                    split_rows=split_rows,
                    child_local_shape=child_local_shape,
                    child_global_shape=child_global_shape,
                ),
            )

    def _ensure_split_checkpoint_state(self, param, state, optim_group, dist_meta) -> None:
        """Initialize lazy split-child Dion state for checkpoint load templates."""
        if optim_group.get("algorithm", "dion") != "dion":
            return
        if not bool(self.optimizer.defaults.get("split_qkv", False)) and not bool(
            self.optimizer.defaults.get("split_linear", False)
        ):
            return

        for split_kind, child_dist_meta in self._iter_split_child_dist_metas(
            param=param,
            state=state,
            dist_meta=dist_meta,
        ):
            if split_kind == "qkvg":
                self._ensure_qkvg_child_state_(
                    parent_param=param,
                    parent_state=state,
                    optim_group=optim_group,
                    child_dist_meta=child_dist_meta,
                )
            elif split_kind == "qkv":
                self._ensure_qkv_child_state_(
                    parent_param=param,
                    parent_state=state,
                    optim_group=optim_group,
                    child_dist_meta=child_dist_meta,
                )
            elif split_kind == "linear":
                self._ensure_linear_child_state_(
                    parent_param=param,
                    parent_state=state,
                    optim_group=optim_group,
                    child_dist_meta=child_dist_meta,
                )
            else:
                raise RuntimeError(f"[DION_INVALID_SPLIT_KIND] {split_kind!r}")

    def _scalar_checkpoint_optimizer(self, optim_group) -> str:
        default_scalar = self.optimizer.defaults.get("scalar_optimizer", "adamw")
        group_algorithm = optim_group.get("algorithm", None)
        if group_algorithm is None or group_algorithm == "dion":
            scalar_optimizer = optim_group.get("scalar_optimizer", default_scalar)
        else:
            scalar_optimizer = group_algorithm
        if scalar_optimizer == "adam":
            return "adamw"
        if scalar_optimizer not in {"adamw", "lion"}:
            raise RuntimeError(f"[DION_INVALID_SCALAR_CHECKPOINT_OPT] {scalar_optimizer!r}")
        return str(scalar_optimizer)

    def _ensure_scalar_checkpoint_state(self, param, state, optim_group) -> None:
        """Match scalar checkpoint tensor keys to the active scalar optimizer."""
        if "first_moment" in state and "exp_avg" in state:
            raise RuntimeError(
                "[DION_SCALAR_STATE_LAYOUT_CONFLICT] found both first_moment and exp_avg"
            )
        if "first_moment" not in state:
            if "exp_avg" in state:
                state["first_moment"] = state.pop("exp_avg")
            elif "momentum" in state:
                state["first_moment"] = state.pop("momentum")
            else:
                momentum_dtype = self.optimizer._mixed_precision_config.momentum_dtype
                if momentum_dtype is None:
                    momentum_dtype = param.dtype
                state["first_moment"] = torch.zeros_like(param, dtype=momentum_dtype)

        if self._scalar_checkpoint_optimizer(optim_group) != "adamw":
            return
        if "second_moment" in state and "exp_avg_sq" in state:
            raise RuntimeError(
                "[DION_SCALAR_STATE_LAYOUT_CONFLICT] found both second_moment and exp_avg_sq"
            )
        if "second_moment" not in state:
            if "exp_avg_sq" in state:
                state["second_moment"] = state.pop("exp_avg_sq")
            elif "variance" in state:
                state["second_moment"] = state.pop("variance")
            else:
                variance_dtype = self.optimizer._mixed_precision_config.variance_dtype
                if variance_dtype is None:
                    variance_dtype = param.dtype
                state["second_moment"] = torch.zeros_like(param, dtype=variance_dtype)

    def _log_grad_issue(
        self,
        kind: str,
        model_param: torch.nn.Parameter,
        shard_param: Optional[torch.nn.Parameter] = None,
        **extra,
    ) -> None:
        """Emit one structured grad-route error log."""
        payload = {
            "kind": kind,
            "param": self._find_param_name(model_param)
            or getattr(model_param, "_param_name", f"id_{id(model_param)}"),
            "model_shape": tuple(model_param.shape),
            "shard_shape": tuple(shard_param.shape) if shard_param is not None else None,
        }
        payload.update(extra)
        logger.error("[DION_GRAD_ISSUE] %s", payload)

    def _restore_bucket_param_views_(
        self, *, dion_only: bool = False, matrix_only: bool = False
    ) -> None:
        """Make selected model params alias canonical bucket.param_data before step updates."""
        dion_only = bool(dion_only or matrix_only)
        if not hasattr(self, "buffers"):
            return
        for buffer in self.buffers:
            for bucket in buffer.buckets:
                if getattr(bucket, "param_data", None) is None:
                    continue
                dion_layout = getattr(bucket, "matrix_layout", None)
                if dion_only:
                    dion_params = collect_matrix_bucket_params(dion_layout)
                    if not dion_params:
                        continue
                    self._set_bucket_param_views(bucket, copy_data=True, params=dion_params)
                else:
                    self._set_bucket_param_views(bucket, copy_data=True)
                    if dion_layout is None or not dion_layout.has_params:
                        continue

                if dion_layout is None or not dion_layout.has_params:
                    continue
                restore_matrix_shards_from_bucket_(
                    matrix_layout=dion_layout,
                    get_full_view_2d=lambda entry: self._bucket_matrix_param_view(bucket, entry),
                    update_data_shard=self._update_data_shard,
                    param_name=lambda param: self._param_name(param) or f'id_{id(param)}',
                )
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
            assert_shard_aliased(
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
        - standard model side canonical grad: `model_param.main_grad`
        - Dion step grad: adapter-stored local shard surface
        - optimizer shard grad: stock DO local view on `shard_param.grad` or `shard_param.decoupled_grad`

        Dion parameters still need their own step surface because the optimizer shard objects are
        separate FP32 tensors and Dion reads FS/TP-local grad views. At the same time,
        optimizer grad fields must keep stock Megatron-Core DO semantics for grad norm and
        clipping.
        """
        set_optimizer_shard_grads(
            is_stub_optimizer=self.is_stub_optimizer,
            use_megatron_fsdp=self.ddp_config.use_megatron_fsdp,
            use_precision_aware_optimizer=bool(
                getattr(self.config, "use_precision_aware_optimizer_no_fp8_or_ds_fp8", False)
            ),
            model_float16_groups=self.model_float16_groups,
            shard_float16_groups=self.shard_float16_groups,
            model_fp32_groups=self.model_fp32_groups,
            shard_fp32_groups=self.shard_fp32_groups,
            shard_fp32_from_float16_groups=self.shard_fp32_from_float16_groups,
            get_param_range=self._get_model_param_range_map,
            get_local_grad=self._get_local_grad,
            log_grad_issue=self._log_grad_issue,
            release_rs_buffers=self._release_rs_buffers,
        )

    def get_main_grads_for_grad_norm(self) -> List[torch.Tensor]:
        """Return grad-norm inputs on the gradient surface."""
        return grad_norm_inputs(self)

    @torch.no_grad()
    def get_grad_norm(self):
        """Compute grad norm on the current gradient surface."""
        return compute_grad_norm(self)

    def clip_grad_norm(self, clip_grad: float) -> float:
        """Clip optimizer grads using the authoritative Dion grad norm."""
        return self.clip_matrix_grad_norm(
            clip_grad,
            is_matrix_model_param=lambda param: getattr(param, "is_dion_param", False),
        )

    def requires_individual_grad_norm_in_chain(self) -> bool:
        """Shared-group chained grad norm must respect Dion's exact local accumulation."""
        return True

    def prepare_grads(self) -> bool:
        """
        Match standard Megatron-Core prepare_grads semantics.

        The standard training loop already calls `finalize_model_grads()`, which in turn
        calls `model_chunk.finish_grad_sync()` before `optimizer.step()`. Repeating
        `finish_grad_sync()` here is incorrect for the multi-instance path because mixed
        standard RS-local buffers have already been flushed and cleared by the first call.
        """
        return super().prepare_grads()

    def _copy_model_params_to_main_params(self, state_dict=None):
        """Copy model params to main params with FS sharding awareness.

        Handles FS-sharded parameters correctly when loading from checkpoint.
        """
        from ...cpu_offloading import HybridDeviceOptimizer
        use_precision_aware_optimizer = bool(
            getattr(self.config, "use_precision_aware_optimizer_no_fp8_or_ds_fp8", False)
        )
        main_shard_groups = getattr(self, "shard_fp32_from_float16_groups", None)
        copy_model_params_to_main_shards(
            is_hybrid_device_optimizer=isinstance(self.optimizer, HybridDeviceOptimizer),
            hybrid_optimizer_update=(
                self.optimizer.update_fp32_param_by_new_param
                if isinstance(self.optimizer, HybridDeviceOptimizer)
                else None
            ),
            use_megatron_fsdp=self.ddp_config.use_megatron_fsdp,
            use_precision_aware_optimizer=use_precision_aware_optimizer,
            state_dict=state_dict,
            build_model_param_to_state_dict_param_map=self._build_model_param_to_state_dict_param_map,
            model_float16_groups=self.model_float16_groups,
            main_shard_groups=main_shard_groups,
            model_fp32_groups=self.model_fp32_groups,
            shard_fp32_groups=self.shard_fp32_groups,
            get_model_param_range_map=self._get_model_param_range_map,
            get_matrix_shard_layout=self._param_shard_layout,
        )

    def _copy_main_params_to_model_params(self):
        """Copy parameters with efficient batch flattening for 2D params.

        Copy updated optimizer shards into local model-param shards.

        Stock DO treats bucket.param_data as the canonical post-step restore / gather
        surface. Point forward-visible model params to that bucket storage before
        writing local shards so the subsequent param-gather path never depends on
        stale lingering aliases.
        """
        use_precision_aware_optimizer = bool(
            getattr(self.config, "use_precision_aware_optimizer_no_fp8_or_ds_fp8", False)
        )
        main_shard_groups = getattr(self, "shard_fp32_from_float16_groups", None)

        seen_main = set()
        if main_shard_groups is not None:
            for model_group, shard_group in zip(self.model_float16_groups, main_shard_groups):
                for model_param, shard_param in zip(model_group, shard_group):
                    if shard_param is None or id(model_param) in seen_main:
                        continue
                    seen_main.add(id(model_param))
        for model_group, shard_group in zip(self.model_fp32_groups, self.shard_fp32_groups):
            for model_param, shard_param in zip(model_group, shard_group):
                if shard_param is None or id(model_param) in seen_main:
                    continue
                seen_main.add(id(model_param))

        if (
            not self.is_stub_optimizer
            and not self.ddp_config.use_megatron_fsdp
            and not use_precision_aware_optimizer
        ):
            quantize_param_shard(
                *self._get_fp8_params_and_shard_fp32_from_fp8(), self.data_parallel_group
            )

        def copy_fsdp_main_to_model_weights() -> None:
            for model_chunk in self.model_chunks:
                model_chunk.param_and_grad_buffer.copy_main_weights_to_model_weights()

        self._empty_range_warning_count = copy_main_params_to_model_shards(
            is_stub_optimizer=self.is_stub_optimizer,
            use_megatron_fsdp=self.ddp_config.use_megatron_fsdp,
            copy_fsdp_main_to_model_weights=copy_fsdp_main_to_model_weights,
            use_precision_aware_optimizer=use_precision_aware_optimizer,
            model_float16_groups=self.model_float16_groups,
            main_shard_groups=main_shard_groups,
            shard_float16_groups=self.shard_float16_groups,
            model_fp32_groups=self.model_fp32_groups,
            shard_fp32_groups=self.shard_fp32_groups,
            get_data_shard=self._get_data_shard,
            get_param_range_map=self._get_model_param_range_map,
            get_matrix_shard_layout=self._param_shard_layout,
            get_bucket_param_data=self._bucket_param_data,
            mark_buckets_full_param_ready=self._mark_buckets_full_param_ready,
            check_main_shards=self._check_main_shards,
            restore_model_params_to_canonical_bucket_storage=self._restore_bucket_param_views_,
            empty_range_warning_count=getattr(self, '_empty_range_warning_count', 0),
        )

    def _refresh_param_groups(self):
        """Update optimizer param groups after rebuilding."""
        from ...cpu_offloading import HybridDeviceOptimizer

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

        finish_bucket_group_grad_sync(self.per_model_bucket_groups)

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

        if timers is not None:
            timers('params-all-gather', log_level=1).start(
                barrier=self.config.barrier_with_L1_time
            )
        if self.ddp_config.use_megatron_fsdp:
            for model_chunk in self.model_chunks:
                model_chunk.start_param_sync()
        else:
            if not self.ddp_config.overlap_param_gather:
                for model_chunk in self.model_chunks:
                    model_chunk.start_param_sync()
        if timers is not None:
            timers('params-all-gather').stop()
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
        `param_uid`.
        """
        from ....dist_checkpointing.mapping import ShardedObject, ShardedTensor

        if model_sharded_state_dict is None:
            model_sharded_state_dict = {}

        dp_rank = self.data_parallel_group.rank()
        dp_size = self.data_parallel_group.size()
        base_key = f'optimizer.distributed.dp_group_idx_{self.data_parallel_group_idx}'
        common_replica_id = (self.distributed_optimizer_instance_id, 0, dp_rank)
        requested_type = resolve_matrix_checkpoint_sharding_type(sharding_type, metadata)
        tp_group = self._resolve_dion_tp_group()
        tp_size = 1 if tp_group is None else self._group_size(tp_group)
        fs_size = int(self.fs_size)
        fs_rank = int(self.fs_rank)
        if fs_size <= 0 or fs_rank < 0 or fs_rank >= fs_size:
            raise RuntimeError(
                "[Dion] invalid FS checkpoint state coordinates: "
                f"fs_size={fs_size} fs_rank={fs_rank}"
            )
        if is_loading:
            self._ensure_dion_checkpoint_state()

        state_replica_group = getattr(self, "state_replica_group", None)
        state_replica_size = (
            1 if state_replica_group is None else self._group_size(state_replica_group)
        )
        state_replica_rank = (
            0 if state_replica_group is None else self._group_rank(state_replica_group)
        )
        state_owner_dp_rank = dp_rank
        if state_replica_group is not None:
            dp_ranks = self._get_group_ranks_for_checkpoint(self.data_parallel_group)
            state_replica_ranks = self._get_group_ranks_for_checkpoint(state_replica_group)
            if not state_replica_ranks:
                raise RuntimeError("[Dion] empty state-replica group for checkpoint")
            state_owner_global_rank = int(state_replica_ranks[0])
            if state_owner_global_rank not in dp_ranks:
                raise RuntimeError(
                    "[Dion] checkpoint state replica owner is outside data-parallel group: "
                    f"owner={state_owner_global_rank} dp_ranks={dp_ranks}"
                )
            state_owner_dp_rank = int(dp_ranks.index(state_owner_global_rank))

        checkpoint_metadata = build_matrix_checkpoint_metadata(
            dp_size=dp_size,
            fs_size=fs_size,
            tp_size=tp_size,
            rp_size=int(self._rp_size) if self._rp_size is not None else 1,
            state_replica_size=state_replica_size,
            requested_type=requested_type,
            topology_signature=self._dion_checkpoint_topology_signature(),
            backend_state_spec=self._require_matrix_backend().state_spec(),
        )
        common_state = self.state_dict()
        return build_distributed_checkpoint_state(
            common_state=common_state,
            param_groups=self.optimizer.param_groups,
            optimizer_state=self.optimizer.state,
            get_param_key=self._shard_param_uid,
            base_key=base_key,
            common_replica_id=common_replica_id,
            state_global_shape=(dp_size,),
            state_global_offset=(dp_rank,),
            state_replica_id=(state_replica_rank,),
            checkpoint_metadata=checkpoint_metadata,
            sharded_object_cls=ShardedObject,
            sharded_tensor_cls=ShardedTensor,
            state_rank_key=str(state_owner_dp_rank),
        )

    @staticmethod
    def _param_group_match_key(param_group):
        keys = []
        for key in param_group_identifier_keys:
            if key in param_group:
                keys.append(param_group[key])
            elif f"pre_{key}" in param_group:
                keys.append(param_group[f"pre_{key}"])
            else:
                raise ValueError(
                    f"Key {key} (or pre_{key}) not found in param_group {param_group}."
                )
        return tuple(keys)

    def _load_dion_common_state_dict(self, state_dict) -> None:
        """Load non-tensor optimizer state without Adam-style param-state allocation."""
        if "optimizer" not in state_dict:
            raise RuntimeError("[Dion] distributed checkpoint missing optimizer common state")

        saved_groups = {
            self._param_group_match_key(param_group): param_group
            for param_group in state_dict["optimizer"]["param_groups"]
        }
        inner_state_dict = self.optimizer.state_dict()
        param_groups = []
        for inner_param_group in inner_state_dict["param_groups"]:
            group_key = self._param_group_match_key(inner_param_group)
            if group_key not in saved_groups:
                raise RuntimeError(
                    "[Dion] checkpoint optimizer param-group metadata mismatch: "
                    f"group_key={group_key}"
                )
            param_groups.append(
                {**saved_groups[group_key], "params": inner_param_group["params"]}
            )

        self.optimizer.load_state_dict(
            {
                "state": inner_state_dict.get("state", {}),
                "param_groups": param_groups,
            }
        )

        if "grad_scaler" in state_dict:
            if self.grad_scaler:
                self.grad_scaler.load_state_dict(state_dict["grad_scaler"])
            else:
                logger.info(
                    "[Dion] checkpoint has grad scaler state but optimizer has no grad scaler; "
                    "skipping grad scaler restore"
                )
        elif self.config.fp16:
            logger.info("[Dion] checkpoint has no grad scaler state; skipping grad scaler restore")

    def load_state_dict(self, state_dict):
        """Load optimizer checkpoint state with standard common-state outer protocol."""
        (
            checkpoint_metadata,
            param_state_payload,
            common_state_dict,
        ) = split_distributed_checkpoint_state(state_dict)

        if param_state_payload is None:
            super().load_state_dict(state_dict)
            return

        tp_group = self._resolve_dion_tp_group()
        tp_size = 1 if tp_group is None else self._group_size(tp_group)
        state_replica_size = (
            1
            if self.state_replica_group is None
            else self._group_size(self.state_replica_group)
        )
        validate_matrix_checkpoint_metadata(
            checkpoint_metadata,
            dp_size=self.data_parallel_group.size(),
            fs_size=int(self.fs_size),
            tp_size=int(tp_size),
            rp_size=int(self._rp_size) if self._rp_size is not None else 1,
            state_replica_size=int(state_replica_size),
            topology_signature=self._dion_checkpoint_topology_signature(),
            backend_state_spec=self._require_matrix_backend().state_spec(),
        )
        self._load_dion_common_state_dict(common_state_dict)
        self._ensure_dion_checkpoint_state()

        restore_summary = restore_distributed_checkpoint_state(
            param_state_payload=param_state_payload,
            param_groups=self.optimizer.param_groups,
            optimizer_state=self.optimizer.state,
            get_param_key=self._shard_param_uid,
            mixed_precision_config=getattr(self.optimizer, "_mixed_precision_config", None),
        )
        logger.info(
            '[Dion] Restored %s Dion param states from distributed checkpoint '
            '(no_payload_entry=%s, unnamed=%s)',
            restore_summary["restored"],
            restore_summary["no_payload_entry"],
            restore_summary["unnamed"],
        )

        # Master params are restored directly from the Dion checkpoint payload.
        # Copying from model params here would lose same-topology fp32 resume state.

    def offload_to_cpu(self):
        """Clean up Dion-specific buffers during offload.

        Standard DO pattern handles most cleanup via:
        1. Buffer level: param_and_grad_buffer.offload_to_cpu()
        2. Optimizer state: move_optimizer("cpu")
        3. Q buffers: Auto-cleared when device changes
        """
        self._offload_runtime_buffers()

    def _offload_runtime_buffers(self):
        """Own distributed Dion runtime-buffer cleanup during optimizer offload."""
        optimizer = self.optimizer

        for optim_group in optimizer.param_groups:
            for p in optim_group['params']:
                if p in optimizer.state:
                    state = optimizer.state[p]
                    if '_q_full_buffer' in state:
                        state['_q_full_buffer'] = None
                    if '_q_gather_buffer' in state:
                        state['_q_gather_buffer'] = None

        if hasattr(optimizer, '_q_full_buffers'):
            optimizer._q_full_buffers.clear()
        if hasattr(optimizer, '_q_gather_buffers'):
            optimizer._q_gather_buffers.clear()
        if hasattr(optimizer, '_q_buffer_device'):
            del optimizer._q_buffer_device
        optimizer._buffer_cache.clear()

        torch.cuda.empty_cache()
