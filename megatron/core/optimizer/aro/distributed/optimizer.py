"""Distributed optimizer wrapper for MCore-native ARO."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist

from .... import parallel_state, tensor_parallel
from ....fp8_utils import is_float8tensor
from ...matrix.checkpoint_io import (
    build_distributed_checkpoint_state,
    build_matrix_checkpoint_metadata,
    restore_distributed_checkpoint_state,
    split_distributed_checkpoint_state,
    validate_matrix_checkpoint_metadata,
)
from ...matrix.distrib_optimizer import DistributedMatrixOptimizer
from ...matrix.gradients import get_inter_instance_grad_buffers
from ...matrix.parameter import (
    build_bucket_param_map,
    build_matrix_shard_entries,
    init_matrix_bucket,
    init_standard_bucket,
    resolve_grad_rank_to_fs_rank,
)
from ...matrix.sharding import (
    attach_fs_shard_,
    compute_fs_shard_range,
    create_fs_shard,
    get_opt_shard,
    param_shard_layout,
    register_matrix_shard,
    update_opt_shard,
)
from ...matrix.splits.linear import (
    get_linear_split_rows,
    iter_linear_child_kinds,
    is_linear_fc1_param,
    linear_child_global_shape,
    linear_child_has_local_overlap,
    linear_child_local_shape,
    linear_child_name,
    linear_child_param_uid,
    linear_child_row_range,
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
from ..algorithm import MegatronAro
from ..backend import AroBackend
from ..kernels import choose_orientation, compute_aro_update, compute_aro_updates_batched
from ..scalar_opts import adamw_update_, lion_update_
from ..state import (
    build_param_config,
    init_matrix_state,
    is_aro_matrix_param,
    mark_aro_bucket_params,
    rotation_shape,
    str_to_dtype,
)
from ..types import AroBatch, AroDistMeta, AroMixedPrecisionConfig, AroParamConfig, AroStepParam
from .batches import build_aro_batches


def _group_size(group) -> int:
    if group is None or not dist.is_available() or not dist.is_initialized():
        return 1
    return int(dist.get_world_size(group))


def _group_rank(group) -> int:
    if group is None or not dist.is_available() or not dist.is_initialized():
        return 0
    return int(dist.get_rank(group))


def _map_shard_dim(orientation: str, shard_dim: int) -> int:
    shard_dim = int(shard_dim)
    if shard_dim not in (0, 1):
        return -1
    if orientation == "normal":
        return shard_dim
    if orientation == "transpose":
        return 1 - shard_dim
    raise RuntimeError(f"[ARO_INVALID_ORIENTATION] orientation={orientation!r}")


class DistributedAroOptimizer(DistributedMatrixOptimizer):
    """MCore distributed optimizer adapter for ARO matrix updates."""

    @classmethod
    def _bucket_fs_get_group_size_rank(cls, param_and_grad_buffer, bucket) -> Tuple[object, int, int]:
        """Return the authoritative FS topology for ARO math on one bucket."""
        is_expert_bucket = any(not getattr(param, "allreduce", True) for param in bucket.params)
        if is_expert_bucket:
            fs_group, fs_size, fs_rank = cls._bucket_shard_get_group_size_rank(
                param_and_grad_buffer,
                bucket,
            )
        else:
            fs_group = getattr(param_and_grad_buffer, "aro_fs_group", None)
            fs_size = getattr(param_and_grad_buffer, "aro_fs_size", None)
            fs_rank = getattr(param_and_grad_buffer, "aro_fs_rank", None)
            if fs_size is None or fs_rank is None:
                fs_group, fs_size, fs_rank = cls._bucket_shard_get_group_size_rank(
                    param_and_grad_buffer,
                    bucket,
                )
        if int(fs_size) <= 0:
            raise RuntimeError(f"[ARO_INVALID_FS_GROUP] bucket={bucket.bucket_id} fs_size={fs_size}")
        return fs_group, int(fs_size), int(fs_rank)

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
    def _build_model_gbuf_range(cls, param_and_grad_buffer, bucket_index):
        """Build stock DO ranges plus canonical ARO matrix FS shard metadata."""
        from ...distrib_optimizer import DistributedOptimizer

        parent_result = DistributedOptimizer._build_model_gbuf_range(
            param_and_grad_buffer,
            bucket_index,
        )
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
        fs_group, fs_size, fs_rank = cls._bucket_fs_get_group_size_rank(
            param_and_grad_buffer,
            bucket,
        )
        grad_rank_to_fs_rank = resolve_grad_rank_to_fs_rank(
            grad_group=dp_group,
            fs_group=fs_group,
            fs_size=fs_size,
            bucket_id=bucket.bucket_id,
        )
        aro_param_count, matrix_info_by_param = mark_aro_bucket_params(
            param_map=param_map,
            param_to_name=getattr(param_and_grad_buffer, "param_to_name", None),
            fs_size=fs_size,
        )

        (
            aro_layout,
            shard_layout_by_param,
            aro_param_count,
        ) = build_matrix_shard_entries(
            bucket=bucket,
            param_map=param_map,
            matrix_info_by_param=matrix_info_by_param,
            fs_size=fs_size,
            fs_rank=fs_rank,
            grad_shard_group_size=dp_world_size,
            grad_rank_to_fs_rank=grad_rank_to_fs_rank,
        )
        for param, range_info in param_map.items():
            range_info["matrix_shard_layout"] = shard_layout_by_param.get(param)

        parent_result["local_total"] = 0 if aro_layout is None else aro_layout.shard_size
        parent_result["matrix_bucket_layout"] = aro_layout
        parent_result["standard_count"] = len(param_map) - aro_param_count
        return parent_result

    def __init__(self, *args, **kwargs):
        self.set_matrix_backend(AroBackend())
        self._shard_layouts_by_param = {}
        self._shards_by_param = {}
        self._matrix_local_grad_by_param = {}
        self._matrix_buckets_by_param = {}
        self._matrix_entries_by_param = {}
        self._aro_group_rank_cache = {}
        self._aro_fs_group = kwargs.pop("aro_fs_group", None)
        self._aro_tp_group = kwargs.pop("aro_tp_group", None)
        self._replica_group = kwargs.pop("replica_group", kwargs.pop("replica_group_override", None))
        self._pure_data_parallel_group = kwargs.pop("pure_data_parallel_group", None)
        self._requested_fs_size = int(kwargs.pop("fully_shard_model_parallel_size", 1) or 1)
        self._requested_rp_size = int(
            kwargs.pop("replica_model_parallel_size", kwargs.pop("rp_size", 1)) or 1
        )
        self._is_expert_aro = bool(kwargs.pop("is_expert_aro", False))
        per_model_buffers = kwargs.get("per_model_buffers", None)
        aro_fs_size = 1 if self._aro_fs_group is None else _group_size(self._aro_fs_group)
        aro_fs_rank = 0 if self._aro_fs_group is None else _group_rank(self._aro_fs_group)
        if per_model_buffers is not None:
            for buffers in per_model_buffers.values():
                for buffer in buffers:
                    buffer.aro_fs_group = self._aro_fs_group
                    buffer.aro_fs_size = int(aro_fs_size)
                    buffer.aro_fs_rank = int(aro_fs_rank)

        super().__init__(*args, **kwargs)
        if getattr(self, "is_stub_optimizer", False):
            return
        if self._requested_fs_size > 1 and self._aro_fs_group is None:
            raise RuntimeError(
                "DistributedAroOptimizer requires an authoritative FS group when "
                f"fully_shard_model_parallel_size={self._requested_fs_size}"
            )
        if self._requested_rp_size > 1 and self._replica_group is None:
            raise RuntimeError(
                "DistributedAroOptimizer requires an authoritative replica group when "
                f"replica_model_parallel_size={self._requested_rp_size}"
            )
        self.fs_group = self._aro_fs_group if self._requested_fs_size > 1 else None
        self.fs_size = _group_size(self.fs_group)
        self.fs_rank = _group_rank(self.fs_group)
        self.rp_group = self._replica_group if self._requested_rp_size > 1 else None
        if self._requested_fs_size > 1 and self.fs_size != self._requested_fs_size:
            raise RuntimeError(
                "DistributedAroOptimizer FS group size mismatch: "
                f"requested={self._requested_fs_size} actual={self.fs_size}"
            )
        if self._requested_rp_size > 1 and _group_size(self.rp_group) != self._requested_rp_size:
            raise RuntimeError(
                "DistributedAroOptimizer RP group size mismatch: "
                f"requested={self._requested_rp_size} actual={_group_size(self.rp_group)}"
            )
        self.aro_tp_group = self._resolve_aro_tp_group()
        self.tp_size = _group_size(self.aro_tp_group)
        self.tp_rank = _group_rank(self.aro_tp_group)
        self.state_replica_group = self._resolve_state_replica_group()
        self._validate_matrix_topology(
            fs_size=int(self.fs_size),
            rp_size=int(self._requested_rp_size),
            tp_size=int(self.tp_size),
            is_expert=bool(self._is_expert_aro),
            split_qkv=bool(getattr(self.config, "aro_split_qkv", False)),
            split_qkvg=bool(getattr(self.config, "aro_split_qkv", False)),
            split_linear=bool(getattr(self.config, "aro_split_linear", False)),
        )
        self._mixed_precision_config = AroMixedPrecisionConfig(
            momentum_dtype=getattr(self.config, "aro_momentum_dtype", None),
            rotation_dtype=getattr(self.config, "aro_rotation_dtype", None),
        )
        self._setup_aro_path()
        self._attach_model_param_links()
        self.dist_metas = self._build_dist_metas()
        self.optimizer.dist_metas = self.dist_metas

    @property
    def aro_fs_group(self):
        return self.fs_group

    def _init_bucket_comm(self, bucket, fs_group) -> None:
        """Attach the ARO shard group to a bucket."""
        bucket.matrix_shard_group = fs_group

    def _resolve_aro_tp_group(self):
        group = getattr(self, "_aro_tp_group", None)
        if group is None:
            group = (
                parallel_state.get_expert_tensor_parallel_group(check_initialized=False)
                if getattr(self, "_is_expert_aro", False)
                else parallel_state.get_tensor_model_parallel_group(check_initialized=False)
            )
        if group is not None:
            self._assert_group_excludes_context_parallel(group, label="aro_tp_group")
        return group

    def _init_aro_bucket(self, *, gbuf_idx: int, buffer, bucket, aro_layout, fs_group) -> None:
        init_matrix_bucket(
            self,
            gbuf_idx=gbuf_idx,
            buffer=buffer,
            bucket=bucket,
            matrix_layout=aro_layout,
            fs_group=fs_group,
        )
        bucket.matrix_layout = aro_layout
        bucket.matrix_optimizer = self
        bucket._tracks_matrix_param_views = True
        bucket._matrix_full_param_ready = True

    def _init_standard_bucket(self, *, gbuf_idx: int, buffer, bucket, fs_group) -> None:
        init_standard_bucket(self, gbuf_idx=gbuf_idx, buffer=buffer, bucket=bucket, fs_group=fs_group)
        bucket.matrix_layout = None
        bucket.matrix_optimizer = self
        bucket._tracks_matrix_param_views = False
        bucket._matrix_full_param_ready = True

    def _init_aro_buckets(self) -> None:
        if not hasattr(self, "gbuf_ranges") or not hasattr(self, "buffers"):
            return
        shard_group = self.data_parallel_group
        for gbuf_idx, gbuf_range_maps in enumerate(self.gbuf_ranges):
            buffer = self.buffers[gbuf_idx]
            dtype_key = (buffer.param_dtype, buffer.grad_dtype)
            if dtype_key not in gbuf_range_maps:
                raise RuntimeError(
                    f"[ARO_MISSING_GBUF_RANGE] buffer={gbuf_idx} dtype={dtype_key}"
                )
            bucket_range_maps = gbuf_range_maps[dtype_key]
            for bucket in buffer.buckets:
                bucket_range_map = bucket_range_maps[bucket.bucket_id]
                aro_layout = bucket_range_map.pop("matrix_bucket_layout", None)
                if aro_layout is not None and aro_layout.has_params:
                    fs_group, _, _ = self._bucket_fs_get_group_size_rank(buffer, bucket)
                    self._init_aro_bucket(
                        gbuf_idx=gbuf_idx,
                        buffer=buffer,
                        bucket=bucket,
                        aro_layout=aro_layout,
                        fs_group=fs_group,
                    )
                else:
                    self._init_standard_bucket(
                        gbuf_idx=gbuf_idx,
                        buffer=buffer,
                        bucket=bucket,
                        fs_group=shard_group,
                    )

    def _setup_aro_path(self) -> None:
        if hasattr(self, "buffers"):
            self._init_aro_buckets()
        if hasattr(self, "optimizer") and isinstance(self.optimizer, MegatronAro):
            if hasattr(self, "gbuf_ranges") and hasattr(self, "buffers") and hasattr(
                self,
                "opt_group_ranges",
            ):
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
                    self.config,
                )
                self._refresh_param_groups()
                self._refresh_aro_shards()
        if dist.is_initialized() and hasattr(self, "data_parallel_group"):
            dist.barrier(group=self.data_parallel_group)

    def _build_param_groups(
        self,
        gbuf_ranges: List,
        param_gbuf_map: Dict,
        opt_group_ranges: List,
        config,
    ):
        use_precision_aware_optimizer = config.use_precision_aware_optimizer_no_fp8_or_ds_fp8
        model_fp16_groups = []
        model_fp32_groups = []
        shard_float16_groups = []
        shard_fp32_groups = []
        main_shard_groups = []
        for group_range in opt_group_ranges:
            model_fp16_params = []
            model_fp32_params = []
            shard_float16_params = []
            shard_fp32_params = []
            main_shard_params = []
            model_fp16_groups.append(model_fp16_params)
            model_fp32_groups.append(model_fp32_params)
            shard_float16_groups.append(shard_float16_params)
            shard_fp32_groups.append(shard_fp32_params)
            main_shard_groups.append(main_shard_params)
            for model_param in group_range["params"]:
                gbuf_index, dtype, bucket_index = param_gbuf_map[model_param]
                gbuf_range = gbuf_ranges[gbuf_index][dtype][bucket_index]
                param_range_info = gbuf_range["param_map"][model_param]
                param_range = param_range_info["param"]
                shard_layout = param_range_info.get("matrix_shard_layout", None)
                if model_param.type() in ["torch.cuda.HalfTensor", "torch.cuda.BFloat16Tensor"]:
                    self._process_float16_param(
                        model_param,
                        param_range,
                        shard_layout,
                        config,
                        model_fp16_params,
                        shard_float16_params,
                        main_shard_params,
                    )
                elif model_param.type() == "torch.cuda.FloatTensor":
                    self._process_float32_param(
                        model_param,
                        param_range,
                        shard_layout,
                        config,
                        model_fp32_params,
                        shard_fp32_params,
                    )
                else:
                    raise TypeError(f"Unsupported parameter type: {model_param.type()}")
            if not use_precision_aware_optimizer:
                group_range["orig_group"]["params"] = [*shard_fp32_params, *main_shard_params]
            else:
                float16_optimizer_params = [
                    shard_main_param if shard_main_param is not None else shard_model_param
                    for shard_model_param, shard_main_param in zip(
                        shard_float16_params,
                        main_shard_params,
                    )
                    if shard_main_param is not None or shard_model_param is not None
                ]
                group_range["orig_group"]["params"] = [*shard_fp32_params, *float16_optimizer_params]
        return (
            model_fp16_groups,
            model_fp32_groups,
            shard_float16_groups,
            shard_fp32_groups,
            main_shard_groups,
        )

    def _copy_param_attrs(self, shard_param, model_param) -> None:
        tensor_parallel.copy_tensor_model_parallel_attributes(shard_param, model_param)
        copy_qkv_split_metadata(shard_param, model_param)
        copy_qkvg_split_metadata(shard_param, model_param)
        if is_linear_fc1_param(model_param):
            shard_param.is_linear_fc1 = True
            shard_param.linear_split_rows = get_linear_split_rows(model_param)
        if hasattr(model_param, "shared"):
            shard_param.shared = model_param.shared
        shard_param.is_aro_param = bool(getattr(model_param, "is_aro_param", False))
        shard_param.use_aro = bool(getattr(model_param, "use_aro", shard_param.is_aro_param))

    def _process_float16_param(
        self,
        model_param,
        param_range,
        shard_layout,
        config,
        model_fp16_params,
        shard_float16_params,
        main_shard_params,
    ) -> None:
        use_precision_aware_optimizer = config.use_precision_aware_optimizer_no_fp8_or_ds_fp8
        if shard_layout is not None:
            shard_model_param = create_fs_shard(self, model_param, shard_layout)
            attach_fs_shard_(self, model_param, shard_model_param)
            self._copy_param_attrs(shard_model_param, model_param)
            shard_main_param = shard_model_param.clone().float()
            shard_main_param._model_param = model_param
            self._copy_param_attrs(shard_main_param, model_param)
            register_matrix_shard(
                self,
                model_param=model_param,
                data_shard=shard_model_param,
                opt_shard=shard_main_param,
                shard_layout=shard_layout,
            )
        else:
            if is_float8tensor(model_param) and config.fp8_recipe != "delayed":
                shard_model_param = None
            else:
                shard_model_param = model_param.detach().view(-1)[param_range.start : param_range.end]
                shard_model_param._model_param = model_param
                self._copy_param_attrs(shard_model_param, model_param)
            if not use_precision_aware_optimizer:
                if is_float8tensor(model_param):
                    if hasattr(model_param, "get_high_precision_init_val"):
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
                self._copy_param_attrs(shard_main_param, model_param)
            else:
                shard_main_param = None
        model_param.main_param = shard_main_param
        model_param.main_param_sharded = True
        model_fp16_params.append(model_param)
        shard_float16_params.append(shard_main_param if use_precision_aware_optimizer and shard_layout is not None else shard_model_param)
        main_shard_params.append(shard_main_param)

    def _process_float32_param(
        self,
        model_param,
        param_range,
        shard_layout,
        config,
        model_fp32_params,
        shard_fp32_params,
    ) -> None:
        del config
        if shard_layout is not None:
            shard_model_param = create_fs_shard(self, model_param, shard_layout)
            attach_fs_shard_(self, model_param, shard_model_param)
            shard_model_param._model_param = model_param
            self._copy_param_attrs(shard_model_param, model_param)
            model_param.main_param = shard_model_param
            model_param.main_param_sharded = True
            register_matrix_shard(
                self,
                model_param=model_param,
                data_shard=shard_model_param,
                opt_shard=shard_model_param,
                shard_layout=shard_layout,
            )
        else:
            shard_model_param = model_param.view(-1)[param_range.start : param_range.end]
            shard_model_param._model_param = model_param
            self._copy_param_attrs(shard_model_param, model_param)
        model_fp32_params.append(model_param)
        shard_fp32_params.append(shard_model_param)

    def _refresh_param_groups(self) -> None:
        from ...cpu_offloading import HybridDeviceOptimizer

        if isinstance(self.optimizer, HybridDeviceOptimizer):
            self.optimizer = HybridDeviceOptimizer(
                params=[group_range["orig_group"] for group_range in self.opt_group_ranges],
                **self.optimizer.defaults,
            )
        else:
            self.optimizer.param_groups = [group_range["orig_group"] for group_range in self.opt_group_ranges]
            self.optimizer.load_state_dict(self.optimizer.state_dict())

    def _refresh_aro_shards(self) -> None:
        for optim_group in self.optimizer.param_groups:
            for shard_param in optim_group["params"]:
                model_param = getattr(shard_param, "_model_param", None)
                if model_param is None:
                    continue
                old_opt_shard = get_opt_shard(self, model_param)
                if old_opt_shard is not None and old_opt_shard is not shard_param:
                    if hasattr(old_opt_shard, "_matrix_param_uid"):
                        shard_param._matrix_param_uid = old_opt_shard._matrix_param_uid
                    if hasattr(old_opt_shard, "_aro_param_uid"):
                        shard_param._aro_param_uid = old_opt_shard._aro_param_uid
                    update_opt_shard(self, model_param, shard_param)

    def _attach_model_param_links(self) -> None:
        for model_group, shard_group in zip(self.model_fp32_groups, self.shard_fp32_groups):
            for model_param, shard_param in zip(model_group, shard_group):
                if shard_param is not None:
                    shard_param._model_param = model_param
        for model_group, shard_group in zip(
            self.model_float16_groups,
            self.shard_fp32_from_float16_groups,
        ):
            for model_param, shard_param in zip(model_group, shard_group):
                if shard_param is not None:
                    shard_param._model_param = model_param

    def _local_2d_layout(self, model_param, shard_param):
        if model_param.ndim != 2:
            return None
        shard_layout = param_shard_layout(self, model_param)
        if shard_layout is not None:
            if int(shard_param.numel()) != int(shard_layout.local_numel):
                return None
            return (
                tuple(int(dim) for dim in shard_layout.local_shape),
                tuple(int(dim) for dim in shard_layout.global_shape),
                (
                    tuple(int(dim) for dim in shard_layout.per_expert_global_shape)
                    if shard_layout.per_expert_global_shape is not None
                    else None
                ),
                int(shard_layout.fs_shard_dim),
                int(shard_layout.start_idx),
                int(shard_layout.end_idx),
            )
        try:
            param_range = self._get_model_param_range_map(model_param)["param"]
        except Exception:
            if int(shard_param.numel()) == int(model_param.numel()):
                return (
                    tuple(int(dim) for dim in model_param.shape),
                    None,
                    None,
                    -1,
                    -1,
                    -1,
                )
            return None
        start, end = int(param_range.start), int(param_range.end)
        rows, cols = int(model_param.shape[0]), int(model_param.shape[1])
        if end - start != int(shard_param.numel()):
            return None
        if start == 0 and end == rows * cols:
            return (rows, cols), None, None, -1, -1, -1
        if start % cols == 0 and end % cols == 0:
            return ((end - start) // cols, cols), None, None, 0, start // cols, end // cols
        return None

    def _tp_shard_dim(self, model_param) -> int:
        if not bool(getattr(model_param, "tensor_model_parallel", False)):
            return -1
        dim = int(getattr(model_param, "partition_dim", -1))
        return dim if dim in (0, 1) and self.tp_size > 1 else -1

    def _global_shape(self, model_param, tp_shard_dim: int) -> tuple[int, int]:
        rows, cols = int(model_param.shape[0]), int(model_param.shape[1])
        if tp_shard_dim == 0:
            rows *= int(self.tp_size)
        elif tp_shard_dim == 1:
            cols *= int(self.tp_size)
        return rows, cols

    def _fs_meta_for_param(self, model_param, *, use_matrix: bool, fs_shard_dim: int):
        if not bool(use_matrix) or int(fs_shard_dim) not in (0, 1):
            return None, 1, 0
        bucket = getattr(self, "_matrix_buckets_by_param", {}).get(model_param)
        fs_group = getattr(bucket, "matrix_shard_group", None) if bucket is not None else None
        if fs_group is None:
            fs_group = self.fs_group
        fs_world_size = _group_size(fs_group)
        fs_rank = _group_rank(fs_group)
        if int(fs_world_size) <= 1:
            return None, 1, 0
        return fs_group, int(fs_world_size), int(fs_rank)

    def _build_dist_metas(self) -> Dict[torch.Tensor, AroDistMeta]:
        dist_metas = {}
        for optim_group in self.optimizer.param_groups:
            for shard_param in optim_group.get("params", ()):
                model_param = getattr(shard_param, "_model_param", shard_param)
                name = self._global_param_name(model_param) or getattr(model_param, "_param_name", "")
                layout = self._local_2d_layout(model_param, shard_param)
                use_matrix = is_aro_matrix_param(model_param) and layout is not None
                tp_shard_dim = self._tp_shard_dim(model_param)
                if use_matrix:
                    (
                        local_shape,
                        layout_global_shape,
                        per_expert_global_shape,
                        fs_dim,
                        fs_start,
                        fs_end,
                    ) = layout
                    if fs_dim < 0 and self.fs_size > 1:
                        use_matrix = False
                else:
                    local_shape, fs_dim, fs_start, fs_end = tuple(shard_param.shape), -1, -1, -1
                    layout_global_shape = None
                    per_expert_global_shape = None
                if use_matrix:
                    global_shape = (
                        tuple(int(dim) for dim in layout_global_shape)
                        if layout_global_shape is not None
                        else self._global_shape(model_param, tp_shard_dim)
                    )
                    logical_shape = (
                        tuple(int(dim) for dim in per_expert_global_shape)
                        if per_expert_global_shape is not None
                        else tuple(int(dim) for dim in global_shape)
                    )
                    orientation = choose_orientation(logical_shape)
                    if not name:
                        raise RuntimeError(
                            "[ARO_MISSING_PARAM_NAME] Matrix ARO checkpoint keys require a "
                            "stable parameter name."
                        )
                    param_uid = (name, tuple(logical_shape))
                    linear_split_rows = (
                        get_linear_split_rows(model_param)
                        if is_linear_fc1_param(model_param)
                        else None
                    )
                else:
                    global_shape = None
                    orientation = "normal"
                    param_uid = (
                        name or f"id_{id(model_param)}",
                        tuple(int(dim) for dim in local_shape),
                    )
                    linear_split_rows = None
                fs_group, fs_world_size, fs_rank = self._fs_meta_for_param(
                    model_param,
                    use_matrix=use_matrix,
                    fs_shard_dim=fs_dim,
                )
                dist_meta = AroDistMeta(
                    shape=tuple(int(dim) for dim in local_shape),
                    global_shape=global_shape,
                    fs_start_idx=int(fs_start),
                    fs_end_idx=int(fs_end),
                    tp_shard_dim=int(tp_shard_dim),
                    fs_shard_dim=int(fs_dim),
                    is_transposed=orientation == "transpose",
                    param_uid=param_uid,
                    is_matrix_param=bool(use_matrix),
                    is_aro_param=bool(use_matrix),
                    param_name=name or f"id_{id(model_param)}",
                    fs_group=fs_group,
                    fs_world_size=int(fs_world_size),
                    fs_rank=int(fs_rank),
                    tp_group=self.aro_tp_group if self.tp_size > 1 and tp_shard_dim in (0, 1) else None,
                    tp_world_size=int(self.tp_size) if tp_shard_dim in (0, 1) else 1,
                    tp_rank=int(self.tp_rank) if tp_shard_dim in (0, 1) else 0,
                    per_expert_global_shape=per_expert_global_shape,
                    local_shape=tuple(int(dim) for dim in local_shape) if use_matrix else None,
                    orientation=orientation,
                    linear_split_rows=linear_split_rows,
                    linear_partition_stride=int(getattr(model_param, "partition_stride", 1)),
                )
                dist_meta.param_config = self._build_param_config(shard_param, dist_meta, optim_group)
                shard_param._matrix_param_uid = dist_meta.param_uid
                shard_param._aro_param_uid = dist_meta.param_uid
                dist_metas[shard_param] = dist_meta
        return dist_metas

    def _build_param_config(self, param, dist_meta, optim_group) -> AroParamConfig:
        del optim_group
        return build_param_config(
            param_ndim=2 if bool(getattr(dist_meta, "is_aro_param", False)) else param.ndim,
            local_shape=getattr(dist_meta, "shape", None),
            dist_meta=dist_meta,
            momentum=float(getattr(self.config, "aro_momentum", 0.95)),
            base_optimizer=getattr(self.config, "aro_base_optimizer", "sinkhorn"),
            sinkhorn_iters=int(getattr(self.config, "aro_sinkhorn_iters", 5)),
            qr_backend=getattr(self.config, "aro_qr_backend", "scqr"),
            scqr_eps=float(getattr(self.config, "aro_scqr_eps", 1e-6)),
            update_rms_scale=float(getattr(self.config, "aro_update_rms_scale", 0.2)),
            scalar_optimizer=getattr(self.config, "aro_scalar_optimizer", "adam"),
            scalar_lr_scale=float(getattr(self.config, "aro_scalar_lr_scale", 1.0)),
            beta1=float(getattr(self.config, "aro_beta1", 0.9)),
            beta2=float(getattr(self.config, "aro_beta2", 0.95)),
            scalar_eps=float(getattr(self.config, "aro_scalar_eps", 1e-8)),
            split_qkv=bool(getattr(self.config, "aro_split_qkv", False)),
            split_qkvg=bool(getattr(self.config, "aro_split_qkv", False)),
            split_linear=bool(getattr(self.config, "aro_split_linear", False)),
        )

    def _refresh_aro_step_metadata(self, *, param, optimizer_state, optim_group, dist_meta):
        del optimizer_state
        if dist_meta is not None:
            dist_meta.param_config = self._build_param_config(param, dist_meta, optim_group)

    def _require_param_config(self, param, dist_meta):
        del param
        return dist_meta.param_config if dist_meta is not None else AroParamConfig()

    def _get_step_param_grad(self, param):
        if self.config.use_precision_aware_optimizer_no_fp8_or_ds_fp8:
            return getattr(param, "decoupled_grad", None)
        return getattr(param, "grad", None)

    def _parent_uses_split(self, param, optimizer_state, dist_meta) -> bool:
        if dist_meta is None or not bool(getattr(dist_meta, "is_aro_param", False)):
            return False
        cfg = getattr(dist_meta, "param_config", None)
        split_qkvg = bool(getattr(cfg, "split_qkvg", getattr(self.config, "aro_split_qkv", False)))
        split_qkv = bool(getattr(cfg, "split_qkv", getattr(self.config, "aro_split_qkv", False)))
        split_linear = bool(getattr(cfg, "split_linear", getattr(self.config, "aro_split_linear", False)))
        if split_qkvg and resolve_qkvg_split_shapes(
            param=param,
            optimizer_state=optimizer_state,
            dist_meta=dist_meta,
        ) is not None:
            return True
        if split_qkv and resolve_qkv_split_shapes(
            param=param,
            optimizer_state=optimizer_state,
            dist_meta=dist_meta,
        ) is not None:
            return True
        if split_linear and resolve_linear_split_rows(
            optimizer_state=optimizer_state,
            dist_meta=dist_meta,
        ) is not None:
            return True
        return False

    def _ensure_optimizer_state(self, param, optim_group):
        del optim_group
        state = self.optimizer.state[param]
        meta = self.dist_metas.get(param)
        if meta is None or not bool(getattr(meta, "is_aro_param", False)):
            return state
        init_matrix_state(
            param,
            state,
            dist_meta=meta,
            mixed_precision_config=self._mixed_precision_config,
            init_rotation=not self._parent_uses_split(param, state, meta),
        )
        return state

    def _should_use_distributed_aro_update(self, param, state, optim_group, dist_meta) -> bool:
        del param, state, optim_group
        return bool(dist_meta is not None and getattr(dist_meta, "is_aro_param", False))

    def _view_2d(self, tensor, meta):
        shape = tuple(int(dim) for dim in meta.shape)
        return tensor if tensor.ndim == 2 and tuple(tensor.shape) == shape else tensor.view(shape)

    def _child_param_config(self, child_meta, child_shape, optim_group) -> AroParamConfig:
        del optim_group
        return build_param_config(
            param_ndim=2,
            local_shape=tuple(int(dim) for dim in child_shape),
            dist_meta=child_meta,
            momentum=float(getattr(self.config, "aro_momentum", 0.95)),
            base_optimizer=getattr(self.config, "aro_base_optimizer", "sinkhorn"),
            sinkhorn_iters=int(getattr(self.config, "aro_sinkhorn_iters", 5)),
            qr_backend=getattr(self.config, "aro_qr_backend", "scqr"),
            scqr_eps=float(getattr(self.config, "aro_scqr_eps", 1e-6)),
            update_rms_scale=float(getattr(self.config, "aro_update_rms_scale", 0.2)),
            scalar_optimizer=getattr(self.config, "aro_scalar_optimizer", "adam"),
            scalar_lr_scale=float(getattr(self.config, "aro_scalar_lr_scale", 1.0)),
            beta1=float(getattr(self.config, "aro_beta1", 0.9)),
            beta2=float(getattr(self.config, "aro_beta2", 0.95)),
            scalar_eps=float(getattr(self.config, "aro_scalar_eps", 1e-8)),
            split_qkv=False,
            split_qkvg=False,
            split_linear=False,
        )

    def _child_meta(
        self,
        parent_meta,
        *,
        child_kind,
        child_name,
        child_uid,
        child_shape,
        child_global_shape,
        split_kind,
        optim_group,
        qkv_shapes=None,
        qkvg_shapes=None,
        linear_rows=None,
    ):
        fs_start = int(parent_meta.fs_start_idx)
        fs_end = int(parent_meta.fs_end_idx)
        row_start = int(getattr(parent_meta, "row_shard_start_idx", -1))
        row_end = int(getattr(parent_meta, "row_shard_end_idx", -1))

        def _child_row_range(parent_start: int, parent_end: int):
            if split_kind == "qkv":
                return qkv_child_row_range(
                    parent_row_start=parent_start,
                    parent_row_end=parent_end,
                    split_shapes=qkv_shapes,
                    child_kind=child_kind,
                )
            if split_kind == "qkvg":
                return qkvg_child_row_range(
                    parent_row_start=parent_start,
                    parent_row_end=parent_end,
                    split_shapes=qkvg_shapes,
                    child_kind=child_kind,
                )
            if split_kind == "linear":
                return linear_child_row_range(
                    parent_row_start=parent_start,
                    parent_row_end=parent_end,
                    split_rows=linear_rows,
                    child_kind=child_kind,
                )
            raise RuntimeError(f"[ARO_INVALID_SPLIT_KIND] split_kind={split_kind!r}")

        if int(parent_meta.fs_shard_dim) == 0:
            row_range = _child_row_range(fs_start, fs_end)
            if row_range is not None:
                fs_start, fs_end = int(row_range[0]), int(row_range[1])
                row_start, row_end = fs_start, fs_end
            elif int(child_shape[0]) == 0:
                fs_start, fs_end = 0, 0
                row_start, row_end = 0, 0

        if int(parent_meta.tp_shard_dim) == 0 and int(getattr(parent_meta, "tp_world_size", 1)) > 1:
            parent_rows = int((getattr(parent_meta, "per_expert_global_shape", None) or parent_meta.global_shape)[0])
            tp_parent_start, tp_parent_end = compute_fs_shard_range(
                parent_rows,
                int(parent_meta.tp_world_size),
                int(parent_meta.tp_rank),
            )
            row_range = _child_row_range(tp_parent_start, tp_parent_end)
            if row_range is not None:
                row_start, row_end = int(row_range[0]), int(row_range[1])
            elif int(child_shape[0]) == 0:
                row_start, row_end = 0, 0

        orientation = choose_orientation(child_global_shape)
        child_meta = replace(
            parent_meta,
            shape=tuple(child_shape),
            local_shape=tuple(child_shape),
            global_shape=tuple(child_global_shape),
            per_expert_global_shape=None,
            fs_start_idx=fs_start,
            fs_end_idx=fs_end,
            row_shard_start_idx=row_start,
            row_shard_end_idx=row_end,
            param_uid=child_uid,
            param_name=child_name,
            parent_param_uid=parent_meta.param_uid,
            parent_param_name=parent_meta.param_name,
            orientation=orientation,
            is_transposed=orientation == "transpose",
            is_qkv_child=split_kind == "qkv",
            qkv_child_kind=child_kind if split_kind == "qkv" else "",
            qkv_split_shapes=qkv_shapes,
            is_qkvg_child=split_kind == "qkvg",
            qkvg_child_kind=child_kind if split_kind == "qkvg" else "",
            qkvg_split_shapes=qkvg_shapes,
            is_linear_child=split_kind == "linear",
            linear_child_kind=child_kind if split_kind == "linear" else "",
            linear_split_rows=linear_rows,
        )
        child_meta.param_config = self._child_param_config(child_meta, child_shape, optim_group)
        return child_meta

    @staticmethod
    def _step_param(param, grad, state, group, cfg, meta, commit):
        return AroStepParam(
            param=param,
            grad=grad,
            optimizer_state=state,
            optim_group=group,
            config=cfg,
            dist_meta=meta,
            commit_update=commit,
        )

    @staticmethod
    def _parent_momentum(optimizer_state, param2d):
        momentum = optimizer_state.get("momentum")
        if momentum is None:
            momentum = torch.zeros_like(param2d)
            optimizer_state["momentum"] = momentum
        if momentum.ndim != 2 or tuple(momentum.shape) != tuple(param2d.shape):
            momentum = momentum.view_as(param2d)
        return momentum

    @staticmethod
    def _split_rotation_key(split_kind: str, child_kind: str) -> str:
        if split_kind == "qkv":
            return qkv_state_key("rotation", child_kind)
        if split_kind == "qkvg":
            return qkvg_state_key("rotation", child_kind)
        if split_kind == "linear":
            return linear_state_key("rotation", child_kind)
        raise RuntimeError(f"[ARO_INVALID_SPLIT_KIND] split_kind={split_kind!r}")

    def _child_state(
        self,
        parent_state,
        child_param,
        child_momentum,
        child_meta,
        *,
        split_kind: str,
        child_kind: str,
    ):
        rotation_key = self._split_rotation_key(split_kind, child_kind)
        orientation_key = f"{split_kind}_{child_kind}_orientation"
        local_shape = tuple(int(dim) for dim in child_meta.shape)
        global_shape = tuple(int(dim) for dim in child_meta.global_shape)
        orientation = str(child_meta.orientation)
        rot_shape = rotation_shape(
            local_shape=local_shape,
            global_shape=global_shape,
            orientation=orientation,
            fs_shard_dim=int(getattr(child_meta, "fs_shard_dim", -1)),
            tp_shard_dim=int(getattr(child_meta, "tp_shard_dim", -1)),
        )
        rotation = parent_state.get(rotation_key)
        if rotation is None or tuple(rotation.shape) != tuple(rot_shape):
            tmp_state = {}
            init_matrix_state(
                child_param,
                tmp_state,
                dist_meta=child_meta,
                mixed_precision_config=self._mixed_precision_config,
            )
            parent_state[rotation_key] = tmp_state["rotation"]
        parent_state[orientation_key] = orientation
        parent_state[f"{split_kind}_split_{split_kind}"] = True
        if split_kind == "qkv":
            parent_state["qkv_split_shapes"] = tuple(int(dim) for dim in child_meta.qkv_split_shapes)
        elif split_kind == "qkvg":
            parent_state["qkvg_split_shapes"] = tuple(int(dim) for dim in child_meta.qkvg_split_shapes)
        elif split_kind == "linear":
            parent_state["linear_split_rows"] = tuple(int(dim) for dim in child_meta.linear_split_rows)
        return {
            "momentum": child_momentum,
            "rotation": parent_state[rotation_key],
            "orientation": orientation,
            "local_shape": local_shape,
            "global_shape": global_shape,
        }

    def _expand_split_aro_params(self, *, param, grad, optimizer_state, optim_group, config, dist_meta):
        if not bool(getattr(dist_meta, "is_aro_param", False)):
            return None
        split_qkv = bool(getattr(config, "split_qkv", getattr(self.config, "aro_split_qkv", False)))
        split_qkvg = bool(getattr(config, "split_qkvg", split_qkv))
        split_linear = bool(
            getattr(config, "split_linear", getattr(self.config, "aro_split_linear", False))
        )
        param2d = self._view_2d(param, dist_meta)
        grad2d = self._view_2d(grad, dist_meta)
        parent_shape = tuple(int(dim) for dim in dist_meta.shape)
        parent_global_shape = tuple(int(dim) for dim in (dist_meta.global_shape or parent_shape))
        if split_qkvg:
            shapes = resolve_qkvg_split_shapes(
                param=param,
                optimizer_state=optimizer_state,
                dist_meta=dist_meta,
            )
            if shapes is not None:
                return self._expand_qkvg(
                    param2d,
                    grad2d,
                    optimizer_state,
                    optim_group,
                    dist_meta,
                    parent_shape,
                    parent_global_shape,
                    shapes,
                )
        if split_qkv:
            shapes = resolve_qkv_split_shapes(
                param=param,
                optimizer_state=optimizer_state,
                dist_meta=dist_meta,
            )
            if shapes is not None:
                return self._expand_qkv(
                    param2d,
                    grad2d,
                    optimizer_state,
                    optim_group,
                    dist_meta,
                    parent_shape,
                    parent_global_shape,
                    shapes,
                )
        if split_linear:
            rows = resolve_linear_split_rows(optimizer_state=optimizer_state, dist_meta=dist_meta)
            if rows is not None:
                return self._expand_linear(
                    param2d,
                    grad2d,
                    optimizer_state,
                    optim_group,
                    dist_meta,
                    parent_shape,
                    parent_global_shape,
                    rows,
                )
        return None

    @staticmethod
    def _include_empty_split_child(meta) -> bool:
        return bool(
            int(getattr(meta, "fs_world_size", 1)) > 1
            or int(getattr(meta, "tp_world_size", 1)) > 1
        )

    def _expand_qkv(self, param2d, grad2d, state, group, meta, parent_shape, parent_global_shape, shapes):
        children = []
        parent_momentum = self._parent_momentum(state, param2d)
        include_empty = self._include_empty_split_child(meta)
        for kind in iter_qkv_child_kinds():
            has_overlap = qkv_child_has_local_overlap(shapes, meta, kind)
            if not has_overlap and not include_empty:
                continue
            child_shape = qkv_child_local_shape(parent_shape, shapes, kind, meta) if has_overlap else (0, int(parent_shape[1]))
            global_shape = qkv_child_global_shape(parent_global_shape, shapes, kind)
            child_param = extract_qkv_child(param2d, shapes, kind, meta) if has_overlap else param2d.new_empty(child_shape)
            child_grad = extract_qkv_child(grad2d, shapes, kind, meta) if has_overlap else grad2d.new_empty(child_shape)
            child_momentum = extract_qkv_child(parent_momentum, shapes, kind, meta) if has_overlap else parent_momentum.new_empty(child_shape)
            child_meta = self._child_meta(
                meta,
                child_kind=kind,
                child_name=qkv_child_name(meta.param_name, kind),
                child_uid=qkv_child_param_uid(meta.param_uid, kind),
                child_shape=child_shape,
                child_global_shape=global_shape,
                split_kind="qkv",
                optim_group=group,
                qkv_shapes=shapes,
            )
            child_state = self._child_state(
                state,
                child_param,
                child_momentum,
                child_meta,
                split_kind="qkv",
                child_kind=kind,
            )

            def _commit_update(updated_param, updated_momentum, *, child_kind=kind, has_local_overlap=has_overlap):
                if not has_local_overlap:
                    return
                scatter_qkv_child_(param2d, updated_param, shapes, child_kind, meta)
                scatter_qkv_child_(parent_momentum, updated_momentum, shapes, child_kind, meta)

            children.append(
                self._step_param(
                    child_param,
                    child_grad,
                    child_state,
                    group,
                    child_meta.param_config,
                    child_meta,
                    _commit_update,
                )
            )
        return children

    def _expand_qkvg(self, param2d, grad2d, state, group, meta, parent_shape, parent_global_shape, shapes):
        children = []
        parent_momentum = self._parent_momentum(state, param2d)
        include_empty = self._include_empty_split_child(meta)
        for kind in iter_qkvg_child_kinds():
            has_overlap = qkvg_child_has_local_overlap(shapes, meta, kind)
            if not has_overlap and not include_empty:
                continue
            child_shape = qkvg_child_local_shape(parent_shape, shapes, kind, meta) if has_overlap else (0, int(parent_shape[1]))
            global_shape = qkvg_child_global_shape(parent_global_shape, shapes, kind)
            child_param = extract_qkvg_child(param2d, shapes, kind, meta) if has_overlap else param2d.new_empty(child_shape)
            child_grad = extract_qkvg_child(grad2d, shapes, kind, meta) if has_overlap else grad2d.new_empty(child_shape)
            child_momentum = extract_qkvg_child(parent_momentum, shapes, kind, meta) if has_overlap else parent_momentum.new_empty(child_shape)
            child_meta = self._child_meta(
                meta,
                child_kind=kind,
                child_name=qkvg_child_name(meta.param_name, kind),
                child_uid=qkvg_child_param_uid(meta.param_uid, kind),
                child_shape=child_shape,
                child_global_shape=global_shape,
                split_kind="qkvg",
                optim_group=group,
                qkvg_shapes=shapes,
            )
            child_state = self._child_state(
                state,
                child_param,
                child_momentum,
                child_meta,
                split_kind="qkvg",
                child_kind=kind,
            )

            def _commit_update(updated_param, updated_momentum, *, child_kind=kind, has_local_overlap=has_overlap):
                if not has_local_overlap:
                    return
                scatter_qkvg_child_(param2d, updated_param, shapes, child_kind, meta)
                scatter_qkvg_child_(parent_momentum, updated_momentum, shapes, child_kind, meta)

            children.append(
                self._step_param(
                    child_param,
                    child_grad,
                    child_state,
                    group,
                    child_meta.param_config,
                    child_meta,
                    _commit_update,
                )
            )
        return children

    def _expand_linear(self, param2d, grad2d, state, group, meta, parent_shape, parent_global_shape, rows):
        children = []
        parent_momentum = self._parent_momentum(state, param2d)
        include_empty = self._include_empty_split_child(meta)
        for kind in iter_linear_child_kinds():
            has_overlap = linear_child_has_local_overlap(rows, meta, kind)
            if not has_overlap and not include_empty:
                continue
            child_shape = linear_child_local_shape(parent_shape, rows, meta, kind) if has_overlap else (0, int(parent_shape[1]))
            global_shape = linear_child_global_shape(parent_global_shape, rows, kind)
            child_param = read_linear_child(param2d, rows, meta, kind) if has_overlap else param2d.new_empty(child_shape)
            child_grad = read_linear_child(grad2d, rows, meta, kind) if has_overlap else grad2d.new_empty(child_shape)
            child_momentum = read_linear_child(parent_momentum, rows, meta, kind) if has_overlap else parent_momentum.new_empty(child_shape)
            child_meta = self._child_meta(
                meta,
                child_kind=kind,
                child_name=linear_child_name(meta.param_name, kind),
                child_uid=linear_child_param_uid(meta.param_uid, kind),
                child_shape=child_shape,
                child_global_shape=global_shape,
                split_kind="linear",
                optim_group=group,
                linear_rows=rows,
            )
            child_state = self._child_state(
                state,
                child_param,
                child_momentum,
                child_meta,
                split_kind="linear",
                child_kind=kind,
            )

            def _commit_update(updated_param, updated_momentum, *, child_kind=kind, has_local_overlap=has_overlap):
                if not has_local_overlap:
                    return
                write_linear_child_(param2d, updated_param, rows, meta, child_kind)
                write_linear_child_(parent_momentum, updated_momentum, rows, meta, child_kind)

            children.append(
                self._step_param(
                    child_param,
                    child_grad,
                    child_state,
                    group,
                    child_meta.param_config,
                    child_meta,
                    _commit_update,
                )
            )
        return children

    def _sync_aro_state(self, matrix_params) -> None:
        del matrix_params

    @staticmethod
    def _get_inter_instance_grad_buffers(bucket):
        return get_inter_instance_grad_buffers(bucket)

    def _matrix_grads_are_replicate_synced(self) -> bool:
        return True

    def _build_aro_batches(self, matrix_params):
        return build_aro_batches(matrix_params, rank_cache=self._aro_group_rank_cache)

    def _route_step_params(self):
        return self._route_matrix_step_params(
            param_groups=self.optimizer.param_groups,
            dist_metas=self.dist_metas,
            get_step_param_grad=self._get_step_param_grad,
            ensure_optimizer_state=self._ensure_optimizer_state,
            require_param_config=self._require_param_config,
        )

    def _matrix_params(self):
        return [
            param
            for group in self.optimizer.param_groups
            for param in group.get("params", ())
            if bool(getattr(self.dist_metas.get(param), "is_aro_param", False))
        ]

    def _apply_weight_decay(self, param, group):
        root_config = getattr(self, "config", None)
        lr = float(group.get("lr", getattr(root_config, "lr", 0.0)))
        wd = float(
            group.get(
                "weight_decay",
                getattr(root_config, "weight_decay", 0.0) * float(group.get("wd_mult", 1.0)),
            )
        )
        if wd:
            param.mul_(1.0 - lr * wd)

    def _row_col_groups(self, meta):
        orientation = getattr(meta, "orientation", "normal")
        row_groups = []
        col_groups = []
        if int(getattr(meta, "fs_world_size", 1)) > 1 and getattr(meta, "fs_group", None) is not None:
            mapped = _map_shard_dim(orientation, getattr(meta, "fs_shard_dim", -1))
            if mapped == 0:
                row_groups.append(meta.fs_group)
            elif mapped == 1:
                col_groups.append(meta.fs_group)
        if int(getattr(meta, "tp_world_size", 1)) > 1 and getattr(meta, "tp_group", None) is not None:
            mapped = _map_shard_dim(orientation, getattr(meta, "tp_shard_dim", -1))
            if mapped == 0:
                row_groups.append(meta.tp_group)
            elif mapped == 1:
                col_groups.append(meta.tp_group)
        return tuple(row_groups), tuple(col_groups)

    def _group_ranks_key(self, group):
        if group is None:
            return ()
        cache_key = id(group)
        cached = self._aro_group_rank_cache.get(cache_key)
        if cached is not None and cached[0] is group:
            return cached[1]
        if not dist.is_available() or not dist.is_initialized():
            ranks = ()
        else:
            ranks = tuple(int(rank) for rank in dist.get_process_group_ranks(group))
        self._aro_group_rank_cache[cache_key] = (group, ranks)
        return ranks

    def _groups_ranks_key(self, groups):
        result = []
        seen = set()
        for group in groups:
            if _group_size(group) <= 1:
                continue
            key = self._group_ranks_key(group)
            if key in seen:
                continue
            seen.add(key)
            result.append(key)
        return tuple(result)

    def _batch_entry_key(self, entry, momentum, rotation, row_groups, col_groups):
        cfg = entry.config
        global_shape = tuple(int(dim) for dim in entry.global_shape)
        orientation = str(entry.orientation)
        if orientation == "transpose":
            oriented_global_shape = (global_shape[1], global_shape[0])
        else:
            oriented_global_shape = global_shape
        return (
            oriented_global_shape,
            orientation,
            str(momentum.dtype),
            str(rotation.dtype),
            str(momentum.device),
            int(rotation.size(1)),
            self._groups_ranks_key(row_groups),
            self._groups_ranks_key(col_groups),
            str(getattr(cfg, "base_optimizer", "sinkhorn")),
            int(getattr(cfg, "sinkhorn_iters", 5)),
            str(getattr(cfg, "qr_backend", "scqr")),
            float(getattr(cfg, "scqr_eps", 1e-6)),
            float(getattr(cfg, "update_rms_scale", 0.2)),
        )

    def _prepare_entry(self, entry):
        param = entry.param if entry.param.ndim == 2 else entry.param.view(entry.param_shape)
        grad = entry.grad if entry.grad.ndim == 2 else entry.grad.view(entry.param_shape)
        state = entry.optimizer_state
        momentum = state["momentum"]
        rotation = state["rotation"]
        beta = float(entry.config.momentum)
        momentum.mul_(beta).add_(grad.view_as(momentum), alpha=1.0 - beta)
        self._apply_weight_decay(param, entry.optim_group or {})
        return param, momentum, rotation

    def _commit(self, entry, param, update, new_rotation):
        lr = float((entry.optim_group or {}).get("lr", getattr(getattr(self, "config", None), "lr", 0.0)))
        param.add_(update.to(param.dtype), alpha=-lr)
        entry.optimizer_state["rotation"].copy_(new_rotation.to(dtype=entry.optimizer_state["rotation"].dtype))
        if entry.commit_update is not None:
            entry.commit_update(param, entry.optimizer_state.get("momentum"))

    def _apply_prepared_aro_entry(self, entry, param, momentum, rotation, row_groups, col_groups):
        update, new_rotation = compute_aro_update(
            momentum=momentum,
            rotation=rotation,
            config=entry.config,
            orientation=entry.orientation,
            row_groups=row_groups,
            column_groups=col_groups,
        )
        self._commit(entry, param, update, new_rotation)

    def _apply_prepared_aro_group(self, items):
        if len(items) <= 1:
            entry, param, momentum, rotation, row_groups, col_groups = items[0]
            self._apply_prepared_aro_entry(entry, param, momentum, rotation, row_groups, col_groups)
            return

        first_entry, _, _, _, row_groups, col_groups = items[0]
        result = compute_aro_updates_batched(
            momentums=[item[2] for item in items],
            rotations=[item[3] for item in items],
            config=first_entry.config,
            orientation=first_entry.orientation,
            row_groups=row_groups,
            column_groups=col_groups,
        )
        if result is None:
            for entry, param, momentum, rotation, row_groups, col_groups in items:
                self._apply_prepared_aro_entry(entry, param, momentum, rotation, row_groups, col_groups)
            return

        updates, new_rotations = result
        for (entry, param, _momentum, _rotation, _row_groups, _col_groups), update, new_rotation in zip(
            items,
            updates,
            new_rotations,
        ):
            self._commit(entry, param, update, new_rotation)

    def _apply_aro_batches(self, batches: Sequence[AroBatch]):
        for batch in batches:
            grouped = defaultdict(list)
            ordered_keys = []
            for entry in batch.entries[: batch.real_batch_size]:
                if entry.param is None or entry.grad is None:
                    continue
                param, momentum, rotation = self._prepare_entry(entry)
                row_groups, col_groups = self._row_col_groups(entry.dist_meta)
                key = self._batch_entry_key(entry, momentum, rotation, row_groups, col_groups)
                if key not in grouped:
                    ordered_keys.append(key)
                grouped[key].append((entry, param, momentum, rotation, row_groups, col_groups))
            for key in ordered_keys:
                self._apply_prepared_aro_group(grouped[key])

    def _apply_scalar_params(self, scalar_params) -> None:
        root_config = getattr(self, "config", None)
        for step_param in scalar_params:
            param = step_param.param
            grad = step_param.grad
            if param is None or grad is None:
                continue
            group = step_param.optim_group or {}
            state = step_param.optimizer_state or self.optimizer.state[param]
            lr = float(group.get("lr", getattr(root_config, "lr", 0.0)))
            weight_decay = float(group.get("weight_decay", getattr(root_config, "weight_decay", 0.0)))
            scalar_optimizer = getattr(root_config, "aro_scalar_optimizer", "adam")
            beta1 = float(getattr(root_config, "aro_beta1", 0.9))
            beta2 = float(getattr(root_config, "aro_beta2", 0.95))
            if scalar_optimizer in ("adam", "adamw"):
                adamw_update_(
                    param=param,
                    grad=grad,
                    state=state,
                    lr=lr,
                    weight_decay=weight_decay,
                    beta1=beta1,
                    beta2=beta2,
                    eps=float(getattr(root_config, "aro_scalar_eps", 1e-8)),
                    lr_scale=float(getattr(root_config, "aro_scalar_lr_scale", 1.0)),
                )
            elif scalar_optimizer == "lion":
                lion_update_(
                    param=param,
                    grad=grad,
                    state=state,
                    lr=lr,
                    weight_decay=weight_decay,
                    beta1=beta1,
                    beta2=beta2,
                    lr_scale=float(getattr(root_config, "aro_scalar_lr_scale", 1.0)),
                )
            else:
                raise RuntimeError(f"[ARO_INVALID_SCALAR_OPTIMIZER] {scalar_optimizer!r}")

    def clip_grad_norm(self, clip_grad: float) -> float:
        return self.clip_matrix_grad_norm(
            clip_grad,
            is_matrix_model_param=lambda param: getattr(param, "is_aro_param", False),
        )

    @torch.no_grad()
    def step_with_ready_grads(self) -> bool:
        batches, scalar_params = self._route_step_params()
        matrix_params = set(self._matrix_params())
        saved_grads = {}
        for param in matrix_params:
            if self.config.use_precision_aware_optimizer_no_fp8_or_ds_fp8:
                saved_grads[param] = getattr(param, "decoupled_grad", None)
                param.decoupled_grad = None
            else:
                saved_grads[param] = getattr(param, "grad", None)
                param.grad = None
        self._apply_scalar_params(scalar_params)
        for param, grad in saved_grads.items():
            if self.config.use_precision_aware_optimizer_no_fp8_or_ds_fp8:
                param.decoupled_grad = grad
            else:
                param.grad = grad
        self._apply_aro_batches(batches)
        self._copy_main_params_to_model_params()
        if not self.ddp_config.overlap_param_gather:
            for model_chunk in self.model_chunks:
                model_chunk.start_param_sync()
        return True

    def _aro_param_key(self, param):
        meta = self.dist_metas.get(param)
        return getattr(meta, "param_uid", None) or getattr(param, "_aro_param_uid", None)

    def _aro_checkpoint_topology_signature(self) -> dict:
        return self._matrix_checkpoint_topology_signature(
            fs_group=getattr(self, "fs_group", None),
            tp_group=self._resolve_aro_tp_group(),
            rp_group=getattr(self, "rp_group", None),
            state_replica_group=getattr(self, "state_replica_group", None),
        )

    @staticmethod
    def _validate_state_shape(*, state, key: str, expected: Tuple[int, ...], param_uid) -> None:
        value = state.get(key)
        if value is None:
            raise RuntimeError(f"[ARO_CHECKPOINT_MISSING_STATE] param_uid={param_uid} key={key}")
        if isinstance(value, torch.Tensor):
            actual = tuple(int(dim) for dim in value.shape)
        else:
            actual = tuple(int(dim) for dim in value)
        if actual != tuple(int(dim) for dim in expected):
            raise RuntimeError(
                "[ARO_CHECKPOINT_STATE_SHAPE_MISMATCH] "
                f"param_uid={param_uid} key={key} saved={actual} expected={expected}"
            )

    def _expected_child_layout(self, *, meta, state, split_kind: str, child_kind: str):
        parent_shape = tuple(int(dim) for dim in meta.shape)
        parent_global_shape = tuple(
            int(dim)
            for dim in (getattr(meta, "per_expert_global_shape", None) or meta.global_shape)
        )
        if split_kind == "qkv":
            shapes = tuple(int(dim) for dim in state["qkv_split_shapes"])
            if qkv_child_has_local_overlap(shapes, meta, child_kind):
                child_shape = qkv_child_local_shape(parent_shape, shapes, child_kind, meta)
            else:
                child_shape = (0, int(parent_shape[1]))
            child_global_shape = qkv_child_global_shape(parent_global_shape, shapes, child_kind)
        elif split_kind == "qkvg":
            shapes = tuple(int(dim) for dim in state["qkvg_split_shapes"])
            if qkvg_child_has_local_overlap(shapes, meta, child_kind):
                child_shape = qkvg_child_local_shape(parent_shape, shapes, child_kind, meta)
            else:
                child_shape = (0, int(parent_shape[1]))
            child_global_shape = qkvg_child_global_shape(parent_global_shape, shapes, child_kind)
        elif split_kind == "linear":
            rows = tuple(int(dim) for dim in state["linear_split_rows"])
            if linear_child_has_local_overlap(rows, meta, child_kind):
                child_shape = linear_child_local_shape(parent_shape, rows, meta, child_kind)
            else:
                child_shape = (0, int(parent_shape[1]))
            child_global_shape = linear_child_global_shape(parent_global_shape, rows, child_kind)
        else:
            raise RuntimeError(f"[ARO_INVALID_SPLIT_KIND] split_kind={split_kind!r}")
        orientation = choose_orientation(child_global_shape)
        rot_shape = rotation_shape(
            local_shape=tuple(int(dim) for dim in child_shape),
            global_shape=tuple(int(dim) for dim in child_global_shape),
            orientation=orientation,
            fs_shard_dim=int(getattr(meta, "fs_shard_dim", -1)),
            tp_shard_dim=int(getattr(meta, "tp_shard_dim", -1)),
        )
        return tuple(int(dim) for dim in child_shape), tuple(
            int(dim) for dim in child_global_shape
        ), orientation, rot_shape

    def _validate_restored_split_state(self, *, meta, state, split_kind: str) -> None:
        if split_kind == "qkv":
            split_shapes_key = "qkv_split_shapes"
            child_kinds = iter_qkv_child_kinds()
        elif split_kind == "qkvg":
            split_shapes_key = "qkvg_split_shapes"
            child_kinds = iter_qkvg_child_kinds()
        elif split_kind == "linear":
            split_shapes_key = "linear_split_rows"
            child_kinds = iter_linear_child_kinds()
        else:
            raise RuntimeError(f"[ARO_INVALID_SPLIT_KIND] split_kind={split_kind!r}")

        if split_shapes_key not in state:
            raise RuntimeError(
                "[ARO_CHECKPOINT_MISSING_SPLIT_METADATA] "
                f"param_uid={meta.param_uid} key={split_shapes_key}"
            )
        for child_kind in child_kinds:
            rotation_key = self._split_rotation_key(split_kind, child_kind)
            orientation_key = f"{split_kind}_{child_kind}_orientation"
            _, _, expected_orientation, expected_rotation_shape = self._expected_child_layout(
                meta=meta,
                state=state,
                split_kind=split_kind,
                child_kind=child_kind,
            )
            actual_orientation = state.get(orientation_key)
            if actual_orientation != expected_orientation:
                raise RuntimeError(
                    "[ARO_CHECKPOINT_SPLIT_ORIENTATION_MISMATCH] "
                    f"param_uid={meta.param_uid} child={split_kind}:{child_kind} "
                    f"saved={actual_orientation!r} expected={expected_orientation!r}"
                )
            self._validate_state_shape(
                state=state,
                key=rotation_key,
                expected=expected_rotation_shape,
                param_uid=meta.param_uid,
            )

    def _validate_restored_aro_state(self) -> None:
        for param, meta in self.dist_metas.items():
            if not bool(getattr(meta, "is_aro_param", False)):
                continue
            state = self.optimizer.state.get(param, {})
            expected_local_shape = tuple(int(dim) for dim in meta.shape)
            expected_global_shape = tuple(
                int(dim)
                for dim in (getattr(meta, "per_expert_global_shape", None) or meta.global_shape)
            )
            self._validate_state_shape(
                state=state,
                key="momentum",
                expected=expected_local_shape,
                param_uid=meta.param_uid,
            )
            self._validate_state_shape(
                state=state,
                key="local_shape",
                expected=expected_local_shape,
                param_uid=meta.param_uid,
            )
            self._validate_state_shape(
                state=state,
                key="global_shape",
                expected=expected_global_shape,
                param_uid=meta.param_uid,
            )
            saved_orientation = state.get("orientation")
            if saved_orientation != meta.orientation:
                raise RuntimeError(
                    "[ARO_CHECKPOINT_ORIENTATION_MISMATCH] "
                    f"param_uid={meta.param_uid} saved={saved_orientation!r} "
                    f"expected={meta.orientation!r}"
                )

            if bool(state.get("qkvg_split_qkvg", False)):
                self._validate_restored_split_state(meta=meta, state=state, split_kind="qkvg")
            elif bool(state.get("qkv_split_qkv", False)):
                self._validate_restored_split_state(meta=meta, state=state, split_kind="qkv")
            elif bool(state.get("linear_split_linear", False)):
                self._validate_restored_split_state(meta=meta, state=state, split_kind="linear")
            else:
                expected_rotation_shape = rotation_shape(
                    local_shape=expected_local_shape,
                    global_shape=expected_global_shape,
                    orientation=meta.orientation,
                    fs_shard_dim=int(getattr(meta, "fs_shard_dim", -1)),
                    tp_shard_dim=int(getattr(meta, "tp_shard_dim", -1)),
                )
                self._validate_state_shape(
                    state=state,
                    key="rotation",
                    expected=expected_rotation_shape,
                    param_uid=meta.param_uid,
                )

    def _cast_restored_aro_state_dtypes(self) -> None:
        momentum_dtype = str_to_dtype(getattr(self._mixed_precision_config, "momentum_dtype", None))
        rotation_dtype = str_to_dtype(getattr(self._mixed_precision_config, "rotation_dtype", None))
        if momentum_dtype is None and rotation_dtype is None:
            return

        for param, meta in self.dist_metas.items():
            if not bool(getattr(meta, "is_aro_param", False)):
                continue
            state = self.optimizer.state.get(param, {})
            if momentum_dtype is not None and isinstance(state.get("momentum"), torch.Tensor):
                state["momentum"] = state["momentum"].to(device=param.device, dtype=momentum_dtype)
            if rotation_dtype is None:
                continue
            for key, value in tuple(state.items()):
                if isinstance(value, torch.Tensor) and (key == "rotation" or key.endswith("_rotation")):
                    state[key] = value.to(device=param.device, dtype=rotation_dtype)

    def sharded_state_dict(
        self,
        model_sharded_state_dict=None,
        is_loading=False,
        sharding_type=None,
        metadata=None,
    ):
        """Build torch_dist-compatible ARO optimizer checkpoint state."""
        from ....dist_checkpointing.mapping import ShardedObject, ShardedTensor

        del model_sharded_state_dict, is_loading, sharding_type
        dp_rank = self.data_parallel_group.rank()
        dp_size = self.data_parallel_group.size()
        base_key = f"optimizer.distributed.dp_group_idx_{self.data_parallel_group_idx}"
        common_replica_id = (self.distributed_optimizer_instance_id, 0, dp_rank)
        tp_group = self._resolve_aro_tp_group()
        tp_size = 1 if tp_group is None else self._group_size(tp_group)
        state_replica_group = getattr(self, "state_replica_group", None)
        state_replica_size = 1 if state_replica_group is None else self._group_size(state_replica_group)
        state_replica_rank = 0 if state_replica_group is None else self._group_rank(state_replica_group)
        rp_group = getattr(self, "rp_group", None)
        rp_size = 1 if rp_group is None else self._group_size(rp_group)
        state_owner_dp_rank = dp_rank
        if state_replica_group is not None:
            dp_ranks = self._get_group_ranks_for_checkpoint(self.data_parallel_group)
            replica_ranks = self._get_group_ranks_for_checkpoint(state_replica_group)
            if replica_ranks:
                state_owner_dp_rank = int(dp_ranks.index(int(replica_ranks[0])))
        checkpoint_metadata = build_matrix_checkpoint_metadata(
            dp_size=dp_size,
            fs_size=int(getattr(self, "fs_size", 1)),
            tp_size=int(tp_size),
            rp_size=int(rp_size),
            state_replica_size=int(state_replica_size),
            requested_type=(metadata or {}).get("distrib_optim_sharding_type", "matrix_fs_rank_state"),
            topology_signature=self._aro_checkpoint_topology_signature(),
            backend_state_spec=self._require_matrix_backend().state_spec(),
        )
        return build_distributed_checkpoint_state(
            common_state=self.state_dict(),
            param_groups=self.optimizer.param_groups,
            optimizer_state=self.optimizer.state,
            get_param_key=self._aro_param_key,
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

    def load_state_dict(self, state_dict):
        if "matrix_checkpoint_metadata" not in state_dict:
            return super().load_state_dict(state_dict)
        metadata, param_state, common = split_distributed_checkpoint_state(state_dict)
        tp_group = self._resolve_aro_tp_group()
        tp_size = 1 if tp_group is None else self._group_size(tp_group)
        state_replica_group = getattr(self, "state_replica_group", None)
        state_replica_size = 1 if state_replica_group is None else self._group_size(state_replica_group)
        rp_group = getattr(self, "rp_group", None)
        rp_size = 1 if rp_group is None else self._group_size(rp_group)
        validate_matrix_checkpoint_metadata(
            metadata,
            dp_size=self.data_parallel_group.size(),
            fs_size=int(getattr(self, "fs_size", 1)),
            tp_size=int(tp_size),
            rp_size=int(rp_size),
            state_replica_size=int(state_replica_size),
            topology_signature=self._aro_checkpoint_topology_signature(),
            backend_state_spec=self._require_matrix_backend().state_spec(),
        )
        result = super().load_state_dict(common)
        restore_distributed_checkpoint_state(
            param_state_payload=param_state,
            param_groups=self.optimizer.param_groups,
            optimizer_state=self.optimizer.state,
            get_param_key=self._aro_param_key,
            mixed_precision_config=self._mixed_precision_config,
        )
        self._cast_restored_aro_state_dtypes()
        self._validate_restored_aro_state()
        return result


__all__ = ["DistributedAroOptimizer"]
