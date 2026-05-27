"""Distributed optimizer wrapper for MCore-native Muon."""

from __future__ import annotations

import math
import logging
import os
import time
from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist

from .... import parallel_state, tensor_parallel
from ....fp8_utils import is_float8tensor
from ...matrix.distrib_optimizer import DistributedMatrixOptimizer
from ...matrix.types import MatrixBucketLayout
from ...matrix.sharding import (
    create_fs_shard,
    get_fs_split_dim,
    get_opt_shard,
    get_tp_split_dim,
    is_tp_enabled,
    param_shard_layout,
    attach_fs_shard_,
    register_matrix_shard,
    update_opt_shard,
)
from ...matrix.splits.linear import (
    iter_linear_child_kinds,
    linear_child_global_shape,
    linear_child_has_local_overlap,
    linear_child_local_shape,
    linear_child_name,
    linear_child_param_uid,
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
    resolve_qkvg_split_shapes,
    scatter_qkvg_child_,
)
from ...matrix.parameter import (
    build_bucket_param_map,
    build_matrix_shard_entries,
    init_matrix_bucket,
    init_standard_bucket,
    resolve_grad_rank_to_fs_rank,
)
from ..backend import MuonBackend
from ..algorithm import MegatronMuon
from ..kernels import (
    apply_muon_momentum,
    get_and_reset_gram_profile,
    get_muon_scale_factor,
    orthogonalize_muon,
    orthogonalize_muon_update,
)
from ..state import build_param_config, is_muon_matrix_param
from ..types import MuonBatch, MuonDistMeta, MuonParamConfig, MuonStepParam
from .batches import build_muon_batches
from .checkpoint_io import (
    build_distributed_checkpoint_state,
    build_muon_checkpoint_metadata,
    restore_muon_param_state_,
    split_distributed_checkpoint_state,
    validate_muon_checkpoint_metadata,
)


logger = logging.getLogger(__name__)


def _group_size(group) -> int:
    if group is None or not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size(group)


def _group_rank(group) -> int:
    if group is None or not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank(group)


def _split_range(size: int, world_size: int, rank: int) -> tuple[int, int]:
    base = int(size) // int(world_size)
    rem = int(size) % int(world_size)
    start = int(rank) * base + min(int(rank), rem)
    return start, start + base + (1 if int(rank) < rem else 0)


def _is_moe_expert_param(param: torch.Tensor, param_name: Optional[str] = None) -> bool:
    del param_name
    if not getattr(param, "allreduce", True):
        return True
    num_local_experts = getattr(param, "num_local_experts", None)
    return num_local_experts is not None and int(num_local_experts) > 1


def _mark_muon_bucket_params(param_map, param_to_name, fs_size: int):
    """Classify bucket params and build static Muon matrix metadata."""
    fs_size = int(fs_size)
    if fs_size <= 0:
        raise RuntimeError(f"[Muon] invalid FS size while marking bucket params: {fs_size}")

    muon_param_count = 0
    muon_info_by_param = {}

    for param in param_map.keys():
        param_name = None
        if param_to_name is not None and param in param_to_name:
            param_name = param_to_name[param]
        if param_name:
            param._param_name = param_name

        param.is_muon_param = is_muon_matrix_param(param)
        param.is_matrix_param = bool(param.is_muon_param)

        is_expert = _is_moe_expert_param(param, param_name)
        raw_tp_split_dim = get_tp_split_dim(param)
        has_tp = is_tp_enabled(param)
        if is_expert and has_tp:
            tp_world_size = parallel_state.get_expert_tensor_parallel_world_size()
        else:
            tp_world_size = parallel_state.get_tensor_model_parallel_world_size() if has_tp else 1
        tp_shard_dim = raw_tp_split_dim if has_tp and tp_world_size > 1 else -1

        if not param.is_muon_param:
            continue

        m_local, n_local = (int(dim) for dim in param.shape)
        fs_shard_dim = get_fs_split_dim(tp_shard_dim)
        split_size = m_local if fs_shard_dim == 0 else n_local
        if fs_size > 1 and split_size < fs_size:
            param.is_muon_param = False
            param.is_matrix_param = False
            param.muon_candidate = False
            param.matrix_optimizer_candidate = False
            continue

        muon_param_count += 1

        if tp_shard_dim == 0:
            m_global = m_local * tp_world_size
            n_global = n_local
        elif tp_shard_dim == 1:
            m_global = m_local
            n_global = n_local * tp_world_size
        else:
            m_global = m_local
            n_global = n_local

        muon_info_by_param[param] = {
            "is_muon": True,
            "global_shape": (m_global, n_global),
            "fs_shard_dim": fs_shard_dim,
            "tp_shard_dim": tp_shard_dim,
            "per_expert_global_shape": None,
        }

    return muon_param_count, muon_info_by_param


class DistributedMuonOptimizer(DistributedMatrixOptimizer):
    """MCore distributed optimizer adapter for Muon matrix updates."""

    @classmethod
    def _bucket_fs_get_group_size_rank(cls, param_and_grad_buffer, bucket) -> Tuple[object, int, int]:
        """Return the authoritative FS topology for Muon math on one bucket."""
        is_expert_bucket = any(not getattr(param, "allreduce", True) for param in bucket.params)
        if is_expert_bucket:
            fs_group, fs_size, fs_rank = cls._bucket_shard_get_group_size_rank(
                param_and_grad_buffer, bucket
            )
        else:
            fs_group = getattr(param_and_grad_buffer, "muon_fs_group", None)
            fs_size = getattr(param_and_grad_buffer, "muon_fs_size", None)
            fs_rank = getattr(param_and_grad_buffer, "muon_fs_rank", None)
            if fs_size is None or fs_rank is None:
                fs_group, fs_size, fs_rank = cls._bucket_shard_get_group_size_rank(
                    param_and_grad_buffer, bucket
                )
        if fs_size <= 0:
            raise RuntimeError(f"[Muon] invalid FS group size for bucket {bucket.bucket_id}: {fs_size}")
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
        """Build stock DO ranges plus canonical Muon matrix FS shard metadata."""
        from ...distrib_optimizer import DistributedOptimizer

        parent_result = DistributedOptimizer._build_model_gbuf_range(
            param_and_grad_buffer, bucket_index
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

        fs_group, fs_size, fs_rank = cls._bucket_fs_get_group_size_rank(param_and_grad_buffer, bucket)
        grad_rank_to_fs_rank = resolve_grad_rank_to_fs_rank(
            grad_group=dp_group,
            fs_group=fs_group,
            fs_size=fs_size,
            bucket_id=bucket.bucket_id,
        )
        muon_param_count, muon_info_by_param = _mark_muon_bucket_params(
            param_map=param_map,
            param_to_name=getattr(param_and_grad_buffer, "param_to_name", None),
            fs_size=fs_size,
        )

        (
            muon_layout,
            muon_shard_layout_by_param,
            muon_param_count,
        ) = build_matrix_shard_entries(
            bucket=bucket,
            param_map=param_map,
            matrix_info_by_param=muon_info_by_param,
            fs_size=fs_size,
            fs_rank=fs_rank,
            grad_shard_group_size=dp_world_size,
            grad_rank_to_fs_rank=grad_rank_to_fs_rank,
        )
        for param, range_info in param_map.items():
            shard_layout = muon_shard_layout_by_param.get(param)
            range_info["matrix_shard_layout"] = shard_layout

        parent_result["local_total"] = 0 if muon_layout is None else muon_layout.shard_size
        parent_result["matrix_bucket_layout"] = muon_layout
        parent_result["standard_count"] = len(param_map) - muon_param_count
        return parent_result

    def __init__(self, *args, **kwargs):
        self.set_matrix_backend(MuonBackend())
        self._shard_layouts_by_param = {}
        self._shards_by_param = {}
        self._matrix_local_grad_by_param = {}
        self._muon_local_grad_by_param = self._matrix_local_grad_by_param
        self._matrix_buckets_by_param = {}
        self._muon_buckets_by_param = self._matrix_buckets_by_param
        self._matrix_entries_by_param = {}
        self._muon_entries_by_param = self._matrix_entries_by_param
        self._muon_fs_group = kwargs.pop("muon_fs_group", None)
        self._muon_tp_group = kwargs.pop("muon_tp_group", None)
        self._replica_group = kwargs.pop("replica_group", kwargs.pop("replica_group_override", None))
        self._pure_data_parallel_group = kwargs.pop("pure_data_parallel_group", None)
        self._requested_fs_size = int(kwargs.pop("fully_shard_model_parallel_size", 1) or 1)
        self._requested_rp_size = int(
            kwargs.pop("replica_model_parallel_size", kwargs.pop("rp_size", 1)) or 1
        )
        self._is_expert_muon = bool(kwargs.pop("is_expert_muon", False))
        self._fs_mode = kwargs.pop("muon_fs_mode", "blockwise")
        self._tp_mode = kwargs.pop("muon_tp_mode", "blockwise")
        self._ns_backend = kwargs.pop("muon_ns_backend", "standard")
        self._muon_profile_ns = os.getenv("MEGATRON_MUON_PROFILE_NS", "0") == "1"
        self._muon_profile_ns_interval = max(
            1, int(os.getenv("MEGATRON_MUON_PROFILE_NS_INTERVAL", "1"))
        )
        self._muon_profile_ns_step = 0
        self._muon_profile_ns_ms = 0.0
        self._muon_profile_ns_calls = 0
        self._muon_profile_ns_launches = 0
        self._muon_profile_ns_groups = os.getenv("MEGATRON_MUON_PROFILE_NS_GROUPS", "0") == "1"
        self._muon_profile_apply_ms = 0.0
        self._muon_profile_apply_batches = 0
        self._muon_profile_fs_a2a_ms = 0.0
        self._muon_profile_fs_a2a_calls = 0

        per_model_buffers = kwargs.get("per_model_buffers", None)
        muon_fs_size = 1 if self._muon_fs_group is None else _group_size(self._muon_fs_group)
        muon_fs_rank = 0 if self._muon_fs_group is None else _group_rank(self._muon_fs_group)
        if per_model_buffers is not None:
            for buffers in per_model_buffers.values():
                for buffer in buffers:
                    buffer.muon_fs_group = self._muon_fs_group
                    buffer.muon_fs_size = int(muon_fs_size)
                    buffer.muon_fs_rank = int(muon_fs_rank)

        super().__init__(*args, **kwargs)
        if getattr(self, "is_stub_optimizer", False):
            return
        if self._requested_fs_size > 1 and self._muon_fs_group is None:
            raise RuntimeError(
                "DistributedMuonOptimizer requires an authoritative FS group when "
                f"fully_shard_model_parallel_size={self._requested_fs_size}"
            )
        if self._requested_rp_size > 1 and self._replica_group is None:
            raise RuntimeError(
                "DistributedMuonOptimizer requires an authoritative replica group when "
                f"replica_model_parallel_size={self._requested_rp_size}"
            )
        self.fs_group = self._muon_fs_group if self._requested_fs_size > 1 else None
        self.fs_size = _group_size(self.fs_group)
        self.fs_rank = _group_rank(self.fs_group)
        self.rp_group = self._replica_group if self._requested_rp_size > 1 else None
        if self._requested_fs_size > 1 and self.fs_size != self._requested_fs_size:
            raise RuntimeError(
                "DistributedMuonOptimizer FS group size mismatch: "
                f"requested={self._requested_fs_size} actual={self.fs_size}"
            )
        if self._requested_rp_size > 1 and _group_size(self.rp_group) != self._requested_rp_size:
            raise RuntimeError(
                "DistributedMuonOptimizer RP group size mismatch: "
                f"requested={self._requested_rp_size} actual={_group_size(self.rp_group)}"
            )
        self.muon_tp_group = self._resolve_muon_tp_group()
        self.tp_size = _group_size(self.muon_tp_group)
        self.tp_rank = _group_rank(self.muon_tp_group)
        self.state_replica_group = self._resolve_state_replica_group()
        self._validate_matrix_topology(
            fs_size=int(self.fs_size),
            rp_size=int(self._requested_rp_size),
            tp_size=int(self.tp_size),
            is_expert=bool(self._is_expert_muon),
            split_qkv=bool(getattr(self.optimizer, "defaults", {}).get("split_qkv", False)),
            split_qkvg=bool(getattr(self.optimizer, "defaults", {}).get("split_qkv", False)),
            split_linear=bool(getattr(self.optimizer, "defaults", {}).get("split_linear", False)),
        )
        self._setup_muon_path()
        self._attach_model_param_links()
        self.dist_metas = self._build_dist_metas()
        self.optimizer.dist_metas = self.dist_metas

    @property
    def muon_fs_group(self):
        return self.fs_group

    @property
    def muon_tp_group(self):
        return self._muon_tp_group

    @muon_tp_group.setter
    def muon_tp_group(self, value):
        self._muon_tp_group = value

    def _init_bucket_comm(self, bucket, fs_group) -> None:
        """Attach the Muon shard group to a bucket."""
        bucket.matrix_shard_group = fs_group

    def _init_muon_bucket(
        self,
        *,
        gbuf_idx: int,
        buffer,
        bucket,
        muon_layout: MatrixBucketLayout,
        fs_group,
    ) -> None:
        init_matrix_bucket(
            self,
            gbuf_idx=gbuf_idx,
            buffer=buffer,
            bucket=bucket,
            matrix_layout=muon_layout,
            fs_group=fs_group,
        )
        bucket.matrix_layout = muon_layout
        bucket.matrix_optimizer = self
        bucket._tracks_matrix_param_views = True
        bucket._matrix_full_param_ready = True

    def _init_standard_bucket(self, *, gbuf_idx: int, buffer, bucket, fs_group) -> None:
        init_standard_bucket(self, gbuf_idx=gbuf_idx, buffer=buffer, bucket=bucket, fs_group=fs_group)
        bucket.matrix_layout = None
        bucket.matrix_optimizer = self
        bucket._tracks_matrix_param_views = False
        bucket._matrix_full_param_ready = True

    def _init_muon_buckets(self) -> None:
        """Configure buffer-level Muon bucket layouts after parent optimizer init."""
        if not hasattr(self, "gbuf_ranges") or not hasattr(self, "buffers"):
            return
        shard_group = self.data_parallel_group
        for gbuf_idx, gbuf_range_maps in enumerate(self.gbuf_ranges):
            buffer = self.buffers[gbuf_idx]
            dtype_key = (buffer.param_dtype, buffer.grad_dtype)
            if dtype_key not in gbuf_range_maps:
                raise RuntimeError(
                    f"[Muon] missing gbuf_ranges entry for buffer={gbuf_idx} dtype={dtype_key}"
                )
            bucket_range_maps = gbuf_range_maps[dtype_key]
            for bucket in buffer.buckets:
                bucket_range_map = bucket_range_maps[bucket.bucket_id]
                muon_layout = bucket_range_map.pop("matrix_bucket_layout", None)
                if muon_layout is not None and muon_layout.has_params:
                    fs_group, _, _ = self._bucket_fs_get_group_size_rank(buffer, bucket)
                    self._init_muon_bucket(
                        gbuf_idx=gbuf_idx,
                        buffer=buffer,
                        bucket=bucket,
                        muon_layout=muon_layout,
                        fs_group=fs_group,
                    )
                else:
                    self._init_standard_bucket(
                        gbuf_idx=gbuf_idx,
                        buffer=buffer,
                        bucket=bucket,
                        fs_group=shard_group,
                    )

    def _setup_muon_path(self) -> None:
        """Run Muon-specific matrix-shard setup after parent optimizer init."""
        if hasattr(self, "buffers"):
            self._init_muon_buckets()
        if hasattr(self, "optimizer") and isinstance(self.optimizer, MegatronMuon):
            if hasattr(self, "gbuf_ranges") and hasattr(self, "buffers"):
                if hasattr(self, "opt_group_ranges"):
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
                    self._refresh_muon_shards()
        if dist.is_initialized() and hasattr(self, "data_parallel_group"):
            dist.barrier(group=self.data_parallel_group)

    def _build_param_groups(
        self,
        gbuf_ranges: List,
        param_gbuf_map: Dict,
        opt_group_ranges: List,
        config,
    ):
        """Build parameter groups with canonical matrix FS shards."""
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
                group_range["orig_group"]["params"] = [
                    *shard_fp32_params,
                    *main_shard_params,
                ]
            else:
                float16_optimizer_params = [
                    shard_main_param if shard_main_param is not None else shard_model_param
                    for shard_model_param, shard_main_param in zip(
                        shard_float16_params,
                        main_shard_params,
                    )
                    if shard_main_param is not None or shard_model_param is not None
                ]
                group_range["orig_group"]["params"] = [
                    *shard_fp32_params,
                    *float16_optimizer_params,
                ]

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
        if hasattr(model_param, "shared"):
            shard_param.shared = model_param.shared

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
        if use_precision_aware_optimizer and shard_layout is not None:
            shard_float16_params.append(shard_main_param)
        else:
            shard_float16_params.append(shard_model_param)
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
                params=[g["orig_group"] for g in self.opt_group_ranges],
                **self.optimizer.defaults,
            )
        else:
            self.optimizer.param_groups = [g["orig_group"] for g in self.opt_group_ranges]
            self.optimizer.load_state_dict(self.optimizer.state_dict())

    def _refresh_muon_shards(self) -> None:
        for optim_group in self.optimizer.param_groups:
            for shard_param in optim_group["params"]:
                model_param = getattr(shard_param, "_model_param", None)
                if model_param is None:
                    continue
                old_opt_shard = get_opt_shard(self, model_param)
                if old_opt_shard is not None and old_opt_shard is not shard_param:
                    if hasattr(old_opt_shard, "_matrix_param_uid"):
                        shard_param._matrix_param_uid = old_opt_shard._matrix_param_uid
                    if hasattr(old_opt_shard, "_muon_param_uid"):
                        shard_param._muon_param_uid = old_opt_shard._muon_param_uid
                    update_opt_shard(self, model_param, shard_param)

    def _attach_model_param_links(self) -> None:
        for model_group, shard_group in zip(self.model_fp32_groups, self.shard_fp32_groups):
            for model_param, shard_param in zip(model_group, shard_group):
                if shard_param is not None:
                    shard_param._model_param = model_param
        for model_group, shard_group in zip(
            self.model_float16_groups, self.shard_fp32_from_float16_groups
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
                int(shard_layout.fs_shard_dim),
                int(shard_layout.start_idx),
                int(shard_layout.end_idx),
            )
        try:
            param_range = self._get_model_param_range_map(model_param)["param"]
        except Exception:
            if int(shard_param.numel()) == int(model_param.numel()):
                return tuple(int(dim) for dim in model_param.shape), -1, -1, -1
            return None
        start, end = int(param_range.start), int(param_range.end)
        rows, cols = int(model_param.shape[0]), int(model_param.shape[1])
        if end - start != int(shard_param.numel()):
            return None
        if start == 0 and end == rows * cols:
            return (rows, cols), -1, -1, -1
        if start % cols == 0 and end % cols == 0:
            return ((end - start) // cols, cols), 0, start // cols, end // cols
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

    def _build_dist_metas(self) -> Dict[torch.Tensor, MuonDistMeta]:
        dist_metas = {}
        for optim_group in self.optimizer.param_groups:
            for shard_param in optim_group.get("params", ()):
                model_param = getattr(shard_param, "_model_param", shard_param)
                name = self._global_param_name(model_param) or getattr(model_param, "_param_name", "")
                layout = self._local_2d_layout(model_param, shard_param)
                use_matrix = is_muon_matrix_param(model_param) and layout is not None
                tp_shard_dim = self._tp_shard_dim(model_param)
                if use_matrix:
                    local_shape, fs_dim, fs_start, fs_end = layout
                    if fs_dim < 0 and self.fs_size > 1:
                        use_matrix = False
                else:
                    local_shape, fs_dim, fs_start, fs_end = tuple(shard_param.shape), -1, -1, -1
                if use_matrix:
                    global_shape = self._global_shape(model_param, tp_shard_dim)
                    is_transposed = bool(global_shape[0] > global_shape[1])
                    param_uid = (name or f"id_{id(model_param)}", tuple(global_shape))
                else:
                    global_shape = None
                    is_transposed = False
                    param_uid = (
                        name or f"id_{id(model_param)}",
                        tuple(int(dim) for dim in local_shape),
                    )
                dist_meta = MuonDistMeta(
                    shape=tuple(int(dim) for dim in local_shape),
                    global_shape=global_shape,
                    fs_start_idx=int(fs_start),
                    fs_end_idx=int(fs_end),
                    tp_shard_dim=int(tp_shard_dim),
                    fs_shard_dim=int(fs_dim),
                    is_transposed=is_transposed,
                    param_uid=param_uid,
                    is_matrix_param=bool(use_matrix),
                    is_muon_param=bool(use_matrix),
                    param_name=name or f"id_{id(model_param)}",
                    fs_group=self.fs_group if self.fs_size > 1 else None,
                    fs_world_size=int(self.fs_size),
                    fs_rank=int(self.fs_rank),
                    tp_group=self.muon_tp_group if self.tp_size > 1 and tp_shard_dim in (0, 1) else None,
                    tp_world_size=int(self.tp_size) if tp_shard_dim in (0, 1) else 1,
                    tp_rank=int(self.tp_rank) if tp_shard_dim in (0, 1) else 0,
                    local_shape=tuple(int(dim) for dim in local_shape) if use_matrix else None,
                )
                dist_meta.param_config = self._build_param_config(shard_param, dist_meta, optim_group)
                shard_param._matrix_param_uid = dist_meta.param_uid
                shard_param._muon_param_uid = dist_meta.param_uid
                dist_metas[shard_param] = dist_meta
        return dist_metas

    def _build_param_config(self, param, dist_meta, optim_group) -> MuonParamConfig:
        cfg = build_param_config(
            param_ndim=2 if bool(getattr(dist_meta, "is_muon_param", False)) else param.ndim,
            local_shape=getattr(dist_meta, "shape", None),
            dist_meta=dist_meta,
            tp_world_size=int(getattr(dist_meta, "tp_world_size", 1)),
            tp_active=int(getattr(dist_meta, "tp_world_size", 1)) > 1,
            momentum_beta=float(getattr(self.config, "muon_momentum", 0.9)),
            use_nesterov=bool(getattr(self.config, "muon_use_nesterov", False)),
            ns_backend=getattr(self.config, "muon_ns_backend", self._ns_backend),
            coefficient_type=getattr(self.config, "muon_coefficient_type", "quintic"),
            num_ns_steps=int(getattr(self.config, "muon_num_ns_steps", 5)),
            ns_epsilon=float(getattr(self.config, "muon_ns_epsilon", 1e-7)),
            gram_restart_iterations=tuple(
                getattr(self.config, "muon_gram_ns_restart_iters", (2,))
            ),
            gram_kernel_policy=getattr(self.config, "muon_gram_ns_kernel_policy", "torch"),
            gram_dtype=getattr(self.config, "muon_gram_ns_dtype", None),
            scale_mode=getattr(self.config, "muon_scale_mode", "spectral"),
            extra_scale_factor=float(getattr(self.config, "muon_extra_scale_factor", 1.0)),
            fs_mode=getattr(self.config, "muon_fs_mode", self._fs_mode),
            tp_mode=getattr(self.config, "muon_tp_mode", self._tp_mode),
            split_qkv=bool(getattr(self.config, "muon_split_qkv", True)),
            split_qkvg=bool(getattr(self.config, "muon_split_qkv", True)),
            split_linear=bool(getattr(self.config, "muon_split_linear", True)),
        )
        if cfg.tp_mode == "duplicated":
            cfg.tp_mode = "duplicated_debug"
        return cfg

    def _refresh_muon_step_metadata(self, *, param, optimizer_state, optim_group, dist_meta):
        del optimizer_state
        if dist_meta is not None:
            dist_meta.param_config = self._build_param_config(param, dist_meta, optim_group)

    def _require_param_config(self, param, dist_meta):
        del param
        return dist_meta.param_config

    def _get_step_param_grad(self, param):
        if self.config.use_precision_aware_optimizer_no_fp8_or_ds_fp8:
            return getattr(param, "decoupled_grad", None)
        return getattr(param, "grad", None)

    def _ensure_optimizer_state(self, param, optim_group):
        state = self.optimizer.state[param]
        meta = self.dist_metas.get(param)
        if meta is None or not bool(getattr(meta, "is_muon_param", False)):
            return state
        shape = tuple(int(dim) for dim in meta.shape)
        momentum = state.get("momentum_buffer")
        if momentum is None or tuple(momentum.shape) != shape:
            state["momentum_buffer"] = torch.zeros(shape, dtype=param.dtype, device=param.device)
        state["local_shape"] = shape
        state["global_shape"] = tuple(int(dim) for dim in meta.global_shape)
        return state

    def _should_use_distributed_muon_update(self, param, state, optim_group, dist_meta) -> bool:
        del param, state, optim_group
        return bool(dist_meta is not None and getattr(dist_meta, "is_muon_param", False))

    def _view_2d(self, tensor, meta):
        shape = tuple(int(dim) for dim in meta.shape)
        return tensor if tensor.ndim == 2 and tuple(tensor.shape) == shape else tensor.view(shape)

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
        if int(parent_meta.fs_shard_dim) == 0:
            if split_kind == "qkv":
                row_range = qkv_child_row_range(parent_row_start=fs_start, parent_row_end=fs_end, split_shapes=qkv_shapes, child_kind=child_kind)
            elif split_kind == "qkvg":
                row_range = qkvg_child_row_range(parent_row_start=fs_start, parent_row_end=fs_end, split_shapes=qkvg_shapes, child_kind=child_kind)
            else:
                row_range = (0, int(child_shape[0]))
            if row_range is not None:
                fs_start, fs_end = int(row_range[0]), int(row_range[1])
            elif int(child_shape[0]) == 0:
                fs_start, fs_end = 0, 0
        child_meta = replace(
            parent_meta,
            shape=tuple(child_shape),
            local_shape=tuple(child_shape),
            global_shape=tuple(child_global_shape),
            fs_start_idx=fs_start,
            fs_end_idx=fs_end,
            param_uid=child_uid,
            param_name=child_name,
            parent_param_uid=parent_meta.param_uid,
            parent_param_name=parent_meta.param_name,
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
        return MuonStepParam(
            param=param,
            grad=grad,
            optimizer_state=state,
            optim_group=group,
            config=cfg,
            dist_meta=meta,
            commit_update=commit,
        )

    def _config_value(self, group, param_config, group_key, config_key, default):
        if param_config is not None and hasattr(param_config, group_key):
            return getattr(param_config, group_key)
        if group is not None and group_key in group:
            return group[group_key]
        defaults = getattr(getattr(self, "optimizer", None), "defaults", {})
        if group_key in defaults:
            return defaults[group_key]
        root_config = getattr(self, "config", None)
        if root_config is not None and hasattr(root_config, config_key):
            return getattr(root_config, config_key)
        return default

    def _child_param_config(self, child_meta, child_shape, optim_group) -> MuonParamConfig:
        parent_config = getattr(child_meta, "param_config", None)
        config = build_param_config(
            param_ndim=2,
            local_shape=tuple(int(dim) for dim in child_shape),
            dist_meta=child_meta,
            tp_world_size=int(getattr(child_meta, "tp_world_size", 1)),
            tp_active=int(getattr(child_meta, "tp_world_size", 1)) > 1,
            momentum_beta=float(
                self._config_value(optim_group, parent_config, "momentum_beta", "muon_momentum", 0.95)
            ),
            use_nesterov=bool(
                self._config_value(
                    optim_group,
                    parent_config,
                    "use_nesterov",
                    "muon_use_nesterov",
                    False,
                )
            ),
            ns_backend=self._config_value(
                optim_group,
                parent_config,
                "ns_backend",
                "muon_ns_backend",
                getattr(self, "_ns_backend", "standard"),
            ),
            coefficient_type=self._config_value(
                optim_group,
                parent_config,
                "coefficient_type",
                "muon_coefficient_type",
                "quintic",
            ),
            num_ns_steps=int(
                self._config_value(optim_group, parent_config, "num_ns_steps", "muon_num_ns_steps", 5)
            ),
            ns_epsilon=float(
                self._config_value(optim_group, parent_config, "ns_epsilon", "muon_ns_epsilon", 1e-7)
            ),
            gram_restart_iterations=tuple(
                self._config_value(
                    optim_group,
                    parent_config,
                    "gram_restart_iterations",
                    "muon_gram_ns_restart_iters",
                    (2,),
                )
            ),
            gram_kernel_policy=self._config_value(
                optim_group,
                parent_config,
                "gram_kernel_policy",
                "muon_gram_ns_kernel_policy",
                "torch",
            ),
            gram_dtype=self._config_value(
                optim_group,
                parent_config,
                "gram_dtype",
                "muon_gram_ns_dtype",
                None,
            ),
            scale_mode=self._config_value(
                optim_group, parent_config, "scale_mode", "muon_scale_mode", "spectral"
            ),
            extra_scale_factor=float(
                self._config_value(
                    optim_group,
                    parent_config,
                    "extra_scale_factor",
                    "muon_extra_scale_factor",
                    1.0,
                )
            ),
            fs_mode=self._config_value(
                optim_group,
                parent_config,
                "fs_mode",
                "muon_fs_mode",
                getattr(self, "_fs_mode", "blockwise"),
            ),
            tp_mode=self._config_value(
                optim_group,
                parent_config,
                "tp_mode",
                "muon_tp_mode",
                getattr(self, "_tp_mode", "blockwise"),
            ),
            split_qkv=bool(
                self._config_value(optim_group, parent_config, "split_qkv", "muon_split_qkv", True)
            ),
            split_qkvg=bool(
                self._config_value(
                    optim_group,
                    parent_config,
                    "split_qkvg",
                    "muon_split_qkv",
                    True,
                )
            ),
            split_linear=bool(
                self._config_value(
                    optim_group,
                    parent_config,
                    "split_linear",
                    "muon_split_linear",
                    True,
                )
            ),
        )
        if config.tp_mode == "duplicated":
            config.tp_mode = "duplicated_debug"
        return config

    @staticmethod
    def _parent_momentum(optimizer_state, param2d):
        momentum = optimizer_state.get("momentum_buffer")
        if momentum is None:
            momentum = optimizer_state.get("momentum")
            if momentum is not None:
                optimizer_state["momentum_buffer"] = momentum
        if momentum is None:
            momentum = torch.zeros_like(param2d)
            optimizer_state["momentum_buffer"] = momentum
        if momentum.ndim != 2 or tuple(momentum.shape) != tuple(param2d.shape):
            momentum = momentum.view_as(param2d)
        return momentum

    @staticmethod
    def _child_state(momentum, child_shape, global_shape):
        return {
            "momentum": momentum,
            "momentum_buffer": momentum,
            "local_shape": tuple(int(dim) for dim in child_shape),
            "global_shape": tuple(int(dim) for dim in global_shape),
        }

    def _expand_split_muon_params(self, *, param, grad, optimizer_state, optim_group, config, dist_meta):
        if not bool(getattr(dist_meta, "is_muon_param", False)):
            return None
        if config is None:
            config = getattr(dist_meta, "param_config", None)
        split_qkv = bool(self._config_value(optim_group, config, "split_qkv", "muon_split_qkv", True))
        split_qkvg = bool(
            self._config_value(optim_group, config, "split_qkvg", "muon_split_qkv", split_qkv)
        )
        split_linear = bool(
            self._config_value(optim_group, config, "split_linear", "muon_split_linear", True)
        )
        param2d = self._view_2d(param, dist_meta)
        grad2d = self._view_2d(grad, dist_meta)
        parent_shape = tuple(int(dim) for dim in dist_meta.shape)
        parent_global_shape = tuple(int(dim) for dim in (dist_meta.global_shape or parent_shape))
        if split_qkvg:
            shapes = resolve_qkvg_split_shapes(param=param, optimizer_state=optimizer_state, dist_meta=dist_meta)
            if shapes is not None:
                return self._expand_qkvg(param2d, grad2d, optimizer_state, optim_group, dist_meta, parent_shape, parent_global_shape, shapes)
        if split_qkv:
            shapes = resolve_qkv_split_shapes(param=param, optimizer_state=optimizer_state, dist_meta=dist_meta)
            if shapes is not None:
                return self._expand_qkv(param2d, grad2d, optimizer_state, optim_group, dist_meta, parent_shape, parent_global_shape, shapes)
        if split_linear:
            rows = resolve_linear_split_rows(optimizer_state=optimizer_state, dist_meta=dist_meta)
            if rows is not None:
                return self._expand_linear(param2d, grad2d, optimizer_state, optim_group, dist_meta, parent_shape, parent_global_shape, rows)
        return None

    def _expand_qkv(self, param2d, grad2d, state, group, meta, parent_shape, parent_global_shape, shapes):
        children = []
        parent_momentum = self._parent_momentum(state, param2d)
        include_empty = getattr(meta.param_config, "fs_mode", "blockwise") == "distributed"
        for kind in iter_qkv_child_kinds():
            has_overlap = qkv_child_has_local_overlap(shapes, meta, kind)
            if not has_overlap and not include_empty:
                continue
            child_shape = (
                qkv_child_local_shape(parent_shape, shapes, kind, meta)
                if has_overlap
                else (0, int(parent_shape[1]))
            )
            global_shape = qkv_child_global_shape(parent_global_shape, shapes, kind)
            child_momentum = (
                extract_qkv_child(parent_momentum, shapes, kind, meta)
                if has_overlap
                else parent_momentum.new_empty(child_shape)
            )
            child_state = self._child_state(child_momentum, child_shape, global_shape)
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

            def _commit_update(
                updated_param,
                updated_momentum,
                *,
                child_kind=kind,
                parent_momentum=parent_momentum,
                has_local_overlap=has_overlap,
            ):
                if not has_local_overlap:
                    return
                scatter_qkv_child_(param2d, updated_param, shapes, child_kind, meta)
                scatter_qkv_child_(parent_momentum, updated_momentum, shapes, child_kind, meta)

            children.append(
                self._step_param(
                    extract_qkv_child(param2d, shapes, kind, meta)
                    if has_overlap
                    else param2d.new_empty(child_shape),
                    extract_qkv_child(grad2d, shapes, kind, meta)
                    if has_overlap
                    else grad2d.new_empty(child_shape),
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
        include_empty = getattr(meta.param_config, "fs_mode", "blockwise") == "distributed"
        for kind in iter_qkvg_child_kinds():
            has_overlap = qkvg_child_has_local_overlap(shapes, meta, kind)
            if not has_overlap and not include_empty:
                continue
            child_shape = (
                qkvg_child_local_shape(parent_shape, shapes, kind, meta)
                if has_overlap
                else (0, int(parent_shape[1]))
            )
            global_shape = qkvg_child_global_shape(parent_global_shape, shapes, kind)
            child_momentum = (
                extract_qkvg_child(parent_momentum, shapes, kind, meta)
                if has_overlap
                else parent_momentum.new_empty(child_shape)
            )
            child_state = self._child_state(child_momentum, child_shape, global_shape)
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

            def _commit_update(
                updated_param,
                updated_momentum,
                *,
                child_kind=kind,
                parent_momentum=parent_momentum,
                has_local_overlap=has_overlap,
            ):
                if not has_local_overlap:
                    return
                scatter_qkvg_child_(param2d, updated_param, shapes, child_kind, meta)
                scatter_qkvg_child_(parent_momentum, updated_momentum, shapes, child_kind, meta)

            children.append(
                self._step_param(
                    extract_qkvg_child(param2d, shapes, kind, meta)
                    if has_overlap
                    else param2d.new_empty(child_shape),
                    extract_qkvg_child(grad2d, shapes, kind, meta)
                    if has_overlap
                    else grad2d.new_empty(child_shape),
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
        include_empty = getattr(meta.param_config, "fs_mode", "blockwise") == "distributed"
        for kind in iter_linear_child_kinds():
            has_overlap = linear_child_has_local_overlap(rows, meta, kind)
            if not has_overlap and not include_empty:
                continue
            child_shape = (
                linear_child_local_shape(parent_shape, rows, meta, kind)
                if has_overlap
                else (0, int(parent_shape[1]))
            )
            global_shape = linear_child_global_shape(parent_global_shape, rows, kind)
            child_momentum = (
                read_linear_child(parent_momentum, rows, meta, kind)
                if has_overlap
                else parent_momentum.new_empty(child_shape)
            )
            child_state = self._child_state(child_momentum, child_shape, global_shape)
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

            def _commit_update(
                updated_param,
                updated_momentum,
                *,
                child_kind=kind,
                parent_momentum=parent_momentum,
                has_local_overlap=has_overlap,
            ):
                if not has_local_overlap:
                    return
                write_linear_child_(param2d, updated_param, rows, meta, child_kind)
                write_linear_child_(parent_momentum, updated_momentum, rows, meta, child_kind)

            children.append(
                self._step_param(
                    read_linear_child(param2d, rows, meta, kind)
                    if has_overlap
                    else param2d.new_empty(child_shape),
                    read_linear_child(grad2d, rows, meta, kind)
                    if has_overlap
                    else grad2d.new_empty(child_shape),
                    child_state,
                    group,
                    child_meta.param_config,
                    child_meta,
                    _commit_update,
                )
            )
        return children

    def _sync_muon_state(self, matrix_params) -> None:
        del matrix_params

    def _build_muon_batches(self, matrix_params):
        return build_muon_batches(
            matrix_params,
            fs_mode=getattr(self, "_fs_mode", "blockwise"),
            tp_mode=getattr(self, "_tp_mode", "blockwise"),
            ns_backend=getattr(self, "_ns_backend", "standard"),
        )

    def _route_step_params(self):
        return self._route_matrix_step_params(
            param_groups=self.optimizer.param_groups,
            dist_metas=self.dist_metas,
            get_step_param_grad=self._get_step_param_grad,
            ensure_optimizer_state=self._ensure_optimizer_state,
            require_param_config=self._require_param_config,
        )

    def _matrix_params(self):
        return [p for g in self.optimizer.param_groups for p in g.get("params", ()) if bool(getattr(self.dist_metas.get(p), "is_muon_param", False))]

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

    def _orthogonalize(self, update, entry, meta=None):
        meta = entry.dist_meta if meta is None else meta
        cfg = entry.config
        matrix_count = int(update.size(0)) if update.ndim > 2 else 1
        elapsed_ms = None
        if bool(getattr(self, "_muon_profile_ns", False)) and update.is_cuda:
            torch.cuda.synchronize(update.device)
            start = time.perf_counter()
        result = orthogonalize_muon_update(
            update,
            ns_backend=cfg.ns_backend,
            coefficient_type=cfg.coefficient_type,
            num_ns_steps=cfg.num_ns_steps,
            eps=cfg.ns_epsilon,
            scale_mode=cfg.scale_mode,
            extra_scale_factor=cfg.extra_scale_factor,
            global_shape=tuple(int(dim) for dim in getattr(meta, "global_shape", entry.global_shape)),
            tp_group=getattr(meta, "tp_group", None),
            partition_dim=getattr(meta, "tp_shard_dim", -1),
            tp_mode=cfg.tp_mode,
            gram_restart_iterations=cfg.gram_restart_iterations,
            gram_dtype=cfg.gram_dtype,
            gram_kernel_policy=cfg.gram_kernel_policy,
            fp32_matmul_prec=getattr(
                getattr(self, "config", None),
                "muon_fp32_matmul_prec",
                "medium",
            ),
        )
        if bool(getattr(self, "_muon_profile_ns", False)) and update.is_cuda:
            torch.cuda.synchronize(update.device)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self._muon_profile_ns_ms += elapsed_ms
            self._muon_profile_ns_calls += matrix_count
            self._muon_profile_ns_launches += 1
        self._maybe_log_ns_group(
            update=update,
            entries=(entry,),
            metas=(meta,),
            elapsed_ms=elapsed_ms,
        )
        return result

    @staticmethod
    def _same_orthogonalize_config(entry, other_entry, meta, other_meta) -> bool:
        cfg = entry.config
        other_cfg = other_entry.config
        if cfg != other_cfg:
            return False
        if getattr(meta, "tp_group", None) is not getattr(other_meta, "tp_group", None):
            return False
        return int(getattr(meta, "tp_shard_dim", -1)) == int(
            getattr(other_meta, "tp_shard_dim", -1)
        )

    def _orthogonalize_batch_key(self, update, entry, meta):
        cfg = entry.config
        return (
            tuple(update.shape),
            update.dtype,
            update.device,
            cfg.ns_backend,
            cfg.coefficient_type,
            int(cfg.num_ns_steps),
            float(cfg.ns_epsilon),
            tuple(int(step) for step in cfg.gram_restart_iterations),
            str(cfg.gram_kernel_policy),
            str(cfg.gram_dtype),
            cfg.tp_mode,
            id(getattr(meta, "tp_group", None)),
            int(getattr(meta, "tp_shard_dim", -1)),
        )

    @staticmethod
    def _scale_update(update, entry, meta):
        cfg = entry.config
        global_shape = tuple(int(dim) for dim in getattr(meta, "global_shape", entry.global_shape))
        scale = get_muon_scale_factor(int(global_shape[0]), int(global_shape[1]), mode=cfg.scale_mode)
        return update * float(scale) * float(cfg.extra_scale_factor)

    def _orthogonalize_stacked_unscaled(self, stacked_update, entries, metas):
        entry = entries[0]
        meta = metas[0]
        cfg = entry.config
        matrix_count = int(stacked_update.size(0)) if stacked_update.ndim > 2 else 1
        elapsed_ms = None
        if bool(getattr(self, "_muon_profile_ns", False)) and stacked_update.is_cuda:
            torch.cuda.synchronize(stacked_update.device)
            start = time.perf_counter()
        result = orthogonalize_muon(
            stacked_update,
            ns_backend=cfg.ns_backend,
            steps=cfg.num_ns_steps,
            coefficient_type=cfg.coefficient_type,
            tp_group=getattr(meta, "tp_group", None),
            partition_dim=getattr(meta, "tp_shard_dim", -1),
            tp_mode=cfg.tp_mode,
            gram_restart_steps=cfg.gram_restart_iterations,
            gram_dtype=cfg.gram_dtype,
            gram_kernel_policy=cfg.gram_kernel_policy,
            eps=cfg.ns_epsilon,
            fp32_matmul_prec=getattr(
                getattr(self, "config", None),
                "muon_fp32_matmul_prec",
                "medium",
            ),
        )
        if bool(getattr(self, "_muon_profile_ns", False)) and stacked_update.is_cuda:
            torch.cuda.synchronize(stacked_update.device)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self._muon_profile_ns_ms += elapsed_ms
            self._muon_profile_ns_calls += matrix_count
            self._muon_profile_ns_launches += 1
        self._maybe_log_ns_group(
            update=stacked_update,
            entries=entries,
            metas=metas,
            elapsed_ms=elapsed_ms,
        )
        return result

    def _orthogonalize_batch(self, updates, entries, metas=None):
        if not updates:
            return []
        if metas is None:
            metas = [entry.dist_meta for entry in entries]
        results = [None] * len(updates)
        groups = {}
        for index, (update, entry, meta) in enumerate(zip(updates, entries, metas)):
            groups.setdefault(self._orthogonalize_batch_key(update, entry, meta), []).append(index)

        for indices in groups.values():
            if len(indices) == 1:
                index = indices[0]
                results[index] = self._orthogonalize(updates[index], entries[index], metas[index])
                continue

            batch_update = torch.stack([updates[index].contiguous() for index in indices], dim=0)
            batch_result = self._orthogonalize_stacked_unscaled(
                batch_update,
                tuple(entries[index] for index in indices),
                tuple(metas[index] for index in indices),
            )
            for batch_index, result_index in enumerate(indices):
                results[result_index] = self._scale_update(
                    batch_result[batch_index].contiguous(),
                    entries[result_index],
                    metas[result_index],
                )
        return results

    def _maybe_log_ns_group(self, *, update, entries, metas, elapsed_ms):
        if not bool(getattr(self, "_muon_profile_ns_groups", False)):
            return
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        if rank != 0:
            return
        cfg = entries[0].config
        unique_global_shapes = sorted(
            {
                tuple(int(dim) for dim in getattr(meta, "global_shape", entry.global_shape))
                for entry, meta in zip(entries, metas)
            }
        )
        print(
            "[MuonNSGroup] "
            f"step={self._muon_profile_ns_step + 1} "
            f"backend={cfg.ns_backend} kernel={cfg.gram_kernel_policy} "
            f"shape={tuple(int(dim) for dim in update.shape)} "
            f"dtype={update.dtype} count={len(entries)} "
            f"global_shapes={unique_global_shapes} "
            f"ms={(0.0 if elapsed_ms is None else elapsed_ms):.3f}",
            flush=True,
        )

    def _maybe_log_ns_profile(self):
        if not bool(getattr(self, "_muon_profile_ns", False)):
            return
        self._muon_profile_ns_step += 1
        gram_stats = get_and_reset_gram_profile()
        if self._muon_profile_ns_step % int(self._muon_profile_ns_interval) != 0:
            self._muon_profile_ns_ms = 0.0
            self._muon_profile_ns_calls = 0
            self._muon_profile_ns_launches = 0
            self._muon_profile_apply_ms = 0.0
            self._muon_profile_apply_batches = 0
            self._muon_profile_fs_a2a_ms = 0.0
            self._muon_profile_fs_a2a_calls = 0
            return

        local_ms = float(self._muon_profile_ns_ms)
        local_calls = int(self._muon_profile_ns_calls)
        local_launches = int(self._muon_profile_ns_launches)
        local_apply_ms = float(self._muon_profile_apply_ms)
        local_apply_batches = int(self._muon_profile_apply_batches)
        local_fs_a2a_ms = float(self._muon_profile_fs_a2a_ms)
        local_fs_a2a_calls = int(self._muon_profile_fs_a2a_calls)
        max_ms = local_ms
        avg_ms = local_ms
        max_apply_ms = local_apply_ms
        avg_apply_ms = local_apply_ms
        max_fs_a2a_ms = local_fs_a2a_ms
        avg_fs_a2a_ms = local_fs_a2a_ms
        max_calls = local_calls
        max_launches = local_launches
        max_apply_batches = local_apply_batches
        max_fs_a2a_calls = local_fs_a2a_calls
        total_gram_stats = dict(gram_stats)
        rank = 0
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
            sum_stats = torch.tensor(
                [
                    local_ms,
                    float(local_calls),
                    float(local_launches),
                    local_apply_ms,
                    float(local_apply_batches),
                    local_fs_a2a_ms,
                    float(local_fs_a2a_calls),
                    float(gram_stats.get("dao_groups", 0)),
                    float(gram_stats.get("torch_groups", 0)),
                    float(gram_stats.get("dao_matrices", 0)),
                    float(gram_stats.get("torch_matrices", 0)),
                    float(gram_stats.get("fallback_policy_torch_groups", 0)),
                    float(gram_stats.get("fallback_device_groups", 0)),
                    float(gram_stats.get("fallback_dtype_groups", 0)),
                    float(gram_stats.get("fallback_tile_groups", 0)),
                    float(gram_stats.get("gram_all_reduce_calls", 0)),
                    float(gram_stats.get("gram_all_reduce_numel", 0)),
                ],
                dtype=torch.float64,
                device=device,
            )
            max_stats = sum_stats.clone()
            dist.all_reduce(sum_stats, op=dist.ReduceOp.SUM)
            dist.all_reduce(max_stats, op=dist.ReduceOp.MAX)
            avg_ms = float(sum_stats[0].item()) / float(dist.get_world_size())
            max_ms = float(max_stats[0].item())
            max_calls = int(max_stats[1].item())
            max_launches = int(max_stats[2].item())
            avg_apply_ms = float(sum_stats[3].item()) / float(dist.get_world_size())
            max_apply_ms = float(max_stats[3].item())
            max_apply_batches = int(max_stats[4].item())
            avg_fs_a2a_ms = float(sum_stats[5].item()) / float(dist.get_world_size())
            max_fs_a2a_ms = float(max_stats[5].item())
            max_fs_a2a_calls = int(max_stats[6].item())
            total_gram_stats = {
                "dao_groups": int(sum_stats[7].item()),
                "torch_groups": int(sum_stats[8].item()),
                "dao_matrices": int(sum_stats[9].item()),
                "torch_matrices": int(sum_stats[10].item()),
                "fallback_policy_torch_groups": int(sum_stats[11].item()),
                "fallback_device_groups": int(sum_stats[12].item()),
                "fallback_dtype_groups": int(sum_stats[13].item()),
                "fallback_tile_groups": int(sum_stats[14].item()),
                "gram_all_reduce_calls": int(sum_stats[15].item()),
                "gram_all_reduce_numel": int(sum_stats[16].item()),
            }

        if rank == 0:
            cfg = getattr(self, "config", None)
            print(
                "[MuonNSProfile] "
                f"step={self._muon_profile_ns_step} "
                f"backend={getattr(cfg, 'muon_ns_backend', self._ns_backend)} "
                f"kernel={getattr(cfg, 'muon_gram_ns_kernel_policy', 'torch')} "
                f"local_ms={local_ms:.3f} avg_ms={avg_ms:.3f} max_ms={max_ms:.3f} "
                f"local_calls={local_calls} max_calls={max_calls} "
                f"local_launches={local_launches} max_launches={max_launches} "
                f"apply_avg_ms={avg_apply_ms:.3f} apply_max_ms={max_apply_ms:.3f} "
                f"apply_batches_max={max_apply_batches} "
                f"fs_a2a_avg_ms={avg_fs_a2a_ms:.3f} fs_a2a_max_ms={max_fs_a2a_ms:.3f} "
                f"fs_a2a_calls_max={max_fs_a2a_calls} "
                f"dao_groups={total_gram_stats.get('dao_groups', 0)} "
                f"torch_groups={total_gram_stats.get('torch_groups', 0)} "
                f"dao_matrices={total_gram_stats.get('dao_matrices', 0)} "
                f"torch_matrices={total_gram_stats.get('torch_matrices', 0)} "
                f"fallback_tile_groups={total_gram_stats.get('fallback_tile_groups', 0)} "
                f"gram_all_reduce_calls={total_gram_stats.get('gram_all_reduce_calls', 0)} "
                f"gram_all_reduce_numel={total_gram_stats.get('gram_all_reduce_numel', 0)}",
                flush=True,
            )
        self._muon_profile_ns_ms = 0.0
        self._muon_profile_ns_calls = 0
        self._muon_profile_ns_launches = 0
        self._muon_profile_apply_ms = 0.0
        self._muon_profile_apply_batches = 0
        self._muon_profile_fs_a2a_ms = 0.0
        self._muon_profile_fs_a2a_calls = 0

    def _prepare_entry(self, entry):
        param = entry.param if entry.param.ndim == 2 else entry.param.view(entry.param_shape)
        grad = entry.grad if entry.grad.ndim == 2 else entry.grad.view(entry.param_shape)
        momentum = entry.optimizer_state.get("momentum_buffer")
        if momentum is None:
            momentum = entry.optimizer_state["momentum"]
            entry.optimizer_state["momentum_buffer"] = momentum
        update = apply_muon_momentum(momentum, grad, beta=entry.config.momentum_beta, nesterov=entry.config.use_nesterov)
        self._apply_weight_decay(param, entry.optim_group or {})
        return param, update

    def _commit(self, entry, param, update):
        lr = float((entry.optim_group or {}).get("lr", getattr(getattr(self, "config", None), "lr", 0.0)))
        param.add_(update.to(param.dtype), alpha=-lr)
        if entry.commit_update is not None:
            entry.commit_update(param, entry.optimizer_state.get("momentum_buffer"))

    def _apply_local_entry(self, entry):
        param, update = self._prepare_entry(entry)
        self._commit(entry, param, self._orthogonalize(update, entry))

    def _apply_local_batch(self, entries):
        prepared = [self._prepare_entry(entry) for entry in entries]
        params = [param for param, _ in prepared]
        updates = [update for _, update in prepared]
        orthogonalized = self._orthogonalize_batch(updates, entries)
        for entry, param, update in zip(entries, params, orthogonalized):
            self._commit(entry, param, update)

    def _fs_full_shape(self, entry):
        meta = entry.dist_meta
        shape = tuple(int(dim) for dim in entry.param_shape)
        global_shape = tuple(int(dim) for dim in entry.global_shape)
        if int(meta.fs_shard_dim) == 0:
            return global_shape[0], shape[1]
        if int(meta.fs_shard_dim) == 1:
            return shape[0], global_shape[1]
        return shape

    def _fs_capacity(self, entry):
        meta = entry.dist_meta
        fs_dim = int(meta.fs_shard_dim)
        full_shape = self._fs_full_shape(entry)
        split_size = int(full_shape[fs_dim])
        other = int(full_shape[1 - fs_dim])
        ranges = self._fs_rank_ranges(entry, split_size)
        return max(e - s for s, e in ranges) * other, ranges, other

    @staticmethod
    def _linear_child_row_range(parent_row_start, parent_row_end, split_rows, child_kind):
        child_start = 0 if child_kind == "gate" else int(split_rows[0])
        child_end = child_start + int(split_rows[0 if child_kind == "gate" else 1])
        overlap_start = max(int(parent_row_start), child_start)
        overlap_end = min(int(parent_row_end), child_end)
        if overlap_end <= overlap_start:
            return None
        return overlap_start - child_start, overlap_end - child_start

    def _fs_rank_ranges(self, entry, split_size: int):
        meta = entry.dist_meta
        world_size = int(meta.fs_world_size)
        if int(meta.fs_shard_dim) != 0:
            return [_split_range(split_size, world_size, r) for r in range(world_size)]

        if bool(getattr(meta, "is_qkv_child", False)):
            shapes = tuple(int(dim) for dim in getattr(meta, "qkv_split_shapes"))
            child_kind = getattr(meta, "qkv_child_kind")
            child_rows_per_group = int(shapes[{"q": 0, "k": 1, "v": 2}[child_kind]])
            parent_rows = int(split_size) // child_rows_per_group * int(sum(shapes))
            return [
                qkv_child_row_range(
                    parent_row_start=parent_start,
                    parent_row_end=parent_end,
                    split_shapes=shapes,
                    child_kind=child_kind,
                )
                or (0, 0)
                for parent_start, parent_end in (
                    _split_range(parent_rows, world_size, rank) for rank in range(world_size)
                )
            ]

        if bool(getattr(meta, "is_qkvg_child", False)):
            shapes = tuple(int(dim) for dim in getattr(meta, "qkvg_split_shapes"))
            child_kind = getattr(meta, "qkvg_child_kind")
            child_rows_per_group = int(shapes[{"q": 0, "gate": 1, "k": 2, "v": 3}[child_kind]])
            parent_rows = int(split_size) // child_rows_per_group * int(sum(shapes))
            return [
                qkvg_child_row_range(
                    parent_row_start=parent_start,
                    parent_row_end=parent_end,
                    split_shapes=shapes,
                    child_kind=child_kind,
                )
                or (0, 0)
                for parent_start, parent_end in (
                    _split_range(parent_rows, world_size, rank) for rank in range(world_size)
                )
            ]

        if bool(getattr(meta, "is_linear_child", False)):
            split_rows = tuple(int(dim) for dim in getattr(meta, "linear_split_rows"))
            child_kind = getattr(meta, "linear_child_kind")
            parent_rows = int(sum(split_rows))
            return [
                self._linear_child_row_range(parent_start, parent_end, split_rows, child_kind)
                or (0, 0)
                for parent_start, parent_end in (
                    _split_range(parent_rows, world_size, rank) for rank in range(world_size)
                )
            ]

        return [_split_range(split_size, world_size, r) for r in range(world_size)]

    def _put_shard(self, full, shard, entry, rank):
        _, ranges, _ = self._fs_capacity(entry)
        s, e = ranges[rank]
        if int(entry.dist_meta.fs_shard_dim) == 0:
            full[s:e, :].copy_(shard)
        else:
            full[:, s:e].copy_(shard)

    def _get_shard(self, full, entry, rank):
        _, ranges, _ = self._fs_capacity(entry)
        s, e = ranges[rank]
        return full[s:e, :].contiguous() if int(entry.dist_meta.fs_shard_dim) == 0 else full[:, s:e].contiguous()

    def _unpack(self, flat, entry, rank):
        _, ranges, other = self._fs_capacity(entry)
        s, e = ranges[rank]
        split = e - s
        shape = (split, other) if int(entry.dist_meta.fs_shard_dim) == 0 else (other, split)
        return flat[: math.prod(shape)].view(shape).contiguous()

    def _apply_fs_duplicated_entry(self, entry):
        meta = entry.dist_meta
        if getattr(meta, "fs_group", None) is None or int(meta.fs_world_size) <= 1:
            self._apply_local_entry(entry)
            return
        param, update = self._prepare_entry(entry)
        cap, _, _ = self._fs_capacity(entry)
        send = update.new_zeros((cap,))
        send[: update.numel()].copy_(update.contiguous().view(-1))
        gathered = [torch.empty_like(send) for _ in range(int(meta.fs_world_size))]
        dist.all_gather(gathered, send, group=meta.fs_group)
        full = update.new_empty(self._fs_full_shape(entry))
        for rank, flat in enumerate(gathered):
            self._put_shard(full, self._unpack(flat, entry, rank), entry, rank)
        full_meta = replace(meta, shape=tuple(full.shape), local_shape=tuple(full.shape), fs_group=None, fs_world_size=1, fs_rank=0)
        local = self._get_shard(self._orthogonalize(full, entry, full_meta), entry, int(meta.fs_rank))
        self._commit(entry, param, local)

    def _apply_fs_duplicated_batch(self, entries):
        if not entries:
            return
        first_meta = entries[0].dist_meta
        if getattr(first_meta, "fs_group", None) is None or int(first_meta.fs_world_size) <= 1:
            self._apply_local_batch(entries)
            return

        params = []
        full_updates = []
        full_metas = []
        for entry in entries:
            meta = entry.dist_meta
            param, update = self._prepare_entry(entry)
            params.append(param)
            cap, _, _ = self._fs_capacity(entry)
            send = update.new_zeros((cap,))
            send[: update.numel()].copy_(update.contiguous().view(-1))
            gathered = [torch.empty_like(send) for _ in range(int(meta.fs_world_size))]
            dist.all_gather(gathered, send, group=meta.fs_group)
            full = update.new_empty(self._fs_full_shape(entry))
            for rank, flat in enumerate(gathered):
                self._put_shard(full, self._unpack(flat, entry, rank), entry, rank)
            full_updates.append(full)
            full_metas.append(
                replace(
                    meta,
                    shape=tuple(full.shape),
                    local_shape=tuple(full.shape),
                    fs_group=None,
                    fs_world_size=1,
                    fs_rank=0,
                )
            )

        orthogonalized = self._orthogonalize_batch(full_updates, entries, full_metas)
        for entry, param, update in zip(entries, params, orthogonalized):
            local = self._get_shard(update, entry, int(entry.dist_meta.fs_rank))
            self._commit(entry, param, local)

    def _distributed_fs_group_info(self, entries):
        active = [
            entry
            for entry in entries
            if int(getattr(entry.dist_meta, "fs_world_size", 1)) > 1
        ]
        if not active:
            return None, 1, 0

        group = getattr(active[0].dist_meta, "fs_group", None)
        if group is None:
            raise RuntimeError("[MUON_FS_DISTRIBUTED_REQUIRES_GROUP]")
        world_size = int(getattr(active[0].dist_meta, "fs_world_size", _group_size(group)))
        rank = int(getattr(active[0].dist_meta, "fs_rank", _group_rank(group)))
        fs_dim = int(getattr(active[0].dist_meta, "fs_shard_dim", -1))
        if fs_dim not in (0, 1):
            raise RuntimeError(f"[MUON_FS_DISTRIBUTED_INVALID_SHARD_DIM] fs_shard_dim={fs_dim}")
        if rank < 0 or rank >= world_size:
            raise RuntimeError(
                f"[MUON_FS_DISTRIBUTED_INVALID_RANK] fs_rank={rank} fs_world_size={world_size}"
            )

        for entry in entries:
            meta = entry.dist_meta
            if int(getattr(meta, "fs_world_size", 1)) != world_size:
                raise RuntimeError("[MUON_FS_DISTRIBUTED_MIXED_WORLD_SIZES]")
            if getattr(meta, "fs_group", group) is not group:
                raise RuntimeError("[MUON_FS_DISTRIBUTED_MIXED_GROUPS]")
            if int(getattr(meta, "fs_shard_dim", fs_dim)) != fs_dim:
                raise RuntimeError("[MUON_FS_DISTRIBUTED_MIXED_SHARD_DIMS]")
        return group, world_size, rank

    @staticmethod
    def _owner_schedule(entry_count: int, world_size: int) -> list[list[int]]:
        owners = [[] for _ in range(int(world_size))]
        for index in range(int(entry_count)):
            owners[index % int(world_size)].append(index)
        return owners

    @staticmethod
    def _pack_flat(dest, offset: int, tensor, capacity: int) -> int:
        flat = tensor.contiguous().view(-1)
        if int(flat.numel()) > int(capacity):
            raise RuntimeError(
                f"[MUON_FS_DISTRIBUTED_PACK_OVERFLOW] numel={int(flat.numel())} "
                f"capacity={int(capacity)}"
            )
        dest.narrow(0, int(offset), int(flat.numel())).copy_(flat)
        return int(offset) + int(capacity)

    def _pack_updates_for_owners(self, updates, entries, owners, capacities, split_sizes):
        send = updates[0].new_zeros((sum(int(size) for size in split_sizes),))
        offset = 0
        for owner, owner_indices in enumerate(owners):
            owner_start = offset
            for index in owner_indices:
                offset = self._pack_flat(send, offset, updates[index], capacities[index])
            expected = owner_start + int(split_sizes[owner])
            if offset != expected:
                raise RuntimeError("[MUON_FS_DISTRIBUTED_PACK_SIZE_MISMATCH]")
        return send

    def _extract_owner_full_updates(self, recv, entries, owners, capacities, owner_rank, world_size):
        owner_indices = owners[owner_rank]
        owner_size = sum(int(capacities[index]) for index in owner_indices)
        full_updates = {}
        prefix = {}
        cursor = 0
        for index in owner_indices:
            prefix[index] = cursor
            cursor += int(capacities[index])

        owner_entries = []
        owner_full_updates = []
        owner_full_metas = []
        for index in owner_indices:
            entry = entries[index]
            full = recv.new_empty(self._fs_full_shape(entry))
            for source_rank in range(int(world_size)):
                offset = source_rank * owner_size + prefix[index]
                shard = self._unpack(
                    recv.narrow(0, int(offset), int(capacities[index])),
                    entry,
                    source_rank,
                )
                self._put_shard(full, shard, entry, source_rank)
            owner_entries.append(entry)
            owner_full_updates.append(full)
            owner_full_metas.append(
                replace(
                    entry.dist_meta,
                    shape=tuple(full.shape),
                    local_shape=tuple(full.shape),
                    fs_group=None,
                    fs_world_size=1,
                    fs_rank=0,
                )
            )
        for index, update in zip(
            owner_indices,
            self._orthogonalize_batch(owner_full_updates, owner_entries, owner_full_metas),
        ):
            full_updates[index] = update
        return full_updates

    def _pack_owner_results(self, full_updates, entries, owners, capacities, owner_rank, world_size):
        owner_indices = owners[owner_rank]
        owner_size = sum(int(capacities[index]) for index in owner_indices)
        if not owner_indices:
            return None
        send = next(iter(full_updates.values())).new_zeros((owner_size * int(world_size),))
        for target_rank in range(int(world_size)):
            offset = target_rank * owner_size
            for index in owner_indices:
                shard = self._get_shard(full_updates[index], entries[index], target_rank)
                offset = self._pack_flat(send, offset, shard, capacities[index])
        return send

    def _unpack_local_results(self, recv, entries, owners, capacities, split_sizes, local_rank):
        local_updates = {}
        offset = 0
        for owner, owner_indices in enumerate(owners):
            owner_start = offset
            for index in owner_indices:
                local_updates[index] = self._unpack(
                    recv.narrow(0, int(offset), int(capacities[index])),
                    entries[index],
                    local_rank,
                )
                offset += int(capacities[index])
            expected = owner_start + int(split_sizes[owner])
            if offset != expected:
                raise RuntimeError("[MUON_FS_DISTRIBUTED_UNPACK_SIZE_MISMATCH]")
        return local_updates

    def _all_to_all_single(self, output, input_, *, output_split_sizes, input_split_sizes, group):
        if bool(getattr(self, "_muon_profile_ns", False)) and output.is_cuda:
            torch.cuda.synchronize(output.device)
            start = time.perf_counter()
            dist.all_to_all_single(
                output,
                input_,
                output_split_sizes=output_split_sizes,
                input_split_sizes=input_split_sizes,
                group=group,
            )
            torch.cuda.synchronize(output.device)
            self._muon_profile_fs_a2a_ms += (time.perf_counter() - start) * 1000.0
            self._muon_profile_fs_a2a_calls += 1
            return
        dist.all_to_all_single(
            output,
            input_,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
        )

    def _apply_fs_distributed_batch(self, batch: MuonBatch):
        entries = tuple(batch.entries)
        group, world_size, rank = self._distributed_fs_group_info(entries)
        if group is None or int(world_size) <= 1:
            for entry in entries:
                self._apply_local_entry(entry)
            return
        if not hasattr(dist, "all_to_all_single"):
            raise RuntimeError("[MUON_FS_DISTRIBUTED_REQUIRES_ALL_TO_ALL_SINGLE]")

        prepared = [self._prepare_entry(entry) for entry in entries]
        params = [param for param, _ in prepared]
        updates = [update for _, update in prepared]
        dtype = updates[0].dtype
        device = updates[0].device
        if any(update.dtype != dtype or update.device != device for update in updates):
            raise RuntimeError("[MUON_FS_DISTRIBUTED_MIXED_DTYPES_OR_DEVICES]")

        owners = self._owner_schedule(len(entries), int(world_size))
        capacities = [int(self._fs_capacity(entry)[0]) for entry in entries]
        split_sizes = [
            sum(int(capacities[index]) for index in owner_indices)
            for owner_indices in owners
        ]

        send = self._pack_updates_for_owners(updates, entries, owners, capacities, split_sizes)
        owner_size = int(split_sizes[rank])
        recv = updates[0].new_empty((owner_size * int(world_size),))
        self._all_to_all_single(
            recv,
            send,
            output_split_sizes=[owner_size] * int(world_size),
            input_split_sizes=[int(size) for size in split_sizes],
            group=group,
        )

        full_updates = self._extract_owner_full_updates(
            recv,
            entries,
            owners,
            capacities,
            int(rank),
            int(world_size),
        )
        reverse_send = self._pack_owner_results(
            full_updates,
            entries,
            owners,
            capacities,
            int(rank),
            int(world_size),
        )
        if reverse_send is None:
            reverse_send = updates[0].new_empty((0,))

        reverse_recv = updates[0].new_empty((sum(int(size) for size in split_sizes),))
        self._all_to_all_single(
            reverse_recv,
            reverse_send,
            output_split_sizes=[int(size) for size in split_sizes],
            input_split_sizes=[owner_size] * int(world_size),
            group=group,
        )

        local_updates = self._unpack_local_results(
            reverse_recv,
            entries,
            owners,
            capacities,
            split_sizes,
            int(rank),
        )
        for index, entry in enumerate(entries):
            self._commit(entry, params[index], local_updates[index])

    def _apply_muon_batches(self, batches: Sequence[MuonBatch]):
        profile = bool(getattr(self, "_muon_profile_ns", False)) and torch.cuda.is_available()
        device = torch.device("cuda", torch.cuda.current_device()) if profile else None
        if profile:
            torch.cuda.synchronize(device)
            start = time.perf_counter()
        try:
            for batch in batches:
                if not batch.entries:
                    continue
                self._muon_profile_apply_batches += 1
                mode = batch.entries[0].config.fs_mode
                if mode == "blockwise":
                    self._apply_local_batch(batch.entries)
                elif mode == "duplicated_debug":
                    self._apply_fs_duplicated_batch(batch.entries)
                elif mode == "distributed":
                    self._apply_fs_distributed_batch(batch)
                else:
                    raise RuntimeError(f"[MUON_INVALID_FS_MODE] fs_mode={mode!r}")
        finally:
            if profile:
                torch.cuda.synchronize(device)
                self._muon_profile_apply_ms += (time.perf_counter() - start) * 1000.0

    def clip_grad_norm(self, clip_grad: float) -> float:
        return self.clip_matrix_grad_norm(
            clip_grad,
            is_matrix_model_param=lambda param: getattr(param, "is_muon_param", False),
        )

    @torch.no_grad()
    def step_with_ready_grads(self) -> bool:
        batches, _ = self._route_step_params()
        matrix_params = set(self._matrix_params())
        saved_grads = {}
        for param in matrix_params:
            if self.config.use_precision_aware_optimizer_no_fp8_or_ds_fp8:
                saved_grads[param] = getattr(param, "decoupled_grad", None)
                param.decoupled_grad = None
            else:
                saved_grads[param] = getattr(param, "grad", None)
                param.grad = None
        self.optimizer.step()
        for param, grad in saved_grads.items():
            if self.config.use_precision_aware_optimizer_no_fp8_or_ds_fp8:
                param.decoupled_grad = grad
            else:
                param.grad = grad
        self._apply_muon_batches(batches)
        self._maybe_log_ns_profile()
        self._copy_main_params_to_model_params()
        if not self.ddp_config.overlap_param_gather:
            for model_chunk in self.model_chunks:
                model_chunk.start_param_sync()
        return True

    def _muon_param_key(self, param):
        meta = self.dist_metas.get(param)
        return getattr(meta, "param_uid", None) or getattr(param, "_muon_param_uid", None)

    def _resolve_muon_tp_group(self):
        group = getattr(self, "_muon_tp_group", None)
        if group is None:
            group = (
                parallel_state.get_expert_tensor_parallel_group(check_initialized=False)
                if getattr(self, "_is_expert_muon", False)
                else parallel_state.get_tensor_model_parallel_group(check_initialized=False)
            )
        if group is not None:
            self._assert_group_excludes_context_parallel(group, label="muon_tp_group")
        return group

    def _muon_checkpoint_topology_signature(self) -> dict:
        return self._matrix_checkpoint_topology_signature(
            fs_group=getattr(self, "fs_group", None),
            tp_group=self._resolve_muon_tp_group(),
            rp_group=getattr(self, "rp_group", None),
            state_replica_group=getattr(self, "state_replica_group", None),
        )

    def sharded_state_dict(self, model_sharded_state_dict=None, is_loading=False, sharding_type=None, metadata=None):
        """Build torch_dist-compatible Muon optimizer checkpoint state."""
        from ....dist_checkpointing.mapping import ShardedObject, ShardedTensor

        del model_sharded_state_dict, sharding_type
        dp_rank = self.data_parallel_group.rank()
        dp_size = self.data_parallel_group.size()
        base_key = f"optimizer.distributed.dp_group_idx_{self.data_parallel_group_idx}"
        common_replica_id = (self.distributed_optimizer_instance_id, 0, dp_rank)
        tp_group = self._resolve_muon_tp_group()
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

        checkpoint_metadata = build_muon_checkpoint_metadata(
            dp_size=dp_size,
            fs_size=int(getattr(self, "fs_size", 1)),
            tp_size=int(tp_size),
            rp_size=int(rp_size),
            state_replica_size=int(state_replica_size),
            requested_type=(metadata or {}).get("distrib_optim_sharding_type", "muon_rank_local_state"),
            topology_signature=self._muon_checkpoint_topology_signature(),
            backend_state_spec=self._require_matrix_backend().state_spec(),
        )
        return build_distributed_checkpoint_state(
            common_state=self.state_dict(),
            param_groups=self.optimizer.param_groups,
            optimizer_state=self.optimizer.state,
            get_param_key=self._muon_param_key,
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
        if "muon_checkpoint_metadata" not in state_dict:
            return super().load_state_dict(state_dict)
        metadata, _, common = split_distributed_checkpoint_state(state_dict)
        tp_group = self._resolve_muon_tp_group()
        tp_size = 1 if tp_group is None else self._group_size(tp_group)
        state_replica_group = getattr(self, "state_replica_group", None)
        state_replica_size = 1 if state_replica_group is None else self._group_size(state_replica_group)
        rp_group = getattr(self, "rp_group", None)
        rp_size = 1 if rp_group is None else self._group_size(rp_group)
        validate_muon_checkpoint_metadata(
            metadata,
            dp_size=self.data_parallel_group.size(),
            fs_size=int(getattr(self, "fs_size", 1)),
            tp_size=int(tp_size),
            rp_size=int(rp_size),
            state_replica_size=int(state_replica_size),
            topology_signature=self._muon_checkpoint_topology_signature(),
            backend_state_spec=self._require_matrix_backend().state_spec(),
        )
        result = super().load_state_dict(common)
        _, param_state, _ = split_distributed_checkpoint_state(state_dict)
        restore_muon_param_state_(
            self.optimizer.param_groups,
            self.optimizer.state,
            param_state,
            self._muon_param_key,
        )
        return result


__all__ = ["DistributedMuonOptimizer"]
