"""Generic base class for matrix-aware distributed optimizers."""

import logging
from typing import Optional

import torch
import torch.distributed as dist

from ... import parallel_state
from ...fp8_utils import quantize_param_shard
from ...transformer.fsdp_dtensor_checkpoint import get_global_unique_param_name
from ..clip_grads import clip_grad_by_total_norm_fp32
from ..distrib_optimizer import DistributedOptimizer
from .backend import MatrixBackend
from .checkpoint_io import copy_main_params_to_model_shards, copy_model_params_to_main_shards
from .grad_norm import compute_grad_norm, grad_norm_inputs, matrix_replica_grads
from .gradients import (
    apply_bucket_grads,
    clear_grad_transport,
    clear_matrix_local_grads,
    get_local_grad,
    get_standard_inter_instance_grad_buffer,
    release_rs_buffers,
    scale_matrix_bucket_grads,
    scale_matrix_local_grads,
    set_optimizer_shard_grads,
    start_matrix_grad_sync,
)
from .parameter import (
    all_gather_bucket_params_,
    bucket_matrix_param_view,
    check_bucket_param_views,
    collect_matrix_bucket_params,
    restore_matrix_shards_from_bucket_,
    set_bucket_param_views,
)
from .sharding import (
    get_data_shard,
    get_opt_shard,
    param_shard_layout,
    update_data_shard,
    update_opt_shard,
)


logger = logging.getLogger(__name__)


class DistributedMatrixOptimizer(DistributedOptimizer):
    """Distributed optimizer base with a matrix backend boundary."""

    matrix_backend: MatrixBackend | None = None

    def set_matrix_backend(self, backend: MatrixBackend) -> None:
        self.matrix_backend = backend

    def _require_matrix_backend(self) -> MatrixBackend:
        backend = self.matrix_backend
        if backend is None:
            raise RuntimeError("DistributedMatrixOptimizer requires a matrix backend")
        return backend

    def _validate_matrix_topology(self, **kwargs) -> None:
        self._require_matrix_backend().validate_topology(**kwargs)

    @staticmethod
    def _group_size(group) -> int:
        return group.size() if hasattr(group, "size") else dist.get_world_size(group)

    @staticmethod
    def _group_rank(group) -> int:
        return group.rank() if hasattr(group, "rank") else dist.get_rank(group)

    @staticmethod
    def _assert_group_excludes_context_parallel(group, *, label: str) -> None:
        if group is None or not dist.is_initialized():
            return
        cp_group = parallel_state.get_context_parallel_group(check_initialized=False)
        if cp_group is None:
            return
        cp_ranks = tuple(int(rank) for rank in dist.get_process_group_ranks(cp_group))
        if len(cp_ranks) <= 1:
            return
        group_ranks = tuple(int(rank) for rank in dist.get_process_group_ranks(group))
        global_rank = int(dist.get_rank())
        overlap = tuple(rank for rank in group_ranks if rank in set(cp_ranks))
        if overlap != (global_rank,):
            raise RuntimeError(
                "[MATRIX_CP_GROUP_LEAK] "
                f"{label} must exclude context-parallel peers: "
                f"group_ranks={group_ranks} context_parallel_ranks={cp_ranks} "
                f"overlap={overlap} global_rank={global_rank}"
            )

    def _resolve_state_replica_group(self):
        """Return the standard DO state-replica group, independent of backend RP."""
        try:
            state_group = parallel_state.get_inter_distributed_optimizer_instance_group(
                check_initialized=False
            )
        except Exception:
            state_group = None
        if state_group is None or self._group_size(state_group) <= 1:
            return None
        self._assert_group_excludes_context_parallel(
            state_group,
            label="state_replica_group",
        )
        return state_group

    def _get_replicate_group(self):
        """Return the matrix optimizer RP group, if this topology has replicas."""
        group = getattr(self, "rp_group", None)
        if group is None:
            group = getattr(self, "_replica_group", None)
        if group is None or self._group_size(group) <= 1:
            return None
        self._assert_group_excludes_context_parallel(group, label="replicate_group")
        return group

    def _shard_param_uid(self, shard_param):
        """Return the backend-neutral identity for one optimizer shard."""
        param_uid = getattr(shard_param, "_matrix_param_uid", None)
        if param_uid is not None:
            return param_uid

        dist_metas = getattr(self, "dist_metas", {})
        dist_meta = dist_metas.get(shard_param)
        if dist_meta is not None and getattr(dist_meta, "param_uid", None) is not None:
            shard_param._matrix_param_uid = dist_meta.param_uid
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
                param_uid = getattr(candidate, "_matrix_param_uid", None)
                if param_uid is not None:
                    shard_param._matrix_param_uid = param_uid
                    candidate_meta = dist_metas.get(candidate)
                    if candidate_meta is not None:
                        dist_metas[shard_param] = candidate_meta
                    return param_uid

                candidate_meta = dist_metas.get(candidate)
                if candidate_meta is not None and getattr(candidate_meta, "param_uid", None) is not None:
                    shard_param._matrix_param_uid = candidate_meta.param_uid
                    dist_metas[shard_param] = candidate_meta
                    return candidate_meta.param_uid

        raise RuntimeError(
            "[Matrix] missing param_uid for optimizer shard "
            f"name={self._param_name(shard_param) or f'id_{id(shard_param)}'} "
            f"shape={tuple(shard_param.shape)}"
        )

    @classmethod
    def _normalize_group_info(cls, group_or_info, *, bucket_id: int, label: str):
        """Return ``(group, size, rank)`` from a group or MCore group-info tuple."""
        if isinstance(group_or_info, tuple) and len(group_or_info) == 3:
            group, group_size, group_rank = group_or_info
            if isinstance(group, (int, str, tuple, list)):
                raise RuntimeError(
                    f"[MATRIX_INVALID_GROUP] invalid {label} process group "
                    f"for bucket {bucket_id}: type={type(group).__name__}"
                )
            if group_size is None:
                group_size = cls._group_size(group)
            if group_rank is None:
                group_rank = cls._group_rank(group)
            return group, int(group_size), int(group_rank)
        if isinstance(group_or_info, tuple):
            raise RuntimeError(
                f"[MATRIX_INVALID_GROUP_INFO] invalid {label} group info "
                f"for bucket {bucket_id}: type={type(group_or_info).__name__} "
                f"len={len(group_or_info)}"
            )
        return group_or_info, cls._group_size(group_or_info), cls._group_rank(group_or_info)

    @classmethod
    def _get_bucket_shard_group(cls, param_and_grad_buffer, bucket):
        """Resolve the MCore group that owns one distributed optimizer bucket shard."""
        shard_group = getattr(bucket, "intra_distributed_optimizer_instance_group", None)
        if shard_group is not None:
            return (
                shard_group,
                int(getattr(bucket, "intra_distributed_optimizer_instance_size", shard_group.size())),
                int(getattr(bucket, "intra_distributed_optimizer_instance_rank", shard_group.rank())),
            )

        shard_group = getattr(bucket, "data_parallel_group", None)
        if shard_group is None:
            shard_group = getattr(param_and_grad_buffer, "data_parallel_group", None)
        if shard_group is None:
            raise RuntimeError(
                "[MATRIX_MISSING_BUCKET_SHARD_GROUP] "
                f"bucket={getattr(bucket, 'bucket_id', -1)} has no MCore shard group"
            )
        return shard_group, cls._group_size(shard_group), cls._group_rank(shard_group)

    @classmethod
    def _bucket_shard_get_group_size_rank(cls, param_and_grad_buffer, bucket):
        """Return the standard bucket shard group and this rank's position in it."""
        shard_group, shard_size, shard_rank = cls._normalize_group_info(
            cls._get_bucket_shard_group(param_and_grad_buffer, bucket),
            bucket_id=bucket.bucket_id,
            label="bucket shard",
        )
        if shard_size <= 0:
            raise RuntimeError(
                "[MATRIX_INVALID_BUCKET_SHARD_GROUP] "
                f"bucket={bucket.bucket_id} shard_size={shard_size}"
            )
        return shard_group, shard_size, shard_rank

    @classmethod
    def _has_local_matrix_shard(cls, range_info: dict) -> bool:
        """Return whether this rank owns a Matrix FS shard for one model param."""
        shard_layout = range_info.get("matrix_shard_layout", None)
        if shard_layout is None:
            return False
        for dim in shard_layout.local_shape:
            if int(dim) <= 0:
                raise RuntimeError(
                    "[Matrix] invalid empty local matrix shard "
                    f"shape={tuple(int(x) for x in shard_layout.local_shape)}"
                )
        return True

    @classmethod
    def _build_model_param_gbuf_map(cls, gbuf_ranges: list[dict]) -> dict[torch.nn.Parameter, tuple]:
        """Create the reverse mapping for this-rank optimizer params."""
        param_gbuf_map = {}
        for gbuf_index, gbuf_range_map in enumerate(gbuf_ranges):
            for dtype, bucket_range_maps in gbuf_range_map.items():
                for bucket_index, bucket_range_map in enumerate(bucket_range_maps):
                    for param, range_info in bucket_range_map["param_map"].items():
                        param_range = range_info.get("param", None)
                        has_local_range = param_range is not None and param_range.size > 0
                        has_local_matrix_shard = cls._has_local_matrix_shard(range_info)
                        if not has_local_range and not has_local_matrix_shard:
                            continue
                        assert param not in param_gbuf_map, (
                            "Param should not appear in model_param_gbuf_map more than once; "
                            "only this-rank optimizer params belong in the map."
                        )
                        param_gbuf_map[param] = (gbuf_index, dtype, bucket_index)
        return param_gbuf_map

    @classmethod
    def _build_optimizer_group_ranges(cls, param_groups: list[dict], gbuf_ranges: list[dict]):
        """Build optimizer groups from this-rank standard params and Matrix shards."""
        world_param_group_map = {}
        for group_index, optim_group in enumerate(param_groups):
            for param in optim_group["params"]:
                assert param.requires_grad
                world_param_group_map[param] = group_index

        local_param_group_map = {}
        group_ranges = [{"params": []} for _ in param_groups]
        for gbuf_range_map in gbuf_ranges:
            for _, bucket_range_maps in gbuf_range_map.items():
                for bucket_range_map in bucket_range_maps:
                    for param, range_info in bucket_range_map["param_map"].items():
                        param_range = range_info.get("param", None)
                        has_local_range = param_range is not None and param_range.size > 0
                        has_local_matrix_shard = cls._has_local_matrix_shard(range_info)
                        if not has_local_range and not has_local_matrix_shard:
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

    @staticmethod
    def _get_group_ranks_for_checkpoint(group) -> tuple[int, ...]:
        if group is None:
            return ()
        if hasattr(group, "ranks"):
            return tuple(int(rank) for rank in group.ranks)
        return tuple(int(rank) for rank in dist.get_process_group_ranks(group))

    def _matrix_checkpoint_topology_signature(
        self,
        *,
        fs_group=None,
        tp_group=None,
        rp_group=None,
        state_replica_group=None,
    ) -> dict:
        return {
            "data_parallel": self._get_group_ranks_for_checkpoint(self.data_parallel_group),
            "fs": self._get_group_ranks_for_checkpoint(fs_group),
            "tp": self._get_group_ranks_for_checkpoint(tp_group),
            "rp": self._get_group_ranks_for_checkpoint(rp_group),
            "state_replica": self._get_group_ranks_for_checkpoint(state_replica_group),
        }

    def save_parameter_state(self, filename: str):
        raise NotImplementedError(
            f"{type(self).__name__} does not support legacy torch optimizer "
            f"checkpoint parameter state ({filename}). Use --ckpt-format torch_dist "
            "for optimizer checkpointing, or --no-save-optim for model-only checkpoints."
        )

    def load_parameter_state(self, filename: str, *, update_legacy_format=False):
        del update_legacy_format
        raise NotImplementedError(
            f"{type(self).__name__} does not support legacy torch optimizer "
            f"checkpoint parameter state ({filename}). Use --ckpt-format torch_dist "
            "for optimizer checkpointing, or --no-load-optim for model-only checkpoints."
        )

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

    def _param_name(self, param) -> Optional[str]:
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

    def _global_param_name(self, param) -> Optional[str]:
        """Return a PP/EP-invariant parameter name when available."""
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

        return self._param_name(param)

    def _set_bucket_runtime_flag(self, attr_name: str, value) -> None:
        """Set one existing bucket runtime flag across all optimizer buckets."""
        if not hasattr(self, "buffers"):
            return
        for buffer in self.buffers:
            for bucket in buffer.buckets:
                if hasattr(bucket, attr_name):
                    setattr(bucket, attr_name, value)

    def _route_matrix_step_params(
        self,
        *,
        param_groups,
        dist_metas,
        get_step_param_grad,
        ensure_optimizer_state,
        require_param_config,
    ):
        from .runtime import route_step_params

        backend = self._require_matrix_backend()
        return route_step_params(
            param_groups=param_groups,
            dist_metas=dist_metas,
            get_step_param_grad=get_step_param_grad,
            ensure_optimizer_state=ensure_optimizer_state,
            require_param_config=require_param_config,
            use_matrix=lambda param, state, optim_group, dist_meta: backend.use_matrix(
                self,
                param=param,
                state=state,
                optim_group=optim_group,
                dist_meta=dist_meta,
            ),
            split_children=lambda param, grad, state, optim_group, config, dist_meta: (
                backend.split_children(
                    self,
                    param=param,
                    grad=grad,
                    state=state,
                    optim_group=optim_group,
                    config=config,
                    dist_meta=dist_meta,
                )
            ),
            refresh_state=lambda param, state, optim_group, dist_meta: backend.refresh_state(
                self,
                param=param,
                state=state,
                optim_group=optim_group,
                dist_meta=dist_meta,
            ),
            sync_state=lambda matrix_params: backend.sync_state(self, matrix_params),
            build_batches=lambda matrix_params: backend.build_batches(self, matrix_params),
        )

    def _all_gather_bucket_params_(self, bucket, async_op=False):
        return all_gather_bucket_params_(self, bucket, async_op=async_op)

    def _get_data_shard(self, model_param):
        return get_data_shard(self, model_param)

    def _get_opt_shard(self, model_param):
        return get_opt_shard(self, model_param)

    def _update_data_shard(self, model_param, new_data_shard) -> None:
        update_data_shard(self, model_param, new_data_shard)

    def _update_opt_shard(self, model_param, new_opt_shard) -> None:
        update_opt_shard(self, model_param, new_opt_shard)

    def _param_shard_layout(self, model_param):
        return param_shard_layout(self, model_param)

    def _bucket_matrix_param_view(self, bucket, entry):
        return bucket_matrix_param_view(self, bucket, entry)

    def _set_bucket_param_views(self, bucket, *, copy_data: bool, params=None) -> None:
        set_bucket_param_views(self, bucket, copy_data=copy_data, params=params)

    def _check_bucket_param_views(self, bucket, *, context: str, params=None) -> None:
        check_bucket_param_views(self, bucket, context=context, params=params)

    def _clear_matrix_local_grads(self, params=None) -> None:
        clear_matrix_local_grads(self, params)

    @staticmethod
    def _clear_matrix_grad_transport(bucket) -> None:
        clear_grad_transport(bucket)

    def _scale_matrix_bucket_grads(
        self,
        *,
        bucket,
        local_data_view,
        communication_group,
        scaling_factor: float,
        use_distributed_optimizer: bool,
    ) -> None:
        scale_matrix_bucket_grads(
            self,
            bucket=bucket,
            local_data_view=local_data_view,
            communication_group=communication_group,
            scaling_factor=scaling_factor,
            use_distributed_optimizer=use_distributed_optimizer,
        )

    def _scale_matrix_local_grads(self, params=None, scaling_factor: float = 1.0) -> None:
        scale_matrix_local_grads(self, params, scaling_factor)

    def _start_matrix_grad_sync(
        self,
        *,
        bucket,
        local_data_view,
        communication_group,
        reduce_op,
        async_op: bool,
        reduce_scatter,
    ):
        return start_matrix_grad_sync(
            self,
            bucket=bucket,
            local_data_view=local_data_view,
            communication_group=communication_group,
            reduce_op=reduce_op,
            async_op=async_op,
            reduce_scatter=reduce_scatter,
        )

    def _apply_bucket_grads(self, *, bucket, local_data_view, communication_group) -> None:
        apply_bucket_grads(
            self,
            bucket=bucket,
            local_data_view=local_data_view,
            communication_group=communication_group,
        )

    @staticmethod
    def _get_standard_inter_instance_grad_buffer(bucket):
        return get_standard_inter_instance_grad_buffer(bucket)

    @staticmethod
    def _get_inter_instance_grad_buffers(bucket):
        standard_grad = get_standard_inter_instance_grad_buffer(bucket)
        return (standard_grad,) if standard_grad is not None and standard_grad.numel() > 0 else ()

    def _matrix_grads_are_replicate_synced(self) -> bool:
        return False

    def _set_local_grad(self, model_param, local_grad) -> None:
        self._matrix_local_grad_by_param[model_param] = local_grad

    def _get_local_grad(self, model_param, shard_param):
        return get_local_grad(self, model_param, shard_param)

    def _release_rs_buffers(self):
        if hasattr(self, "buffers"):
            release_rs_buffers(self.buffers)

    def _log_grad_issue(self, kind: str, model_param, shard_param=None, **extra) -> None:
        payload = {
            "backend": self._require_matrix_backend().name,
            "kind": kind,
            "param": self._find_param_name(model_param)
            or getattr(model_param, "_param_name", f"id_{id(model_param)}"),
            "model_shape": tuple(model_param.shape),
            "shard_shape": tuple(shard_param.shape) if shard_param is not None else None,
        }
        payload.update(extra)
        logger.error("[MATRIX_GRAD_ISSUE] %s", payload)

    def _copy_model_grads_to_main_grads(self):
        """Map canonical model-side grads onto optimizer-side matrix shard grads."""
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

    def _restore_bucket_param_views_(self, *, matrix_only: bool = False) -> None:
        """Make selected model params alias canonical bucket.param_data before updates."""
        if not hasattr(self, "buffers"):
            return
        for buffer in self.buffers:
            for bucket in buffer.buckets:
                if getattr(bucket, "param_data", None) is None:
                    continue
                matrix_layout = getattr(bucket, "matrix_layout", None)
                if matrix_only:
                    matrix_params = collect_matrix_bucket_params(matrix_layout)
                    if not matrix_params:
                        continue
                    self._set_bucket_param_views(bucket, copy_data=True, params=matrix_params)
                else:
                    self._set_bucket_param_views(bucket, copy_data=True)
                    matrix_params = None
                if matrix_layout is None or not matrix_layout.has_params:
                    continue
                restore_matrix_shards_from_bucket_(
                    matrix_layout=matrix_layout,
                    get_full_view_2d=lambda entry: self._bucket_matrix_param_view(bucket, entry),
                    update_data_shard=lambda param, shard: update_data_shard(self, param, shard),
                    param_name=lambda param: self._param_name(param) or f"id_{id(param)}",
                )
                self._check_bucket_param_views(
                    bucket,
                    context="copy_main_params_to_model_params",
                    params=matrix_params,
                )

    def _bucket_param_data(self, model_param):
        gbuf_index, _, bucket_index = self.model_param_gbuf_map[model_param]
        return self.buffers[gbuf_index].buckets[bucket_index].param_data

    def _check_main_shards(self, main_shard_groups) -> None:
        del main_shard_groups

    def _mark_buckets_full_param_ready(self, ready: bool) -> None:
        self._set_bucket_runtime_flag("_matrix_full_param_ready", ready)

    def _copy_model_params_to_main_params(self, state_dict=None):
        from ..cpu_offloading import HybridDeviceOptimizer

        use_precision_aware_optimizer = bool(
            getattr(self.config, "use_precision_aware_optimizer_no_fp8_or_ds_fp8", False)
        )
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
            main_shard_groups=getattr(self, "shard_fp32_from_float16_groups", None),
            model_fp32_groups=self.model_fp32_groups,
            shard_fp32_groups=self.shard_fp32_groups,
            get_model_param_range_map=self._get_model_param_range_map,
            get_matrix_shard_layout=lambda param: param_shard_layout(self, param),
        )

    def _copy_main_params_to_model_params(self):
        use_precision_aware_optimizer = bool(
            getattr(self.config, "use_precision_aware_optimizer_no_fp8_or_ds_fp8", False)
        )
        if (
            not self.is_stub_optimizer
            and not self.ddp_config.use_megatron_fsdp
            and not use_precision_aware_optimizer
        ):
            quantize_param_shard(
                *self._get_fp8_params_and_shard_fp32_from_fp8(),
                self.data_parallel_group,
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
            main_shard_groups=getattr(self, "shard_fp32_from_float16_groups", None),
            shard_float16_groups=self.shard_float16_groups,
            model_fp32_groups=self.model_fp32_groups,
            shard_fp32_groups=self.shard_fp32_groups,
            get_data_shard=lambda param: get_data_shard(self, param),
            get_param_range_map=self._get_model_param_range_map,
            get_matrix_shard_layout=lambda param: param_shard_layout(self, param),
            get_bucket_param_data=self._bucket_param_data,
            mark_buckets_full_param_ready=self._mark_buckets_full_param_ready,
            check_main_shards=self._check_main_shards,
            restore_model_params_to_canonical_bucket_storage=self._restore_bucket_param_views_,
            empty_range_warning_count=getattr(self, "_empty_range_warning_count", 0),
        )

    def _matrix_grads_for_norm(self, matrix_params, count_matrix_grad: bool):
        return matrix_replica_grads(self, matrix_params, count_matrix_grad)

    def _resolve_matrix_tp_group(self):
        backend_name = self._require_matrix_backend().name
        resolver = getattr(self, f"_resolve_{backend_name}_tp_group", None)
        if resolver is not None:
            return resolver()
        return None

    def get_main_grads_for_grad_norm(self):
        return grad_norm_inputs(self)

    @torch.no_grad()
    def get_grad_norm(self):
        return compute_grad_norm(self)

    def clip_matrix_grad_norm(self, clip_grad: float, *, is_matrix_model_param) -> float:
        params = self.get_parameters()
        grad_norm = self.get_grad_norm() if params else 0.0
        if not params:
            return grad_norm

        matrix_model_params = []
        matrix_model_param_ids = set()
        matrix_shard_params = []
        standard_params = []
        for param in params:
            model_param = getattr(param, "_model_param", None)
            if model_param is not None and is_matrix_model_param(model_param):
                matrix_shard_params.append((model_param, param))
                if id(model_param) not in matrix_model_param_ids:
                    matrix_model_param_ids.add(id(model_param))
                    matrix_model_params.append(model_param)
                continue
            standard_params.append(param)

        if standard_params:
            clip_grad_by_total_norm_fp32(
                standard_params,
                clip_grad,
                grad_norm,
                self.config.use_precision_aware_optimizer_no_fp8_or_ds_fp8,
            )
        clip_coeff = float(clip_grad) / (float(grad_norm) + 1.0e-6)
        if matrix_model_params and clip_coeff < 1.0:
            self._scale_matrix_local_grads(matrix_model_params, clip_coeff)
            for model_param, shard_param in matrix_shard_params:
                local_grad = self._get_local_grad(model_param, shard_param)
                if self.config.use_precision_aware_optimizer_no_fp8_or_ds_fp8:
                    shard_param.decoupled_grad = local_grad
                    shard_param.grad = None
                else:
                    shard_param.decoupled_grad = None
                    shard_param.grad = (
                        local_grad if local_grad.dtype == torch.float32 else local_grad.float()
                    )
        return grad_norm
