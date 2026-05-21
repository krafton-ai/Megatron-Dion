"""Generic base class for matrix-aware distributed optimizers."""

from typing import Optional

import torch
import torch.distributed as dist

from ... import parallel_state
from ...transformer.fsdp_dtensor_checkpoint import get_global_unique_param_name
from .backend import MatrixBackend
from ..distrib_optimizer import DistributedOptimizer


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
