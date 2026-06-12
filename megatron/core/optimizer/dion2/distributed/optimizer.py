"""Distributed optimizer wrapper for MCore-native Dion2."""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import torch

from ...matrix.checkpoint_io import (
    build_distributed_checkpoint_state,
    build_matrix_checkpoint_metadata,
    resolve_matrix_checkpoint_sharding_type,
    restore_distributed_checkpoint_state,
    split_distributed_checkpoint_state,
    validate_matrix_checkpoint_metadata,
)
from ...muon.distributed.optimizer import DistributedMuonOptimizer
from ..backend import Dion2Backend
from ..kernels import (
    dion2_prepare_selected,
    dion2_scatter_update,
    orthogonalize_selected,
)
from ..state import build_param_config, is_dion2_matrix_param
from ..types import Dion2Batch, Dion2ParamConfig
from .batches import build_dion2_batches


class DistributedDion2Optimizer(DistributedMuonOptimizer):
    """MCore distributed optimizer adapter for Dion2 matrix updates.

    The wrapper deliberately reuses Muon's Matrix shard ownership, bucket layout,
    overlap, and optimizer-only split machinery. Dion2 replaces only the matrix
    update rule and the per-parameter config/state semantics.
    """

    def __init__(self, *args, **kwargs):
        dion2_fs_group = kwargs.pop("dion2_fs_group", None)
        dion2_tp_group = kwargs.pop("dion2_tp_group", None)
        dion2_fs_mode = kwargs.pop("dion2_fs_mode", "distributed")
        dion2_tp_mode = kwargs.pop("dion2_tp_mode", "distributed")
        dion2_ns_backend = kwargs.pop("dion2_ns_backend", "standard")
        kwargs["muon_fs_group"] = dion2_fs_group
        kwargs["muon_tp_group"] = dion2_tp_group
        kwargs["muon_fs_mode"] = dion2_fs_mode
        kwargs["muon_tp_mode"] = dion2_tp_mode
        kwargs["muon_ns_backend"] = dion2_ns_backend
        kwargs["is_expert_muon"] = kwargs.pop("is_expert_dion2", False)
        super().__init__(*args, **kwargs)
        self.set_matrix_backend(Dion2Backend())
        self._dion2_group_rank_cache = self._muon_group_rank_cache
        self._fs_mode = dion2_fs_mode
        self._tp_mode = dion2_tp_mode
        self._ns_backend = dion2_ns_backend
        self._mixed_precision_config = getattr(self.optimizer, "mixed_precision_config", None)

    @property
    def dion2_fs_group(self):
        return self.fs_group

    @property
    def dion2_tp_group(self):
        return self.muon_tp_group

    def _split_for_group(self, group) -> bool:
        default = bool(getattr(self.config, "dion2_split_parameters", True))
        if group is None:
            return default
        return bool(group.get("dion2_split_parameters", group.get("split_parameters", default)))

    def _build_param_config(self, param, dist_meta, optim_group) -> Dion2ParamConfig:
        cfg = build_param_config(
            param_ndim=2 if bool(getattr(dist_meta, "is_muon_param", False)) else param.ndim,
            local_shape=getattr(dist_meta, "shape", None),
            dist_meta=dist_meta,
            tp_world_size=int(getattr(dist_meta, "tp_world_size", 1)),
            tp_active=int(getattr(dist_meta, "tp_world_size", 1)) > 1,
            fraction=float(getattr(self.config, "dion2_fraction", 0.25)),
            ef_decay=float(getattr(self.config, "dion2_ef_decay", 0.95)),
            adjust_lr=getattr(self.config, "dion2_adjust_lr", "spectral_norm"),
            select_dim=getattr(self.config, "dion2_select_dim", "auto"),
            selection_policy=getattr(self.config, "dion2_selection_policy", "local_shard"),
            ns_backend=getattr(self.config, "dion2_ns_backend", self._ns_backend),
            coefficient_type=getattr(self.config, "dion2_coefficient_type", "polar_express"),
            num_ns_steps=int(getattr(self.config, "dion2_num_ns_steps", 5)),
            ns_epsilon=float(getattr(self.config, "dion2_ns_epsilon", 1e-7)),
            gram_restart_iterations=tuple(
                getattr(self.config, "dion2_gram_ns_restart_iters", (2,))
            ),
            gram_kernel_policy=getattr(self.config, "dion2_gram_ns_kernel_policy", "torch"),
            gram_dtype=getattr(self.config, "dion2_gram_ns_dtype", None),
            fp32_matmul_prec=getattr(self.config, "dion2_fp32_matmul_prec", "medium"),
            fs_mode=getattr(self.config, "dion2_fs_mode", self._fs_mode),
            tp_mode=getattr(self.config, "dion2_tp_mode", self._tp_mode),
            split_parameters=self._split_for_group(optim_group),
        )
        return cfg

    def _child_param_config(self, child_meta, child_shape, optim_group) -> Dion2ParamConfig:
        parent_config = getattr(child_meta, "param_config", None)
        del parent_config
        child_meta = replace(
            child_meta,
            is_matrix_param=True,
        )
        return build_param_config(
            param_ndim=2,
            local_shape=tuple(int(dim) for dim in child_shape),
            dist_meta=child_meta,
            tp_world_size=int(getattr(child_meta, "tp_world_size", 1)),
            tp_active=int(getattr(child_meta, "tp_world_size", 1)) > 1,
            fraction=float(getattr(self.config, "dion2_fraction", 0.25)),
            ef_decay=float(getattr(self.config, "dion2_ef_decay", 0.95)),
            adjust_lr=getattr(self.config, "dion2_adjust_lr", "spectral_norm"),
            select_dim=getattr(self.config, "dion2_select_dim", "auto"),
            selection_policy=getattr(self.config, "dion2_selection_policy", "local_shard"),
            ns_backend=getattr(self.config, "dion2_ns_backend", self._ns_backend),
            coefficient_type=getattr(self.config, "dion2_coefficient_type", "polar_express"),
            num_ns_steps=int(getattr(self.config, "dion2_num_ns_steps", 5)),
            ns_epsilon=float(getattr(self.config, "dion2_ns_epsilon", 1e-7)),
            gram_restart_iterations=tuple(
                getattr(self.config, "dion2_gram_ns_restart_iters", (2,))
            ),
            gram_kernel_policy=getattr(self.config, "dion2_gram_ns_kernel_policy", "torch"),
            gram_dtype=getattr(self.config, "dion2_gram_ns_dtype", None),
            fp32_matmul_prec=getattr(self.config, "dion2_fp32_matmul_prec", "medium"),
            fs_mode=getattr(self.config, "dion2_fs_mode", self._fs_mode),
            tp_mode=getattr(self.config, "dion2_tp_mode", self._tp_mode),
            split_parameters=self._split_for_group(optim_group),
        )

    def _refresh_dion2_step_metadata(self, *, param, optimizer_state, optim_group, dist_meta):
        del optimizer_state
        if dist_meta is None:
            return
        signature = (
            tuple(int(dim) for dim in getattr(dist_meta, "shape", ()) or ()),
            tuple(int(dim) for dim in getattr(dist_meta, "global_shape", ()) or ()),
            int(getattr(dist_meta, "tp_world_size", 1)),
            int(getattr(dist_meta, "fs_world_size", 1)),
            int(getattr(dist_meta, "tp_shard_dim", -1)),
            int(getattr(dist_meta, "fs_shard_dim", -1)),
            float(getattr(self.config, "dion2_fraction", 0.25)),
            float(getattr(self.config, "dion2_ef_decay", 0.95)),
            getattr(self.config, "dion2_adjust_lr", "spectral_norm"),
            getattr(self.config, "dion2_select_dim", "auto"),
            getattr(self.config, "dion2_selection_policy", "local_shard"),
            getattr(self.config, "dion2_ns_backend", self._ns_backend),
            getattr(self.config, "dion2_coefficient_type", "polar_express"),
            int(getattr(self.config, "dion2_num_ns_steps", 5)),
            float(getattr(self.config, "dion2_ns_epsilon", 1e-7)),
            tuple(getattr(self.config, "dion2_gram_ns_restart_iters", (2,))),
            getattr(self.config, "dion2_gram_ns_kernel_policy", "torch"),
            getattr(self.config, "dion2_gram_ns_dtype", None),
            getattr(self.config, "dion2_fp32_matmul_prec", "medium"),
            getattr(self.config, "dion2_fs_mode", self._fs_mode),
            getattr(self.config, "dion2_tp_mode", self._tp_mode),
            self._split_for_group(optim_group),
        )
        if (
            getattr(dist_meta, "_dion2_param_config_signature", None) == signature
            and getattr(dist_meta, "param_config", None) is not None
        ):
            return
        dist_meta.param_config = self._build_param_config(param, dist_meta, optim_group)
        dist_meta._dion2_param_config_signature = signature

    def _ensure_optimizer_state(self, param, optim_group):
        state = self.optimizer.state[param]
        meta = self.dist_metas.get(param)
        if (
            meta is None
            or not bool(getattr(meta, "is_muon_param", False))
            or optim_group.get("algorithm", "dion2") != "dion2"
        ):
            return state
        shape = tuple(int(dim) for dim in meta.shape)
        momentum = state.get("momentum")
        if momentum is None:
            momentum = state.get("momentum_buffer")
        if momentum is None or tuple(momentum.shape) != shape:
            momentum = torch.zeros(shape, dtype=param.dtype, device=param.device)
            state["momentum"] = momentum
            state["momentum_buffer"] = momentum
        else:
            state["momentum"] = momentum
            state["momentum_buffer"] = momentum
        state["local_shape"] = shape
        state["global_shape"] = tuple(int(dim) for dim in meta.global_shape)
        return state

    def _should_use_distributed_dion2_update(self, param, state, optim_group, dist_meta) -> bool:
        del param, state
        return bool(
            optim_group.get("algorithm", "dion2") == "dion2"
            and dist_meta is not None
            and getattr(dist_meta, "is_muon_param", False)
        )

    def _expand_split_dion2_params(self, *, param, grad, optimizer_state, optim_group, config, dist_meta):
        if optim_group.get("algorithm", "dion2") != "dion2":
            return None
        if not bool(getattr(dist_meta, "is_muon_param", False)):
            return None
        fake_group = dict(optim_group)
        fake_group["algorithm"] = "muon"
        children = super()._expand_split_muon_params(
            param=param,
            grad=grad,
            optimizer_state=optimizer_state,
            optim_group=fake_group,
            config=config,
            dist_meta=dist_meta,
        )
        if children is None:
            return None
        for child in children:
            child.optim_group = optim_group
        return children

    def _sync_dion2_state(self, matrix_params) -> None:
        del matrix_params

    def _build_dion2_batches(self, matrix_params):
        return build_dion2_batches(
            matrix_params,
            fs_mode=getattr(self, "_fs_mode", "distributed"),
            tp_mode=getattr(self, "_tp_mode", "distributed"),
            ns_backend=getattr(self, "_ns_backend", "standard"),
            rank_cache=self._dion2_group_rank_cache,
        )

    def _apply_dion2_batches(self, batches: Sequence[Dion2Batch]):
        for batch in batches:
            prepared = []
            groups = {}
            for entry in batch.entries[: batch.real_batch_size]:
                if entry.param is None or entry.grad is None:
                    continue
                param = entry.param if entry.param.ndim == 2 else entry.param.view(entry.param_shape)
                grad = entry.grad if entry.grad.ndim == 2 else entry.grad.view(entry.param_shape)
                momentum = entry.optimizer_state.get("momentum")
                if momentum is None:
                    momentum = entry.optimizer_state.get("momentum_buffer")
                    entry.optimizer_state["momentum"] = momentum
                if momentum is None:
                    momentum = torch.zeros_like(param)
                    entry.optimizer_state["momentum"] = momentum
                    entry.optimizer_state["momentum_buffer"] = momentum
                momentum = momentum if momentum.ndim == 2 else momentum.view(entry.param_shape)
                lr = float(
                    (entry.optim_group or {}).get(
                        "lr",
                        getattr(getattr(self, "config", None), "lr", 0.0),
                    )
                )
                weight_decay = float(
                    (entry.optim_group or {}).get(
                        "weight_decay",
                        getattr(getattr(self, "config", None), "weight_decay", 0.0),
                    )
                )
                if weight_decay != 0.0:
                    param.mul_(1.0 - lr * weight_decay)
                canonical, indices, select_dim, layout, lr_scale = dion2_prepare_selected(
                    grad=grad,
                    momentum=momentum,
                    config=entry.config,
                    global_shape=tuple(int(dim) for dim in entry.global_shape),
                    dist_meta=entry.dist_meta,
                )
                item_index = len(prepared)
                prepared.append(
                    (
                        entry,
                        param,
                        momentum,
                        canonical,
                        indices,
                        select_dim,
                        layout,
                        lr_scale,
                        lr,
                    )
                )
                if int(indices.numel()) == 0:
                    continue
                cfg = entry.config
                key = (
                    tuple(int(dim) for dim in canonical.shape),
                    str(canonical.dtype),
                    str(canonical.device),
                    int(select_dim),
                    tuple(int(dim) for dim in layout.logical_shape),
                    id(layout.row_group),
                    int(layout.row_partition_dim),
                    layout.row_mode,
                    layout.row_partition_sizes,
                    id(layout.col_group),
                    int(layout.col_partition_dim),
                    layout.col_mode,
                    cfg.ns_backend,
                    cfg.coefficient_type,
                    int(cfg.num_ns_steps),
                    float(cfg.ns_epsilon),
                    tuple(int(step) for step in cfg.gram_restart_iterations),
                    str(cfg.gram_kernel_policy),
                    str(cfg.gram_dtype),
                    str(cfg.fp32_matmul_prec),
                )
                groups.setdefault(key, []).append(item_index)

            orthogonalized = [None] * len(prepared)
            for indices_in_group in groups.values():
                first = prepared[indices_in_group[0]]
                first_entry = first[0]
                first_layout = first[6]
                if len(indices_in_group) == 1:
                    idx = indices_in_group[0]
                    orthogonalized[idx] = orthogonalize_selected(
                        prepared[idx][3],
                        layout=prepared[idx][6],
                        config=prepared[idx][0].config,
                    )
                    continue
                stacked = torch.stack([prepared[idx][3] for idx in indices_in_group], dim=0)
                stacked_orth = orthogonalize_selected(
                    stacked,
                    layout=first_layout,
                    config=first_entry.config,
                )
                for batch_index, item_index in enumerate(indices_in_group):
                    orthogonalized[item_index] = stacked_orth[batch_index].contiguous()

            for item_index, item in enumerate(prepared):
                entry, param, momentum, _canonical, selected_indices, select_dim, _layout, lr_scale, lr = item
                orth = orthogonalized[item_index]
                if orth is not None:
                    update = dion2_scatter_update(
                        target_shape=tuple(int(dim) for dim in param.shape),
                        orthogonalized_canonical=orth,
                        indices=selected_indices,
                        select_dim=select_dim,
                        lr_scale=lr_scale,
                        dtype=param.dtype,
                    )
                    param.add_(update, alpha=-lr)
                if entry.commit_update is not None:
                    entry.commit_update(param, momentum)

    @torch.no_grad()
    def step_with_ready_grads(self) -> bool:
        batches, _ = self._route_step_params()
        saved_grads = {}
        for param in self._matrix_param_tuple():
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
        self._apply_dion2_batches(batches)
        self._copy_main_params_to_model_params()
        if self.ddp_config.use_megatron_fsdp or not self.ddp_config.overlap_param_gather:
            for model_chunk in self.model_chunks:
                model_chunk.start_param_sync()
        return True

    def clip_grad_norm(self, clip_grad: float) -> float:
        return self.clip_matrix_grad_norm(
            clip_grad,
            is_matrix_model_param=lambda param: is_dion2_matrix_param(param)
            or getattr(param, "is_muon_param", False),
        )

    def _dion2_param_key(self, param):
        meta = self.dist_metas.get(param)
        return getattr(meta, "param_uid", None) or getattr(param, "_matrix_param_uid", None)

    def _dion2_checkpoint_topology_signature(self) -> dict:
        return self._matrix_checkpoint_topology_signature(
            fs_group=getattr(self, "fs_group", None),
            tp_group=self._resolve_muon_tp_group(),
            rp_group=getattr(self, "rp_group", None),
            state_replica_group=getattr(self, "state_replica_group", None),
        )

    def _ensure_dion2_checkpoint_state(self) -> None:
        for group in self.optimizer.param_groups:
            for param in group.get("params", ()):
                meta = self.dist_metas.get(param)
                if meta is None or not bool(getattr(meta, "is_muon_param", False)):
                    continue
                if group.get("algorithm", "dion2") != "dion2":
                    continue
                state = self.optimizer.state[param]
                shape = tuple(int(dim) for dim in meta.shape)
                momentum = state.get("momentum")
                if momentum is None:
                    momentum = state.get("momentum_buffer")
                if momentum is None or tuple(momentum.shape) != shape:
                    momentum = torch.zeros(shape, dtype=param.dtype, device=param.device)
                state["momentum"] = momentum
                state["momentum_buffer"] = momentum
                state["local_shape"] = shape
                state["global_shape"] = tuple(
                    int(dim)
                    for dim in (
                        getattr(meta, "per_expert_global_shape", None) or meta.global_shape
                    )
                )

    def sharded_state_dict(
        self,
        model_sharded_state_dict=None,
        is_loading=False,
        sharding_type=None,
        metadata=None,
    ):
        """Build torch_dist-compatible Dion2 optimizer checkpoint state."""
        from ....dist_checkpointing.mapping import ShardedObject, ShardedTensor

        del model_sharded_state_dict
        if is_loading:
            self._ensure_dion2_checkpoint_state()
        self._ensure_dion2_checkpoint_state()
        dp_rank = self.data_parallel_group.rank()
        dp_size = self.data_parallel_group.size()
        base_key = f"optimizer.distributed.dp_group_idx_{self.data_parallel_group_idx}"
        common_replica_id = (self.distributed_optimizer_instance_id, 0, dp_rank)
        state_type = resolve_matrix_checkpoint_sharding_type(sharding_type, metadata)
        tp_group = self._resolve_muon_tp_group()
        tp_size = 1 if tp_group is None else self._group_size(tp_group)
        state_replica_group = getattr(self, "state_replica_group", None)
        state_replica_size = 1 if state_replica_group is None else self._group_size(state_replica_group)
        state_replica_rank = 0 if state_replica_group is None else self._group_rank(state_replica_group)
        rp_group = getattr(self, "rp_group", None)
        rp_size = 1 if rp_group is None else self._group_size(rp_group)
        checkpoint_metadata = build_matrix_checkpoint_metadata(
            dp_size=dp_size,
            fs_size=int(getattr(self, "fs_size", 1)),
            tp_size=int(tp_size),
            rp_size=int(rp_size),
            state_replica_size=int(state_replica_size),
            requested_type=state_type,
            topology_signature=self._dion2_checkpoint_topology_signature(),
            backend_state_spec=self._require_matrix_backend().state_spec(),
        )
        return build_distributed_checkpoint_state(
            common_state=self.state_dict(),
            param_groups=self.optimizer.param_groups,
            optimizer_state=self.optimizer.state,
            get_param_key=self._dion2_param_key,
            base_key=base_key,
            common_replica_id=common_replica_id,
            state_global_shape=(dp_size,),
            state_global_offset=(dp_rank,),
            state_replica_id=(state_replica_rank,),
            checkpoint_metadata=checkpoint_metadata,
            sharded_object_cls=ShardedObject,
            sharded_tensor_cls=ShardedTensor,
            state_rank_key=str(dp_rank),
        )

    def load_state_dict(self, state_dict):
        if "matrix_checkpoint_metadata" not in state_dict:
            return super().load_state_dict(state_dict)
        metadata, param_state, common = split_distributed_checkpoint_state(state_dict)
        tp_group = self._resolve_muon_tp_group()
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
            topology_signature=self._dion2_checkpoint_topology_signature(),
            backend_state_spec=self._require_matrix_backend().state_spec(),
        )
        result = self._load_matrix_common_state_dict(common, label="Dion2")
        restore_distributed_checkpoint_state(
            param_state_data=param_state,
            param_groups=self.optimizer.param_groups,
            optimizer_state=self.optimizer.state,
            get_param_key=self._dion2_param_key,
            mixed_precision_config=self._mixed_precision_config,
        )
        self._ensure_dion2_checkpoint_state()
        return result


__all__ = ["DistributedDion2Optimizer"]
