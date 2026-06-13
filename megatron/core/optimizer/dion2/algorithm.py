"""MCore-native Dion2 optimizer."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

from megatron.core.optimizer_param_scheduler import ParamGroupOverride
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import MegatronModule

from ..matrix.splits.gdn import (
    extract_gdn_child,
    iter_gdn_child_kinds,
    resolve_gdn_split_axis,
    scatter_gdn_child_,
)
from ..matrix.splits.linear import (
    iter_linear_child_kinds,
    read_linear_child,
    resolve_linear_child_kinds,
    resolve_linear_split_axis,
    write_linear_child_,
)
from ..matrix.splits.qkv import (
    extract_qkv_child,
    iter_qkv_child_kinds,
    resolve_qkv_split_axis,
    scatter_qkv_child_,
)
from ..matrix.splits.qkvg import (
    extract_qkvg_child,
    iter_qkvg_child_kinds,
    resolve_qkvg_split_axis,
    scatter_qkvg_child_,
)
from ..muon.algorithm import MegatronMuon
from ..muon.kernels import logical_shape_for_tp
from ..optimizer import MegatronOptimizer
from ..optimizer_config import OptimizerConfig, ParamKey
from .kernels import dion2_compute_update
from .state import (
    Dion2MixedPrecisionConfig,
    init_matrix_state,
    init_scalar_state,
    is_dion2_matrix_param,
)
from .types import Dion2ParamConfig


def _tp_group_for_param(
    param: torch.Tensor,
    pg_collection: Optional[ProcessGroupCollection],
):
    if pg_collection is None:
        return None
    if getattr(param, "expert_tp", False) and hasattr(pg_collection, "expt_tp"):
        return pg_collection.expt_tp
    return getattr(pg_collection, "tp", None)


class MegatronDion2(MegatronMuon):
    """Dion2 optimizer with MCore-native kernels and scalar Adam fallback."""

    def __init__(
        self,
        params,
        lr: float = 3e-4,
        fraction: float = 0.25,
        ef_decay: float = 0.95,
        weight_decay: float = 0.01,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        adjust_lr: Optional[str] = "spectral_norm",
        scale_mode: str = "spectral",
        extra_scale_factor: float = 0.2,
        select_dim: str | int = "auto",
        selection_policy: str = "local_shard",
        split_parameters: bool = True,
        fp32_matmul_prec: str = "medium",
        coefficient_type: str = "dion2_polar_express",
        num_ns_steps: int = 5,
        ns_backend: str = "standard",
        gram_restart_steps: tuple[int, ...] = (2,),
        gram_dtype: Optional[torch.dtype | str] = None,
        gram_kernel_policy: str = "torch",
        fs_mode: str = "distributed",
        tp_mode: str = "distributed",
        scalar_optimizer: str = "adam",
        scalar_lr_scale: float = 1.0,
        mixed_precision_config: Optional[Dion2MixedPrecisionConfig] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ) -> None:
        super().__init__(
            params,
            lr=lr,
            momentum_beta=0.0,
            use_nesterov=False,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
            split_parameters=split_parameters,
            fp32_matmul_prec=fp32_matmul_prec,
            coefficient_type=coefficient_type,
            num_ns_steps=num_ns_steps,
            ns_backend=ns_backend,
            gram_restart_steps=gram_restart_steps,
            gram_dtype=gram_dtype,
            gram_kernel_policy=gram_kernel_policy,
            scale_mode=scale_mode,
            extra_scale_factor=extra_scale_factor,
            scalar_optimizer=scalar_optimizer,
            scalar_lr_scale=scalar_lr_scale,
            fs_mode=fs_mode,
            tp_mode=tp_mode,
            pg_collection=pg_collection,
        )
        self.defaults.update(
            algorithm="dion2",
            fraction=float(fraction),
            ef_decay=float(ef_decay),
            adjust_lr=adjust_lr,
            scale_mode=scale_mode,
            extra_scale_factor=float(extra_scale_factor),
            select_dim=select_dim,
            selection_policy=selection_policy,
        )
        for group in self.param_groups:
            group.setdefault("algorithm", "dion2")
            group.setdefault("fraction", float(fraction))
            group.setdefault("ef_decay", float(ef_decay))
            group.setdefault("adjust_lr", adjust_lr)
            group.setdefault("scale_mode", scale_mode)
            group.setdefault("extra_scale_factor", float(extra_scale_factor))
            group.setdefault("select_dim", select_dim)
            group.setdefault("selection_policy", selection_policy)
        self.mixed_precision_config = mixed_precision_config or Dion2MixedPrecisionConfig()

    def _init_state(self, param: torch.Tensor, state: dict, group: dict) -> None:
        if group.get("algorithm", "dion2") == "dion2" and is_dion2_matrix_param(param):
            init_matrix_state(param, state, self.mixed_precision_config)
            return
        init_scalar_state(
            param=param,
            state=state,
            mixed_precision_config=self.mixed_precision_config,
            scalar_optimizer=group.get("scalar_optimizer", self.defaults["scalar_optimizer"]),
        )

    def _param_config(self, parent_param: torch.Tensor, update: torch.Tensor, group: dict) -> Dion2ParamConfig:
        partition_dim = getattr(parent_param, "partition_dim", None)
        if partition_dim == -1:
            partition_dim = None
        tp_group = _tp_group_for_param(parent_param, self.pg_collection)
        logical_shape = logical_shape_for_tp(
            update,
            partition_dim=partition_dim,
            tp_group=tp_group,
        )
        return Dion2ParamConfig(
            fraction=float(group.get("fraction", self.defaults["fraction"])),
            ef_decay=float(group.get("ef_decay", self.defaults["ef_decay"])),
            adjust_lr=group.get("adjust_lr", self.defaults["adjust_lr"]),
            scale_mode=group.get("scale_mode", self.defaults["scale_mode"]),
            extra_scale_factor=float(
                group.get("extra_scale_factor", self.defaults["extra_scale_factor"])
            ),
            select_dim=group.get("select_dim", self.defaults["select_dim"]),
            selection_policy=group.get("selection_policy", self.defaults["selection_policy"]),
            ns_backend=group.get("ns_backend", self.defaults["ns_backend"]),
            coefficient_type=group.get("coefficient_type", self.defaults["coefficient_type"]),
            num_ns_steps=int(group.get("num_ns_steps", self.defaults["num_ns_steps"])),
            gram_restart_iterations=tuple(
                group.get("gram_restart_steps", self.defaults["gram_restart_steps"])
            ),
            gram_dtype=group.get("gram_dtype", self.defaults["gram_dtype"]),
            gram_kernel_policy=group.get(
                "gram_kernel_policy",
                self.defaults["gram_kernel_policy"],
            ),
            fp32_matmul_prec=group.get(
                "fp32_matmul_prec",
                self.defaults["fp32_matmul_prec"],
            ),
            tp_mode=group.get("tp_mode", self.defaults["tp_mode"]),
            fs_mode=group.get("fs_mode", self.defaults["fs_mode"]),
            split_parameters=bool(group.get("split_parameters", self.defaults["split_parameters"])),
        ), logical_shape

    def _one_matrix_update(
        self,
        momentum: torch.Tensor,
        grad: torch.Tensor,
        parent_param: torch.Tensor,
        group: dict,
    ) -> torch.Tensor:
        config, global_shape = self._param_config(parent_param, grad, group)
        return dion2_compute_update(
            grad=grad,
            momentum=momentum,
            config=config,
            global_shape=tuple(int(dim) for dim in global_shape),
        )

    def _matrix_child_updates(
        self,
        param: torch.Tensor,
        grad: torch.Tensor,
        momentum: torch.Tensor,
        group: dict,
    ) -> torch.Tensor:
        split_parameters = bool(group.get("split_parameters", True))
        if split_parameters and getattr(param, "is_qkv", False):
            split_shapes = tuple(int(x) for x in getattr(param, "qkv_split_shapes"))
            split_axis = resolve_qkv_split_axis(param=param)
            merged = torch.zeros_like(grad)
            for kind in iter_qkv_child_kinds():
                grad_child = extract_qkv_child(grad, split_shapes, kind, split_axis=split_axis)
                mom_child = extract_qkv_child(momentum, split_shapes, kind, split_axis=split_axis)
                update = self._one_matrix_update(mom_child, grad_child, param, group)
                scatter_qkv_child_(momentum, mom_child, split_shapes, kind, split_axis=split_axis)
                scatter_qkv_child_(merged, update, split_shapes, kind, split_axis=split_axis)
            return merged
        if split_parameters and getattr(param, "is_qkvg", False):
            split_shapes = tuple(int(x) for x in getattr(param, "qkvg_split_shapes"))
            split_axis = resolve_qkvg_split_axis(param=param)
            merged = torch.zeros_like(grad)
            for kind in iter_qkvg_child_kinds():
                grad_child = extract_qkvg_child(grad, split_shapes, kind, split_axis=split_axis)
                mom_child = extract_qkvg_child(momentum, split_shapes, kind, split_axis=split_axis)
                update = self._one_matrix_update(mom_child, grad_child, param, group)
                scatter_qkvg_child_(momentum, mom_child, split_shapes, kind, split_axis=split_axis)
                scatter_qkvg_child_(merged, update, split_shapes, kind, split_axis=split_axis)
            return merged
        if split_parameters and getattr(param, "is_gdn", False):
            split_shapes = tuple(int(x) for x in getattr(param, "gdn_split_shapes"))
            split_axis = resolve_gdn_split_axis(param=param)
            merged = torch.zeros_like(grad)
            for kind in iter_gdn_child_kinds():
                grad_child = extract_gdn_child(grad, split_shapes, kind, split_axis=split_axis)
                mom_child = extract_gdn_child(momentum, split_shapes, kind, split_axis=split_axis)
                update = self._one_matrix_update(mom_child, grad_child, param, group)
                scatter_gdn_child_(momentum, mom_child, split_shapes, kind, split_axis=split_axis)
                scatter_gdn_child_(merged, update, split_shapes, kind, split_axis=split_axis)
            return merged
        if split_parameters and getattr(param, "is_linear_split", False):
            split_rows = tuple(int(x) for x in getattr(param, "linear_split_rows"))
            split_axis = resolve_linear_split_axis(param=param)
            child_kinds = resolve_linear_child_kinds(param=param)
            merged = torch.zeros_like(grad)
            for kind in iter_linear_child_kinds(child_kinds):
                grad_child = read_linear_child(
                    grad,
                    split_rows,
                    None,
                    kind,
                    split_axis=split_axis,
                    child_kinds=child_kinds,
                )
                mom_child = read_linear_child(
                    momentum,
                    split_rows,
                    None,
                    kind,
                    split_axis=split_axis,
                    child_kinds=child_kinds,
                )
                update = self._one_matrix_update(mom_child, grad_child, param, group)
                write_linear_child_(
                    momentum,
                    mom_child,
                    split_rows,
                    None,
                    kind,
                    split_axis=split_axis,
                    child_kinds=child_kinds,
                )
                write_linear_child_(
                    merged,
                    update,
                    split_rows,
                    None,
                    kind,
                    split_axis=split_axis,
                    child_kinds=child_kinds,
                )
            return merged
        return self._one_matrix_update(momentum, grad, param, group)

    def _step_matrix_param(self, param: torch.Tensor, grad: torch.Tensor, state: dict, group: dict):
        momentum = state["momentum"]
        update = self._matrix_child_updates(param, grad, momentum, group)
        lr = float(group.get("lr", self.defaults["lr"]))
        weight_decay = float(group.get("weight_decay", self.defaults["weight_decay"]))
        if weight_decay > 0.0:
            param.mul_(1.0 - lr * weight_decay)
        param.add_(update.to(param.dtype), alpha=-lr)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            scalar_items = []
            for param in group["params"]:
                grad = param.grad
                if grad is None:
                    grad = getattr(param, "decoupled_grad", None)
                if grad is None:
                    continue
                state = self.state[param]
                self._init_state(param, state, group)
                if group.get("algorithm", "dion2") == "dion2" and is_dion2_matrix_param(param):
                    self._step_matrix_param(param, grad, state, group)
                else:
                    scalar_items.append((param, grad, state))
            self._step_scalar_batch(scalar_items, group)
        return loss


TensorParallelDion2 = MegatronDion2


def build_dion2_optimizer(
    *,
    config: OptimizerConfig,
    param_groups,
    pg_collection: Optional[ProcessGroupCollection] = None,
    **kwargs,
):
    """Build the base Dion2 optimizer used by MCore wrappers."""
    del kwargs
    mixed_precision_config = Dion2MixedPrecisionConfig(
        momentum_dtype=getattr(config, "dion2_momentum_dtype", None),
        scalar_momentum_dtype=getattr(config, "dion2_scalar_momentum_dtype", None),
        scalar_variance_dtype=getattr(config, "dion2_scalar_variance_dtype", None),
    )
    return MegatronDion2(
        param_groups,
        lr=config.lr,
        fraction=config.dion2_fraction,
        ef_decay=config.dion2_ef_decay,
        weight_decay=config.weight_decay,
        betas=(config.dion2_beta1, config.dion2_beta2),
        eps=config.dion2_scalar_eps,
        adjust_lr=config.dion2_adjust_lr,
        scale_mode=config.dion2_scale_mode,
        extra_scale_factor=config.dion2_extra_scale_factor,
        select_dim=config.dion2_select_dim,
        selection_policy=config.dion2_selection_policy,
        split_parameters=(
            False
            if bool(getattr(config, "use_distributed_optimizer", False))
            else config.dion2_split_parameters
        ),
        fp32_matmul_prec=config.dion2_fp32_matmul_prec,
        coefficient_type=config.dion2_coefficient_type,
        num_ns_steps=config.dion2_num_ns_steps,
        ns_backend=config.dion2_ns_backend,
        gram_restart_steps=tuple(config.dion2_gram_ns_restart_iters),
        gram_dtype=config.dion2_gram_ns_dtype,
        gram_kernel_policy=config.dion2_gram_ns_kernel_policy,
        fs_mode=config.dion2_fs_mode,
        tp_mode=config.dion2_tp_mode,
        scalar_optimizer=config.dion2_scalar_optimizer,
        scalar_lr_scale=config.dion2_scalar_lr_scale,
        mixed_precision_config=mixed_precision_config,
        pg_collection=pg_collection,
    )


def init_dion2_state(opt, config=None):
    """Initialize Dion2 state for checkpointing wrappers."""
    del config
    for group in opt.param_groups:
        for param in group["params"]:
            state = opt.state[param]
            opt._init_state(param, state, group)


def get_megatron_dion2_optimizer(
    config: OptimizerConfig,
    model_chunks: List[MegatronModule],
    config_overrides: Optional[Dict[ParamKey, ParamGroupOverride]] = None,
    use_gloo_process_groups: bool = True,
    layer_wise_distributed_optimizer: bool = False,
    pg_collection: Optional[ProcessGroupCollection] = None,
) -> MegatronOptimizer:
    """Compatibility entrypoint that routes to the MCore optimizer factory."""
    del layer_wise_distributed_optimizer
    from .. import get_megatron_optimizer

    return get_megatron_optimizer(
        config=config,
        model_chunks=model_chunks,
        config_overrides=config_overrides,
        use_gloo_process_groups=use_gloo_process_groups,
        pg_collection=pg_collection,
    )


__all__ = [
    "MegatronDion2",
    "TensorParallelDion2",
    "build_dion2_optimizer",
    "get_megatron_dion2_optimizer",
    "init_dion2_state",
]
