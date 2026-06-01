"""MCore-native Muon optimizer."""

from __future__ import annotations

import logging
import math
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch.optim.optimizer import Optimizer

from megatron.core.optimizer_param_scheduler import ParamGroupOverride
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import MegatronModule
from megatron.core.utils import log_single_rank

from ..optimizer import MegatronOptimizer
from ..optimizer_config import OptimizerConfig, ParamKey
from .kernels import (
    logical_shape_for_tp,
    muon_scale_factor,
    nesterov_update,
    orthogonalize_muon,
    update_momentum,
)
from .state import init_matrix_state, init_scalar_state, is_muon_matrix_param

logger = logging.getLogger(__name__)


def _tp_group_for_param(
    param: torch.Tensor,
    pg_collection: Optional[ProcessGroupCollection],
):
    if pg_collection is None:
        return None
    if getattr(param, "expert_tp", False) and hasattr(pg_collection, "expt_tp"):
        return pg_collection.expt_tp
    return getattr(pg_collection, "tp", None)


def _split_grouped_rows(
    tensor: torch.Tensor,
    split_shapes: Tuple[int, ...],
) -> list[torch.Tensor]:
    rows = int(tensor.size(0))
    total = int(sum(split_shapes))
    if rows % total != 0:
        raise RuntimeError(
            "[MUON_SPLIT_ROWS_MISMATCH] "
            f"rows={rows} split_shapes={tuple(split_shapes)}"
        )
    groups = rows // total
    parts = torch.split(tensor.view(groups, total, -1), tuple(int(x) for x in split_shapes), dim=1)
    return [part.reshape(-1, tensor.size(-1)) for part in parts]


def _merge_grouped_rows(parts: list[torch.Tensor], split_shapes: Tuple[int, ...]) -> torch.Tensor:
    if not parts:
        raise RuntimeError("[MUON_EMPTY_SPLIT_MERGE]")
    cols = int(parts[0].size(-1))
    groups = int(parts[0].size(0)) // int(split_shapes[0])
    grouped = [
        part.reshape(groups, int(split_shapes[index]), cols)
        for index, part in enumerate(parts)
    ]
    return torch.cat(grouped, dim=1).reshape(-1, cols)


class MegatronMuon(torch.optim.AdamW):
    """Muon optimizer with MCore-native kernels and scalar fallback."""

    def __init__(
        self,
        params,
        lr: float = 3e-4,
        momentum_beta: float = 0.95,
        use_nesterov: bool = False,
        weight_decay: float = 0.01,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        split_qkv: bool = True,
        split_linear: bool = True,
        fp32_matmul_prec: str = "medium",
        coefficient_type: str = "quintic",
        num_ns_steps: int = 5,
        ns_backend: str = "standard",
        gram_restart_steps: tuple[int, ...] = (2,),
        gram_dtype: Optional[torch.dtype | str] = None,
        gram_kernel_policy: str = "torch",
        scale_mode: str = "spectral",
        extra_scale_factor: float = 1.0,
        fs_mode: str = "blockwise",
        tp_mode: str = "blockwise",
        pg_collection: Optional[ProcessGroupCollection] = None,
    ) -> None:
        if int(num_ns_steps) < 1:
            raise ValueError(f"num_ns_steps must be at least 1, got {num_ns_steps}")
        if fs_mode == "duplicated_debug":
            fs_mode = "duplicated"
        if tp_mode == "duplicated_debug":
            tp_mode = "duplicated"
        if fs_mode not in ("blockwise", "duplicated", "distributed"):
            raise ValueError(f"invalid Muon fs_mode: {fs_mode}")
        if tp_mode not in ("blockwise", "duplicated", "distributed"):
            raise ValueError(f"invalid Muon tp_mode: {tp_mode}")
        if ns_backend not in ("standard", "gram"):
            raise ValueError(f"invalid Muon ns_backend: {ns_backend}")

        defaults = dict(
            lr=lr,
            momentum_beta=momentum_beta,
            use_nesterov=use_nesterov,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
            split_qkv=bool(split_qkv),
            split_linear=bool(split_linear),
            fp32_matmul_prec=fp32_matmul_prec,
            coefficient_type=coefficient_type,
            num_ns_steps=int(num_ns_steps),
            ns_backend=ns_backend,
            gram_restart_steps=tuple(int(step) for step in gram_restart_steps),
            gram_dtype=gram_dtype,
            gram_kernel_policy=gram_kernel_policy,
            scale_mode=scale_mode,
            extra_scale_factor=extra_scale_factor,
            fs_mode=fs_mode,
            tp_mode=tp_mode,
        )
        Optimizer.__init__(self, params, defaults)
        self.pg_collection = pg_collection

    def _init_state(self, param: torch.Tensor, state: dict, group: dict) -> None:
        if is_muon_matrix_param(param):
            init_matrix_state(param, state)
            return
        init_scalar_state(param=param, state=state)

    def _matrix_child_updates(
        self,
        param: torch.Tensor,
        grad: torch.Tensor,
        momentum: torch.Tensor,
        group: dict,
    ) -> torch.Tensor:
        split_qkv = bool(group.get("split_qkv", True))
        split_linear = bool(group.get("split_linear", True))
        if split_qkv and getattr(param, "is_qkv", False):
            split_shapes = tuple(int(x) for x in getattr(param, "qkv_split_shapes"))
            grad_parts = _split_grouped_rows(grad, split_shapes)
            mom_parts = _split_grouped_rows(momentum, split_shapes)
            updates = [
                self._one_matrix_update(part, grad_parts[index], param, group)
                for index, part in enumerate(mom_parts)
            ]
            momentum.copy_(_merge_grouped_rows(mom_parts, split_shapes))
            return _merge_grouped_rows(updates, split_shapes)
        if split_qkv and getattr(param, "is_qkvg", False):
            split_shapes = tuple(int(x) for x in getattr(param, "qkvg_split_shapes"))
            grad_parts = _split_grouped_rows(grad, split_shapes)
            mom_parts = _split_grouped_rows(momentum, split_shapes)
            updates = [
                self._one_matrix_update(part, grad_parts[index], param, group)
                for index, part in enumerate(mom_parts)
            ]
            momentum.copy_(_merge_grouped_rows(mom_parts, split_shapes))
            return _merge_grouped_rows(updates, split_shapes)
        if split_linear and getattr(param, "is_linear_fc1", False):
            split_rows = tuple(int(x) for x in getattr(param, "linear_split_rows"))
            grad_parts = list(torch.split(grad, split_rows, dim=0))
            mom_parts = list(torch.split(momentum, split_rows, dim=0))
            updates = [
                self._one_matrix_update(part, grad_parts[index], param, group)
                for index, part in enumerate(mom_parts)
            ]
            return torch.cat(updates, dim=0)
        return self._one_matrix_update(momentum, grad, param, group)

    def _one_matrix_update(
        self,
        momentum: torch.Tensor,
        grad: torch.Tensor,
        parent_param: torch.Tensor,
        group: dict,
    ) -> torch.Tensor:
        beta = float(group.get("momentum_beta", self.defaults["momentum_beta"]))
        update_momentum(momentum, grad, beta=beta)
        update = (
            nesterov_update(momentum, grad, beta=beta)
            if bool(group.get("use_nesterov", self.defaults["use_nesterov"]))
            else momentum
        )

        tp_group = _tp_group_for_param(parent_param, self.pg_collection)
        partition_dim = getattr(parent_param, "partition_dim", None)
        if partition_dim == -1:
            partition_dim = None
        logical_shape = logical_shape_for_tp(
            update,
            partition_dim=partition_dim,
            tp_group=tp_group,
        )

        orth_update = orthogonalize_muon(
            update,
            ns_backend=group.get("ns_backend", self.defaults["ns_backend"]),
            steps=int(group.get("num_ns_steps", self.defaults["num_ns_steps"])),
            coefficient_type=group.get("coefficient_type", self.defaults["coefficient_type"]),
            tp_group=tp_group,
            partition_dim=partition_dim,
            tp_mode=group.get("tp_mode", self.defaults["tp_mode"]),
            logical_shape=logical_shape,
            gram_restart_steps=group.get(
                "gram_restart_steps", self.defaults["gram_restart_steps"]
            ),
            gram_dtype=group.get("gram_dtype", self.defaults["gram_dtype"]),
            gram_kernel_policy=group.get(
                "gram_kernel_policy", self.defaults["gram_kernel_policy"]
            ),
            fp32_matmul_prec=group.get(
                "fp32_matmul_prec", self.defaults["fp32_matmul_prec"]
            ),
        )
        m, n = logical_shape
        scale = muon_scale_factor(m, n, group.get("scale_mode", self.defaults["scale_mode"]))
        return orth_update * (scale * float(group.get("extra_scale_factor", 1.0)))

    def orthogonalize(self, param: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
        """Return the scaled Muon orthogonalized update for a parameter."""
        group = self.param_groups[0] if self.param_groups else self.defaults
        tp_group = _tp_group_for_param(param, self.pg_collection)
        partition_dim = getattr(param, "partition_dim", None)
        if partition_dim == -1:
            partition_dim = None
        logical_shape = logical_shape_for_tp(update, partition_dim=partition_dim, tp_group=tp_group)
        orth_update = orthogonalize_muon(
            update,
            ns_backend=group.get("ns_backend", self.defaults["ns_backend"]),
            steps=int(group.get("num_ns_steps", self.defaults["num_ns_steps"])),
            coefficient_type=group.get("coefficient_type", self.defaults["coefficient_type"]),
            tp_group=tp_group,
            partition_dim=partition_dim,
            tp_mode=group.get("tp_mode", self.defaults["tp_mode"]),
            logical_shape=logical_shape,
            gram_restart_steps=group.get(
                "gram_restart_steps", self.defaults["gram_restart_steps"]
            ),
            gram_dtype=group.get("gram_dtype", self.defaults["gram_dtype"]),
            gram_kernel_policy=group.get(
                "gram_kernel_policy", self.defaults["gram_kernel_policy"]
            ),
            fp32_matmul_prec=group.get(
                "fp32_matmul_prec", self.defaults["fp32_matmul_prec"]
            ),
        )
        m, n = logical_shape
        scale = muon_scale_factor(m, n, group.get("scale_mode", self.defaults["scale_mode"]))
        return orth_update * (scale * float(group.get("extra_scale_factor", 1.0)))

    def _step_matrix_param(self, param: torch.Tensor, grad: torch.Tensor, state: dict, group: dict):
        momentum = state["momentum_buffer"]
        update = self._matrix_child_updates(param, grad, momentum, group)
        lr = float(group.get("lr", self.defaults["lr"]))
        weight_decay = float(group.get("weight_decay", self.defaults["weight_decay"]))
        if weight_decay > 0.0:
            param.mul_(1.0 - lr * weight_decay)
        param.add_(update.to(param.dtype), alpha=-lr)

    def _step_scalar_param(self, param: torch.Tensor, grad: torch.Tensor, state: dict, group: dict):
        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]
        beta1, beta2 = group.get("betas", self.defaults["betas"])
        lr = float(group.get("lr", self.defaults["lr"]))
        eps = float(group.get("eps", self.defaults["eps"]))
        weight_decay = float(group.get("weight_decay", self.defaults["weight_decay"]))
        state["step"] = int(state.get("step", 0)) + 1
        grad_for_state = grad.to(exp_avg.dtype)
        exp_avg.lerp_(grad_for_state, 1.0 - float(beta1))
        exp_avg_sq.mul_(float(beta2)).addcmul_(grad_for_state, grad_for_state, value=1.0 - float(beta2))
        bias_correction1 = 1.0 - float(beta1) ** int(state["step"])
        bias_correction2 = 1.0 - float(beta2) ** int(state["step"])
        denom = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(eps)
        update = exp_avg / bias_correction1 / denom
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
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                state = self.state[param]
                self._init_state(param, state, group)
                if is_muon_matrix_param(param):
                    self._step_matrix_param(param, grad, state, group)
                else:
                    self._step_scalar_param(param, grad, state, group)
        return loss


TensorParallelMuon = MegatronMuon


def build_muon_optimizer(
    *,
    config: OptimizerConfig,
    param_groups,
    pg_collection: Optional[ProcessGroupCollection] = None,
    **kwargs,
):
    """Build the base Muon optimizer used by MCore wrappers."""
    del kwargs
    return MegatronMuon(
        param_groups,
        lr=config.lr,
        momentum_beta=config.muon_momentum,
        use_nesterov=config.muon_use_nesterov,
        weight_decay=config.weight_decay,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_eps,
        split_qkv=config.muon_split_qkv,
        split_linear=getattr(config, "muon_split_linear", True),
        fp32_matmul_prec=config.muon_fp32_matmul_prec,
        coefficient_type=getattr(config, "muon_coefficient_type", "quintic"),
        num_ns_steps=config.muon_num_ns_steps,
        ns_backend=getattr(config, "muon_ns_backend", "standard"),
        gram_restart_steps=tuple(
            getattr(
                config,
                "muon_gram_restart_steps",
                getattr(config, "muon_gram_ns_restart_iters", (2,)),
            )
        ),
        gram_dtype=getattr(config, "muon_gram_ns_dtype", None),
        gram_kernel_policy=getattr(config, "muon_gram_ns_kernel_policy", "torch"),
        scale_mode=config.muon_scale_mode,
        extra_scale_factor=config.muon_extra_scale_factor,
        fs_mode=getattr(config, "muon_fs_mode", "blockwise"),
        tp_mode=config.muon_tp_mode,
        pg_collection=pg_collection,
    )


def init_muon_state(opt, config=None):
    """Initialize Muon state for checkpointing wrappers."""
    del config
    for group in opt.param_groups:
        for param in group["params"]:
            state = opt.state[param]
            opt._init_state(param, state, group)


def get_megatron_muon_optimizer(
    config: OptimizerConfig,
    model_chunks: List[MegatronModule],
    config_overrides: Optional[Dict[ParamKey, ParamGroupOverride]] = None,
    use_gloo_process_groups: bool = True,
    layer_wise_distributed_optimizer: bool = False,
    pg_collection: Optional[ProcessGroupCollection] = None,
) -> MegatronOptimizer:
    """Compatibility entrypoint that routes to the MCore optimizer factory."""
    if layer_wise_distributed_optimizer or getattr(config, "optimizer", None) == "dist_muon":
        from ..muon_reference import get_megatron_muon_optimizer as get_reference

        return get_reference(
            config=config,
            model_chunks=model_chunks,
            config_overrides=config_overrides,
            use_gloo_process_groups=use_gloo_process_groups,
            layer_wise_distributed_optimizer=layer_wise_distributed_optimizer,
            pg_collection=pg_collection,
        )

    from .. import get_megatron_optimizer

    return get_megatron_optimizer(
        config=config,
        model_chunks=model_chunks,
        config_overrides=config_overrides,
        use_gloo_process_groups=use_gloo_process_groups,
        pg_collection=pg_collection,
    )
