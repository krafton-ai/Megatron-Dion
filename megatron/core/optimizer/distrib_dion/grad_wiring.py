"""Grad wiring helpers for Dion distributed optimizer.

This module extracts small, self-contained helpers from the Dion distributed
optimizer wrapper. The intent is mechanical refactor only: keep numerics and
collective patterns unchanged.
"""

from __future__ import annotations

from typing import Callable

import torch

from .fs_layout import slice_fs_shard_2d


def validate_dion_local_shard_grad_(
    *,
    model_param: torch.nn.Parameter,
    shard_param: torch.nn.Parameter,
    shard_view: torch.Tensor | None,
    log_grad_issue_fn: Callable,
) -> torch.Tensor:
    """Validate one adapter-published Dion local grad shard before optimizer bind."""
    if shard_view is None:
        log_grad_issue_fn("DION_LOCAL_GRAD_NONE", model_param, shard_param)
        raise RuntimeError(
            "[Dion] Dion grad bind requires published local grad shard "
            f"param_shape={tuple(model_param.shape)} shard_shape={tuple(shard_param.shape)}"
        )

    if shard_view.ndim != shard_param.ndim:
        raise RuntimeError(
            "[Dion] Dion canonical shard grad ndim mismatch "
            f"shard_view_ndim={int(shard_view.ndim)} shard_param_ndim={int(shard_param.ndim)}"
        )

    if shard_view.nelement() != shard_param.nelement():
        log_grad_issue_fn(
            "DION_MAIN_GRAD_NUMEL_MISMATCH",
            model_param,
            shard_param,
            shard_view_numel=int(shard_view.nelement()),
            shard_param_numel=int(shard_param.nelement()),
        )
        raise RuntimeError(
            "[Dion] Dion canonical shard grad numel mismatch "
            f"shard_view_numel={int(shard_view.nelement())} shard_param_numel={int(shard_param.nelement())}"
        )
    if tuple(shard_view.shape) != tuple(shard_param.shape):
        log_grad_issue_fn(
            "DION_MAIN_GRAD_SHAPE_MISMATCH",
            model_param,
            shard_param,
            shard_view_shape=tuple(shard_view.shape),
            shard_param_shape=tuple(shard_param.shape),
        )
        raise RuntimeError(
            "[Dion] Dion canonical shard grad shape mismatch "
            f"shard_view_shape={tuple(shard_view.shape)} shard_param_shape={tuple(shard_param.shape)}"
        )
    return shard_view

def slice_non_dion_shard_grad_(
    *,
    model_param: torch.nn.Parameter,
    shard_main_param: torch.nn.Parameter,
    param_range,
    log_grad_issue_fn: Callable,
) -> torch.Tensor:
    """Slice one non-Dion optimizer shard grad from canonical `model_param.main_grad`."""
    model_grad = model_param.main_grad
    if model_grad is None:
        log_grad_issue_fn("NON_DION_MODEL_MAIN_GRAD_NONE", model_param, shard_main_param)
        raise RuntimeError(
            "[Dion] non-Dion standard shard grad requires canonical model_param.main_grad "
            f"param_shape={tuple(model_param.shape)} shard_shape={tuple(shard_main_param.shape)}"
        )

    flat_grad = model_grad.view(-1)
    start = int(param_range.start)
    end = int(param_range.end)
    if end > flat_grad.numel():
        log_grad_issue_fn(
            "NON_DION_STOCK_SLICE_OOB",
            model_param,
            shard_main_param,
            grad_numel=int(flat_grad.numel()),
            start=int(start),
            end=int(end),
        )
        raise RuntimeError(
            "[Dion] non-Dion standard shard grad slice exceeded canonical main_grad "
            f"grad_numel={int(flat_grad.numel())} start={int(start)} end={int(end)}"
        )
    shard_view = flat_grad[start:end]
    if shard_view.numel() != shard_main_param.nelement():
        log_grad_issue_fn(
            "NON_DION_STOCK_SLICE_SIZE_MISMATCH",
            model_param,
            shard_main_param,
            expected_numel=int(shard_main_param.nelement()),
            got_numel=int(shard_view.numel()),
            start=int(start),
            end=int(end),
        )
        raise RuntimeError(
            "[Dion] non-Dion standard shard grad size mismatch "
            f"expected={int(shard_main_param.nelement())} got={int(shard_view.numel())}"
        )
    return shard_view.view(shard_main_param.shape)


def bind_dion_shard_grad_(
    *,
    model_param: torch.nn.Parameter,
    shard_param: torch.nn.Parameter,
    get_dion_local_shard_grad_fn: Callable[[torch.nn.Parameter, torch.nn.Parameter], torch.Tensor],
    log_grad_issue_fn: Callable,
    use_precision_aware_optimizer: bool,
) -> None:
    """Bind one Dion grad shard onto an optimizer shard param.

    Dion optimizer shards keep their Dion-local 2D layout, and their optimizer-side
    grad source is the adapter-published Dion-local grad surface.
    """
    canonical_shard_grad = validate_dion_local_shard_grad_(
        model_param=model_param,
        shard_param=shard_param,
        shard_view=get_dion_local_shard_grad_fn(model_param, shard_param),
        log_grad_issue_fn=log_grad_issue_fn,
    )

    shard_param.is_dion_param = True
    shard_param.main_grad = None
    if use_precision_aware_optimizer:
        shard_param.decoupled_grad = canonical_shard_grad
        shard_param.grad = None
    else:
        shard_param.decoupled_grad = None
        shard_param.grad = canonical_shard_grad.float()


def bind_non_dion_shard_grad_(
    *,
    model_param: torch.nn.Parameter,
    shard_param: torch.nn.Parameter,
    get_param_range_fn: Callable[[torch.nn.Parameter], object],
    log_grad_issue_fn: Callable,
    use_precision_aware_optimizer: bool,
) -> None:
    """Bind one non-Dion grad shard onto an optimizer shard param."""
    param_range = get_param_range_fn(model_param)["param"]

    model_grad = getattr(model_param, "main_grad", None)
    if model_grad is None:
        raise RuntimeError(
            "[Dion] non-Dion optimizer grad bind requires canonical model_param.main_grad "
            f"param={getattr(model_param, '_param_name', f'id_{id(model_param)}')} "
            f"shard_shape={tuple(shard_param.shape)}"
        )

    shard_grad = slice_non_dion_shard_grad_(
        model_param=model_param,
        shard_main_param=shard_param,
        param_range=param_range,
        log_grad_issue_fn=log_grad_issue_fn,
    )

    if use_precision_aware_optimizer:
        shard_param.main_grad = None
        shard_param.decoupled_grad = shard_grad
        shard_param.grad = None
        shard_param.is_dion_param = False
        return

    shard_param.main_grad = None
    shard_param.decoupled_grad = None
    shard_param.grad = shard_grad.float()
    shard_param.is_dion_param = False


def bind_non_dion_optimizer_shard_grads_(
    *,
    model_groups,
    shard_groups,
    get_param_range_fn: Callable[[torch.nn.Parameter], object],
    log_grad_issue_fn: Callable,
    use_precision_aware_optimizer: bool,
) -> None:
    """Project canonical model-side non-Dion grads onto optimizer shards.

    This mirrors Megatron-Core DistributedOptimizer semantics as closely as
    possible:
    - source of truth on the model side is `model_param.main_grad`
    - optimizer-side canonical grad is `shard_param.grad` / `decoupled_grad`
    - Dion params are excluded entirely from this path
    """
    for model_group, shard_param_group in zip(model_groups, shard_groups):
        if len(model_group) != len(shard_param_group):
            raise RuntimeError(
                "[Dion] non-Dion grad copy requires equal model/shard group lengths "
                f"model_len={len(model_group)} shard_len={len(shard_param_group)}"
            )
        for model_param, shard_param in zip(model_group, shard_param_group):
            if shard_param is None or getattr(model_param, "is_dion_param", False):
                continue
            bind_non_dion_shard_grad_(
                model_param=model_param,
                shard_param=shard_param,
                get_param_range_fn=get_param_range_fn,
                log_grad_issue_fn=log_grad_issue_fn,
                use_precision_aware_optimizer=use_precision_aware_optimizer,
            )


def bind_dion_optimizer_shard_grads_(
    *,
    model_groups,
    shard_groups,
    get_dion_local_shard_grad_fn: Callable[[torch.nn.Parameter, torch.nn.Parameter], torch.Tensor],
    log_grad_issue_fn: Callable,
    use_precision_aware_optimizer: bool,
) -> None:
    """Bind Dion local shard grads onto optimizer shard params for one or more param groups."""
    for model_group, shard_param_group in zip(model_groups, shard_groups):
        if len(model_group) != len(shard_param_group):
            raise RuntimeError(
                "[Dion] Dion grad copy requires equal model/shard group lengths "
                f"model_len={len(model_group)} shard_len={len(shard_param_group)}"
            )
        for model_param, shard_param in zip(model_group, shard_param_group):
            if not getattr(model_param, "is_dion_param", False):
                continue
            bind_dion_shard_grad_(
                model_param=model_param,
                shard_param=shard_param,
                get_dion_local_shard_grad_fn=get_dion_local_shard_grad_fn,
                log_grad_issue_fn=log_grad_issue_fn,
                use_precision_aware_optimizer=use_precision_aware_optimizer,
            )


def build_grad_route_(
    *,
    use_precision_aware_optimizer: bool,
    model_float16_groups,
    shard_float16_groups,
    model_fp32_groups,
    shard_fp32_groups,
    shard_fp32_from_float16_groups,
    log_grad_issue_fn: Callable,
) -> dict:
    """Build the model-surface to optimizer-shard grad route."""
    if use_precision_aware_optimizer:
        grad_group_pairs = (
            (model_float16_groups, shard_float16_groups),
            (model_fp32_groups, shard_fp32_groups),
        )
    else:
        grad_group_pairs = (
            (model_float16_groups, shard_fp32_from_float16_groups),
            (model_fp32_groups, shard_fp32_groups),
        )
    return {
        "use_precision_aware_optimizer": use_precision_aware_optimizer,
        "grad_group_pairs": grad_group_pairs,
        "log_grad_issue": log_grad_issue_fn,
    }


def bind_optimizer_shard_grads_(
    *,
    is_stub_optimizer: bool,
    use_megatron_fsdp: bool,
    use_precision_aware_optimizer: bool,
    model_float16_groups,
    shard_float16_groups,
    model_fp32_groups,
    shard_fp32_groups,
    shard_fp32_from_float16_groups,
    get_param_range_fn: Callable[[torch.nn.Parameter], object],
    get_dion_local_shard_grad_fn: Callable[[torch.nn.Parameter, torch.nn.Parameter], torch.Tensor],
    log_grad_issue_fn: Callable,
    release_rs_buffers_fn: Callable,
) -> None:
    """Bind model-side grad surfaces onto optimizer shards and release RS buffers."""
    if is_stub_optimizer or use_megatron_fsdp:
        return

    grad_route = build_grad_route_(
        use_precision_aware_optimizer=use_precision_aware_optimizer,
        model_float16_groups=model_float16_groups,
        shard_float16_groups=shard_float16_groups,
        model_fp32_groups=model_fp32_groups,
        shard_fp32_groups=shard_fp32_groups,
        shard_fp32_from_float16_groups=shard_fp32_from_float16_groups,
        log_grad_issue_fn=log_grad_issue_fn,
    )
    use_precision_aware_optimizer = grad_route["use_precision_aware_optimizer"]
    log_grad_issue = grad_route["log_grad_issue"]
    grad_group_pairs = grad_route["grad_group_pairs"]

    for model_groups, shard_groups in grad_group_pairs:
        bind_non_dion_optimizer_shard_grads_(
            model_groups=model_groups,
            shard_groups=shard_groups,
            get_param_range_fn=get_param_range_fn,
            log_grad_issue_fn=log_grad_issue,
            use_precision_aware_optimizer=use_precision_aware_optimizer,
        )
        bind_dion_optimizer_shard_grads_(
            model_groups=model_groups,
            shard_groups=shard_groups,
            get_dion_local_shard_grad_fn=get_dion_local_shard_grad_fn,
            log_grad_issue_fn=log_grad_issue,
            use_precision_aware_optimizer=use_precision_aware_optimizer,
        )

    release_rs_buffers_fn()
