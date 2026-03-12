"""Grad-norm and clipping helpers for the Dion distributed optimizer."""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from typing import Callable, List, Sequence

import torch
import torch.distributed as dist


logger = logging.getLogger(__name__)
_NORM_ENTRY_DIAG_SEEN = set()


def get_optimizer_grad_(shard_param: torch.nn.Parameter):
    """Return the optimizer-side canonical grad tensor for one shard param."""
    grad = getattr(shard_param, "decoupled_grad", None)
    if grad is not None:
        return grad
    grad = getattr(shard_param, "grad", None)
    if grad is not None:
        return grad
    return None


def _norm_entry_diag_enabled(param_name: str) -> bool:
    spec = os.getenv("DION_NORM_ENTRY_DIAG", "")
    if not spec:
        return False
    if spec == "1":
        return True
    return any(token and token in param_name for token in spec.split(","))


def _maybe_log_norm_entry_(
    *,
    kind: str,
    model_param: torch.nn.Parameter,
    shard_param: torch.nn.Parameter,
    grad: torch.Tensor,
    name_fn,
    step: int | None,
) -> None:
    param_name = name_fn(model_param)
    if not param_name or not _norm_entry_diag_enabled(param_name):
        return
    rank = dist.get_rank() if dist.is_initialized() else 0
    target_step_env = os.getenv("DION_NORM_ENTRY_STEP", "")
    if target_step_env:
        try:
            target_step = int(target_step_env)
        except ValueError:
            target_step = None
        if target_step is not None and step != target_step:
            return
    key = (rank, step, kind, param_name)
    if key in _NORM_ENTRY_DIAG_SEEN:
        return
    _NORM_ENTRY_DIAG_SEEN.add(key)
    grad_fp32 = grad.reshape(-1).float()
    logger.info(
        "[DION_NORM_ENTRY] step=%s rank=%s kind=%s param=%s shard_shape=%s grad_shape=%s ptr=%s sum=%.6f abs_sum=%.6f sq_sum=%.6f amax=%.6f",
        step,
        rank,
        kind,
        param_name,
        tuple(shard_param.shape),
        tuple(grad.shape),
        grad.data_ptr(),
        float(grad_fp32.sum().item()),
        float(grad_fp32.abs().sum().item()),
        float(torch.dot(grad_fp32, grad_fp32).item()),
        float(grad_fp32.abs().max().item()) if grad.numel() > 0 else 0.0,
    )


@dataclass(frozen=True)
class DionNormEntry:
    """Pre-filtered Dion grad-norm/clip entry."""

    model_param: torch.nn.Parameter
    shard_param: torch.nn.Parameter


@dataclass(frozen=True)
class NonDionNormEntry:
    """Pre-filtered non-Dion grad-norm/clip entry."""

    model_param: torch.nn.Parameter
    shard_param: torch.nn.Parameter


def build_grad_norm_entries_(
    *,
    model_groups: Sequence[Sequence[torch.nn.Parameter]],
    shard_groups: Sequence[Sequence[torch.nn.Parameter]],
    is_unshared: Callable[[torch.nn.Parameter], bool],
    is_tp_unique: Callable[[torch.nn.Parameter], bool],
) -> tuple[List[DionNormEntry], List[NonDionNormEntry]]:
    """Build pre-filtered grad-norm/clip contribution entries."""
    dion_entries: List[DionNormEntry] = []
    non_dion_entries: List[NonDionNormEntry] = []
    debug_tp_entries = bool(torch.distributed.is_initialized()) and bool(
        os.environ.get("DION_DEBUG_TP_NORM_ENTRIES")
    )

    for model_group, shard_group in zip(model_groups, shard_groups):
        if len(model_group) != len(shard_group):
            raise RuntimeError(
                "[Dion] grad-norm entry build requires equal model/shard group lengths "
                f"model_len={len(model_group)} shard_len={len(shard_group)}"
            )
        for zipped_model_param, shard_param in zip(model_group, shard_group):
            if shard_param is None:
                continue
            if shard_param.numel() == 0:
                continue
            model_param = getattr(shard_param, "_model_param", None)
            if model_param is None:
                raise RuntimeError(
                    "[Dion] grad-norm entry build requires shard_param._model_param "
                    f"for shard shape={tuple(shard_param.shape)}"
                )
            if model_param is not zipped_model_param:
                raise RuntimeError(
                    "[Dion] grad-norm entry zip/backlink mismatch: "
                    f"zip_id={id(zipped_model_param)} backlink_id={id(model_param)} "
                    f"shard_shape={tuple(shard_param.shape)}"
                )
            if not (is_unshared(model_param) and is_tp_unique(model_param)):
                continue

            if getattr(model_param, "is_dion_param", False):
                dion_entries.append(
                    DionNormEntry(
                        model_param=model_param,
                        shard_param=shard_param,
                    )
                )
            else:
                if debug_tp_entries:
                    logger.info(
                        "[DION_TP_NORM_ENTRY] kind=non_dion name=%s tp_unique=%s tensor_model_parallel=%s model_shape=%s shard_shape=%s",
                        getattr(model_param, "_param_name", getattr(model_param, "name", f"id_{id(model_param)}")),
                        bool(is_tp_unique(model_param)),
                        bool(getattr(model_param, "tensor_model_parallel", False)),
                        tuple(model_param.shape),
                        tuple(shard_param.shape),
                    )
                non_dion_entries.append(
                    NonDionNormEntry(
                        model_param=model_param,
                        shard_param=shard_param,
                    )
                )

    return dion_entries, non_dion_entries


def log_zero_global_grad_norm_(optimizer, grads_for_norm_count: int) -> None:
    """Emit an always-on log when the global grad norm is exactly zero."""
    try:
        global_rank = dist.get_rank() if dist.is_initialized() else 0
    except Exception:
        global_rank = -1

    if global_rank not in (0, -1):
        return

    step = None
    try:
        param_groups = getattr(optimizer, "param_groups", None)
        if param_groups:
            step = int(param_groups[0].get("step", -1))
    except Exception:
        step = None

    logger.error(
        "[DION_ZERO_GLOBAL_GRAD_NORM] step=%s global_rank=%s grads_for_norm=%d",
        step,
        global_rank,
        grads_for_norm_count,
    )


def append_precomputed_norm_grads_(
    *,
    grads_for_norm: List[torch.Tensor],
    dion_entries: Sequence[DionNormEntry],
    non_dion_entries: Sequence[NonDionNormEntry],
    name_fn,
    step: int | None = None,
) -> None:
    """Append grad-norm tensors from pre-filtered cached entries."""
    for entry in dion_entries:
        grad = get_optimizer_grad_(entry.shard_param)
        if grad is None:
            continue
        if grad.numel() == 0:
            continue
        _maybe_log_norm_entry_(
            kind="dion",
            model_param=entry.model_param,
            shard_param=entry.shard_param,
            grad=grad,
            name_fn=name_fn,
            step=step,
        )
        grads_for_norm.append(grad.reshape(-1))

    for entry in non_dion_entries:
        grad = get_optimizer_grad_(entry.shard_param)
        if grad is None:
            continue
        if grad.numel() == 0:
            continue
        _maybe_log_norm_entry_(
            kind="non_dion",
            model_param=entry.model_param,
            shard_param=entry.shard_param,
            grad=grad,
            name_fn=name_fn,
            step=step,
        )
        grads_for_norm.append(grad.reshape(-1))


def debug_validate_precomputed_norm_grads_(
    *,
    dion_entries: Sequence[DionNormEntry],
    non_dion_entries: Sequence[NonDionNormEntry],
    name_fn,
) -> None:
    """Best-effort validator to identify the exact grad shard that breaks clip-grad.

    Used only during CP debug. This intentionally synchronizes to surface illegal
    memory access before multi-tensor clipping obscures the culprit.
    """
    for idx, entry in enumerate(dion_entries):
        grad = get_optimizer_grad_(entry.shard_param)
        if grad is None:
            continue
        if grad.numel() == 0:
            continue
        try:
            probe = grad.reshape(-1).float().abs().amax()
            torch.cuda.synchronize(probe.device)
            del probe
        except Exception as exc:
            logger.error(
                "[DION_CP_GRADNORM_BAD] kind=dion idx=%s param=%s shard_shape=%s grad_shape=%s exc=%r",
                idx,
                name_fn(entry.model_param),
                tuple(entry.shard_param.shape),
                tuple(grad.shape),
                exc,
            )
            raise

    for idx, entry in enumerate(non_dion_entries):
        grad = get_optimizer_grad_(entry.shard_param)
        if grad is None:
            continue
        if grad.numel() == 0:
            continue
        try:
            probe = grad.reshape(-1).float().abs().amax()
            torch.cuda.synchronize(probe.device)
            del probe
        except Exception as exc:
            logger.error(
                "[DION_CP_GRADNORM_BAD] kind=non_dion idx=%s param=%s shard_shape=%s grad_shape=%s exc=%r",
                idx,
                name_fn(entry.shard_param),
                tuple(entry.shard_param.shape),
                tuple(grad.shape),
                exc,
            )
            raise


def log_grad_norm_contributors_(
    *,
    dion_entries: Sequence[DionNormEntry],
    non_dion_entries: Sequence[NonDionNormEntry],
    name_fn,
    step: int,
    topk: int = 12,
) -> None:
    """Log local grad-norm contributors once for focused CP debug."""
    try:
        global_rank = dist.get_rank() if dist.is_initialized() else 0
    except Exception:
        global_rank = -1

    if global_rank not in (0, -1):
        return

    contribs = []
    grad_ptr_map = {}
    totals = {
        "dion_unscaled": 0.0,
        "non_dion_unscaled": 0.0,
    }
    counts = {key: 0 for key in totals}

    def _record(kind: str, param_obj, grad: torch.Tensor) -> None:
        grad_fp32 = grad.reshape(-1).float()
        raw_sq = torch.dot(grad_fp32, grad_fp32).item()
        eff_sq = raw_sq
        bucket = f"{kind}_unscaled"
        totals[bucket] += eff_sq
        counts[bucket] += 1
        grad_ptr = grad.data_ptr()
        existing = grad_ptr_map.get(grad_ptr)
        if existing is None:
            grad_ptr_map[grad_ptr] = (kind, name_fn(param_obj), grad.numel())
        else:
            logger.warning(
                "[DION_GRADNORM_DUP_PTR] step=%s ptr=%s first_kind=%s first_param=%s first_numel=%s "
                "dup_kind=%s dup_param=%s dup_numel=%s",
                step,
                grad_ptr,
                existing[0],
                existing[1],
                existing[2],
                kind,
                name_fn(param_obj),
                grad.numel(),
            )
        contribs.append(
            {
                "kind": kind,
                "name": name_fn(param_obj),
                "numel": grad.numel(),
                "scaled": False,
                "raw_l2": math.sqrt(raw_sq),
                "eff_l2": math.sqrt(eff_sq),
                "amax": grad_fp32.abs().max().item(),
            }
        )

    for entry in dion_entries:
        grad = get_optimizer_grad_(entry.shard_param)
        if grad is None or grad.numel() == 0:
            continue
        _record("dion", entry.model_param, grad)

    for entry in non_dion_entries:
        grad = get_optimizer_grad_(entry.shard_param)
        if grad is None or grad.numel() == 0:
            continue
        _record("non_dion", entry.model_param, grad)

    contribs.sort(key=lambda item: item["eff_l2"], reverse=True)
    top_items = contribs[:topk]
    summary = " ".join(
        f"{key}={math.sqrt(value):.6f}[n={counts[key]}]" for key, value in totals.items()
    )
    logger.info(
        "[DION_GRADNORM_SUMMARY] step=%s total_entries=%s %s",
        step,
        len(contribs),
        summary,
    )
    for idx, item in enumerate(top_items):
        logger.info(
            "[DION_GRADNORM_TOPK] step=%s idx=%s kind=%s scaled=%s eff_l2=%.6f raw_l2=%.6f "
            "amax=%.6f numel=%s param=%s",
            step,
            idx,
            item["kind"],
            int(item["scaled"]),
            item["eff_l2"],
            item["raw_l2"],
            item["amax"],
            item["numel"],
            item["name"],
        )


def clip_precomputed_grad_groups_(
    *,
    dion_entries: Sequence[DionNormEntry],
    non_dion_entries: Sequence[NonDionNormEntry],
    clip_coeff: float,
) -> None:
    """Clip gradients using pre-filtered cached entries."""
    for entry in dion_entries:
        grad = get_optimizer_grad_(entry.shard_param)
        if grad is not None and grad.numel() > 0:
            grad.mul_(clip_coeff)

    for entry in non_dion_entries:
        grad = get_optimizer_grad_(entry.shard_param)
        if grad is not None and grad.numel() > 0:
            grad.mul_(clip_coeff)
