"""Grad wiring helpers for Dion distributed optimizer.

This module extracts small, self-contained helpers from the Dion distributed
optimizer wrapper. The intent is mechanical refactor only: keep numerics and
collective patterns unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Callable, Optional, Sequence, Tuple
import os

import torch

from ..distrib_optimizer import Range
from .grad_diag import log_dion_copy_debug_
from .param_selection import is_moe_expert_param

logger = logging.getLogger(__name__)

_GRAD_CONTRACT_DIAG_SEEN = set()
_BUCKET_SLICE_FP_DIAG_SEEN = set()


def _grad_contract_diag_enabled(param_name: str) -> bool:
    spec = os.getenv("DION_GRAD_CONTRACT_DIAG", "")
    if not spec:
        return False
    if spec == "1":
        return True
    return any(token and token in param_name for token in spec.split(","))


def _maybe_log_grad_contract_(
    *,
    kind: str,
    model_param: torch.nn.Parameter,
    shard_param: torch.nn.Parameter,
    model_grad: Optional[torch.Tensor],
    shard_grad: Optional[torch.Tensor],
    param_range=None,
    extra: str = "",
    step: int | None = None,
) -> None:
    param_name = getattr(model_param, "_param_name", "") or getattr(shard_param, "_param_name", "")
    if not param_name or not _grad_contract_diag_enabled(param_name):
        return
    target_step_env = os.getenv("DION_GRAD_CONTRACT_STEP", "")
    if target_step_env:
        try:
            target_step = int(target_step_env)
        except ValueError:
            target_step = None
        if target_step is not None and step != target_step:
            return
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    key = (rank, step, kind, param_name)
    if key in _GRAD_CONTRACT_DIAG_SEEN:
        return
    _GRAD_CONTRACT_DIAG_SEEN.add(key)

    def _fp(t: Optional[torch.Tensor]):
        if t is None:
            return None
        tf = t.float()
        return (
            tuple(t.shape),
            float(tf.sum().item()),
            float(tf.abs().sum().item()),
            float((tf ** 2).sum().item()),
            float(tf.abs().max().item()) if t.numel() > 0 else 0.0,
        )

    logger.info(
        "[DION_GRAD_CONTRACT] step=%s rank=%s kind=%s param=%s range=(%s,%s) model=%s shard=%s extra=%s",
        step,
        rank,
        kind,
        param_name,
        int(param_range.start) if param_range is not None else -1,
        int(param_range.end) if param_range is not None else -1,
        _fp(model_grad),
        _fp(shard_grad),
        extra,
    )


def _bucket_slice_fp_diag_enabled(param_name: str) -> bool:
    spec = os.getenv("DION_BUCKET_SLICE_FP_NAMES", "")
    if not spec:
        return False
    if spec == "1":
        return True
    return any(token and token in param_name for token in spec.split(","))


def _maybe_log_bucket_slice_fp_(
    *,
    model_param: torch.nn.Parameter,
    bucket_slice: Optional[torch.Tensor],
    state_replica_group,
    rs_start: Optional[int],
    rs_end: Optional[int],
    local_shape,
    name_fn: Callable[[torch.nn.Parameter], str],
) -> None:
    param_name = name_fn(model_param)
    if not param_name or not _bucket_slice_fp_diag_enabled(param_name):
        return
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    key = (rank, param_name, int(rs_start) if rs_start is not None else -1, int(rs_end) if rs_end is not None else -1)
    if key in _BUCKET_SLICE_FP_DIAG_SEEN:
        return
    _BUCKET_SLICE_FP_DIAG_SEEN.add(key)

    def _fp(t: Optional[torch.Tensor]):
        if t is None:
            return None
        tf = t.detach().float()
        return {
            "shape": tuple(tf.shape),
            "sum": float(tf.sum().item()),
            "abs_sum": float(tf.abs().sum().item()),
            "sq_sum": float((tf ** 2).sum().item()),
            "amax": float(tf.abs().max().item()) if tf.numel() > 0 else 0.0,
            "sample": tf.view(-1)[: min(8, tf.numel())].cpu().tolist(),
        }

    local_fp = {
        "param": param_name,
        "local_shape": tuple(local_shape),
        "rs": (
            int(rs_start) if rs_start is not None else -1,
            int(rs_end) if rs_end is not None else -1,
        ),
        "slice": _fp(bucket_slice),
    }

    if state_replica_group is not None and torch.distributed.get_world_size(state_replica_group) > 1:
        gathered = [None] * torch.distributed.get_world_size(state_replica_group)
        torch.distributed.all_gather_object(gathered, local_fp, group=state_replica_group)
        logger.info(
            "[DION_BUCKET_SLICE_FP] rank=%s group=%s gathered=%s",
            rank,
            torch.distributed.get_process_group_ranks(state_replica_group),
            gathered,
        )
    else:
        logger.info("[DION_BUCKET_SLICE_FP] rank=%s gathered=%s", rank, [local_fp])


def _get_dion_rs_local_buffer(bucket):
    """Return the canonical Dion RS local-output buffer for one bucket."""
    grad_local_view = getattr(bucket, "dion_grad_local_view", None)
    if grad_local_view is not None:
        return grad_local_view
    raise RuntimeError(
        "[Dion] Missing canonical dion_grad_local_view; Dion grad fetch must read only "
        "the stock-DO-aligned local shard output."
    )


def _get_optimizer_grad(param: torch.nn.Parameter) -> Optional[torch.Tensor]:
    """Return the optimizer-side canonical grad tensor for one shard param."""
    grad = getattr(param, "decoupled_grad", None)
    if grad is not None:
        return grad
    grad = getattr(param, "grad", None)
    if grad is not None:
        return grad
    return None


def _numel_from_shape(shape) -> int:
    if len(shape) == 2:
        return int(shape[0]) * int(shape[1])
    return int(shape[0])


def _slice_model_grad_local_shard(
    model_param: torch.nn.Parameter, local_shape, dion_info: Optional[dict] = None
) -> Optional[torch.Tensor]:
    if dion_info is None:
        dion_info = getattr(model_param, "dion_info", None)
    model_grad = getattr(model_param, "main_grad", None)
    if dion_info is None or model_grad is None or model_grad.ndim != 2:
        return None
    start_idx = dion_info.get("start_idx")
    end_idx = dion_info.get("end_idx")
    fs_split_dim = dion_info.get("fs_split_dim")
    if start_idx is None or end_idx is None or fs_split_dim not in (0, 1):
        return None
    if fs_split_dim == 0:
        shard = model_grad[start_idx:end_idx, :]
    else:
        shard = model_grad[:, start_idx:end_idx]
    if tuple(shard.shape) != tuple(local_shape):
        return None
    return shard


@dataclass
class DionGradFetchResult:
    """Resolved Dion grad-buffer slice information for one optimizer shard."""

    bucket_slice: Optional[torch.Tensor]
    buffer_idx: Optional[int]
    bucket_idx: Optional[int]
    expected_len: int
    bucket: Optional[object] = None
    rs_start: Optional[int] = None
    rs_end: Optional[int] = None
    used_direct_range: bool = False


def fetch_dion_grad_from_bucket_(
    *,
    model_param: torch.nn.Parameter,
    shard_main_param: torch.nn.Parameter,
    local_shape,
    dion_info,
    param_range_start: int,
    model_param_gbuf_map,
    buffers,
    name_fn: Callable[[torch.nn.Parameter], str],
    log_grad_issue_fn: Callable,
) -> DionGradFetchResult:
    """Resolve the Dion reduce-scatter output slice for one parameter."""
    buffer_idx = dion_info.get("buffer_idx")
    bucket_idx = dion_info.get("bucket_idx")

    if buffer_idx is None or bucket_idx is None:
        try:
            gbuf_index, _, found_bucket_idx = model_param_gbuf_map[model_param]
            buffer_idx = gbuf_index if buffer_idx is None else buffer_idx
            bucket_idx = found_bucket_idx if bucket_idx is None else bucket_idx
            logger.warning(
                "[Dion] Filled buffer_idx=%s, bucket_idx=%s for param id=%s",
                buffer_idx,
                bucket_idx,
                id(model_param),
            )
        except Exception as error:
            logger.error(
                "[Dion] Could not infer buffer/bucket for param id=%s: %s",
                id(model_param),
                error,
            )

    expected_len = local_shape[0] * local_shape[1] if len(local_shape) == 2 else local_shape[0]
    result = DionGradFetchResult(
        bucket_slice=None,
        buffer_idx=buffer_idx,
        bucket_idx=bucket_idx,
        expected_len=expected_len,
    )

    try:
        bucket = buffers[buffer_idx].buckets[bucket_idx]
        result.bucket = bucket
        rs_start = dion_info.get("rs_start")
        rs_end = dion_info.get("rs_end")

        if rs_start is not None and rs_end is not None:
            result.rs_start = int(rs_start)
            result.rs_end = int(rs_end)
            result.used_direct_range = True
            actual_len = rs_end - rs_start
            if actual_len != expected_len:
                param_name = name_fn(model_param)
                logger.error(
                    "[Dion] CACHED GRAD RANGE MISMATCH param=%s rs_range=(%s,%s) actual_len=%s expected=%s",
                    param_name,
                    rs_start,
                    rs_end,
                    actual_len,
                    expected_len,
                )
                log_grad_issue_fn(
                    "DION_CACHED_RS_RANGE_LEN_MISMATCH",
                    model_param,
                    shard_main_param,
                    buffer_idx=buffer_idx,
                    bucket_idx=bucket_idx,
                    bucket_id=getattr(bucket, "bucket_id", None),
                    rs_start=int(rs_start),
                    rs_end=int(rs_end),
                    expected_len=int(expected_len),
                    actual_len=int(actual_len),
                )
                raise RuntimeError(
                    f"[Dion] Cached Dion RS range length mismatch for {param_name}: "
                    f"actual={actual_len} expected={expected_len}"
                )
            dion_rs_local_buffer = _get_dion_rs_local_buffer(bucket)
            result.bucket_slice = dion_rs_local_buffer[rs_start:rs_end].view(local_shape)
            return result

        if hasattr(bucket, "dion_param_shard_range") and model_param in bucket.dion_param_shard_range:
            rs_start, rs_end = bucket.dion_param_shard_range[model_param]
            result.rs_start = int(rs_start)
            result.rs_end = int(rs_end)
            result.used_direct_range = True
            actual_len = rs_end - rs_start
            if actual_len != expected_len:
                param_name = name_fn(model_param)
                logger.error(
                    "[Dion] GRAD RANGE MISMATCH param=%s rs_range=(%s,%s) actual_len=%s expected=%s",
                    param_name,
                    rs_start,
                    rs_end,
                    actual_len,
                    expected_len,
                )
                log_grad_issue_fn(
                    "DION_RS_RANGE_LEN_MISMATCH",
                    model_param,
                    shard_main_param,
                    buffer_idx=buffer_idx,
                    bucket_idx=bucket_idx,
                    bucket_id=getattr(bucket, "bucket_id", None),
                    rs_start=int(rs_start),
                    rs_end=int(rs_end),
                    expected_len=int(expected_len),
                    actual_len=int(actual_len),
                )
                raise RuntimeError(
                    f"[Dion] Dion RS range length mismatch for {param_name}: "
                    f"actual={actual_len} expected={expected_len}"
                )
            dion_rs_local_buffer = _get_dion_rs_local_buffer(bucket)
            result.bucket_slice = dion_rs_local_buffer[rs_start:rs_end].view(local_shape)
            return result

        param_name = name_fn(model_param)
        has_range_map = hasattr(bucket, "dion_param_shard_range")
        range_keys = len(bucket.dion_param_shard_range) if has_range_map else 0
        log_grad_issue_fn(
            "DION_RS_RANGE_MISSING",
            model_param,
            shard_main_param,
            buffer_idx=buffer_idx,
            bucket_idx=bucket_idx,
            bucket_id=getattr(bucket, "bucket_id", None),
            has_dion_param_shard_range=bool(has_range_map),
            range_keys=int(range_keys),
            expected_len=int(expected_len),
        )
        raise RuntimeError(
            f"[Dion] Missing canonical dion_param_shard_range for {param_name} "
            f"(bucket={getattr(bucket, 'bucket_id', None)} expected_len={expected_len})"
        )
    except Exception as error:
        logger.error(
            "[Dion] Failed to fetch bucket slice for param id=%s: %s",
            id(model_param),
            error,
        )
        log_grad_issue_fn(
            "DION_BUCKET_SLICE_EXCEPTION",
            model_param,
            shard_main_param,
            buffer_idx=buffer_idx,
            bucket_idx=bucket_idx,
            err=str(error),
        )

    return result


def build_non_dion_shard_grad_from_main_(
    *,
    model_param: torch.nn.Parameter,
    shard_main_param: torch.nn.Parameter,
    param_range,
    log_grad_issue_fn: Callable,
) -> torch.Tensor:
    """Build the stock-DO optimizer shard grad from canonical `model_param.main_grad`."""
    model_grad = model_param.main_grad
    if model_grad is None:
        log_grad_issue_fn("NON_DION_MODEL_MAIN_GRAD_NONE", model_param, shard_main_param)
        raise RuntimeError(
            "[Dion] non-Dion stock shard grad requires canonical model_param.main_grad "
            f"param_shape={tuple(model_param.shape)} shard_shape={tuple(shard_main_param.shape)}"
        )

    flat_grad = model_grad.view(-1)
    stock_start = getattr(model_param, "_stock_param_start", None)
    stock_end = getattr(model_param, "_stock_param_end", None)
    if stock_start is not None and stock_end is not None:
        start = int(stock_start)
        end = int(stock_end)
    else:
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
            "[Dion] non-Dion stock shard grad slice exceeded canonical main_grad "
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
            "[Dion] non-Dion stock shard grad size mismatch "
            f"expected={int(shard_main_param.nelement())} got={int(shard_view.numel())}"
        )
    return shard_view.view(shard_main_param.shape)


def bind_dion_shard_grad_(
    *,
    model_param: torch.nn.Parameter,
    shard_param: torch.nn.Parameter,
    get_param_range_fn: Callable[[torch.nn.Parameter], object],
    get_dion_info_fn: Callable[[torch.nn.Parameter], dict],
    model_param_gbuf_map,
    buffers,
    name_fn: Callable[[torch.nn.Parameter], str],
    log_grad_issue_fn: Callable,
    use_prec_opt: bool,
    state_replica_group,
    debug_copy_count: int,
    max_infos: list,
    max_tensors: list,
    step: int | None = None,
) -> int:
    """Bind one Dion grad shard onto an optimizer shard param.

    Dion keeps its own canonical packed local RS shard. This differs from the
    flat stock local-shard contract used by pure non-Dion params because Dion
    FS/TP sharding does not generally align with flat bucket-local optimizer
    shards.
    """
    dion_info = get_dion_info_fn(model_param)
    buffer_idx = dion_info.get("buffer_idx")
    bucket_idx = dion_info.get("bucket_idx")
    local_shape = dion_info.get("shape", model_param.shape)
    param_range = get_param_range_fn(model_param)["param"]

    bucket_grad = fetch_dion_grad_from_bucket_(
        model_param=model_param,
        shard_main_param=shard_param,
        dion_info=dion_info,
        local_shape=local_shape,
        param_range_start=int(param_range.start),
        model_param_gbuf_map=model_param_gbuf_map,
        buffers=buffers,
        name_fn=name_fn,
        log_grad_issue_fn=log_grad_issue_fn,
    )
    if bucket_grad.bucket_slice is None:
        raise RuntimeError(
            "[Dion] Missing canonical Dion local RS shard for optimizer grad bind "
            f"param={name_fn(model_param)} local_shape={tuple(int(x) for x in local_shape)}"
        )

    param_range = Range(int(bucket_grad.rs_start), int(bucket_grad.rs_end))
    shard_grad_view = bucket_grad.bucket_slice
    shard_grad = shard_grad_view.float()

    shard_param.is_dion_param = True
    shard_param.main_grad = None
    if use_prec_opt:
        shard_param.decoupled_grad = shard_grad
        shard_param.grad = None
        _maybe_log_grad_contract_(
            kind="dion",
            model_param=model_param,
            shard_param=shard_param,
            model_grad=shard_grad_view,
            shard_grad=shard_grad,
            param_range=param_range,
            step=step,
            extra=(
                f"buffer={buffer_idx} bucket={bucket_idx} "
                f"rs=({int(bucket_grad.rs_start)},{int(bucket_grad.rs_end)})"
            ),
        )
    else:
        shard_param.decoupled_grad = None
        shard_param.grad = shard_grad
        _maybe_log_grad_contract_(
            kind="dion",
            model_param=model_param,
            shard_param=shard_param,
            model_grad=shard_grad_view,
            shard_grad=shard_grad,
            param_range=param_range,
            step=step,
            extra=(
                f"buffer={buffer_idx} bucket={bucket_idx} "
                f"rs=({int(bucket_grad.rs_start)},{int(bucket_grad.rs_end)})"
            ),
        )

    try:
        max_infos.append(
            (
                model_param,
                shard_param,
                int(buffer_idx) if buffer_idx is not None else -1,
                int(bucket_idx) if bucket_idx is not None else -1,
                tuple(int(x) for x in local_shape),
                int(shard_grad.numel()),
            )
        )
        optimizer_grad = _get_optimizer_grad(shard_param)
        if optimizer_grad is not None:
            max_tensors.append(optimizer_grad.abs().amax())
    except Exception:
        pass

    return debug_copy_count


def bind_non_dion_shard_grad_(
    *,
    model_param: torch.nn.Parameter,
    shard_param: torch.nn.Parameter,
    get_param_range_fn: Callable[[torch.nn.Parameter], object],
    model_param_gbuf_map,
    buffers,
    name_fn: Callable[[torch.nn.Parameter], str],
    log_grad_issue_fn: Callable,
    use_prec_opt: bool,
    step: int | None = None,
) -> None:
    """Bind one non-Dion grad shard onto an optimizer shard param."""
    param_range = get_param_range_fn(model_param)["param"]

    model_grad = getattr(model_param, "main_grad", None)
    if model_grad is None:
        raise RuntimeError(
            "[Dion] non-Dion optimizer grad bind requires canonical model_param.main_grad "
            f"param={name_fn(model_param)} shard_shape={tuple(shard_param.shape)}"
        )

    shard_grad = build_non_dion_shard_grad_from_main_(
        model_param=model_param,
        shard_main_param=shard_param,
        param_range=param_range,
        log_grad_issue_fn=log_grad_issue_fn,
    )

    if use_prec_opt:
        shard_param.main_grad = None
        shard_param.decoupled_grad = shard_grad
        shard_param.grad = None
        shard_param.is_dion_param = False
        _maybe_log_grad_contract_(
            kind="non_dion",
            model_param=model_param,
            shard_param=shard_param,
            model_grad=model_grad,
            shard_grad=shard_grad,
            param_range=param_range,
            step=step,
            extra=(
                f"shared={int(bool(getattr(model_param, 'shared', False)))} "
                f"shared_embedding={int(bool(getattr(model_param, 'shared_embedding', False)))}"
            ),
        )
        return

    shard_param.main_grad = None
    shard_param.grad = shard_grad.float()
    shard_param.is_dion_param = False
    _maybe_log_grad_contract_(
        kind="non_dion",
        model_param=model_param,
        shard_param=shard_param,
        model_grad=model_grad,
        shard_grad=shard_param.grad,
        param_range=param_range,
        step=step,
        extra=(
            f"shared={int(bool(getattr(model_param, 'shared', False)))} "
            f"shared_embedding={int(bool(getattr(model_param, 'shared_embedding', False)))}"
        ),
    )


def copy_stock_non_dion_grads_(
    *,
    model_groups,
    shard_groups,
    get_param_range_fn: Callable[[torch.nn.Parameter], object],
    log_grad_issue_fn: Callable,
    use_prec_opt: bool,
    step: int | None = None,
) -> None:
    """Project canonical model-side non-Dion grads onto optimizer shards.

    This mirrors Megatron-Core DistributedOptimizer semantics as closely as
    possible:
    - source of truth on the model side is `model_param.main_grad`
    - optimizer-side canonical grad is `shard_param.grad` / `decoupled_grad`
    - Dion params are excluded entirely from this path
    """
    for model_group, shard_group in zip(model_groups, shard_groups):
        for model_param, shard_param in zip(model_group, shard_group):
            if shard_param is None or getattr(model_param, "is_dion_param", False):
                continue
            bind_non_dion_shard_grad_(
                model_param=model_param,
                shard_param=shard_param,
                get_param_range_fn=get_param_range_fn,
                model_param_gbuf_map=None,
                buffers=None,
                name_fn=lambda p: getattr(p, "_param_name", f"id_{id(p)}"),
                log_grad_issue_fn=log_grad_issue_fn,
                use_prec_opt=use_prec_opt,
                step=step,
            )


def copy_grad_groups_(
    *,
    model_groups,
    shard_groups,
    get_param_range_fn: Callable[[torch.nn.Parameter], object],
    get_dion_info_fn: Callable[[torch.nn.Parameter], dict],
    model_param_gbuf_map,
    buffers,
    name_fn: Callable[[torch.nn.Parameter], str],
    log_grad_issue_fn: Callable,
    use_prec_opt: bool,
    state_replica_group,
    debug_copy_count: int,
    max_infos: list,
    max_tensors: list,
    step: int | None = None,
) -> int:
    """Copy Dion model grads to optimizer shard grads for one or more param groups."""
    for model_group, shard_group in zip(model_groups, shard_groups):
        for model_param, shard_param in zip(model_group, shard_group):
            if not getattr(model_param, "is_dion_param", False):
                continue
            debug_copy_count = bind_dion_shard_grad_(
                model_param=model_param,
                shard_param=shard_param,
                get_param_range_fn=get_param_range_fn,
                get_dion_info_fn=get_dion_info_fn,
                model_param_gbuf_map=model_param_gbuf_map,
                buffers=buffers,
                name_fn=name_fn,
                log_grad_issue_fn=log_grad_issue_fn,
                use_prec_opt=use_prec_opt,
                state_replica_group=state_replica_group,
                debug_copy_count=debug_copy_count,
                max_infos=max_infos,
                max_tensors=max_tensors,
                step=step,
            )

    return debug_copy_count


def fix_zero_dion_grads_(
    *,
    max_tensors: Sequence[torch.Tensor],
    max_infos: Sequence[Tuple[torch.nn.Parameter, torch.nn.Parameter, int, int, Tuple[int, int], int]],
    max_zero_grad_warnings: int,
    name_fn: Callable[[torch.nn.Parameter], str],
    get_dion_info_fn: Callable[[torch.nn.Parameter], dict],
    buffers,
    use_prec_opt: bool,
) -> int:
    """Detect exact-zero dense Dion shard grads and fail fast.

    Exact-zero dense Dion shard grads are treated as invalid runtime states.
    Sparse MoE expert weights are exempt because routing can legitimately skip
    an expert on a step.
    """
    if not max_tensors:
        return 0

    try:
        max_cpu = torch.stack(list(max_tensors)).detach().cpu()
        zero_indices = (max_cpu == 0).nonzero(as_tuple=False).view(-1).tolist()
    except Exception:
        zero_indices = []

    warned = 0
    for idx in zero_indices:
        if warned >= max_zero_grad_warnings:
            break
        try:
            model_param, shard_main_param, buffer_idx, bucket_idx, local_shape, expected_len = max_infos[idx]
        except Exception:
            continue

        param_name = name_fn(model_param)
        is_moe_expert = is_moe_expert_param(model_param, param_name)
        if is_moe_expert:
            # Sparse MoE routing can legitimately produce exact-zero gradients for
            # individual expert weights on some steps. Treat these as valid zeros.
            continue

        dion_info = get_dion_info_fn(model_param) or {}
        model_grad = getattr(model_param, "main_grad", None)
        input_local_max = 0.0
        rs_local_max = 0.0
        fs_group_size = None
        fs_group_rank = None
        try:
            if buffer_idx >= 0 and bucket_idx >= 0:
                bucket = buffers[buffer_idx].buckets[bucket_idx]
                dion_grad_buffer = getattr(bucket, "dion_grad_buffer", None)
                pack_total = int(getattr(bucket, "fs_pack_total", 0))
                rs_start = dion_info.get("rs_start")
                rs_end = dion_info.get("rs_end")
                fs_group = getattr(bucket, "dion_comm_group", None)
                if fs_group is not None:
                    fs_group_size = int(fs_group.size())
                    fs_group_rank = int(torch.distributed.get_rank(group=fs_group))
                if (
                    dion_grad_buffer is not None
                    and pack_total > 0
                    and rs_start is not None
                    and rs_end is not None
                    and fs_group_rank is not None
                ):
                    input_start = fs_group_rank * pack_total + int(rs_start)
                    input_end = input_start + int(expected_len)
                    if input_end <= dion_grad_buffer.numel():
                        input_local_max = float(dion_grad_buffer[input_start:input_end].abs().amax())
                rs_buffer = _get_dion_rs_local_buffer(bucket)
                if rs_buffer is not None and rs_start is not None and rs_end is not None:
                    if int(rs_end) <= rs_buffer.numel():
                        rs_local_max = float(rs_buffer[int(rs_start):int(rs_end)].abs().amax())
        except Exception:
            input_local_max = 0.0
            rs_local_max = 0.0

        entry_param_same = None
        entry_param_id = None
        entry_main_grad_ptr = None
        entry_main_grad_max = 0.0
        model_main_grad_ptr = None
        try:
            if model_grad is not None:
                model_main_grad_ptr = int(model_grad.untyped_storage().data_ptr())
            if buffer_idx >= 0 and bucket_idx >= 0:
                bucket = buffers[buffer_idx].buckets[bucket_idx]
                entry = None
                name_to_entry = getattr(bucket, "dion_param_name_to_entry", None)
                if name_to_entry:
                    entry = name_to_entry.get(param_name)
                if entry is not None:
                    entry_param = entry.get("param")
                    if entry_param is not None:
                        entry_param_same = bool(entry_param is model_param)
                        entry_param_id = int(id(entry_param))
                        entry_main_grad = getattr(entry_param, "main_grad", None)
                        if entry_main_grad is not None:
                            entry_main_grad_ptr = int(entry_main_grad.untyped_storage().data_ptr())
                            entry_main_grad_max = float(entry_main_grad.abs().amax())
        except Exception:
            entry_param_same = None

        try:
            optimizer_grad = _get_optimizer_grad(shard_main_param)
            grad_shape = tuple(optimizer_grad.shape) if optimizer_grad is not None else None
        except Exception:
            grad_shape = None
        model_grad_shape = None
        full_alt_max = 0.0
        try:
            if model_grad is not None:
                model_grad_shape = tuple(model_grad.shape)
                full_alt_max = float(model_grad.abs().amax())
        except Exception:
            model_grad_shape = None
            full_alt_max = 0.0

        fs_split_dim = dion_info.get("fs_split_dim")
        start_idx = dion_info.get("start_idx")
        end_idx = dion_info.get("end_idx")
        raise RuntimeError(
            "[Dion] exact-zero dense Dion shard grad detected; this is an invalid runtime state "
            f"param={param_name} grad_shape={grad_shape} buffer={buffer_idx} bucket={bucket_idx} "
            f"expected_len={expected_len} local_shape={local_shape} model_grad_shape={model_grad_shape} "
            f"fs_split_dim={fs_split_dim} start={start_idx} end={end_idx} "
            f"full_main_grad_max={full_alt_max:.3e} input_local_max={input_local_max:.3e} "
            f"rs_local_max={rs_local_max:.3e} fs_group_size={fs_group_size} "
            f"fs_group_rank={fs_group_rank} entry_same={entry_param_same} entry_id={entry_param_id} "
            f"model_id={id(model_param)} entry_grad_ptr={entry_main_grad_ptr} "
            f"model_grad_ptr={model_main_grad_ptr} entry_grad_max={entry_main_grad_max:.3e}"
        )

    return warned
