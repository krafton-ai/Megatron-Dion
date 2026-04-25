"""Dion grad norm helpers."""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.distributed as dist

from ..dion.runtime import replicate_reduce_op
from ... import tensor_parallel
from ..dion.dense_grad_cache import (
    dense_cache_entries as _dense_cache_entries,
    dense_cache_state as _dense_cache_state,
    mark_dense_grad_reduced as _mark_dense_grad_reduced,
)
from ...utils import get_data_parallel_group_if_dtensor, to_local_if_dtensor


_GRAD_NORM_FP64_CHUNK_BYTES = 128 * 1024 * 1024


def _base_optimizer(optimizer):
    return getattr(optimizer, "optimizer", optimizer)


def _dist_meta_for_shard(optimizer, shard_param):
    for source in (optimizer, _base_optimizer(optimizer)):
        dist_metas = getattr(source, "dist_metas", None)
        if dist_metas is None:
            continue
        dist_meta = dist_metas.get(shard_param, None)
        if dist_meta is not None:
            return dist_meta
    return None


def _can_reuse_dense_rp_grad(optimizer, shard_param) -> bool:
    dist_meta = _dist_meta_for_shard(optimizer, shard_param)
    if dist_meta is None:
        return False
    if getattr(dist_meta, "qkv_split_shapes", None) is not None:
        return False
    if getattr(dist_meta, "linear_split_rows", None) is not None:
        return False
    config = getattr(dist_meta, "param_config", None)
    if config is None:
        return False
    return not bool(getattr(config, "use_low_rank_sync", False))


def _grad_sum_sq_fp64(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.detach()
    total_sq = torch.zeros(1, dtype=torch.float64, device=tensor.device)
    numel = int(tensor.numel())
    if numel <= 0:
        return total_sq

    flat_tensor = tensor.view(-1) if tensor.is_contiguous() else tensor.reshape(-1)
    max_chunk_numel = max(1, int(_GRAD_NORM_FP64_CHUNK_BYTES) // 8)
    for start in range(0, numel, max_chunk_numel):
        chunk_numel = min(max_chunk_numel, numel - start)
        chunk = flat_tensor.narrow(0, start, chunk_numel).to(dtype=torch.float64)
        chunk.mul_(chunk)
        total_sq += chunk.sum()
    return total_sq


def dion_replica_grads(
    optimizer,
    dion_params: List[Tuple[torch.nn.Parameter, torch.nn.Parameter]],
    count_dion_grad: bool,
) -> List[torch.Tensor]:
    if not dion_params:
        return []

    dion_params = sorted(
        dion_params,
        key=lambda pair: repr(optimizer._shard_param_uid(pair[1])),
    )
    local_grads = [
        optimizer._get_local_grad(model_param, shard_param)
        for model_param, shard_param in dion_params
    ]
    replica_group = optimizer._get_replicate_group() if dist.is_initialized() else None
    if replica_group is None or optimizer._group_size(replica_group) <= 1:
        return local_grads if count_dion_grad else []

    groups_by_dtype_device = {}
    for index, grad in enumerate(local_grads):
        groups_by_dtype_device.setdefault((grad.dtype, grad.device), []).append(index)

    replica_grads = {}
    for indices in groups_by_dtype_device.values():
        dtype = local_grads[indices[0]].dtype
        device = local_grads[indices[0]].device
        total_numel = sum(int(local_grads[index].numel()) for index in indices)
        if total_numel <= 0:
            continue
        flat_grad = torch.empty(total_numel, dtype=dtype, device=device)
        cursor = 0
        for index in indices:
            grad = local_grads[index].detach()
            next_cursor = cursor + int(grad.numel())
            flat_grad[cursor:next_cursor].copy_(grad.reshape(-1))
            cursor = next_cursor

        dist.all_reduce(flat_grad, op=replicate_reduce_op(optimizer.optimizer), group=replica_group)

        if not count_dion_grad:
            continue
        cursor = 0
        for index in indices:
            grad = local_grads[index]
            next_cursor = cursor + int(grad.numel())
            replica_grads[index] = flat_grad[cursor:next_cursor].view_as(grad)
            cursor = next_cursor

    if not count_dion_grad:
        return []
    return [
        replica_grads[index]
        for index in range(len(local_grads))
        if index in replica_grads
    ]


def _dion_grad_norm_sq(
    optimizer,
    dion_params: List[Tuple[torch.nn.Parameter, torch.nn.Parameter]],
    count_dion_grad: bool,
) -> torch.Tensor | None:
    """Return exact Dion grad-norm contribution without materializing reduced views."""
    if not dion_params:
        return None

    dion_params = sorted(
        dion_params,
        key=lambda pair: repr(optimizer._shard_param_uid(pair[1])),
    )
    local_grads = [
        optimizer._get_local_grad(model_param, shard_param)
        for model_param, shard_param in dion_params
    ]
    dense_reuse = [
        _can_reuse_dense_rp_grad(optimizer, shard_param)
        for _, shard_param in dion_params
    ]
    replica_group = optimizer._get_replicate_group() if dist.is_initialized() else None
    if replica_group is None or optimizer._group_size(replica_group) <= 1:
        if not count_dion_grad:
            return None
        total_sq = torch.zeros(1, dtype=torch.float64, device=local_grads[0].device)
        for grad in local_grads:
            total_sq += _grad_sum_sq_fp64(grad)
        return total_sq

    total_sq_by_device = {}
    groups_by_dtype_device = {}
    for index, grad in enumerate(local_grads):
        groups_by_dtype_device.setdefault((grad.dtype, grad.device), []).append(index)

    base_optimizer = _base_optimizer(optimizer)
    reduce_op = replicate_reduce_op(base_optimizer)
    before_step = int(getattr(base_optimizer, "_step_count", 0))
    _dense_cache_entries(base_optimizer, before_step, create=True)

    for indices in groups_by_dtype_device.values():
        reduce_indices = []
        for index in indices:
            if dense_reuse[index]:
                state = _dense_cache_state(
                    base_optimizer,
                    local_grads[index],
                    replica_group=replica_group,
                    op=reduce_op,
                    before_step=before_step,
                )
                if state == "match":
                    if count_dion_grad:
                        total_sq = total_sq_by_device.get(local_grads[index].device)
                        if total_sq is None:
                            total_sq = torch.zeros(
                                1, dtype=torch.float64, device=local_grads[index].device
                            )
                            total_sq_by_device[local_grads[index].device] = total_sq
                        total_sq += _grad_sum_sq_fp64(local_grads[index])
                    continue
                if state == "mismatch":
                    raise RuntimeError(
                        "[DION_DENSE_RP_GRAD_CACHE_MISMATCH] "
                        f"param_uid={optimizer._shard_param_uid(dion_params[index][1])}"
                    )
            reduce_indices.append(index)

        if not reduce_indices:
            continue
        dtype = local_grads[indices[0]].dtype
        device = local_grads[indices[0]].device
        total_numel = sum(int(local_grads[index].numel()) for index in reduce_indices)
        if total_numel <= 0:
            continue
        flat_grad = torch.empty(total_numel, dtype=dtype, device=device)
        cursor = 0
        for index in reduce_indices:
            grad = local_grads[index].detach()
            next_cursor = cursor + int(grad.numel())
            flat_grad[cursor:next_cursor].copy_(grad.reshape(-1))
            cursor = next_cursor

        dist.all_reduce(flat_grad, op=reduce_op, group=replica_group)
        if count_dion_grad:
            total_sq = total_sq_by_device.get(device)
            if total_sq is None:
                total_sq = torch.zeros(1, dtype=torch.float64, device=device)
                total_sq_by_device[device] = total_sq
            total_sq += _grad_sum_sq_fp64(flat_grad)
        cursor = 0
        for index in reduce_indices:
            grad = local_grads[index]
            next_cursor = cursor + int(grad.numel())
            if dense_reuse[index]:
                grad.copy_(flat_grad[cursor:next_cursor].view_as(grad))
                _mark_dense_grad_reduced(
                    base_optimizer,
                    grad,
                    replica_group=replica_group,
                    op=reduce_op,
                    before_step=before_step,
                )
            cursor = next_cursor

    if not count_dion_grad or not total_sq_by_device:
        return None

    first_total = None
    for total_sq in total_sq_by_device.values():
        if first_total is None:
            first_total = total_sq
        else:
            first_total += total_sq.to(device=first_total.device)
    return first_total


def contributes_dion_grad(optimizer) -> bool:
    if not dist.is_initialized():
        return True
    replica_group = optimizer._get_replicate_group()
    if replica_group is None or optimizer._group_size(replica_group) <= 1:
        return True

    replica_ranks = tuple(int(rank) for rank in dist.get_process_group_ranks(replica_group))
    stats_group = optimizer.get_grad_stats_parallel_group()
    if stats_group is None:
        stats_rank_set = set(int(rank) for rank in range(dist.get_world_size()))
    else:
        stats_rank_set = set(int(rank) for rank in dist.get_process_group_ranks(stats_group))

    replica_ranks_in_stats = tuple(rank for rank in replica_ranks if rank in stats_rank_set)
    if len(replica_ranks_in_stats) <= 1:
        return True
    return int(dist.get_rank()) == int(replica_ranks_in_stats[0])


def _grad_norm_routes(optimizer):
    from ...transformer.module import param_is_not_shared

    main_grad_view_by_param_id = {}
    params = []
    dion_params = []
    seen_param_ids = set()
    for model_group, shard_group in zip(
        optimizer.model_float16_groups + optimizer.model_fp32_groups,
        optimizer.shard_fp32_from_float16_groups + optimizer.shard_fp32_groups,
    ):
        if len(model_group) != len(shard_group):
            raise RuntimeError(
                "[Dion] grad norm route requires equal model/shard group lengths "
                f"model_len={len(model_group)} shard_len={len(shard_group)}"
            )
        for model_param, shard_param in zip(model_group, shard_group):
            if shard_param is None:
                continue
            shard_param_id = id(shard_param)
            if shard_param_id in seen_param_ids:
                continue
            seen_param_ids.add(shard_param_id)
            params.append(shard_param)
            if getattr(model_param, "is_dion_param", False):
                continue
            model_grad = getattr(model_param, "main_grad", None)
            if model_grad is None:
                continue
            param_range = optimizer._get_model_param_range_map(model_param)["param"]
            if param_range.size == 0:
                continue
            main_grad_view_by_param_id[shard_param_id] = model_grad.view(-1)[
                int(param_range.start) : int(param_range.end)
            ]
    grads_for_norm = []
    count_dion_grad = contributes_dion_grad(optimizer)

    def _stats_grad(param):
        main_grad_view = main_grad_view_by_param_id.get(id(param), None)
        if main_grad_view is not None:
            return main_grad_view
        model_param = getattr(param, "_model_param", None)
        if model_param is not None and getattr(model_param, "is_dion_param", False):
            return None
        if getattr(param, "__fsdp_param__", False):
            return param.grad._local_tensor if param.grad is not None else None
        if optimizer.config.use_precision_aware_optimizer_no_fp8_or_ds_fp8:
            return param.decoupled_grad if hasattr(param, "decoupled_grad") else None
        return param.grad

    for param in params:
        grad = _stats_grad(param)
        is_not_shared = param_is_not_shared(param)
        is_not_tp_duplicate = tensor_parallel.param_is_not_tensor_parallel_duplicate(
            param, optimizer._resolve_dion_tp_group()
        )
        model_param = getattr(param, "_model_param", None)
        if model_param is not None and getattr(model_param, "is_dion_param", False):
            if is_not_shared and is_not_tp_duplicate:
                dion_params.append((model_param, param))
            continue

        if grad is not None and is_not_shared and is_not_tp_duplicate:
            grads_for_norm.append(grad)

    return grads_for_norm, dion_params, count_dion_grad


def grad_norm_inputs(optimizer) -> List[torch.Tensor]:
    grads_for_norm, dion_params, count_dion_grad = _grad_norm_routes(optimizer)
    grads_for_norm.extend(
        dion_replica_grads(
            optimizer,
            dion_params=dion_params,
            count_dion_grad=count_dion_grad,
        )
    )
    return grads_for_norm


@torch.no_grad()
def compute_grad_norm(optimizer):
    grads_for_norm, dion_params, count_dion_grad = _grad_norm_routes(optimizer)
    total_sq = torch.zeros(1, dtype=torch.float64, device=torch.cuda.current_device())
    data_parallel_group = None
    for grad in grads_for_norm:
        data_parallel_group = get_data_parallel_group_if_dtensor(grad, data_parallel_group)
        local_grad = to_local_if_dtensor(grad).detach()
        total_sq += _grad_sum_sq_fp64(local_grad)

    dion_sq = _dion_grad_norm_sq(
        optimizer,
        dion_params=dion_params,
        count_dion_grad=count_dion_grad,
    )
    if dion_sq is not None:
        total_sq += dion_sq.to(device=total_sq.device)

    if data_parallel_group is not None:
        torch.distributed.all_reduce(
            total_sq, op=torch.distributed.ReduceOp.SUM, group=data_parallel_group
        )
    torch.distributed.all_reduce(
        total_sq, op=torch.distributed.ReduceOp.SUM, group=optimizer.get_grad_stats_parallel_group()
    )
    return float(total_sq.sqrt().item())
