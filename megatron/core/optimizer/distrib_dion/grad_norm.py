"""Dion grad norm helpers."""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.distributed as dist

from ..dion.runtime import replicate_reduce_op
from ... import tensor_parallel
from ...utils import get_data_parallel_group_if_dtensor, to_local_if_dtensor


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


def grad_norm_inputs(optimizer) -> List[torch.Tensor]:
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
    grads_for_norm = grad_norm_inputs(optimizer)
    total_sq = torch.zeros(1, dtype=torch.float64, device=torch.cuda.current_device())
    data_parallel_group = None
    for grad in grads_for_norm:
        data_parallel_group = get_data_parallel_group_if_dtensor(grad, data_parallel_group)
        local_grad = to_local_if_dtensor(grad).detach()
        total_sq += local_grad.to(dtype=torch.float64).pow(2).sum()

    if data_parallel_group is not None:
        torch.distributed.all_reduce(
            total_sq, op=torch.distributed.ReduceOp.SUM, group=data_parallel_group
        )
    torch.distributed.all_reduce(
        total_sq, op=torch.distributed.ReduceOp.SUM, group=optimizer.get_grad_stats_parallel_group()
    )
    return float(total_sq.sqrt().item())
