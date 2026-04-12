"""Package-private distributed runtime helpers for Dion."""
from typing import Callable, Generator, List, Optional

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
from torch import Tensor

from .async_runtime import AsyncTask
from .kernels import local_column_sum_sq, normalize_columns
from .types import DionBatchCollectives, DionBatchRoute, DionBatch, DionParamConfig

_REPLICATE_GROUP_UNSET = object()


def resolve_async_task_limit(
    *,
    max_concurrent_tasks: Optional[int],
    task_count: int,
) -> int:
    """Resolve async runtime width without relying on hidden global defaults."""
    if task_count <= 0:
        raise RuntimeError(f"[Dion] invalid async task_count={task_count}")
    if max_concurrent_tasks is None:
        return task_count
    if int(max_concurrent_tasks) <= 0:
        raise RuntimeError(
            f"[Dion] invalid max_concurrent_tasks={max_concurrent_tasks}"
        )
    return min(task_count, int(max_concurrent_tasks))


def iter_dist_tasks(optimizer) -> Generator[AsyncTask, None, None]:
    """Create async tasks for distributed Dion execution."""
    route_step_params_fn = getattr(
        optimizer,
        "_route_step_params_fn",
        None,
    )
    if route_step_params_fn is None:
        raise RuntimeError(
            "[DION_MISSING_DIST_STEP_ITEMS_CALLBACK] "
            f"step={optimizer._step_count} rank={optimizer._global_rank}"
        )

    dion_batches, scalar_params = route_step_params_fn()
    optimizer._dion_update_count += sum(int(batch.real_batch_size) for batch in dion_batches)
    optimizer._adamw_update_count += len(scalar_params)

    if dion_batches:
        yield AsyncTask(run_dion_batch_async(optimizer, dion_batches))

    if scalar_params:
        yield AsyncTask(optimizer._run_scalar_bucket_async(scalar_params))


def run_dion_batch_async(
    optimizer,
    dion_batches: List[DionBatch],
) -> Generator[None, None, None]:
    """Process adapter-authored Dion batches with async operations."""
    if not dion_batches:
        return

    for dion_batch in dion_batches:
        batch_route = dion_batch.batch_route

        params = list(dion_batch.params or [])
        momentums = list(dion_batch.momentums or [])
        q_tensors = list(dion_batch.q_tensors or [])
        configs = list(dion_batch.configs or [])
        dist_metas = list(dion_batch.dist_metas or [])
        optim_groups = list(dion_batch.optim_groups or [])
        grads_to_process = list(dion_batch.grads or [])
        optimizer_states = list(dion_batch.optimizer_states or [])
        param_shapes = list(dion_batch.param_shapes)
        real_batch_size = int(dion_batch.real_batch_size)
        batch_collectives = dion_batch.batch_collectives
        global_param_offset = int(dion_batch.global_param_offset)

        if params:
            yield from optimizer._batch_dion_update_async(
                params,
                momentums,
                q_tensors,
                configs,
                dist_metas,
                optim_groups,
                grads_to_process,
                optimizer_states,
                param_shapes,
                real_batch_size,
                global_param_offset,
                batch_route,
                batch_collectives,
            )


def replicate_reduce_op(optimizer):
    """Replicate-domain collapse must match `dion_reference.py` semantics."""
    del optimizer
    return dist.ReduceOp.AVG


def collapse_grads_across_replicas(
    optimizer,
    grads: List[Tensor],
    *,
    replicate_group=_REPLICATE_GROUP_UNSET,
) -> Generator[None, None, None]:
    """Collapse true Dion replicate replicas on G for non-compressed batches."""
    if replicate_group is _REPLICATE_GROUP_UNSET:
        raise RuntimeError(
            "[DION_MISSING_REPLICATE_GROUP_BINDING] "
            f"step={optimizer._step_count} rank={optimizer._global_rank}"
        )
    if replicate_group is None or not grads:
        return

    op = replicate_reduce_op(optimizer)
    for grad in grads:
        if grad.is_contiguous():
            dist.all_reduce(grad, op=op, group=replicate_group)
            continue

        reduced_grad = grad.contiguous()
        dist.all_reduce(reduced_grad, op=op, group=replicate_group)
        grad.copy_(reduced_grad)
    yield


def collapse_grads_over_replicate_subset(
    optimizer,
    grads: List[Tensor],
    *,
    primary_group: torch.distributed.ProcessGroup,
    subset_ranks: List[int],
) -> Generator[None, None, None]:
    """Average `grads` across a deterministic subset of an existing group."""
    del optimizer
    if primary_group is None:
        raise RuntimeError("[DION_INVALID_PRIMARY_GROUP_FOR_SUBSET_COLLAPSE]")
    if not subset_ranks or not grads:
        return

    primary_ranks = dist.get_process_group_ranks(primary_group)
    rank_to_index = {rank: idx for idx, rank in enumerate(primary_ranks)}
    subset_indices = [rank_to_index[rank] for rank in subset_ranks]
    index_tensor = None

    for grad in grads:
        gathered = funcol.all_gather_tensor(
            grad.contiguous(),
            gather_dim=0,
            group=primary_group,
        )
        yield

        gathered = gathered.view(len(primary_ranks), *grad.shape)
        if index_tensor is None or index_tensor.device != grad.device:
            index_tensor = torch.tensor(
                subset_indices,
                device=grad.device,
                dtype=torch.long,
            )
        reduced = gathered.index_select(0, index_tensor).mean(dim=0)
        grad.copy_(reduced)


def collapse_batch_across_replicas(
    optimizer,
    batch: Tensor,
    *,
    replicate_group=_REPLICATE_GROUP_UNSET,
) -> Generator[None, None, None]:
    """Collapse true Dion replicate replicas on a batched tensor."""
    if replicate_group is _REPLICATE_GROUP_UNSET:
        raise RuntimeError(
            "[DION_MISSING_REPLICATE_GROUP_BINDING] "
            f"step={optimizer._step_count} rank={optimizer._global_rank}"
        )
    if replicate_group is None:
        return

    dist.all_reduce(batch, op=replicate_reduce_op(optimizer), group=replicate_group)
    yield


def collapse_batch_over_replicate_subset(
    optimizer,
    batch: Tensor,
    *,
    primary_group: torch.distributed.ProcessGroup,
    subset_ranks: List[int],
) -> Generator[None, None, None]:
    """Average `batch` across a deterministic subset of an existing group."""
    del optimizer
    if primary_group is None:
        raise RuntimeError("[DION_INVALID_PRIMARY_GROUP_FOR_SUBSET_COLLAPSE]")
    if not subset_ranks:
        raise RuntimeError("[DION_EMPTY_SUBSET_FOR_SUBSET_COLLAPSE]")

    primary_ranks = dist.get_process_group_ranks(primary_group)
    rank_to_index = {rank: idx for idx, rank in enumerate(primary_ranks)}
    subset_indices = [rank_to_index[rank] for rank in subset_ranks]

    gathered = funcol.all_gather_tensor(
        batch.contiguous(),
        gather_dim=0,
        group=primary_group,
    )
    yield

    gathered = gathered.view(len(primary_ranks), *batch.shape)
    index_tensor = torch.tensor(
        subset_indices,
        device=batch.device,
        dtype=torch.long,
    )
    reduced = gathered.index_select(0, index_tensor).mean(dim=0)
    batch.copy_(reduced)


def enable_distributed_mode(
    optimizer,
    *,
    route_step_params_fn: Optional[Callable] = None,
) -> None:
    """Apply distributed bootstrap to MegatronDion."""
    optimizer.is_distributed_mode = True
    if route_step_params_fn is None:
        raise RuntimeError(
            "[DION_MISSING_DIST_STEP_ITEMS_CALLBACK] "
            f"step={optimizer._step_count} rank={optimizer._global_rank}"
        )
    optimizer._route_step_params_fn = route_step_params_fn


def unshard_q_batch(
    optimizer,
    Qs: List[Tensor],
    configs: List[DionParamConfig],
    dist_metas: Optional[List] = None,
    batch_collectives: Optional[DionBatchCollectives] = None,
) -> Generator[List[Tensor], None, None]:
    """All-gather TP-sharded Q blocks needed for local matmul."""
    batch_size = len(Qs)
    q_for_matmul: List[Optional[Tensor]] = [None] * batch_size
    if batch_collectives is None:
        raise RuntimeError(
            "[DION_MISSING_BATCH_AXIS_PLAN] "
            f"step={optimizer._step_count} rank={optimizer._global_rank}"
        )
    tp_unshard_indices = {
        idx for plan in batch_collectives.tp_q_gathers for idx in plan.indices
    }

    for i in range(batch_size):
        if i not in tp_unshard_indices:
            q_for_matmul[i] = Qs[i]

    if batch_collectives.tp_q_gathers:
        pending = []
        for group_seq, plan in enumerate(batch_collectives.tp_q_gathers):
            indices = list(plan.indices)
            tp_group = plan.process_group
            if tp_group is None:
                raise RuntimeError(
                    "[DION_MISSING_TP_GROUP_FOR_Q_UNSHARD] "
                    f"step={optimizer._step_count} rank={optimizer._global_rank} indices={indices}"
                )
            if not indices:
                continue
            q0 = Qs[indices[0]]
            n = q0.size(0)
            r_local = q0.size(1)
            dtype = q0.dtype
            device = q0.device
            for idx in indices[1:]:
                q_local = Qs[idx]
                if (
                    q_local.size(0) != n
                    or q_local.size(1) != r_local
                    or q_local.dtype != dtype
                    or q_local.device != device
                ):
                    raise RuntimeError(
                        "[DION_INCONSISTENT_Q_UNSHARD_PLAN] "
                        f"step={optimizer._step_count} rank={optimizer._global_rank} indices={indices} "
                        f"first_shape={(n, r_local)} first_dtype={dtype} idx={idx} "
                        f"local_shape={tuple(q_local.shape)} local_dtype={q_local.dtype}"
                    )
            tp_size = plan.world_size
            local_batch = optimizer._cached_buffer(
                f"q_local_batch_{group_seq}",
                (len(indices), n, r_local),
                dtype,
                device,
            )
            for slot, idx in enumerate(indices):
                local_batch[slot].copy_(Qs[idx])

            gathered_batch = optimizer._cached_buffer(
                f"q_gather_batch_{group_seq}",
                (tp_size, len(indices), n, r_local),
                dtype,
                device,
            )
            handle = dist.all_gather_into_tensor(
                gathered_batch.view(-1),
                local_batch.view(-1),
                group=tp_group,
                async_op=True,
            )
            pending.append((indices, tp_size, n, r_local, local_batch, gathered_batch, handle))

        yield

        for indices, tp_size, n, r_local, local_batch, gathered_batch, handle in pending:
            handle.wait()
            q_full_batch = (
                gathered_batch.permute(1, 2, 0, 3)
                .contiguous()
                .view(len(indices), n, tp_size * r_local)
            )
            for slot, idx in enumerate(indices):
                q_for_matmul[idx] = q_full_batch[slot]
            del local_batch

    return [q for q in q_for_matmul]


def reduce_p_across_fs_groups(
    optimizer,
    P_batch: Tensor,
    configs: List[DionParamConfig],
    dist_metas: List,
    batch_collectives: Optional[DionBatchCollectives] = None,
) -> Generator[None, None, None]:
    """Reduce STEP 3.5 P batches across the correct FS shard groups."""
    del dist_metas
    if not optimizer.use_fs_collectives:
        return

    if batch_collectives is None:
        raise RuntimeError(
            "[DION_MISSING_BATCH_AXIS_PLAN] "
            f"step={optimizer._step_count} rank={optimizer._global_rank}"
        )
    if not batch_collectives.fs_p_collectives:
        return

    did_yield = False
    batch_size = len(configs)
    for plan in batch_collectives.fs_p_collectives:
        process_group = plan.process_group
        indices = list(plan.indices)
        if process_group and plan.world_size > 1:
            if len(indices) == batch_size and len(batch_collectives.fs_p_collectives) == 1:
                dist.all_reduce(P_batch, op=dist.ReduceOp.SUM, group=process_group)
            else:
                tensors = [P_batch[idx] for idx in indices]
                reduced = funcol.all_reduce_coalesced(
                    tensors,
                    reduceOp="sum",
                    group=process_group,
                )
                for idx, tensor in zip(indices, reduced):
                    P_batch[idx].copy_(tensor)
                del tensors, reduced
            if not did_yield:
                yield
                did_yield = True

    if not did_yield:
        yield


def reduce_r_across_tp(
    optimizer,
    R_batch: Tensor,
    configs: List[DionParamConfig],
    dist_metas: Optional[List] = None,
    batch_collectives: Optional[DionBatchCollectives] = None,
) -> Generator[None, None, None]:
    """Reduce STEP 5 R batches across TP when required."""
    del configs, dist_metas
    if batch_collectives is None:
        raise RuntimeError(
            "[DION_MISSING_BATCH_AXIS_PLAN] "
            f"step={optimizer._step_count} rank={optimizer._global_rank}"
        )
    if not batch_collectives.tp_r_collectives:
        return

    for plan in batch_collectives.tp_r_collectives:
        tp_group = plan.process_group
        need_tp_r = list(plan.indices)
        if len(need_tp_r) == R_batch.size(0) and len(batch_collectives.tp_r_collectives) == 1:
            dist.all_reduce(R_batch, op=dist.ReduceOp.SUM, group=tp_group)
        else:
            tensors = [R_batch[i] for i in need_tp_r]
            reduced = funcol.all_reduce_coalesced(tensors, reduceOp="sum", group=tp_group)
            for i, tensor in zip(need_tp_r, reduced):
                R_batch[i].copy_(tensor)
            del tensors, reduced

    yield


def normalize_cols_async(
    optimizer,
    R_batch: Tensor,
    configs: List[DionParamConfig],
    dist_metas: List,
    real_batch_size: int = None,
    global_param_offset: int = 0,
    batch_route: Optional[DionBatchRoute] = None,
) -> Generator[Tensor, None, None]:
    """Async batch column normalization with yields for communication."""
    del configs, global_param_offset
    batch_size = R_batch.shape[0]
    if real_batch_size is None:
        real_batch_size = batch_size

    result = torch.empty_like(R_batch)
    epsilon = optimizer.defaults['epsilon']
    if real_batch_size <= 0:
        return result

    col_sum_sq_real = local_column_sum_sq(R_batch[:real_batch_size])

    if batch_route is None:
        raise RuntimeError(
            "[DION_MISSING_Q_NORM_BINDING] "
            f"step={optimizer._step_count} rank={optimizer._global_rank}"
        )
    del dist_metas
    q_norm_group = batch_route.q_norm_group

    if q_norm_group is not None:
        dist.all_reduce(col_sum_sq_real, op=dist.ReduceOp.SUM, group=q_norm_group)
        yield

    q_new_real, post_col_sum_sq_real = normalize_columns(
        R_batch[:real_batch_size],
        col_sum_sq_real,
        epsilon=epsilon,
    )
    result[:real_batch_size].copy_(q_new_real)
    if q_norm_group is not None:
        dist.all_reduce(post_col_sum_sq_real, op=dist.ReduceOp.SUM, group=q_norm_group)
        yield

    if real_batch_size < batch_size:
        result[real_batch_size:].copy_(R_batch[real_batch_size:])

    return result
