"""Package-private distributed runtime helpers for Dion."""
import logging
from typing import Callable, Generator, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
from torch import Tensor

from .kernels import (
    apply_error_feedback,
    compute_update_batch,
    fix_all_zero_or_nan,
    local_column_sum_sq,
    normalize_columns,
    scaled_lr_for_shape,
)
from .ortho import (
    _dion_math_precision_context,
    distributed_orthogonalize,
    make_sketch,
    orthogonalize,
    reshard_q_along_tp,
)
from .state import has_tp_shard, is_fs_only_config
from .types import DionBatch, DionBatchCollectives, DionBatchGroup, DionParamConfig
from .utils import get_global_shape

_REPLICATE_GROUP_UNSET = object()
logger = logging.getLogger(__name__)


class AsyncTask:
    """Wrapper for async generator tasks to enable concurrent execution."""

    def __init__(self, generator: Generator[None, None, None]):
        self.generator = generator
        self.completed = False
        self._running = self.run()

    def run(self) -> bool:
        """Execute one step of the async task. Returns True while still running."""
        try:
            next(self.generator)
            return True
        except StopIteration:
            self.completed = True
            if self.generator is not None:
                self.generator.close()
                self.generator = None
            return False


class AsyncRuntime:
    """Runtime for managing and executing async tasks concurrently."""

    def __init__(self, tasks: Generator[AsyncTask, None, None], max_concurrent_tasks: int):
        if int(max_concurrent_tasks) <= 0:
            raise ValueError(f"Invalid max_concurrent_tasks={max_concurrent_tasks}")
        self.tasks = tasks
        self.max_concurrent = int(max_concurrent_tasks)

    def run(self):
        """Execute all tasks with controlled concurrency."""
        have_new_tasks = True
        previous_tasks: List[AsyncTask] = []
        task_iter = iter(self.tasks)

        while have_new_tasks or previous_tasks:
            running_tasks: List[AsyncTask] = []

            if have_new_tasks and len(previous_tasks) < self.max_concurrent:
                try:
                    new_task = next(task_iter)
                except StopIteration:
                    have_new_tasks = False
                else:
                    if new_task._running:
                        running_tasks.append(new_task)

            for task in previous_tasks:
                if task.run():
                    running_tasks.append(task)

            previous_tasks = running_tasks


def resolve_async_task_limit(
    *,
    max_concurrent_tasks: Optional[int],
    task_count: Optional[int] = None,
) -> int:
    """Resolve async runtime width without relying on hidden global defaults."""
    if max_concurrent_tasks is None:
        # Reference Dion hard-codes AsyncRuntime(..., max_concurrent_tasks=3).
        limit = 3
    else:
        if int(max_concurrent_tasks) <= 0:
            raise RuntimeError(
                f"[Dion] invalid max_concurrent_tasks={max_concurrent_tasks}"
            )
        limit = int(max_concurrent_tasks)
    if task_count is None:
        return limit
    if task_count <= 0:
        raise RuntimeError(f"[Dion] invalid async task_count={task_count}")
    return min(task_count, limit)


def _collective_signature(collectives) -> tuple:
    return tuple(
        (
            id(collective.process_group),
            int(collective.world_size),
            int(collective.rank),
            tuple(int(index) for index in collective.indices),
        )
        for collective in collectives
    )


def _fsdp_tp_compressed_run_key(optimizer, dion_batch: DionBatch) -> Optional[tuple]:
    batch_group = dion_batch.batch_group
    batch_collectives = dion_batch.batch_collectives
    if batch_group is None or batch_collectives is None:
        return None
    if batch_group.kernel_kind != "fsdp_tp":
        return None
    replicate_group = batch_group.replicate_group
    if replicate_group is None or dist.get_world_size(replicate_group) <= 1:
        return None
    if int(dion_batch.real_batch_size) != int(batch_group.batch_world_size):
        return None
    configs = [config for config in dion_batch.configs if config is not None]
    if not configs:
        return None
    if not optimizer.use_compressed_comm or not all(
        bool(config.compressed_all_reduce) for config in configs
    ):
        return None
    ortho_group = batch_group.ortho_group
    return (
        dion_batch.batch_key,
        id(replicate_group),
        dist.get_world_size(replicate_group),
        dist.get_rank(replicate_group),
        id(ortho_group),
        1 if ortho_group is None else dist.get_world_size(ortho_group),
        0 if ortho_group is None else dist.get_rank(ortho_group),
        int(batch_group.batch_world_size),
        int(dion_batch.real_batch_size),
        _collective_signature(batch_collectives.tp_q_gathers),
        _collective_signature(batch_collectives.fs_p_collectives),
        _collective_signature(batch_collectives.tp_r_collectives),
    )


def _iter_coalesced_dion_batch_runs(optimizer, dion_batches: List[DionBatch]):
    current_run: List[DionBatch] = []
    current_key = None
    for dion_batch in dion_batches:
        run_key = _fsdp_tp_compressed_run_key(optimizer, dion_batch)
        if (
            run_key is None
            or current_key is None
            or run_key != current_key
        ):
            if current_run:
                yield current_run
                current_run = []
            current_key = run_key
        if run_key is None:
            yield [dion_batch]
            current_key = None
            continue
        current_run.append(dion_batch)
    if current_run:
        yield current_run


def _split_stacked_batches(stacked_batch: Tensor, batch_sizes: List[int]) -> List[Tensor]:
    split_batches = []
    cursor = 0
    for batch_size in batch_sizes:
        next_cursor = cursor + int(batch_size)
        split_batches.append(stacked_batch[cursor:next_cursor].contiguous())
        cursor = next_cursor
    return split_batches


def _validate_batch_update_contract(
    optimizer,
    *,
    optim_groups: List[dict],
    optimizer_states: List[dict],
    dist_metas: List,
    param_shapes: List[Tuple[int, int]],
    real_batch_size: int,
) -> None:
    if real_batch_size <= 0:
        return

    update_contract_rows = []
    for index in range(real_batch_size):
        optimizer_state = optimizer_states[index]
        dist_meta = dist_metas[index]
        param_shape = param_shapes[index]
        true_global_shape = tuple(
            int(dim)
            for dim in optimizer_state.get(
                "true_global_shape",
                get_global_shape(dist_meta, param_shape[0], param_shape[1]),
            )
        )
        per_expert_global_shape = optimizer_state.get("per_expert_global_shape", None)
        if per_expert_global_shape is not None:
            per_expert_global_shape = tuple(int(dim) for dim in per_expert_global_shape)
            m_for_lr, n_for_lr = per_expert_global_shape
        else:
            m_for_lr, n_for_lr = true_global_shape

        optim_group = optim_groups[index]
        lr = float(optim_group.get("lr", optimizer.defaults["lr"]))
        wd_mult = float(optim_group.get("wd_mult", 1.0))
        weight_decay = float(
            optim_group.get("weight_decay", optimizer.defaults["weight_decay"] * wd_mult)
        )
        mu = float(optim_group.get("mu", optimizer.defaults["mu"]))
        update_contract_rows.append(
            {
                "batch_index": int(index),
                "param_uid": (
                    getattr(dist_meta, "param_uid", None) if dist_meta is not None else None
                ),
                "param_name": getattr(dist_meta, "param_name", "") if dist_meta is not None else "",
                "true_global_shape": true_global_shape,
                "per_expert_global_shape": per_expert_global_shape,
                "m_for_lr": int(m_for_lr),
                "n_for_lr": int(n_for_lr),
                "lr": lr,
                "weight_decay": weight_decay,
                "mu": mu,
                "r": int(optimizer_state.get("r", -1)),
            }
        )

    expected_contract = {
        key: update_contract_rows[0][key]
        for key in (
            "true_global_shape",
            "per_expert_global_shape",
            "m_for_lr",
            "n_for_lr",
            "lr",
            "weight_decay",
            "mu",
            "r",
        )
    }
    mismatched_rows = [
        row
        for row in update_contract_rows[1:]
        if any(row[key] != expected_contract[key] for key in expected_contract)
    ]
    if mismatched_rows:
        raise RuntimeError(
            "[DION_BATCH_UPDATE_CONTRACT_MISMATCH] "
            f"step={optimizer._step_count} rank={optimizer._global_rank} "
            f"expected={expected_contract} mismatched_rows={mismatched_rows}"
        )


def _all_reduce_stacked_collectives(
    stacked_batch: Tensor,
    *,
    template_collectives,
    per_batch_size: int,
    run_count: int,
    op,
) -> Generator[None, None, None]:
    if not template_collectives:
        yield
        return

    full_batch = int(per_batch_size) * int(run_count)
    did_yield = False
    for collective in template_collectives:
        process_group = collective.process_group
        if process_group is None or int(collective.world_size) <= 1:
            continue
        indices = []
        base_indices = [int(index) for index in collective.indices]
        for run_idx in range(int(run_count)):
            offset = run_idx * int(per_batch_size)
            indices.extend(offset + index for index in base_indices)
        if len(indices) == full_batch and len(template_collectives) == 1:
            handle = dist.all_reduce(
                stacked_batch,
                op=op,
                group=process_group,
                async_op=True,
            )
        else:
            tensors = [stacked_batch[index] for index in indices]
            handle = dist.all_reduce_coalesced(
                tensors,
                op=op,
                group=process_group,
                async_op=True,
            )
        yield
        handle.wait()
        if len(indices) != full_batch or len(template_collectives) != 1:
            del tensors
        did_yield = True

    if not did_yield:
        yield


def iter_dist_tasks(optimizer) -> Generator[AsyncTask, None, None]:
    """Create async tasks for distributed Dion execution."""
    route_step_params = getattr(
        optimizer,
        "_route_step_params",
        None,
    )
    if route_step_params is None:
        raise RuntimeError(
            "[DION_MISSING_DIST_STEP_ITEMS_CALLBACK] "
            f"step={optimizer._step_count} rank={optimizer._global_rank}"
        )

    dion_batches, scalar_params = route_step_params()
    optimizer._dion_update_count += sum(int(batch.real_batch_size) for batch in dion_batches)
    optimizer._scalar_update_count += len(scalar_params)

    for dion_batch in dion_batches:
        yield AsyncTask(run_dion_batch_async(optimizer, dion_batch))

    if scalar_params:
        yield AsyncTask(optimizer._apply_scalar_buckets(scalar_params))


def run_dion_batch_async(
    optimizer,
    dion_batch: DionBatch,
) -> Generator[None, None, None]:
    """Process one adapter-authored Dion batch with async operations."""
    if dion_batch is None:
        return

    batch_group = dion_batch.batch_group

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
        yield from batch_dion_update_async(
            optimizer,
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
            batch_group,
            batch_collectives,
            commit_updates=[entry.commit_update for entry in dion_batch.entries[:real_batch_size]],
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
            "[DION_MISSING_REPLICATE_GROUP] "
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


def collapse_batch_across_replicas(
    optimizer,
    batch: Tensor,
    *,
    replicate_group=_REPLICATE_GROUP_UNSET,
) -> Generator[Tensor, None, None]:
    """Collapse true Dion replicate replicas on a batched tensor."""
    if replicate_group is _REPLICATE_GROUP_UNSET:
        raise RuntimeError(
            "[DION_MISSING_REPLICATE_GROUP] "
            f"step={optimizer._step_count} rank={optimizer._global_rank}"
        )
    if replicate_group is None:
        return batch

    reduced_batch = funcol.all_reduce(
        batch,
        reduceOp="avg" if replicate_reduce_op(optimizer) == dist.ReduceOp.AVG else "sum",
        group=replicate_group,
    )
    yield
    return reduced_batch


def enable_distributed_mode(
    optimizer,
    *,
    route_step_params: Optional[Callable] = None,
) -> None:
    """Apply distributed bootstrap to MegatronDion."""
    optimizer.is_distributed_mode = True
    if route_step_params is None:
        raise RuntimeError(
            "[DION_MISSING_DIST_STEP_ITEMS_CALLBACK] "
            f"step={optimizer._step_count} rank={optimizer._global_rank}"
        )
    optimizer._route_step_params = route_step_params


def unshard_q_batch(
    optimizer,
    Qs: List[Tensor],
    configs: List[DionParamConfig],
    dist_metas: Optional[List] = None,
    batch_collectives: Optional[DionBatchCollectives] = None,
    cache_scope: Optional[int] = None,
) -> Generator[List[Tensor], None, None]:
    """All-gather TP-sharded Q blocks needed for local matmul."""
    del configs, dist_metas
    batch_size = len(Qs)
    q_for_matmul: List[Optional[Tensor]] = [None] * batch_size
    if batch_collectives is None:
        raise RuntimeError(
            "[DION_MISSING_BATCH_COLLECTIVES] "
            f"step={optimizer._step_count} rank={optimizer._global_rank}"
        )
    tp_unshard_indices = {
        idx for collective in batch_collectives.tp_q_gathers for idx in collective.indices
    }

    for i in range(batch_size):
        if i not in tp_unshard_indices:
            q_for_matmul[i] = Qs[i]

    if batch_collectives.tp_q_gathers:
        pending = []
        scope = "global" if cache_scope is None else str(int(cache_scope))
        for group_seq, collective in enumerate(batch_collectives.tp_q_gathers):
            indices = list(collective.indices)
            tp_group = collective.process_group
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
                        "[DION_INCONSISTENT_Q_UNSHARD_LAYOUT] "
                        f"step={optimizer._step_count} rank={optimizer._global_rank} indices={indices} "
                        f"first_shape={(n, r_local)} first_dtype={dtype} idx={idx} "
                        f"local_shape={tuple(q_local.shape)} local_dtype={q_local.dtype}"
                    )
            tp_size = collective.world_size
            local_batch = optimizer._cached_buffer(
                f"q_local_batch_{scope}_{group_seq}",
                (len(indices), n, r_local),
                dtype,
                device,
            )
            for local_index, idx in enumerate(indices):
                local_batch[local_index].copy_(Qs[idx])

            gathered_batch = optimizer._cached_buffer(
                f"q_gather_batch_{scope}_{group_seq}",
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
            for local_index, idx in enumerate(indices):
                q_for_matmul[idx] = q_full_batch[local_index]
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
            "[DION_MISSING_BATCH_COLLECTIVES] "
            f"step={optimizer._step_count} rank={optimizer._global_rank}"
        )
    if not batch_collectives.fs_p_collectives:
        return

    batch_size = len(configs)
    for collective in batch_collectives.fs_p_collectives:
        process_group = collective.process_group
        indices = list(collective.indices)
        if process_group and collective.world_size > 1:
            if len(indices) == batch_size and len(batch_collectives.fs_p_collectives) == 1:
                handle = dist.all_reduce(
                    P_batch,
                    op=dist.ReduceOp.SUM,
                    group=process_group,
                    async_op=True,
                )
            else:
                tensors = [P_batch[idx] for idx in indices]
                handle = dist.all_reduce_coalesced(
                    tensors,
                    op=dist.ReduceOp.SUM,
                    group=process_group,
                    async_op=True,
                )
            yield
            handle.wait()
            if len(indices) != batch_size or len(batch_collectives.fs_p_collectives) != 1:
                del tensors
    yield


def reduce_r_across_tp(
    optimizer,
    R_batch: Tensor,
    configs: List[DionParamConfig],
    dist_metas: Optional[List] = None,
    batch_collectives: Optional[DionBatchCollectives] = None,
) -> Generator[None, None, None]:
    """Reduce STEP 5 R batches across TP when TP reduction is enabled."""
    del configs, dist_metas
    if batch_collectives is None:
        raise RuntimeError(
            "[DION_MISSING_BATCH_COLLECTIVES] "
            f"step={optimizer._step_count} rank={optimizer._global_rank}"
        )
    if not batch_collectives.tp_r_collectives:
        return

    for collective in batch_collectives.tp_r_collectives:
        tp_group = collective.process_group
        need_tp_r = list(collective.indices)
        if len(need_tp_r) == R_batch.size(0) and len(batch_collectives.tp_r_collectives) == 1:
            handle = dist.all_reduce(
                R_batch,
                op=dist.ReduceOp.SUM,
                group=tp_group,
                async_op=True,
            )
        else:
            tensors = [R_batch[i] for i in need_tp_r]
            handle = dist.all_reduce_coalesced(
                tensors,
                op=dist.ReduceOp.SUM,
                group=tp_group,
                async_op=True,
            )
        yield
        handle.wait()
        if len(need_tp_r) != R_batch.size(0) or len(batch_collectives.tp_r_collectives) != 1:
            del tensors
    yield


def normalize_cols_async(
    optimizer,
    R_batch: Tensor,
    configs: List[DionParamConfig],
    dist_metas: List,
    real_batch_size: int = None,
    global_param_offset: int = 0,
    batch_group: Optional[DionBatchGroup] = None,
) -> Generator[Tensor, None, None]:
    """Async batch column normalization with yields for communication."""
    batch_size = R_batch.shape[0]
    if real_batch_size is None:
        real_batch_size = batch_size

    result = torch.empty_like(R_batch)
    epsilon = optimizer.defaults['epsilon']
    if real_batch_size <= 0:
        return result

    col_sum_sq_real = local_column_sum_sq(R_batch[:real_batch_size])

    if batch_group is None:
        raise RuntimeError(
            "[DION_MISSING_Q_NORM_GROUP] "
            f"step={optimizer._step_count} rank={optimizer._global_rank}"
        )
    q_norm_group = batch_group.q_norm_group

    if q_norm_group is not None:
        col_sum_sq_real = funcol.all_reduce(
            col_sum_sq_real,
            reduceOp="sum",
            group=q_norm_group,
        )
        yield

    q_new_real, _ = normalize_columns(
        R_batch[:real_batch_size],
        col_sum_sq_real,
        epsilon=epsilon,
    )
    del dist_metas
    result[:real_batch_size].copy_(q_new_real)

    if real_batch_size < batch_size:
        result[real_batch_size:].copy_(R_batch[real_batch_size:])

    return result

def apply_batch_updates(
    optimizer,
    params: List[Tensor],
    momentums: List[Tensor],
    Qs: List[Tensor],
    Q_new_batch: torch.Tensor,
    Q_state_batch: torch.Tensor,
    P_batch: torch.Tensor,
    R_batch: torch.Tensor,
    configs: List[DionParamConfig],
    optim_groups: List[dict],
    optimizer_states: List[dict],
    dist_metas: List,
    param_shapes: List[Tuple[int, int]],
    real_batch_size: int,
    global_param_offset: int,
    batch_group: DionBatchGroup,
    batch_collectives: Optional[DionBatchCollectives] = None,
    commit_updates: Optional[List[Callable[[Tensor, Tensor], None]]] = None,
) -> None:
    """Apply weight decay, Dion delta update, and TP re-sharding for Q."""
    del global_param_offset, R_batch
    if real_batch_size <= 0:
        return
    _validate_batch_update_contract(
        optimizer,
        optim_groups=optim_groups,
        optimizer_states=optimizer_states,
        dist_metas=dist_metas,
        param_shapes=param_shapes,
        real_batch_size=real_batch_size,
    )

    delta_batch = compute_update_batch(
        Q_new_batch,
        P_batch,
        configs,
        real_batch_size=real_batch_size,
        delta_shape=params[0].shape,
    )

    optimizer_state0 = optimizer_states[0]
    if "true_global_shape" in optimizer_state0:
        m_global, n_global = optimizer_state0["true_global_shape"]
    else:
        m_global, n_global = get_global_shape(
            dist_metas[0],
            param_shapes[0][0],
            param_shapes[0][1],
        )

    m_for_lr, n_for_lr = m_global, n_global
    if "per_expert_global_shape" in optimizer_state0:
        m_for_lr, n_for_lr = optimizer_state0["per_expert_global_shape"]

    lr = optim_groups[0].get("lr", optimizer.defaults["lr"])
    scaled_lr = scaled_lr_for_shape(
        lr=lr,
        m_for_lr=m_for_lr,
        n_for_lr=n_for_lr,
        rule=optimizer.defaults.get("lr_scaling_rule", "moonlight"),
        rank_fraction=optimizer.defaults.get("rank_fraction", 0.25),
    )

    wd_mult = optim_groups[0].get("wd_mult", 1.0)
    weight_decay = optim_groups[0].get(
        "weight_decay", optimizer.defaults["weight_decay"] * wd_mult
    )
    if batch_collectives is None:
        raise RuntimeError(
            "[DION_MISSING_BATCH_COLLECTIVES] "
            f"step={optimizer._step_count} rank={optimizer._global_rank}"
        )
    tp_q_reshard_by_index = {
        idx: collective for collective in batch_collectives.tp_q_reshards for idx in collective.indices
    }

    for index in range(real_batch_size):
        param = params[index]
        delta = delta_batch[index]
        if delta.shape != param.shape:
            delta = delta.contiguous().view(param.shape)

        if weight_decay > 0:
            param.mul_(1 - lr * weight_decay)
        param.add_(delta.to(param.dtype), alpha=-scaled_lr)

        q_state = Q_state_batch[index].to(Qs[index].dtype)
        tp_q_reshard = tp_q_reshard_by_index.get(index)
        if tp_q_reshard is not None:
            tp_group = tp_q_reshard.process_group
            if tp_group is None:
                raise RuntimeError(
                    "[DION_MISSING_TP_GROUP_FOR_Q_RESHARD] "
                    f"step={optimizer._step_count} rank={optimizer._global_rank} "
                    f"param_uid={getattr(dist_metas[index], 'param_uid', None) if dist_metas is not None else None}"
                )
            q_state = reshard_q_along_tp(q_state, tp_group, tp_q_reshard.rank)
        elif optimizer.is_distributed_mode and has_tp_shard(configs[index]):
            raise RuntimeError(
                "[DION_MISSING_TP_Q_RESHARD] "
                f"step={optimizer._step_count} rank={optimizer._global_rank} "
                f"param_uid={getattr(dist_metas[index], 'param_uid', None) if dist_metas is not None else None}"
            )
        Qs[index].copy_(q_state)
        if (
            dist_metas is not None
            and getattr(dist_metas[index], "is_qkv_child", False)
            and (commit_updates is None or commit_updates[index] is None)
        ):
            raise RuntimeError(
                "[DION_QKV_CHILD_MISSING_COMMIT_UPDATE] "
                f"step={optimizer._step_count} rank={optimizer._global_rank} "
                f"param_uid={getattr(dist_metas[index], 'param_uid', None)} "
                f"param_name={getattr(dist_metas[index], 'param_name', '')}"
            )
        if commit_updates is not None and commit_updates[index] is not None:
            commit_updates[index](param, momentums[index])

    del delta_batch


def run_compressed_comm_async(
    optimizer,
    P_batch: torch.Tensor,
    M_batch: torch.Tensor,
    Q_batch: torch.Tensor,
    configs: List[DionParamConfig],
    dist_metas: List,
    batch_group: Optional[DionBatchGroup] = None,
    batch_collectives: Optional[DionBatchCollectives] = None,
    skip_p_replicate: bool = False,
    skip_r_replicate: bool = False,
) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
    """Run the compressed Dion communication path selected by the adapter."""
    if batch_group is None:
        raise RuntimeError(
            "[DION_MISSING_BATCH_GROUPS] "
            f"step={optimizer._step_count} rank={optimizer._global_rank}"
        )
    if batch_collectives is None:
        raise RuntimeError(
            "[DION_MISSING_BATCH_COLLECTIVES] "
            f"step={optimizer._step_count} rank={optimizer._global_rank}"
        )
    comm_group = batch_group.replicate_group
    comm_world_size = dist.get_world_size(comm_group) if comm_group else 1
    ortho_group = batch_group.ortho_group
    kernel_kind = batch_group.kernel_kind
    batch_size = P_batch.size(0)
    fs_collective = batch_collectives.fs_collective
    if (
        kernel_kind == "fsdp"
        and optimizer.use_fs_collectives
        and optimizer.is_distributed_mode
        and fs_collective is None
    ):
        raise RuntimeError(
            "[DION_MISSING_FS_ONLY_ORTHO_GROUP_FOR_COMPRESSED_ROUTE] "
            f"step={optimizer._step_count} rank={optimizer._global_rank}"
        )
    fs_group = fs_collective.process_group if fs_collective is not None else None
    fs_group_world = fs_collective.world_size if fs_collective is not None else 1
    compressed_replicate_group = batch_group.compressed_replicate_group

    if kernel_kind == "fsdp" and optimizer.use_fs_collectives and fs_group and fs_group_world > 1:
        batch_size = int(P_batch.size(0))
        if batch_collectives is None or batch_collectives.fs_collective is None:
            raise RuntimeError(
                "[DION_MISSING_FS_ONLY_ORTHO_GROUP] "
                f"step={optimizer._step_count} rank={optimizer._global_rank}"
            )
        if batch_size != int(fs_group_world):
            raise RuntimeError(
                "[DION_FSONLY_BATCH_SIZE_MISMATCH] "
                f"batch_size={batch_size} fs_world_size={int(fs_group_world)}"
            )
        if len(fs_collective.indices) != batch_size:
            raise RuntimeError(
                "[DION_FSONLY_ORTHOGONALIZE_GROUP_SIZE_MISMATCH] "
                f"batch_size={batch_size} group_size={len(fs_collective.indices)}"
            )

        P_single = funcol.reduce_scatter_tensor(
            P_batch.contiguous(),
            reduceOp="sum",
            scatter_dim=0,
            group=fs_group,
        )
        yield

        if (
            compressed_replicate_group is not None
            and dist.get_world_size(compressed_replicate_group) > 1
        ):
            dist.all_reduce(
                P_single,
                op=replicate_reduce_op(optimizer),
                group=compressed_replicate_group,
            )
            yield

        fs_rank = int(fs_collective.rank)
        batch_indices = tuple(int(index) for index in fs_collective.indices)
        if fs_rank >= len(batch_indices):
            raise RuntimeError(
                "[DION_FSONLY_ORTHOGONALIZE_RANK_MISMATCH] "
                f"fs_rank={fs_rank} group_size={len(batch_indices)}"
            )
        rank_index = batch_indices[fs_rank]
        rank_meta = dist_metas[rank_index]
        if rank_meta is None:
            torch._assert_async(
                torch.count_nonzero(P_single) == 0,
                f"[DION_NONZERO_FSONLY_PADDED_MATRIX] shape={tuple(P_single.shape)}",
            )
            P_single = torch.zeros_like(P_single).contiguous()
        else:
            sketch = make_sketch(
                dist_metas=[rank_meta],
                contract="fsdp",
                step_count=optimizer._step_count,
            )
            with _dion_math_precision_context():
                P_single = orthogonalize(
                    P_single,
                    rcqr_oversample=optimizer.defaults["rcqr_oversample"],
                    make_sketch=sketch,
                ).to(torch.float32)
            P_single = P_single.to(P_batch.dtype).contiguous()

        P_batch = funcol.all_gather_tensor(
            P_single.contiguous(),
            gather_dim=0,
            group=fs_group,
        )
        yield
        if sorted(batch_indices) != list(range(batch_size)):
            raise RuntimeError(
                "[DION_FSONLY_GATHER_PERMUTATION_INVALID] "
                f"batch_size={batch_size} indices={batch_indices}"
            )
        if batch_indices != tuple(range(batch_size)):
            gathered_p = P_batch
            P_batch = torch.empty_like(gathered_p)
            for gather_rank, batch_index in enumerate(batch_indices):
                P_batch[batch_index].copy_(gathered_p[gather_rank])
        with _dion_math_precision_context():
            R_batch = M_batch.mT @ P_batch
        yield from reduce_r_across_tp(
            optimizer,
            R_batch,
            configs,
            dist_metas,
            batch_collectives=batch_collectives,
        )
        if compressed_replicate_group is not None:
            R_batch = yield from collapse_batch_across_replicas(
                optimizer,
                R_batch,
                replicate_group=compressed_replicate_group,
            )
        return P_batch, R_batch

    if comm_group is None or comm_world_size <= 1:
        P_batch = yield from collapse_batch_across_replicas(
            optimizer, P_batch, replicate_group=comm_group
        )
        if ortho_group is not None:
            P_batch = distributed_orthogonalize(
                optimizer,
                P_batch,
                ortho_group=ortho_group,
                oversample=optimizer.defaults["rcqr_oversample"],
                dist_metas=dist_metas,
            )
        else:
            with _dion_math_precision_context():
                P_batch = orthogonalize(
                    P_batch,
                    rcqr_oversample=optimizer.defaults["rcqr_oversample"],
                ).to(torch.float32)
            P_batch = P_batch.to(M_batch.dtype).contiguous()

        R_batch = M_batch.mT @ P_batch
        yield from reduce_r_across_tp(
            optimizer,
            R_batch,
            configs,
            dist_metas,
            batch_collectives=batch_collectives,
        )
        R_batch = yield from collapse_batch_across_replicas(
            optimizer, R_batch, replicate_group=comm_group
        )
        return P_batch, R_batch

    original_batch_size = batch_size

    if kernel_kind == "fsdp_tp":
        if skip_p_replicate and (comm_group is None or comm_world_size <= 1):
            raise RuntimeError(
                "[DION_INVALID_SKIP_P_REPLICATE] "
                f"step={optimizer._step_count} rank={optimizer._global_rank}"
            )
        if skip_r_replicate and (comm_group is None or comm_world_size <= 1):
            raise RuntimeError(
                "[DION_INVALID_SKIP_R_REPLICATE] "
                f"step={optimizer._step_count} rank={optimizer._global_rank}"
            )
        if comm_group is not None and comm_world_size > 1 and not skip_p_replicate:
            P_batch = yield from collapse_batch_across_replicas(
                optimizer,
                P_batch,
                replicate_group=comm_group,
            )
        if ortho_group is not None:
            P_batch = distributed_orthogonalize(
                optimizer,
                P_batch,
                ortho_group=ortho_group,
                oversample=optimizer.defaults["rcqr_oversample"],
                dist_metas=dist_metas,
            )
        else:
            with _dion_math_precision_context():
                P_batch = orthogonalize(
                    P_batch,
                    rcqr_oversample=optimizer.defaults["rcqr_oversample"],
                ).to(torch.float32)
            P_batch = P_batch.to(M_batch.dtype).contiguous()
        with _dion_math_precision_context():
            R_batch = M_batch.mT @ P_batch
        yield from reduce_r_across_tp(
            optimizer,
            R_batch,
            configs,
            dist_metas,
            batch_collectives=batch_collectives,
        )
        if comm_group is not None and comm_world_size > 1 and not skip_r_replicate:
            R_batch = yield from collapse_batch_across_replicas(
                optimizer,
                R_batch,
                replicate_group=comm_group,
            )
        return P_batch[:original_batch_size], R_batch[:original_batch_size]

    if comm_group is not None and comm_world_size > 1:
        P_batch = yield from collapse_batch_across_replicas(
            optimizer,
            P_batch,
            replicate_group=comm_group,
        )

    use_replicated_orthogonalize = (
        comm_world_size > 1
        and ortho_group is None
        and not (configs and all(is_fs_only_config(config) for config in configs))
    )
    if use_replicated_orthogonalize:
        if comm_world_size <= 1:
            yield
            with _dion_math_precision_context():
                P_batch = orthogonalize(
                    P_batch,
                    rcqr_oversample=optimizer.defaults["rcqr_oversample"],
                ).to(torch.float32)
            P_batch = P_batch.to(M_batch.dtype).contiguous()
        else:
            batch_size = P_batch.size(0)
            real_batch_size = batch_size
            if batch_size % comm_world_size != 0:
                pad = comm_world_size - (batch_size % comm_world_size)
                padded_batch_size = batch_size + pad
                P_padded = optimizer._cached_buffer(
                    "replicated_p_padded",
                    (padded_batch_size, P_batch.size(1), P_batch.size(2)),
                    P_batch.dtype,
                    P_batch.device,
                    zero=True,
                )
                P_padded[:batch_size].copy_(P_batch)
                P_batch = P_padded
                batch_size = P_batch.size(0)

            comm_rank = dist.get_rank(comm_group)
            P_batch = optimizer._cached_buffer(
                "replicated_p_ortho_full",
                P_batch.shape,
                P_batch.dtype,
                P_batch.device,
            )

            for chunk_start in range(0, batch_size, comm_world_size):
                chunk_end = chunk_start + comm_world_size
                P_chunk = P_batch[chunk_start:chunk_end].contiguous()
                P_single = P_chunk[comm_rank : comm_rank + 1].clone()
                with _dion_math_precision_context():
                    P_single = orthogonalize(
                        P_single,
                        rcqr_oversample=optimizer.defaults["rcqr_oversample"],
                    ).to(torch.float32)
                P_single = P_single.to(P_chunk.dtype).contiguous()
                P_chunk = funcol.all_gather_tensor(
                    P_single.contiguous(),
                    gather_dim=0,
                    group=comm_group,
                )
                yield
                P_batch[chunk_start:chunk_end].copy_(P_chunk)

            P_batch = P_batch[:real_batch_size]
    else:
        if ortho_group is not None:
            P_batch = distributed_orthogonalize(
                optimizer,
                P_batch,
                ortho_group=ortho_group,
                oversample=optimizer.defaults["rcqr_oversample"],
                dist_metas=dist_metas,
                real_batch_size=original_batch_size,
            )
        else:
            with _dion_math_precision_context():
                P_batch = orthogonalize(
                    P_batch,
                    rcqr_oversample=optimizer.defaults["rcqr_oversample"],
                ).to(torch.float32)
            P_batch = P_batch.to(M_batch.dtype).contiguous()

    with _dion_math_precision_context():
        R_batch = M_batch.mT @ P_batch
    yield from reduce_r_across_tp(
        optimizer,
        R_batch,
        configs,
        dist_metas,
        batch_collectives=batch_collectives,
    )
    if comm_group is not None and comm_world_size > 1:
        R_batch = yield from collapse_batch_across_replicas(
            optimizer,
            R_batch,
            replicate_group=comm_group,
        )

    P_batch = P_batch[:original_batch_size]
    R_batch = R_batch[:original_batch_size]

    return P_batch, R_batch


def batch_dion_update_async(
    optimizer,
    params: List[Tensor],
    momentums: List[Tensor],
    Qs: List[Tensor],
    configs: List[DionParamConfig],
    dist_metas: List,
    optim_groups: List[dict],
    grads: Optional[List[Tensor]] = None,
    optimizer_states: Optional[List[dict]] = None,
    param_shapes: Optional[List[Tuple[int, int]]] = None,
    real_batch_size: Optional[int] = None,
    global_param_offset: int = 0,
    batch_group: Optional[DionBatchGroup] = None,
    batch_collectives: Optional[DionBatchCollectives] = None,
    commit_updates: Optional[List[Callable[[Tensor, Tensor], None]]] = None,
) -> Generator[None, None, None]:
    """Perform one adapter-authored batched Dion update with async communication."""
    batch_size = len(params)
    if real_batch_size is None:
        real_batch_size = batch_size

    if batch_group is None:
        raise RuntimeError(
            "[DION_MISSING_BATCH_GROUPS] "
            f"step={optimizer._step_count} rank={optimizer._global_rank}"
        )
    if batch_collectives is None:
        raise RuntimeError(
            "[DION_MISSING_BATCH_COLLECTIVES] "
            f"step={optimizer._step_count} rank={optimizer._global_rank}"
        )
    _validate_batch_update_contract(
        optimizer,
        optim_groups=optim_groups,
        optimizer_states=optimizer_states,
        dist_metas=dist_metas,
        param_shapes=param_shapes,
        real_batch_size=real_batch_size,
    )
    replicate_group = batch_group.replicate_group
    ortho_group = batch_group.ortho_group
    replicate_world_size = (
        dist.get_world_size(replicate_group) if replicate_group is not None else 1
    )
    use_compressed = (
        optimizer.use_compressed_comm
        and replicate_world_size > 1
        and any(config.compressed_all_reduce for config in configs)
    )

    if not use_compressed:
        yield from collapse_grads_across_replicas(
            optimizer,
            grads,
            replicate_group=replicate_group,
        )

    if grads:
        if momentums[0].dtype == grads[0].dtype:
            torch._foreach_add_(momentums, grads)
        else:
            for momentum, grad in zip(momentums, grads):
                momentum.add_(grad.to(momentum.dtype))

    Q_for_matmul = yield from unshard_q_batch(
        optimizer,
        Qs,
        configs,
        dist_metas,
        batch_collectives=batch_collectives,
        cache_scope=global_param_offset,
    )

    M_for_matmul = [
        (momentum.mT if config.is_transposed else momentum).to(torch.float32)
        for momentum, config in zip(momentums, configs)
    ]
    M_batch = torch.stack(M_for_matmul, dim=0)
    Q_batch = torch.stack([q_tensor.to(torch.float32) for q_tensor in Q_for_matmul], dim=0)
    del Q_for_matmul

    with _dion_math_precision_context():
        P_batch = M_batch @ Q_batch

    if not (configs and all(is_fs_only_config(config) for config in configs)):
        yield from reduce_p_across_fs_groups(
            optimizer,
            P_batch,
            configs,
            dist_metas,
            batch_collectives=batch_collectives,
        )

    if use_compressed:
        P_batch, R_batch = yield from run_compressed_comm_async(
            optimizer,
            P_batch,
            M_batch,
            Q_batch,
            configs,
            dist_metas,
            batch_group,
            batch_collectives,
        )
    else:
        use_replicated_orthogonalize = (
            replicate_world_size > 1
            and ortho_group is None
            and not (configs and all(is_fs_only_config(config) for config in configs))
        )
        if use_replicated_orthogonalize:
            if replicate_world_size <= 1:
                yield
                with _dion_math_precision_context():
                    P_batch = orthogonalize(
                        P_batch,
                        rcqr_oversample=optimizer.defaults["rcqr_oversample"],
                    ).to(torch.float32)
                P_batch = P_batch.to(dtype).contiguous()
            else:
                batch_size = P_batch.size(0)
                real_batch_size = batch_size
                scope = str(int(global_param_offset))
                if batch_size % replicate_world_size != 0:
                    pad = replicate_world_size - (batch_size % replicate_world_size)
                    padded_batch_size = batch_size + pad
                    P_padded = optimizer._cached_buffer(
                        f"replicated_p_padded_{scope}",
                        (padded_batch_size, P_batch.size(1), P_batch.size(2)),
                        P_batch.dtype,
                        P_batch.device,
                        zero=True,
                    )
                    P_padded[:batch_size].copy_(P_batch)
                    P_batch = P_padded
                    batch_size = P_batch.size(0)

                replicate_rank = dist.get_rank(replicate_group)
                P_batch = optimizer._cached_buffer(
                    f"replicated_p_ortho_full_{scope}",
                    P_batch.shape,
                    P_batch.dtype,
                    P_batch.device,
                )

                for chunk_start in range(0, batch_size, replicate_world_size):
                    chunk_end = chunk_start + replicate_world_size
                    P_chunk = P_batch[chunk_start:chunk_end].contiguous()
                    P_single = P_chunk[replicate_rank : replicate_rank + 1].clone()
                    with _dion_math_precision_context():
                        P_single = orthogonalize(
                            P_single,
                            rcqr_oversample=optimizer.defaults["rcqr_oversample"],
                        ).to(torch.float32)
                    P_single = P_single.to(P_chunk.dtype).contiguous()
                    P_chunk = funcol.all_gather_tensor(
                        P_single.contiguous(),
                        gather_dim=0,
                        group=replicate_group,
                    )
                    yield
                    P_batch[chunk_start:chunk_end].copy_(P_chunk)

                P_batch = P_batch[:real_batch_size]
        else:
            if configs and all(is_fs_only_config(config) for config in configs):
                if batch_collectives is None or batch_collectives.fs_collective is None:
                    raise RuntimeError(
                        "[DION_MISSING_FS_ONLY_ORTHO_GROUP] "
                        f"step={optimizer._step_count} rank={optimizer._global_rank}"
                    )
                fs_collective = batch_collectives.fs_collective
                fs_group = fs_collective.process_group
                fs_world_size = int(fs_collective.world_size)
                fs_rank = int(fs_collective.rank)
                batch_size = int(P_batch.size(0))

                if fs_group is None or fs_world_size <= 1:
                    original_dtype = P_batch.dtype
                    with _dion_math_precision_context():
                        P_batch = orthogonalize(
                            P_batch,
                            rcqr_oversample=optimizer.defaults["rcqr_oversample"],
                        ).to(torch.float32)
                    P_batch = P_batch.to(original_dtype).contiguous()
                else:
                    if batch_size != fs_world_size:
                        raise RuntimeError(
                            "[DION_FSONLY_BATCH_SIZE_MISMATCH] "
                            f"batch_size={batch_size} fs_world_size={fs_world_size}"
                        )
                    if len(fs_collective.indices) != batch_size:
                        raise RuntimeError(
                            "[DION_FSONLY_ORTHOGONALIZE_GROUP_SIZE_MISMATCH] "
                            f"batch_size={batch_size} group_size={len(fs_collective.indices)}"
                        )
                    if fs_rank >= len(fs_collective.indices):
                        raise RuntimeError(
                            "[DION_FSONLY_ORTHOGONALIZE_RANK_MISMATCH] "
                            f"fs_rank={fs_rank} group_size={len(fs_collective.indices)}"
                        )

                    P_single = funcol.reduce_scatter_tensor(
                        P_batch.contiguous(),
                        reduceOp="sum",
                        scatter_dim=0,
                        group=fs_group,
                    )
                    yield

                    batch_indices = tuple(int(index) for index in fs_collective.indices)
                    rank_index = batch_indices[fs_rank]
                    rank_meta = dist_metas[rank_index]
                    if rank_meta is None:
                        torch._assert_async(
                            torch.count_nonzero(P_single) == 0,
                            f"[DION_NONZERO_FSONLY_PADDED_MATRIX] shape={tuple(P_single.shape)}",
                        )
                        P_single = torch.zeros_like(P_single).contiguous()
                    else:
                        sketch = make_sketch(
                            dist_metas=[rank_meta],
                            contract="fsdp",
                            step_count=optimizer._step_count,
                        )
                        with _dion_math_precision_context():
                            P_single = orthogonalize(
                                P_single,
                                rcqr_oversample=optimizer.defaults["rcqr_oversample"],
                                make_sketch=sketch,
                            ).to(torch.float32)
                    P_single = P_single.to(P_batch.dtype).contiguous()

                    P_batch = funcol.all_gather_tensor(
                        P_single.contiguous(),
                        gather_dim=0,
                        group=fs_group,
                    )
                    yield
                    if sorted(batch_indices) != list(range(batch_size)):
                        raise RuntimeError(
                            "[DION_FSONLY_GATHER_PERMUTATION_INVALID] "
                            f"batch_size={batch_size} indices={batch_indices}"
                        )
                    if batch_indices != tuple(range(batch_size)):
                        gathered_p = P_batch
                        P_batch = torch.empty_like(gathered_p)
                        for gather_rank, batch_index in enumerate(batch_indices):
                            P_batch[batch_index].copy_(gathered_p[gather_rank])
            elif ortho_group is not None:
                P_batch = distributed_orthogonalize(
                    optimizer,
                    P_batch,
                    ortho_group=ortho_group,
                    oversample=optimizer.defaults["rcqr_oversample"],
                    dist_metas=dist_metas,
                    real_batch_size=real_batch_size,
                )
            else:
                original_dtype = P_batch.dtype
                with _dion_math_precision_context():
                    P_batch = orthogonalize(
                        P_batch,
                        rcqr_oversample=optimizer.defaults["rcqr_oversample"],
                    ).to(torch.float32)
                P_batch = P_batch.to(original_dtype).contiguous()
        with _dion_math_precision_context():
            R_batch = M_batch.mT @ P_batch
        yield from reduce_r_across_tp(
            optimizer,
            R_batch,
            configs,
            dist_metas,
            batch_collectives=batch_collectives,
        )
    P_batch, R_batch = fix_all_zero_or_nan(
        P_batch,
        R_batch,
        Q_batch,
        M_batch,
    )

    apply_error_feedback(
        momentums,
        P_batch,
        R_batch,
        configs,
        optim_groups,
        default_mu=optimizer.defaults["mu"],
    )

    Q_new_batch = yield from normalize_cols_async(
        optimizer,
        R_batch,
        configs,
        dist_metas,
        real_batch_size,
        global_param_offset,
        batch_group=batch_group,
    )
    Q_state_batch = Q_new_batch

    apply_batch_updates(
        optimizer,
        params,
        momentums,
        Qs,
        Q_new_batch,
        Q_state_batch,
        P_batch,
        R_batch,
        configs,
        optim_groups,
        optimizer_states,
        dist_metas,
        param_shapes,
        real_batch_size,
        global_param_offset,
        batch_group,
        batch_collectives,
        commit_updates,
    )

    del M_batch, Q_batch, P_batch, R_batch, Q_new_batch, Q_state_batch
