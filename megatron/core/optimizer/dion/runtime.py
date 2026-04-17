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
    sanitize_zero_or_nan_dion_batch,
    local_column_sum_sq,
    normalize_columns,
    orthogonalize_dense_replicate_batch_async,
    orthogonalize_fs_only_batch,
    scaled_lr_for_shape,
)
from .ortho import (
    _dion_math_precision_context,
    distributed_orthogonalize,
    orthogonalize,
    orthogonalize_local_matrix_batch,
    reshard_q_along_tp,
)
from .state import has_tp_shard, is_fs_only_config
from .types import DionBatchCollectives, DionBatch, DionBatchGroup, DionParamConfig
from .utils import get_global_shape

_REPLICATE_GROUP_UNSET = object()
logger = logging.getLogger(__name__)


class AsyncTask:
    """Wrapper for async generator tasks to enable concurrent execution."""

    def __init__(self, generator: Generator[None, None, None]):
        self.generator = generator
        self.completed = False

    def step(self) -> bool:
        """Execute one step of the async task. Returns True when completed."""
        try:
            next(self.generator)
            return False
        except StopIteration:
            self.completed = True
            if self.generator is not None:
                self.generator.close()
                self.generator = None
            return True


class AsyncRuntime:
    """Runtime for managing and executing async tasks concurrently."""

    def __init__(self, tasks: Generator[AsyncTask, None, None], max_concurrent_tasks: int):
        if int(max_concurrent_tasks) <= 0:
            raise ValueError(f"Invalid max_concurrent_tasks={max_concurrent_tasks}")
        self.tasks: List[AsyncTask] = list(tasks)
        self.max_concurrent = int(max_concurrent_tasks)

    def run(self):
        """Execute all tasks with controlled concurrency."""
        active_tasks: List[AsyncTask] = []
        task_iter = iter(self.tasks)

        for _ in range(min(self.max_concurrent, len(self.tasks))):
            try:
                active_tasks.append(next(task_iter))
            except StopIteration:
                break

        while active_tasks:
            completed_indices = []
            for i, task in enumerate(active_tasks):
                if task.step():
                    completed_indices.append(i)
            for i in reversed(completed_indices):
                active_tasks.pop(i)
                try:
                    active_tasks.append(next(task_iter))
                except StopIteration:
                    pass

        for task in self.tasks:
            if task.generator is not None:
                task.generator.close()
                task.generator = None
        self.tasks.clear()
        del active_tasks, task_iter


def resolve_async_task_limit(
    *,
    max_concurrent_tasks: Optional[int],
    task_count: int,
) -> int:
    """Resolve async runtime width without relying on hidden global defaults."""
    if task_count <= 0:
        raise RuntimeError(f"[Dion] invalid async task_count={task_count}")
    if max_concurrent_tasks is None:
        # Reference Dion hard-codes AsyncRuntime(..., max_concurrent_tasks=3).
        return min(task_count, 3)
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

    for dion_batch in dion_batches:
        yield AsyncTask(run_dion_batch_async(optimizer, dion_batch))

    if scalar_params:
        yield AsyncTask(optimizer._run_scalar_bucket_async(scalar_params))


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
            batch_group,
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

    reduced_batch = funcol.all_reduce(
        batch,
        reduceOp="avg" if replicate_reduce_op(optimizer) == dist.ReduceOp.AVG else "sum",
        group=replicate_group,
    )
    yield
    batch.copy_(reduced_batch)
    yield


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
    cache_scope: Optional[int] = None,
) -> Generator[List[Tensor], None, None]:
    """All-gather TP-sharded Q blocks needed for local matmul."""
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
                        "[DION_INCONSISTENT_Q_UNSHARD_PLAN] "
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
            for slot, idx in enumerate(indices):
                local_batch[slot].copy_(Qs[idx])

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
            "[DION_MISSING_BATCH_COLLECTIVES] "
            f"step={optimizer._step_count} rank={optimizer._global_rank}"
        )
    if not batch_collectives.fs_p_collectives:
        return

    did_yield = False
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
                yield
                handle.wait()
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
                del tensors
            if not did_yield:
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
            yield
            handle.wait()
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
    del configs, global_param_offset
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
            "[DION_MISSING_Q_NORM_BINDING] "
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


def orthogonalize_p_batch(
    optimizer,
    P_batch: torch.Tensor,
    configs: List[DionParamConfig],
    dist_metas: List,
    real_batch_size: Optional[int] = None,
    ortho_group=None,
    batch_collectives: Optional[DionBatchCollectives] = None,
) -> Generator[torch.Tensor, None, None]:
    """Orthogonalize a P batch using the correct local or distributed path."""
    if configs and all(is_fs_only_config(config) for config in configs):
        P_batch = yield from orthogonalize_fs_only_batch(
            optimizer,
            P_batch,
            dist_metas,
            batch_collectives=batch_collectives,
        )
        return P_batch

    if ortho_group is not None:
        P_batch = distributed_orthogonalize(
            optimizer,
            P_batch,
            ortho_group=ortho_group,
            oversample=optimizer.defaults["rcqr_oversample"],
            dist_metas=dist_metas,
            real_batch_size=real_batch_size,
        )
    else:
        P_batch = orthogonalize_local_matrix_batch(
            optimizer,
            P_batch,
            dist_metas=dist_metas,
            tag="dense_replicate_local",
        )
    yield
    return P_batch


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
) -> None:
    """Apply weight decay, Dion delta update, and TP re-sharding for Q."""
    del batch_group, global_param_offset, momentums, R_batch
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
                "slot": int(index),
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

    del delta_batch


def sanitize_dion_batch_for_update(
    optimizer,
    P_batch: Tensor,
    R_batch: Tensor,
    Q_batch: Tensor,
    M_batch: Tensor,
    *,
    real_batch_size: Optional[int] = None,
    global_param_offset: int = 0,
    configs: Optional[List[DionParamConfig]] = None,
    dist_metas: Optional[List] = None,
) -> Tuple[Tensor, Tensor]:
    """Apply reference-aligned NaN/zero sanitization and optimizer-local warning logs."""
    raw_p_nan_mask = torch.isnan(P_batch).view(P_batch.size(0), -1).any(dim=1)
    raw_r_nan_mask = torch.isnan(R_batch).view(R_batch.size(0), -1).any(dim=1)
    is_all_zero = (M_batch == 0).view(M_batch.size(0), -1).all(dim=1)
    has_nan = raw_p_nan_mask | raw_r_nan_mask
    unexpected_nan = has_nan & (~is_all_zero)
    P_batch, R_batch = sanitize_zero_or_nan_dion_batch(
        P_batch,
        R_batch,
        Q_batch,
        M_batch,
    )
    real_slot_mask = None
    if real_batch_size is not None:
        real_slot_mask = torch.zeros(
            unexpected_nan.size(0),
            dtype=torch.bool,
            device=unexpected_nan.device,
        )
        real_slot_mask[:real_batch_size] = True
    warn_mask = unexpected_nan.view(unexpected_nan.size(0), -1).any(dim=1)
    if real_slot_mask is not None:
        warn_mask = warn_mask & real_slot_mask
    if warn_mask.any():
        warn_count = getattr(optimizer, "_unexpected_nan_warn_count", 0)
        if warn_count < 8:
            setattr(optimizer, "_unexpected_nan_warn_count", warn_count + 1)
            bad_slots = []
            for index in torch.nonzero(warn_mask, as_tuple=False).view(-1).tolist():
                config = configs[index] if (configs is not None and index < len(configs)) else None
                dist_meta = (
                    dist_metas[index]
                    if (dist_metas is not None and index < len(dist_metas))
                    else None
                )
                bad_slots.append(
                    {
                        "slot": int(index),
                        "global_offset": int(global_param_offset + index),
                        "is_real": bool(real_batch_size is None or index < real_batch_size),
                        "param_uid": getattr(dist_meta, "param_uid", None)
                        if dist_meta is not None
                        else None,
                        "param_name": getattr(dist_meta, "param_name", "")
                        if dist_meta is not None
                        else "",
                        "shape": tuple(P_batch[index].shape),
                        "is_tr": bool(config.is_transposed) if config is not None else None,
                        "tp": getattr(config, "tp_shard_dim", None)
                        if config is not None
                        else None,
                        "fs": getattr(config, "fs_shard_dim", None)
                        if config is not None
                        else None,
                        "raw_p_nan": bool(raw_p_nan_mask[index].item()),
                        "raw_r_nan": bool(raw_r_nan_mask[index].item()),
                        "m_nan": bool(torch.isnan(M_batch[index]).any().item()),
                        "q_nan": bool(torch.isnan(Q_batch[index]).any().item()),
                        "p_nan": bool(torch.isnan(P_batch[index]).any().item()),
                        "r_nan": bool(torch.isnan(R_batch[index]).any().item()),
                        "m_sq": float(M_batch[index].float().square().sum().item()),
                        "q_sq": float(Q_batch[index].float().square().sum().item()),
                        "m_maxabs": float(M_batch[index].float().abs().amax().item()),
                        "q_maxabs": float(Q_batch[index].float().abs().amax().item()),
                    }
                )
            logger.warning(
                "[DION_UNEXPECTED_NAN] rank=%s step=%s count=%s real_batch_size=%s bad_slots=%s",
                optimizer._global_rank,
                optimizer._step_count,
                int(unexpected_nan.sum().item()),
                int(real_batch_size) if real_batch_size is not None else -1,
                bad_slots,
            )

    return P_batch, R_batch


def apply_error_feedback_to_momentum(
    optimizer,
    momentums: List[Tensor],
    P_batch: Tensor,
    R_batch: Tensor,
    configs: List[DionParamConfig],
    optim_groups: List[dict],
) -> None:
    """Apply Dion error feedback using the optimizer default mu contract."""
    apply_error_feedback(
        momentums,
        P_batch,
        R_batch,
        configs,
        optim_groups,
        default_mu=optimizer.defaults["mu"],
    )


def run_compressed_comm_async(
    optimizer,
    P_batch: torch.Tensor,
    M_batch: torch.Tensor,
    Q_batch: torch.Tensor,
    configs: List[DionParamConfig],
    dist_metas: List,
    batch_group: Optional[DionBatchGroup] = None,
    batch_collectives: Optional[DionBatchCollectives] = None,
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
    fs_orthogonalize = batch_collectives.fs_orthogonalize
    if (
        kernel_kind == "fsdp"
        and optimizer.use_fs_collectives
        and optimizer.is_distributed_mode
        and fs_orthogonalize is None
    ):
        raise RuntimeError(
            "[DION_MISSING_FS_ONLY_ORTHO_PLAN_FOR_COMPRESSED_ROUTE] "
            f"step={optimizer._step_count} rank={optimizer._global_rank}"
        )
    fs_group = fs_orthogonalize.process_group if fs_orthogonalize is not None else None
    fs_group_world = fs_orthogonalize.world_size if fs_orthogonalize is not None else 1
    fs_only_replicate_group = batch_group.compressed_replicate_group

    if kernel_kind == "fsdp" and optimizer.use_fs_collectives and fs_group and fs_group_world > 1:
        P_ortho_full = yield from orthogonalize_fs_only_batch(
            optimizer,
            P_batch,
            dist_metas,
            batch_collectives=batch_collectives,
            replicate_group=fs_only_replicate_group,
        )
        with _dion_math_precision_context():
            R_batch = M_batch.mT @ P_ortho_full
        yield from reduce_r_across_tp(
            optimizer,
            R_batch,
            configs,
            dist_metas,
            batch_collectives=batch_collectives,
        )
        if fs_only_replicate_group is not None:
            yield from collapse_batch_across_replicas(
                optimizer,
                R_batch,
                replicate_group=fs_only_replicate_group,
            )
        return P_ortho_full, R_batch

    if comm_group is None or comm_world_size <= 1:
        yield from collapse_batch_across_replicas(optimizer, P_batch, replicate_group=comm_group)
        if ortho_group is not None:
            P_ortho = distributed_orthogonalize(
                optimizer,
                P_batch,
                ortho_group=ortho_group,
                oversample=optimizer.defaults["rcqr_oversample"],
                dist_metas=dist_metas,
            )
        else:
            P_ortho = P_batch.clone()
            for index in range(P_batch.size(0)):
                P_ortho[index] = orthogonalize(
                    P_batch[index],
                    rcqr_oversample=optimizer.defaults["rcqr_oversample"],
                )

        yield

        R_batch = M_batch.mT @ P_ortho
        yield from reduce_r_across_tp(
            optimizer,
            R_batch,
            configs,
            dist_metas,
            batch_collectives=batch_collectives,
        )
        yield from collapse_batch_across_replicas(optimizer, R_batch, replicate_group=comm_group)
        return P_ortho, R_batch

    original_batch_size = batch_size

    if kernel_kind == "fsdp_tp":
        if comm_group is not None and comm_world_size > 1:
            yield from collapse_batch_across_replicas(
                optimizer,
                P_batch,
                replicate_group=comm_group,
            )
        if ortho_group is not None:
            P_ortho_full = distributed_orthogonalize(
                optimizer,
                P_batch,
                ortho_group=ortho_group,
                oversample=optimizer.defaults["rcqr_oversample"],
                dist_metas=dist_metas,
            )
        else:
            P_ortho_full = orthogonalize_local_matrix_batch(
                optimizer,
                P_batch,
                dist_metas=dist_metas,
                tag="compressed_local",
            )
        yield
        with _dion_math_precision_context():
            R_batch = M_batch.mT @ P_ortho_full
        yield from reduce_r_across_tp(
            optimizer,
            R_batch,
            configs,
            dist_metas,
            batch_collectives=batch_collectives,
        )
        if comm_group is not None and comm_world_size > 1:
            yield from collapse_batch_across_replicas(
                optimizer,
                R_batch,
                replicate_group=comm_group,
            )
        return P_ortho_full[:original_batch_size], R_batch[:original_batch_size]

    if comm_group is not None and comm_world_size > 1:
        yield from collapse_batch_across_replicas(
            optimizer,
            P_batch,
            replicate_group=comm_group,
        )

    use_dense_replicate_batch = (
        comm_world_size > 1
        and ortho_group is None
        and not (configs and all(is_fs_only_config(config) for config in configs))
    )
    if use_dense_replicate_batch:
        P_ortho_full = yield from orthogonalize_dense_replicate_batch_async(
            optimizer,
            P_batch,
            dist_metas,
            comm_group=comm_group,
            cache_scope=global_param_offset,
        )
    else:
        P_ortho_full = yield from orthogonalize_p_batch(
            optimizer,
            P_batch,
            configs,
            dist_metas,
            real_batch_size=original_batch_size,
            ortho_group=ortho_group,
            batch_collectives=batch_collectives,
        )

    with _dion_math_precision_context():
        R_batch = M_batch.mT @ P_ortho_full
    yield from reduce_r_across_tp(
        optimizer,
        R_batch,
        configs,
        dist_metas,
        batch_collectives=batch_collectives,
    )
    if comm_group is not None and comm_world_size > 1:
        yield from collapse_batch_across_replicas(
            optimizer,
            R_batch,
            replicate_group=comm_group,
        )

    P_ortho_full = P_ortho_full[:original_batch_size]
    R_batch = R_batch[:original_batch_size]

    return P_ortho_full, R_batch


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
        use_dense_replicate_batch = (
            replicate_world_size > 1
            and ortho_group is None
            and not (configs and all(is_fs_only_config(config) for config in configs))
        )
        if use_dense_replicate_batch:
            P_batch = yield from orthogonalize_dense_replicate_batch_async(
                optimizer,
                P_batch,
                dist_metas,
                comm_group=replicate_group,
                cache_scope=global_param_offset,
            )
        else:
            P_batch = yield from orthogonalize_p_batch(
                optimizer,
                P_batch,
                configs,
                dist_metas,
                real_batch_size=real_batch_size,
                ortho_group=ortho_group,
                batch_collectives=batch_collectives,
            )
        with _dion_math_precision_context():
            R_batch = M_batch.mT @ P_batch
        yield from reduce_r_across_tp(
            optimizer,
            R_batch,
            configs,
            dist_metas,
            batch_collectives=batch_collectives,
        )
    P_batch, R_batch = sanitize_dion_batch_for_update(
        optimizer,
        P_batch,
        R_batch,
        Q_batch,
        M_batch,
        real_batch_size=real_batch_size,
        global_param_offset=global_param_offset,
        configs=configs,
        dist_metas=dist_metas,
    )

    apply_error_feedback_to_momentum(
        optimizer, momentums, P_batch, R_batch, configs, optim_groups
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
    )

    del M_batch, Q_batch, P_batch, R_batch, Q_new_batch, Q_state_batch
