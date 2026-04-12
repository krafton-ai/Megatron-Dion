"""Package-private batch-update runtime helpers for MegatronDion."""

import logging
from typing import Generator, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
from torch import Tensor

from .kernels import (
    apply_error_feedback as apply_error_feedback_,
    compute_update_batch,
    sanitize_dion_intermediate_batch as sanitize_dion_intermediate_batch_,
    scaled_lr_for_shape,
)
from .ortho import orthogonalize, reshard_q_along_tp
from .ortho_runtime import distributed_orthogonalize, orthogonalize_local_matrix_batch
from .reference_kernels import (
    orthogonalize_dense_replicate_batch_async_,
    orthogonalize_fs_only_batch_,
)
from .runtime import (
    collapse_batch_across_replicas,
    collapse_batch_over_replicate_subset,
    collapse_grads_across_replicas,
    collapse_grads_over_replicate_subset,
    normalize_cols_async,
    reduce_p_across_fs_groups,
    reduce_r_across_tp,
    unshard_q_batch,
)
from .state import has_tp_shard as has_tp_shard_, is_fs_only_config as is_fs_only_config_
from .types import DionBatchCollectives, DionBatchRoute, DionParamConfig
from .utils import get_global_shape


logger = logging.getLogger(__name__)


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
    batch_route: Optional[DionBatchRoute] = None,
    batch_collectives: Optional[DionBatchCollectives] = None,
) -> Generator[None, None, None]:
    """Perform one adapter-authored batched Dion update with async communication."""
    batch_size = len(params)
    if real_batch_size is None:
        real_batch_size = batch_size

    if batch_route is None:
        raise RuntimeError(
            "[DION_MISSING_BATCH_ROUTE] "
            f"step={optimizer._step_count} rank={optimizer._global_rank}"
        )
    if batch_collectives is None:
        raise RuntimeError(
            "[DION_MISSING_BATCH_COLLECTIVES] "
            f"step={optimizer._step_count} rank={optimizer._global_rank}"
        )
    replicate_group = batch_route.replicate_group
    replicate_subset_ranks = (
        list(batch_route.replicate_subset_ranks)
        if batch_route.replicate_subset_ranks is not None
        else None
    )
    ortho_group = batch_route.ortho_group
    replicate_world_size = (
        dist.get_world_size(replicate_group) if replicate_group is not None else 1
    )
    use_compressed = (
        optimizer.use_compressed_comm
        and replicate_world_size > 1
        and any(config.compressed_all_reduce for config in configs)
    )

    if not use_compressed:
        if replicate_subset_ranks is not None:
            yield from collapse_grads_over_replicate_subset(
                optimizer,
                grads,
                primary_group=replicate_group,
                subset_ranks=replicate_subset_ranks,
            )
        else:
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
    )

    M_for_matmul = [
        (momentum.mT if config.is_transposed else momentum).to(torch.float32)
        for momentum, config in zip(momentums, configs)
    ]
    M_batch = torch.stack(M_for_matmul, dim=0)
    Q_batch = torch.stack([q_tensor.to(torch.float32) for q_tensor in Q_for_matmul], dim=0)
    del Q_for_matmul

    P_batch = M_batch @ Q_batch

    if not (configs and all(is_fs_only_config_(config) for config in configs)):
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
            batch_route,
            batch_collectives,
        )
    else:
        use_dense_replicate_batch = (
            replicate_world_size > 1
            and ortho_group is None
            and not (configs and all(is_fs_only_config_(config) for config in configs))
        )
        if use_dense_replicate_batch:
            P_batch = yield from orthogonalize_dense_replicate_batch_async(
                P_batch,
                dist_metas,
                comm_group=replicate_group,
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
        batch_route=batch_route,
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
        configs,
        optim_groups,
        optimizer_states,
        dist_metas,
        param_shapes,
        real_batch_size,
        batch_collectives,
    )

    del M_batch, Q_batch, P_batch, R_batch, Q_new_batch, Q_state_batch


def run_compressed_comm_async(
    optimizer,
    P_batch: torch.Tensor,
    M_batch: torch.Tensor,
    Q_batch: torch.Tensor,
    configs: List[DionParamConfig],
    dist_metas: List,
    batch_route: Optional[DionBatchRoute] = None,
    batch_collectives: Optional[DionBatchCollectives] = None,
) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
    """Run the compressed Dion communication path selected by the adapter."""
    if batch_route is None:
        raise RuntimeError(
            "[DION_MISSING_BATCH_ROUTE] "
            f"step={optimizer._step_count} rank={optimizer._global_rank}"
        )
    if batch_collectives is None:
        raise RuntimeError(
            "[DION_MISSING_BATCH_COLLECTIVES] "
            f"step={optimizer._step_count} rank={optimizer._global_rank}"
        )
    comm_group = batch_route.replicate_group
    comm_subset_ranks = (
        list(batch_route.replicate_subset_ranks)
        if batch_route.replicate_subset_ranks is not None
        else None
    )
    comm_world_size = dist.get_world_size(comm_group) if comm_group else 1
    ortho_group = batch_route.ortho_group
    kernel_kind = batch_route.kernel_kind
    batch_size = P_batch.size(0)
    fs_orthogonalize = (
        batch_collectives.fs_orthogonalize if batch_collectives is not None else None
    )
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
    fs_only_replicate_group = batch_route.compressed_replicate_group
    fs_only_replicate_ranks = (
        list(batch_route.compressed_replicate_ranks)
        if batch_route.compressed_replicate_ranks is not None
        else None
    )

    if kernel_kind == "fsdp" and optimizer.use_fs_collectives and fs_group and fs_group_world > 1:
        P_ortho_full = yield from orthogonalize_fs_only_batch_(
            optimizer,
            P_batch,
            dist_metas,
            axis_plan=batch_collectives,
            replicate_group=fs_only_replicate_group,
            replicate_ranks=fs_only_replicate_ranks,
        )
        R_batch = M_batch.mT @ P_ortho_full
        yield from reduce_r_across_tp(
            optimizer,
            R_batch,
            configs,
            dist_metas,
            batch_collectives=batch_collectives,
        )
        if fs_only_replicate_group is not None:
            if fs_only_replicate_ranks is not None:
                yield from collapse_batch_over_replicate_subset(
                    optimizer,
                    R_batch,
                    primary_group=fs_only_replicate_group,
                    subset_ranks=fs_only_replicate_ranks,
                )
            else:
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
                orthogonalize_mesh=(
                    batch_collectives.orthogonalize_mesh
                    if batch_collectives is not None
                    else None
                ),
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
        if comm_subset_ranks is not None:
            yield from collapse_batch_over_replicate_subset(
                optimizer,
                P_batch,
                primary_group=comm_group,
                subset_ranks=comm_subset_ranks,
            )
        else:
            yield from collapse_batch_across_replicas(
                optimizer,
                P_batch,
                replicate_group=comm_group,
            )
        P_ortho_full = distributed_orthogonalize(
            optimizer,
            P_batch,
            ortho_group=ortho_group,
            orthogonalize_mesh=(
                batch_collectives.orthogonalize_mesh
                if batch_collectives is not None
                else None
            ),
            oversample=optimizer.defaults["rcqr_oversample"],
            dist_metas=dist_metas,
        )
    else:
        if batch_size % comm_world_size != 0:
            pad = comm_world_size - (batch_size % comm_world_size)
            padded_batch_size = batch_size + pad
            P_padded = optimizer._cached_buffer(
                "compressed_p_padded",
                (padded_batch_size, P_batch.size(1), P_batch.size(2)),
                P_batch.dtype,
                P_batch.device,
                zero=True,
            )
            M_padded = optimizer._cached_buffer(
                "compressed_m_padded",
                (padded_batch_size, M_batch.size(1), M_batch.size(2)),
                M_batch.dtype,
                M_batch.device,
                zero=True,
            )
            Q_padded = optimizer._cached_buffer(
                "compressed_q_padded",
                (padded_batch_size, Q_batch.size(1), Q_batch.size(2)),
                Q_batch.dtype,
                Q_batch.device,
                zero=True,
            )
            P_padded[:batch_size].copy_(P_batch)
            M_padded[:batch_size].copy_(M_batch)
            Q_padded[:batch_size].copy_(Q_batch)
            P_batch = P_padded
            M_batch = M_padded
            Q_batch = Q_padded
            if isinstance(configs, list):
                dummy_cfg = DionParamConfig()
                configs = list(configs) + [dummy_cfg] * pad
            if isinstance(dist_metas, list):
                dist_metas = list(dist_metas) + [None] * pad
            batch_size = P_batch.size(0)

        comm_rank = dist.get_rank(comm_group)
        P_ortho_full = optimizer._cached_buffer(
            "compressed_p_ortho_full",
            P_batch.shape,
            P_batch.dtype,
            P_batch.device,
        )

        for chunk_start in range(0, batch_size, comm_world_size):
            chunk_end = chunk_start + comm_world_size
            P_chunk = P_batch[chunk_start:chunk_end].contiguous()
            P_single = funcol.reduce_scatter_tensor(
                P_chunk,
                reduceOp="avg",
                scatter_dim=0,
                group=comm_group,
            )
            yield

            if P_single.size(0) != 1:
                raise RuntimeError(
                    "[DION_INVALID_DENSE_COMPRESSED_LOCAL_CHUNK] "
                    f"step={optimizer._step_count} comm_world_size={comm_world_size} "
                    f"local_chunk={P_single.size(0)} chunk_start={chunk_start}"
                )

            local_index = chunk_start + comm_rank
            local_metas = dist_metas[local_index : local_index + 1] if dist_metas else None
            P_ortho_single = orthogonalize_local_matrix_batch(
                optimizer,
                P_single,
                dist_metas=local_metas,
                tag="compressed_local",
            )
            P_ortho_chunk = funcol.all_gather_tensor(
                P_ortho_single.contiguous(),
                gather_dim=0,
                group=comm_group,
            )
            yield
            P_ortho_full[chunk_start:chunk_end].copy_(P_ortho_chunk)

    R_batch = M_batch.mT @ P_ortho_full
    yield from reduce_r_across_tp(
            optimizer,
            R_batch,
            configs,
            dist_metas,
            batch_collectives=batch_collectives,
        )
    if kernel_kind == "fsdp" and optimizer.use_fs_collectives and fs_group and dist.get_world_size(fs_group) > 1:
        if fs_only_replicate_ranks is not None:
            yield from collapse_batch_over_replicate_subset(
                optimizer,
                R_batch,
                primary_group=fs_only_replicate_group,
                subset_ranks=fs_only_replicate_ranks,
            )
        else:
            yield from collapse_batch_across_replicas(
                optimizer,
                R_batch,
                replicate_group=fs_only_replicate_group,
            )
    else:
        if comm_subset_ranks is not None:
            yield from collapse_batch_over_replicate_subset(
                optimizer,
                R_batch,
                primary_group=comm_group,
                subset_ranks=comm_subset_ranks,
            )
        else:
            yield from collapse_batch_across_replicas(
                optimizer,
                R_batch,
                replicate_group=comm_group,
            )

    P_ortho_full = P_ortho_full[:original_batch_size]
    R_batch = R_batch[:original_batch_size]

    return P_ortho_full, R_batch


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
    if configs and all(is_fs_only_config_(config) for config in configs):
        P_batch = yield from orthogonalize_fs_only_batch_(
            optimizer,
            P_batch,
            dist_metas,
            axis_plan=batch_collectives,
        )
        return P_batch

    if ortho_group is not None:
        P_batch = distributed_orthogonalize(
            optimizer,
            P_batch,
            ortho_group=ortho_group,
            orthogonalize_mesh=(
                batch_collectives.orthogonalize_mesh
                if batch_collectives is not None
                else None
            ),
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


orthogonalize_dense_replicate_batch_async = orthogonalize_dense_replicate_batch_async_


def apply_batch_updates(
    optimizer,
    params: List[Tensor],
    momentums: List[Tensor],
    Qs: List[Tensor],
    Q_new_batch: torch.Tensor,
    Q_state_batch: torch.Tensor,
    P_batch: torch.Tensor,
    configs: List[DionParamConfig],
    optim_groups: List[dict],
    optimizer_states: List[dict],
    dist_metas: List,
    param_shapes: List[Tuple[int, int]],
    real_batch_size: int,
    batch_collectives: Optional[DionBatchCollectives] = None,
) -> None:
    """Apply weight decay, Dion delta update, and TP re-sharding for Q."""
    if real_batch_size <= 0:
        return

    update_contract_rows = []
    for index in range(real_batch_size):
        optimizer_state = optimizer_states[index]
        dist_meta = dist_metas[index]
        param_shape = param_shapes[index]
        true_global_shape = tuple(
            int(dim) for dim in optimizer_state.get(
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

    reference_contract = {
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
        if any(row[key] != reference_contract[key] for key in reference_contract)
    ]
    if mismatched_rows:
        raise RuntimeError(
            "[DION_BATCH_UPDATE_CONTRACT_MISMATCH] "
            f"step={optimizer._step_count} rank={optimizer._global_rank} "
            f"expected={reference_contract} mismatched_rows={mismatched_rows}"
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
    tp_q_writeback_by_index = {
        idx: plan for plan in batch_collectives.tp_q_reshards for idx in plan.indices
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
        tp_writeback_plan = tp_q_writeback_by_index.get(index)
        if tp_writeback_plan is not None:
            tp_group = tp_writeback_plan.process_group
            if tp_group is None:
                raise RuntimeError(
                    "[DION_MISSING_TP_GROUP_FOR_Q_RESHARD] "
                    f"step={optimizer._step_count} rank={optimizer._global_rank} "
                    f"param_uid={getattr(dist_metas[index], 'param_uid', None) if dist_metas is not None else None}"
                )
            q_state = reshard_q_along_tp(q_state, tp_group, tp_writeback_plan.rank)
        elif optimizer.is_distributed_mode and has_tp_shard_(configs[index]):
            raise RuntimeError(
                "[DION_MISSING_TP_Q_WRITEBACK_PLAN] "
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
) -> Tuple[Tensor, Tensor]:
    """Apply reference-aligned NaN/zero sanitization and optimizer-local warning logs."""
    P_batch, R_batch, unexpected_nan, _ = sanitize_dion_intermediate_batch_(
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
                bad_slots.append(
                    {
                        "slot": int(index),
                        "global_offset": int(global_param_offset + index),
                        "is_real": bool(real_batch_size is None or index < real_batch_size),
                        "shape": tuple(P_batch[index].shape),
                        "is_tr": bool(config.is_transposed) if config is not None else None,
                        "tp": getattr(config, "tp_shard_dim", None)
                        if config is not None
                        else None,
                        "fs": getattr(config, "fs_shard_dim", None)
                        if config is not None
                        else None,
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
    apply_error_feedback_(
        momentums,
        P_batch,
        R_batch,
        configs,
        optim_groups,
        default_mu=optimizer.defaults["mu"],
    )
