"""Reference communication-kernel helpers for distributed Dion updates."""

from typing import Generator, List, Optional

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol

from .ortho_runtime import (
    orthogonalize_local_matrix_batch,
    orthogonalize_local_slice,
)
from .runtime import collapse_batch_over_replicate_subset, replicate_reduce_op


def orthogonalize_fs_only_batch_(
    optimizer,
    P_batch: torch.Tensor,
    dist_metas: List,
    *,
    axis_plan=None,
    replicate_group: Optional[torch.distributed.ProcessGroup] = None,
    replicate_ranks: Optional[List[int]] = None,
) -> Generator[torch.Tensor, None, None]:
    """Match `dion_reference.dion_update_fsdp()` for fs-only batches."""
    ortho_plan = axis_plan.fs_orthogonalize if axis_plan is not None else None
    if ortho_plan is None:
        raise RuntimeError(
            "[DION_MISSING_FS_ONLY_ORTHO_PLAN] "
            f"step={optimizer._step_count} rank={optimizer._global_rank}"
        )
    outer_shard_group = ortho_plan.process_group
    outer_shard_world = ortho_plan.world_size
    outer_shard_rank = ortho_plan.rank

    if outer_shard_group is None or outer_shard_world <= 1:
        yield
        return orthogonalize_local_slice(
            optimizer,
            P_batch,
            dist_metas=dist_metas,
            tag="fs_only_local",
            sketch_tag="logical_local",
        )

    batch_size = P_batch.size(0)
    if batch_size != outer_shard_world:
        raise RuntimeError(
            "[DION_FSONLY_BATCH_SIZE_MISMATCH] "
            f"batch_size={batch_size} outer_shard_world_size={outer_shard_world}"
        )

    local_slot = outer_shard_rank
    if len(ortho_plan.indices) != batch_size:
        raise RuntimeError(
            "[DION_FSONLY_ORTHO_PLAN_SIZE_MISMATCH] "
            f"batch_size={batch_size} plan_size={len(ortho_plan.indices)}"
        )
    if outer_shard_rank >= len(ortho_plan.indices):
        raise RuntimeError(
            "[DION_FSONLY_ORTHO_PLAN_RANK_MISMATCH] "
            f"outer_shard_rank={outer_shard_rank} plan_size={len(ortho_plan.indices)}"
        )
    local_slot = ortho_plan.indices[outer_shard_rank]

    local_dist_metas = (
        [dist_metas[local_slot]]
        if dist_metas and local_slot < len(dist_metas)
        else None
    )
    P_single = funcol.reduce_scatter_tensor(
        P_batch.contiguous(),
        reduceOp="sum",
        scatter_dim=0,
        group=outer_shard_group,
    )
    yield

    if replicate_group is not None and dist.get_world_size(replicate_group) > 1:
        if replicate_ranks is not None:
            yield from collapse_batch_over_replicate_subset(
                optimizer,
                P_single,
                primary_group=replicate_group,
                subset_ranks=replicate_ranks,
            )
        else:
            dist.all_reduce(
                P_single,
                op=replicate_reduce_op(optimizer),
                group=replicate_group,
            )
            yield
    # FS-only compressed reference orthogonalizes one full local matrix after RS/collapse.
    P_single_ortho = orthogonalize_local_matrix_batch(
        optimizer,
        P_single,
        dist_metas=local_dist_metas,
        tag="fs_only_local",
    ).contiguous().to(P_single.dtype)

    P_ortho = funcol.all_gather_tensor(
        P_single_ortho.contiguous(),
        gather_dim=0,
        group=outer_shard_group,
    )
    yield
    canonical_slots = tuple(int(idx) for idx in ortho_plan.indices)
    if sorted(canonical_slots) != list(range(batch_size)):
        raise RuntimeError(
            "[DION_FSONLY_GATHER_PERMUTATION_INVALID] "
            f"batch_size={batch_size} indices={canonical_slots}"
        )
    if canonical_slots != tuple(range(batch_size)):
        gathered_p_ortho = P_ortho
        P_ortho = torch.empty_like(gathered_p_ortho)
        for owner_rank, batch_slot in enumerate(canonical_slots):
            P_ortho[batch_slot].copy_(gathered_p_ortho[owner_rank])
    return P_ortho


def orthogonalize_dense_replicate_batch_async_(
    optimizer,
    P_batch: torch.Tensor,
    dist_metas: List,
    *,
    comm_group: torch.distributed.ProcessGroup,
) -> Generator[torch.Tensor, None, None]:
    """Match `dion_reference.py::dion_update_ddp()` for dense replicate orthogonalization."""
    comm_world_size = dist.get_world_size(comm_group)
    if comm_world_size <= 1:
        yield
        return orthogonalize_local_matrix_batch(
            optimizer,
            P_batch,
            dist_metas=dist_metas,
            tag="dense_replicate_local",
        )

    batch_size = P_batch.size(0)
    original_batch_size = batch_size
    if batch_size % comm_world_size != 0:
        pad = comm_world_size - (batch_size % comm_world_size)
        padded_batch_size = batch_size + pad
        P_padded = optimizer._cached_buffer(
            "dense_replicate_p_padded",
            (padded_batch_size, P_batch.size(1), P_batch.size(2)),
            P_batch.dtype,
            P_batch.device,
            zero=True,
        )
        P_padded[:batch_size].copy_(P_batch)
        P_batch = P_padded
        dist_metas = list(dist_metas) + [None] * pad
        batch_size = P_batch.size(0)

    comm_rank = dist.get_rank(comm_group)
    P_ortho_full = optimizer._cached_buffer(
        "dense_replicate_p_ortho_full",
        P_batch.shape,
        P_batch.dtype,
        P_batch.device,
    )

    for chunk_start in range(0, batch_size, comm_world_size):
        chunk_end = chunk_start + comm_world_size
        P_chunk = P_batch[chunk_start:chunk_end].contiguous()
        P_single = P_chunk[comm_rank : comm_rank + 1]
        local_index = chunk_start + comm_rank
        local_dist_metas = (
            dist_metas[local_index : local_index + 1] if dist_metas else None
        )
        P_single_ortho = orthogonalize_local_matrix_batch(
            optimizer,
            P_single,
            dist_metas=local_dist_metas,
            tag="dense_replicate_local",
        )
        P_ortho_chunk = funcol.all_gather_tensor(
            P_single_ortho.contiguous(),
            gather_dim=0,
            group=comm_group,
        )
        yield
        P_ortho_full[chunk_start:chunk_end].copy_(P_ortho_chunk)

    return P_ortho_full[:original_batch_size]
