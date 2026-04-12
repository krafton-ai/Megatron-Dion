"""Batch-order synchronization for distributed Dion batch grouping."""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch.distributed as dist

from ..dion.batching import (
    batch_member_id,
    canonical_reorder_indices,
    canonicalize_gathered_sequences,
    deserialize_batch_key,
    serialize_batch_key,
)
from ..dion.types import DionBatchGroup

def sync_batch_keys_(
    *,
    local_batch_keys: List[Tuple],
    sync_group,
    batch_key_cache: Dict,
) -> List[Tuple]:
    """Return the canonical cross-rank batch-key order for one sync group."""
    if sync_group is None or dist.get_world_size(sync_group) <= 1:
        return local_batch_keys

    cache_key = (id(sync_group), tuple(local_batch_keys))
    if cache_key in batch_key_cache:
        return batch_key_cache[cache_key]

    local_keys_serialized = [serialize_batch_key(sk) for sk in local_batch_keys]
    gathered_keys_list = [None] * dist.get_world_size(sync_group)
    dist.all_gather_object(gathered_keys_list, local_keys_serialized, group=sync_group)

    canonical_keys, mismatch_ranks = canonicalize_gathered_sequences(gathered_keys_list)
    if mismatch_ranks:
        try:
            group_ranks = dist.get_process_group_ranks(sync_group)
        except Exception:
            group_ranks = []
        raise RuntimeError(
            "[DION_BATCH_KEY_MISMATCH] "
            f"group_ranks={group_ranks} mismatch_local_ranks={mismatch_ranks} "
            f"local_batch_keys={local_keys_serialized} gathered_batch_keys={gathered_keys_list}"
        )

    ordered_keys = [deserialize_batch_key(key_data) for key_data in canonical_keys]
    batch_key_cache[cache_key] = ordered_keys
    return ordered_keys


def align_group_data_order_(
    *,
    batch_group: DionBatchGroup,
    sync_group,
) -> None:
    """Reorder one shape-group's param lists to the canonical cross-rank order."""
    if sync_group is None or dist.get_world_size(sync_group) <= 1:
        return

    local_ids = [batch_member_id(dist_meta) for dist_meta in (batch_group.dist_metas or [])]
    gathered_ids = [None] * dist.get_world_size(sync_group)
    dist.all_gather_object(gathered_ids, local_ids, group=sync_group)

    canonical_ids, mismatch_ranks = canonicalize_gathered_sequences(gathered_ids)
    if mismatch_ranks:
        try:
            group_ranks = dist.get_process_group_ranks(sync_group)
        except Exception:
            group_ranks = []
        raise RuntimeError(
            "[DION_PARAM_ID_CONTENT_MISMATCH] "
            f"group_ranks={group_ranks} mismatch_local_ranks={mismatch_ranks} "
            f"local_ids={local_ids} gathered_ids={gathered_ids}"
        )

    if tuple(local_ids) == canonical_ids:
        return

    reorder_indices = canonical_reorder_indices(
        local_ids,
        canonical_ids,
        error_tag="DION_PARAM_ID_REORDER_MISMATCH",
    )
    batch_group.params = [batch_group.params[idx] for idx in reorder_indices]
    batch_group.grads = [batch_group.grads[idx] for idx in reorder_indices]
    batch_group.optimizer_states = [batch_group.optimizer_states[idx] for idx in reorder_indices]
    batch_group.optim_groups = [batch_group.optim_groups[idx] for idx in reorder_indices]
    batch_group.configs = [batch_group.configs[idx] for idx in reorder_indices]
    batch_group.dist_metas = [batch_group.dist_metas[idx] for idx in reorder_indices]


def validate_distributed_batch_group_cardinality_(
    *,
    batch_key,
    batch_group: DionBatchGroup,
    batch_route,
) -> None:
    """Validate per-sync-group local batch cardinality before batch assembly."""
    local_num_params = len(batch_group.params or [])
    for sync_group in batch_route.sync_groups:
        world_size = dist.get_world_size(sync_group)
        gathered_counts = [None] * world_size
        dist.all_gather_object(gathered_counts, local_num_params, group=sync_group)
        mismatch_ranks = [
            idx for idx, rank_count in enumerate(gathered_counts)
            if int(rank_count) != int(local_num_params)
        ]
        if mismatch_ranks:
            try:
                group_ranks = dist.get_process_group_ranks(sync_group)
            except Exception:
                group_ranks = []
            raise RuntimeError(
                "[DION_PARAM_COUNT_MISMATCH] "
                f"batch_key={batch_key} group_ranks={group_ranks} "
                f"mismatch_local_ranks={mismatch_ranks} gathered_counts={gathered_counts}"
            )


def _validate_param_batches(
    *,
    batch_key,
    batch_group: DionBatchGroup,
    batch_route,
) -> None:
    """Validate per-batch canonical ids on the adapter-authored ordered data."""
    batch_size = batch_route.batch_world_size
    if batch_size <= 0:
        raise RuntimeError(
            "[DION_INVALID_BATCH_WORLD_SIZE] "
            f"batch_key={batch_key} batch_world_size={batch_size}"
        )

    local_num_params = len(batch_group.params or [])
    for batch_start in range(0, local_num_params, batch_size):
        batch_end = min(batch_start + batch_size, local_num_params)
        local_batch_ids = [
            batch_member_id(dist_meta)
            for dist_meta in (batch_group.dist_metas or [])[batch_start:batch_end]
        ]
        for sync_group in batch_route.sync_groups:
            world_size = dist.get_world_size(sync_group)
            gathered_batch_ids = [None] * world_size
            dist.all_gather_object(gathered_batch_ids, local_batch_ids, group=sync_group)
            canonical_batch_ids = tuple(gathered_batch_ids[0])
            mismatch_ranks = [
                idx for idx, rank_ids in enumerate(gathered_batch_ids)
                if tuple(rank_ids) != canonical_batch_ids
            ]
            if mismatch_ranks:
                try:
                    group_ranks = dist.get_process_group_ranks(sync_group)
                except Exception:
                    group_ranks = []
                raise RuntimeError(
                    "[DION_BATCH_ID_MISMATCH] "
                    f"batch_key={batch_key} batch_start={batch_start} batch_end={batch_end} "
                    f"group_ranks={group_ranks} mismatch_local_ranks={mismatch_ranks} "
                    f"local_batch_ids={local_batch_ids} gathered_batch_ids={gathered_batch_ids}"
                )
