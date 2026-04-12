"""Adapter-owned topology/bootstrap validation helpers for distributed Dion enablement."""

from __future__ import annotations

import torch
import torch.distributed as dist


def validate_uniform_fs_topology_(
    *,
    fs_group,
    dp_group,
    global_rank: int,
) -> None:
    """Fail fast unless every DP rank has the authoritative FS group."""
    have_fs = torch.tensor(
        [1 if fs_group is not None else 0],
        device=torch.cuda.current_device(),
        dtype=torch.int64,
    )
    dist.all_reduce(have_fs, op=dist.ReduceOp.MIN, group=dp_group)
    all_have_fs = int(have_fs.item()) == 1

    if all_have_fs:
        return
    if fs_group is not None:
        raise RuntimeError(
            f"Global rank {global_rank}: FS groups exist only on subset of ranks! "
            f"Ensure rp_group/fs_group are provided uniformly to prevent new_group() mismatch."
        )
    raise RuntimeError(
        f"Global rank {global_rank}: No FS group found! "
        f"FS group must be provided from the standard Megatron-Core optimizer topology."
    )


def validate_enabled_rp_topology_(
    *,
    expected_rp_size: int,
    rp_group,
    data_parallel_group,
    global_rank: int,
    dist_metas_sharded,
    logger_error_fn,
) -> None:
    """Fail fast unless RP topology and Dion eligibility are uniform across RP groups."""
    if expected_rp_size <= 1:
        return

    have_rp = torch.tensor(
        [1 if rp_group is not None else 0],
        device=torch.cuda.current_device(),
        dtype=torch.int64,
    )
    dist.all_reduce(have_rp, op=dist.ReduceOp.MIN, group=data_parallel_group)
    all_have_rp = int(have_rp.item()) == 1
    if not all_have_rp:
        raise RuntimeError(
            f"[Dion] Global rank {global_rank}: not all DP ranks have rp_group "
            f"(MIN={have_rp.item()}); RP topology must be identical across the data-parallel group"
        )
    if rp_group is None:
        raise RuntimeError(
            f"Global rank {global_rank}: all_have_rp=True but rp_group is None! "
            f"This indicates a bug in group creation or collective voting logic."
        )

    my_dion_count = sum(
        1 for dist_meta in dist_metas_sharded.values() if dist_meta.is_dion_param
    )
    my_cnt_tensor = torch.tensor(
        [my_dion_count],
        device=torch.cuda.current_device(),
        dtype=torch.int64,
    )

    rp_world_size = dist.get_world_size(rp_group)
    gathered = [torch.zeros_like(my_cnt_tensor) for _ in range(rp_world_size)]
    dist.all_gather(gathered, my_cnt_tensor, group=rp_group)

    gathered_counts = [int(t.item()) for t in gathered]
    if all(count == my_dion_count for count in gathered_counts):
        return

    my_dion_offsets = sorted(
        [
            dist_meta.param_uid
            for dist_meta in dist_metas_sharded.values()
            if dist_meta.is_dion_param
        ]
    )
    gathered_offsets = [None] * rp_world_size
    dist.all_gather_object(gathered_offsets, my_dion_offsets, group=rp_group)

    for rp_rank, offsets in enumerate(gathered_offsets):
        if offsets != my_dion_offsets:
            my_set = set(my_dion_offsets)
            other_set = set(offsets)
            only_mine = my_set - other_set
            only_other = other_set - my_set
            logger_error_fn(
                "[Dion] RP rank %s differs from me: Only in mine: %s, Only in theirs: %s",
                rp_rank,
                sorted(only_mine)[:5],
                sorted(only_other)[:5],
            )

    raise RuntimeError(
        f"CRITICAL: Dion eligibility mismatch within RP group! "
        f"My Dion count: {my_dion_count}, RP group counts: {gathered_counts}. "
        f"This will cause collective operation hangs. "
        f"DistributedOptimizer did uniform sharding across DP (RP×FS), "
        f"so RP group members have different param chunks. "
        f"Consider disabling DistributedOptimizer sharding or implementing custom FS-aware sharding."
    )
