"""Runtime validation helpers for distributed Dion bootstrap."""

from __future__ import annotations

from typing import Callable

import torch
import torch.distributed as dist

def _validate_step_groups(
    *,
    full_data_parallel_group,
    tp_group,
    rp_group,
    fs_group,
    state_replica_group,
    route_step_params_fn: Callable,
    group_size_fn: Callable,
    group_rank_fn: Callable,
    use_compressed_comm: bool,
    use_fs_collectives: bool,
) -> Callable:
    """Validate distributed bootstrap inputs and return the step-routing callback."""
    global_rank = dist.get_rank()

    if dist.is_initialized() and full_data_parallel_group is not None:
        have_rp_arg = torch.tensor(
            [1 if rp_group is not None else 0],
            device=torch.cuda.current_device(),
            dtype=torch.int64,
        )
        have_fs_arg = torch.tensor(
            [1 if fs_group is not None else 0],
            device=torch.cuda.current_device(),
            dtype=torch.int64,
        )
        have_state_replica_arg = torch.tensor(
            [1 if state_replica_group is not None else 0],
            device=torch.cuda.current_device(),
            dtype=torch.int64,
        )
        dist.all_reduce(have_rp_arg, op=dist.ReduceOp.MIN, group=full_data_parallel_group)
        dist.all_reduce(have_fs_arg, op=dist.ReduceOp.MIN, group=full_data_parallel_group)
        dist.all_reduce(
            have_state_replica_arg,
            op=dist.ReduceOp.MIN,
            group=full_data_parallel_group,
        )

        if int(have_rp_arg.item()) != 1 and rp_group is not None:
            raise RuntimeError(
                f"Global rank {global_rank}: Inconsistent rp_group provision! "
                f"This rank received rp_group, but some DP ranks did not (MIN=0). "
                f"Ensure rp_group is provided uniformly to all ranks."
            )
        if int(have_fs_arg.item()) != 1 and fs_group is not None:
            raise RuntimeError(
                f"Global rank {global_rank}: Inconsistent fs_group provision! "
                f"This rank received fs_group, but some DP ranks did not (MIN=0). "
                f"Ensure fs_group is provided uniformly to all ranks."
            )
        if int(have_state_replica_arg.item()) != 1 and state_replica_group is not None:
            raise RuntimeError(
                f"Global rank {global_rank}: Inconsistent state_replica_group provision! "
                f"This rank received state_replica_group, but some DP ranks did not (MIN=0)."
            )

    def _validate_group_membership(group, label):
        if group is None:
            return 1, 0
        world_size = group_size_fn(group)
        rank = group_rank_fn(group)
        group_ranks = dist.get_process_group_ranks(group)
        if len(group_ranks) != world_size or dist.get_rank() not in group_ranks:
            raise RuntimeError(
                f"Global rank {global_rank}: invalid {label} membership: "
                f"size={world_size} ranks={group_ranks}"
            )
        return world_size, rank

    fs_world_size, _ = _validate_group_membership(fs_group, "fs_group")
    rp_world_size, _ = _validate_group_membership(rp_group, "rp_group")
    _validate_group_membership(state_replica_group, "state_replica_group")
    _validate_group_membership(tp_group, "tp_group")

    if dist.is_initialized() and full_data_parallel_group is not None:
        world_size = dist.get_world_size(full_data_parallel_group)
        if world_size > 1:
            local_config = {
                'use_compressed_comm': bool(use_compressed_comm),
                'use_fs_collectives': bool(use_fs_collectives),
                'rp_group_size': rp_world_size if rp_group is not None else 0,
                'fs_group_size': fs_world_size if fs_group is not None else 0,
            }
            gathered_configs = [None] * world_size
            dist.all_gather_object(
                gathered_configs,
                local_config,
                group=full_data_parallel_group,
            )
            for rank_idx, rank_config in enumerate(gathered_configs):
                if rank_config != gathered_configs[0]:
                    raise ValueError(
                        "Dion config mismatch! "
                        f"Rank 0 config: {gathered_configs[0]}, "
                        f"Rank {rank_idx} config: {rank_config}. "
                        "All ranks must have identical use_compressed_comm, "
                        "use_fs_collectives, AND same rp_group_size/fs_group_size. "
                        "This ensures identical distributed Dion execution."
                    )

    return route_step_params_fn
