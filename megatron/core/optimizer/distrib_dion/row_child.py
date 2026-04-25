"""Shared split-child row-group helpers."""

from __future__ import annotations

from dataclasses import dataclass

import torch.distributed as dist


@dataclass(frozen=True)
class RowChildLayout:
    group: object
    world_size: int
    rank: int
    start_idx: int
    end_idx: int
    row_shard_sizes: tuple[int, ...]

    def as_tuple(self):
        return (
            self.group,
            self.world_size,
            self.rank,
            self.start_idx,
            self.end_idx,
            self.row_shard_sizes,
        )


def resolve_row_child_layout(
    *,
    parent_group,
    parent_world_size: int,
    parent_rank: int,
    child_rows: int,
    child_ranges,
    label: str,
    detail: str,
    error_prefix: str,
    create_group: bool,
    make_group,
) -> RowChildLayout:
    parent_world_size = int(parent_world_size)
    parent_rank = int(parent_rank)
    child_rows = int(child_rows)
    if parent_group is None:
        raise RuntimeError(f"[DION_{error_prefix}_MISSING_PARENT_{label}_GROUP] {detail}")
    if parent_rank < 0 or parent_rank >= parent_world_size:
        raise RuntimeError(
            f"[DION_{error_prefix}_INVALID_PARENT_{label}_RANK] "
            f"{detail} rank={parent_rank} world_size={parent_world_size}"
        )

    parent_group_ranks = tuple(int(rank) for rank in dist.get_process_group_ranks(parent_group))
    if len(parent_group_ranks) != parent_world_size:
        raise RuntimeError(
            f"[DION_{error_prefix}_PARENT_{label}_GROUP_SIZE_MISMATCH] "
            f"{detail} meta_world_size={parent_world_size} "
            f"actual_group_size={len(parent_group_ranks)}"
        )

    member_global_ranks: list[int] = []
    member_parent_ranks: list[int] = []
    member_ranges: list[tuple[int, int]] = []
    if len(child_ranges) != parent_world_size:
        raise RuntimeError(
            f"[DION_{error_prefix}_{label}_RANGE_COUNT_MISMATCH] "
            f"{detail} range_count={len(child_ranges)} world_size={parent_world_size}"
        )
    for rank_idx, child_range in enumerate(child_ranges):
        if child_range is None:
            continue
        start_idx, end_idx = (int(dim) for dim in child_range)
        if end_idx <= start_idx:
            continue
        member_parent_ranks.append(int(rank_idx))
        member_global_ranks.append(int(parent_group_ranks[rank_idx]))
        member_ranges.append((start_idx, end_idx))

    if not member_global_ranks:
        raise RuntimeError(f"[DION_{error_prefix}_NO_{label}_OWNERS] {detail}")
    if member_ranges[0][0] != 0 or member_ranges[-1][1] != child_rows:
        raise RuntimeError(
            f"[DION_{error_prefix}_{label}_COVERAGE_MISMATCH] "
            f"{detail} child_global_rows={child_rows} member_ranges={member_ranges}"
        )
    for prev_range, next_range in zip(member_ranges, member_ranges[1:]):
        if prev_range[1] != next_range[0]:
            raise RuntimeError(
                f"[DION_{error_prefix}_{label}_NONCONTIGUOUS_COVERAGE] "
                f"{detail} member_ranges={member_ranges}"
            )

    child_ranks = tuple(member_global_ranks)
    if child_ranks == parent_group_ranks:
        child_group = parent_group
    else:
        child_group = make_group(
            child_ranks,
            create_group=bool(create_group),
        )
    child_world_size = len(member_parent_ranks)
    row_shard_sizes = tuple(int(end_idx - start_idx) for start_idx, end_idx in member_ranges)

    if parent_rank not in member_parent_ranks:
        return RowChildLayout(child_group, child_world_size, -1, -1, -1, row_shard_sizes)

    child_rank = member_parent_ranks.index(parent_rank)
    start_idx, end_idx = member_ranges[child_rank]
    return RowChildLayout(
        child_group,
        child_world_size,
        int(child_rank),
        int(start_idx),
        int(end_idx),
        row_shard_sizes,
    )
