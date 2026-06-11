"""Shared ownership helpers for row-split optimizer children."""

from __future__ import annotations

from dataclasses import dataclass

import torch.distributed as dist

from megatron.core import parallel_state

from ..sharding import compute_fs_shard_range


_GROUP_CACHE: dict[tuple[int, ...], object] = {}
_GROUP_READY: set[tuple[int, ...]] = set()
_PENDING_GROUPS: set[tuple[int, ...]] = set()


@dataclass(frozen=True)
class RowChildLayout:
    group: object
    world_size: int
    rank: int
    start_idx: int
    end_idx: int
    row_sizes: tuple[int, ...]

    def as_tuple(self):
        return (
            self.group,
            self.world_size,
            self.rank,
            self.start_idx,
            self.end_idx,
            self.row_sizes,
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
    group_desc: str,
    namespace: str = "MATRIX",
) -> RowChildLayout:
    """Return child-local ownership for a row-sharded split child."""
    namespace = str(namespace)
    parent_world_size = int(parent_world_size)
    parent_rank = int(parent_rank)
    child_rows = int(child_rows)
    if parent_group is None:
        raise RuntimeError(f"[{namespace}_{error_prefix}_MISSING_PARENT_{label}_GROUP] {detail}")
    if parent_rank < 0 or parent_rank >= parent_world_size:
        raise RuntimeError(
            f"[{namespace}_{error_prefix}_INVALID_PARENT_{label}_RANK] "
            f"{detail} rank={parent_rank} world_size={parent_world_size}"
        )

    parent_group_ranks = tuple(int(rank) for rank in dist.get_process_group_ranks(parent_group))
    if len(parent_group_ranks) != parent_world_size:
        raise RuntimeError(
            f"[{namespace}_{error_prefix}_PARENT_{label}_GROUP_SIZE_MISMATCH] "
            f"{detail} meta_world_size={parent_world_size} "
            f"actual_group_size={len(parent_group_ranks)}"
        )

    member_global_ranks = []
    member_parent_ranks = []
    member_ranges = []
    if len(child_ranges) != parent_world_size:
        raise RuntimeError(
            f"[{namespace}_{error_prefix}_{label}_RANGE_COUNT_MISMATCH] "
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
        raise RuntimeError(f"[{namespace}_{error_prefix}_NO_{label}_OWNERS] {detail}")
    if member_ranges[0][0] != 0 or member_ranges[-1][1] != child_rows:
        raise RuntimeError(
            f"[{namespace}_{error_prefix}_{label}_COVERAGE_MISMATCH] "
            f"{detail} child_rows={child_rows} member_ranges={member_ranges}"
        )
    for prev_range, next_range in zip(member_ranges, member_ranges[1:]):
        if prev_range[1] != next_range[0]:
            raise RuntimeError(
                f"[{namespace}_{error_prefix}_{label}_NONCONTIGUOUS_COVERAGE] "
                f"{detail} member_ranges={member_ranges}"
            )

    child_ranks = tuple(member_global_ranks)
    child_group = parent_group if child_ranks == parent_group_ranks else _row_child_group(
        child_ranks,
        create_group=bool(create_group),
        group_desc=group_desc,
        namespace=namespace,
        sync_group=parent_group,
    )
    row_sizes = tuple(int(end_idx - start_idx) for start_idx, end_idx in member_ranges)
    if parent_rank not in member_parent_ranks:
        return RowChildLayout(child_group, len(member_parent_ranks), -1, -1, -1, row_sizes)

    child_rank = member_parent_ranks.index(parent_rank)
    start_idx, end_idx = member_ranges[child_rank]
    return RowChildLayout(
        child_group,
        len(member_parent_ranks),
        int(child_rank),
        int(start_idx),
        int(end_idx),
        row_sizes,
    )


def _row_child_group(
    ranks: tuple[int, ...],
    *,
    create_group: bool,
    group_desc: str = "MATRIX_SPLIT_CHILD_GROUP",
    namespace: str = "MATRIX",
    sync_group=None,
) -> object:
    """Return a cached child group, or register it for finalization."""
    del sync_group
    namespace = str(namespace)
    ranks = tuple(int(rank) for rank in ranks)
    if len(ranks) <= 1:
        return None
    rank = dist.get_rank()
    group = _GROUP_CACHE.get(ranks)
    if group is not None:
        return group
    if ranks in _GROUP_READY:
        return None
    if not create_group:
        if rank not in ranks:
            return None
        raise RuntimeError(f"[{namespace}_SPLIT_CHILD_GROUP_NOT_PREPARED] ranks={ranks}")
    _PENDING_GROUPS.add(ranks)
    return None


def finalize_row_child_groups(group_desc: str = "MATRIX_SPLIT_CHILD_GROUP") -> None:
    """Create registered split-child groups in one deterministic global order."""
    if not dist.is_available() or not dist.is_initialized():
        _PENDING_GROUPS.clear()
        return
    world_size = dist.get_world_size()
    gathered: list[object] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, tuple(sorted(_PENDING_GROUPS)))
    group_specs = sorted(
        {
            tuple(int(rank) for rank in ranks)
            for rank_list in gathered
            if rank_list is not None
            for ranks in rank_list
            if len(tuple(ranks)) > 1
        }
    )
    rank = dist.get_rank()
    for ranks in group_specs:
        if ranks in _GROUP_READY:
            continue
        group = parallel_state.create_group(
            list(ranks),
            use_local_synchronization=False,
            group_desc=group_desc,
        )
        if rank in ranks:
            _GROUP_CACHE[ranks] = group
        _GROUP_READY.add(ranks)
    _PENDING_GROUPS.clear()


def resolve_child_shard_layout(
    *,
    parent_group,
    parent_world_size: int,
    parent_rank: int,
    parent_shard_dim: int,
    parent_start_idx: int,
    parent_end_idx: int,
    parent_rows: int,
    child_rows: int,
    split_axis: int,
    child_range,
    label: str,
    detail: str,
    error_prefix: str,
    create_group: bool,
    group_desc: str = "MATRIX_SPLIT_CHILD_GROUP",
    namespace: str = "MATRIX",
    child_rank_range=None,
) -> tuple[object, int, int, int, int, tuple[int, ...] | None]:
    """Return the child ownership layout for one sharded axis."""
    parent_world_size = int(parent_world_size)
    parent_rank = int(parent_rank)
    parent_shard_dim = int(parent_shard_dim)
    split_axis = int(split_axis)
    if parent_shard_dim != split_axis or parent_world_size <= 1:
        return (
            parent_group,
            parent_world_size,
            parent_rank,
            int(parent_start_idx),
            int(parent_end_idx),
            None,
        )

    child_ranges = []
    for rank_idx in range(parent_world_size):
        if child_rank_range is not None:
            rank_range = child_rank_range(parent_world_size, rank_idx)
        else:
            parent_rank_start, parent_rank_end = compute_fs_shard_range(
                int(parent_rows),
                parent_world_size,
                rank_idx,
            )
            rank_range = child_range(parent_rank_start, parent_rank_end)
        child_ranges.append(rank_range)

    return resolve_row_child_layout(
        parent_group=parent_group,
        parent_world_size=parent_world_size,
        parent_rank=parent_rank,
        child_rows=int(child_rows),
        child_ranges=tuple(child_ranges),
        label=label,
        detail=detail,
        error_prefix=error_prefix,
        create_group=create_group,
        group_desc=group_desc,
        namespace=namespace,
    ).as_tuple()


def resolve_child_layouts(
    parent_meta,
    *,
    child_kind,
    child_global_shape,
    split_kind: str,
    split_axis: int,
    child_range,
    child_rank_range=None,
    create_group: bool,
    group_desc: str = "MATRIX_SPLIT_CHILD_GROUP",
    error_prefix: str = "SPLIT_CHILD",
    namespace: str = "MATRIX",
) -> tuple | None:
    """Return FS/TP ownership layouts for one split child, or None off-owner."""
    split_axis = int(split_axis)
    parent_global_shape = tuple(int(dim) for dim in parent_meta.global_shape)
    detail = (
        f"param_uid={parent_meta.param_uid} param_name={parent_meta.param_name} "
        f"split_kind={split_kind} child_kind={child_kind}"
    )
    fs_layout = resolve_child_shard_layout(
        parent_group=getattr(parent_meta, "fs_group", None),
        parent_world_size=int(getattr(parent_meta, "fs_world_size", 1)),
        parent_rank=int(getattr(parent_meta, "fs_rank", 0)),
        parent_shard_dim=int(getattr(parent_meta, "fs_shard_dim", -1)),
        parent_start_idx=int(getattr(parent_meta, "fs_start_idx", -1)),
        parent_end_idx=int(getattr(parent_meta, "fs_end_idx", -1)),
        parent_rows=int(parent_global_shape[split_axis]),
        child_rows=int(child_global_shape[split_axis]),
        split_axis=split_axis,
        child_range=child_range,
        label="FS",
        detail=detail,
        error_prefix=error_prefix,
        create_group=bool(create_group),
        group_desc=group_desc,
        namespace=namespace,
    )
    tp_layout = resolve_child_shard_layout(
        parent_group=getattr(parent_meta, "tp_group", None),
        parent_world_size=int(getattr(parent_meta, "tp_world_size", 1)),
        parent_rank=int(getattr(parent_meta, "tp_rank", 0)),
        parent_shard_dim=int(getattr(parent_meta, "tp_shard_dim", -1)),
        parent_start_idx=-1,
        parent_end_idx=-1,
        parent_rows=int(parent_global_shape[split_axis]),
        child_rows=int(child_global_shape[split_axis]),
        split_axis=split_axis,
        child_range=child_range,
        child_rank_range=child_rank_range,
        label="TP",
        detail=detail,
        error_prefix=error_prefix,
        create_group=bool(create_group),
        group_desc=group_desc,
        namespace=namespace,
    )
    if int(fs_layout[2]) < 0 or int(tp_layout[2]) < 0:
        return None
    return fs_layout, tp_layout


def child_row_layout(fs_layout, tp_layout):
    """Return the active row layout from FS/TP child ownership layouts."""
    fs_sizes = fs_layout[5]
    tp_sizes = tp_layout[5]
    if fs_sizes is not None and tp_sizes is not None:
        raise RuntimeError("[MATRIX_SPLIT_CHILD_MULTIPLE_ROW_SHARDS]")
    if tp_sizes is not None:
        return int(tp_layout[3]), int(tp_layout[4]), tuple(int(size) for size in tp_sizes)
    if fs_sizes is not None:
        return int(fs_layout[3]), int(fs_layout[4]), tuple(int(size) for size in fs_sizes)
    return -1, -1, None
