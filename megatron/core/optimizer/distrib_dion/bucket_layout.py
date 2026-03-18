"""Bucket layout helpers for Dion transport under standard DO ownership."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import FrozenSet, List, Tuple

import torch

from .fs_layout import compute_fs_flat_segments, compute_fs_shard_range
from .shard_info import DionShardLayout


@dataclass(frozen=True)
class DionBucketEntry:
    """Bucket-local Dion shard layout for one parameter."""

    param: torch.nn.Parameter
    shard_layout: DionShardLayout
    size_per_rank: int
    shard_capacity: int
    shard_offset: int
    canonical_bucket_start: int
    canonical_bucket_end: int
    canonical_rank_flat_segments: Tuple[Tuple[Tuple[int, int], ...], ...]
    rank_split_ranges: Tuple[Tuple[int, int], ...]

    @property
    def local_shape(self) -> Tuple[int, int]:
        return self.shard_layout.local_shape

    @property
    def global_shape(self) -> Tuple[int, int]:
        return self.shard_layout.global_shape

    @property
    def fs_split_dim(self) -> int:
        return int(self.shard_layout.fs_split_dim)

    @property
    def start_idx(self) -> int:
        return int(self.shard_layout.start_idx)

    @property
    def end_idx(self) -> int:
        return int(self.shard_layout.end_idx)

    @property
    def local_numel(self) -> int:
        return int(self.shard_layout.local_numel)


@dataclass(frozen=True)
class DionBucketLayout:
    """Bucket-local Dion transport layout under standard DO ownership."""

    entries: Tuple[DionBucketEntry, ...]
    shard_size: int
    gathered_numel: int
    param_ids: FrozenSet[int]

    @property
    def has_params(self) -> bool:
        return bool(self.entries)

    @property
    def entry_count(self) -> int:
        return len(self.entries)

    @property
    def max_shard_capacity(self) -> int:
        return max((int(entry.shard_capacity) for entry in self.entries), default=0)


def build_dion_entries_(
    *,
    bucket,
    param_map,
    dion_static_info_by_param,
    fs_size: int,
    fs_rank: int,
) -> tuple[DionBucketLayout | None, dict, int]:
    """Build bucket-local Dion shard layouts without mutating parent `param_map`."""
    entries: List[DionBucketEntry] = []
    dion_shard_layout_by_param = {}
    shard_offset = 0

    for param in param_map.keys():
        if not getattr(param, "is_dion_param", False):
            continue

        static_info = dion_static_info_by_param.get(param)
        if static_info is None:
            continue

        global_shape = tuple(static_info["global_shape"])
        fs_split_dim = int(static_info["fs_split_dim"])
        m, n = tuple(int(dim) for dim in param.shape)

        split_size = m if fs_split_dim == 0 else n
        start_idx, end_idx = compute_fs_shard_range(split_size, fs_size, fs_rank)
        local_split_size = int(end_idx) - int(start_idx)
        size_per_rank = math.ceil(split_size / fs_size)

        if fs_split_dim == 0:
            local_shape = (local_split_size, n)
            shard_capacity = size_per_rank * n
        else:
            local_shape = (m, local_split_size)
            shard_capacity = m * size_per_rank

        if not hasattr(bucket, "param_to_index") or param not in bucket.param_to_index:
            raise RuntimeError(
                "[Dion] missing canonical bucket span for Dion param "
                f"param={getattr(param, '_param_name', f'id_{id(param)}')}"
            )

        canonical_bucket_start, canonical_bucket_end = bucket.param_to_index[param]
        canonical_bucket_start = int(canonical_bucket_start)
        canonical_bucket_end = int(canonical_bucket_end)

        canonical_rank_flat_segments = []
        rank_split_ranges = []
        for rank_i in range(fs_size):
            rank_start, rank_end = compute_fs_shard_range(split_size, fs_size, rank_i)
            rank_split_ranges.append((int(rank_start), int(rank_end)))
            canonical_rank_flat_segments.append(
                compute_fs_flat_segments(
                    full_start=canonical_bucket_start,
                    m=m,
                    n=n,
                    fs_split_dim=fs_split_dim,
                    start_idx=rank_start,
                    end_idx=rank_end,
                )
            )

        shard_layout = DionShardLayout(
            local_shape=tuple(int(dim) for dim in local_shape),
            global_shape=tuple(int(dim) for dim in global_shape),
            fs_split_dim=fs_split_dim,
            start_idx=int(start_idx),
            end_idx=int(end_idx),
            per_expert_global_shape=static_info.get("per_expert_global_shape"),
        )
        dion_shard_layout_by_param[param] = shard_layout
        entries.append(
            DionBucketEntry(
                param=param,
                shard_layout=shard_layout,
                size_per_rank=int(size_per_rank),
                shard_capacity=int(shard_capacity),
                shard_offset=int(shard_offset),
                canonical_bucket_start=canonical_bucket_start,
                canonical_bucket_end=canonical_bucket_end,
                canonical_rank_flat_segments=tuple(
                    tuple((int(seg_start), int(seg_end)) for seg_start, seg_end in rank_segments)
                    for rank_segments in canonical_rank_flat_segments
                ),
                rank_split_ranges=tuple(rank_split_ranges),
            )
        )
        shard_offset += int(shard_capacity)

    bucket_layout = None
    if entries:
        bucket_layout = DionBucketLayout(
            entries=tuple(entries),
            shard_size=int(shard_offset),
            gathered_numel=int(shard_offset) * int(fs_size),
            param_ids=frozenset(id(entry.param) for entry in entries),
        )

    return bucket_layout, dion_shard_layout_by_param, len(entries)


def select_non_dion_bucket_params_(*, bucket_params, dion_layout: DionBucketLayout | None):
    """Return bucket params that are not carried by Dion transport layout entries."""
    dion_param_ids = frozenset() if dion_layout is None else dion_layout.param_ids
    return [param for param in bucket_params if id(param) not in dion_param_ids]


def bucket_rank_range_(full_start: int, full_end: int, shard_size: int, rank: int) -> tuple[int, int]:
    """Return the intersection of one full bucket span with one standard local shard."""
    shard_abs_start = rank * shard_size
    shard_abs_end = shard_abs_start + shard_size
    local_abs_start = max(int(full_start), shard_abs_start)
    local_abs_end = min(int(full_end), shard_abs_end)
    if local_abs_end <= local_abs_start:
        return 0, 0
    return local_abs_start - shard_abs_start, local_abs_end - shard_abs_start


def param_rank_range_(full_start: int, full_end: int, shard_size: int, rank: int) -> tuple[int, int]:
    """Return the intersection of one full bucket span in param-local coordinates."""
    shard_abs_start = rank * shard_size
    shard_abs_end = shard_abs_start + shard_size
    local_abs_start = max(int(full_start), shard_abs_start)
    local_abs_end = min(int(full_end), shard_abs_end)
    if local_abs_end <= local_abs_start:
        return 0, 0
    return local_abs_start - int(full_start), local_abs_end - int(full_start)
