"""Param-gather and canonical local-shard rebinding helpers for Dion DO."""

from __future__ import annotations

from typing import Callable, Iterable

import torch

from .bucket_layout import DionBucketLayout
from .fs_layout import slice_fs_shard_2d


def collect_dion_bucket_params_(dion_layout: DionBucketLayout | None) -> list[torch.nn.Parameter]:
    """Return unique bucket-local Dion params in canonical entry order."""
    if dion_layout is None or not dion_layout.has_params:
        return []
    params = []
    seen_param_ids = set()
    for entry in dion_layout.entries:
        if entry.param is None or id(entry.param) in seen_param_ids:
            continue
        seen_param_ids.add(id(entry.param))
        params.append(entry.param)
    return params


def serialize_dion_bucket_gather_layout_(dion_layout: DionBucketLayout) -> tuple[int, ...]:
    """Serialize bucket-global Dion gather invariants for cross-rank validation."""
    payload: list[int] = [
        int(dion_layout.entry_count),
        int(dion_layout.shard_size),
        int(dion_layout.gathered_numel),
        int(dion_layout.max_shard_capacity),
    ]
    for entry in dion_layout.entries:
        payload.extend(
            [
                int(entry.shard_offset),
                int(entry.shard_capacity),
                int(entry.canonical_bucket_start),
                int(entry.canonical_bucket_end),
                int(entry.fs_shard_dim),
                int(entry.size_per_rank),
            ]
        )
        payload.append(len(entry.canonical_rank_flat_segments))
        for rank_segments in entry.canonical_rank_flat_segments:
            payload.append(len(rank_segments))
            for seg_start, seg_end in rank_segments:
                payload.extend([int(seg_start), int(seg_end)])
    return tuple(payload)


def bind_dion_local_shard_(
    *,
    entry,
    full_view_2d: torch.Tensor,
    update_data_shard_fn: Callable[[torch.nn.Parameter, torch.Tensor], None],
    param_name_fn: Callable[[torch.nn.Parameter], str],
) -> None:
    """Rebind one Dion param's canonical local shard view from a full bucket view."""
    local_target = slice_fs_shard_2d(
        full_view_2d,
        int(entry.fs_shard_dim),
        int(entry.start_idx),
        int(entry.end_idx),
    )
    local_numel = int(entry.local_numel)
    if local_numel != local_target.numel():
        raise RuntimeError(
            "[Dion] local restore shard size mismatch "
            f"param={param_name_fn(entry.param)} "
            f"source={local_numel} target={int(local_target.numel())}"
        )
    update_data_shard_fn(entry.param, local_target)
    entry.param._fs_shard = local_target


def rebind_dion_local_shards_from_bucket_(
    *,
    dion_layout: DionBucketLayout | None,
    get_full_view_2d_fn: Callable[[object], torch.Tensor],
    update_data_shard_fn: Callable[[torch.nn.Parameter, torch.Tensor], None],
    param_name_fn: Callable[[torch.nn.Parameter], str],
) -> None:
    """Rebind all Dion local shard aliases from canonical bucket.param_data views."""
    if dion_layout is None or not dion_layout.has_params:
        return
    for entry in dion_layout.entries:
        bind_dion_local_shard_(
            entry=entry,
            full_view_2d=get_full_view_2d_fn(entry),
            update_data_shard_fn=update_data_shard_fn,
            param_name_fn=param_name_fn,
        )


def _restore_dion_bucket(
    *,
    bucket,
    prepared_entries: Iterable[object],
    gathered_buffer: torch.Tensor,
    shard_group_size: int,
    update_data_shard_fn: Callable[[torch.nn.Parameter, torch.Tensor], None],
    param_name_fn: Callable[[torch.nn.Parameter], str],
) -> None:
    """Restore canonical bucket storage from one packed gathered Dion shard buffer."""
    if gathered_buffer.dim() != 2:
        raise RuntimeError(
            f"[Dion] gathered Dion bucket buffer must be 2D, got shape={tuple(gathered_buffer.shape)}"
        )
    if gathered_buffer.size(0) != shard_group_size:
        raise RuntimeError(
            "[Dion] gathered Dion bucket buffer group-size mismatch "
            f"bucket={getattr(bucket, 'bucket_id', -1)} "
            f"buffer_group={int(gathered_buffer.size(0))} expected_group={int(shard_group_size)}"
        )

    for entry, full_view_2d in prepared_entries:
        full_flat = full_view_2d.view(-1)
        shard_start = int(entry.shard_offset)
        shard_end = shard_start + int(entry.shard_capacity)

        for rank_i, rank_segments in enumerate(entry.canonical_rank_flat_segments):
            rank_source = gathered_buffer[rank_i, shard_start:shard_end]
            source_offset = 0
            for seg_start, seg_end in rank_segments:
                seg_len = int(seg_end) - int(seg_start)
                if seg_len <= 0:
                    continue
                local_start = int(seg_start) - int(entry.canonical_bucket_start)
                local_end = int(seg_end) - int(entry.canonical_bucket_start)
                full_flat[local_start:local_end].copy_(
                    rank_source[source_offset : source_offset + seg_len]
                )
                source_offset += seg_len

        bind_dion_local_shard_(
            entry=entry,
            full_view_2d=full_view_2d,
            update_data_shard_fn=update_data_shard_fn,
            param_name_fn=param_name_fn,
        )
