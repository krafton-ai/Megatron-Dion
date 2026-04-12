"""Bucket grad-sync helpers for Dion distributed optimizer integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.distributed as dist


@dataclass
class DionBucketGradSync:
    """Bucket-local Dion grad sync payload kept through reduce-scatter completion."""

    local_grad_shard: torch.Tensor
    bucket_grad_shards: torch.Tensor | None = None
    group_ranks: tuple[int, ...] | None = None
    group_rank: int | None = None


def _copy_flat_segments(
    *,
    dst_buffer: torch.Tensor,
    full_bucket_flat: torch.Tensor,
    flat_segments,
) -> None:
    """Pack one logical local shard from canonical bucket-flat segments."""
    dst_buffer.zero_()
    source_offset = 0
    for seg_start, seg_end in flat_segments:
        seg_len = int(seg_end) - int(seg_start)
        if seg_len <= 0:
            continue
        dst_buffer[source_offset : source_offset + seg_len].copy_(
            full_bucket_flat[int(seg_start) : int(seg_end)]
        )
        source_offset += seg_len


def _build_bucket_grad_shards(*, bucket, group_size: int) -> torch.Tensor:
    """Pack bucket.grad_data into per-rank Dion local-shard RS payloads."""
    dion_layout = getattr(bucket, "dion_layout", None)
    if dion_layout is None or not dion_layout.has_params:
        raise RuntimeError(
            f"[Dion] missing Dion layout for bucket grad sync bucket={getattr(bucket, 'bucket_id', -1)}"
        )
    if group_size <= 0:
        raise RuntimeError(
            f"[Dion] invalid bucket grad sync group size for bucket={getattr(bucket, 'bucket_id', -1)}: {group_size}"
        )

    shard_size = int(dion_layout.shard_size)
    expected_total = int(dion_layout.gathered_numel)
    if shard_size <= 0 or expected_total != shard_size * group_size:
        raise RuntimeError(
            "[Dion] invalid packed bucket grad sync layout "
            f"bucket={getattr(bucket, 'bucket_id', -1)} shard_size={shard_size} "
            f"expected_total={expected_total} group_size={group_size}"
        )

    full_bucket_flat = bucket.grad_data.view(-1)
    packed_input = torch.zeros(
        expected_total,
        dtype=full_bucket_flat.dtype,
        device=full_bucket_flat.device,
    )
    packed_view = packed_input.view(group_size, shard_size)

    for entry in dion_layout.entries:
        shard_start = int(entry.shard_offset)
        shard_end = shard_start + int(entry.shard_capacity)
        for rank_i, rank_segments in enumerate(entry.canonical_rank_flat_segments):
            _copy_flat_segments(
                dst_buffer=packed_view[rank_i, shard_start:shard_end],
                full_bucket_flat=full_bucket_flat,
                flat_segments=rank_segments,
            )

    return packed_input


def _launch_dion_bucket_grad_sync(
    *,
    bucket,
    communication_group,
    reduce_op,
    async_op: bool,
    reduce_scatter_fn,
    stash_grad_sync_fn: Callable[[object, DionBucketGradSync], None],
):
    """Launch bucket-wise packed RS so the Dion output already matches local shard layout."""
    dion_layout = getattr(bucket, "dion_layout", None)
    if dion_layout is None or not dion_layout.has_params:
        return None

    group_size = 1 if communication_group is None else int(dist.get_world_size(communication_group))
    local_grad_shard = torch.empty(
        int(dion_layout.shard_size),
        dtype=bucket.grad_data.dtype,
        device=bucket.grad_data.device,
    )
    packed_input = _build_bucket_grad_shards(bucket=bucket, group_size=group_size)

    if group_size == 1:
        local_grad_shard.copy_(packed_input.view(1, -1)[0])
        stash_grad_sync_fn(
            bucket,
            DionBucketGradSync(local_grad_shard=local_grad_shard),
        )
        return None

    grad_sync = DionBucketGradSync(
        local_grad_shard=local_grad_shard,
        bucket_grad_shards=packed_input,
        group_ranks=tuple(dist.get_process_group_ranks(communication_group)),
        group_rank=int(dist.get_rank(communication_group)),
    )
    stash_grad_sync_fn(bucket, grad_sync)
    return reduce_scatter_fn(
        local_grad_shard,
        packed_input,
        op=reduce_op,
        group=communication_group,
        async_op=async_op,
    )


def _set_bucket_local_grads(
    *,
    bucket,
    publish_local_grad_fn: Callable[[torch.nn.Parameter, torch.Tensor], None],
    grad_sync: DionBucketGradSync | None,
) -> None:
    """Publish Dion local-grad views from packed RS output without full-bucket reconstruction."""
    dion_layout = getattr(bucket, "dion_layout", None)
    if dion_layout is None or not dion_layout.has_params:
        return

    if grad_sync is None:
        raise RuntimeError(
            "[Dion] missing packed Dion bucket grad sync "
            f"for bucket={getattr(bucket, 'bucket_id', -1)}"
        )
    local_grad_shard = grad_sync.local_grad_shard
    expected_shard_size = int(dion_layout.shard_size)
    if local_grad_shard.ndim != 1 or local_grad_shard.numel() != expected_shard_size:
        raise RuntimeError(
            "[Dion] invalid packed Dion bucket grad sync output "
            f"bucket={getattr(bucket, 'bucket_id', -1)} "
            f"shape={tuple(local_grad_shard.shape)} expected_numel={expected_shard_size}"
        )

    for entry in dion_layout.entries:
        local_numel = int(entry.local_numel)
        shard_capacity = int(entry.shard_capacity)
        if local_numel <= 0 or local_numel > shard_capacity:
            raise RuntimeError(
                "[Dion] invalid packed Dion grad shard metadata "
                f"bucket={getattr(bucket, 'bucket_id', -1)} "
                f"param={getattr(entry.param, '_param_name', id(entry.param))} "
                f"local_numel={local_numel} shard_capacity={shard_capacity}"
            )
        shard_start = int(entry.shard_offset)
        local_shard = local_grad_shard[shard_start : shard_start + local_numel].view(
            entry.local_shape
        )
        publish_local_grad_fn(entry.param, local_shard)
