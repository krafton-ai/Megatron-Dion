"""FS all-gather pack/unpack helpers for Dion distributed optimizer."""

from __future__ import annotations

import os
import logging
from typing import Optional

import torch
from .checkpoint_io import all_gather_fs_shards_2d_

from .fs_layout import (
    compute_fs_shard_range,
    slice_fs_shard_2d,
    write_fs_shard_2d_,
)

logger = logging.getLogger(__name__)


def _fs_view_aliases_param_(*, param: torch.Tensor, shard: torch.Tensor, fs_split_dim: int, start_idx: int, end_idx: int) -> bool:
    """Whether `shard` already aliases the expected local FS slice of `param.data`."""
    expected_view = slice_fs_shard_2d(param.data, fs_split_dim, start_idx, end_idx)
    return shard.shape == expected_view.shape and shard.data_ptr() == expected_view.data_ptr()


def pack_fs_shards_(
    *,
    optimizer,
    buffer,
    pack_buffer: torch.Tensor,
    dion_param_layout,
) -> None:
    """Pack local FS shards into `pack_buffer` according to `dion_param_layout`.

    This mirrors the existing logic in `DistributedOptimizerForDion._all_gather_params_bucket`
    and intentionally keeps behavior identical.
    """
    for entry in dion_param_layout:
        param = entry["param"]
        bucket = buffer.param_to_bucket.get(param) if hasattr(buffer, "param_to_bucket") else None
        full_view = (
            optimizer._bucket_param_view(bucket, param)
            if hasattr(optimizer, "_bucket_param_view")
            else None
        )
        local_shape = entry["local_shape"]
        pack_offset = entry["pack_offset"]
        segment_size = entry["segment_size"]
        fs_split_dim = entry["fs_split_dim"]
        start_idx = entry["start_idx"]
        end_idx = entry["end_idx"]

        # Get param name for logging.
        if hasattr(optimizer, "param_to_name"):
            param_name = optimizer.param_to_name.get(param, f"id_{id(param)}")
        else:
            param_name = f"id_{id(param)}"

        expected_numel = local_shape[0] * local_shape[1]
        if segment_size < expected_numel:
            raise RuntimeError(
                f"[Dion] segment_size too small: seg={segment_size} expected>={expected_numel} "
                f"shape={local_shape} param={param_name}"
            )

        expected_shape = tuple(local_shape)
        source_param = full_view if full_view is not None else param.data
        expected_view = slice_fs_shard_2d(source_param, fs_split_dim, start_idx, end_idx)
        existing_shard = optimizer._get_data_shard(param)
        fs_shard = getattr(param, "_fs_shard", None)

        if existing_shard is not None and _fs_view_aliases_param_(
            param=param,
            shard=existing_shard,
            fs_split_dim=fs_split_dim,
            start_idx=start_idx,
            end_idx=end_idx,
        ):
            local_shard_2d = existing_shard
        elif fs_shard is not None and _fs_view_aliases_param_(
            param=param,
            shard=fs_shard,
            fs_split_dim=fs_split_dim,
            start_idx=start_idx,
            end_idx=end_idx,
        ):
            local_shard_2d = fs_shard
            optimizer._update_data_shard(param, local_shard_2d)
        else:
            local_shard_2d = expected_view
            optimizer._update_data_shard(param, local_shard_2d)
            param._fs_shard = local_shard_2d

        shard_numel = local_shape[0] * local_shape[1]
        # View-based FS shards can be non-contiguous (e.g., column shards). Pack through
        # reshape so the communication path stays correct while preserving alias-based storage.
        local_shard = local_shard_2d.reshape(-1)
        if local_shard.device != pack_buffer.device:
            local_shard = local_shard.to(pack_buffer.device, non_blocking=True)

        pack_buffer[pack_offset : pack_offset + shard_numel].copy_(local_shard[:shard_numel])

        # Zero padding if needed.
        if shard_numel < segment_size:
            pack_buffer[pack_offset + shard_numel : pack_offset + segment_size].zero_()


def unpack_fs_shards_(
    *,
    optimizer,
    gathered_buffer: torch.Tensor,
    dion_param_layout,
    pack_total: int,
    fs_size: int,
    fs_rank: int,
    buffer: Optional[object] = None,
) -> None:
    """Unpack full params from `gathered_buffer` into `param.data` in-place.

    The `buffer` argument is accepted for API compatibility but is not used.
    """
    for entry in dion_param_layout:
        param = entry["param"]
        bucket = buffer.param_to_bucket.get(param) if hasattr(buffer, "param_to_bucket") else None
        full_view = (
            optimizer._bucket_param_view(bucket, param)
            if hasattr(optimizer, "_bucket_param_view")
            else None
        )
        pack_offset = entry["pack_offset"]
        fs_split_dim = entry["fs_split_dim"]
        global_shape = entry["global_shape"]
        pre_unpack_local = None

        # FS restores one dimension; other stays TP-local.
        if fs_split_dim == 0:
            m_full = global_shape[0]
            n_full = param.shape[1]
        else:
            m_full = param.shape[0]
            n_full = global_shape[1]

        full_param_base = full_view if full_view is not None else param.data
        full_param_2d = full_param_base.view(m_full, n_full)
        split_dim_size = m_full if fs_split_dim == 0 else n_full
        local_rank_segment_2d = None
        checked_dims = getattr(optimizer, "_debug_fs_pack_checked_dims", set())
        debug_compare_once = (
            os.getenv("DION_DEBUG_COMPARE_FS_PACK", "0") == "1"
            and getattr(optimizer, "_debug_fs_pack_compare_armed", False)
            and fs_split_dim not in checked_dims
        )
        if debug_compare_once:
            pre_unpack_local = (
                slice_fs_shard_2d(full_param_2d, fs_split_dim, my_start_idx := compute_fs_shard_range(split_dim_size, fs_size, fs_rank)[0], my_end_idx := compute_fs_shard_range(split_dim_size, fs_size, fs_rank)[1])
                .contiguous()
            )

        for rank_i in range(fs_size):
            rank_pack_offset = rank_i * pack_total + pack_offset
            segment_size = entry["segment_size"]
            rank_segment = gathered_buffer[rank_pack_offset : rank_pack_offset + segment_size]

            start_idx, end_idx = compute_fs_shard_range(split_dim_size, fs_size, rank_i)
            actual_size = end_idx - start_idx

            if fs_split_dim == 0:
                rank_segment_2d = rank_segment[: actual_size * n_full].view(actual_size, n_full)
            else:
                rank_segment_2d = rank_segment[: actual_size * m_full].view(m_full, actual_size)

            write_fs_shard_2d_(full_param_2d, fs_split_dim, start_idx, end_idx, rank_segment_2d)
            if rank_i == fs_rank:
                local_rank_segment_2d = rank_segment_2d

        # Refresh local shard metadata only if it does not already alias param.data.
        full_m, full_n = full_param_2d.shape
        split_dim_size = full_m if fs_split_dim == 0 else full_n
        my_start_idx, my_end_idx = compute_fs_shard_range(split_dim_size, fs_size, fs_rank)
        expected_view = slice_fs_shard_2d(full_param_2d, fs_split_dim, my_start_idx, my_end_idx)
        target_shard = optimizer._get_data_shard(param)
        if (
            target_shard is None
            or not _fs_view_aliases_param_(
                param=param,
                shard=target_shard,
                fs_split_dim=fs_split_dim,
                start_idx=my_start_idx,
                end_idx=my_end_idx,
            )
        ):
            target_shard = expected_view
            optimizer._update_data_shard(param, target_shard)
            param._fs_shard = target_shard

        if debug_compare_once and pre_unpack_local is not None:
            expected_full = torch.zeros_like(full_param_2d)
            write_fs_shard_2d_(
                expected_full, fs_split_dim, my_start_idx, my_end_idx, pre_unpack_local
            )
            all_gather_fs_shards_2d_(
                expected_full,
                fs_split_dim=fs_split_dim,
                start_idx=my_start_idx,
                end_idx=my_end_idx,
                fs_group=optimizer.fs_group,
                fs_size=fs_size,
            )
            max_diff = (expected_full - full_param_2d).abs().max().item()
            param_name = (
                optimizer.param_to_name.get(param, f"id_{id(param)}")
                if hasattr(optimizer, "param_to_name")
                else f"id_{id(param)}"
            )
            logger.error(
                "[DION_FS_PACK_COMPARE] param=%s fs_split_dim=%s range=[%s:%s] max_diff=%s",
                param_name,
                fs_split_dim,
                my_start_idx,
                my_end_idx,
                max_diff,
            )
            checked_dims = set(getattr(optimizer, "_debug_fs_pack_checked_dims", set()))
            checked_dims.add(fs_split_dim)
            optimizer._debug_fs_pack_checked_dims = checked_dims
            if 0 in checked_dims and 1 in checked_dims:
                optimizer._debug_fs_pack_compare_armed = False
