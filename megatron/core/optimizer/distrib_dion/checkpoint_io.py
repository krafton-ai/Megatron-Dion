"""Checkpoint and export helpers for Dion distributed optimizer.

These helpers are intended to be behavior-preserving refactors: no changes to
math or collective patterns, only code organization.
"""

from __future__ import annotations

import logging
from typing import Callable

import torch
import torch.distributed as dist

from ..distrib_optimizer import Range
from .fs_layout import compute_fs_shard_range, slice_fs_shard_2d, write_fs_shard_2d_
from .param_naming import get_param_name

logger = logging.getLogger(__name__)


def build_named_dion_param_state(param_groups, optimizer_state, get_param_name_fn) -> dict:
    """Build persistent Dion optimizer state keyed by deterministic param name.

    This is intentionally narrower than the raw optimizer state:
    scratch buffers and transient sync flags are excluded.
    """
    name_to_state = {}
    for group in param_groups:
        for param in group["params"]:
            state = optimizer_state.get(param)
            if not state:
                continue
            name = get_param_name_fn(param)
            if name is None:
                continue
            persistent_state = {}
            for key, value in state.items():
                if key.startswith("_"):
                    continue
                persistent_state[key] = value
            if persistent_state:
                name_to_state[name] = persistent_state
    return name_to_state


def restore_named_dion_param_state_(
    *,
    param_groups,
    optimizer_state,
    get_param_name_fn,
    name_to_state: dict,
) -> dict[str, int]:
    """Restore persistent Dion optimizer state keyed by deterministic param name.

    Returns:
        Summary counts with distinct buckets:
        - `restored`
        - `unnamed`
        - `no_payload_entry`
        The helper intentionally does not guess whether a missing payload entry is
        a correctness problem; save-time omitted/default state and load-time
        initialized default state are indistinguishable here.
    """
    summary = {
        "restored": 0,
        "unnamed": 0,
        "no_payload_entry": 0,
    }

    for group in param_groups:
        for param in group["params"]:
            param_name = get_param_name_fn(param)
            current_state = optimizer_state.get(param, {})
            if param_name is None:
                summary["unnamed"] += 1
                continue
            if param_name not in name_to_state:
                summary["no_payload_entry"] += 1
                continue

            saved_state = name_to_state[param_name]
            new_state = {k: v for k, v in current_state.items() if k.startswith("_")}
            new_state["_needs_state_replica_q_sync"] = False
            if "_q_full_buffer" in new_state:
                new_state["_q_full_buffer"] = None
            if "_q_gather_buffer" in new_state:
                new_state["_q_gather_buffer"] = None

            for key, value in saved_state.items():
                if isinstance(value, torch.Tensor):
                    current_value = current_state.get(key)
                    if isinstance(current_value, torch.Tensor) and current_value.shape != value.shape:
                        raise RuntimeError(
                            f"[Dion] checkpoint tensor shape mismatch for {param_name}.{key}: "
                            f"saved={tuple(value.shape)} current={tuple(current_value.shape)}"
                        )
                    new_state[key] = value.to(device=param.device)
                else:
                    new_state[key] = value

            optimizer_state[param] = new_state
            summary["restored"] += 1

    return summary


def all_gather_flat_shards_(
    param_flat: torch.Tensor,
    *,
    flat_start: int,
    flat_end: int,
    group: dist.ProcessGroup,
    world_size: int,
) -> None:
    """All-gather uneven 1D shards back into `param_flat` in-place.

    This matches the existing Dion code path:
    - pad each rank's shard to max_shard_size
    - `dist.all_gather` padded shards
    - `dist.all_gather` each rank's true (start,end) range
    - unpack back into the correct slices
    """
    local_shard_size = flat_end - flat_start
    if local_shard_size <= 0:
        return

    local_shard = param_flat[flat_start:flat_end].contiguous()

    total_numel = param_flat.numel()
    max_shard_size = (total_numel + world_size - 1) // world_size

    padded_shard = torch.zeros(
        max_shard_size, dtype=param_flat.dtype, device=param_flat.device
    )
    padded_shard[:local_shard_size].copy_(local_shard)

    gathered_shards = [torch.empty_like(padded_shard) for _ in range(world_size)]
    dist.all_gather(gathered_shards, padded_shard, group=group)

    local_range_info = torch.tensor(
        [flat_start, flat_end], device=param_flat.device, dtype=torch.long
    )
    all_range_infos = [torch.empty_like(local_range_info) for _ in range(world_size)]
    dist.all_gather(all_range_infos, local_range_info, group=group)

    for rank_i in range(world_size):
        r_start, r_end = all_range_infos[rank_i].tolist()
        r_size = r_end - r_start
        if r_size > 0:
            param_flat[r_start:r_end].copy_(gathered_shards[rank_i][:r_size])


def all_gather_fs_shards_2d_(
    param_2d: torch.Tensor,
    *,
    fs_split_dim: int,
    start_idx: int,
    end_idx: int,
    fs_group: dist.ProcessGroup,
    fs_size: int,
) -> None:
    """All-gather FS-sharded 2D shards back into `param_2d` in-place.

    This matches the existing Dion code path: pad to max shard size along the FS
    axis, all_gather, then unpack each rank's actual shard slice.
    """
    local_shard_size = end_idx - start_idx
    if local_shard_size <= 0:
        return

    local_shard = slice_fs_shard_2d(param_2d, fs_split_dim, start_idx, end_idx).contiguous()
    other_dim_size = param_2d.shape[1] if fs_split_dim == 0 else param_2d.shape[0]

    split_dim_size = param_2d.shape[fs_split_dim]
    max_shard_size = (split_dim_size + fs_size - 1) // fs_size

    if fs_split_dim == 0:
        padded_shard = torch.zeros(
            max_shard_size, other_dim_size, dtype=param_2d.dtype, device=param_2d.device
        )
        padded_shard[:local_shard_size, :].copy_(local_shard)
    else:
        padded_shard = torch.zeros(
            other_dim_size, max_shard_size, dtype=param_2d.dtype, device=param_2d.device
        )
        padded_shard[:, :local_shard_size].copy_(local_shard)

    gathered_shards = [torch.empty_like(padded_shard) for _ in range(fs_size)]
    dist.all_gather(gathered_shards, padded_shard, group=fs_group)

    for rank_i in range(fs_size):
        r_start, r_end = compute_fs_shard_range(split_dim_size, fs_size, rank_i)
        actual_size = r_end - r_start
        if fs_split_dim == 0:
            rank_shard_2d = gathered_shards[rank_i][:actual_size, :]
        else:
            rank_shard_2d = gathered_shards[rank_i][:, :actual_size]
        write_fs_shard_2d_(param_2d, fs_split_dim, r_start, r_end, rank_shard_2d)


def copy_param_to_shard_main_(
    *,
    source_param: torch.Tensor,
    shard_main_param: torch.nn.Parameter,
    dion_info,
    param_range,
) -> None:
    """Copy a source model/state-dict param into the local optimizer shard."""
    from ...fp8_utils import dequantize_fp8_tensor, is_float8tensor

    if is_float8tensor(source_param):
        source_param_fp32 = dequantize_fp8_tensor(source_param)
    else:
        source_param_fp32 = source_param

    if dion_info.get("is_dion", False):
        start_idx = dion_info["start_idx"]
        end_idx = dion_info["end_idx"]
        fs_split_dim = dion_info["fs_split_dim"]

        if source_param_fp32.ndim == 2:
            source_2d = source_param_fp32
        else:
            source_2d = source_param_fp32.view(shard_main_param.shape)

        shard_model_param = (
            slice_fs_shard_2d(source_2d, fs_split_dim, start_idx, end_idx).clone().view(-1)
        )
        assert shard_model_param.numel() == shard_main_param.numel(), (
            f"FS shard size mismatch: shard_model_param={shard_model_param.numel()}, "
            f"shard_main_param={shard_main_param.numel()}, "
            f"fs_split_dim={fs_split_dim}, start_idx={start_idx}, end_idx={end_idx}, "
            f"source_shape={source_param_fp32.shape}, "
            f"global_shape={dion_info['global_shape']}"
        )
        shard_main_param.data.copy_(shard_model_param.reshape(shard_main_param.shape))
        return

    shard_model_param = source_param_fp32.view(-1)[param_range.start : param_range.end]
    assert param_range.size == shard_main_param.nelement(), (
        f"Non-Dion param size mismatch: param_range.size={param_range.size}, "
        f"shard_main={shard_main_param.nelement()}, "
        f"source_shape={source_param.shape}, "
        f"param_range=[{param_range.start}:{param_range.end}], "
        f"is_2D={source_param.dim() >= 2}"
    )
    if shard_model_param.shape != shard_main_param.shape:
        shard_model_param = shard_model_param.view_as(shard_main_param)
    shard_main_param.data.copy_(shard_model_param)


def copy_group_params_to_main_shards_(
    *,
    model_groups,
    shard_main_groups,
    get_source_param_fn: Callable[[torch.nn.Parameter], torch.Tensor],
    get_param_range_fn: Callable[[torch.nn.Parameter], object],
    get_dion_info_fn: Callable[[torch.nn.Parameter], dict],
) -> None:
    """Copy one or more model param groups into their local optimizer shards."""
    for model_group, shard_main_group in zip(model_groups, shard_main_groups):
        for model_param, shard_main_param in zip(model_group, shard_main_group):
            if shard_main_param is None:
                continue

            copy_param_to_shard_main_(
                source_param=get_source_param_fn(model_param),
                shard_main_param=shard_main_param,
                dion_info=get_dion_info_fn(model_param),
                param_range=get_param_range_fn(model_param),
            )


def apply_optimizer_shard_to_model_param_(
    *,
    model_param: torch.nn.Parameter,
    opt_shard: torch.Tensor,
    data_shard: torch.Tensor,
    param_range_map,
    dion_info,
    get_bucket_param_data_fn: Callable[[torch.nn.Parameter], torch.Tensor] | None,
    zero_range_warned: int,
) -> int:
    """Copy an updated optimizer shard into `data_shard` and `model_param.data`."""
    data_shard.data.copy_(opt_shard.to(data_shard.dtype))
    model_param._fs_shard = data_shard

    if dion_info.get("is_dion", False):
        fs_split_dim = dion_info["fs_split_dim"]
        start_idx = dion_info["start_idx"]
        end_idx = dion_info["end_idx"]

        if start_idx == 0 and end_idx == 0 and zero_range_warned < 5:
            param_name = getattr(model_param, "_param_name", f"shape={model_param.shape}")
            logger.error(
                "[ZERO RANGE] start_idx=0, end_idx=0 for param %s! No data will be copied to model_param.data! dion_info=%s",
                param_name,
                dion_info,
            )
            zero_range_warned += 1

        if fs_split_dim == 0:
            expected_rows = end_idx - start_idx
            expected_cols = model_param.data.shape[1]
        else:
            expected_rows = model_param.data.shape[0]
            expected_cols = end_idx - start_idx

        if data_shard.numel() != expected_rows * expected_cols:
            param_name = getattr(model_param, "_param_name", f"id_{id(model_param)}")
            raise RuntimeError(
                f"[Dion] FS shard shape mismatch: param={param_name}, "
                f"data_shard.numel()={data_shard.numel()}, "
                f"expected={expected_rows}x{expected_cols}={expected_rows * expected_cols}, "
                f"fs_split_dim={fs_split_dim}, range=[{start_idx}:{end_idx}]"
            )

        target_slice = slice_fs_shard_2d(model_param.data, fs_split_dim, start_idx, end_idx)
        src_view = data_shard.view(expected_rows, expected_cols)
        if src_view.data_ptr() == target_slice.data_ptr():
            return zero_range_warned
        param_name = getattr(model_param, "_param_name", f"id_{id(model_param)}")
        logger.error(
            "[DION_FS_ALIAS_MISMATCH] param=%s data_shard_ptr=%s target_ptr=%s fs_split_dim=%s range=[%s:%s]",
            param_name,
            src_view.data_ptr(),
            target_slice.data_ptr(),
            fs_split_dim,
            start_idx,
            end_idx,
        )
        raise RuntimeError(
            f"[Dion] FS shard alias mismatch for {param_name}: "
            f"data_shard no longer aliases canonical model_param.data slice "
            f"(fs_split_dim={fs_split_dim}, range=[{start_idx}:{end_idx}])"
        )

    if param_range_map is None or "gbuf_world_in_bucket" not in param_range_map:
        param_name = get_param_name(model_param, fallback_style="shape")
        raise RuntimeError(
            "[Dion] non-Dion write-back requires canonical gbuf_world_in_bucket "
            f"for {param_name}"
        )

    stock_world_start = getattr(model_param, "_stock_world_bucket_start", None)
    stock_world_end = getattr(model_param, "_stock_world_bucket_end", None)
    if stock_world_start is not None and stock_world_end is not None:
        world_range = Range(int(stock_world_start), int(stock_world_end))
    else:
        world_range = param_range_map["gbuf_world_in_bucket"]
    if get_bucket_param_data_fn is None:
        param_name = get_param_name(model_param, fallback_style="shape")
        raise RuntimeError(
            "[Dion] non-Dion write-back requires canonical bucket.param_data "
            f"for {param_name}"
        )
    bucket_param_data = get_bucket_param_data_fn(model_param)
    if bucket_param_data is None:
        param_name = get_param_name(model_param, fallback_style="shape")
        raise RuntimeError(
            "[Dion] non-Dion write-back missing bucket.param_data "
            f"for {param_name}"
        )
    target_flat = bucket_param_data.view(-1)[world_range.start : world_range.end]
    if target_flat.numel() != opt_shard.numel():
        param_name = get_param_name(model_param, fallback_style="shape")
        stock_param_start = getattr(model_param, "_stock_param_start", None)
        stock_param_end = getattr(model_param, "_stock_param_end", None)
        stock_world_start = getattr(model_param, "_stock_world_bucket_start", None)
        stock_world_end = getattr(model_param, "_stock_world_bucket_end", None)
        stock_bucket = getattr(model_param, "_stock_bucket_index", None)
        stock_gbuf = getattr(model_param, "_stock_gbuf_index", None)
        raise RuntimeError(
            "[Dion] non-Dion write-back size mismatch "
            f"param={param_name} world_range_size={target_flat.numel()} "
            f"opt_shard_numel={opt_shard.numel()} "
            f"current_world_range=[{world_range.start}:{world_range.end}] "
            f"stock_param_range=[{stock_param_start}:{stock_param_end}] "
            f"stock_world_range=[{stock_world_start}:{stock_world_end}] "
            f"stock_gbuf={stock_gbuf} stock_bucket={stock_bucket}"
        )

    target_view = target_flat.view_as(opt_shard)
    if target_view.data_ptr() != data_shard.data_ptr():
        target_view.copy_(opt_shard.to(target_view.dtype))
    return zero_range_warned


def apply_stock_non_dion_shards_(
    *,
    model_groups,
    shard_groups,
    get_bucket_param_data_fn: Callable[[torch.nn.Parameter], torch.Tensor] | None,
) -> int:
    """Write back non-Dion optimizer shards using the standard DO local-shard contract."""
    param_count = 0
    for model_group, shard_group in zip(model_groups, shard_groups):
        for model_param, shard_param in zip(model_group, shard_group):
            if shard_param is None or getattr(model_param, "is_dion_param", False):
                continue
            if get_bucket_param_data_fn is None:
                param_name = get_param_name(model_param, fallback_style="shape")
                raise RuntimeError(
                    "[Dion] non-Dion write-back requires canonical bucket.param_data "
                    f"for {param_name}"
                )
            bucket_param_data = get_bucket_param_data_fn(model_param)
            if bucket_param_data is None:
                param_name = get_param_name(model_param, fallback_style="shape")
                raise RuntimeError(
                    "[Dion] non-Dion write-back missing bucket.param_data "
                    f"for {param_name}"
                )
            stock_world_start = getattr(model_param, "_stock_world_bucket_start", None)
            stock_world_end = getattr(model_param, "_stock_world_bucket_end", None)
            if stock_world_start is None or stock_world_end is None:
                param_name = get_param_name(model_param, fallback_style="shape")
                raise RuntimeError(
                    "[Dion] non-Dion write-back missing preserved stock world range "
                    f"for {param_name}"
                )
            world_range = Range(int(stock_world_start), int(stock_world_end))
            target_flat = bucket_param_data.view(-1)[world_range.start : world_range.end]
            if target_flat.numel() != shard_param.nelement():
                param_name = get_param_name(model_param, fallback_style="shape")
                raise RuntimeError(
                    "[Dion] non-Dion stock write-back size mismatch "
                    f"param={param_name} world_range_size={target_flat.numel()} "
                    f"opt_shard_numel={shard_param.nelement()}"
                )
            target_flat.view_as(shard_param).copy_(shard_param)
            param_count += 1
    return param_count


def apply_group_shards_to_model_params_(
    *,
    model_groups,
    shard_groups,
    shard16_groups,
    get_data_shard_fn: Callable[[torch.nn.Parameter], torch.Tensor],
    get_param_range_map_fn: Callable[[torch.nn.Parameter], dict],
    get_dion_info_fn: Callable[[torch.nn.Parameter], dict],
    get_bucket_param_data_fn: Callable[[torch.nn.Parameter], torch.Tensor] | None,
    zero_range_warned: int,
) -> tuple[int, int]:
    """Apply updated Dion optimizer shards to grouped model params."""
    param_count = 0

    for model_group, shard_group, shard16_group in zip(
        model_groups,
        shard_groups,
        shard16_groups,
    ):
        for model_param, shard_param, shard16_param in zip(
            model_group, shard_group, shard16_group
        ):
            if shard_param is None:
                continue
            if not get_dion_info_fn(model_param).get("is_dion", False):
                continue

            data_shard = get_data_shard_fn(model_param)
            if data_shard is None:
                data_shard = shard16_param

            zero_range_warned = apply_optimizer_shard_to_model_param_(
                model_param=model_param,
                opt_shard=shard_param,
                data_shard=data_shard,
                param_range_map=get_param_range_map_fn(model_param),
                dion_info=get_dion_info_fn(model_param),
                get_bucket_param_data_fn=get_bucket_param_data_fn,
                zero_range_warned=zero_range_warned,
            )
            param_count += 1

    return param_count, zero_range_warned


def restore_full_model_param_(
    *,
    model_param: torch.nn.Parameter,
    param_range,
    dion_info,
    fs_group: dist.ProcessGroup,
    fs_size: int,
) -> bool:
    """Restore `model_param.data` to its full view by all-gathering local shards."""
    if not dion_info.get("is_dion", False):
        if param_range is None:
            return False
        flat_start = param_range.start
        flat_end = param_range.end
        if flat_end <= flat_start:
            return False
        param_flat = model_param.data.view(-1)
        all_gather_flat_shards_(
            param_flat,
            flat_start=flat_start,
            flat_end=flat_end,
            group=fs_group,
            world_size=fs_size,
        )
        return True

    fs_split_dim = dion_info["fs_split_dim"]
    start_idx = dion_info["start_idx"]
    end_idx = dion_info["end_idx"]
    if start_idx == end_idx:
        return False

    all_gather_fs_shards_2d_(
        model_param.data,
        fs_split_dim=fs_split_dim,
        start_idx=start_idx,
        end_idx=end_idx,
        fs_group=fs_group,
        fs_size=fs_size,
    )
    return True


def restore_group_params_(
    *,
    model_groups,
    shard_groups,
    get_param_range_fn,
    get_dion_info_fn,
    fs_group: dist.ProcessGroup,
    fs_size: int,
) -> tuple[int, int]:
    """Restore full params for a group list and return `(dion_count, non_dion_count)`."""
    dion_count = 0
    non_dion_count = 0

    for model_group, shard_group in zip(model_groups, shard_groups):
        for model_param, shard_param in zip(model_group, shard_group):
            if shard_param is None:
                continue

            restored = restore_full_model_param_(
                model_param=model_param,
                param_range=get_param_range_fn(model_param),
                dion_info=get_dion_info_fn(model_param),
                fs_group=fs_group,
                fs_size=fs_size,
            )
            if not restored:
                continue

            if get_dion_info_fn(model_param).get("is_dion", False):
                dion_count += 1
            else:
                non_dion_count += 1

    return dion_count, non_dion_count


def restore_non_dion_group_params_(
    *,
    model_groups,
    shard_groups,
    get_param_range_fn,
    get_dion_info_fn,
    fs_group: dist.ProcessGroup,
    fs_size: int,
) -> int:
    """Restore only non-Dion params in grouped model params.

    This is a diagnostic stepping stone toward standard DO-style no-restore Dion:
    Dion params can stay on the custom FS all-gather path, while mixed-bucket
    non-Dion params still need a full-param restore until they are moved onto a
    standard forward-visible gather path.
    """
    non_dion_count = 0
    for model_group, shard_group in zip(model_groups, shard_groups):
        for model_param, shard_param in zip(model_group, shard_group):
            if shard_param is None:
                continue
            if get_dion_info_fn(model_param).get("is_dion", False):
                continue
            restored = restore_full_model_param_(
                model_param=model_param,
                param_range=get_param_range_fn(model_param),
                dion_info=get_dion_info_fn(model_param),
                fs_group=fs_group,
                fs_size=fs_size,
            )
            if restored:
                non_dion_count += 1
    return non_dion_count
