"""Checkpoint and export helpers for Dion distributed optimizer.

These helpers are intended to be behavior-preserving refactors: no changes to
math or collective patterns, only code organization.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import torch
import torch.distributed as dist

from ..dion.qkv import iter_qkv_child_kinds, qkv_state_key
from .sharding import (
    DionShardLayout,
    compute_fs_shard_range,
    fs_shard_view_2d,
    set_fs_shard_view_2d,
)

logger = logging.getLogger(__name__)


def _param_name(param: torch.Tensor) -> str:
    """Return a deterministic best-effort parameter name for errors/logs."""
    name = getattr(param, "_param_name", None)
    if name is not None:
        return name
    model_param = getattr(param, "_model_param", None)
    if model_param is not None:
        name = getattr(model_param, "_param_name", None)
        if name is not None:
            return name
    return f"id_{id(param)}"


def build_persistent_dion_param_state(param_groups, optimizer_state, get_param_key) -> dict:
    """Build persistent Dion optimizer state keyed by `param_uid`.

    This is intentionally narrower than the raw optimizer state:
    per-call gather buffers and transient sync flags are excluded.
    """
    key_to_state = {}
    for param_group in param_groups:
        for param in param_group["params"]:
            state = optimizer_state.get(param)
            if not state:
                continue
            param_key = get_param_key(param)
            if param_key is None:
                continue
            persistent_state = {}
            for key, value in state.items():
                if key.startswith("_"):
                    continue
                persistent_state[key] = value
            if persistent_state:
                key_to_state[param_key] = persistent_state
    return key_to_state


def restore_persistent_dion_param_state_(
    *,
    param_groups,
    optimizer_state,
    get_param_key,
    key_to_state: dict,
) -> dict[str, int]:
    """Restore persistent Dion optimizer state keyed by `param_uid`.

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

    for param_group in param_groups:
        for param in param_group["params"]:
            param_key = get_param_key(param)
            current_state = optimizer_state.get(param, {})
            if param_key is None:
                summary["unnamed"] += 1
                continue
            if param_key not in key_to_state:
                summary["no_payload_entry"] += 1
                continue

            saved_state = key_to_state[param_key]
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
                            f"[Dion] checkpoint tensor shape mismatch for {param_key}.{key}: "
                            f"saved={tuple(value.shape)} current={tuple(current_value.shape)}"
                        )
                    new_state[key] = value.to(device=param.device)
                else:
                    new_state[key] = value

            if bool(new_state.get("qkv_split_qkv", False)):
                for child_kind in iter_qkv_child_kinds():
                    q_key = qkv_state_key("Q", child_kind)
                    if q_key in new_state:
                        new_state[f"_qkv_{child_kind}_needs_state_replica_q_sync"] = True

            optimizer_state[param] = new_state
            summary["restored"] += 1

    return summary


def build_distributed_dion_checkpoint_state(
    *,
    common_state: dict,
    param_groups,
    optimizer_state,
    get_param_key,
    base_key: str,
    replica_id,
    sharded_object_cls,
) -> dict:
    """Build distributed checkpoint state with standard common-state outer layout."""
    state_dict = {
        key: sharded_object_cls(
            f"{base_key}.{key}",
            value,
            (1,),
            (0,),
            replica_id=replica_id,
        )
        for key, value in common_state.items()
    }

    dion_param_state = build_persistent_dion_param_state(
        param_groups,
        optimizer_state,
        get_param_key,
    )
    state_dict["dion_param_state"] = sharded_object_cls(
        f"{base_key}.dion_param_state",
        dion_param_state,
        (1,),
        (0,),
        replica_id=replica_id,
    )
    return state_dict


def split_distributed_dion_checkpoint_state(state_dict: dict) -> tuple[Optional[dict], dict]:
    """Split Dion payload from the standard common-state outer protocol."""
    dion_param_state = state_dict.get("dion_param_state", None)
    if dion_param_state is None and state_dict.get("param_state_sharding_type") == "dion_non_reshardable":
        dion_param_state = state_dict.get("param_state", None)

    common_state_dict = {
        key: value
        for key, value in state_dict.items()
        if key not in ("dion_param_state", "param_state", "param_state_sharding_type")
    }
    return dion_param_state, common_state_dict


def ensure_dion_state_initialized_for_load(
    *,
    param_groups,
    optimizer_state,
    init_state,
) -> None:
    """Initialize Dion optimizer state tensors before loading persistent payload."""
    if init_state is None:
        return

    for param_group in param_groups:
        for param in param_group["params"]:
            state = optimizer_state.get(param)
            if state is None:
                optimizer_state[param] = {}
                state = optimizer_state[param]
            if len(state) == 0:
                init_state(param, state, param_group)


def restore_distributed_dion_checkpoint_state(
    *,
    dion_param_state,
    param_groups,
    optimizer_state,
    get_param_key,
) -> dict[str, int]:
    """Restore Dion-specific distributed checkpoint payload and validate completeness."""
    if not isinstance(dion_param_state, dict) or len(dion_param_state) == 0:
        raise RuntimeError("[Dion] distributed checkpoint missing Dion param state payload")

    restore_summary = restore_persistent_dion_param_state_(
        param_groups=param_groups,
        optimizer_state=optimizer_state,
        get_param_key=get_param_key,
        key_to_state=dion_param_state,
    )
    if restore_summary["unnamed"] > 0 or restore_summary["no_payload_entry"] > 0:
        raise RuntimeError(
            "[Dion] distributed checkpoint restore left unresolved param state entries "
            f"(restored={restore_summary['restored']} "
            f"no_payload_entry={restore_summary['no_payload_entry']} "
            f"unnamed={restore_summary['unnamed']})"
        )
    return restore_summary


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
    - write them back into the correct ranges
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


def all_gather_fs_shards_2d(
    param_2d: torch.Tensor,
    *,
    fs_shard_dim: int,
    start_idx: int,
    end_idx: int,
    fs_group: dist.ProcessGroup,
    fs_size: int,
) -> None:
    """All-gather FS-sharded 2D shards back into `param_2d` in-place.

    This matches the existing Dion code path: pad to max shard size along the FS
    axis, all_gather, then write each rank's actual shard view back in place.
    """
    local_shard_size = end_idx - start_idx
    if local_shard_size <= 0:
        return

    local_shard = fs_shard_view_2d(param_2d, fs_shard_dim, start_idx, end_idx).contiguous()
    other_dim_size = param_2d.shape[1] if fs_shard_dim == 0 else param_2d.shape[0]

    split_dim_size = param_2d.shape[fs_shard_dim]
    max_shard_size = (split_dim_size + fs_size - 1) // fs_size

    if fs_shard_dim == 0:
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
        if fs_shard_dim == 0:
            rank_shard_2d = gathered_shards[rank_i][:actual_size, :]
        else:
            rank_shard_2d = gathered_shards[rank_i][:, :actual_size]
        set_fs_shard_view_2d(param_2d, fs_shard_dim, r_start, r_end, rank_shard_2d)


def copy_param_to_shard_main_(
    *,
    source_param: torch.Tensor,
    shard_main_param: torch.nn.Parameter,
    dion_shard_layout: Optional[DionShardLayout],
    param_range,
) -> None:
    """Copy a source model/state-dict param into the local optimizer shard."""
    from ...fp8_utils import dequantize_fp8_tensor, is_float8tensor

    if is_float8tensor(source_param):
        source_param_fp32 = dequantize_fp8_tensor(source_param)
    else:
        source_param_fp32 = source_param

    if dion_shard_layout is not None:
        start_idx = int(dion_shard_layout.start_idx)
        end_idx = int(dion_shard_layout.end_idx)
        fs_shard_dim = int(dion_shard_layout.fs_shard_dim)

        if source_param_fp32.ndim == 2:
            source_2d = source_param_fp32
        else:
            source_2d = source_param_fp32.view(shard_main_param.shape)

        shard_model_param = (
            fs_shard_view_2d(source_2d, fs_shard_dim, start_idx, end_idx).clone().view(-1)
        )
        assert shard_model_param.numel() == shard_main_param.numel(), (
            f"FS shard size mismatch: shard_model_param={shard_model_param.numel()}, "
            f"shard_main_param={shard_main_param.numel()}, "
            f"fs_shard_dim={fs_shard_dim}, start_idx={start_idx}, end_idx={end_idx}, "
            f"source_shape={source_param_fp32.shape}, "
            f"global_shape={tuple(dion_shard_layout.global_shape)}"
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
    get_source_param: Callable[[torch.nn.Parameter], torch.Tensor],
    get_param_range: Callable[[torch.nn.Parameter], object],
    get_dion_shard_layout: Callable[[torch.nn.Parameter], Optional[DionShardLayout]],
) -> None:
    """Copy one or more model param groups into their local optimizer shards."""
    for model_group, shard_main_group in zip(model_groups, shard_main_groups):
        for model_param, shard_main_param in zip(model_group, shard_main_group):
            if shard_main_param is None:
                continue

            copy_param_to_shard_main_(
                source_param=get_source_param(model_param),
                shard_main_param=shard_main_param,
                dion_shard_layout=get_dion_shard_layout(model_param),
                param_range=get_param_range(model_param),
            )


def copy_model_params_to_main_shards(
    *,
    is_hybrid_device_optimizer: bool,
    hybrid_optimizer_update: Callable | None,
    use_megatron_fsdp: bool,
    use_precision_aware_optimizer: bool,
    state_dict,
    build_model_param_to_state_dict_param_map: Callable,
    model_float16_groups,
    main_shard_groups,
    model_fp32_groups,
    shard_fp32_groups,
    get_model_param_range_map: Callable,
    get_dion_shard_layout: Callable,
) -> None:
    """Copy model params onto optimizer main shards using the canonical adapter mapping."""
    if is_hybrid_device_optimizer:
        if hybrid_optimizer_update is None:
            raise RuntimeError(
                "[Dion] HybridDeviceOptimizer path requires update_fp32_param_by_new_param callback"
            )
        hybrid_optimizer_update()
        return

    if use_megatron_fsdp or use_precision_aware_optimizer:
        return

    model_param_to_state_dict_param_map = None
    if state_dict is not None:
        model_param_to_state_dict_param_map = build_model_param_to_state_dict_param_map(
            state_dict
        )

    if model_param_to_state_dict_param_map is not None:
        get_source_param = lambda model_param: model_param_to_state_dict_param_map[model_param]
    else:
        get_source_param = lambda model_param: model_param
    get_param_range = lambda model_param: get_model_param_range_map(model_param)["param"]

    copy_group_params_to_main_shards_(
        model_groups=model_float16_groups,
        shard_main_groups=main_shard_groups,
        get_source_param=get_source_param,
        get_param_range=get_param_range,
        get_dion_shard_layout=get_dion_shard_layout,
    )
    copy_group_params_to_main_shards_(
        model_groups=model_fp32_groups,
        shard_main_groups=shard_fp32_groups,
        get_source_param=get_source_param,
        get_param_range=get_param_range,
        get_dion_shard_layout=get_dion_shard_layout,
    )


def apply_optimizer_shard_to_model_param_(
    *,
    model_param: torch.nn.Parameter,
    opt_shard: torch.Tensor,
    data_shard: torch.Tensor,
    param_range_map,
    dion_shard_layout: Optional[DionShardLayout],
    get_bucket_param_data: Callable[[torch.nn.Parameter], torch.Tensor] | None,
    zero_range_warned: int,
) -> int:
    """Copy an updated optimizer shard into `data_shard` and `model_param.data`."""
    data_shard.data.copy_(opt_shard.to(data_shard.dtype))
    model_param._fs_shard = data_shard

    if dion_shard_layout is not None:
        fs_shard_dim = int(dion_shard_layout.fs_shard_dim)
        start_idx = int(dion_shard_layout.start_idx)
        end_idx = int(dion_shard_layout.end_idx)

        if start_idx == 0 and end_idx == 0 and zero_range_warned < 5:
            param_name = getattr(model_param, "_param_name", f"shape={model_param.shape}")
            logger.error(
                "[ZERO RANGE] start_idx=0, end_idx=0 for param %s! No data will be copied to model_param.data! dion_shard_layout=%s",
                param_name,
                dion_shard_layout,
            )
            zero_range_warned += 1

        if fs_shard_dim == 0:
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
                f"fs_shard_dim={fs_shard_dim}, range=[{start_idx}:{end_idx}]"
            )

        target_view = fs_shard_view_2d(model_param.data, fs_shard_dim, start_idx, end_idx)
        src_view = data_shard.view(expected_rows, expected_cols)
        if src_view.data_ptr() == target_view.data_ptr():
            return zero_range_warned
        param_name = getattr(model_param, "_param_name", f"id_{id(model_param)}")
        logger.error(
            "[DION_FS_ALIAS_MISMATCH] param=%s data_shard_ptr=%s target_ptr=%s fs_shard_dim=%s range=[%s:%s]",
            param_name,
            src_view.data_ptr(),
            target_view.data_ptr(),
            fs_shard_dim,
            start_idx,
            end_idx,
        )
        raise RuntimeError(
            f"[Dion] FS shard alias mismatch for {param_name}: "
            f"data_shard no longer aliases canonical model_param.data view "
            f"(fs_shard_dim={fs_shard_dim}, range=[{start_idx}:{end_idx}])"
        )

    if param_range_map is None or "gbuf_world_in_bucket" not in param_range_map:
        param_name = _param_name(model_param)
        raise RuntimeError(
            "[Dion] non-Dion write-back requires canonical gbuf_world_in_bucket "
            f"for {param_name}"
        )

    world_range = param_range_map["gbuf_world_in_bucket"]
    if get_bucket_param_data is None:
        param_name = _param_name(model_param)
        raise RuntimeError(
            "[Dion] non-Dion write-back requires canonical bucket.param_data "
            f"for {param_name}"
        )
    bucket_param_data = get_bucket_param_data(model_param)
    if bucket_param_data is None:
        param_name = _param_name(model_param)
        raise RuntimeError(
            "[Dion] non-Dion write-back missing bucket.param_data "
            f"for {param_name}"
        )
    target_flat = bucket_param_data.view(-1)[world_range.start : world_range.end]
    if target_flat.numel() != opt_shard.numel():
        param_name = _param_name(model_param)
        param_local_range = param_range_map.get("param", None)
        raise RuntimeError(
            "[Dion] non-Dion write-back size mismatch "
            f"param={param_name} world_range_size={target_flat.numel()} "
            f"opt_shard_numel={opt_shard.numel()} "
            f"current_world_range=[{world_range.start}:{world_range.end}] "
            f"param_local_range=["
            f"{None if param_local_range is None else param_local_range.start}:"
            f"{None if param_local_range is None else param_local_range.end}]"
        )

    target_view = target_flat.view_as(opt_shard)
    if target_view.data_ptr() != data_shard.data_ptr():
        target_view.copy_(opt_shard.to(target_view.dtype))
    return zero_range_warned


def apply_non_dion_shards_(
    *,
    model_groups,
    shard_groups,
    get_param_range_map: Callable[[torch.nn.Parameter], dict],
    get_bucket_param_data: Callable[[torch.nn.Parameter], torch.Tensor] | None,
) -> int:
    """Write back non-Dion optimizer shards using the standard DO local-shard contract."""
    param_count = 0
    for model_group, shard_param_group in zip(model_groups, shard_groups):
        for model_param, shard_param in zip(model_group, shard_param_group):
            if shard_param is None or getattr(model_param, "is_dion_param", False):
                continue
            if get_bucket_param_data is None:
                param_name = _param_name(model_param)
                raise RuntimeError(
                    "[Dion] non-Dion write-back requires canonical bucket.param_data "
                    f"for {param_name}"
                )
            bucket_param_data = get_bucket_param_data(model_param)
            if bucket_param_data is None:
                param_name = _param_name(model_param)
                raise RuntimeError(
                    "[Dion] non-Dion write-back missing bucket.param_data "
                    f"for {param_name}"
                )
            param_range_map = get_param_range_map(model_param)
            if param_range_map is None or "gbuf_world_in_bucket" not in param_range_map:
                param_name = _param_name(model_param)
                raise RuntimeError(
                    "[Dion] non-Dion write-back requires canonical gbuf_world_in_bucket "
                    f"for {param_name}"
                )
            world_range = param_range_map["gbuf_world_in_bucket"]
            target_flat = bucket_param_data.view(-1)[world_range.start : world_range.end]
            if target_flat.numel() != shard_param.nelement():
                param_name = _param_name(model_param)
                raise RuntimeError(
                    "[Dion] non-Dion standard write-back size mismatch "
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
    get_data_shard: Callable[[torch.nn.Parameter], torch.Tensor],
    get_param_range_map: Callable[[torch.nn.Parameter], dict],
    get_dion_shard_layout: Callable[[torch.nn.Parameter], Optional[DionShardLayout]],
    get_bucket_param_data: Callable[[torch.nn.Parameter], torch.Tensor] | None,
    zero_range_warned: int,
) -> tuple[int, int]:
    """Apply updated Dion optimizer shards to grouped model params."""
    param_count = 0

    for model_group, shard_param_group, shard16_param_group in zip(
        model_groups,
        shard_groups,
        shard16_groups,
    ):
        for model_param, shard_param, shard16_param in zip(
            model_group, shard_param_group, shard16_param_group
        ):
            if shard_param is None:
                continue
            dion_shard_layout = get_dion_shard_layout(model_param)
            if dion_shard_layout is None:
                continue

            data_shard = get_data_shard(model_param)
            if data_shard is None:
                data_shard = shard16_param

            zero_range_warned = apply_optimizer_shard_to_model_param_(
                model_param=model_param,
                opt_shard=shard_param,
                data_shard=data_shard,
                param_range_map=get_param_range_map(model_param),
                dion_shard_layout=dion_shard_layout,
                get_bucket_param_data=get_bucket_param_data,
                zero_range_warned=zero_range_warned,
            )
            param_count += 1

    return param_count, zero_range_warned


def copy_main_params_to_model_shards(
    *,
    is_stub_optimizer: bool,
    use_megatron_fsdp: bool,
    copy_fsdp_main_to_model_weights: Callable | None,
    use_precision_aware_optimizer: bool,
    model_float16_groups,
    main_shard_groups,
    shard_float16_groups,
    model_fp32_groups,
    shard_fp32_groups,
    get_data_shard: Callable[[torch.nn.Parameter], torch.Tensor],
    get_param_range_map: Callable[[torch.nn.Parameter], dict],
    get_dion_shard_layout: Callable[[torch.nn.Parameter], Optional[DionShardLayout]],
    get_bucket_param_data: Callable[[torch.nn.Parameter], torch.Tensor] | None,
    mark_buckets_full_param_ready: Callable[[bool], None],
    check_main_shards: Callable,
    restore_model_params_to_canonical_bucket_storage: Callable,
    zero_range_warned: int,
) -> int:
    """Copy updated optimizer main shards back onto model-param local shards."""
    if is_stub_optimizer:
        return zero_range_warned

    if use_megatron_fsdp:
        if copy_fsdp_main_to_model_weights is None:
            raise RuntimeError(
                "[Dion] FSDP param restore requires copy_main_weights_to_model_weights callback"
            )
        copy_fsdp_main_to_model_weights()
        return zero_range_warned

    if use_precision_aware_optimizer:
        return zero_range_warned

    mark_buckets_full_param_ready(False)
    check_main_shards(main_shard_groups)

    apply_non_dion_shards_(
        model_groups=model_float16_groups,
        shard_groups=main_shard_groups,
        get_param_range_map=get_param_range_map,
        get_bucket_param_data=get_bucket_param_data,
    )
    _, zero_range_warned = apply_group_shards_to_model_params_(
        model_groups=model_float16_groups,
        shard_groups=main_shard_groups,
        shard16_groups=shard_float16_groups,
        get_data_shard=get_data_shard,
        get_param_range_map=get_param_range_map,
        get_dion_shard_layout=get_dion_shard_layout,
        get_bucket_param_data=get_bucket_param_data,
        zero_range_warned=zero_range_warned,
    )

    apply_non_dion_shards_(
        model_groups=model_fp32_groups,
        shard_groups=shard_fp32_groups,
        get_param_range_map=get_param_range_map,
        get_bucket_param_data=get_bucket_param_data,
    )

    restore_model_params_to_canonical_bucket_storage(dion_only=True)

    _, zero_range_warned = apply_group_shards_to_model_params_(
        model_groups=model_fp32_groups,
        shard_groups=shard_fp32_groups,
        shard16_groups=shard_fp32_groups,
        get_data_shard=get_data_shard,
        get_param_range_map=get_param_range_map,
        get_dion_shard_layout=get_dion_shard_layout,
        get_bucket_param_data=get_bucket_param_data,
        zero_range_warned=zero_range_warned,
    )

    return zero_range_warned


def restore_full_model_param_(
    *,
    model_param: torch.nn.Parameter,
    param_range,
    dion_shard_layout: Optional[DionShardLayout],
    fs_group: dist.ProcessGroup,
    fs_size: int,
) -> bool:
    """Restore `model_param.data` to its full view by all-gathering local shards."""
    if dion_shard_layout is None:
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

    fs_shard_dim = int(dion_shard_layout.fs_shard_dim)
    start_idx = int(dion_shard_layout.start_idx)
    end_idx = int(dion_shard_layout.end_idx)
    if start_idx == end_idx:
        return False

    all_gather_fs_shards_2d(
        model_param.data,
        fs_shard_dim=fs_shard_dim,
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
    get_param_range,
    get_dion_shard_layout,
    fs_group: dist.ProcessGroup,
    fs_size: int,
) -> tuple[int, int]:
    """Restore full params for a group list and return `(dion_count, non_dion_count)`."""
    dion_count = 0
    non_dion_count = 0

    for model_group, shard_param_group in zip(model_groups, shard_groups):
        for model_param, shard_param in zip(model_group, shard_param_group):
            if shard_param is None:
                continue

            restored = restore_full_model_param_(
                model_param=model_param,
                param_range=get_param_range(model_param),
                dion_shard_layout=get_dion_shard_layout(model_param),
                fs_group=fs_group,
                fs_size=fs_size,
            )
            if not restored:
                continue

            if get_dion_shard_layout(model_param) is not None:
                dion_count += 1
            else:
                non_dion_count += 1

    return dion_count, non_dion_count
