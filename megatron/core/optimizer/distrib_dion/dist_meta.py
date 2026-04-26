"""Distributed metadata helpers for the Dion optimizer."""

from __future__ import annotations

from typing import Callable

import torch.distributed as dist

from ... import parallel_state
from ..dion.linear import get_linear_split_rows, is_linear_fc1_param
from ..dion.qkv import get_qkv_split_shapes, is_qkv_param, validate_qkv_split_shapes_for_rows
from ..dion.state import build_param_config
from ..dion.types import DionDistMeta
from ..dion.utils import get_local_shape
from .parameter import is_moe_expert_param
from .sharding import get_tp_split_dim, is_tp_enabled
from ...transformer.fsdp_dtensor_checkpoint import get_expert_index_from_key


def get_expert_layout(*, model_param, shard_param, param_name: str, dion_shard_layout):
    """Return local-shape metadata for one expert Dion object."""
    per_expert_global_shape = dion_shard_layout.per_expert_global_shape
    if per_expert_global_shape is None:
        return None

    global_shape = tuple(int(dim) for dim in dion_shard_layout.global_shape)
    per_expert_global_shape = tuple(int(dim) for dim in per_expert_global_shape)
    differing_axes = [axis for axis in (0, 1) if global_shape[axis] != per_expert_global_shape[axis]]
    if not differing_axes:
        return None
    if len(differing_axes) != 1:
        raise RuntimeError(
            "[DION_INVALID_EXPERT_AXIS_LAYOUT] expected exactly one expert axis "
            f"param={param_name} global_shape={global_shape} per_expert_global_shape={per_expert_global_shape}"
        )

    expert_axis = int(differing_axes[0])
    combined_global = int(global_shape[expert_axis])
    expert_global = int(per_expert_global_shape[expert_axis])
    if expert_global <= 0 or combined_global % expert_global != 0:
        raise RuntimeError(
            "[DION_INVALID_EXPERT_LAYOUT_RATIO] "
            f"param={param_name} global_shape={global_shape} per_expert_global_shape={per_expert_global_shape}"
        )
    num_local_experts = combined_global // expert_global
    if num_local_experts <= 1:
        return None

    model_num_local_experts = getattr(model_param, "num_local_experts", None)
    if model_num_local_experts is not None and int(model_num_local_experts) != int(num_local_experts):
        raise RuntimeError(
            "[DION_EXPERT_COUNT_MISMATCH] "
            f"param={param_name} model_num_local_experts={int(model_num_local_experts)} "
            f"derived_num_local_experts={int(num_local_experts)} "
            f"global_shape={global_shape} per_expert_global_shape={per_expert_global_shape}"
        )

    local_shape = tuple(int(dim) for dim in shard_param.shape)
    if local_shape[expert_axis] % num_local_experts != 0:
        raise RuntimeError(
            "[DION_INVALID_LOCAL_EXPERT_LAYOUT] "
            f"param={param_name} local_shape={local_shape} expert_axis={expert_axis} "
            f"num_local_experts={int(num_local_experts)}"
        )

    expert_local_shape = list(local_shape)
    expert_local_shape[expert_axis] //= int(num_local_experts)

    expert_index = get_expert_index_from_key(param_name)
    if expert_index is None:
        raise RuntimeError(
            "[DION_MISSING_EXPERT_INDEX] "
            f"param={param_name} cannot derive local expert index for this expert tensor"
        )
    local_expert_index = int(expert_index) % int(num_local_experts)

    return {
        "local_shape": tuple(int(dim) for dim in expert_local_shape),
        "expert_axis": int(expert_axis),
        "num_local_experts": int(num_local_experts),
        "local_expert_index": int(local_expert_index),
    }


def get_group_size_rank(group) -> tuple[int, int]:
    """Return ``(world_size, rank)`` for a process group, or ``(1, -1)`` if absent."""
    if group is None:
        return 1, -1
    return dist.get_world_size(group), dist.get_rank(group)


def get_group_ranks(group):
    """Return global ranks for a process group, or ``None`` when the group is absent."""
    if group is None:
        return None
    return tuple(dist.get_process_group_ranks(group))


def _linear_split_rows_for_layout(*, model_param, param_name: str, dion_shard_layout, expert_layout):
    if not is_linear_fc1_param(model_param):
        return None
    split_rows = get_linear_split_rows(model_param)
    if split_rows is None:
        return None

    global_rows = int(dion_shard_layout.global_shape[0])
    per_expert_shape = dion_shard_layout.per_expert_global_shape
    if per_expert_shape is None:
        if int(sum(split_rows)) != global_rows:
            raise RuntimeError(
                "[DION_LINEAR_SPLIT_ROWS_MISMATCH] "
                f"param={param_name or id(model_param)} "
                f"split_rows={split_rows} global_rows={global_rows}"
            )
        return split_rows

    per_expert_rows = int(per_expert_shape[0])
    if int(sum(split_rows)) == per_expert_rows:
        return split_rows

    if expert_layout is not None and int(expert_layout["expert_axis"]) == 0:
        num_local_experts = int(expert_layout["num_local_experts"])
        if (
            num_local_experts > 1
            and int(sum(split_rows)) == global_rows
            and all(int(row) % num_local_experts == 0 for row in split_rows)
        ):
            converted = tuple(int(row) // num_local_experts for row in split_rows)
            if int(sum(converted)) == per_expert_rows:
                return converted

    raise RuntimeError(
        "[DION_EXPERT_LINEAR_SPLIT_ROWS_MISMATCH] "
        f"param={param_name or id(model_param)} split_rows={split_rows} "
        f"per_expert_global_shape={per_expert_shape} global_shape={dion_shard_layout.global_shape}"
    )


def assert_context_parallel_excluded(*, label: str, group, extra: str = "") -> None:
    """Fail fast when a Dion group contains the caller's context-parallel peers."""
    if group is None:
        return
    cp_group = parallel_state.get_context_parallel_group(check_initialized=False)
    if cp_group is None:
        return
    cp_ranks = tuple(int(rank) for rank in dist.get_process_group_ranks(cp_group))
    if len(cp_ranks) <= 1:
        return
    group_ranks = tuple(int(rank) for rank in dist.get_process_group_ranks(group))
    global_rank = int(dist.get_rank())
    cp_rank_set = set(cp_ranks)
    overlap = tuple(rank for rank in group_ranks if rank in cp_rank_set)
    if overlap != (global_rank,):
        raise RuntimeError(
            f"[Dion][CP] {label} must exclude context-parallel peers: "
            f"group_ranks={group_ranks} context_parallel_ranks={cp_ranks} "
            f"overlap={overlap} global_rank={global_rank}. {extra}".strip()
        )


def get_expected_expert_fs_group():
    """Return the authoritative Megatron-Core expert-local shard group."""
    group = parallel_state.get_expert_data_parallel_group(
        check_initialized=False,
        partial_expert_data_parallel=True,
    )
    if group is None:
        raise RuntimeError(
            "[Dion][EP] missing expert-local shard group "
            "(expected intra_expt_dp_group for expert param/bucket)."
        )
    return group


def assert_same_group_ranks(*, label: str, actual_group, expected_group, extra: str = "") -> None:
    """Fail fast when two runtime groups do not contain the same global-rank membership."""
    actual_ranks = get_group_ranks(actual_group)
    expected_ranks = get_group_ranks(expected_group)
    if actual_ranks != expected_ranks:
        raise RuntimeError(
            f"[Dion][EP] {label} group mismatch: "
            f"actual={actual_ranks} expected={expected_ranks}. {extra}".strip()
        )


def select_fs_group(*, model_param, fs_group):
    """Return the authoritative local-shard group for one model param."""
    if not getattr(model_param, "allreduce", True):
        expert_group = get_expected_expert_fs_group()
        assert_same_group_ranks(
            label="expert param fs_group",
            actual_group=fs_group,
            expected_group=expert_group,
            extra=(
                f"param={getattr(model_param, '_param_name', '') or id(model_param)} "
                "Dion EP must keep expert params on the standard expert-local DO shard group."
            ),
        )
        assert_context_parallel_excluded(
            label="expert param fs_group",
            group=expert_group,
            extra=(
                f"param={getattr(model_param, '_param_name', '') or id(model_param)} "
                "CP must not be part of Dion expert FS/orthogonalization domains."
            ),
        )
        return expert_group
    return fs_group


def select_tp_group(model_param):
    """Return the authoritative TP group for one model param, if TP is active."""
    if not is_tp_enabled(model_param):
        return None
    if is_moe_expert_param(model_param, getattr(model_param, "_param_name", None)):
        group = parallel_state.get_expert_tensor_parallel_group(check_initialized=False)
        if group is None:
            raise RuntimeError(
                "[Dion][EP] missing expert TP group for expert TP-sharded param "
                f"param={getattr(model_param, '_param_name', '') or id(model_param)}"
            )
        assert_context_parallel_excluded(
            label="expert TP group",
            group=group,
            extra=(
                f"param={getattr(model_param, '_param_name', '') or id(model_param)} "
                "CP must not be part of Dion distributed orthogonalization domains."
            ),
        )
        return group
    group = parallel_state.get_tensor_model_parallel_group(check_initialized=False)
    assert_context_parallel_excluded(
        label="dense TP group",
        group=group,
        extra=(
            f"param={getattr(model_param, '_param_name', '') or id(model_param)} "
            "CP must not be part of Dion distributed orthogonalization domains."
        ),
    )
    return group


def make_param_uid(
    *,
    param_name: str,
    global_shape=None,
    is_dion_param: bool,
):
    """Build a topology-independent optimizer-state identity."""
    if not param_name:
        raise RuntimeError("[Dion] Missing param_name while building param_uid")
    return (
        str(param_name or ""),
        tuple(int(dim) for dim in global_shape) if global_shape is not None else (),
        bool(is_dion_param),
    )


def build_param_dist_meta(
    *,
    model_param,
    shard_param,
    fs_group,
    tp_group=None,
    shard_layouts_by_param,
    get_param_name: Callable,
    rank_fraction_default: float,
    rank_multiple_of_default: int,
    use_low_rank_sync: bool,
) -> DionDistMeta:
    """Build one Dion distributed metadata record from adapter shard metadata."""
    param_name = get_param_name(model_param) or ""
    fs_group = select_fs_group(model_param=model_param, fs_group=fs_group)
    fs_world_size, fs_rank = get_group_size_rank(fs_group)
    if is_tp_enabled(model_param):
        if tp_group is None:
            tp_group = select_tp_group(model_param)
        else:
            assert_context_parallel_excluded(
                label="expert TP group" if is_moe_expert_param(model_param, param_name) else "dense TP group",
                group=tp_group,
                extra=(
                    f"param={param_name or id(model_param)} "
                    "CP must not be part of Dion distributed orthogonalization domains."
                ),
            )
    else:
        tp_group = None
    tp_world_size, tp_rank = get_group_size_rank(tp_group)
    tp_shard_dim = (
        get_tp_split_dim(model_param)
        if is_tp_enabled(model_param) and tp_world_size > 1
        else -1
    )
    dion_shard_layout = shard_layouts_by_param.get(model_param)
    if dion_shard_layout is None:
        raise RuntimeError(
            "[Dion] missing dion_shard_layout while building distributed metadata "
            f"for param={param_name or id(model_param)}"
        )

    expert_layout = get_expert_layout(
        model_param=model_param,
        shard_param=shard_param,
        param_name=param_name,
        dion_shard_layout=dion_shard_layout,
    )

    qkv_split_shapes = get_qkv_split_shapes(model_param) if is_qkv_param(model_param) else None
    if qkv_split_shapes is not None:
        qkv_global_rows = int(
            (
                dion_shard_layout.per_expert_global_shape
                if dion_shard_layout.per_expert_global_shape is not None
                else dion_shard_layout.global_shape
            )[0]
        )
        validate_qkv_split_shapes_for_rows(
            qkv_split_shapes,
            rows=qkv_global_rows,
            context=f"param={param_name or id(model_param)}",
        )

    dist_meta = DionDistMeta(
        shape=shard_param.shape,
        global_shape=dion_shard_layout.global_shape,
        fs_start_idx=int(dion_shard_layout.start_idx),
        fs_end_idx=int(dion_shard_layout.end_idx),
        tp_shard_dim=tp_shard_dim,
        fs_shard_dim=dion_shard_layout.fs_shard_dim,
        rank_fraction=rank_fraction_default,
        is_dion_param=getattr(model_param, "is_dion_param", True),
        is_transposed=bool(shard_param.shape[0] < shard_param.shape[1]),
        param_uid=make_param_uid(
            param_name=param_name,
            global_shape=(
                dion_shard_layout.per_expert_global_shape
                if dion_shard_layout.per_expert_global_shape is not None
                else dion_shard_layout.global_shape
            ),
            is_dion_param=True,
        ),
        param_name=param_name,
        fs_group=fs_group,
        fs_world_size=fs_world_size,
        fs_rank=fs_rank,
        tp_group=tp_group,
        tp_world_size=tp_world_size,
        tp_rank=tp_rank,
        per_expert_global_shape=dion_shard_layout.per_expert_global_shape,
        local_shape=(
            expert_layout["local_shape"] if expert_layout is not None else None
        ),
        expert_axis=(
            int(expert_layout["expert_axis"]) if expert_layout is not None else -1
        ),
        num_local_experts=(
            int(expert_layout["num_local_experts"]) if expert_layout is not None else 1
        ),
        local_expert_index=(
            int(expert_layout["local_expert_index"]) if expert_layout is not None else -1
        ),
        qkv_split_shapes=qkv_split_shapes,
        linear_split_rows=_linear_split_rows_for_layout(
            model_param=model_param,
            param_name=param_name,
            dion_shard_layout=dion_shard_layout,
            expert_layout=expert_layout,
        ),
        linear_partition_stride=(
            int(getattr(model_param, "partition_stride", 1))
            if is_linear_fc1_param(model_param)
            else 1
        ),
    )

    config_local_shape = get_local_shape(
        dist_meta,
        int(shard_param.shape[0]),
        int(shard_param.shape[1]),
    )
    dist_meta.param_config = build_param_config(
        param_ndim=shard_param.ndim,
        local_shape=config_local_shape if shard_param.ndim == 2 else None,
        dist_meta=dist_meta,
        use_low_rank_sync=bool(use_low_rank_sync),
        r_global_override=None,
        rank_fraction_default=rank_fraction_default,
        rank_multiple_of_default=rank_multiple_of_default,
        tp_world_size=tp_world_size,
        tp_active=bool(tp_shard_dim in (0, 1) and tp_world_size > 1),
    )

    return dist_meta


def add_standard_metas(
    *,
    param_groups,
    dist_metas_sharded,
    get_param_name: Callable,
    get_direct_param_name: Callable,
    rank_fraction_default: float,
    rank_multiple_of_default: int,
    use_low_rank_sync: bool,
) -> None:
    """Add dist_metas for standard parameters."""
    for param_group in param_groups:
        for param in param_group['params']:
            if param in dist_metas_sharded:
                continue

            model_param = getattr(param, '_model_param', None)
            if model_param is None:
                raise RuntimeError(
                    "[Dion] standard optimizer shard is missing _model_param backlink; "
                    f"shape={tuple(param.shape)} id={id(param)}"
                )

            param_name = (
                get_param_name(model_param)
                or get_param_name(param)
                or get_direct_param_name(model_param)
                or get_direct_param_name(param)
                or ""
            )
            param_uid = make_param_uid(
                param_name=param_name,
                global_shape=None,
                is_dion_param=getattr(model_param, 'is_dion_param', False),
            )
            tp_group = select_tp_group(model_param)
            tp_world_size, tp_rank = get_group_size_rank(tp_group)
            dist_meta = DionDistMeta(
                shape=param.shape,
                global_shape=tuple(model_param.shape) if model_param.ndim == 2 else None,
                fs_start_idx=-1,
                fs_end_idx=-1,
                tp_shard_dim=-1,
                rank_fraction=rank_fraction_default,
                is_dion_param=getattr(model_param, 'is_dion_param', False),
                is_transposed=bool(param.ndim == 2 and param.shape[0] < param.shape[1]),
                param_uid=param_uid,
                param_name=param_name,
                tp_group=tp_group,
                tp_world_size=tp_world_size,
                tp_rank=tp_rank,
                qkv_split_shapes=(
                    get_qkv_split_shapes(model_param)
                    if is_qkv_param(model_param)
                    else None
                ),
                linear_split_rows=None,
                linear_partition_stride=1,
            )
            dist_meta.param_config = build_param_config(
                param_ndim=param.ndim,
                local_shape=tuple(param.shape) if param.ndim == 2 else None,
                dist_meta=dist_meta,
                use_low_rank_sync=bool(use_low_rank_sync),
                r_global_override=None,
                rank_fraction_default=rank_fraction_default,
                rank_multiple_of_default=rank_multiple_of_default,
                tp_world_size=1,
                tp_active=False,
            )
            dist_metas_sharded[param] = dist_meta
            param._dion_param_uid = param_uid


def build_all_dist_metas(
    *,
    shard_pairs_by_param,
    build_param_dist_meta: Callable,
    add_standard_metas: Callable,
    validate_dist_meta_uids: Callable,
):
    """Create distributed metadata for Dion and standard optimizer shards."""
    dist_metas_sharded = {}

    for model_param, shard_pair in shard_pairs_by_param.items():
        _, shard_param = shard_pair
        dist_meta = build_param_dist_meta(
            model_param=model_param,
            shard_param=shard_param,
        )
        dist_metas_sharded[shard_param] = dist_meta
        shard_param._dion_param_uid = dist_meta.param_uid

    add_standard_metas(dist_metas_sharded)
    validate_dist_meta_uids(dist_metas_sharded)
    return dist_metas_sharded


def validate_dist_meta_uids(
    *,
    dist_metas_sharded,
    get_param_name: Callable,
    log_error: Callable,
) -> None:
    """Fail fast when multiple local shards share the same param uid."""
    uid_to_entries = {}
    for shard_param, dist_meta in dist_metas_sharded.items():
        uid_to_entries.setdefault(dist_meta.param_uid, []).append(
            {
                "name": getattr(dist_meta, "param_name", "")
                or get_param_name(shard_param),
                "shape": (
                    tuple(dist_meta.shape)
                    if dist_meta.shape is not None
                    else tuple(shard_param.shape)
                ),
                "is_dion": bool(getattr(dist_meta, "is_dion_param", False)),
            }
        )
    duplicate_uids = {
        uid: entries for uid, entries in uid_to_entries.items() if len(entries) > 1
    }
    if not duplicate_uids:
        return

    for uid, entries in sorted(duplicate_uids.items(), key=lambda item: str(item[0])):
        log_error("[Dion] Duplicate param_uid=%s entries=%s", uid, entries)
    raise RuntimeError(
        "[Dion] Duplicate param_uid detected in dist_metas; "
        "optimizer state identity would be ambiguous"
    )
