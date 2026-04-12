"""Adapter-owned distributed metadata helpers for Dion optimizer wiring."""

from __future__ import annotations

from typing import Callable

import torch.distributed as dist

from ... import parallel_state
from ..dion.state import build_param_config
from ..dion.types import DionDistMeta
from .param_selection import is_moe_expert_param
from .param_utils import get_tp_split_dim, is_tp_enabled


def group_info_(group) -> tuple[int, int]:
    """Return ``(world_size, rank)`` for a process group, or ``(1, -1)`` if absent."""
    if group is None:
        return 1, -1
    return dist.get_world_size(group), dist.get_rank(group)


def group_ranks_(group):
    """Return global ranks for a process group, or ``None`` when the group is absent."""
    if group is None:
        return None
    return tuple(dist.get_process_group_ranks(group))


def expected_expert_outer_shard_group_():
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


def assert_group_matches_(*, label: str, actual_group, expected_group, extra: str = "") -> None:
    """Fail fast when two runtime groups do not publish the same global-rank membership."""
    actual_ranks = group_ranks_(actual_group)
    expected_ranks = group_ranks_(expected_group)
    if actual_ranks != expected_ranks:
        raise RuntimeError(
            f"[Dion][EP] {label} group mismatch: "
            f"actual={actual_ranks} expected={expected_ranks}. {extra}".strip()
        )


def select_outer_shard_group_(*, model_param, fs_group):
    """Return the authoritative local-shard group for one model param."""
    if not getattr(model_param, "allreduce", True):
        expert_group = expected_expert_outer_shard_group_()
        assert_group_matches_(
            label="expert param outer_shard_group",
            actual_group=fs_group,
            expected_group=expert_group,
            extra=(
                f"param={getattr(model_param, '_param_name', '') or id(model_param)} "
                "Dion EP must keep expert params on the standard expert-local DO shard group."
            ),
        )
        return expert_group
    return fs_group


def select_tp_group_(model_param):
    """Return the authoritative TP group binding for one model param, if TP is active."""
    if not is_tp_enabled(model_param):
        return None
    if is_moe_expert_param(model_param, getattr(model_param, "_param_name", None)):
        group = parallel_state.get_expert_tensor_parallel_group(check_initialized=False)
        if group is None:
            raise RuntimeError(
                "[Dion][EP] missing expert TP group for expert TP-sharded param "
                f"param={getattr(model_param, '_param_name', '') or id(model_param)}"
            )
        return group
    return parallel_state.get_tensor_model_parallel_group(check_initialized=False)


def make_param_uid_(
    *,
    param_name: str,
    logical_global_shape=None,
    is_dion_param: bool,
):
    """Build a topology-independent logical optimizer-state identity."""
    if not param_name:
        raise RuntimeError("[Dion] Missing param_name while building logical param_uid")
    return (
        str(param_name or ""),
        tuple(int(dim) for dim in logical_global_shape) if logical_global_shape is not None else (),
        bool(is_dion_param),
    )


def build_dist_meta_(
    *,
    model_param,
    shard_param,
    fs_group,
    shard_layouts_by_param,
    logical_param_name_fn: Callable,
    rank_fraction_default: float,
    rank_multiple_of_default: int,
    use_compressed_comm: bool,
) -> DionDistMeta:
    """Build one Dion distributed metadata record from adapter-owned shard bindings."""
    param_name = logical_param_name_fn(model_param) or ""
    outer_shard_group = select_outer_shard_group_(model_param=model_param, fs_group=fs_group)
    fs_world_size, fs_rank = group_info_(outer_shard_group)
    tp_group = select_tp_group_(model_param)
    tp_world_size, tp_rank = group_info_(tp_group)
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

    dist_meta = DionDistMeta(
        shape=shard_param.shape,
        global_shape=dion_shard_layout.global_shape,
        tp_shard_dim=tp_shard_dim,
        fs_shard_dim=dion_shard_layout.fs_shard_dim,
        rank_fraction=rank_fraction_default,
        is_dion_param=getattr(model_param, "is_dion_param", True),
        is_transposed=bool(shard_param.shape[0] < shard_param.shape[1]),
        param_uid=make_param_uid_(
            param_name=param_name,
            logical_global_shape=(
                dion_shard_layout.per_expert_global_shape
                if dion_shard_layout.per_expert_global_shape is not None
                else dion_shard_layout.global_shape
            ),
            is_dion_param=True,
        ),
        param_name=param_name,
        outer_shard_group=outer_shard_group,
        fs_world_size=fs_world_size,
        fs_rank=fs_rank,
        tp_group=tp_group,
        tp_world_size=tp_world_size,
        tp_rank=tp_rank,
        per_expert_global_shape=dion_shard_layout.per_expert_global_shape,
    )

    dist_meta.param_config = build_param_config(
        param_ndim=shard_param.ndim,
        local_shape=tuple(shard_param.shape) if shard_param.ndim == 2 else None,
        dist_meta=dist_meta,
        use_compressed_comm=bool(use_compressed_comm),
        r_global_override=None,
        rank_fraction_default=rank_fraction_default,
        rank_multiple_of_default=rank_multiple_of_default,
        tp_world_size=tp_world_size,
        use_tp_shard=bool(tp_shard_dim in (0, 1) and tp_world_size > 1),
    )

    return dist_meta


def add_non_dion_metas_(
    *,
    param_groups,
    dist_metas_sharded,
    logical_param_name_fn: Callable,
    canonical_param_name_fn: Callable,
    rank_fraction_default: float,
    rank_multiple_of_default: int,
    use_compressed_comm: bool,
) -> None:
    """Add dist_metas for non-Dion parameters."""
    for param_group in param_groups:
        for param in param_group['params']:
            if param in dist_metas_sharded:
                continue

            model_param = getattr(param, '_model_param', None)
            if model_param is None:
                raise RuntimeError(
                    "[Dion] non-Dion optimizer shard is missing _model_param backlink; "
                    f"shape={tuple(param.shape)} id={id(param)}"
                )

            param_name = (
                logical_param_name_fn(model_param)
                or logical_param_name_fn(param)
                or canonical_param_name_fn(model_param)
                or canonical_param_name_fn(param)
                or ""
            )
            param_uid = make_param_uid_(
                param_name=param_name,
                logical_global_shape=None,
                is_dion_param=getattr(model_param, 'is_dion_param', False),
            )
            tp_group = select_tp_group_(model_param)
            tp_world_size, tp_rank = group_info_(tp_group)
            dist_meta = DionDistMeta(
                shape=param.shape,
                global_shape=tuple(model_param.shape) if model_param.ndim == 2 else None,
                tp_shard_dim=-1,
                rank_fraction=rank_fraction_default,
                is_dion_param=getattr(model_param, 'is_dion_param', False),
                is_transposed=bool(param.ndim == 2 and param.shape[0] < param.shape[1]),
                param_uid=param_uid,
                param_name=param_name,
                tp_group=tp_group,
                tp_world_size=tp_world_size,
                tp_rank=tp_rank,
            )
            dist_meta.param_config = build_param_config(
                param_ndim=param.ndim,
                local_shape=tuple(param.shape) if param.ndim == 2 else None,
                dist_meta=dist_meta,
                use_compressed_comm=bool(use_compressed_comm),
                r_global_override=None,
                rank_fraction_default=rank_fraction_default,
                rank_multiple_of_default=rank_multiple_of_default,
                tp_world_size=1,
                use_tp_shard=False,
            )
            dist_metas_sharded[param] = dist_meta
            param._dion_param_uid = param_uid


def build_dist_metas_(
    *,
    shard_bindings_by_param,
    build_dist_meta_fn: Callable,
    add_non_dion_metas_fn: Callable,
    validate_dist_meta_uids_fn: Callable,
):
    """Create distributed metadata for Dion and non-Dion optimizer shards."""
    dist_metas_sharded = {}

    for model_param, shard_pair in shard_bindings_by_param.items():
        _, shard_param = shard_pair
        dist_meta = build_dist_meta_fn(
            model_param=model_param,
            shard_param=shard_param,
        )
        dist_metas_sharded[shard_param] = dist_meta
        shard_param._dion_param_uid = dist_meta.param_uid

    add_non_dion_metas_fn(dist_metas_sharded)
    validate_dist_meta_uids_fn(dist_metas_sharded)
    return dist_metas_sharded


def validate_dist_meta_uids_(
    *,
    dist_metas_sharded,
    canonical_param_name_fn: Callable,
    logger_error_fn: Callable,
) -> None:
    """Fail fast when multiple local shards publish the same logical param uid."""
    uid_to_entries = {}
    for shard_param, dist_meta in dist_metas_sharded.items():
        uid_to_entries.setdefault(dist_meta.param_uid, []).append(
            {
                "name": getattr(dist_meta, "param_name", "")
                or canonical_param_name_fn(shard_param),
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
        logger_error_fn("[Dion] Duplicate param_uid=%s entries=%s", uid, entries)
    raise RuntimeError(
        "[Dion] Duplicate param_uid detected in dist_metas; "
        "optimizer state identity would be ambiguous"
    )
