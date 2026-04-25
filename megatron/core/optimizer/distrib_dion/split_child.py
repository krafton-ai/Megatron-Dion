"""Split-child metadata helpers for Dion."""

from __future__ import annotations

from dataclasses import replace

from ..dion.state import build_param_config, is_p_fs_sharded, is_p_tp_sharded


def _tensor_row_shard_sizes(fs_row_shard_sizes, tp_row_shard_sizes, *, error_prefix: str, detail: str):
    if fs_row_shard_sizes is not None and tp_row_shard_sizes is not None:
        raise RuntimeError(f"[DION_{error_prefix}_MULTIPLE_ROW_SHARDS] {detail}")
    if fs_row_shard_sizes is not None:
        return tuple(int(size) for size in fs_row_shard_sizes)
    if tp_row_shard_sizes is not None:
        return tuple(int(size) for size in tp_row_shard_sizes)
    return None


def _set_row_layout(
    child_dist_meta,
    *,
    fs_row_shard_sizes,
    fs_row_shard_start_idx: int,
    fs_row_shard_end_idx: int,
    tp_row_shard_sizes,
    tp_row_shard_start_idx: int,
    tp_row_shard_end_idx: int,
) -> None:
    config = child_dist_meta.param_config
    if config is None:
        raise RuntimeError(
            "[DION_SPLIT_CHILD_MISSING_PARAM_CONFIG] "
            f"param_uid={child_dist_meta.param_uid} param_name={child_dist_meta.param_name}"
        )
    tp_active = bool(
        getattr(child_dist_meta, "tp_shard_dim", -1) in (0, 1)
        and int(getattr(child_dist_meta, "tp_world_size", 1)) > 1
    )
    if tp_row_shard_sizes is not None and is_p_tp_sharded(config, tp_active=tp_active):
        child_dist_meta.row_shard_start_idx = int(tp_row_shard_start_idx)
        child_dist_meta.row_shard_end_idx = int(tp_row_shard_end_idx)
        child_dist_meta.row_shard_sizes = tuple(int(size) for size in tp_row_shard_sizes)
        return
    if fs_row_shard_sizes is not None and is_p_fs_sharded(config):
        child_dist_meta.row_shard_start_idx = int(fs_row_shard_start_idx)
        child_dist_meta.row_shard_end_idx = int(fs_row_shard_end_idx)
        child_dist_meta.row_shard_sizes = tuple(int(size) for size in fs_row_shard_sizes)
        return
    child_dist_meta.row_shard_start_idx = -1
    child_dist_meta.row_shard_end_idx = -1
    child_dist_meta.row_shard_sizes = None


def build_split_child_dist_meta(
    *,
    parent_dist_meta,
    child_uid,
    child_name: str,
    child_local_shape: tuple[int, int],
    child_global_shape: tuple[int, int],
    fs_layout,
    tp_layout,
    child_fields: dict,
    error_prefix: str,
    use_low_rank_sync: bool,
    rank_fraction_default: float,
    rank_multiple_of_default: int,
):
    (
        child_fs_group,
        child_fs_world_size,
        child_fs_rank,
        child_fs_start_idx,
        child_fs_end_idx,
        fs_row_shard_sizes,
    ) = fs_layout
    (
        child_tp_group,
        child_tp_world_size,
        child_tp_rank,
        tp_row_shard_start_idx,
        tp_row_shard_end_idx,
        tp_row_shard_sizes,
    ) = tp_layout
    detail = (
        f"param_uid={parent_dist_meta.param_uid} "
        f"param_name={parent_dist_meta.param_name} "
        f"child_name={child_name}"
    )
    tensor_row_shard_sizes = _tensor_row_shard_sizes(
        fs_row_shard_sizes,
        tp_row_shard_sizes,
        error_prefix=error_prefix,
        detail=detail,
    )
    child_dist_meta = replace(
        parent_dist_meta,
        shape=tuple(int(dim) for dim in child_local_shape),
        global_shape=tuple(int(dim) for dim in child_global_shape),
        fs_group=child_fs_group,
        fs_world_size=int(child_fs_world_size),
        fs_rank=int(child_fs_rank),
        fs_start_idx=int(child_fs_start_idx),
        fs_end_idx=int(child_fs_end_idx),
        tp_group=child_tp_group,
        tp_world_size=int(child_tp_world_size),
        tp_rank=int(child_tp_rank),
        param_uid=child_uid,
        param_name=child_name,
        per_expert_global_shape=None,
        local_shape=tuple(int(dim) for dim in child_local_shape),
        tensor_row_shard_sizes=tensor_row_shard_sizes,
        row_shard_start_idx=-1,
        row_shard_end_idx=-1,
        row_shard_sizes=None,
        expert_axis=-1,
        num_local_experts=1,
        local_expert_index=-1,
        parent_param_uid=parent_dist_meta.param_uid,
        parent_param_name=parent_dist_meta.param_name,
        param_config=None,
        **child_fields,
    )
    child_dist_meta.param_config = build_param_config(
        param_ndim=2,
        local_shape=child_local_shape,
        dist_meta=child_dist_meta,
        use_low_rank_sync=bool(use_low_rank_sync),
        r_global_override=None,
        rank_fraction_default=rank_fraction_default,
        rank_multiple_of_default=rank_multiple_of_default,
        tp_world_size=int(getattr(child_dist_meta, "tp_world_size", 1)),
        tp_active=bool(
            getattr(child_dist_meta, "tp_shard_dim", -1) in (0, 1)
            and int(getattr(child_dist_meta, "tp_world_size", 1)) > 1
        ),
    )
    _set_row_layout(
        child_dist_meta,
        fs_row_shard_sizes=fs_row_shard_sizes,
        fs_row_shard_start_idx=child_fs_start_idx,
        fs_row_shard_end_idx=child_fs_end_idx,
        tp_row_shard_sizes=tp_row_shard_sizes,
        tp_row_shard_start_idx=tp_row_shard_start_idx,
        tp_row_shard_end_idx=tp_row_shard_end_idx,
    )
    return child_dist_meta
