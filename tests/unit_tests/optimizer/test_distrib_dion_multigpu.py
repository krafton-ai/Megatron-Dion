import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.core.optimizer.dion.types import DionDistMeta, DionParamConfig
from megatron.core.optimizer.dion_distrib_optimizer import (
    _CHILD_GROUP_CACHE,
    DionDistributedOptimizer,
)
from megatron.core.optimizer.distrib_dion.batches import (
    build_batch_collectives,
    resolve_batch_group,
)
from megatron.core.optimizer.distrib_dion.integration import (
    _get_dion_replica_group,
    _resolve_fs_group,
)
from megatron.core.optimizer.distrib_dion.parameter import (
    is_dion_param,
    resolve_grad_rank_to_fs_rank,
)


def _require_multigpu(min_world_size=4):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if "RANK" not in os.environ or "LOCAL_RANK" not in os.environ:
        pytest.skip("requires torchrun")
    if world_size < min_world_size:
        pytest.skip(f"requires torchrun with WORLD_SIZE >= {min_world_size}")
    if torch.cuda.device_count() < local_world_size:
        pytest.skip(
            f"requires at least {local_world_size} visible CUDA devices for local torchrun ranks"
        )


def _init_model_parallel(**kwargs):
    os.environ.pop("NVTE_FLASH_ATTN", None)
    os.environ.pop("NVTE_FUSED_ATTN", None)
    os.environ.pop("NVTE_UNFUSED_ATTN", None)

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank % torch.cuda.device_count())
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timedelta(minutes=5),
        )
    parallel_state.destroy_model_parallel()
    parallel_state.initialize_model_parallel(
        create_gloo_process_groups=False,
        **kwargs,
    )


def _destroy_model_parallel():
    if dist.is_initialized():
        dist.barrier()
    parallel_state.destroy_model_parallel()
    if dist.is_initialized():
        dist.barrier()


def _new_optimizer():
    return object.__new__(DionDistributedOptimizer)


def _all_gather_object(obj):
    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, obj)
    return gathered


def _layout_info(layout, global_rank):
    group, world_size, child_rank, start, end, row_sizes = layout
    if group is None:
        group_ranks = (global_rank,) if child_rank >= 0 else ()
    elif child_rank >= 0:
        group_ranks = tuple(int(rank) for rank in dist.get_process_group_ranks(group))
    else:
        group_ranks = ()
    return {
        "world_size": int(world_size),
        "rank": int(child_rank),
        "start": int(start),
        "end": int(end),
        "row_sizes": tuple(int(size) for size in row_sizes),
        "group_ranks": group_ranks,
    }


def _check_layout(info, *, tp_rank, member_tp_ranks, row_sizes):
    expected_world_size = len(member_tp_ranks)
    assert info["world_size"] == expected_world_size
    assert info["row_sizes"] == tuple(row_sizes)
    if tp_rank not in member_tp_ranks:
        assert info["rank"] == -1
        assert (info["start"], info["end"]) == (-1, -1)
        return

    child_rank = member_tp_ranks.index(tp_rank)
    start = sum(row_sizes[:child_rank])
    end = start + row_sizes[child_rank]
    assert info["rank"] == child_rank
    assert (info["start"], info["end"]) == (start, end)


def _assert_group_sum(group, expected_sum):
    value = torch.tensor(
        [float(dist.get_rank() + 1)],
        device=torch.cuda.current_device(),
        dtype=torch.float32,
    )
    if group is not None:
        dist.all_reduce(value, group=group)
    assert int(value.item()) == int(expected_sum)


def _assert_real_group_collective(group, expected_size=None):
    ranks = tuple(int(rank) for rank in dist.get_process_group_ranks(group))
    if expected_size is not None:
        assert len(ranks) == expected_size
    _assert_group_sum(group, sum(rank + 1 for rank in ranks))
    return ranks


def _assert_dist_optimizer_groups(*, instance_size, intra_dp_size, inter_size):
    _assert_real_group_collective(
        parallel_state.get_intra_distributed_optimizer_instance_group(),
        expected_size=instance_size,
    )
    _assert_real_group_collective(
        parallel_state.get_intra_distributed_optimizer_instance_data_parallel_group(),
        expected_size=intra_dp_size,
    )
    _assert_real_group_collective(
        parallel_state.get_inter_distributed_optimizer_instance_group(),
        expected_size=inter_size,
    )


def _group_ranks(group):
    return tuple(int(rank) for rank in dist.get_process_group_ranks(group))


def _assert_same_group_ranks(first, second):
    assert _group_ranks(first) == _group_ranks(second)


def _assert_group_excludes_cp_peers(group):
    cp_group = parallel_state.get_context_parallel_group(check_initialized=False)
    if cp_group is None:
        return
    cp_ranks = set(_group_ranks(cp_group))
    if len(cp_ranks) <= 1:
        return
    overlap = tuple(rank for rank in _group_ranks(group) if rank in cp_ranks)
    assert overlap == (int(dist.get_rank()),)


def _assert_qkv_tp_layouts(optimizer, meta, split_shapes):
    tp_rank = int(meta.tp_rank)
    tp_ranks = tuple(int(rank) for rank in dist.get_process_group_ranks(meta.tp_group))
    optimizer._init_qkv_child_groups(dist_meta=meta, split_shapes=split_shapes)
    layouts = {
        child: optimizer._resolve_qkv_child_tp_shard_layout(
            parent_dist_meta=meta,
            split_shapes=split_shapes,
            child_kind=child,
            create_group=False,
        )
        for child in ("q", "k", "v")
    }
    report = {
        "global_rank": dist.get_rank(),
        "tp_rank": tp_rank,
        "tp_ranks": tp_ranks,
        "layouts": {
            child: _layout_info(layout, dist.get_rank()) for child, layout in layouts.items()
        },
    }

    reports = _all_gather_object(report)
    for item in reports:
        item_tp_rank = item["tp_rank"]
        item_tp_ranks = tuple(item["tp_ranks"])
        item_layouts = item["layouts"]
        expected = {
            "q": ((0, 2), (4, 4), (item_tp_ranks[0], item_tp_ranks[2])),
            "k": ((0, 1, 2, 3), (2, 2, 2, 2), item_tp_ranks),
            "v": ((1, 3), (4, 4), (item_tp_ranks[1], item_tp_ranks[3])),
        }
        for child, (member_tp_ranks, row_sizes, child_ranks) in expected.items():
            child_info = item_layouts[child]
            _check_layout(
                child_info,
                tp_rank=item_tp_rank,
                member_tp_ranks=member_tp_ranks,
                row_sizes=row_sizes,
            )
            if child_info["rank"] >= 0:
                assert child_info["group_ranks"] == tuple(child_ranks)

    for child, member_tp_ranks in {"q": (0, 2), "k": (0, 1, 2, 3), "v": (1, 3)}.items():
        if tp_rank in member_tp_ranks:
            child_ranks = tuple(tp_ranks[index] for index in member_tp_ranks)
            _assert_group_sum(layouts[child][0], sum(rank + 1 for rank in child_ranks))


def _assert_linear_tp_layouts(optimizer, meta, split_rows):
    tp_rank = int(meta.tp_rank)
    tp_ranks = tuple(int(rank) for rank in dist.get_process_group_ranks(meta.tp_group))
    optimizer._init_linear_child_groups(dist_meta=meta, split_rows=split_rows)
    layouts = {
        child: optimizer._resolve_linear_child_tp_shard_layout(
            parent_dist_meta=meta,
            split_rows=split_rows,
            child_kind=child,
            create_group=False,
        )
        for child in ("gate", "up")
    }
    report = {
        "global_rank": dist.get_rank(),
        "tp_rank": tp_rank,
        "tp_ranks": tp_ranks,
        "layouts": {
            child: _layout_info(layout, dist.get_rank()) for child, layout in layouts.items()
        },
    }

    reports = _all_gather_object(report)
    for item in reports:
        item_tp_rank = item["tp_rank"]
        item_tp_ranks = tuple(item["tp_ranks"])
        item_layouts = item["layouts"]
        expected = {
            "gate": ((0, 1), (4, 4), (item_tp_ranks[0], item_tp_ranks[1])),
            "up": ((2, 3), (4, 4), (item_tp_ranks[2], item_tp_ranks[3])),
        }
        for child, (member_tp_ranks, row_sizes, child_ranks) in expected.items():
            child_info = item_layouts[child]
            _check_layout(
                child_info,
                tp_rank=item_tp_rank,
                member_tp_ranks=member_tp_ranks,
                row_sizes=row_sizes,
            )
            if child_info["rank"] >= 0:
                assert child_info["group_ranks"] == tuple(child_ranks)

    for child, member_tp_ranks in {"gate": (0, 1), "up": (2, 3)}.items():
        if tp_rank in member_tp_ranks:
            child_ranks = tuple(tp_ranks[index] for index in member_tp_ranks)
            _assert_group_sum(layouts[child][0], sum(rank + 1 for rank in child_ranks))


def test_qkv_tp4_split_groups_run_real_nccl_collectives():
    _require_multigpu(4)
    _init_model_parallel(
        tensor_model_parallel_size=4,
        pipeline_model_parallel_size=1,
    )
    _CHILD_GROUP_CACHE.clear()
    try:
        optimizer = _new_optimizer()
        tp_group = parallel_state.get_tensor_model_parallel_group()
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        tp_ranks = tuple(int(rank) for rank in dist.get_process_group_ranks(tp_group))
        meta = DionDistMeta(
            shape=(6, 4),
            local_shape=(6, 4),
            global_shape=(24, 4),
            fs_shard_dim=1,
            fs_world_size=1,
            fs_rank=-1,
            fs_start_idx=0,
            fs_end_idx=4,
            tp_shard_dim=0,
            tp_world_size=4,
            tp_rank=tp_rank,
            tp_group=tp_group,
            is_dion_param=True,
            param_uid=("qkv_tp4",),
            param_name="attention.linear_qkv.weight",
        )

        split_shapes = (4, 4, 4)
        optimizer._init_qkv_child_groups(dist_meta=meta, split_shapes=split_shapes)
        layouts = {
            child: optimizer._resolve_qkv_child_tp_shard_layout(
                parent_dist_meta=meta,
                split_shapes=split_shapes,
                child_kind=child,
                create_group=False,
            )
            for child in ("q", "k", "v")
        }
        report = {
            "global_rank": dist.get_rank(),
            "tp_rank": tp_rank,
            "tp_ranks": tp_ranks,
            "layouts": {
                child: _layout_info(layout, dist.get_rank()) for child, layout in layouts.items()
            },
        }

        reports = _all_gather_object(report)
        for item in reports:
            item_tp_rank = item["tp_rank"]
            item_tp_ranks = tuple(item["tp_ranks"])
            item_layouts = item["layouts"]
            expected = {
                "q": ((0, 2), (4, 4), (item_tp_ranks[0], item_tp_ranks[2])),
                "k": ((0, 1, 2, 3), (2, 2, 2, 2), item_tp_ranks),
                "v": ((1, 3), (4, 4), (item_tp_ranks[1], item_tp_ranks[3])),
            }
            for child, (member_tp_ranks, row_sizes, child_ranks) in expected.items():
                child_info = item_layouts[child]
                _check_layout(
                    child_info,
                    tp_rank=item_tp_rank,
                    member_tp_ranks=member_tp_ranks,
                    row_sizes=row_sizes,
                )
                if child_info["rank"] >= 0:
                    assert child_info["group_ranks"] == tuple(child_ranks)

        for child, member_tp_ranks in {"q": (0, 2), "k": (0, 1, 2, 3), "v": (1, 3)}.items():
            if tp_rank in member_tp_ranks:
                child_ranks = tuple(tp_ranks[index] for index in member_tp_ranks)
                _assert_group_sum(layouts[child][0], sum(rank + 1 for rank in child_ranks))
        dist.barrier()
    finally:
        _CHILD_GROUP_CACHE.clear()
        _destroy_model_parallel()


def test_qkv_tp2_cp2_split_groups_do_not_include_cp_peers():
    _require_multigpu(4)
    _init_model_parallel(
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=1,
        context_parallel_size=2,
    )
    _CHILD_GROUP_CACHE.clear()
    try:
        optimizer = _new_optimizer()
        tp_group = parallel_state.get_tensor_model_parallel_group()
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        tp_ranks = tuple(int(rank) for rank in dist.get_process_group_ranks(tp_group))
        cp_ranks = tuple(int(rank) for rank in parallel_state.get_context_parallel_global_ranks())
        meta = DionDistMeta(
            shape=(6, 4),
            local_shape=(6, 4),
            global_shape=(12, 4),
            fs_shard_dim=1,
            fs_world_size=1,
            fs_rank=-1,
            fs_start_idx=0,
            fs_end_idx=4,
            tp_shard_dim=0,
            tp_world_size=2,
            tp_rank=tp_rank,
            tp_group=tp_group,
            is_dion_param=True,
            param_uid=("qkv_tp2_cp2",),
            param_name="attention.linear_qkv.weight",
        )

        split_shapes = (4, 4, 4)
        optimizer._init_qkv_child_groups(dist_meta=meta, split_shapes=split_shapes)
        layouts = {
            child: optimizer._resolve_qkv_child_tp_shard_layout(
                parent_dist_meta=meta,
                split_shapes=split_shapes,
                child_kind=child,
                create_group=False,
            )
            for child in ("q", "k", "v")
        }
        report = {
            "global_rank": dist.get_rank(),
            "tp_rank": tp_rank,
            "tp_ranks": tp_ranks,
            "cp_ranks": cp_ranks,
            "layouts": {
                child: _layout_info(layout, dist.get_rank()) for child, layout in layouts.items()
            },
        }

        reports = _all_gather_object(report)
        for item in reports:
            item_tp_rank = item["tp_rank"]
            item_tp_ranks = tuple(item["tp_ranks"])
            item_cp_ranks = set(item["cp_ranks"])
            item_layouts = item["layouts"]
            expected = {
                "q": ((0,), (4,), (item_tp_ranks[0],)),
                "k": ((0, 1), (2, 2), item_tp_ranks),
                "v": ((1,), (4,), (item_tp_ranks[1],)),
            }
            for child, (member_tp_ranks, row_sizes, child_ranks) in expected.items():
                child_info = item_layouts[child]
                _check_layout(
                    child_info,
                    tp_rank=item_tp_rank,
                    member_tp_ranks=member_tp_ranks,
                    row_sizes=row_sizes,
                )
                if child_info["rank"] >= 0:
                    assert child_info["group_ranks"] == tuple(child_ranks)
                    assert set(child_info["group_ranks"]).isdisjoint(
                        item_cp_ranks.difference(item_tp_ranks)
                    )

        if tp_rank in (0, 1):
            _assert_group_sum(layouts["k"][0], sum(rank + 1 for rank in tp_ranks))
        dist.barrier()
    finally:
        _CHILD_GROUP_CACHE.clear()
        _destroy_model_parallel()


def test_partial_data_parallel_with_cp_runs_real_nccl_collectives():
    _require_multigpu(4)
    _init_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=2,
        num_distributed_optimizer_instances=2,
    )
    try:
        full_group = parallel_state.get_data_parallel_group(with_context_parallel=True)
        partial_group = parallel_state.get_data_parallel_group(
            with_context_parallel=True,
            partial_data_parallel=True,
        )
        full_ranks = tuple(int(rank) for rank in dist.get_process_group_ranks(full_group))
        partial_ranks = tuple(int(rank) for rank in dist.get_process_group_ranks(partial_group))

        report = {
            "global_rank": dist.get_rank(),
            "full_ranks": full_ranks,
            "partial_ranks": partial_ranks,
            "cp_ranks": tuple(
                int(rank) for rank in parallel_state.get_context_parallel_global_ranks()
            ),
        }
        reports = _all_gather_object(report)
        for item in reports:
            assert len(item["full_ranks"]) % 2 == 0
            assert len(item["partial_ranks"]) == len(item["full_ranks"]) // 2
            assert set(item["partial_ranks"]).issubset(set(item["full_ranks"]))
            assert set(item["cp_ranks"]).issubset(set(item["full_ranks"]))

        _assert_group_sum(partial_group, sum(rank + 1 for rank in partial_ranks))
        _assert_group_sum(full_group, sum(rank + 1 for rank in full_ranks))
        dist.barrier()
    finally:
        _destroy_model_parallel()


def test_32gpu_pp_cp_partial_dp_and_split_children_run_real_nccl_collectives():
    _require_multigpu(32)
    _init_model_parallel(
        tensor_model_parallel_size=4,
        pipeline_model_parallel_size=2,
        context_parallel_size=2,
        num_distributed_optimizer_instances=2,
        order="tp-cp-dp-pp",
    )
    _CHILD_GROUP_CACHE.clear()
    try:
        optimizer = _new_optimizer()
        tp_group = parallel_state.get_tensor_model_parallel_group()
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        meta = DionDistMeta(
            shape=(6, 4),
            local_shape=(6, 4),
            global_shape=(24, 4),
            fs_shard_dim=1,
            fs_world_size=1,
            fs_rank=-1,
            fs_start_idx=0,
            fs_end_idx=4,
            tp_shard_dim=0,
            tp_world_size=4,
            tp_rank=tp_rank,
            tp_group=tp_group,
            is_dion_param=True,
            param_uid=("qkv_32gpu_pp_cp",),
            param_name="attention.linear_qkv.weight",
        )
        linear_meta = DionDistMeta(
            shape=(4, 4),
            local_shape=(4, 4),
            global_shape=(16, 4),
            fs_shard_dim=1,
            fs_world_size=1,
            fs_rank=-1,
            fs_start_idx=0,
            fs_end_idx=4,
            tp_shard_dim=0,
            tp_world_size=4,
            tp_rank=tp_rank,
            tp_group=tp_group,
            is_dion_param=True,
            param_uid=("linear_32gpu_pp_cp",),
            param_name="mlp.linear_fc1.weight",
        )

        assert parallel_state.get_tensor_model_parallel_world_size() == 4
        assert parallel_state.get_pipeline_model_parallel_world_size() == 2
        assert parallel_state.get_context_parallel_world_size() == 2
        _assert_real_group_collective(tp_group, expected_size=4)
        _assert_real_group_collective(
            parallel_state.get_pipeline_model_parallel_group(),
            expected_size=2,
        )
        _assert_real_group_collective(
            parallel_state.get_context_parallel_group(),
            expected_size=2,
        )
        _assert_real_group_collective(
            parallel_state.get_data_parallel_group(with_context_parallel=True),
            expected_size=4,
        )
        _assert_real_group_collective(
            parallel_state.get_data_parallel_group(
                with_context_parallel=True,
                partial_data_parallel=True,
            ),
            expected_size=2,
        )
        _assert_dist_optimizer_groups(instance_size=16, intra_dp_size=1, inter_size=2)

        _assert_qkv_tp_layouts(optimizer, meta, (4, 4, 4))
        _assert_linear_tp_layouts(optimizer, linear_meta, (8, 8))
        dist.barrier()
    finally:
        _CHILD_GROUP_CACHE.clear()
        _destroy_model_parallel()


def test_32gpu_dense_tp_pp_cp_dp_fs_rp_sp_contracts_run_real_nccl_collectives():
    _require_multigpu(32)
    _init_model_parallel(
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=2,
        context_parallel_size=2,
        num_distributed_optimizer_instances=2,
        order="tp-cp-dp-pp",
    )
    try:
        tp_group = parallel_state.get_tensor_model_parallel_group()
        fs_group = parallel_state.get_intra_distributed_optimizer_instance_data_parallel_group()
        rp_group = parallel_state.get_inter_distributed_optimizer_instance_group()
        full_dp_group = parallel_state.get_data_parallel_group(with_context_parallel=True)
        pure_dp_group = parallel_state.get_data_parallel_group(with_context_parallel=False)
        partial_dp_group = parallel_state.get_data_parallel_group(
            with_context_parallel=True,
            partial_data_parallel=True,
        )

        assert parallel_state.get_tensor_model_parallel_world_size() == 2
        assert parallel_state.get_pipeline_model_parallel_world_size() == 2
        assert parallel_state.get_context_parallel_world_size() == 2
        _assert_real_group_collective(tp_group, expected_size=2)
        _assert_real_group_collective(
            parallel_state.get_pipeline_model_parallel_group(),
            expected_size=2,
        )
        _assert_real_group_collective(
            parallel_state.get_context_parallel_group(),
            expected_size=2,
        )
        _assert_real_group_collective(full_dp_group, expected_size=8)
        _assert_real_group_collective(partial_dp_group, expected_size=4)
        _assert_dist_optimizer_groups(instance_size=16, intra_dp_size=2, inter_size=2)
        _assert_real_group_collective(fs_group, expected_size=2)
        _assert_real_group_collective(rp_group, expected_size=2)
        _assert_group_excludes_cp_peers(fs_group)
        _assert_group_excludes_cp_peers(rp_group)

        resolved_fs_group = _resolve_fs_group(
            dense_fs_group=fs_group,
            pure_data_parallel_group=pure_dp_group,
            is_expert_parallel=False,
            requested_fs_world_size=2,
            requested_rp_world_size=2,
        )
        resolved_rp_group = _get_dion_replica_group(
            pg_collection=None,
            pure_dp_group=pure_dp_group,
            requested_fs_world_size=2,
            requested_rp_world_size=2,
            is_expert_parallel=False,
        )
        _assert_same_group_ranks(resolved_fs_group, fs_group)
        _assert_same_group_ranks(resolved_rp_group, rp_group)

        optimizer = _new_optimizer()
        optimizer._pure_data_parallel_group = pure_dp_group
        optimizer.data_parallel_group = full_dp_group
        optimizer._dion_fs_group = fs_group
        optimizer._is_expert_dion = False
        optimizer._fs_size = 2
        optimizer._rp_size = 2
        optimizer._replica_group = rp_group
        optimizer._global_rank = dist.get_rank()
        optimizer._init_groups()
        _assert_same_group_ranks(optimizer.fs_group, fs_group)
        _assert_same_group_ranks(optimizer.rp_group, rp_group)
        assert optimizer.fs_size == 2
        assert optimizer.fs_rank == dist.get_rank(fs_group)

        grad_rank_to_fs_rank = resolve_grad_rank_to_fs_rank(
            grad_group=full_dp_group,
            fs_group=fs_group,
            fs_size=2,
            bucket_id=7,
        )
        assert len(grad_rank_to_fs_rank) == dist.get_world_size(full_dp_group)
        assert set(grad_rank_to_fs_rank) == {0, 1}

        config = DionParamConfig(
            has_fs_shard=True,
            use_fs_shard=True,
            fs_shard_dim=1,
            has_tp_shard=True,
            use_tp_shard=True,
            tp_shard_dim=0,
            use_low_rank_sync=True,
        )
        dist_meta = DionDistMeta(
            shape=(4, 4),
            local_shape=(4, 4),
            global_shape=(8, 8),
            fs_shard_dim=1,
            fs_world_size=2,
            fs_rank=dist.get_rank(fs_group),
            fs_start_idx=0,
            fs_end_idx=4,
            fs_group=fs_group,
            tp_shard_dim=0,
            tp_world_size=2,
            tp_rank=parallel_state.get_tensor_model_parallel_rank(),
            tp_group=tp_group,
            is_dion_param=True,
            param_uid=("dense_all_parallel", dist.get_rank()),
            param_name="layers.0.mlp.linear_fc1.weight",
        )
        batch_group = resolve_batch_group(
            config=config,
            dist_meta=dist_meta,
            use_fs_collectives=True,
            state_replica_group=rp_group,
            replica_validation_group=pure_dp_group,
            group_size=dist.get_world_size,
            get_replicate_group=lambda: rp_group,
            resolve_ortho_group=lambda _config, _dist_meta: tp_group,
        )
        sync_rank_sets = {_group_ranks(group) for group in batch_group.sync_groups}
        assert _group_ranks(fs_group) in sync_rank_sets
        assert _group_ranks(tp_group) in sync_rank_sets
        assert _group_ranks(rp_group) in sync_rank_sets
        assert batch_group.kernel_kind == "fsdp_tp"
        assert batch_group.batch_world_size == 2

        collectives = build_batch_collectives(
            q_tensors=[torch.zeros(4, 2, device=torch.cuda.current_device())],
            configs=[config],
            dist_metas=[dist_meta],
            real_batch_size=1,
            use_fs_collectives=True,
            resolve_tp_group=lambda meta, expect_group: meta.tp_group,
            resolve_fs_group_from_meta=lambda meta, expect_group: meta.fs_group,
        )
        assert len(collectives.tp_q_gathers) == 1
        assert len(collectives.tp_q_reshards) == 1
        assert len(collectives.tp_r_collectives) == 1
        assert len(collectives.fs_p_collectives) == 1
        _assert_same_group_ranks(collectives.tp_q_gathers[0].process_group, tp_group)
        _assert_same_group_ranks(collectives.tp_r_collectives[0].process_group, tp_group)
        _assert_same_group_ranks(collectives.fs_p_collectives[0].process_group, fs_group)

        dense_param = torch.nn.Parameter(torch.empty(2, 2, device=torch.cuda.current_device()))
        dense_param.dion_candidate = True
        assert is_dion_param(dense_param, "layers.0.mlp.linear_fc1.weight")

        sp_param = torch.nn.Parameter(torch.empty(2, 2, device=torch.cuda.current_device()))
        sp_param.dion_candidate = True
        sp_param.sequence_parallel = True
        assert not is_dion_param(sp_param, "layers.0.mlp.linear_fc1.weight")
        dist.barrier()
    finally:
        _destroy_model_parallel()


@pytest.mark.parametrize("rank_order", ["tp-cp-ep-dp-pp", "tp-ep-dp-cp-pp"])
def test_32gpu_pp_cp_ep_partial_groups_run_real_nccl_collectives(rank_order):
    _require_multigpu(32)
    _init_model_parallel(
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=2,
        context_parallel_size=2,
        expert_model_parallel_size=4,
        expert_tensor_parallel_size=2,
        num_distributed_optimizer_instances=2,
        order=rank_order,
    )
    try:
        assert parallel_state.get_tensor_model_parallel_world_size() == 2
        assert parallel_state.get_pipeline_model_parallel_world_size() == 2
        assert parallel_state.get_context_parallel_world_size() == 2
        assert parallel_state.get_expert_model_parallel_world_size() == 4
        assert parallel_state.get_expert_tensor_parallel_world_size() == 2
        assert parallel_state.get_expert_tensor_and_model_parallel_world_size() == 8
        assert parallel_state.get_expert_data_parallel_world_size() == 2
        assert parallel_state.get_expert_data_parallel_world_size(
            partial_expert_data_parallel=True
        ) == 1

        _assert_real_group_collective(
            parallel_state.get_tensor_model_parallel_group(),
            expected_size=2,
        )
        _assert_real_group_collective(
            parallel_state.get_pipeline_model_parallel_group(),
            expected_size=2,
        )
        _assert_real_group_collective(
            parallel_state.get_context_parallel_group(),
            expected_size=2,
        )
        _assert_real_group_collective(
            parallel_state.get_data_parallel_group(with_context_parallel=True),
            expected_size=8,
        )
        _assert_real_group_collective(
            parallel_state.get_data_parallel_group(
                with_context_parallel=True,
                partial_data_parallel=True,
            ),
            expected_size=4,
        )
        _assert_real_group_collective(
            parallel_state.get_expert_model_parallel_group(),
            expected_size=4,
        )
        _assert_real_group_collective(
            parallel_state.get_expert_tensor_parallel_group(),
            expected_size=2,
        )
        _assert_real_group_collective(
            parallel_state.get_expert_tensor_and_model_parallel_group(),
            expected_size=8,
        )
        _assert_real_group_collective(
            parallel_state.get_expert_data_parallel_group(),
            expected_size=2,
        )
        _assert_real_group_collective(
            parallel_state.get_expert_data_parallel_group(partial_expert_data_parallel=True),
            expected_size=1,
        )
        _assert_dist_optimizer_groups(instance_size=16, intra_dp_size=2, inter_size=2)
        dist.barrier()
    finally:
        _destroy_model_parallel()


def test_ep_split_qkv_children_run_real_nccl_collectives():
    _require_multigpu(4)
    _init_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=2,
        expert_tensor_parallel_size=2,
    )
    _CHILD_GROUP_CACHE.clear()
    try:
        optimizer = _new_optimizer()
        expert_tp_group = parallel_state.get_expert_tensor_parallel_group()
        expert_tp_rank = parallel_state.get_expert_tensor_parallel_rank()
        expert_tp_ranks = tuple(int(rank) for rank in dist.get_process_group_ranks(expert_tp_group))
        meta = DionDistMeta(
            shape=(12, 4),
            local_shape=(6, 4),
            global_shape=(24, 4),
            per_expert_global_shape=(12, 4),
            expert_axis=0,
            num_local_experts=2,
            local_expert_index=0,
            fs_shard_dim=1,
            fs_world_size=1,
            fs_rank=-1,
            fs_start_idx=0,
            fs_end_idx=4,
            tp_shard_dim=0,
            tp_world_size=2,
            tp_rank=expert_tp_rank,
            tp_group=expert_tp_group,
            is_dion_param=True,
            param_uid=("expert_qkv_split", dist.get_rank()),
            param_name="decoder.layers.0.mlp.experts.local_experts.0.attention.qkv.weight",
        )

        optimizer._init_qkv_child_groups(dist_meta=meta, split_shapes=(4, 4, 4))
        parent_meta = optimizer._split_parent_dist_meta(meta)
        layouts = {
            child: optimizer._resolve_qkv_child_tp_shard_layout(
                parent_dist_meta=parent_meta,
                split_shapes=(4, 4, 4),
                child_kind=child,
                create_group=False,
            )
            for child in ("q", "k", "v")
        }
        report = {
            "global_rank": dist.get_rank(),
            "expert_tp_rank": expert_tp_rank,
            "expert_tp_ranks": expert_tp_ranks,
            "layouts": {
                child: _layout_info(layout, dist.get_rank()) for child, layout in layouts.items()
            },
        }
        reports = _all_gather_object(report)
        for item in reports:
            item_tp_rank = item["expert_tp_rank"]
            item_tp_ranks = tuple(item["expert_tp_ranks"])
            item_layouts = item["layouts"]
            expected = {
                "q": ((0,), (4,), (item_tp_ranks[0],)),
                "k": ((0, 1), (2, 2), item_tp_ranks),
                "v": ((1,), (4,), (item_tp_ranks[1],)),
            }
            for child, (member_tp_ranks, row_sizes, child_ranks) in expected.items():
                child_info = item_layouts[child]
                _check_layout(
                    child_info,
                    tp_rank=item_tp_rank,
                    member_tp_ranks=member_tp_ranks,
                    row_sizes=row_sizes,
                )
                if child_info["rank"] >= 0:
                    assert child_info["group_ranks"] == tuple(child_ranks)

        if expert_tp_rank in (0, 1):
            _assert_group_sum(layouts["k"][0], sum(rank + 1 for rank in expert_tp_ranks))
        dist.barrier()
    finally:
        _CHILD_GROUP_CACHE.clear()
        _destroy_model_parallel()


def test_ep_split_linear_children_run_real_nccl_collectives():
    _require_multigpu(4)
    _init_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=2,
        expert_tensor_parallel_size=2,
    )
    _CHILD_GROUP_CACHE.clear()
    try:
        optimizer = _new_optimizer()
        expert_tp_group = parallel_state.get_expert_tensor_parallel_group()
        expert_tp_rank = parallel_state.get_expert_tensor_parallel_rank()
        expert_tp_ranks = tuple(int(rank) for rank in dist.get_process_group_ranks(expert_tp_group))
        meta = DionDistMeta(
            shape=(12, 4),
            local_shape=(6, 4),
            global_shape=(24, 4),
            per_expert_global_shape=(12, 4),
            expert_axis=0,
            num_local_experts=2,
            local_expert_index=0,
            fs_shard_dim=1,
            fs_world_size=1,
            fs_rank=-1,
            fs_start_idx=0,
            fs_end_idx=4,
            tp_shard_dim=0,
            tp_world_size=2,
            tp_rank=expert_tp_rank,
            tp_group=expert_tp_group,
            linear_partition_stride=1,
            is_dion_param=True,
            param_uid=("expert_linear_split", dist.get_rank()),
            param_name="decoder.layers.0.mlp.experts.local_experts.0.linear_fc1.weight",
        )

        split_rows = (4, 8)
        optimizer._init_linear_child_groups(dist_meta=meta, split_rows=split_rows)
        parent_meta = optimizer._split_parent_dist_meta(meta)
        layouts = {
            child: optimizer._resolve_linear_child_tp_shard_layout(
                parent_dist_meta=parent_meta,
                split_rows=split_rows,
                child_kind=child,
                create_group=False,
            )
            for child in ("gate", "up")
        }
        report = {
            "global_rank": dist.get_rank(),
            "expert_tp_rank": expert_tp_rank,
            "expert_tp_ranks": expert_tp_ranks,
            "layouts": {
                child: _layout_info(layout, dist.get_rank()) for child, layout in layouts.items()
            },
        }
        reports = _all_gather_object(report)
        for item in reports:
            item_tp_rank = item["expert_tp_rank"]
            item_tp_ranks = tuple(item["expert_tp_ranks"])
            item_layouts = item["layouts"]
            expected = {
                "gate": ((0,), (4,), (item_tp_ranks[0],)),
                "up": ((0, 1), (2, 6), item_tp_ranks),
            }
            for child, (member_tp_ranks, row_sizes, child_ranks) in expected.items():
                child_info = item_layouts[child]
                _check_layout(
                    child_info,
                    tp_rank=item_tp_rank,
                    member_tp_ranks=member_tp_ranks,
                    row_sizes=row_sizes,
                )
                if child_info["rank"] >= 0:
                    assert child_info["group_ranks"] == tuple(child_ranks)

        if expert_tp_rank in (0, 1):
            _assert_group_sum(layouts["up"][0], sum(rank + 1 for rank in expert_tp_ranks))
        dist.barrier()
    finally:
        _CHILD_GROUP_CACHE.clear()
        _destroy_model_parallel()
