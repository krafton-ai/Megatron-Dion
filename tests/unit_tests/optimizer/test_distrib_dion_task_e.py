from collections import OrderedDict
from types import SimpleNamespace

import pytest
import torch

from megatron.core.optimizer import _get_param_groups
from megatron.core.optimizer.dion.linear import (
    get_linear_split_rows,
    linear_child_has_local_overlap,
    linear_child_local_shape,
    linear_state_key,
    read_linear_child,
    write_linear_child_,
)
from megatron.core.optimizer.dion.qkv import (
    extract_qkv_child,
    qkv_child_has_local_overlap,
    qkv_child_local_shape,
    qkv_child_row_range,
    qkv_state_key,
    scatter_qkv_child_,
)
from megatron.core.optimizer.dion.types import DionBatchGroup, DionDistMeta, DionQInit, DionQLayout
from megatron.core.optimizer.dion_distrib_optimizer import (
    _CHILD_GROUP_CACHE,
    DionDistributedOptimizer,
    _ensure_child_group,
)
from megatron.core.optimizer.distrib_dion.batches import (
    _batch_group_key,
    _batch_key_for_sync,
    _sync_group_batch_metadata,
    build_batch_key,
)
from megatron.core.optimizer.distrib_dion import integration as dion_integration
from megatron.core.optimizer.distrib_dion.parameter import (
    is_combined_grouped_mlp_param,
    is_dion_param,
    mark_dion_bucket_params,
    resolve_grad_rank_to_fs_rank,
)
from megatron.core.optimizer.distrib_dion.dist_meta import build_param_dist_meta, select_tp_group
from megatron.core.optimizer.distrib_dion.sharding import DionShardLayout


class FakeGroup:
    def __init__(self, ranks):
        self.ranks = tuple(int(rank) for rank in ranks)


def make_optimizer_stub():
    optimizer = object.__new__(DionDistributedOptimizer)
    optimizer.optimizer = SimpleNamespace(
        use_low_rank_sync=True,
        _mixed_precision_config=None,
        defaults={"rank_fraction": 0.25, "rank_multiple_of": 1},
    )
    return optimizer


def split_range(size, world_size, rank):
    size_per_rank = int(size) // int(world_size)
    remainder = int(size) % int(world_size)
    if rank < remainder:
        start = rank * (size_per_rank + 1)
        return start, start + size_per_rank + 1
    start = remainder * (size_per_rank + 1) + (rank - remainder) * size_per_rank
    return start, start + size_per_rank


def expected_qkv_child(global_tensor, parent_start, parent_end, split_shapes, child_kind):
    child_index = {"q": 0, "k": 1, "v": 2}[child_kind]
    total = sum(split_shapes)
    child_rows = split_shapes[child_index]
    child_offset = sum(split_shapes[:child_index])
    rows = []
    for row in range(parent_start, parent_end):
        group = row // total
        row_in_group = row % total
        if child_offset <= row_in_group < child_offset + child_rows:
            rows.append(group * child_rows + row_in_group - child_offset)
    child_global = torch.cat(
        [
            global_tensor[group_start + child_offset : group_start + child_offset + child_rows]
            for group_start in range(0, int(global_tensor.size(0)), total)
        ],
        dim=0,
    )
    if not rows:
        return None
    return child_global[rows]


def expected_linear_child(global_tensor, parent_start, parent_end, split_rows, child_kind):
    child_start = 0 if child_kind == "gate" else int(split_rows[0])
    child_end = child_start + (int(split_rows[0]) if child_kind == "gate" else int(split_rows[1]))
    rows = []
    for row in range(parent_start, parent_end):
        if child_start <= row < child_end:
            rows.append(row - child_start)
    child_global = global_tensor[child_start:child_end]
    if not rows:
        return None
    return child_global[rows]


def test_resolve_grad_rank_to_fs_rank_accepts_cp_strided_order(monkeypatch):
    grad_group = FakeGroup((10, 20, 11, 21, 12, 22, 13, 23))
    fs_group = FakeGroup((10, 11, 12, 13))
    monkeypatch.setattr(
        "torch.distributed.get_process_group_ranks",
        lambda group: group.ranks,
    )

    assert resolve_grad_rank_to_fs_rank(
        grad_group=grad_group,
        fs_group=fs_group,
        fs_size=4,
        bucket_id=7,
    ) == (0, 0, 1, 1, 2, 2, 3, 3)


def test_resolve_grad_rank_to_fs_rank_accepts_contiguous_repeated_order(monkeypatch):
    grad_group = FakeGroup((10, 11, 12, 13, 20, 21, 22, 23))
    fs_group = FakeGroup((20, 21, 22, 23))
    monkeypatch.setattr(
        "torch.distributed.get_process_group_ranks",
        lambda group: group.ranks,
    )

    assert resolve_grad_rank_to_fs_rank(
        grad_group=grad_group,
        fs_group=fs_group,
        fs_size=4,
        bucket_id=8,
    ) == (0, 1, 2, 3, 0, 1, 2, 3)


def test_resolve_grad_rank_to_fs_rank_accepts_cp_interleaved_fs_order(monkeypatch):
    grad_group = FakeGroup((1, 3, 5, 7, 9, 11, 13, 15))
    fs_group = FakeGroup((3, 7))
    monkeypatch.setattr(
        "torch.distributed.get_process_group_ranks",
        lambda group: group.ranks,
    )

    assert resolve_grad_rank_to_fs_rank(
        grad_group=grad_group,
        fs_group=fs_group,
        fs_size=2,
        bucket_id=10,
    ) == (0, 0, 1, 1, 0, 0, 1, 1)


def test_resolve_grad_rank_to_fs_rank_rejects_unproved_order(monkeypatch):
    grad_group = FakeGroup((0, 1, 2, 3, 4, 5))
    fs_group = FakeGroup((0, 3, 4))
    monkeypatch.setattr(
        "torch.distributed.get_process_group_ranks",
        lambda group: group.ranks,
    )

    with pytest.raises(RuntimeError, match="cannot prove FS rank order"):
        resolve_grad_rank_to_fs_rank(
            grad_group=grad_group,
            fs_group=fs_group,
            fs_size=3,
            bucket_id=9,
        )


def test_combined_grouped_mlp_params_are_not_dion_candidates():
    param = torch.nn.Parameter(torch.empty(4, 8))
    param.dion_candidate = True
    param.num_local_experts = 2

    assert is_combined_grouped_mlp_param(param, "decoder.layers.0.mlp.experts.weight1")
    assert not is_dion_param(param, "decoder.layers.0.mlp.experts.weight1")
    assert not is_combined_grouped_mlp_param(
        param,
        "decoder.layers.0.mlp.experts.linear_fc1.weight1",
    )


def test_per_expert_named_grouped_linear_keeps_full_local_shape():
    param = torch.nn.Parameter(torch.empty(4, 8))
    param.dion_candidate = True
    param.num_local_experts = 2
    param.tensor_model_parallel = False
    param._param_name = "decoder.layers.0.mlp.experts.linear_fc1.weight0"

    count, info_by_param = mark_dion_bucket_params(
        None,
        OrderedDict([(param, object())]),
        {param: param._param_name},
        fs_size=1,
    )

    assert count == 1
    assert param.is_dion_param
    assert info_by_param[param]["global_shape"] == (4, 8)
    assert info_by_param[param]["per_expert_global_shape"] is None


def test_dense_tp_group_must_exclude_context_parallel_peers(monkeypatch):
    param = torch.nn.Parameter(torch.empty(4, 8))
    param.tensor_model_parallel = True
    param.partition_dim = 0
    dense_tp_group = FakeGroup((0, 1))
    cp_group = FakeGroup((0, 1))

    monkeypatch.setattr(
        "megatron.core.optimizer.distrib_dion.dist_meta.parallel_state.get_tensor_model_parallel_group",
        lambda check_initialized=False: dense_tp_group,
    )
    monkeypatch.setattr(
        "megatron.core.optimizer.distrib_dion.dist_meta.parallel_state.get_context_parallel_group",
        lambda check_initialized=False: cp_group,
    )
    monkeypatch.setattr(
        "megatron.core.optimizer.distrib_dion.dist_meta.dist.get_process_group_ranks",
        lambda group: group.ranks,
    )
    monkeypatch.setattr(
        "megatron.core.optimizer.distrib_dion.dist_meta.dist.get_rank",
        lambda: 0,
    )

    with pytest.raises(RuntimeError, match="dense TP group must exclude context-parallel peers"):
        select_tp_group(param)


def test_allreduce_false_tp_param_uses_expert_tp_group(monkeypatch):
    param = torch.nn.Parameter(torch.empty(4, 8))
    param.tensor_model_parallel = True
    param.partition_dim = 0
    param.allreduce = False
    dense_tp_group = FakeGroup((0, 1, 2, 3))
    expert_tp_group = FakeGroup((0, 2))

    monkeypatch.setattr(
        "megatron.core.optimizer.distrib_dion.dist_meta.parallel_state.get_tensor_model_parallel_group",
        lambda check_initialized=False: dense_tp_group,
    )
    monkeypatch.setattr(
        "megatron.core.optimizer.distrib_dion.dist_meta.parallel_state.get_expert_tensor_parallel_group",
        lambda check_initialized=False: expert_tp_group,
    )
    monkeypatch.setattr(
        "megatron.core.optimizer.distrib_dion.dist_meta.parallel_state.get_context_parallel_group",
        lambda check_initialized=False: None,
    )

    assert select_tp_group(param) is expert_tp_group


def test_replica_group_uses_process_group_collection_inter_dist_opt(monkeypatch):
    pure_dp_group = FakeGroup((0, 1, 2, 3))
    replica_group = FakeGroup((0, 2))
    pg_collection = SimpleNamespace(inter_dist_opt=replica_group)

    monkeypatch.setattr(dion_integration.dist, "get_world_size", lambda group: len(group.ranks))
    monkeypatch.setattr(
        dion_integration.dist,
        "get_process_group_ranks",
        lambda group: group.ranks,
    )
    monkeypatch.setattr(dion_integration.dist, "is_initialized", lambda: False)

    group = dion_integration._get_dion_replica_group(
        pg_collection,
        pure_dp_group,
        requested_fs_world_size=2,
        requested_rp_world_size=2,
        is_expert_parallel=False,
    )

    assert group is replica_group


def test_dense_fs_group_uses_authoritative_cp_excluded_group(monkeypatch):
    pure_dp_group = FakeGroup((0, 1, 2, 3))
    fs_group = FakeGroup((0, 2))

    monkeypatch.setattr(
        dion_integration.parallel_state,
        "get_intra_distributed_optimizer_instance_data_parallel_group",
        lambda check_initialized=False: (_ for _ in ()).throw(AssertionError("unused fallback")),
    )
    monkeypatch.setattr(
        dion_integration.dist,
        "get_process_group_ranks",
        lambda group: group.ranks,
    )
    monkeypatch.setattr(dion_integration.dist, "is_initialized", lambda: False)

    group = dion_integration._resolve_fs_group(
        dense_fs_group=fs_group,
        pure_data_parallel_group=pure_dp_group,
        is_expert_parallel=False,
        requested_fs_world_size=2,
        requested_rp_world_size=1,
    )

    assert group is fs_group


def test_dion_split_qkv_rejects_attention_output_gate():
    param = torch.nn.Parameter(torch.empty(16, 8))

    class ModelChunk:
        config = SimpleNamespace(
            attention_output_gate=True,
            num_attention_heads=8,
            num_query_groups=2,
            kv_channels=4,
            ffn_hidden_size=16,
        )

        def named_parameters(self):
            yield "decoder.layers.0.self_attention.linear_qkv.weight", param

    config = SimpleNamespace(
        dion_split_qkv=True,
        lr=1.0,
        min_lr=0.0,
    )

    with pytest.raises(RuntimeError, match="DION_QKV_SPLIT_OUTPUT_GATE_UNSUPPORTED"):
        _get_param_groups([ModelChunk()], config=config, config_overrides={})


def test_dion_split_qkv_tags_qwen_style_grouped_qkv_shapes(monkeypatch):
    param = torch.nn.Parameter(torch.empty(40, 16))

    class ModelChunk:
        config = SimpleNamespace(
            attention_output_gate=False,
            num_attention_heads=8,
            num_query_groups=1,
            kv_channels=4,
            ffn_hidden_size=16,
        )

        def named_parameters(self):
            yield "decoder.layers.0.self_attention.linear_qkv.weight", param

    monkeypatch.setattr("torch.distributed.get_world_size", lambda group=None: 1)

    def fake_all_gather_object(gathered, local_object, group=None):
        gathered[0] = local_object

    monkeypatch.setattr("torch.distributed.all_gather_object", fake_all_gather_object)

    config = SimpleNamespace(
        dion_split_qkv=True,
        lr=1.0,
        min_lr=0.0,
    )

    param_groups = _get_param_groups([ModelChunk()], config=config, config_overrides={})

    assert param.is_qkv
    assert param.qkv_split_shapes == (32, 4, 4)
    assert any(param_in_group is param for group in param_groups for param_in_group in group["params"])


def test_tp_late_reduction_params_are_not_dion_candidates():
    param = torch.nn.Parameter(torch.empty(4, 8))
    param.dion_candidate = True
    param.sequence_parallel = True
    assert not is_dion_param(param, "router.weight")

    param.sequence_parallel = False
    param.average_gradients_across_tp_domain = True
    assert not is_dion_param(param, "hf_adapter.weight")


def test_qkv_child_overlap_skips_non_owner_fs_rank():
    dist_meta = SimpleNamespace(
        fs_shard_dim=0,
        fs_world_size=8,
        fs_rank=-1,
        global_shape=(12, 4),
        fs_start_idx=-1,
        fs_end_idx=-1,
        param_uid=("qkv",),
        param_name="attention.qkv",
    )

    assert not qkv_child_has_local_overlap((2, 1, 1), dist_meta, "q")


def test_qkv_child_split_handles_fs_shard_inside_qkv_group():
    split_shapes = (2, 1, 1)
    dist_meta = SimpleNamespace(
        fs_shard_dim=0,
        fs_world_size=5,
        fs_rank=1,
        global_shape=(12, 2),
        fs_start_idx=3,
        fs_end_idx=6,
        shape=(3, 2),
        local_shape=(3, 2),
        param_uid=("qkv",),
        param_name="attention.qkv",
    )
    parent = torch.tensor(
        [
            [30.0, 31.0],
            [40.0, 41.0],
            [50.0, 51.0],
        ]
    )

    assert qkv_child_row_range(
        parent_row_start=3,
        parent_row_end=6,
        split_shapes=split_shapes,
        child_kind="q",
    ) == (2, 4)
    assert qkv_child_row_range(
        parent_row_start=3,
        parent_row_end=6,
        split_shapes=split_shapes,
        child_kind="k",
    ) is None
    assert qkv_child_has_local_overlap(split_shapes, dist_meta, "q")
    assert not qkv_child_has_local_overlap(split_shapes, dist_meta, "k")
    assert qkv_child_local_shape((3, 2), split_shapes, "q", dist_meta=dist_meta) == (2, 2)

    q_child = extract_qkv_child(parent, split_shapes, "q", dist_meta=dist_meta)
    assert torch.equal(q_child, parent[1:3])

    updated_parent = parent.clone()
    scatter_qkv_child_(
        updated_parent,
        torch.full((2, 2), -7.0),
        split_shapes,
        "q",
        dist_meta=dist_meta,
    )
    assert torch.equal(updated_parent[0], parent[0])
    assert torch.equal(updated_parent[1:3], torch.full((2, 2), -7.0))


def test_qkv_child_split_handles_tp_shard_crossing_qkv_boundary():
    split_shapes = (2, 2, 2)
    rank0_meta = SimpleNamespace(
        tp_shard_dim=0,
        tp_world_size=2,
        tp_rank=0,
        fs_shard_dim=1,
        fs_world_size=1,
        global_shape=(6, 2),
        shape=(3, 2),
        local_shape=(3, 2),
        param_uid=("qkv",),
        param_name="attention.qkv",
    )
    rank1_meta = SimpleNamespace(
        tp_shard_dim=0,
        tp_world_size=2,
        tp_rank=1,
        fs_shard_dim=1,
        fs_world_size=1,
        global_shape=(6, 2),
        shape=(3, 2),
        local_shape=(3, 2),
        param_uid=("qkv",),
        param_name="attention.qkv",
    )
    rank0_parent = torch.tensor(
        [
            [0.0, 1.0],
            [2.0, 3.0],
            [4.0, 5.0],
        ]
    )
    rank1_parent = torch.tensor(
        [
            [6.0, 7.0],
            [8.0, 9.0],
            [10.0, 11.0],
        ]
    )

    assert qkv_child_has_local_overlap(split_shapes, rank0_meta, "q")
    assert qkv_child_has_local_overlap(split_shapes, rank0_meta, "k")
    assert not qkv_child_has_local_overlap(split_shapes, rank0_meta, "v")
    assert not qkv_child_has_local_overlap(split_shapes, rank1_meta, "q")
    assert qkv_child_has_local_overlap(split_shapes, rank1_meta, "k")
    assert qkv_child_has_local_overlap(split_shapes, rank1_meta, "v")

    assert qkv_child_local_shape((3, 2), split_shapes, "q", dist_meta=rank0_meta) == (2, 2)
    assert qkv_child_local_shape((3, 2), split_shapes, "k", dist_meta=rank0_meta) == (1, 2)
    assert qkv_child_local_shape((3, 2), split_shapes, "k", dist_meta=rank1_meta) == (1, 2)
    assert qkv_child_local_shape((3, 2), split_shapes, "v", dist_meta=rank1_meta) == (2, 2)

    assert torch.equal(
        extract_qkv_child(rank0_parent, split_shapes, "q", dist_meta=rank0_meta),
        rank0_parent[0:2],
    )
    assert torch.equal(
        extract_qkv_child(rank0_parent, split_shapes, "k", dist_meta=rank0_meta),
        rank0_parent[2:3],
    )
    assert torch.equal(
        extract_qkv_child(rank1_parent, split_shapes, "k", dist_meta=rank1_meta),
        rank1_parent[0:1],
    )
    assert torch.equal(
        extract_qkv_child(rank1_parent, split_shapes, "v", dist_meta=rank1_meta),
        rank1_parent[1:3],
    )

    updated_rank0 = rank0_parent.clone()
    updated_rank1 = rank1_parent.clone()
    scatter_qkv_child_(
        updated_rank0,
        torch.full((1, 2), -3.0),
        split_shapes,
        "k",
        dist_meta=rank0_meta,
    )
    scatter_qkv_child_(
        updated_rank1,
        torch.full((1, 2), -5.0),
        split_shapes,
        "k",
        dist_meta=rank1_meta,
    )
    assert torch.equal(updated_rank0[0:2], rank0_parent[0:2])
    assert torch.equal(updated_rank0[2:3], torch.full((1, 2), -3.0))
    assert torch.equal(updated_rank1[0:1], torch.full((1, 2), -5.0))
    assert torch.equal(updated_rank1[1:3], rank1_parent[1:3])


def test_qkv_child_split_handles_qwen_style_tp_shard_crossing_kv_boundary():
    split_shapes = (32, 4, 4)
    dist_meta = SimpleNamespace(
        tp_shard_dim=0,
        tp_world_size=4,
        tp_rank=1,
        fs_shard_dim=1,
        fs_world_size=1,
        global_shape=(80, 2),
        shape=(20, 2),
        local_shape=(20, 2),
        param_uid=("qwen_gqa_qkv",),
        param_name="attention.linear_qkv.weight",
    )
    global_parent = torch.arange(80 * 2, dtype=torch.float32).view(80, 2)
    parent = global_parent[20:40].clone()

    assert qkv_child_row_range(
        parent_row_start=20,
        parent_row_end=40,
        split_shapes=split_shapes,
        child_kind="q",
    ) == (20, 32)
    assert qkv_child_row_range(
        parent_row_start=20,
        parent_row_end=40,
        split_shapes=split_shapes,
        child_kind="k",
    ) == (0, 4)
    assert qkv_child_row_range(
        parent_row_start=20,
        parent_row_end=40,
        split_shapes=split_shapes,
        child_kind="v",
    ) == (0, 4)

    assert qkv_child_has_local_overlap(split_shapes, dist_meta, "q")
    assert qkv_child_has_local_overlap(split_shapes, dist_meta, "k")
    assert qkv_child_has_local_overlap(split_shapes, dist_meta, "v")
    assert qkv_child_local_shape((20, 2), split_shapes, "q", dist_meta=dist_meta) == (12, 2)
    assert qkv_child_local_shape((20, 2), split_shapes, "k", dist_meta=dist_meta) == (4, 2)
    assert qkv_child_local_shape((20, 2), split_shapes, "v", dist_meta=dist_meta) == (4, 2)

    for child_kind in ("q", "k", "v"):
        assert torch.equal(
            extract_qkv_child(parent, split_shapes, child_kind, dist_meta=dist_meta),
            expected_qkv_child(global_parent, 20, 40, split_shapes, child_kind),
        )

    updated_parent = parent.clone()
    scatter_qkv_child_(
        updated_parent,
        torch.full((12, 2), -3.0),
        split_shapes,
        "q",
        dist_meta=dist_meta,
    )
    scatter_qkv_child_(
        updated_parent,
        torch.full((4, 2), -5.0),
        split_shapes,
        "k",
        dist_meta=dist_meta,
    )
    scatter_qkv_child_(
        updated_parent,
        torch.full((4, 2), -7.0),
        split_shapes,
        "v",
        dist_meta=dist_meta,
    )
    assert torch.equal(updated_parent[:12], torch.full((12, 2), -3.0))
    assert torch.equal(updated_parent[12:16], torch.full((4, 2), -5.0))
    assert torch.equal(updated_parent[16:20], torch.full((4, 2), -7.0))


def test_qkv_child_tp_layout_compacts_owner_subgroup(monkeypatch):
    _CHILD_GROUP_CACHE.clear()
    created = []
    parent_tp_group = FakeGroup((0, 1, 2, 3))

    def fake_create_group(ranks, *, use_local_synchronization, group_desc):
        created.append((tuple(ranks), use_local_synchronization, group_desc))
        return FakeGroup(ranks)

    monkeypatch.setattr(
        "torch.distributed.get_process_group_ranks",
        lambda process_group: process_group.ranks,
    )
    monkeypatch.setattr(
        "megatron.core.parallel_state.create_group",
        fake_create_group,
    )

    optimizer = make_optimizer_stub()
    parent_rank3_meta = SimpleNamespace(
        tp_shard_dim=0,
        tp_world_size=4,
        tp_rank=3,
        tp_group=parent_tp_group,
        fs_shard_dim=1,
        fs_world_size=1,
        global_shape=(18, 2),
        param_uid=("qkv",),
        param_name="attention.qkv",
    )

    group, world_size, rank, start, end, row_sizes = (
        optimizer._resolve_qkv_child_tp_shard_layout(
            parent_dist_meta=parent_rank3_meta,
            split_shapes=(2, 2, 2),
            child_kind="q",
            create_group=True,
        )
    )

    assert group.ranks == (0, 1, 2)
    assert world_size == 3
    assert rank == -1
    assert (start, end) == (-1, -1)
    assert row_sizes == (2, 2, 2)
    assert created == [((0, 1, 2), True, "DION_SPLIT_CHILD_GROUP")]
    _CHILD_GROUP_CACHE.clear()


def test_qkv_split_child_groups_use_per_expert_layout(monkeypatch):
    parent_tp_group = FakeGroup((0, 1))
    monkeypatch.setattr(
        "torch.distributed.get_process_group_ranks",
        lambda process_group: process_group.ranks,
    )
    optimizer = make_optimizer_stub()
    dist_meta = DionDistMeta(
        shape=(16, 8),
        local_shape=(8, 8),
        global_shape=(16, 8),
        per_expert_global_shape=(8, 8),
        expert_axis=0,
        num_local_experts=2,
        local_expert_index=1,
        fs_shard_dim=1,
        fs_world_size=1,
        fs_rank=-1,
        fs_start_idx=0,
        fs_end_idx=8,
        tp_shard_dim=0,
        tp_world_size=2,
        tp_rank=1,
        tp_group=parent_tp_group,
        is_dion_param=True,
        param_uid=("qkv",),
        param_name="decoder.layers.0.mlp.experts.local_experts.1.attention.qkv.weight",
    )

    optimizer._init_qkv_child_groups(
        dist_meta=dist_meta,
        split_shapes=(2, 1, 1),
    )
    child_meta = optimizer._build_qkv_child_dist_meta(
        parent_dist_meta=optimizer._split_parent_dist_meta(dist_meta),
        child_kind="q",
        split_shapes=(2, 1, 1),
        child_local_shape=(2, 8),
        child_global_shape=(4, 8),
    )

    assert child_meta.global_shape == (4, 8)
    assert child_meta.per_expert_global_shape is None
    assert child_meta.num_local_experts == 1
    assert child_meta.local_expert_index == -1
    assert ("expert", 1) in child_meta.param_uid


def test_expand_split_qkv_expert_param_routes_and_commits_selected_expert():
    optimizer = make_optimizer_stub()
    optimizer.optimizer.defaults["split_qkv"] = True
    optimizer.optimizer.defaults["split_linear"] = False
    parent = torch.nn.Parameter(torch.arange(16 * 4, dtype=torch.float32).view(16, 4))
    grad = torch.arange(1000, 1000 + 16 * 4, dtype=torch.float32).view(16, 4)
    momentum = torch.arange(2000, 2000 + 16 * 4, dtype=torch.float32).view(16, 4)
    split_shapes = (2, 1, 1)
    dist_meta = DionDistMeta(
        shape=(16, 4),
        local_shape=(8, 4),
        global_shape=(16, 4),
        per_expert_global_shape=(8, 4),
        expert_axis=0,
        num_local_experts=2,
        local_expert_index=1,
        fs_shard_dim=1,
        fs_world_size=1,
        fs_rank=-1,
        fs_start_idx=0,
        fs_end_idx=4,
        tp_shard_dim=-1,
        tp_world_size=1,
        is_dion_param=True,
        param_uid=("qkv_parent",),
        param_name="decoder.layers.0.mlp.experts.local_experts.1.attention.qkv.weight",
        qkv_split_shapes=split_shapes,
    )
    state = {
        "momentum": momentum.clone(),
        "qkv_split_qkv": True,
        "qkv_split_shapes": split_shapes,
    }
    for child_kind, rows in {"q": 4, "k": 2, "v": 2}.items():
        state[qkv_state_key("Q", child_kind)] = torch.zeros(4, 1)
        state[qkv_state_key("r", child_kind)] = 1
        state[qkv_state_key("local_shape", child_kind)] = (rows, 4)
        state[qkv_state_key("global_shape", child_kind)] = (rows, 4)

    children = optimizer._expand_split_qkv_params(
        param=parent,
        grad=grad,
        optimizer_state=state,
        optim_group={"algorithm": "dion"},
        config=None,
        dist_meta=dist_meta,
    )

    assert [child.dist_meta.qkv_child_kind for child in children] == ["q", "k", "v"]
    expert_param = parent.detach()[8:16]
    expert_grad = grad[8:16]
    expert_momentum = state["momentum"][8:16]
    for child in children:
        child_kind = child.dist_meta.qkv_child_kind
        assert torch.equal(
            child.param,
            expected_qkv_child(expert_param, 0, 8, split_shapes, child_kind),
        )
        assert torch.equal(
            child.grad,
            expected_qkv_child(expert_grad, 0, 8, split_shapes, child_kind),
        )
        assert torch.equal(
            child.optimizer_state["momentum"],
            expected_qkv_child(expert_momentum, 0, 8, split_shapes, child_kind),
        )
    assert all(child.dist_meta.per_expert_global_shape is None for child in children)
    assert all(("expert", 1) in child.dist_meta.param_uid for child in children)

    before_parent = parent.detach().clone()
    before_momentum = state["momentum"].clone()
    with torch.no_grad():
        for child in children:
            child.commit_update(
                torch.full_like(child.param, -7.0),
                torch.full_like(child.optimizer_state["momentum"], -9.0),
            )

    assert torch.equal(parent.detach()[:8], before_parent[:8])
    assert torch.equal(state["momentum"][:8], before_momentum[:8])
    for child_kind, rows in {"q": 4, "k": 2, "v": 2}.items():
        assert torch.equal(
            extract_qkv_child(parent.detach()[8:16], split_shapes, child_kind),
            torch.full((rows, 4), -7.0),
        )
        assert torch.equal(
            extract_qkv_child(state["momentum"][8:16], split_shapes, child_kind),
            torch.full((rows, 4), -9.0),
        )


def test_expand_split_qkv_column_packed_expert_routes_and_commits_selected_expert():
    optimizer = make_optimizer_stub()
    optimizer.optimizer.defaults["split_qkv"] = True
    optimizer.optimizer.defaults["split_linear"] = False
    parent = torch.nn.Parameter(torch.arange(8 * 8, dtype=torch.float32).view(8, 8))
    grad = torch.arange(1000, 1000 + 8 * 8, dtype=torch.float32).view(8, 8)
    momentum = torch.arange(2000, 2000 + 8 * 8, dtype=torch.float32).view(8, 8)
    split_shapes = (2, 1, 1)
    dist_meta = DionDistMeta(
        shape=(8, 8),
        local_shape=(8, 4),
        global_shape=(8, 8),
        per_expert_global_shape=(8, 4),
        expert_axis=1,
        num_local_experts=2,
        local_expert_index=1,
        fs_shard_dim=-1,
        fs_world_size=1,
        fs_rank=-1,
        fs_start_idx=-1,
        fs_end_idx=-1,
        tp_shard_dim=-1,
        tp_world_size=1,
        is_dion_param=True,
        param_uid=("qkv_column_expert",),
        param_name="decoder.layers.0.mlp.experts.local_experts.1.attention.qkv.weight",
        qkv_split_shapes=split_shapes,
    )
    state = {
        "momentum": momentum.clone(),
        "qkv_split_qkv": True,
        "qkv_split_shapes": split_shapes,
    }
    for child_kind, rows in {"q": 4, "k": 2, "v": 2}.items():
        state[qkv_state_key("Q", child_kind)] = torch.zeros(4, 1)
        state[qkv_state_key("r", child_kind)] = 1
        state[qkv_state_key("local_shape", child_kind)] = (rows, 4)
        state[qkv_state_key("global_shape", child_kind)] = (rows, 4)

    children = optimizer._expand_split_qkv_params(
        param=parent,
        grad=grad,
        optimizer_state=state,
        optim_group={"algorithm": "dion"},
        config=None,
        dist_meta=dist_meta,
    )

    expert_param = parent.detach()[:, 4:8]
    expert_grad = grad[:, 4:8]
    expert_momentum = state["momentum"][:, 4:8]
    for child in children:
        child_kind = child.dist_meta.qkv_child_kind
        assert torch.equal(
            child.param,
            expected_qkv_child(expert_param, 0, 8, split_shapes, child_kind),
        )
        assert torch.equal(
            child.grad,
            expected_qkv_child(expert_grad, 0, 8, split_shapes, child_kind),
        )
        assert torch.equal(
            child.optimizer_state["momentum"],
            expected_qkv_child(expert_momentum, 0, 8, split_shapes, child_kind),
        )
    assert all(child.dist_meta.global_shape[1] == 4 for child in children)
    assert all(child.dist_meta.per_expert_global_shape is None for child in children)

    before_parent = parent.detach().clone()
    before_momentum = state["momentum"].clone()
    with torch.no_grad():
        for child in children:
            child.commit_update(
                torch.full_like(child.param, -11.0),
                torch.full_like(child.optimizer_state["momentum"], -13.0),
            )

    assert torch.equal(parent.detach()[:, :4], before_parent[:, :4])
    assert torch.equal(state["momentum"][:, :4], before_momentum[:, :4])
    for child_kind, rows in {"q": 4, "k": 2, "v": 2}.items():
        assert torch.equal(
            extract_qkv_child(parent.detach()[:, 4:8], split_shapes, child_kind),
            torch.full((rows, 4), -11.0),
        )
        assert torch.equal(
            extract_qkv_child(state["momentum"][:, 4:8], split_shapes, child_kind),
            torch.full((rows, 4), -13.0),
        )


def test_expand_split_qkv_expert_initializes_child_q_from_child_meta(monkeypatch):
    optimizer = make_optimizer_stub()
    optimizer.optimizer.defaults["split_qkv"] = True
    optimizer.optimizer.defaults["split_linear"] = False
    parent = torch.nn.Parameter(torch.arange(16 * 4, dtype=torch.float32).view(16, 4))
    grad = torch.ones_like(parent)
    momentum = torch.zeros_like(parent)
    split_shapes = (2, 1, 1)
    dist_meta = DionDistMeta(
        shape=(16, 4),
        local_shape=(8, 4),
        global_shape=(16, 4),
        per_expert_global_shape=(8, 4),
        expert_axis=0,
        num_local_experts=2,
        local_expert_index=1,
        fs_shard_dim=-1,
        fs_world_size=1,
        fs_rank=-1,
        fs_start_idx=-1,
        fs_end_idx=-1,
        tp_shard_dim=-1,
        tp_world_size=1,
        is_dion_param=True,
        param_uid=("qkv_parent",),
        param_name="decoder.layers.0.mlp.experts.local_experts.1.attention.qkv.weight",
        qkv_split_shapes=split_shapes,
    )
    state = {
        "momentum": momentum.clone(),
        "qkv_split_qkv": True,
        "qkv_split_shapes": split_shapes,
    }
    captured = []

    def fake_build_q_init(*, dist_meta, **kwargs):
        captured.append((dist_meta.qkv_child_kind, dist_meta.local_shape, dist_meta.global_shape))
        q_local_shape = (int(dist_meta.local_shape[0]), 1)
        q_global_shape = (int(dist_meta.global_shape[0]), 1)
        return DionQInit(
            tp_world_size=1,
            tp_rank=0,
            q_layout=DionQLayout(
                q_global_shape=q_global_shape,
                q_local_shape=q_local_shape,
                q_gathered_shape=q_global_shape,
                r_global=1,
                r_local=1,
            ),
            broadcast_q=lambda q_state: None,
        )

    def fake_init_q_state(*, q_layout, **kwargs):
        return torch.full(tuple(int(dim) for dim in q_layout.q_local_shape), 5.0)

    monkeypatch.setattr(
        "megatron.core.optimizer.dion_distrib_optimizer.build_q_init",
        fake_build_q_init,
    )
    monkeypatch.setattr(
        "megatron.core.optimizer.dion_distrib_optimizer.init_q_state",
        fake_init_q_state,
    )
    monkeypatch.setattr(
        "megatron.core.optimizer.dion_distrib_optimizer.resolve_base_training_seed",
        lambda: 1234,
    )

    children = optimizer._expand_split_qkv_params(
        param=parent,
        grad=grad,
        optimizer_state=state,
        optim_group={"algorithm": "dion"},
        config=None,
        dist_meta=dist_meta,
    )

    assert captured == [
        ("q", (4, 4), (4, 4)),
        ("k", (2, 4), (2, 4)),
        ("v", (2, 4), (2, 4)),
    ]
    for child in children:
        child_kind = child.dist_meta.qkv_child_kind
        assert state[qkv_state_key("Q", child_kind)].shape == (
            child.dist_meta.local_shape[0],
            1,
        )
        assert state[qkv_state_key("local_shape", child_kind)] == child.dist_meta.local_shape
        assert state[qkv_state_key("global_shape", child_kind)] == child.dist_meta.global_shape


def test_expand_split_linear_expert_param_routes_and_commits_selected_expert():
    optimizer = make_optimizer_stub()
    optimizer.optimizer.defaults["split_qkv"] = False
    optimizer.optimizer.defaults["split_linear"] = True
    parent = torch.nn.Parameter(torch.arange(12 * 4, dtype=torch.float32).view(12, 4))
    grad = torch.arange(1000, 1000 + 12 * 4, dtype=torch.float32).view(12, 4)
    momentum = torch.arange(2000, 2000 + 12 * 4, dtype=torch.float32).view(12, 4)
    dist_meta = DionDistMeta(
        shape=(12, 4),
        local_shape=(6, 4),
        global_shape=(12, 4),
        per_expert_global_shape=(6, 4),
        expert_axis=0,
        num_local_experts=2,
        local_expert_index=1,
        fs_shard_dim=1,
        fs_world_size=1,
        fs_rank=-1,
        fs_start_idx=0,
        fs_end_idx=4,
        tp_shard_dim=-1,
        tp_world_size=1,
        is_dion_param=True,
        param_uid=("linear_parent",),
        param_name="decoder.layers.0.mlp.experts.local_experts.1.linear_fc1.weight",
        linear_split_rows=(6, 6),
    )
    state = {
        "momentum": momentum.clone(),
        "linear_split_linear": True,
        "linear_split_rows": (6, 6),
    }
    for child_kind in ("gate", "up"):
        state[linear_state_key("Q", child_kind)] = torch.zeros(4, 1)
        state[linear_state_key("r", child_kind)] = 1
        state[linear_state_key("local_shape", child_kind)] = (3, 4)
        state[linear_state_key("global_shape", child_kind)] = (3, 4)

    children = optimizer._expand_split_linear_params(
        param=parent,
        grad=grad,
        optimizer_state=state,
        optim_group={"algorithm": "dion"},
        config=None,
        dist_meta=dist_meta,
    )

    assert [child.dist_meta.linear_child_kind for child in children] == ["gate", "up"]
    expert_param = parent.detach()[6:12]
    expert_grad = grad[6:12]
    expert_momentum = state["momentum"][6:12]
    assert torch.equal(children[0].param, expert_param[:3])
    assert torch.equal(children[0].grad, expert_grad[:3])
    assert torch.equal(children[0].optimizer_state["momentum"], expert_momentum[:3])
    assert torch.equal(children[1].param, expert_param[3:6])
    assert torch.equal(children[1].grad, expert_grad[3:6])
    assert torch.equal(children[1].optimizer_state["momentum"], expert_momentum[3:6])
    assert all(child.dist_meta.global_shape == (3, 4) for child in children)
    assert all(child.dist_meta.per_expert_global_shape is None for child in children)

    before_parent = parent.detach().clone()
    before_momentum = state["momentum"].clone()
    with torch.no_grad():
        children[0].commit_update(
            torch.full_like(children[0].param, -3.0),
            torch.full_like(children[0].optimizer_state["momentum"], -5.0),
        )
        children[1].commit_update(
            torch.full_like(children[1].param, -4.0),
            torch.full_like(children[1].optimizer_state["momentum"], -6.0),
        )

    assert torch.equal(parent.detach()[:6], before_parent[:6])
    assert torch.equal(state["momentum"][:6], before_momentum[:6])
    assert torch.equal(parent.detach()[6:9], torch.full((3, 4), -3.0))
    assert torch.equal(parent.detach()[9:12], torch.full((3, 4), -4.0))
    assert torch.equal(state["momentum"][6:9], torch.full((3, 4), -5.0))
    assert torch.equal(state["momentum"][9:12], torch.full((3, 4), -6.0))


def test_build_param_dist_meta_normalizes_expert_linear_split_rows():
    model_param = torch.nn.Parameter(torch.empty(12, 4))
    model_param.is_linear_fc1 = True
    model_param.linear_split_rows = (6, 6)
    model_param.num_local_experts = 2
    param_name = "decoder.layers.0.mlp.experts.local_experts.1.linear_fc1.weight"
    shard_layout = DionShardLayout(
        local_shape=(12, 4),
        global_shape=(12, 4),
        fs_shard_dim=-1,
        start_idx=-1,
        end_idx=-1,
        per_expert_global_shape=(6, 4),
    )

    dist_meta = build_param_dist_meta(
        model_param=model_param,
        shard_param=model_param,
        fs_group=None,
        shard_layouts_by_param={model_param: shard_layout},
        get_param_name=lambda param: param_name if param is model_param else "",
        rank_fraction_default=0.25,
        rank_multiple_of_default=1,
        use_low_rank_sync=True,
    )

    assert dist_meta.linear_split_rows == (3, 3)
    assert dist_meta.local_shape == (6, 4)
    assert dist_meta.per_expert_global_shape == (6, 4)
    assert dist_meta.local_expert_index == 1


def test_expand_split_linear_column_packed_expert_routes_and_commits_selected_expert():
    optimizer = make_optimizer_stub()
    optimizer.optimizer.defaults["split_qkv"] = False
    optimizer.optimizer.defaults["split_linear"] = True
    parent = torch.nn.Parameter(torch.arange(8 * 8, dtype=torch.float32).view(8, 8))
    grad = torch.arange(1000, 1000 + 8 * 8, dtype=torch.float32).view(8, 8)
    momentum = torch.arange(2000, 2000 + 8 * 8, dtype=torch.float32).view(8, 8)
    dist_meta = DionDistMeta(
        shape=(8, 8),
        local_shape=(8, 4),
        global_shape=(8, 8),
        per_expert_global_shape=(8, 4),
        expert_axis=1,
        num_local_experts=2,
        local_expert_index=1,
        fs_shard_dim=-1,
        fs_world_size=1,
        fs_rank=-1,
        fs_start_idx=-1,
        fs_end_idx=-1,
        tp_shard_dim=-1,
        tp_world_size=1,
        is_dion_param=True,
        param_uid=("linear_column_expert",),
        param_name="decoder.layers.0.mlp.experts.local_experts.1.linear_fc1.weight",
        linear_split_rows=(4, 4),
    )
    state = {
        "momentum": momentum.clone(),
        "linear_split_linear": True,
        "linear_split_rows": (4, 4),
    }
    for child_kind in ("gate", "up"):
        state[linear_state_key("Q", child_kind)] = torch.zeros(4, 1)
        state[linear_state_key("r", child_kind)] = 1
        state[linear_state_key("local_shape", child_kind)] = (4, 4)
        state[linear_state_key("global_shape", child_kind)] = (4, 4)

    children = optimizer._expand_split_linear_params(
        param=parent,
        grad=grad,
        optimizer_state=state,
        optim_group={"algorithm": "dion"},
        config=None,
        dist_meta=dist_meta,
    )

    expert_param = parent.detach()[:, 4:8]
    expert_grad = grad[:, 4:8]
    expert_momentum = state["momentum"][:, 4:8]
    assert torch.equal(children[0].param, expert_param[:4])
    assert torch.equal(children[0].grad, expert_grad[:4])
    assert torch.equal(children[0].optimizer_state["momentum"], expert_momentum[:4])
    assert torch.equal(children[1].param, expert_param[4:8])
    assert torch.equal(children[1].grad, expert_grad[4:8])
    assert torch.equal(children[1].optimizer_state["momentum"], expert_momentum[4:8])
    assert all(child.dist_meta.global_shape == (4, 4) for child in children)
    assert all(child.dist_meta.per_expert_global_shape is None for child in children)

    before_parent = parent.detach().clone()
    before_momentum = state["momentum"].clone()
    with torch.no_grad():
        children[0].commit_update(
            torch.full_like(children[0].param, -17.0),
            torch.full_like(children[0].optimizer_state["momentum"], -19.0),
        )
        children[1].commit_update(
            torch.full_like(children[1].param, -23.0),
            torch.full_like(children[1].optimizer_state["momentum"], -29.0),
        )

    assert torch.equal(parent.detach()[:, :4], before_parent[:, :4])
    assert torch.equal(state["momentum"][:, :4], before_momentum[:, :4])
    assert torch.equal(parent.detach()[:4, 4:8], torch.full((4, 4), -17.0))
    assert torch.equal(parent.detach()[4:8, 4:8], torch.full((4, 4), -23.0))
    assert torch.equal(state["momentum"][:4, 4:8], torch.full((4, 4), -19.0))
    assert torch.equal(state["momentum"][4:8, 4:8], torch.full((4, 4), -29.0))


def test_expand_split_linear_expert_initializes_child_q_from_child_meta(monkeypatch):
    optimizer = make_optimizer_stub()
    optimizer.optimizer.defaults["split_qkv"] = False
    optimizer.optimizer.defaults["split_linear"] = True
    parent = torch.nn.Parameter(torch.arange(12 * 4, dtype=torch.float32).view(12, 4))
    grad = torch.ones_like(parent)
    momentum = torch.zeros_like(parent)
    dist_meta = DionDistMeta(
        shape=(12, 4),
        local_shape=(6, 4),
        global_shape=(12, 4),
        per_expert_global_shape=(6, 4),
        expert_axis=0,
        num_local_experts=2,
        local_expert_index=1,
        fs_shard_dim=-1,
        fs_world_size=1,
        fs_rank=-1,
        fs_start_idx=-1,
        fs_end_idx=-1,
        tp_shard_dim=-1,
        tp_world_size=1,
        is_dion_param=True,
        param_uid=("linear_parent",),
        param_name="decoder.layers.0.mlp.experts.local_experts.1.linear_fc1.weight",
        linear_split_rows=(6, 6),
    )
    state = {
        "momentum": momentum.clone(),
        "linear_split_linear": True,
        "linear_split_rows": (6, 6),
    }
    captured = []

    def fake_build_q_init(*, dist_meta, **kwargs):
        captured.append((dist_meta.linear_child_kind, dist_meta.local_shape, dist_meta.global_shape))
        q_local_shape = (int(dist_meta.local_shape[0]), 1)
        q_global_shape = (int(dist_meta.global_shape[0]), 1)
        return DionQInit(
            tp_world_size=1,
            tp_rank=0,
            q_layout=DionQLayout(
                q_global_shape=q_global_shape,
                q_local_shape=q_local_shape,
                q_gathered_shape=q_global_shape,
                r_global=1,
                r_local=1,
            ),
            broadcast_q=lambda q_state: None,
        )

    def fake_init_q_state(*, q_layout, **kwargs):
        return torch.full(tuple(int(dim) for dim in q_layout.q_local_shape), 7.0)

    monkeypatch.setattr(
        "megatron.core.optimizer.dion_distrib_optimizer.build_q_init",
        fake_build_q_init,
    )
    monkeypatch.setattr(
        "megatron.core.optimizer.dion_distrib_optimizer.init_q_state",
        fake_init_q_state,
    )
    monkeypatch.setattr(
        "megatron.core.optimizer.dion_distrib_optimizer.resolve_base_training_seed",
        lambda: 1234,
    )

    children = optimizer._expand_split_linear_params(
        param=parent,
        grad=grad,
        optimizer_state=state,
        optim_group={"algorithm": "dion"},
        config=None,
        dist_meta=dist_meta,
    )

    assert captured == [
        ("gate", (3, 4), (3, 4)),
        ("up", (3, 4), (3, 4)),
    ]
    for child in children:
        child_kind = child.dist_meta.linear_child_kind
        assert state[linear_state_key("Q", child_kind)].shape == (
            child.dist_meta.local_shape[0],
            1,
        )
        assert state[linear_state_key("local_shape", child_kind)] == child.dist_meta.local_shape
        assert state[linear_state_key("global_shape", child_kind)] == child.dist_meta.global_shape


def test_linear_child_tp_layout_records_noncanonical_row_sizes(monkeypatch):
    _CHILD_GROUP_CACHE.clear()
    parent_tp_group = FakeGroup((0, 1, 2, 3))
    created = []

    def fake_create_group(ranks, *, use_local_synchronization, group_desc):
        created.append((tuple(ranks), use_local_synchronization, group_desc))
        return FakeGroup(ranks)

    monkeypatch.setattr(
        "torch.distributed.get_process_group_ranks",
        lambda process_group: process_group.ranks,
    )
    monkeypatch.setattr(
        "megatron.core.parallel_state.create_group",
        fake_create_group,
    )

    optimizer = object.__new__(DionDistributedOptimizer)
    parent_rank1_meta = SimpleNamespace(
        tp_shard_dim=0,
        tp_world_size=4,
        tp_rank=1,
        tp_group=parent_tp_group,
        fs_shard_dim=1,
        fs_world_size=1,
        global_shape=(12, 2),
        param_uid=("linear",),
        param_name="mlp.linear_fc1",
    )

    group, world_size, rank, start, end, row_sizes = (
        optimizer._resolve_linear_child_tp_shard_layout(
            parent_dist_meta=parent_rank1_meta,
            split_rows=(5, 7),
            child_kind="up",
            create_group=True,
        )
    )

    assert group.ranks == (1, 2, 3)
    assert world_size == 3
    assert rank == 0
    assert (start, end) == (0, 1)
    assert row_sizes == (1, 3, 3)
    assert created == [((1, 2, 3), True, "DION_SPLIT_CHILD_GROUP")]
    _CHILD_GROUP_CACHE.clear()


def test_qkv_child_tp_row_layout_sets_ortho_row_sizes(monkeypatch):
    parent_tp_group = FakeGroup((0, 1, 2))
    monkeypatch.setattr(
        "torch.distributed.get_process_group_ranks",
        lambda process_group: process_group.ranks,
    )
    optimizer = make_optimizer_stub()
    parent_meta = DionDistMeta(
        shape=(6, 4),
        global_shape=(18, 4),
        fs_shard_dim=1,
        fs_world_size=1,
        fs_rank=-1,
        fs_start_idx=0,
        fs_end_idx=4,
        tp_shard_dim=0,
        tp_world_size=3,
        tp_rank=0,
        tp_group=parent_tp_group,
        is_dion_param=True,
        param_uid=("qkv",),
        param_name="attention.qkv",
    )

    child_meta = optimizer._build_qkv_child_dist_meta(
        parent_dist_meta=parent_meta,
        child_kind="q",
        split_shapes=(2, 2, 2),
        child_local_shape=(2, 4),
        child_global_shape=(6, 4),
    )

    assert child_meta.tp_group is parent_tp_group
    assert child_meta.tp_world_size == 3
    assert child_meta.tp_rank == 0
    assert child_meta.row_shard_sizes == (2, 2, 2)
    assert (child_meta.row_shard_start_idx, child_meta.row_shard_end_idx) == (0, 2)


def test_qkv_child_fs_row_tp_column_layout_does_not_set_ortho_row_sizes(monkeypatch):
    parent_fs_group = FakeGroup((0, 1, 2))
    parent_tp_group = FakeGroup((0, 3))
    monkeypatch.setattr(
        "torch.distributed.get_process_group_ranks",
        lambda process_group: process_group.ranks,
    )
    optimizer = make_optimizer_stub()
    parent_meta = DionDistMeta(
        shape=(6, 4),
        global_shape=(18, 8),
        fs_shard_dim=0,
        fs_world_size=3,
        fs_rank=0,
        fs_start_idx=0,
        fs_end_idx=6,
        fs_group=parent_fs_group,
        tp_shard_dim=1,
        tp_world_size=2,
        tp_rank=0,
        tp_group=parent_tp_group,
        is_dion_param=True,
        param_uid=("qkv",),
        param_name="attention.qkv",
    )

    child_meta = optimizer._build_qkv_child_dist_meta(
        parent_dist_meta=parent_meta,
        child_kind="q",
        split_shapes=(2, 2, 2),
        child_local_shape=(2, 4),
        child_global_shape=(6, 8),
    )

    assert child_meta.fs_group is parent_fs_group
    assert child_meta.fs_world_size == 3
    assert child_meta.fs_rank == 0
    assert (child_meta.fs_start_idx, child_meta.fs_end_idx) == (0, 2)
    assert child_meta.tp_group is parent_tp_group
    assert child_meta.tp_world_size == 2
    assert child_meta.row_shard_sizes is None
    assert (child_meta.row_shard_start_idx, child_meta.row_shard_end_idx) == (-1, -1)


def test_linear_child_fs_row_tp_column_layout_does_not_set_ortho_row_sizes(monkeypatch):
    _CHILD_GROUP_CACHE.clear()
    parent_fs_group = FakeGroup((0, 1, 2, 3))
    parent_tp_group = FakeGroup((1, 5))
    created = []

    def fake_create_group(ranks, *, use_local_synchronization, group_desc):
        created.append((tuple(ranks), use_local_synchronization, group_desc))
        return FakeGroup(ranks)

    monkeypatch.setattr(
        "torch.distributed.get_process_group_ranks",
        lambda process_group: process_group.ranks,
    )
    monkeypatch.setattr(
        "megatron.core.parallel_state.create_group",
        fake_create_group,
    )
    optimizer = make_optimizer_stub()
    parent_meta = DionDistMeta(
        shape=(3, 4),
        global_shape=(12, 8),
        fs_shard_dim=0,
        fs_world_size=4,
        fs_rank=1,
        fs_start_idx=3,
        fs_end_idx=6,
        fs_group=parent_fs_group,
        tp_shard_dim=1,
        tp_world_size=2,
        tp_rank=0,
        tp_group=parent_tp_group,
        is_dion_param=True,
        param_uid=("linear",),
        param_name="mlp.linear_fc1",
    )
    optimizer._resolve_linear_child_fs_shard_layout(
        parent_dist_meta=parent_meta,
        split_rows=(5, 7),
        child_kind="up",
        create_group=True,
    )

    child_meta = optimizer._build_linear_child_dist_meta(
        parent_dist_meta=parent_meta,
        child_kind="up",
        split_rows=(5, 7),
        child_local_shape=(1, 4),
        child_global_shape=(7, 8),
    )

    assert child_meta.fs_group.ranks == (1, 2, 3)
    assert child_meta.fs_world_size == 3
    assert child_meta.fs_rank == 0
    assert (child_meta.fs_start_idx, child_meta.fs_end_idx) == (0, 1)
    assert child_meta.tp_group is parent_tp_group
    assert child_meta.row_shard_sizes is None
    assert (child_meta.row_shard_start_idx, child_meta.row_shard_end_idx) == (-1, -1)
    assert created == [((1, 2, 3), True, "DION_SPLIT_CHILD_GROUP")]
    _CHILD_GROUP_CACHE.clear()


def test_linear_child_split_handles_tp_shard_crossing_gate_up_boundary():
    split_rows = (3, 5)
    dist_meta = SimpleNamespace(
        tp_shard_dim=0,
        tp_world_size=3,
        tp_rank=1,
        fs_shard_dim=1,
        fs_world_size=1,
        global_shape=(8, 2),
        shape=(3, 2),
        local_shape=(3, 2),
        param_uid=("linear",),
        param_name="mlp.linear_fc1",
    )
    parent = torch.tensor(
        [
            [30.0, 31.0],
            [40.0, 41.0],
            [50.0, 51.0],
        ]
    )

    assert not linear_child_has_local_overlap(split_rows, dist_meta, "gate")
    assert linear_child_has_local_overlap(split_rows, dist_meta, "up")
    assert linear_child_local_shape((3, 2), split_rows, dist_meta, "up") == (3, 2)
    assert torch.equal(read_linear_child(parent, split_rows, dist_meta, "up"), parent)

    updated_parent = parent.clone()
    write_linear_child_(
        updated_parent,
        torch.full((3, 2), 9.0),
        split_rows,
        dist_meta,
        "up",
    )
    assert torch.equal(updated_parent, torch.full((3, 2), 9.0))


def test_linear_child_split_handles_strided_tp_layout():
    split_rows = (4, 4)
    dist_meta = SimpleNamespace(
        tp_shard_dim=0,
        tp_world_size=2,
        tp_rank=1,
        linear_partition_stride=2,
        fs_shard_dim=1,
        fs_world_size=1,
        global_shape=(8, 2),
        shape=(4, 2),
        local_shape=(4, 2),
        param_uid=("linear",),
        param_name="mlp.linear_fc1",
    )
    parent = torch.tensor(
        [
            [20.0, 21.0],
            [30.0, 31.0],
            [60.0, 61.0],
            [70.0, 71.0],
        ]
    )

    assert linear_child_has_local_overlap(split_rows, dist_meta, "gate")
    assert linear_child_has_local_overlap(split_rows, dist_meta, "up")
    assert linear_child_local_shape((4, 2), split_rows, dist_meta, "gate") == (2, 2)
    assert linear_child_local_shape((4, 2), split_rows, dist_meta, "up") == (2, 2)
    assert torch.equal(read_linear_child(parent, split_rows, dist_meta, "gate"), parent[:2])
    assert torch.equal(read_linear_child(parent, split_rows, dist_meta, "up"), parent[2:])

    updated_parent = parent.clone()
    write_linear_child_(
        updated_parent,
        torch.full((2, 2), -1.0),
        split_rows,
        dist_meta,
        "gate",
    )
    write_linear_child_(
        updated_parent,
        torch.full((2, 2), -2.0),
        split_rows,
        dist_meta,
        "up",
    )
    assert torch.equal(updated_parent[:2], torch.full((2, 2), -1.0))
    assert torch.equal(updated_parent[2:], torch.full((2, 2), -2.0))


def test_linear_child_tp_layout_uses_partition_stride(monkeypatch):
    parent_tp_group = FakeGroup((10, 11))
    monkeypatch.setattr(
        "torch.distributed.get_process_group_ranks",
        lambda process_group: process_group.ranks,
    )
    optimizer = object.__new__(DionDistributedOptimizer)
    parent_meta = SimpleNamespace(
        tp_shard_dim=0,
        tp_world_size=2,
        tp_rank=0,
        tp_group=parent_tp_group,
        linear_partition_stride=2,
        fs_shard_dim=1,
        fs_world_size=1,
        global_shape=(8, 2),
        param_uid=("linear",),
        param_name="mlp.linear_fc1",
    )

    group, world_size, rank, start, end, row_sizes = (
        optimizer._resolve_linear_child_tp_shard_layout(
            parent_dist_meta=parent_meta,
            split_rows=(5, 3),
            child_kind="gate",
            create_group=True,
        )
    )

    assert group is parent_tp_group
    assert world_size == 2
    assert rank == 0
    assert (start, end) == (0, 3)
    assert row_sizes == (3, 2)

    group, world_size, rank, start, end, row_sizes = (
        optimizer._resolve_linear_child_tp_shard_layout(
            parent_dist_meta=parent_meta,
            split_rows=(5, 3),
            child_kind="up",
            create_group=True,
        )
    )

    assert group is parent_tp_group
    assert world_size == 2
    assert rank == 0
    assert (start, end) == (0, 2)
    assert row_sizes == (2, 1)


def test_qkv_fs_row_tensor_layout_separates_batch_keys(monkeypatch):
    parent_fs_group = FakeGroup((0, 1, 2))
    monkeypatch.setattr(
        "torch.distributed.get_process_group_ranks",
        lambda process_group: process_group.ranks,
    )
    optimizer = make_optimizer_stub()
    parent_meta = DionDistMeta(
        shape=(5, 4),
        global_shape=(16, 4),
        fs_shard_dim=0,
        fs_world_size=3,
        fs_rank=1,
        fs_start_idx=6,
        fs_end_idx=11,
        fs_group=parent_fs_group,
        tp_shard_dim=1,
        tp_world_size=1,
        is_dion_param=True,
        param_uid=("qkv",),
        param_name="attention.qkv",
    )

    k_meta = optimizer._build_qkv_child_dist_meta(
        parent_dist_meta=parent_meta,
        child_kind="k",
        split_shapes=(2, 1, 1),
        child_local_shape=(2, 4),
        child_global_shape=(4, 4),
    )
    v_meta = optimizer._build_qkv_child_dist_meta(
        parent_dist_meta=parent_meta,
        child_kind="v",
        split_shapes=(2, 1, 1),
        child_local_shape=(1, 4),
        child_global_shape=(4, 4),
    )

    assert k_meta.tensor_row_shard_sizes == (1, 2, 1)
    assert v_meta.tensor_row_shard_sizes == (1, 1, 2)
    assert k_meta.row_shard_sizes is None
    assert v_meta.row_shard_sizes is None

    k_key = build_batch_key(
        k_meta.local_shape,
        k_meta.param_config,
        torch.float32,
        global_shape=k_meta.global_shape,
        tensor_row_shard_sizes=k_meta.tensor_row_shard_sizes,
        row_shard_sizes=k_meta.row_shard_sizes,
    )
    v_key = build_batch_key(
        v_meta.local_shape,
        v_meta.param_config,
        torch.float32,
        global_shape=v_meta.global_shape,
        tensor_row_shard_sizes=v_meta.tensor_row_shard_sizes,
        row_shard_sizes=v_meta.row_shard_sizes,
    )

    assert k_key != v_key


def test_linear_split_rows_come_from_param_metadata():
    param = torch.nn.Parameter(torch.empty(6, 2))
    param.is_linear_fc1 = True
    param.linear_split_rows = (2, 4)

    assert get_linear_split_rows(param, global_rows=6) == (2, 4)

    with pytest.raises(RuntimeError, match="DION_LINEAR_SPLIT_ROWS_MISMATCH"):
        get_linear_split_rows(param, global_rows=8)

    missing = torch.nn.Parameter(torch.empty(6, 2))
    missing.is_linear_fc1 = True
    with pytest.raises(RuntimeError, match="DION_LINEAR_SPLIT_PARAM_MISSING_ROWS"):
        get_linear_split_rows(missing, global_rows=6)


def test_split_child_group_uses_mcore_group_creation(monkeypatch):
    _CHILD_GROUP_CACHE.clear()
    created = []

    def fake_create_group(ranks, *, use_local_synchronization, group_desc):
        created.append((tuple(ranks), use_local_synchronization, group_desc))
        return FakeGroup(ranks)

    monkeypatch.setattr(
        "megatron.core.parallel_state.create_group",
        fake_create_group,
    )

    group = _ensure_child_group((0, 2, 4), create_group=True)
    cached_group = _ensure_child_group((0, 2, 4), create_group=True)

    assert group is cached_group
    assert created == [((0, 2, 4), True, "DION_SPLIT_CHILD_GROUP")]
    _CHILD_GROUP_CACHE.clear()


def test_split_child_group_must_be_prepared():
    _CHILD_GROUP_CACHE.clear()
    with pytest.raises(RuntimeError, match="DION_SPLIT_CHILD_GROUP_NOT_PREPARED"):
        _ensure_child_group((0, 2, 4), create_group=False)


def test_batch_group_key_separates_collective_domains(monkeypatch):
    first_group = FakeGroup((0, 1))
    second_group = FakeGroup((0, 2))
    monkeypatch.setattr(
        "torch.distributed.get_process_group_ranks",
        lambda group: group.ranks,
    )

    first_key = _batch_group_key(
        DionBatchGroup(
            kernel_kind="fsdp",
            q_norm_group=first_group,
            sync_groups=(first_group,),
            batch_world_size=2,
        )
    )
    second_key = _batch_group_key(
        DionBatchGroup(
            kernel_kind="fsdp",
            q_norm_group=second_group,
            sync_groups=(second_group,),
            batch_world_size=2,
        )
    )

    assert first_key != second_key


def test_process_group_rank_cache_preserves_distinct_batch_group_keys(monkeypatch):
    first_group = FakeGroup((0, 1))
    second_group = FakeGroup((0, 2))
    rank_cache = {}
    calls = []

    def fake_get_process_group_ranks(group):
        calls.append(group)
        return group.ranks

    monkeypatch.setattr(
        "torch.distributed.get_process_group_ranks",
        fake_get_process_group_ranks,
    )

    first_key = _batch_group_key(
        DionBatchGroup(
            kernel_kind="fsdp",
            q_norm_group=first_group,
            sync_groups=(first_group,),
            batch_world_size=2,
        ),
        rank_cache,
    )
    second_key = _batch_group_key(
        DionBatchGroup(
            kernel_kind="fsdp",
            q_norm_group=second_group,
            sync_groups=(second_group,),
            batch_world_size=2,
        ),
        rank_cache,
    )
    repeated_first_key = _batch_group_key(
        DionBatchGroup(
            kernel_kind="fsdp",
            q_norm_group=first_group,
            sync_groups=(first_group,),
            batch_world_size=2,
        ),
        rank_cache,
    )

    assert first_key != second_key
    assert repeated_first_key == first_key
    assert calls == [first_group, second_group]


def test_batch_metadata_cache_skips_repeated_object_gather(monkeypatch):
    group = FakeGroup((0, 1))
    batch_key = (
        ("shape", 1),
        ("optim",),
        ("ddp", 2, group.ranks, (), (), (), (group.ranks,)),
    )
    batch_group = DionBatchGroup(params=[object()])
    cache = {}
    calls = []

    monkeypatch.setattr("torch.distributed.get_world_size", lambda process_group: 2)
    monkeypatch.setattr("torch.distributed.get_rank", lambda: 0)
    monkeypatch.setattr(
        "torch.distributed.get_process_group_ranks",
        lambda process_group: process_group.ranks,
    )

    def fake_all_gather_object(gathered, local_counts, group):
        calls.append((local_counts, group))
        gathered[0] = dict(local_counts)
        gathered[1] = dict(local_counts)

    monkeypatch.setattr("torch.distributed.all_gather_object", fake_all_gather_object)

    first_ordered, first_multiplicity = _sync_group_batch_metadata(
        sync_group=group,
        local_batch_keys=[batch_key],
        batch_group_by_key={batch_key: batch_group},
        batch_key_cache=cache,
    )
    second_ordered, second_multiplicity = _sync_group_batch_metadata(
        sync_group=group,
        local_batch_keys=[batch_key],
        batch_group_by_key={batch_key: batch_group},
        batch_key_cache=cache,
    )

    assert first_ordered == second_ordered == [batch_key]
    assert first_multiplicity == second_multiplicity == {batch_key: 1}
    assert len(calls) == 1


def test_batch_metadata_cache_ignores_dynamic_optimizer_scalars(monkeypatch):
    group = FakeGroup((0, 1))
    static_group_key = ("fsdp_tp", 2, (), group.ranks, (0, 2), (), (group.ranks, (0, 2)))
    first_batch_key = (
        ("shape", 1),
        (0.0, 0.1, 1.0, 0.95, 0.25, 512),
        static_group_key,
    )
    second_batch_key = (
        ("shape", 1),
        (3e-7, 0.1, 1.0, 0.95, 0.25, 512),
        static_group_key,
    )
    dist_meta = SimpleNamespace(param_uid=("param", 0))
    cache = {}
    calls = []

    monkeypatch.setattr("torch.distributed.get_world_size", lambda process_group: 2)
    monkeypatch.setattr("torch.distributed.get_rank", lambda: 0)
    monkeypatch.setattr(
        "torch.distributed.get_process_group_ranks",
        lambda process_group: process_group.ranks,
    )

    def fake_all_gather_object(gathered, local_counts, group):
        calls.append((local_counts, group))
        gathered[0] = dict(local_counts)
        gathered[1] = dict(local_counts)

    monkeypatch.setattr("torch.distributed.all_gather_object", fake_all_gather_object)

    _sync_group_batch_metadata(
        sync_group=group,
        local_batch_keys=[first_batch_key],
        batch_group_by_key={
            first_batch_key: DionBatchGroup(params=[object()], dist_metas=[dist_meta])
        },
        batch_key_cache=cache,
    )
    ordered, multiplicity = _sync_group_batch_metadata(
        sync_group=group,
        local_batch_keys=[second_batch_key],
        batch_group_by_key={
            second_batch_key: DionBatchGroup(params=[object()], dist_metas=[dist_meta])
        },
        batch_key_cache=cache,
    )

    assert ordered == [second_batch_key]
    assert multiplicity == {second_batch_key: 1}
    assert len(calls) == 1


def test_batch_metadata_sync_projects_other_collective_domains(monkeypatch):
    tp_group = FakeGroup((28, 29, 30, 31))
    fs_groups = [
        (0, 4, 8, 12, 16, 20, 24, 28),
        (1, 5, 9, 13, 17, 21, 25, 29),
        (2, 6, 10, 14, 18, 22, 26, 30),
        (3, 7, 11, 15, 19, 23, 27, 31),
    ]

    def make_batch_key(fs_group):
        return (
            ("matrix", (2048, 4096), "tp-fs"),
            ("optim", 3e-5, 0.25, 512),
            (
                "fsdp_tp",
                4,
                (),
                tp_group.ranks,
                tuple(fs_group),
                (),
                (tp_group.ranks, tuple(fs_group)),
            ),
        )

    local_key = make_batch_key(fs_groups[0])
    remote_keys = [make_batch_key(fs_group) for fs_group in fs_groups]
    param_uids = tuple(("param", index) for index in range(48))
    batch_group = DionBatchGroup(
        params=[object() for _ in range(48)],
        dist_metas=[SimpleNamespace(param_uid=param_uid) for param_uid in param_uids],
    )

    monkeypatch.setattr("torch.distributed.get_world_size", lambda process_group: 4)
    monkeypatch.setattr("torch.distributed.get_rank", lambda: 28)
    monkeypatch.setattr(
        "torch.distributed.get_process_group_ranks",
        lambda process_group: process_group.ranks,
    )

    def fake_all_gather_object(gathered, local_counts, group):
        assert group is tp_group
        gathered[0] = dict(local_counts)
        for rank in range(1, 4):
            gathered[rank] = {
                _batch_key_for_sync(remote_keys[rank], tp_group, param_uids=param_uids): 48
            }

    monkeypatch.setattr("torch.distributed.all_gather_object", fake_all_gather_object)

    ordered, multiplicity = _sync_group_batch_metadata(
        sync_group=tp_group,
        local_batch_keys=[local_key],
        batch_group_by_key={local_key: batch_group},
        batch_key_cache={},
    )

    assert ordered == [local_key]
    assert multiplicity == {local_key: 48}


def test_batch_metadata_sync_still_rejects_missing_params(monkeypatch):
    tp_group = FakeGroup((28, 29, 30, 31))
    batch_key = (
        ("matrix", (2048, 4096), "tp-fs"),
        ("optim", 3e-5, 0.25, 512),
        (
            "fsdp_tp",
            4,
            (),
            tp_group.ranks,
            (0, 4, 8, 12, 16, 20, 24, 28),
            (),
            (tp_group.ranks, (0, 4, 8, 12, 16, 20, 24, 28)),
        ),
    )
    param_uids = tuple(("param", index) for index in range(48))
    batch_group = DionBatchGroup(
        params=[object() for _ in range(48)],
        dist_metas=[SimpleNamespace(param_uid=param_uid) for param_uid in param_uids],
    )

    monkeypatch.setattr("torch.distributed.get_world_size", lambda process_group: 4)
    monkeypatch.setattr("torch.distributed.get_rank", lambda: 28)
    monkeypatch.setattr(
        "torch.distributed.get_process_group_ranks",
        lambda process_group: process_group.ranks,
    )

    def fake_all_gather_object(gathered, local_counts, group):
        assert group is tp_group
        gathered[0] = dict(local_counts)
        gathered[1] = dict(local_counts)
        gathered[2] = {}
        gathered[3] = dict(local_counts)

    monkeypatch.setattr("torch.distributed.all_gather_object", fake_all_gather_object)

    with pytest.raises(RuntimeError, match="DION_BATCH_KEY_MULTIPLICITY_MISMATCH"):
        _sync_group_batch_metadata(
            sync_group=tp_group,
            local_batch_keys=[batch_key],
            batch_group_by_key={batch_key: batch_group},
            batch_key_cache={},
        )
