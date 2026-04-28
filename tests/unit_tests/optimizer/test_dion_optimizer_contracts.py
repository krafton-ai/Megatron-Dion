import importlib
from types import SimpleNamespace

import pytest
import torch

from megatron.core.optimizer import dion_distrib_optimizer as dion_do
from megatron.core.distributed.distributed_data_parallel import DistributedDataParallel
from megatron.core.optimizer.dion import algorithm as dion_algorithm
from megatron.core.optimizer.dion import kernels as dion_kernels
from megatron.core.optimizer.dion import runtime as dion_runtime
from megatron.core.optimizer.dion.types import (
    DionBatchCollectives,
    DionBatchGroup,
    DionMixedPrecisionConfig,
    DionParamConfig,
    ElementwiseStepParam,
)
from megatron.core.optimizer.distrib_dion import checkpoint_io
from megatron.core.optimizer.distrib_dion import grad_norm as dion_grad_norm
from megatron.core.optimizer.distrib_dion import gradients as dion_gradients
from megatron.core.optimizer.distrib_dion.integration import get_dion_param_override
from megatron.core.optimizer.optimizer_config import DionOptimizerConfig
from megatron.core.distributed.param_and_grad_buffer import (
    _ParamAndGradBucketGroup,
    _ParamAndGradBuffer,
)

finalize_grads = importlib.import_module("megatron.core.distributed.finalize_model_grads")


class _Group:
    def __init__(self, *, size=1, rank=0, ranks=None):
        self._size = size
        self._rank = rank
        self.ranks = tuple(range(size)) if ranks is None else tuple(ranks)

    def size(self):
        return self._size

    def rank(self):
        return self._rank


class _ParamSyncChunk:
    def __init__(self):
        self.param_sync_calls = 0

    def start_param_sync(self):
        self.param_sync_calls += 1


def _make_dion_distributed_optimizer_stub(*, overlap_param_gather, use_megatron_fsdp):
    optimizer = dion_do.DionDistributedOptimizer.__new__(dion_do.DionDistributedOptimizer)
    optimizer.is_stub_optimizer = True
    optimizer.config = SimpleNamespace(
        timers=None,
        barrier_with_L1_time=False,
        reuse_grad_buf_for_mxfp8_param_ag=False,
        overlap_param_gather=overlap_param_gather,
    )
    optimizer.ddp_config = SimpleNamespace(
        use_megatron_fsdp=use_megatron_fsdp,
        overlap_param_gather=overlap_param_gather,
    )
    chunk = _ParamSyncChunk()
    optimizer.model_chunks = [chunk]
    return optimizer, chunk


def test_step_with_ready_grads_starts_sync_for_non_overlap_param_gather():
    optimizer, chunk = _make_dion_distributed_optimizer_stub(
        overlap_param_gather=False,
        use_megatron_fsdp=False,
    )

    assert optimizer.step_with_ready_grads() is True

    assert chunk.param_sync_calls == 1


def test_step_with_ready_grads_defers_sync_for_overlap_param_gather():
    optimizer, chunk = _make_dion_distributed_optimizer_stub(
        overlap_param_gather=True,
        use_megatron_fsdp=False,
    )

    assert optimizer.step_with_ready_grads() is True

    assert chunk.param_sync_calls == 0


def test_step_with_ready_grads_starts_sync_for_megatron_fsdp():
    optimizer, chunk = _make_dion_distributed_optimizer_stub(
        overlap_param_gather=True,
        use_megatron_fsdp=True,
    )

    assert optimizer.step_with_ready_grads() is True

    assert chunk.param_sync_calls == 1


def test_bucket_group_info_accepts_mcore_group_tuple():
    group = object()

    resolved_group, group_size, group_rank = dion_do.DionDistributedOptimizer._normalize_group_info(
        (group, 8, 3),
        bucket_id=7,
        label="bucket shard",
    )

    assert resolved_group is group
    assert group_size == 8
    assert group_rank == 3


def test_bucket_group_info_fills_missing_tuple_size_rank(monkeypatch):
    group = object()
    monkeypatch.setattr(dion_do.DionDistributedOptimizer, "_group_size", lambda *_args: 4)
    monkeypatch.setattr(dion_do.DionDistributedOptimizer, "_group_rank", lambda *_args: 2)

    resolved_group, group_size, group_rank = dion_do.DionDistributedOptimizer._normalize_group_info(
        (group, None, None),
        bucket_id=7,
        label="bucket shard",
    )

    assert resolved_group is group
    assert group_size == 4
    assert group_rank == 2


def test_bucket_group_stamps_standard_shard_group_metadata():
    bucket = SimpleNamespace(
        params_list=[],
        params=set(),
        bucket_id=0,
        dion_layout=None,
    )
    ddp_config = SimpleNamespace(
        use_distributed_optimizer=True,
        num_distributed_optimizer_instances=1,
        reduce_scatter_with_fp32_accumulation=False,
    )
    group = _Group(size=4, rank=2)

    _ParamAndGradBucketGroup([bucket], ddp_config, group, 4)

    assert bucket.intra_distributed_optimizer_instance_group is group
    assert bucket.intra_distributed_optimizer_instance_size == 4
    assert bucket.intra_distributed_optimizer_instance_rank == 2


def test_bucket_shard_group_prefers_bucket_group_metadata(monkeypatch):
    class _Param:
        allreduce = False

    shard_group = _Group(size=1, rank=0)
    fallback_group = _Group(size=32, rank=7)
    param = _Param()
    bucket = SimpleNamespace(
        params={param},
        bucket_id=0,
        intra_distributed_optimizer_instance_group=shard_group,
        intra_distributed_optimizer_instance_size=1,
        intra_distributed_optimizer_instance_rank=0,
    )
    buffer = SimpleNamespace(data_parallel_group=fallback_group)
    monkeypatch.setattr(
        dion_do.parallel_state,
        "get_expert_data_parallel_group",
        lambda **_kwargs: fallback_group,
    )

    group, size, rank = dion_do.DionDistributedOptimizer._bucket_shard_get_group_size_rank(
        buffer,
        bucket,
    )

    assert group is shard_group
    assert size == 1
    assert rank == 0


def test_state_replica_group_is_independent_from_dion_rp_override(monkeypatch):
    override_rp_group = _Group(size=4, rank=1)
    state_replica_group = _Group(size=2, rank=0)
    pure_dp_group = _Group(size=4, rank=1)
    standard_dp_group = _Group(size=8, rank=3)

    monkeypatch.setattr(
        dion_do.parallel_state,
        "get_inter_distributed_optimizer_instance_group",
        lambda check_initialized=False: state_replica_group,
    )
    monkeypatch.setattr(
        dion_do.parallel_state,
        "get_context_parallel_group",
        lambda check_initialized=False: None,
    )

    wrapper = dion_do.DionDistributedOptimizer.__new__(dion_do.DionDistributedOptimizer)
    wrapper._global_rank = 0
    wrapper._pure_data_parallel_group = pure_dp_group
    wrapper.data_parallel_group = standard_dp_group
    wrapper._is_expert_dion = False
    wrapper._dion_fs_group = None
    wrapper._fs_size = 1
    wrapper._rp_size = 4
    wrapper._replica_group = override_rp_group

    wrapper._init_groups()

    assert wrapper.rp_group is override_rp_group
    assert wrapper.state_replica_group is state_replica_group


def test_init_groups_does_not_infer_dion_topology_from_standard_dp(monkeypatch):
    standard_dp_group = _Group(size=16, rank=11)
    pure_dp_group = _Group(size=8, rank=3)
    dion_fs_group = _Group(size=4, rank=1)
    dion_rp_group = _Group(size=2, rank=0)

    monkeypatch.setattr(
        dion_do.parallel_state,
        "get_inter_distributed_optimizer_instance_group",
        lambda check_initialized=False: None,
    )
    monkeypatch.setattr(
        dion_do.parallel_state,
        "get_context_parallel_group",
        lambda check_initialized=False: None,
    )

    wrapper = dion_do.DionDistributedOptimizer.__new__(dion_do.DionDistributedOptimizer)
    wrapper._global_rank = 0
    wrapper._pure_data_parallel_group = pure_dp_group
    wrapper.data_parallel_group = standard_dp_group
    wrapper._is_expert_dion = False
    wrapper._dion_fs_group = dion_fs_group
    wrapper._fs_size = 4
    wrapper._rp_size = 2
    wrapper._replica_group = dion_rp_group

    wrapper._init_groups()

    assert wrapper.validation_data_parallel_group is pure_dp_group
    assert wrapper.shard_group is standard_dp_group
    assert wrapper.fs_group is dion_fs_group
    assert wrapper.rp_group is dion_rp_group
    assert wrapper.state_replica_group is None


def test_enable_dion_runtime_validates_expert_tp_group(monkeypatch):
    dense_tp_group = _Group(size=4, rank=2)
    expert_tp_group = _Group(size=2, rank=1)
    captured = {}

    def fake_enable_distributed_dion(**kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(dion_do, "enable_distributed_dion", fake_enable_distributed_dion)
    monkeypatch.setattr(
        dion_do.parallel_state,
        "get_tensor_model_parallel_group",
        lambda check_initialized=False: dense_tp_group,
    )
    monkeypatch.setattr(
        dion_do.parallel_state,
        "get_expert_tensor_parallel_group",
        lambda check_initialized=False: expert_tp_group,
    )

    wrapper = dion_do.DionDistributedOptimizer.__new__(dion_do.DionDistributedOptimizer)
    wrapper.optimizer = dion_algorithm.MegatronDion([torch.nn.Parameter(torch.ones(1))])
    wrapper._is_expert_dion = True
    wrapper._global_rank = 0
    wrapper.validation_data_parallel_group = _Group(size=4, rank=0)
    wrapper.rp_group = None
    wrapper.fs_group = _Group(size=4, rank=0)
    wrapper.state_replica_group = None
    wrapper._rp_size = 1
    wrapper._get_replicate_group = lambda: None

    wrapper._enable_dion_runtime()

    assert captured["tp_group"] is expert_tp_group
    assert captured["tp_group"] is not dense_tp_group


def test_expert_checkpoint_topology_uses_expert_tp_group(monkeypatch):
    dense_tp_group = _Group(size=4, rank=0, ranks=(0, 1, 2, 3))
    expert_tp_group = _Group(size=2, rank=0, ranks=(0, 2))

    monkeypatch.setattr(
        dion_do.parallel_state,
        "get_context_parallel_group",
        lambda check_initialized=False: None,
    )

    wrapper = dion_do.DionDistributedOptimizer.__new__(dion_do.DionDistributedOptimizer)
    wrapper._is_expert_dion = True
    wrapper._dion_tp_group = expert_tp_group
    wrapper.tp_group = dense_tp_group
    wrapper.data_parallel_group = _Group(size=8, rank=0, ranks=tuple(range(8)))
    wrapper.fs_group = _Group(size=4, rank=0, ranks=(0, 2, 4, 6))
    wrapper.rp_group = None
    wrapper.state_replica_group = None

    topology = wrapper._dion_checkpoint_topology_signature()

    assert topology["tp"] == expert_tp_group.ranks
    assert topology["tp"] != dense_tp_group.ranks


def test_ddp_returns_dion_local_grad_for_pipeline_parallel_sync():
    param = torch.nn.Parameter(torch.empty(2, 2))
    param.is_dion_param = True
    local_grad = torch.ones(1, 2)
    optimizer = SimpleNamespace(_dion_local_grad_by_param={param: local_grad})
    bucket = SimpleNamespace(
        has_dion_params=True,
        dion_param_ids={id(param)},
        dion_optimizer=optimizer,
    )
    bucket_group = SimpleNamespace(param_to_bucket={param: bucket})
    ddp = DistributedDataParallel.__new__(DistributedDataParallel)
    ddp.param_to_bucket_group = {param: bucket_group}

    assert ddp.get_dion_local_grad(param) is local_grad


def test_conditional_embedding_allreduce_updates_dion_local_grads(monkeypatch):
    class Chunk(torch.nn.Module):
        def __init__(self, main_value, dion_value):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.empty(2, 2))
            self.weight.pipeline_parallel = True
            self.weight.main_grad = torch.full((2, 2), float(main_value))
            self.dion_grad = torch.full((1, 2), float(dion_value))

        def get_dion_local_grad(self, param):
            return self.dion_grad if param is self.weight else None

    class PPGroup:
        def size(self):
            return 2

    def fake_all_reduce(tensor, group):
        assert isinstance(group, PPGroup)
        tensor.mul_(2.0)

    monkeypatch.setattr(finalize_grads.torch.distributed, "all_reduce", fake_all_reduce)
    chunks = [Chunk(1.0, 10.0), Chunk(2.0, 20.0)]

    finalize_grads._allreduce_conditional_embedding_grads(
        chunks,
        SimpleNamespace(has_cond_embedder=True),
        PPGroup(),
    )

    assert torch.equal(chunks[0].weight.main_grad, torch.full((2, 2), 6.0))
    assert torch.equal(chunks[1].weight.main_grad, torch.full((2, 2), 6.0))
    assert torch.equal(chunks[0].dion_grad, torch.full((1, 2), 60.0))
    assert torch.equal(chunks[1].dion_grad, torch.full((1, 2), 60.0))


def test_dion_grad_sync_cache_waits_before_buffer_reuse(monkeypatch):
    class _Handle:
        def __init__(self):
            self.waited = False

        def wait(self):
            self.waited = True

    route = dion_gradients.DionGradRoute(
        group_size=1,
        group_rank=0,
        standard_shard_size=0,
        dion_numel=2,
        standard_numel=0,
        payload_numel=2,
        standard_rank_segments=((),),
        standard_local_segments=(),
    )
    bucket = SimpleNamespace(
        bucket_id=0,
        grad_data=torch.ones(2, dtype=torch.float32),
        dion_layout=SimpleNamespace(has_params=True, entries=()),
    )
    handles = []
    buffer_ptrs = []

    def fake_build_grad_reduce_input(*, bucket, route, reduce_input):
        del bucket, route
        reduce_input.fill_(1.0)
        return reduce_input

    def fake_reduce_scatter(output, input_tensor, op, group, async_op):
        del op, group, async_op
        if handles:
            assert handles[-1].waited
        buffer_ptrs.append((output.data_ptr(), input_tensor.data_ptr()))
        handle = _Handle()
        handles.append(handle)
        return handle

    monkeypatch.setattr(dion_gradients, "_get_grad_route", lambda **_kwargs: route)
    monkeypatch.setattr(
        dion_gradients,
        "_build_grad_reduce_input",
        fake_build_grad_reduce_input,
    )

    dion_gradients.start_dion_grad_sync(
        SimpleNamespace(),
        bucket=bucket,
        local_data_view=None,
        communication_group=None,
        reduce_op=torch.distributed.ReduceOp.SUM,
        async_op=True,
        reduce_scatter=fake_reduce_scatter,
    )
    dion_gradients.start_dion_grad_sync(
        SimpleNamespace(),
        bucket=bucket,
        local_data_view=None,
        communication_group=None,
        reduce_op=torch.distributed.ReduceOp.SUM,
        async_op=True,
        reduce_scatter=fake_reduce_scatter,
    )

    assert handles[0].waited is True
    assert handles[1].waited is False
    assert buffer_ptrs[1] == buffer_ptrs[0]

    dion_gradients.wait_grad_transport(bucket)
    assert handles[1].waited is True


def test_bucket_group_dispatches_dion_param_gather_hook_with_mutating_name():
    class _Optimizer:
        def __init__(self):
            self.calls = []

        def _all_gather_bucket_params_(self, bucket, async_op=False):
            self.calls.append((bucket, async_op))
            return "handle"

    optimizer = _Optimizer()
    bucket = SimpleNamespace(
        has_dion_params=True,
        has_standard_params=False,
        dion_optimizer=optimizer,
        bucket_id=6,
    )
    bucket_group = _ParamAndGradBucketGroup.__new__(_ParamAndGradBucketGroup)
    bucket_group.buckets = [bucket]

    standard_indices, handles = bucket_group._collect_param_gather_launches(async_op=True)

    assert standard_indices == []
    assert handles == ["handle"]
    assert optimizer.calls == [(bucket, True)]


def test_replicate_reduce_op_matches_mcore_ddp_scaling_mode():
    optimizer = SimpleNamespace(defaults={"rp_average_in_collective": False})

    assert dion_runtime.replicate_reduce_op(optimizer) == torch.distributed.ReduceOp.SUM

    optimizer.defaults["rp_average_in_collective"] = True
    assert dion_runtime.replicate_reduce_op(optimizer) == torch.distributed.ReduceOp.AVG


def test_distributed_optimizer_configures_dion_rp_reduce_policy():
    wrapper = dion_do.DionDistributedOptimizer.__new__(dion_do.DionDistributedOptimizer)
    wrapper.optimizer = dion_algorithm.MegatronDion(
        [torch.nn.Parameter(torch.ones(1))],
    )
    wrapper.ddp_config = SimpleNamespace(average_in_collective=False)
    wrapper.fs_size = 1

    wrapper._configure_dion_runtime_policy()

    assert wrapper.optimizer.defaults["rp_average_in_collective"] is False

    wrapper.ddp_config.average_in_collective = True
    wrapper._configure_dion_runtime_policy()
    assert wrapper.optimizer.defaults["rp_average_in_collective"] is True


def test_step_clears_runtime_buffer_cache(monkeypatch):
    param = torch.nn.Parameter(torch.ones(1))
    optimizer = dion_algorithm.MegatronDion([param], max_concurrent_tasks=2)
    optimizer.is_distributed_mode = True
    seen = []

    def make_tasks(active_optimizer):
        assert active_optimizer is optimizer
        if False:
            yield None

    class Runtime:
        def __init__(self, tasks, *, max_concurrent_tasks):
            seen.append(("init", max_concurrent_tasks, list(tasks)))

        def run(self):
            seen.append(("run",))
            optimizer._cached_buffer(
                "transient",
                (2,),
                torch.float32,
                torch.device("cpu"),
            )

    optimizer._cached_buffer("before", (1,), torch.float32, torch.device("cpu"))
    monkeypatch.setattr(dion_algorithm, "iter_dist_tasks", make_tasks)
    monkeypatch.setattr(dion_algorithm, "AsyncRuntime", Runtime)

    optimizer.step()

    assert seen == [("init", 2, []), ("run",)]
    assert optimizer._buffer_cache == {}


def test_dion_grad_norm_uses_runtime_rp_reduce_policy(monkeypatch):
    wrapper = dion_do.DionDistributedOptimizer.__new__(dion_do.DionDistributedOptimizer)
    wrapper.optimizer = SimpleNamespace(defaults={"rp_average_in_collective": False})
    replica_group = object()
    model_param_a = torch.nn.Parameter(torch.ones(1))
    model_param_b = torch.nn.Parameter(torch.ones(1))
    shard_param_a = torch.nn.Parameter(torch.ones(1))
    shard_param_b = torch.nn.Parameter(torch.ones(1))
    grads = {
        shard_param_a: torch.tensor([1.0, 2.0]),
        shard_param_b: torch.tensor([3.0]),
    }
    uids = {shard_param_a: ("a",), shard_param_b: ("b",)}
    reduce_ops = []

    def fake_all_reduce(tensor, op, group):
        assert group is replica_group
        reduce_ops.append(op)
        if op == torch.distributed.ReduceOp.SUM:
            tensor.mul_(2.0)

    wrapper._get_replicate_group = lambda: replica_group
    wrapper._group_size = lambda group: 2
    wrapper._shard_param_uid = lambda param: uids[param]
    wrapper._get_local_grad = lambda model_param, shard_param: grads[shard_param]
    monkeypatch.setattr(dion_do.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dion_do.dist, "all_reduce", fake_all_reduce)

    reduced = wrapper._dion_grads_for_norm(
        [(model_param_a, shard_param_a), (model_param_b, shard_param_b)],
        count_dion_grad=True,
    )

    assert reduce_ops == [torch.distributed.ReduceOp.SUM]
    assert [grad.tolist() for grad in reduced] == [[2.0, 4.0], [6.0]]

    reduce_ops.clear()
    wrapper.optimizer.defaults["rp_average_in_collective"] = True
    wrapper._dion_grads_for_norm(
        [(model_param_a, shard_param_a), (model_param_b, shard_param_b)],
        count_dion_grad=True,
    )

    assert reduce_ops == [torch.distributed.ReduceOp.AVG]


def test_dion_grad_norm_sq_does_not_materialize_reduced_views(monkeypatch):
    wrapper = dion_do.DionDistributedOptimizer.__new__(dion_do.DionDistributedOptimizer)
    wrapper.optimizer = SimpleNamespace(defaults={"rp_average_in_collective": False})
    replica_group = object()
    model_param_a = torch.nn.Parameter(torch.ones(1))
    model_param_b = torch.nn.Parameter(torch.ones(1))
    shard_param_a = torch.nn.Parameter(torch.ones(1))
    shard_param_b = torch.nn.Parameter(torch.ones(1))
    grads = {
        shard_param_a: torch.tensor([1.0, 2.0]),
        shard_param_b: torch.tensor([3.0]),
    }
    uids = {shard_param_a: ("a",), shard_param_b: ("b",)}
    reduce_ops = []

    def fake_all_reduce(tensor, op, group):
        assert group is replica_group
        reduce_ops.append(op)
        if op == torch.distributed.ReduceOp.SUM:
            tensor.mul_(2.0)

    wrapper._get_replicate_group = lambda: replica_group
    wrapper._group_size = lambda group: 2
    wrapper._shard_param_uid = lambda param: uids[param]
    wrapper._get_local_grad = lambda model_param, shard_param: grads[shard_param]
    monkeypatch.setattr(dion_grad_norm.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dion_grad_norm.dist, "all_reduce", fake_all_reduce)

    total_sq = dion_grad_norm._dion_grad_norm_sq(
        wrapper,
        [(model_param_a, shard_param_a), (model_param_b, shard_param_b)],
        count_dion_grad=True,
    )

    assert reduce_ops == [torch.distributed.ReduceOp.SUM]
    assert total_sq.item() == pytest.approx(2.0**2 + 4.0**2 + 6.0**2)
    assert grads[shard_param_a].tolist() == [1.0, 2.0]
    assert grads[shard_param_b].tolist() == [3.0]

    reduce_ops.clear()
    total_sq = dion_grad_norm._dion_grad_norm_sq(
        wrapper,
        [(model_param_a, shard_param_a), (model_param_b, shard_param_b)],
        count_dion_grad=False,
    )
    assert total_sq is None
    assert reduce_ops == [torch.distributed.ReduceOp.SUM]


def test_dion_grad_norm_reuses_only_dense_reduced_rp_surface(monkeypatch):
    wrapper = dion_do.DionDistributedOptimizer.__new__(dion_do.DionDistributedOptimizer)
    base_optimizer = SimpleNamespace(defaults={"rp_average_in_collective": False}, _step_count=3)
    wrapper.optimizer = base_optimizer
    replica_group = object()
    model_param_dense = torch.nn.Parameter(torch.ones(1))
    model_param_low_rank = torch.nn.Parameter(torch.ones(1))
    shard_param_dense = torch.nn.Parameter(torch.ones(1))
    shard_param_low_rank = torch.nn.Parameter(torch.ones(1))
    dense_grad = torch.tensor([1.0, 2.0])
    low_rank_grad = torch.tensor([3.0])
    grads = {
        shard_param_dense: dense_grad,
        shard_param_low_rank: low_rank_grad,
    }
    uids = {shard_param_dense: ("dense",), shard_param_low_rank: ("low-rank",)}
    wrapper.dist_metas = {
        shard_param_dense: SimpleNamespace(
            param_config=DionParamConfig(use_low_rank_sync=False),
        ),
        shard_param_low_rank: SimpleNamespace(
            param_config=DionParamConfig(use_low_rank_sync=True),
        ),
    }
    reduce_ops = []

    def fake_all_reduce(tensor, op, group):
        assert group is replica_group
        reduce_ops.append(op)
        tensor.mul_(2.0)

    wrapper._get_replicate_group = lambda: replica_group
    wrapper._group_size = lambda group: 2
    wrapper._shard_param_uid = lambda param: uids[param]
    wrapper._get_local_grad = lambda model_param, shard_param: grads[shard_param]
    monkeypatch.setattr(dion_grad_norm.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dion_grad_norm.dist, "all_reduce", fake_all_reduce)

    total_sq = dion_grad_norm._dion_grad_norm_sq(
        wrapper,
        [
            (model_param_low_rank, shard_param_low_rank),
            (model_param_dense, shard_param_dense),
        ],
        count_dion_grad=True,
    )

    assert reduce_ops == [torch.distributed.ReduceOp.SUM]
    assert total_sq.item() == pytest.approx(2.0**2 + 4.0**2 + 6.0**2)
    assert torch.equal(dense_grad, torch.tensor([2.0, 4.0]))
    assert torch.equal(low_rank_grad, torch.tensor([3.0]))
    cache = getattr(base_optimizer, "_dion_dense_grad_reduction_cache")
    assert len(cache["entries"]) == 1

    reduce_ops.clear()
    total_sq = dion_grad_norm._dion_grad_norm_sq(
        wrapper,
        [
            (model_param_low_rank, shard_param_low_rank),
            (model_param_dense, shard_param_dense),
        ],
        count_dion_grad=True,
    )

    assert reduce_ops == [torch.distributed.ReduceOp.SUM]
    assert total_sq.item() == pytest.approx(2.0**2 + 4.0**2 + 6.0**2)
    assert torch.equal(dense_grad, torch.tensor([2.0, 4.0]))
    assert torch.equal(low_rank_grad, torch.tensor([3.0]))


def test_dense_rp_step_consumes_grad_norm_reduced_surface_after_clipping(monkeypatch):
    wrapper = dion_do.DionDistributedOptimizer.__new__(dion_do.DionDistributedOptimizer)
    base_optimizer = SimpleNamespace(
        defaults={"rp_average_in_collective": False},
        _step_count=0,
        _global_rank=0,
    )
    wrapper.optimizer = base_optimizer
    replica_group = object()
    model_param = torch.nn.Parameter(torch.ones(1))
    shard_param = torch.nn.Parameter(torch.ones(1))
    grad = torch.tensor([1.0, 2.0])
    wrapper.dist_metas = {
        shard_param: SimpleNamespace(param_config=DionParamConfig(use_low_rank_sync=False)),
    }

    wrapper._get_replicate_group = lambda: replica_group
    wrapper._group_size = lambda group: 2
    wrapper._shard_param_uid = lambda param: ("dense",)
    wrapper._get_local_grad = lambda model_param, shard_param: grad
    monkeypatch.setattr(dion_grad_norm.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(
        dion_grad_norm.dist,
        "all_reduce",
        lambda tensor, op, group: tensor.mul_(2.0),
    )

    dion_grad_norm._dion_grad_norm_sq(
        wrapper,
        [(model_param, shard_param)],
        count_dion_grad=True,
    )
    grad.mul_(0.25)
    base_optimizer._step_count = 1

    def unexpected_all_reduce(*args, **kwargs):
        raise AssertionError("dense RP all-reduce should be reused")

    monkeypatch.setattr(dion_runtime.funcol, "all_reduce_coalesced", unexpected_all_reduce)

    assert list(
        dion_runtime.all_reduce_grads_across_replicas(
            base_optimizer,
            [grad],
            replicate_group=replica_group,
        )
    ) == []
    assert torch.equal(grad, torch.tensor([0.5, 1.0]))
    assert not hasattr(base_optimizer, "_dion_dense_grad_reduction_cache")


def test_dense_rp_step_rejects_partial_grad_norm_reuse(monkeypatch):
    wrapper = dion_do.DionDistributedOptimizer.__new__(dion_do.DionDistributedOptimizer)
    base_optimizer = SimpleNamespace(
        defaults={"rp_average_in_collective": False},
        _step_count=0,
        _global_rank=0,
    )
    wrapper.optimizer = base_optimizer
    replica_group = object()
    model_param = torch.nn.Parameter(torch.ones(1))
    shard_param = torch.nn.Parameter(torch.ones(1))
    cached_grad = torch.tensor([1.0, 2.0])
    missing_grad = torch.tensor([5.0])
    wrapper.dist_metas = {
        shard_param: SimpleNamespace(param_config=DionParamConfig(use_low_rank_sync=False)),
    }

    wrapper._get_replicate_group = lambda: replica_group
    wrapper._group_size = lambda group: 2
    wrapper._shard_param_uid = lambda param: ("dense",)
    wrapper._get_local_grad = lambda model_param, shard_param: cached_grad
    monkeypatch.setattr(dion_grad_norm.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(
        dion_grad_norm.dist,
        "all_reduce",
        lambda tensor, op, group: tensor.mul_(2.0),
    )

    dion_grad_norm._dion_grad_norm_sq(
        wrapper,
        [(model_param, shard_param)],
        count_dion_grad=True,
    )
    base_optimizer._step_count = 1

    monkeypatch.setattr(
        dion_runtime.funcol,
        "all_reduce_coalesced",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected fallback")),
    )

    with pytest.raises(RuntimeError, match="DION_DENSE_RP_GRAD_CACHE_PARTIAL"):
        list(
            dion_runtime.all_reduce_grads_across_replicas(
                base_optimizer,
                [cached_grad, missing_grad],
                replicate_group=replica_group,
            )
        )


def test_dense_rp_step_ignores_stale_grad_norm_reuse(monkeypatch):
    grad = torch.tensor([1.0, 2.0])
    replica_group = object()
    optimizer = SimpleNamespace(
        defaults={"rp_average_in_collective": False},
        _step_count=2,
        _global_rank=0,
    )
    stale_entry = dion_runtime._tensor_region(grad)
    stale_entry.update(
        {
            "group": replica_group,
            "op": torch.distributed.ReduceOp.SUM,
            "before_step": 0,
        }
    )
    setattr(
        optimizer,
        "_dion_dense_grad_reduction_cache",
        {"entries": [stale_entry]},
    )
    calls = []

    def fake_all_reduce_coalesced(grads, reduceOp, group):
        calls.append((reduceOp, group))
        return [local_grad * 2.0 for local_grad in grads]

    monkeypatch.setattr(
        dion_runtime.funcol,
        "all_reduce_coalesced",
        fake_all_reduce_coalesced,
    )

    assert list(
        dion_runtime.all_reduce_grads_across_replicas(
            optimizer,
            [grad],
            replicate_group=replica_group,
        )
    ) == [None]
    assert calls == [("sum", replica_group)]
    assert torch.equal(grad, torch.tensor([2.0, 4.0]))
    assert not hasattr(optimizer, "_dion_dense_grad_reduction_cache")


def test_low_rank_replicated_ortho_uses_reference_avg_reduce(monkeypatch):
    class DummyOptimizer:
        _step_count = 1
        _global_rank = 0

        def __init__(self, average_in_collective):
            self.defaults = {
                "rp_average_in_collective": average_in_collective,
                "rcqr_oversample": 0,
            }
            self.use_fs_collectives = False
            self.is_distributed_mode = True

        def _cached_buffer(self, name, shape, dtype, device, *, zero=False):
            del name
            tensor = torch.empty(shape, dtype=dtype, device=device)
            if zero:
                tensor.zero_()
            return tensor

    replica_group = object()
    reduce_scatter_ops = []

    def fake_reduce_scatter_tensor(input_tensor, reduceOp, scatter_dim, group):
        assert group is replica_group
        assert scatter_dim == 0
        reduce_scatter_ops.append(reduceOp)
        return input_tensor[:1].clone()

    def fake_all_gather_tensor(input_tensor, gather_dim, group):
        assert group is replica_group
        assert gather_dim == 0
        return input_tensor.repeat(2, 1, 1)

    def fake_all_reduce(input_tensor, reduceOp, group):
        del reduceOp
        assert group is replica_group
        return input_tensor

    def run_once(average_in_collective):
        optimizer = DummyOptimizer(average_in_collective)
        gen = dion_runtime.run_low_rank_sync_async(
            optimizer,
            P_batch=torch.ones(2, 3, 1),
            M_batch=torch.ones(2, 3, 2),
            Q_batch=torch.empty(0),
            configs=[DionParamConfig(), DionParamConfig()],
            dist_metas=[object(), object()],
            batch_group=DionBatchGroup(
                kernel_kind="ddp",
                replicate_group=replica_group,
            ),
            batch_collectives=DionBatchCollectives(),
            real_batch_size=2,
        )
        try:
            while True:
                next(gen)
        except StopIteration as exc:
            return exc.value

    monkeypatch.setattr(dion_runtime.dist, "get_world_size", lambda group: 2)
    monkeypatch.setattr(dion_runtime.dist, "get_rank", lambda group: 0)
    monkeypatch.setattr(dion_runtime.funcol, "reduce_scatter_tensor", fake_reduce_scatter_tensor)
    monkeypatch.setattr(dion_runtime.funcol, "all_gather_tensor", fake_all_gather_tensor)
    monkeypatch.setattr(dion_runtime.funcol, "all_reduce", fake_all_reduce)
    monkeypatch.setattr(dion_runtime, "orthogonalize", lambda tensor, **kwargs: tensor)

    run_once(average_in_collective=False)
    assert reduce_scatter_ops == ["avg"]

    reduce_scatter_ops.clear()
    run_once(average_in_collective=True)
    assert reduce_scatter_ops == ["avg"]


def test_dion_rejects_disabled_fs_collectives_when_fs_sharded():
    wrapper = dion_do.DionDistributedOptimizer.__new__(dion_do.DionDistributedOptimizer)
    wrapper.optimizer = dion_algorithm.MegatronDion(
        [torch.nn.Parameter(torch.ones(1))],
        use_fs_collectives=False,
    )
    wrapper.ddp_config = SimpleNamespace(average_in_collective=False)
    wrapper.fs_size = 2

    with pytest.raises(RuntimeError, match="no-dion-use-fs-collectives"):
        wrapper._configure_dion_runtime_policy()


def test_copy_main_params_quantizes_fp8_before_model_writeback(monkeypatch):
    optimizer = dion_do.DionDistributedOptimizer.__new__(dion_do.DionDistributedOptimizer)
    optimizer.is_stub_optimizer = False
    optimizer.ddp_config = SimpleNamespace(use_megatron_fsdp=False)
    optimizer.config = SimpleNamespace(use_precision_aware_optimizer_no_fp8_or_ds_fp8=False)
    optimizer.data_parallel_group = object()
    optimizer.model_chunks = []
    optimizer.model_float16_groups = []
    optimizer.shard_fp32_from_float16_groups = []
    optimizer.shard_float16_groups = []
    optimizer.model_fp32_groups = []
    optimizer.shard_fp32_groups = []
    optimizer._get_data_shard = lambda param: None
    optimizer._get_model_param_range_map = lambda param: {}
    optimizer._param_shard_layout = lambda param: None
    optimizer._bucket_param_data = lambda param: None
    optimizer._mark_buckets_full_param_ready = lambda ready: None
    optimizer._check_main_shards = lambda main_shard_groups: None
    optimizer._restore_bucket_param_views_ = lambda dion_only=False: None

    fp8_param = object()
    main_param = object()
    optimizer._get_fp8_params_and_shard_fp32_from_fp8 = lambda: ([fp8_param], [main_param], [3])

    calls = []
    monkeypatch.setattr(
        dion_do,
        "quantize_param_shard",
        lambda model_params, main_params, offsets, group: calls.append(
            (model_params, main_params, offsets, group)
        ),
    )
    monkeypatch.setattr(
        dion_do,
        "copy_main_params_to_model_shards",
        lambda **kwargs: kwargs["empty_range_warning_count"],
    )

    optimizer._copy_main_params_to_model_params()

    assert calls == [([fp8_param], [main_param], [3], optimizer.data_parallel_group)]


def test_standard_writeback_skips_fp8_after_quantize(monkeypatch):
    model_param = torch.nn.Parameter(torch.ones(2))
    shard_param = torch.ones(2)

    monkeypatch.setattr(checkpoint_io, "is_float8tensor", lambda param: param is model_param)

    copied = checkpoint_io.write_standard_shards_to_model_(
        model_groups=[[model_param]],
        shard_groups=[[shard_param]],
        get_param_range_map=lambda param: (_ for _ in ()).throw(AssertionError("unexpected copy")),
        get_bucket_param_data=lambda param: (_ for _ in ()).throw(AssertionError("unexpected copy")),
    )

    assert copied == 0


def test_precision_aware_dion_float16_uses_fp32_optimizer_shard():
    optimizer = dion_do.DionDistributedOptimizer.__new__(dion_do.DionDistributedOptimizer)
    optimizer._param_name = lambda param: "param"
    optimizer._create_fs_shard = lambda model_param, shard_layout: model_param.detach()
    optimizer._attach_fs_shard_ = lambda model_param, shard: None
    optimizer._check_dion_param = lambda model_param, context: None
    registered = {}
    optimizer._register_dion_shard = lambda **kwargs: registered.update(kwargs)

    model_param = torch.nn.Parameter(torch.ones(2, 2, dtype=torch.bfloat16))
    model_param.is_dion_param = True
    config = SimpleNamespace(use_precision_aware_optimizer_no_fp8_or_ds_fp8=True)
    model_fp16_params = []
    shard_float16_params = []
    main_shard_params = []

    optimizer._process_float16_param(
        model_param,
        SimpleNamespace(start=0, end=4),
        object(),
        config,
        model_fp16_params,
        shard_float16_params,
        main_shard_params,
    )

    assert main_shard_params[0].dtype == torch.float32
    assert model_param.main_param is main_shard_params[0]
    assert shard_float16_params[0] is main_shard_params[0]
    assert registered["opt_shard"] is main_shard_params[0]


def test_elementwise_state_defaults_follow_param_dtype(monkeypatch):
    monkeypatch.setattr(dion_algorithm, "adamw_update_foreach", lambda *args, **kwargs: None)
    param = torch.nn.Parameter(torch.ones(2, dtype=torch.float16))
    grad = torch.ones_like(param)
    state = {}
    optimizer = dion_algorithm.MegatronDion(
        [param],
        mixed_precision_config=DionMixedPrecisionConfig(),
        elementwise_optimizer="adamw",
    )
    optimizer.param_groups[0]["step"] = 1

    with torch.no_grad():
        list(
            optimizer._apply_elementwise_batches(
                [
                    ElementwiseStepParam(
                        param=param,
                        grad=grad,
                        optimizer_state=state,
                        optim_group=optimizer.param_groups[0],
                    )
                ]
            )
        )

    assert state["first_moment"].dtype == torch.float16
    assert state["second_moment"].dtype == torch.float16


def test_elementwise_weight_decay_respects_wd_mult_without_scheduler():
    param = torch.nn.Parameter(torch.ones(1, dtype=torch.float32))
    grad = torch.zeros_like(param)
    state = {}
    optimizer = dion_algorithm.MegatronDion(
        [{"params": [param], "wd_mult": 0.0}],
        lr=1.0,
        weight_decay=0.1,
        elementwise_optimizer="adamw",
    )
    optimizer.param_groups[0]["step"] = 1

    with torch.no_grad():
        list(
            optimizer._apply_elementwise_batches(
                [
                    ElementwiseStepParam(
                        param=param,
                        grad=grad,
                        optimizer_state=state,
                        optim_group=optimizer.param_groups[0],
                    )
                ]
            )
        )

    assert torch.equal(param.detach(), torch.ones_like(param))
    assert optimizer.param_groups[0]["weight_decay"] == 0.0


def test_elementwise_weight_decay_respects_wd_mult_with_explicit_group_weight_decay():
    param = torch.nn.Parameter(torch.ones(1, dtype=torch.float32))
    grad = torch.zeros_like(param)
    state = {}
    optimizer = dion_algorithm.MegatronDion(
        [{"params": [param], "weight_decay": 0.1, "wd_mult": 0.0}],
        lr=1.0,
        weight_decay=0.1,
        elementwise_optimizer="adamw",
    )
    optimizer.param_groups[0]["step"] = 1

    with torch.no_grad():
        list(
            optimizer._apply_elementwise_batches(
                [
                    ElementwiseStepParam(
                        param=param,
                        grad=grad,
                        optimizer_state=state,
                        optim_group=optimizer.param_groups[0],
                    )
                ]
            )
        )

    assert torch.equal(param.detach(), torch.ones_like(param))
    assert optimizer.param_groups[0]["weight_decay"] == 0.0


def test_dion_local_grad_is_unscale_surface_for_non_precision_aware():
    model_param = torch.nn.Parameter(torch.empty((2, 2), dtype=torch.bfloat16))
    local_grad = torch.ones((2, 2), dtype=torch.bfloat16)
    shard_layout = SimpleNamespace(local_shape=(2, 2))
    optimizer = SimpleNamespace(
        config=SimpleNamespace(use_precision_aware_optimizer_no_fp8_or_ds_fp8=False),
        _dion_local_grad_by_param={},
        _param_shard_layout=lambda param: shard_layout,
        _param_name=lambda param: "param",
    )

    dion_gradients.set_local_grad(optimizer, model_param, local_grad)

    stored = optimizer._dion_local_grad_by_param[model_param]
    assert stored.dtype == torch.float32

    shard_param = torch.nn.Parameter(torch.empty((2, 2), dtype=torch.float32))
    dion_gradients.install_dion_shard_grad_(
        model_param=model_param,
        shard_param=shard_param,
        get_local_grad=lambda model_param, shard_param: stored,
        log_grad_issue=lambda *args, **kwargs: None,
        use_precision_aware_optimizer=False,
    )
    assert shard_param.grad is stored

    shard_param.grad.mul_(0.5)
    assert torch.equal(optimizer._dion_local_grad_by_param[model_param], torch.full((2, 2), 0.5))


def test_scale_gradients_leaves_dion_only_grad_buffer_to_adapter_surface():
    calls = []

    class Optimizer:
        def _scale_dion_bucket_grads(self, **kwargs):
            calls.append(kwargs)

    bucket = SimpleNamespace(
        bucket_id=0,
        grad_data=torch.ones(4, dtype=torch.float32),
        has_dion_params=True,
        has_standard_params=False,
        dion_optimizer=Optimizer(),
    )
    bucket_group = SimpleNamespace(
        buckets=[bucket],
        ddp_config=SimpleNamespace(use_distributed_optimizer=True),
        intra_distributed_optimizer_instance_group=object(),
    )

    _ParamAndGradBuffer.scale_gradients(bucket_group, 0.5)

    assert torch.equal(bucket.grad_data, torch.ones(4, dtype=torch.float32))
    assert len(calls) == 1
    assert calls[0]["bucket"] is bucket
    assert calls[0]["local_data_view"] is None
    assert calls[0]["scaling_factor"] == 0.5
    assert calls[0]["use_distributed_optimizer"] is True


def test_scale_dion_bucket_grads_scales_only_active_surfaces(monkeypatch):
    standard_param = torch.nn.Parameter(torch.empty(4))
    dion_param = torch.nn.Parameter(torch.empty(4))
    dion_local_grad = torch.ones(2, dtype=torch.float32)
    bucket = SimpleNamespace(
        bucket_id=0,
        grad_data=torch.arange(8, dtype=torch.float32),
        params_list=[standard_param, dion_param],
        param_to_index={standard_param: (0, 4), dion_param: (4, 8)},
        dion_param_ids=frozenset({id(dion_param)}),
        has_standard_params=True,
        dion_layout=SimpleNamespace(
            has_params=True,
            shard_size=2,
            entries=(SimpleNamespace(param=dion_param),),
        ),
    )
    optimizer = SimpleNamespace(
        _dion_local_grad_by_param={dion_param: dion_local_grad},
        _param_name=lambda param: "dion_param",
    )
    group = object()
    local_data_view = bucket.grad_data[:4]

    monkeypatch.setattr(dion_gradients.dist, "get_world_size", lambda process_group: 2)
    monkeypatch.setattr(dion_gradients.dist, "get_rank", lambda process_group: 0)

    dion_gradients.scale_dion_bucket_grads(
        optimizer,
        bucket=bucket,
        local_data_view=local_data_view,
        communication_group=group,
        scaling_factor=0.5,
        use_distributed_optimizer=True,
    )

    assert torch.equal(bucket.grad_data[:4], torch.tensor([0.0, 0.5, 1.0, 1.5]))
    assert torch.equal(bucket.grad_data[4:], torch.tensor([4.0, 5.0, 6.0, 7.0]))
    assert torch.equal(dion_local_grad, torch.full((2,), 0.5))


def test_scale_dion_bucket_grads_scales_standard_ranges_without_distopt():
    standard_param = torch.nn.Parameter(torch.empty(4))
    dion_param = torch.nn.Parameter(torch.empty(4))
    dion_local_grad = torch.ones(2, dtype=torch.float32)
    bucket = SimpleNamespace(
        bucket_id=0,
        grad_data=torch.arange(8, dtype=torch.float32),
        params_list=[standard_param, dion_param],
        param_to_index={standard_param: (0, 4), dion_param: (4, 8)},
        dion_param_ids=frozenset({id(dion_param)}),
        has_standard_params=True,
        dion_layout=SimpleNamespace(
            has_params=True,
            entries=(SimpleNamespace(param=dion_param),),
        ),
    )
    optimizer = SimpleNamespace(
        _dion_local_grad_by_param={dion_param: dion_local_grad},
        _param_name=lambda param: "dion_param",
    )

    dion_gradients.scale_dion_bucket_grads(
        optimizer,
        bucket=bucket,
        local_data_view=None,
        communication_group=None,
        scaling_factor=0.25,
        use_distributed_optimizer=False,
    )

    assert torch.equal(bucket.grad_data[:4], torch.tensor([0.0, 0.25, 0.5, 0.75]))
    assert torch.equal(bucket.grad_data[4:], torch.tensor([4.0, 5.0, 6.0, 7.0]))
    assert torch.equal(dion_local_grad, torch.full((2,), 0.25))


def test_error_feedback_skips_padded_batch_entries(monkeypatch):
    momentums = [
        torch.zeros(2, 2, dtype=torch.float32),
        torch.full((2, 2), 5.0, dtype=torch.float32),
    ]
    P_batch = torch.ones(2, 2, 1, dtype=torch.float32)
    R_batch = torch.ones(2, 2, 1, dtype=torch.float32)

    def fake_apply_batched_matmul(X, A, B, *, alpha, beta, transpose):
        assert len(X) == 1
        assert A.size(0) == B.size(0) == 1
        assert transpose is False
        X[0].mul_(beta)
        X[0].add_(A[0] @ B[0].mT, alpha=alpha)

    monkeypatch.setattr(dion_kernels, "apply_batched_matmul", fake_apply_batched_matmul)

    dion_kernels.apply_error_feedback(
        momentums,
        P_batch,
        R_batch,
        [DionParamConfig(), DionParamConfig()],
        [{"mu": 0.0}, {"mu": 0.0}],
        default_mu=0.0,
        real_batch_size=1,
    )

    assert torch.equal(momentums[0], torch.full((2, 2), -1.0))
    assert torch.equal(momentums[1], torch.full((2, 2), 5.0))


def test_fix_all_zero_or_nan_only_processes_real_batch_entries():
    P_batch = torch.tensor(
        [
            [[float("nan")], [2.0]],
            [[1.0], [3.0]],
            [[float("nan")], [7.0]],
        ],
        dtype=torch.float32,
    )
    R_batch = torch.tensor(
        [
            [[float("nan")], [5.0], [6.0]],
            [[9.0], [10.0], [11.0]],
            [[float("nan")], [12.0], [13.0]],
        ],
        dtype=torch.float32,
    )
    Q_batch = torch.tensor(
        [
            [[4.0], [5.0], [6.0]],
            [[float("nan")], [8.0], [9.0]],
            [[20.0], [21.0], [float("nan")]],
        ],
        dtype=torch.float32,
    )
    M_batch = torch.ones((3, 2, 3), dtype=torch.float32)
    M_batch[1].zero_()

    fixed_p, fixed_r = dion_kernels.fix_all_zero_or_nan(
        P_batch,
        R_batch,
        Q_batch,
        M_batch,
        real_batch_size=2,
    )

    assert torch.equal(fixed_p[0], torch.tensor([[0.0], [2.0]]))
    assert torch.equal(fixed_r[0], torch.tensor([[0.0], [5.0], [6.0]]))
    assert torch.equal(fixed_p[1], torch.zeros((2, 1)))
    assert torch.equal(fixed_r[1], torch.tensor([[0.0], [8.0], [9.0]]))
    assert torch.isnan(fixed_p[2, 0, 0])
    assert fixed_p[2, 1, 0].item() == 7.0
    assert torch.isnan(fixed_r[2, 0, 0])
    assert torch.equal(fixed_r[2, 1:], torch.tensor([[12.0], [13.0]]))


def test_stack_tensors_as_dtype_casts_directly_to_final_dtype():
    tensors = [
        torch.ones(2, 2, dtype=torch.float16),
        torch.full((2, 2), 2.0, dtype=torch.float16),
    ]

    stacked = dion_runtime._stack_tensors_as_dtype(tensors, torch.float32)

    assert stacked.dtype == torch.float32
    assert torch.equal(stacked[0], torch.ones(2, 2, dtype=torch.float32))
    assert torch.equal(stacked[1], torch.full((2, 2), 2.0, dtype=torch.float32))


def test_zero_numel_dion_shard_is_rejected_in_local_optimizer_maps():
    param = torch.nn.Parameter(torch.empty((0, 4)))
    shard_layout = SimpleNamespace(local_shape=(0, 4), local_numel=0)
    empty_range = SimpleNamespace(size=0)
    gbuf_ranges = [
        {
            torch.float32: [
                {
                    "param_map": {
                        param: {
                            "param": empty_range,
                            "dion_shard_layout": shard_layout,
                        }
                    }
                }
            ]
        }
    ]
    param_groups = [{"params": [param]}]

    with pytest.raises(RuntimeError, match="invalid empty Dion local shard"):
        dion_do.DionDistributedOptimizer._build_model_param_gbuf_map(gbuf_ranges)
    with pytest.raises(RuntimeError, match="invalid empty Dion local shard"):
        dion_do.DionDistributedOptimizer._build_optimizer_group_ranges(
            param_groups,
            gbuf_ranges,
        )


def test_compute_update_batch_matches_reference_operand_dtype(monkeypatch):
    monkeypatch.setattr(
        dion_kernels,
        "_compute_update_batch_regular",
        lambda q_new, p_batch: torch.bmm(p_batch, q_new.transpose(1, 2)),
    )
    q_new = torch.ones(1, 3, 2, dtype=torch.float16)
    p_batch = torch.ones(1, 4, 2, dtype=torch.float32)
    delta = dion_kernels.compute_update_batch(
        q_new,
        p_batch,
        [SimpleNamespace(is_transposed=False)],
        real_batch_size=1,
        delta_shape=(4, 3),
    )

    assert delta.dtype == p_batch.dtype


def test_dion_scale_mode_defaults_match_muon_style_naming():
    config = DionOptimizerConfig()

    assert config.dion_scale_mode == "spectral"
    assert config.dion_extra_scale_factor == 0.2


def test_scaled_lr_uses_rank_fraction_for_all_scale_modes():
    spectral_lr = dion_kernels.scaled_lr_for_shape(
        lr=1.0,
        m_global=16,
        n_global=4,
        scale_mode="spectral",
        rank_fraction=0.25,
        extra_scale_factor=0.2,
    )
    unit_rms_lr = dion_kernels.scaled_lr_for_shape(
        lr=1.0,
        m_global=16,
        n_global=4,
        scale_mode="unit_rms_norm",
        rank_fraction=0.25,
        extra_scale_factor=0.2,
    )
    shape_lr = dion_kernels.scaled_lr_for_shape(
        lr=1.0,
        m_global=4,
        n_global=16,
        scale_mode="shape_scaling",
        rank_fraction=0.25,
        extra_scale_factor=0.2,
    )

    assert spectral_lr == pytest.approx(1.6)
    assert unit_rms_lr == pytest.approx(0.8)
    assert shape_lr == pytest.approx(0.4)


def test_dion_output_layer_override_uses_standard_embedding_role_name():
    config = DionOptimizerConfig(lr=1.0, min_lr=0.25, dion_scale_mode="unit_rms_norm")
    param = torch.nn.Parameter(torch.empty(8, 4))
    param.is_embedding_or_output_parameter = True

    override = get_dion_param_override(
        config,
        param,
        None,
        "decoder.output_layer.weight",
    )

    assert override["max_lr"] == 0.5
    assert override["min_lr"] == 0.125
