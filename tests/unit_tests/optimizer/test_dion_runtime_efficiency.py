from types import SimpleNamespace

import pytest
import torch

from megatron.core.optimizer.dion import runtime as dion_runtime
from megatron.core.optimizer.dion.types import (
    DionAxisCollective,
    DionBatchCollectives,
    DionDistMeta,
    DionParamConfig,
)


class _Handle:
    def __init__(self):
        self.waited = False

    def wait(self):
        self.waited = True


class _BufferOptimizer:
    _step_count = 1
    _global_rank = 0

    def __init__(self):
        self.buffers = {}

    def _cached_buffer(self, name, shape, dtype, device, *, zero=False):
        tensor = torch.empty(shape, dtype=dtype, device=device)
        if zero:
            tensor.zero_()
        self.buffers[name] = tensor
        return tensor


def _finish_generator(gen):
    while True:
        try:
            next(gen)
        except StopIteration as exc:
            return exc.value


def test_replica_batch_all_reduce_uses_in_place_when_allowed(monkeypatch):
    group = object()
    batch = torch.ones(2, 3)
    handles = []

    def fake_all_reduce(tensor, op, group=None, async_op=False):
        assert tensor is batch
        assert op == torch.distributed.ReduceOp.AVG
        assert async_op is True
        tensor.mul_(4.0)
        handle = _Handle()
        handles.append(handle)
        return handle

    monkeypatch.setattr(dion_runtime.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dion_runtime.dist, "all_reduce", fake_all_reduce)
    monkeypatch.setattr(
        dion_runtime.funcol,
        "all_reduce",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected fallback")),
    )

    gen = dion_runtime.all_reduce_batch_across_replicas(
        SimpleNamespace(_step_count=1, _global_rank=0),
        batch,
        replicate_group=group,
        allow_in_place=True,
    )

    assert next(gen) is None
    assert torch.equal(batch, torch.full((2, 3), 4.0))
    with pytest.raises(StopIteration) as stop:
        next(gen)

    assert stop.value.value is batch
    assert handles[0].waited is True


def test_replica_batch_all_reduce_keeps_functional_fallback_when_not_allowed(monkeypatch):
    group = object()
    batch = torch.ones(2, 3)
    calls = []

    def fake_functional_all_reduce(tensor, reduceOp, group=None):
        calls.append((tensor, reduceOp, group))
        return tensor + 2.0

    monkeypatch.setattr(
        dion_runtime.dist,
        "all_reduce",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected in-place")),
    )
    monkeypatch.setattr(dion_runtime.funcol, "all_reduce", fake_functional_all_reduce)

    gen = dion_runtime.all_reduce_batch_across_replicas(
        SimpleNamespace(_step_count=1, _global_rank=0),
        batch,
        replicate_group=group,
    )

    assert next(gen) is None
    with pytest.raises(StopIteration) as stop:
        next(gen)

    assert len(calls) == 1
    assert calls[0][0] is batch
    assert calls[0][1:] == ("avg", group)
    assert stop.value.value is not batch
    assert torch.equal(stop.value.value, torch.full((2, 3), 3.0))
    assert torch.equal(batch, torch.ones(2, 3))


def test_dense_rp_grad_cache_miss_uses_in_place_coalesced_for_contiguous(monkeypatch):
    group = object()
    grad_a = torch.tensor([1.0, 2.0])
    grad_b = torch.tensor([3.0])
    handles = []

    def fake_all_reduce_coalesced(tensors, op, group=None, async_op=False):
        assert len(tensors) == 2
        assert tensors[0] is grad_a
        assert tensors[1] is grad_b
        assert op == torch.distributed.ReduceOp.SUM
        assert async_op is True
        for tensor in tensors:
            tensor.mul_(2.0)
        handle = _Handle()
        handles.append(handle)
        return handle

    monkeypatch.setattr(dion_runtime.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(
        dion_runtime.dist,
        "all_reduce_coalesced",
        fake_all_reduce_coalesced,
        raising=False,
    )
    monkeypatch.setattr(
        dion_runtime.funcol,
        "all_reduce_coalesced",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected fallback")),
    )

    gen = dion_runtime.all_reduce_grads_across_replicas(
        SimpleNamespace(
            defaults={"rp_average_in_collective": False},
            _step_count=1,
            _global_rank=0,
        ),
        [grad_a, grad_b],
        replicate_group=group,
    )

    assert next(gen) is None
    assert torch.equal(grad_a, torch.tensor([2.0, 4.0]))
    assert torch.equal(grad_b, torch.tensor([6.0]))
    with pytest.raises(StopIteration):
        next(gen)
    assert handles[0].waited is True


def test_dense_rp_grad_cache_miss_keeps_non_contiguous_functional_fallback(monkeypatch):
    group = object()
    base = torch.arange(6.0).view(2, 3)
    grad = base.t()
    expected = grad.clone() + 10.0
    calls = []

    def fake_functional_all_reduce_coalesced(tensors, reduceOp, group=None):
        assert len(tensors) == 1
        assert tensors[0].is_contiguous()
        assert tensors[0].data_ptr() != grad.data_ptr()
        calls.append((reduceOp, group))
        return [tensors[0] + 10.0]

    monkeypatch.setattr(
        dion_runtime.dist,
        "all_reduce_coalesced",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected in-place")),
        raising=False,
    )
    monkeypatch.setattr(
        dion_runtime.funcol,
        "all_reduce_coalesced",
        fake_functional_all_reduce_coalesced,
    )

    gen = dion_runtime.all_reduce_grads_across_replicas(
        SimpleNamespace(
            defaults={"rp_average_in_collective": True},
            _step_count=1,
            _global_rank=0,
        ),
        [grad],
        replicate_group=group,
    )

    assert next(gen) is None
    with pytest.raises(StopIteration):
        next(gen)

    assert calls == [("avg", group)]
    assert torch.equal(grad, expected)


def test_unshard_q_batch_exposes_direct_full_batch_for_proven_full_order(monkeypatch):
    group = object()
    q0 = torch.tensor([[1.0], [2.0]])
    q1 = torch.tensor([[3.0], [4.0]])

    def fake_all_gather_into_tensor(output, input_tensor, group=None, async_op=False):
        assert group is not None
        assert async_op is True
        gathered = torch.empty((2, 2, 2, 1), dtype=input_tensor.dtype)
        gathered[0].copy_(input_tensor.view(2, 2, 1))
        gathered[1, 0, :, 0] = torch.tensor([10.0, 20.0])
        gathered[1, 1, :, 0] = torch.tensor([30.0, 40.0])
        output.copy_(gathered.view(-1))
        return _Handle()

    monkeypatch.setattr(dion_runtime.dist, "all_gather_into_tensor", fake_all_gather_into_tensor)
    optimizer = _BufferOptimizer()
    collectives = DionBatchCollectives(
        tp_q_gathers=(
            DionAxisCollective(indices=(0, 1), process_group=group, world_size=2, rank=0),
        )
    )

    gen = dion_runtime.unshard_q_batch(
        optimizer,
        [q0, q1],
        [DionParamConfig(use_tp_shard=True), DionParamConfig(use_tp_shard=True)],
        [DionDistMeta(param_uid=("a",)), DionDistMeta(param_uid=("b",))],
        batch_collectives=collectives,
        optimizer_states=[{"r": 2}, {"r": 2}],
        batch_cache_key=7,
        real_batch_size=2,
        direct_batch_dtype=torch.float32,
        direct_batch_device=torch.device("cpu"),
        direct_batch_q_rows=2,
    )

    assert next(gen) is None
    q_inputs, direct_q_batch = _finish_generator(gen)

    expected = torch.tensor(
        [
            [[1.0, 10.0], [2.0, 20.0]],
            [[3.0, 30.0], [4.0, 40.0]],
        ]
    )
    assert direct_q_batch is optimizer.buffers["q_full_batch_7_0"]
    assert torch.equal(direct_q_batch, expected)

    monkeypatch.setattr(
        dion_runtime,
        "_stack_tensors_as_dtype",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected restack")),
    )
    assert (
        dion_runtime._q_batch_from_inputs(
            q_inputs,
            direct_q_batch,
            torch.float32,
            torch.device("cpu"),
        )
        is direct_q_batch
    )


def test_unshard_q_batch_direct_full_batch_rejects_padding(monkeypatch):
    group = object()
    q0 = torch.tensor([[1.0], [2.0]])
    q1 = torch.zeros_like(q0)

    def fake_all_gather_into_tensor(output, input_tensor, group=None, async_op=False):
        del group, async_op
        gathered = torch.zeros((2, 2, 2, 1), dtype=input_tensor.dtype)
        gathered[0].copy_(input_tensor.view(2, 2, 1))
        output.copy_(gathered.view(-1))
        return _Handle()

    monkeypatch.setattr(dion_runtime.dist, "all_gather_into_tensor", fake_all_gather_into_tensor)
    collectives = DionBatchCollectives(
        tp_q_gathers=(
            DionAxisCollective(indices=(0, 1), process_group=group, world_size=2, rank=0),
        )
    )

    gen = dion_runtime.unshard_q_batch(
        _BufferOptimizer(),
        [q0, q1],
        [DionParamConfig(use_tp_shard=True), DionParamConfig(use_tp_shard=True)],
        [DionDistMeta(param_uid=("a",)), None],
        batch_collectives=collectives,
        optimizer_states=[{"r": 2}, None],
        batch_cache_key=8,
        real_batch_size=1,
        direct_batch_dtype=torch.float32,
        direct_batch_device=torch.device("cpu"),
        direct_batch_q_rows=2,
    )

    assert next(gen) is None
    q_inputs, direct_q_batch = _finish_generator(gen)

    assert isinstance(q_inputs, list)
    assert direct_q_batch is None
