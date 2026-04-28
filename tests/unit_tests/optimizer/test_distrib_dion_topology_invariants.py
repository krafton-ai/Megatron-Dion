import math
from types import SimpleNamespace

import pytest
import torch

from megatron.core.optimizer.dion import ortho, runtime
from megatron.core.optimizer.dion.ortho import reshard_q_along_tp
from megatron.core.optimizer.dion.state import (
    init_q_state,
    require_2d_local_shape,
    resolve_q_state_layout,
)
from megatron.core.optimizer.dion.types import (
    DionAxisCollective,
    DionBatchCollectives,
    DionMixedPrecisionConfig,
    DionParamConfig,
    DionDistMeta,
)
from megatron.core.optimizer.distrib_dion.batches import build_batch_collectives, build_batch_key
from megatron.core.optimizer.distrib_dion.parameter import build_dion_shard_entries


def test_resolve_q_state_layout_keeps_reference_rank_with_uneven_tp():
    config = DionParamConfig(has_tp_shard=True, use_tp_shard=True, tp_shard_dim=1)
    global_shape = (17, 19)
    expected_rank = math.ceil(0.5 * min(global_shape))
    local_cols = []

    for tp_rank in range(4):
        layout = resolve_q_state_layout(
            17,
            5,
            config,
            tp_world_size=4,
            tp_rank=tp_rank,
            use_q_unshard=True,
            global_shape=global_shape,
            rank_fraction=0.5,
            rank_multiple_of=1,
        )
        assert layout.r_global == expected_rank
        local_cols.append(layout.r_local)

    assert expected_rank == 9
    assert local_cols == [3, 2, 2, 2]


def test_empty_2d_local_shape_is_rejected():
    param = torch.nn.Parameter(torch.empty((0, 8), dtype=torch.float32))
    dist_meta = DionDistMeta(
        shape=(0, 8),
        global_shape=(2, 8),
        is_dion_param=True,
        param_uid=("param",),
        param_name="param",
    )

    with pytest.raises(RuntimeError, match="empty local 2D shape"):
        require_2d_local_shape(param, dist_meta)


def test_resolve_q_state_layout_rejects_empty_tp_rank_shard():
    config = DionParamConfig(has_tp_shard=True, use_tp_shard=True, tp_shard_dim=1)

    with pytest.raises(RuntimeError, match="DION_EMPTY_Q_SHARD"):
        resolve_q_state_layout(
            4,
            1,
            config,
            tp_world_size=4,
            tp_rank=1,
            use_q_unshard=True,
            global_shape=(4, 4),
            rank_fraction=0.25,
            rank_multiple_of=1,
        )


def test_init_q_state_matches_seeded_full_q_block_without_full_init():
    config = DionParamConfig(
        has_fs_shard=True,
        use_fs_shard=True,
        fs_shard_dim=0,
        is_transposed=True,
    )
    q_layout = resolve_q_state_layout(
        2,
        11,
        config,
        tp_world_size=2,
        tp_rank=1,
        use_q_unshard=True,
        global_shape=(7, 11),
        rank_fraction=0.7,
        rank_multiple_of=1,
    )
    dist_meta = DionDistMeta(
        fs_world_size=3,
        fs_rank=1,
        fs_start_idx=2,
        fs_end_idx=4,
        param_uid=("param",),
        param_name="param",
    )
    param = torch.empty((2, 11), dtype=torch.float32)
    q_seed = 12345

    q_state = init_q_state(
        param=param,
        mixed_precision_config=DionMixedPrecisionConfig(),
        config=config,
        dist_meta=dist_meta,
        q_layout=q_layout,
        q_seed=q_seed,
        tp_world_size=2,
        tp_rank=1,
        use_q_unshard=True,
    )

    gen = torch.Generator(device="cpu")
    gen.manual_seed(q_seed)
    full_q = torch.randn(q_layout.q_global_shape, dtype=torch.float32, generator=gen)
    assert torch.equal(q_state, full_q[2:4, 3:5])


def test_init_q_state_rejects_empty_fs_and_tp_q_shard():
    config = DionParamConfig(
        has_fs_shard=True,
        use_fs_shard=True,
        fs_shard_dim=0,
        has_tp_shard=True,
        use_tp_shard=True,
        tp_shard_dim=1,
        is_transposed=True,
    )

    with pytest.raises(RuntimeError, match="DION_EMPTY_Q_SHARD"):
        resolve_q_state_layout(
            0,
            4,
            config,
            tp_world_size=4,
            tp_rank=3,
            use_q_unshard=True,
            global_shape=(2, 4),
            rank_fraction=0.25,
            rank_multiple_of=1,
        )


def test_reshard_q_along_tp_rejects_empty_rank_shard(monkeypatch):
    monkeypatch.setattr("megatron.core.optimizer.dion.ortho.dist.get_world_size", lambda group: 4)

    with pytest.raises(RuntimeError, match="DION_EMPTY_Q_SHARD"):
        reshard_q_along_tp(torch.ones((3, 1)), object(), tp_rank=2)


def test_unshard_q_batch_rejects_empty_local_rank_shard(monkeypatch):
    class DummyHandle:
        def wait(self):
            return None

    def fake_all_gather_into_tensor(output, input_tensor, group=None, async_op=False):
        del input_tensor, group, async_op
        gathered = torch.zeros((4, 1, 2, 1), dtype=output.dtype)
        gathered[0, 0, :, 0] = torch.tensor([10.0, 20.0])
        output.copy_(gathered.view(-1))
        return DummyHandle()

    class DummyOptimizer:
        _step_count = 1
        _global_rank = 2

        def __init__(self):
            self.buffers = {}

        def _cached_buffer(self, name, shape, dtype, device, *, zero=False):
            tensor = torch.empty(shape, dtype=dtype, device=device)
            if zero:
                tensor.zero_()
            self.buffers[name] = tensor
            return tensor

    monkeypatch.setattr(runtime.dist, "all_gather_into_tensor", fake_all_gather_into_tensor)
    q_local = torch.empty((2, 0), dtype=torch.float32)
    collectives = DionBatchCollectives(
        tp_q_gathers=(
            DionAxisCollective(indices=(0,), process_group=object(), world_size=4, rank=2),
        )
    )
    gen = runtime.unshard_q_batch(
        DummyOptimizer(),
        [q_local],
        [DionParamConfig(has_tp_shard=True, use_tp_shard=True, tp_shard_dim=1)],
        [DionDistMeta(param_uid=("param",), param_name="param")],
        batch_collectives=collectives,
        optimizer_states=[{"r": 1}],
        batch_cache_key=0,
    )

    with pytest.raises(RuntimeError, match="DION_EMPTY_Q_SHARD"):
        next(gen)


def test_unshard_q_batch_gathers_padded_entries_without_state(monkeypatch):
    class DummyHandle:
        def wait(self):
            return None

    def fake_all_gather_into_tensor(output, input_tensor, group=None, async_op=False):
        del group, async_op
        gathered = torch.zeros((2, 2, 2, 1), dtype=output.dtype)
        gathered[0].copy_(input_tensor.view(2, 2, 1))
        output.copy_(gathered.view(-1))
        return DummyHandle()

    class DummyOptimizer:
        _step_count = 1
        _global_rank = 0

        def _cached_buffer(self, name, shape, dtype, device, *, zero=False):
            del name
            tensor = torch.empty(shape, dtype=dtype, device=device)
            if zero:
                tensor.zero_()
            return tensor

    monkeypatch.setattr(runtime.dist, "all_gather_into_tensor", fake_all_gather_into_tensor)
    q_real = torch.tensor([[1.0], [2.0]])
    q_pad = torch.zeros_like(q_real)
    collectives = DionBatchCollectives(
        tp_q_gathers=(
            DionAxisCollective(indices=(0, 1), process_group=object(), world_size=2, rank=0),
        )
    )
    gen = runtime.unshard_q_batch(
        DummyOptimizer(),
        [q_real, q_pad],
        [DionParamConfig(has_tp_shard=True, use_tp_shard=True, tp_shard_dim=1)] * 2,
        [DionDistMeta(param_uid=("param",), param_name="param"), None],
        batch_collectives=collectives,
        optimizer_states=[{"r": 2}, None],
        batch_cache_key=0,
    )

    try:
        next(gen)
        q_full, direct_q_batch = next(gen)
    except StopIteration as exc:
        q_full, direct_q_batch = exc.value

    assert direct_q_batch is None
    assert tuple(q_full[0].shape) == (2, 2)
    assert tuple(q_full[1].shape) == (2, 2)
    assert torch.equal(q_full[0], torch.tensor([[1.0, 0.0], [2.0, 0.0]]))
    assert torch.equal(q_full[1], torch.zeros((2, 2)))


def test_batch_collectives_keep_padded_tp_entries_for_batched_shapes(monkeypatch):
    class DummyGroup:
        size = 4
        rank = 1

    group = DummyGroup()

    monkeypatch.setattr(
        "megatron.core.optimizer.distrib_dion.batches.dist.get_world_size",
        lambda process_group=None: process_group.size if process_group is not None else 1,
    )
    monkeypatch.setattr(
        "megatron.core.optimizer.distrib_dion.batches.dist.get_rank",
        lambda process_group=None: process_group.rank if process_group is not None else 0,
    )

    config = DionParamConfig(
        has_fs_shard=True,
        use_fs_shard=True,
        fs_shard_dim=1,
        has_tp_shard=True,
        use_tp_shard=True,
        tp_shard_dim=0,
    )
    dist_meta = DionDistMeta(param_uid=("param",), param_name="param")

    collectives = build_batch_collectives(
        q_tensors=[torch.empty((2, 1)), torch.empty((2, 1))],
        configs=[config, config],
        dist_metas=[dist_meta, None],
        real_batch_size=1,
        use_fs_collectives=True,
        resolve_tp_group=lambda *_args, **_kwargs: group,
        resolve_fs_group_from_meta=lambda *_args, **_kwargs: group,
    )

    assert collectives.tp_q_gathers[0].indices == (0, 1)
    assert collectives.tp_q_reshards[0].indices == (0, 1)
    assert collectives.tp_r_collectives[0].indices == (0, 1)
    assert collectives.fs_p_collectives[0].indices == (0, 1)


def test_batch_collectives_keep_padded_fs_only_entries_for_reduce_scatter(monkeypatch):
    class DummyGroup:
        size = 4
        rank = 1

    group = DummyGroup()

    monkeypatch.setattr(
        "megatron.core.optimizer.distrib_dion.batches.dist.get_world_size",
        lambda process_group=None: process_group.size if process_group is not None else 1,
    )
    monkeypatch.setattr(
        "megatron.core.optimizer.distrib_dion.batches.dist.get_rank",
        lambda process_group=None: process_group.rank if process_group is not None else 0,
    )

    config = DionParamConfig(
        has_fs_shard=True,
        use_fs_shard=True,
        fs_shard_dim=0,
    )
    dist_meta = DionDistMeta(param_uid=("param",), param_name="param")

    collectives = build_batch_collectives(
        q_tensors=[torch.empty((2, 1)), torch.empty((2, 1))],
        configs=[config, config],
        dist_metas=[dist_meta, None],
        real_batch_size=1,
        use_fs_collectives=True,
        resolve_tp_group=lambda *_args, **_kwargs: group,
        resolve_fs_group_from_meta=lambda *_args, **_kwargs: group,
    )

    assert collectives.fs_collective.indices == (0, 1)


def test_distributed_orthogonalize_uses_real_batch_size_for_local_path(monkeypatch):
    calls = []

    def fake_orthogonalize(P, rcqr_oversample=1.25, make_sketch=None):
        del rcqr_oversample, make_sketch
        calls.append(tuple(P.shape))
        return P + 1.0

    monkeypatch.setattr(ortho, "orthogonalize", fake_orthogonalize)
    optimizer = SimpleNamespace(defaults={"rcqr_oversample": 1.25}, _step_count=1)
    P_batch = torch.tensor(
        [[[0.0], [0.0], [0.0]], [[float("nan")], [7.0], [8.0]]],
        dtype=torch.float32,
    )

    P_ortho = ortho.distributed_orthogonalize(
        optimizer,
        P_batch,
        ortho_group=None,
        real_batch_size=1,
    )

    assert calls == [(1, 3, 1)]
    assert torch.equal(P_ortho[0], torch.ones((3, 1)))
    assert torch.isnan(P_ortho[1, 0, 0])
    assert torch.equal(P_ortho[1, 1:], torch.tensor([[7.0], [8.0]]))


def test_runtime_local_orthogonalize_preserves_padded_entries(monkeypatch):
    calls = []

    def fake_orthogonalize(P, rcqr_oversample=1.25, make_sketch=None):
        del rcqr_oversample, make_sketch
        calls.append(tuple(P.shape))
        return P + 2.0

    monkeypatch.setattr(runtime, "orthogonalize", fake_orthogonalize)
    optimizer = SimpleNamespace(defaults={"rcqr_oversample": 1.25})
    P_batch = torch.zeros((2, 3, 1), dtype=torch.float32)

    P_ortho = runtime._orthogonalize_real_batch(
        optimizer,
        P_batch,
        real_batch_size=1,
    )

    assert calls == [(1, 3, 1)]
    assert torch.equal(P_ortho[0], torch.full((3, 1), 2.0))
    assert torch.equal(P_ortho[1], torch.zeros((3, 1)))


def test_build_batch_key_uses_global_shape_for_uneven_local_shapes():
    config = DionParamConfig(has_fs_shard=True, use_fs_shard=True, fs_shard_dim=0)

    first_key = build_batch_key(
        (3, 8),
        config,
        torch.float32,
        global_shape=(5, 8),
    )
    second_key = build_batch_key(
        (2, 8),
        config,
        torch.float32,
        global_shape=(5, 8),
    )

    assert first_key == second_key


def test_build_batch_key_separates_explicit_row_shard_layouts():
    config = DionParamConfig(has_tp_shard=True, use_tp_shard=True, tp_shard_dim=0)

    first_key = build_batch_key(
        (2, 8),
        config,
        torch.float32,
        global_shape=(6, 8),
        row_shard_sizes=(2, 2, 2),
    )
    second_key = build_batch_key(
        (2, 8),
        config,
        torch.float32,
        global_shape=(6, 8),
        row_shard_sizes=(1, 1, 2, 2),
    )

    assert first_key != second_key


def test_build_batch_key_separates_tensor_row_layouts_without_ortho_row_layout():
    config = DionParamConfig(
        has_fs_shard=True,
        use_fs_shard=True,
        fs_shard_dim=0,
        is_transposed=True,
    )

    first_key = build_batch_key(
        (2, 8),
        config,
        torch.float32,
        global_shape=(4, 8),
        tensor_row_shard_sizes=(1, 2, 1),
    )
    second_key = build_batch_key(
        (1, 8),
        config,
        torch.float32,
        global_shape=(4, 8),
        tensor_row_shard_sizes=(1, 1, 2),
    )

    assert first_key != second_key


def test_orthogonalize_row_sizes_use_explicit_child_layout():
    config = DionParamConfig(
        has_tp_shard=True,
        use_tp_shard=True,
        tp_shard_dim=0,
        is_transposed=False,
    )
    dist_meta = DionDistMeta(
        global_shape=(6, 8),
        param_config=config,
        row_shard_sizes=(1, 1, 2, 2),
        param_uid=("qkv", "v"),
        param_name="attention.qkv::v",
    )

    row_sizes, global_rows = ortho._resolve_row_sizes_from_dist_meta(
        dist_metas=[dist_meta],
        batch_size=1,
        ortho_world_size=4,
        ortho_rank=2,
        local_rows=2,
    )

    assert row_sizes == [1, 1, 2, 2]
    assert global_rows == 6


def test_dion_bucket_layout_rejects_empty_fs_rank_payload():
    param = torch.nn.Parameter(torch.arange(6, dtype=torch.float32).view(2, 3))
    param._param_name = "param"
    param.is_dion_param = True
    bucket = SimpleNamespace(
        bucket_id=0,
        param_to_index={param: (0, 6)},
    )

    with pytest.raises(RuntimeError, match="DION_EMPTY_FS_SHARD"):
        build_dion_shard_entries(
            bucket=bucket,
            param_map={param: SimpleNamespace()},
            dion_info_by_param={
                param: {
                    "global_shape": (2, 3),
                    "fs_shard_dim": 0,
                }
            },
            fs_size=3,
            fs_rank=2,
            grad_shard_group_size=3,
            grad_rank_to_fs_rank=(0, 1, 2),
        )


@pytest.mark.parametrize(
    ("row_sizes", "row_rank"),
    [
        ((2, 3, 1), 0),
        ((2, 3, 1), 1),
        ((2, 3, 1), 2),
        ((1, 4, 2, 3), 0),
        ((1, 4, 2, 3), 1),
        ((1, 4, 2, 3), 2),
        ((1, 4, 2, 3), 3),
    ],
)
def test_make_sharded_sketch_matches_global_row_major_sketch(row_sizes, row_rank):
    sketch_seeds = (123, 456)
    sketch = ortho._make_sharded_sketch(
        sketch_seeds=sketch_seeds,
        row_sizes=row_sizes,
        row_rank=row_rank,
        r=2,
        oversample=1.0,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    k = sketch.shape[1]
    global_rows = sum(row_sizes)
    expected = torch.empty(
        (len(sketch_seeds), k, row_sizes[row_rank]),
        dtype=torch.float32,
    )
    for index, seed in enumerate(sketch_seeds):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        full_sketch = torch.empty((k, global_rows), dtype=torch.float32)
        full_sketch.normal_(mean=0.0, std=math.sqrt(1.0 / k), generator=generator)
        start = sum(row_sizes[:row_rank])
        end = start + row_sizes[row_rank]
        expected[index].copy_(full_sketch[:, start:end])

    assert torch.equal(sketch, expected)


def _make_cuda_expected_sharded_sketch(
    *,
    sketch_seeds,
    oversample,
    row_sizes,
    row_rank,
    r,
):
    local_rows = int(row_sizes[row_rank])
    prior_rows = sum(int(row_size) for row_size in row_sizes[:row_rank])
    global_rows = sum(int(row_size) for row_size in row_sizes)
    k = math.ceil(oversample * r / 128.0) * 128
    std = math.sqrt(1.0 / k)
    sketch = torch.empty(
        (len(sketch_seeds), k, local_rows),
        device="cuda",
        dtype=torch.float32,
    )
    for index, seed in enumerate(sketch_seeds):
        generator = torch.Generator(device="cuda")
        generator.manual_seed(int(seed))
        for row in range(k):
            element_start = int(row) * int(global_rows) + int(prior_rows)
            aligned_start = (element_start // 4) * 4
            prefix = element_start - aligned_start
            generator.set_offset(aligned_start)
            row_values = torch.empty(
                prefix + local_rows,
                device="cuda",
                dtype=torch.float32,
            )
            row_values.normal_(mean=0.0, std=std, generator=generator)
            sketch[index, row].copy_(row_values[prefix:])
    return sketch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_make_sharded_sketch_cuda_reuses_one_row_buffer(monkeypatch):
    sketch_seeds = (123, 456, 789)
    row_sizes = (1, 4, 2, 3)
    row_rank = 2
    r = 2
    oversample = 1.0
    expected = _make_cuda_expected_sharded_sketch(
        sketch_seeds=sketch_seeds,
        oversample=oversample,
        row_sizes=row_sizes,
        row_rank=row_rank,
        r=r,
    )

    real_empty = torch.empty
    empty_calls = []

    def counted_empty(*args, **kwargs):
        device = kwargs.get("device")
        if isinstance(device, str):
            device = torch.device(device)
        if isinstance(device, torch.device) and device.type == "cuda":
            shape = args[0] if args else kwargs["size"]
            if isinstance(shape, (tuple, list, torch.Size)):
                empty_calls.append(tuple(int(dim) for dim in shape))
            else:
                empty_calls.append((int(shape),))
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", counted_empty)

    sketch = ortho._make_sharded_sketch(
        sketch_seeds=sketch_seeds,
        row_sizes=row_sizes,
        row_rank=row_rank,
        r=r,
        oversample=oversample,
        device=torch.device("cuda"),
        dtype=torch.float32,
    )
    torch.cuda.synchronize()

    assert torch.equal(sketch, expected)
    assert empty_calls == [
        (len(sketch_seeds), sketch.shape[1], row_sizes[row_rank]),
        (row_sizes[row_rank] + 3,),
    ]
