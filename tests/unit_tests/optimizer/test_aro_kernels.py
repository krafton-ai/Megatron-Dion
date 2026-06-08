import os
import pytest
import tempfile
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from types import SimpleNamespace

from megatron.core.optimizer.aro import kernels as aro_kernels
from megatron.core.optimizer.aro.algorithm import MegatronAro
from megatron.core.optimizer.aro.kernels import (
    choose_orientation,
    compute_aro_update,
    compute_aro_updates_batched,
    orient_tensor,
    shifted_cholesky_qr,
    sinkhorn_project,
)
from megatron.core.optimizer.aro.backend import AroBackend
from megatron.core.optimizer.aro.distributed.batches import build_batch_key
from megatron.core.optimizer.aro.distributed.integration import build_aro_optimizer
from megatron.core.optimizer.aro.distributed.optimizer import DistributedAroOptimizer
from megatron.core.optimizer.aro.state import (
    init_matrix_state,
    is_aro_matrix_param,
    mark_aro_bucket_params,
    prepare_aro_params,
    state_backend_keys,
)
from megatron.core.optimizer.aro.types import AroDistMeta
from megatron.core.optimizer.aro.types import AroMixedPrecisionConfig
from megatron.core.optimizer.aro.types import AroParamConfig
from megatron.core.optimizer.aro.types import AroStepParam
from megatron.core.optimizer.matrix.backend import MatrixStateSpec
from megatron.core.optimizer.matrix.checkpoint_io import (
    build_matrix_checkpoint_metadata,
    build_persistent_param_state,
    restore_distributed_checkpoint_state,
    validate_matrix_checkpoint_metadata,
)
from megatron.core.optimizer.matrix.parameter import is_matrix_param, mark_matrix_bucket_params
from megatron.core.optimizer.matrix.runtime import route_step_params
from megatron.core.optimizer.matrix.splits.linear import linear_state_key
from megatron.core.optimizer.matrix.splits.qkv import qkv_state_key
from megatron.core.optimizer.matrix.splits.qkvg import qkvg_state_key


def _devices():
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    return devices


def _reference_sinkhorn(x, *, iters, eps):
    y = x
    target_norm = x.new_tensor(float(x.numel()), dtype=torch.float32).sqrt()
    for _ in range(iters):
        row_norm = y.float().square().sum(dim=1, keepdim=True).clamp_min(eps).sqrt()
        col_norm = y.float().square().sum(dim=0, keepdim=True).clamp_min(eps).sqrt()
        y = y / row_norm.to(y.dtype) / col_norm.to(y.dtype)
        norm = y.float().square().sum().sqrt().clamp_min(eps)
        y = y * (target_norm / norm).to(y.dtype)
    return y


@pytest.mark.parametrize("device", _devices())
def test_sinkhorn_project_matches_reference(device):
    x = torch.arange(1, 13, dtype=torch.float32, device=device).view(3, 4)

    actual = sinkhorn_project(x, iters=4, eps=1e-8)
    expected = _reference_sinkhorn(x, iters=4, eps=1e-8)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("device", _devices())
def test_shifted_cholesky_qr_matches_qr_subspace(device):
    torch.manual_seed(123)
    a = torch.randn(5, 5, dtype=torch.float32, device=device)

    q = shifted_cholesky_qr(a, eps=1e-5)

    eye = torch.eye(5, device=device)
    torch.testing.assert_close(q.T @ q, eye, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("shape,expected", [((3, 7), "normal"), ((7, 3), "transpose")])
def test_choose_orientation_uses_smaller_rotation_dim(shape, expected):
    assert choose_orientation(shape) == expected


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("shape", [(3, 5), (5, 3)])
def test_compute_aro_update_matches_reference_equations(device, shape):
    torch.manual_seed(321)
    momentum = torch.randn(shape, dtype=torch.float32, device=device)
    orientation = choose_orientation(shape)
    oriented = orient_tensor(momentum, orientation).contiguous()
    rotation = torch.eye(oriented.size(0), dtype=torch.float32, device=device)
    config = AroParamConfig(
        momentum=0.95,
        sinkhorn_iters=3,
        qr_backend="qr",
        scqr_eps=1e-8,
        update_rms_scale=0.0,
        orientation=orientation,
    )

    actual, new_rotation = compute_aro_update(
        momentum=momentum,
        rotation=rotation.clone(),
        config=config,
        orientation=orientation,
    )

    y_old = _reference_sinkhorn(rotation.T @ oriented, iters=3, eps=1e-8)
    q, r = torch.linalg.qr(oriented @ y_old.T, mode="reduced")
    signs = torch.where(
        torch.diagonal(r, 0) < 0,
        -torch.ones(q.size(1), device=device),
        torch.ones(q.size(1), device=device),
    )
    q = q * signs.unsqueeze(0)
    y_new = _reference_sinkhorn(q.T @ oriented, iters=3, eps=1e-8)
    expected = q @ y_new
    if orientation == "transpose":
        expected = expected.T

    torch.testing.assert_close(new_rotation, q, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("device", _devices())
def test_compute_aro_update_applies_rms_scale(device):
    torch.manual_seed(432)
    momentum = torch.randn(4, 6, dtype=torch.float32, device=device)
    orientation = choose_orientation(momentum.shape)
    rotation = torch.eye(4, dtype=torch.float32, device=device)
    target_rms = 2.5
    config = AroParamConfig(
        sinkhorn_iters=3,
        qr_backend="qr",
        scqr_eps=1e-8,
        update_rms_scale=target_rms,
        orientation=orientation,
    )

    actual, _ = compute_aro_update(
        momentum=momentum,
        rotation=rotation,
        config=config,
        orientation=orientation,
    )

    rms = actual.float().square().mean().sqrt()
    torch.testing.assert_close(rms, torch.tensor(target_rms, device=device), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("device", _devices())
def test_compute_aro_updates_batched_matches_single_updates(device):
    torch.manual_seed(987)
    momentums = [
        torch.randn(4, 6, dtype=torch.float32, device=device),
        torch.randn(4, 6, dtype=torch.float32, device=device),
        torch.randn(4, 6, dtype=torch.float32, device=device),
    ]
    rotations = [torch.eye(4, dtype=torch.float32, device=device) for _ in momentums]
    config = AroParamConfig(
        sinkhorn_iters=3,
        qr_backend="scqr",
        scqr_eps=1e-5,
        update_rms_scale=1.0,
        orientation="normal",
    )

    batched = compute_aro_updates_batched(
        momentums=momentums,
        rotations=[rotation.clone() for rotation in rotations],
        config=config,
        orientation="normal",
    )

    assert batched is not None
    batched_updates, batched_rotations = batched
    atol = rtol = 5e-2 if device.type == "cuda" else 2e-4
    for momentum, rotation, batch_update, batch_rotation in zip(
        momentums,
        rotations,
        batched_updates,
        batched_rotations,
    ):
        update, new_rotation = compute_aro_update(
            momentum=momentum,
            rotation=rotation.clone(),
            config=config,
            orientation="normal",
        )
        torch.testing.assert_close(batch_update, update, rtol=rtol, atol=atol)
        torch.testing.assert_close(batch_rotation, new_rotation, rtol=rtol, atol=atol)


def test_compute_aro_updates_batched_uses_single_entry_fallback_for_qr_backend():
    momentum = torch.randn(4, 6, dtype=torch.float32)
    rotation = torch.eye(4, dtype=torch.float32)
    config = AroParamConfig(qr_backend="qr")

    assert (
        compute_aro_updates_batched(
            momentums=[momentum, momentum.clone()],
            rotations=[rotation, rotation.clone()],
            config=config,
            orientation="normal",
        )
        is None
    )


def test_local_aro_step_batches_same_invariant_matrix_params(monkeypatch):
    from megatron.core.optimizer.aro import algorithm as aro_algorithm

    torch.manual_seed(1234)
    params = [
        torch.nn.Parameter(torch.randn(4, 6, dtype=torch.float32)),
        torch.nn.Parameter(torch.randn(4, 6, dtype=torch.float32)),
    ]
    for param in params:
        param.matrix_optimizer_ready = True
        param.grad = torch.randn_like(param)
    opt = MegatronAro(
        params,
        lr=1e-3,
        weight_decay=0.0,
        sinkhorn_iters=2,
        scqr_eps=1e-5,
    )
    calls = []
    real_batched = aro_algorithm.compute_aro_updates_batched

    def _record_batched(**kwargs):
        calls.append(len(kwargs["momentums"]))
        return real_batched(**kwargs)

    monkeypatch.setattr(aro_algorithm, "compute_aro_updates_batched", _record_batched)

    opt.step()

    assert calls == [2]


def test_local_aro_rejects_split_flags():
    param = torch.nn.Parameter(torch.zeros(4, 4, dtype=torch.float32))
    with pytest.raises(ValueError, match="split flags require"):
        MegatronAro([param], split_qkv=True)
    with pytest.raises(ValueError, match="split flags require"):
        MegatronAro([param], split_linear=True)
    with pytest.raises(ValueError, match="split flags require"):
        MegatronAro([{"params": [param], "split_qkv": True}])
    opt = MegatronAro([param])
    param.grad = torch.ones_like(param)
    param.matrix_optimizer_ready = True
    opt.param_groups[0]["split_linear"] = True
    with pytest.raises(ValueError, match="split flags require"):
        opt.step()


def test_distributed_aro_inner_optimizer_allows_wrapper_owned_split_flags():
    param = torch.nn.Parameter(torch.zeros(4, 4, dtype=torch.float32))
    config = SimpleNamespace(
        lr=1e-3,
        weight_decay=0.1,
        use_distributed_optimizer=True,
        aro_momentum=0.95,
        aro_base_optimizer="sinkhorn",
        aro_sinkhorn_iters=5,
        aro_qr_backend="scqr",
        aro_scqr_eps=1e-6,
        aro_update_rms_scale=1.0,
        aro_scalar_optimizer="adam",
        aro_scalar_lr_scale=1.0,
        aro_beta1=0.9,
        aro_beta2=0.95,
        aro_scalar_eps=1e-8,
        aro_split_qkv=True,
        aro_split_linear=True,
        aro_momentum_dtype=None,
        aro_rotation_dtype=None,
    )

    opt = build_aro_optimizer(config=config, param_groups=[{"params": [param]}])

    assert isinstance(opt, MegatronAro)
    assert opt.defaults["split_qkv"] is False
    assert opt.defaults["split_linear"] is False


def test_init_matrix_state_preserves_orientation_and_rotation_rows():
    param = torch.zeros(2, 6, dtype=torch.float32)
    meta = AroDistMeta(
        shape=(2, 6),
        local_shape=(2, 6),
        global_shape=(4, 6),
        fs_shard_dim=0,
        fs_start_idx=2,
        fs_end_idx=4,
        fs_world_size=2,
        fs_rank=1,
        is_matrix_param=True,
        is_aro_param=True,
        orientation="normal",
    )
    state = {}

    init_matrix_state(param, state, dist_meta=meta)

    assert state["local_shape"] == (2, 6)
    assert state["global_shape"] == (4, 6)
    assert state["orientation"] == "normal"
    assert tuple(state["rotation"].shape) == (2, 4)
    expected = torch.zeros(2, 4)
    expected[0, 2] = 1.0
    expected[1, 3] = 1.0
    torch.testing.assert_close(state["rotation"], expected)


def test_init_matrix_state_uses_tp_row_offset_for_rotation_rows():
    param = torch.zeros(2, 6, dtype=torch.float32)
    meta = AroDistMeta(
        shape=(2, 6),
        local_shape=(2, 6),
        global_shape=(4, 6),
        tp_shard_dim=0,
        tp_world_size=2,
        tp_rank=1,
        is_matrix_param=True,
        is_aro_param=True,
        orientation="normal",
    )
    state = {}

    init_matrix_state(param, state, dist_meta=meta)

    assert tuple(state["rotation"].shape) == (2, 4)
    expected = torch.zeros(2, 4)
    expected[0, 2] = 1.0
    expected[1, 3] = 1.0
    torch.testing.assert_close(state["rotation"], expected)


def test_init_matrix_state_projects_fs_tp_orientation_for_rotation_rows():
    normal_param = torch.zeros(2, 3, dtype=torch.float32)
    normal_meta = AroDistMeta(
        shape=(2, 3),
        local_shape=(2, 3),
        global_shape=(4, 6),
        fs_shard_dim=0,
        fs_start_idx=2,
        fs_end_idx=4,
        fs_world_size=2,
        fs_rank=1,
        tp_shard_dim=1,
        tp_world_size=2,
        tp_rank=1,
        is_matrix_param=True,
        is_aro_param=True,
        orientation="normal",
    )
    normal_state = {}

    init_matrix_state(normal_param, normal_state, dist_meta=normal_meta)

    normal_expected = torch.zeros(2, 4)
    normal_expected[0, 2] = 1.0
    normal_expected[1, 3] = 1.0
    torch.testing.assert_close(normal_state["rotation"], normal_expected)

    transpose_param = torch.zeros(3, 2, dtype=torch.float32)
    transpose_meta = AroDistMeta(
        shape=(3, 2),
        local_shape=(3, 2),
        global_shape=(6, 4),
        fs_shard_dim=0,
        fs_start_idx=3,
        fs_end_idx=6,
        fs_world_size=2,
        fs_rank=1,
        tp_shard_dim=1,
        tp_world_size=2,
        tp_rank=1,
        is_matrix_param=True,
        is_aro_param=True,
        orientation="transpose",
    )
    transpose_state = {}

    init_matrix_state(transpose_param, transpose_state, dist_meta=transpose_meta)

    transpose_expected = torch.zeros(2, 4)
    transpose_expected[0, 2] = 1.0
    transpose_expected[1, 3] = 1.0
    torch.testing.assert_close(transpose_state["rotation"], transpose_expected)


def _empty_aro_adapter():
    adapter = object.__new__(DistributedAroOptimizer)
    adapter.config = SimpleNamespace()
    return adapter


def test_aro_qkv_child_tp_row_shard_uses_child_row_offset_for_rotation_rows():
    adapter = _empty_aro_adapter()
    parent_meta = AroDistMeta(
        shape=(4, 16),
        local_shape=(4, 16),
        global_shape=(12, 16),
        tp_shard_dim=0,
        tp_world_size=3,
        tp_rank=1,
        is_matrix_param=True,
        is_aro_param=True,
        orientation="normal",
        param_uid=("qkv_parent",),
        param_name="layer.linear_qkv.weight",
    )

    child_meta = adapter._child_meta(
        parent_meta,
        child_kind="q",
        child_name="layer.linear_qkv.weight::q",
        child_uid=("qkv_parent", "q"),
        child_shape=(2, 16),
        child_global_shape=(6, 16),
        split_kind="qkv",
        optim_group={},
        qkv_shapes=(2, 1, 1),
    )
    assert child_meta.row_shard_start_idx == 2
    assert child_meta.row_shard_end_idx == 4

    state = {}
    init_matrix_state(torch.zeros(2, 16), state, dist_meta=child_meta)

    expected = torch.zeros(2, 6)
    expected[0, 2] = 1.0
    expected[1, 3] = 1.0
    torch.testing.assert_close(state["rotation"], expected)


def test_aro_qkvg_child_tp_row_shard_uses_child_row_offset_for_rotation_rows():
    adapter = _empty_aro_adapter()
    parent_meta = AroDistMeta(
        shape=(4, 16),
        local_shape=(4, 16),
        global_shape=(12, 16),
        tp_shard_dim=0,
        tp_world_size=3,
        tp_rank=1,
        is_matrix_param=True,
        is_aro_param=True,
        orientation="normal",
        param_uid=("qkvg_parent",),
        param_name="layer.linear_qkv.weight",
    )

    child_meta = adapter._child_meta(
        parent_meta,
        child_kind="q",
        child_name="layer.linear_qkv.weight::q",
        child_uid=("qkvg_parent", "q"),
        child_shape=(2, 16),
        child_global_shape=(4, 16),
        split_kind="qkvg",
        optim_group={},
        qkvg_shapes=(2, 2, 1, 1),
    )
    assert child_meta.row_shard_start_idx == 2
    assert child_meta.row_shard_end_idx == 4

    state = {}
    init_matrix_state(torch.zeros(2, 16), state, dist_meta=child_meta)

    expected = torch.zeros(2, 4)
    expected[0, 2] = 1.0
    expected[1, 3] = 1.0
    torch.testing.assert_close(state["rotation"], expected)


def test_aro_linear_child_row_shards_use_child_row_offset_for_rotation_rows():
    adapter = _empty_aro_adapter()
    parent_meta = AroDistMeta(
        shape=(2, 16),
        local_shape=(2, 16),
        global_shape=(8, 16),
        fs_start_idx=2,
        fs_end_idx=4,
        fs_shard_dim=0,
        fs_world_size=4,
        fs_rank=1,
        is_matrix_param=True,
        is_aro_param=True,
        orientation="normal",
        param_uid=("linear_parent",),
        param_name="layer.linear_fc1.weight",
    )

    child_meta = adapter._child_meta(
        parent_meta,
        child_kind="gate",
        child_name="layer.linear_fc1.weight::gate",
        child_uid=("linear_parent", "gate"),
        child_shape=(2, 16),
        child_global_shape=(4, 16),
        split_kind="linear",
        optim_group={},
        linear_rows=(4, 4),
    )
    assert child_meta.fs_start_idx == 2
    assert child_meta.fs_end_idx == 4
    assert child_meta.row_shard_start_idx == 2
    assert child_meta.row_shard_end_idx == 4

    state = {}
    init_matrix_state(torch.zeros(2, 16), state, dist_meta=child_meta)

    expected = torch.zeros(2, 4)
    expected[0, 2] = 1.0
    expected[1, 3] = 1.0
    torch.testing.assert_close(state["rotation"], expected)


def test_init_matrix_state_uses_per_expert_global_shape():
    param = torch.zeros(2, 6, dtype=torch.float32)
    meta = AroDistMeta(
        shape=(2, 6),
        local_shape=(2, 6),
        global_shape=(8, 6),
        per_expert_global_shape=(4, 6),
        fs_shard_dim=0,
        fs_start_idx=0,
        fs_end_idx=2,
        fs_world_size=2,
        fs_rank=0,
        is_matrix_param=True,
        is_aro_param=True,
        orientation="normal",
    )
    state = {}

    init_matrix_state(param, state, dist_meta=meta)

    assert state["global_shape"] == (4, 6)
    assert state["per_expert_global_shape"] == (4, 6)
    assert tuple(state["rotation"].shape) == (2, 4)


def test_aro_backend_state_spec_matches_checkpoint_invariant():
    spec = AroBackend().state_spec()

    assert spec.backend == "aro"
    assert spec.state_keys == state_backend_keys()
    assert spec.state_keys == (
        "momentum",
        "rotation",
        "orientation",
        "local_shape",
        "global_shape",
        "per_expert_global_shape",
        "qkv_split_shapes",
        "qkvg_split_shapes",
        "linear_split_rows",
    )


def test_aro_build_batch_key_uses_group_ranks_not_process_group_identity(monkeypatch):
    class FakeGroup:
        pass

    fs_group = FakeGroup()
    tp_group = FakeGroup()
    monkeypatch.setattr(
        "megatron.core.optimizer.aro.distributed.batches.dist.is_available",
        lambda: True,
    )
    monkeypatch.setattr(
        "megatron.core.optimizer.aro.distributed.batches.dist.is_initialized",
        lambda: True,
    )
    monkeypatch.setattr(
        "megatron.core.optimizer.aro.distributed.batches.dist.get_process_group_ranks",
        lambda group: (0, 2) if group is fs_group else (0, 1),
    )
    step_param = type(
        "StepParam",
        (),
        {
            "param": torch.zeros(2, 2),
            "config": AroParamConfig(),
            "dist_meta": AroDistMeta(
                shape=(2, 2),
                global_shape=(2, 2),
                fs_shard_dim=0,
                fs_world_size=2,
                tp_shard_dim=1,
                tp_world_size=2,
                fs_group=fs_group,
                tp_group=tp_group,
                is_matrix_param=True,
                is_aro_param=True,
            ),
        },
    )()

    key = build_batch_key(step_param)

    assert key[-2:] == ((0, 2), (0, 1))
    assert id(fs_group) not in key
    assert id(tp_group) not in key


def test_aro_build_batch_key_separates_topology_domains(monkeypatch):
    class FakeGroup:
        pass

    fs_group = FakeGroup()
    tp_group = FakeGroup()
    other_fs_group = FakeGroup()
    group_ranks = {
        fs_group: (0, 2),
        tp_group: (0, 1),
        other_fs_group: (1, 3),
    }
    monkeypatch.setattr(
        "megatron.core.optimizer.aro.distributed.batches.dist.is_available",
        lambda: True,
    )
    monkeypatch.setattr(
        "megatron.core.optimizer.aro.distributed.batches.dist.is_initialized",
        lambda: True,
    )
    monkeypatch.setattr(
        "megatron.core.optimizer.aro.distributed.batches.dist.get_process_group_ranks",
        lambda group: group_ranks[group],
    )

    def make_step(**overrides):
        values = {
            "fs_shard_dim": 0,
            "fs_world_size": 2,
            "fs_group": fs_group,
            "tp_shard_dim": 1,
            "tp_world_size": 2,
            "tp_group": tp_group,
            "orientation": "normal",
        }
        values.update(overrides)
        return SimpleNamespace(
            param=torch.zeros(2, 2),
            config=AroParamConfig(orientation=values["orientation"]),
            dist_meta=AroDistMeta(
                shape=(2, 2),
                global_shape=(4, 4),
                fs_shard_dim=values["fs_shard_dim"],
                fs_world_size=values["fs_world_size"],
                fs_group=values["fs_group"],
                tp_shard_dim=values["tp_shard_dim"],
                tp_world_size=values["tp_world_size"],
                tp_group=values["tp_group"],
                orientation=values["orientation"],
                is_matrix_param=True,
                is_aro_param=True,
            ),
        )

    base_key = build_batch_key(make_step())

    assert build_batch_key(make_step(fs_world_size=4)) != base_key
    assert build_batch_key(make_step(tp_world_size=4)) != base_key
    assert build_batch_key(make_step(fs_shard_dim=1)) != base_key
    assert build_batch_key(make_step(tp_shard_dim=0)) != base_key
    assert build_batch_key(make_step(orientation="transpose")) != base_key
    assert build_batch_key(make_step(fs_group=other_fs_group)) != base_key


def test_aro_embedding_and_lm_head_params_use_full_model_matrix_path():
    embedding = torch.nn.Parameter(torch.zeros(4, 4))
    embedding.matrix_optimizer_ready = True
    embedding.is_embedding_or_output_parameter = True

    lm_head = torch.nn.Parameter(torch.zeros(4, 4))
    lm_head.matrix_optimizer_ready = True
    lm_head.is_lm_head_parameter = True

    assert not is_matrix_param(embedding)
    assert not is_matrix_param(lm_head)
    assert is_aro_matrix_param(embedding)
    assert is_aro_matrix_param(lm_head)


def test_prepare_aro_params_includes_embedding_and_lm_head():
    module = torch.nn.Module()
    module.embedding = torch.nn.Parameter(torch.zeros(4, 4))
    module.embedding.is_embedding_or_output_parameter = True
    module.lm_head = torch.nn.Parameter(torch.zeros(4, 4))
    module.lm_head.is_lm_head_parameter = True
    module.linear_qkv = torch.nn.Linear(2, 4, bias=False)

    prepare_aro_params(module)
    assert module.embedding.matrix_optimizer_ready
    assert module.embedding.use_aro is True
    assert module.lm_head.matrix_optimizer_ready
    assert module.lm_head.use_aro is True
    assert module.linear_qkv.weight.use_aro is True
    assert module.linear_qkv.weight.is_qkv is True


def test_aro_bucket_marking_includes_embedding_and_lm_head():
    embedding = torch.nn.Parameter(torch.zeros(4, 4))
    embedding.matrix_optimizer_ready = True
    embedding.is_embedding_or_output_parameter = True

    lm_head = torch.nn.Parameter(torch.zeros(4, 4))
    lm_head.matrix_optimizer_ready = True
    lm_head.is_lm_head_parameter = True

    linear = torch.nn.Parameter(torch.zeros(4, 4))
    linear.matrix_optimizer_ready = True

    param_map = {embedding: {}, lm_head: {}, linear: {}}
    param_to_name = {
        embedding: "embedding.word_embeddings.weight",
        lm_head: "decoder.output_layer.weight",
        linear: "decoder.layers.0.mlp.linear_fc2.weight",
    }

    default_count, default_info = mark_matrix_bucket_params(
        param_map=param_map,
        param_to_name=param_to_name,
        fs_size=1,
    )
    assert default_count == 1
    assert embedding not in default_info
    assert lm_head not in default_info
    assert linear in default_info

    aro_count, aro_info = mark_aro_bucket_params(
        param_map=param_map,
        param_to_name=param_to_name,
        fs_size=1,
    )
    assert aro_count == 3
    assert embedding in aro_info
    assert lm_head in aro_info
    assert linear in aro_info


def test_aro_route_step_params_replaces_split_parent_with_children():
    parent = torch.nn.Parameter(torch.ones(4, 2))
    parent.grad = torch.full_like(parent, 2.0)
    child = torch.nn.Parameter(torch.ones(2, 2))
    child_grad = torch.full_like(child, 3.0)
    group = {"params": [parent]}
    state_by_param = {parent: {}, child: {"momentum": torch.zeros_like(child)}}
    parent_meta = AroDistMeta(
        shape=(4, 2),
        global_shape=(4, 2),
        is_matrix_param=True,
        is_aro_param=True,
        param_uid=("parent",),
        param_name="linear_qkv.weight",
    )
    child_meta = AroDistMeta(
        shape=(2, 2),
        global_shape=(2, 2),
        is_matrix_param=True,
        is_aro_param=True,
        parent_param_uid=("parent",),
        param_uid=("parent", "q"),
        param_name="linear_qkv.weight::q",
    )
    child_step_param = AroStepParam(
        param=child,
        grad=child_grad,
        optimizer_state=state_by_param[child],
        optim_group=group,
        config=AroParamConfig(),
        dist_meta=child_meta,
    )
    use_matrix_calls = []
    synced_params = []

    batches, scalar_params = route_step_params(
        param_groups=[group],
        dist_metas={parent: parent_meta},
        get_step_param_grad=lambda param: param.grad,
        ensure_optimizer_state=lambda param, _group: state_by_param[param],
        require_param_config=lambda _param, _meta: AroParamConfig(),
        use_matrix=lambda **kwargs: use_matrix_calls.append(kwargs["param"]) or True,
        split_children=lambda **kwargs: [child_step_param] if kwargs["param"] is parent else None,
        sync_state=lambda matrix_params: synced_params.extend(matrix_params),
        build_batches=lambda matrix_params: [tuple(matrix_params)],
    )

    assert scalar_params == []
    assert use_matrix_calls == []
    assert synced_params == [child_step_param]
    assert batches == [(child_step_param,)]


def test_aro_route_step_params_separates_matrix_and_scalar_paths():
    matrix_param = torch.nn.Parameter(torch.ones(2, 2))
    scalar_param = torch.nn.Parameter(torch.ones(2))
    matrix_param.grad = torch.full_like(matrix_param, 2.0)
    scalar_param.grad = torch.full_like(scalar_param, 3.0)
    group = {"params": [scalar_param, matrix_param]}
    state_by_param = {matrix_param: {}, scalar_param: {}}
    matrix_meta = AroDistMeta(
        shape=(2, 2),
        global_shape=(2, 2),
        is_matrix_param=True,
        is_aro_param=True,
        param_uid=("matrix",),
        param_name="linear.weight",
    )
    synced_params = []

    batches, scalar_params = route_step_params(
        param_groups=[group],
        dist_metas={matrix_param: matrix_meta},
        get_step_param_grad=lambda param: param.grad,
        ensure_optimizer_state=lambda param, _group: state_by_param[param],
        require_param_config=lambda _param, _meta: AroParamConfig(),
        use_matrix=lambda **kwargs: bool(
            getattr(kwargs["dist_meta"], "is_aro_param", False)
        ),
        split_children=lambda **_kwargs: None,
        sync_state=lambda matrix_params: synced_params.extend(matrix_params),
        build_batches=lambda matrix_params: [tuple(matrix_params)],
    )

    assert len(scalar_params) == 1
    assert scalar_params[0].param is scalar_param
    assert len(batches) == 1
    assert len(batches[0]) == 1
    matrix_step_param = batches[0][0]
    assert matrix_step_param.param is matrix_param
    assert matrix_step_param.dist_meta is matrix_meta
    assert synced_params == [matrix_step_param]


def test_aro_distributed_scalar_params_use_configured_lion_optimizer():
    param = torch.nn.Parameter(torch.tensor([1.0, -1.0], dtype=torch.float32))
    grad = torch.tensor([2.0, -3.0], dtype=torch.float32)
    state = {}
    adapter = object.__new__(DistributedAroOptimizer)
    adapter.config = SimpleNamespace(
        aro_scalar_optimizer="lion",
        aro_beta1=0.5,
        aro_beta2=0.25,
        aro_scalar_lr_scale=0.25,
        weight_decay=0.0,
    )
    adapter.optimizer = SimpleNamespace(state={param: state})
    step_param = SimpleNamespace(
        param=param,
        grad=grad,
        optimizer_state=state,
        optim_group={"lr": 0.2, "weight_decay": 0.0},
    )

    adapter._apply_scalar_params([step_param])

    torch.testing.assert_close(param.detach(), torch.tensor([0.95, -0.95]))
    torch.testing.assert_close(state["exp_avg"], grad * 0.75)
    assert "exp_avg_sq" not in state


def test_aro_row_col_groups_project_shards_through_orientation():
    adapter = object.__new__(DistributedAroOptimizer)
    fs_group = object()
    tp_group = object()

    normal_meta = AroDistMeta(
        shape=(2, 3),
        global_shape=(4, 6),
        fs_shard_dim=0,
        fs_world_size=2,
        fs_group=fs_group,
        tp_shard_dim=1,
        tp_world_size=2,
        tp_group=tp_group,
        orientation="normal",
        is_matrix_param=True,
        is_aro_param=True,
    )
    row_groups, col_groups = adapter._row_col_groups(normal_meta)

    assert row_groups == (fs_group,)
    assert col_groups == (tp_group,)

    transpose_meta = AroDistMeta(
        shape=(3, 2),
        global_shape=(6, 4),
        fs_shard_dim=0,
        fs_world_size=2,
        fs_group=fs_group,
        tp_shard_dim=1,
        tp_world_size=2,
        tp_group=tp_group,
        orientation="transpose",
        is_matrix_param=True,
        is_aro_param=True,
    )
    row_groups, col_groups = adapter._row_col_groups(transpose_meta)

    assert row_groups == (tp_group,)
    assert col_groups == (fs_group,)


def test_aro_row_col_groups_support_two_axes_on_same_oriented_dimension():
    adapter = object.__new__(DistributedAroOptimizer)
    fs_group = object()
    tp_group = object()

    normal_meta = AroDistMeta(
        shape=(2, 3),
        global_shape=(4, 6),
        fs_shard_dim=0,
        fs_world_size=2,
        fs_group=fs_group,
        tp_shard_dim=0,
        tp_world_size=2,
        tp_group=tp_group,
        orientation="normal",
        is_matrix_param=True,
        is_aro_param=True,
    )
    row_groups, col_groups = adapter._row_col_groups(normal_meta)

    assert row_groups == (fs_group, tp_group)
    assert col_groups == ()

    transpose_meta = AroDistMeta(
        shape=(3, 2),
        global_shape=(6, 4),
        fs_shard_dim=0,
        fs_world_size=2,
        fs_group=fs_group,
        tp_shard_dim=0,
        tp_world_size=2,
        tp_group=tp_group,
        orientation="transpose",
        is_matrix_param=True,
        is_aro_param=True,
    )
    row_groups, col_groups = adapter._row_col_groups(transpose_meta)

    assert row_groups == ()
    assert col_groups == (fs_group, tp_group)


def test_aro_collectives_deduplicate_equivalent_rank_groups(monkeypatch):
    class FakeGroup:
        pass

    first_group = FakeGroup()
    second_group = FakeGroup()
    calls = []

    monkeypatch.setattr(aro_kernels, "_group_size", lambda group: 2)
    monkeypatch.setattr(aro_kernels, "_group_key", lambda group: (0, 1))
    monkeypatch.setattr(
        aro_kernels.dist,
        "all_reduce",
        lambda tensor, op, group: calls.append(group),
    )

    sinkhorn_project(
        torch.arange(1, 7, dtype=torch.float32).view(2, 3),
        iters=3,
        column_groups=(first_group, second_group),
    )

    assert len(calls) == 7
    assert calls == [first_group] * 7


def test_aro_collective_helpers_deduplicate_combined_equivalent_groups(monkeypatch):
    class FakeGroup:
        pass

    first_group = FakeGroup()
    second_group = FakeGroup()
    calls = []

    monkeypatch.setattr(aro_kernels, "_group_size", lambda group: 2)
    monkeypatch.setattr(aro_kernels, "_group_key", lambda group: (0, 1))
    monkeypatch.setattr(
        aro_kernels.dist,
        "all_reduce",
        lambda tensor, op, group: calls.append(group),
    )

    stats = torch.ones(2)
    aro_kernels._all_reduce_sum_(stats, (first_group, second_group, first_group))

    assert calls == [first_group]


def test_compute_aro_update_participates_in_rms_collective_for_empty_row_shard(monkeypatch):
    class FakeGroup:
        pass

    group = FakeGroup()
    all_reduce_shapes = []

    monkeypatch.setattr(aro_kernels, "_group_size", lambda group: 2)
    monkeypatch.setattr(aro_kernels, "_group_key", lambda group: (0, 1))
    monkeypatch.setattr(
        aro_kernels.dist,
        "all_reduce",
        lambda tensor, op, group: all_reduce_shapes.append(tuple(tensor.shape)),
    )

    update, new_rotation = compute_aro_update(
        momentum=torch.empty(0, 6, dtype=torch.float32),
        rotation=torch.empty(0, 4, dtype=torch.float32),
        config=AroParamConfig(sinkhorn_iters=1, qr_backend="scqr", update_rms_scale=1.0),
        orientation="normal",
        row_groups=(group,),
    )

    assert tuple(update.shape) == (0, 6)
    assert tuple(new_rotation.shape) == (0, 4)
    assert (2,) in all_reduce_shapes


def test_compute_aro_update_scqr_normal_path_does_not_gather_rows(monkeypatch):
    class FakeGroup:
        pass

    row_group = FakeGroup()
    col_group = FakeGroup()
    all_reduce_calls = []

    monkeypatch.setattr(aro_kernels, "_group_size", lambda group: 2)
    monkeypatch.setattr(
        aro_kernels,
        "_group_key",
        lambda group: (0, 1) if group is row_group else (0, 2),
    )
    monkeypatch.setattr(
        aro_kernels.dist,
        "all_reduce",
        lambda tensor, op, group: all_reduce_calls.append(group),
    )

    def _fail_all_gather(*_args, **_kwargs):
        raise AssertionError("ARO SCQR normal path should not gather rows")

    monkeypatch.setattr(aro_kernels.dist, "all_gather", _fail_all_gather)

    momentum = torch.eye(2, 2, dtype=torch.float32)
    rotation = torch.eye(2, dtype=torch.float32)
    update, new_rotation = compute_aro_update(
        momentum=momentum,
        rotation=rotation,
        config=AroParamConfig(sinkhorn_iters=1, qr_backend="scqr", update_rms_scale=1.0),
        orientation="normal",
        row_groups=(row_group,),
        column_groups=(col_group,),
    )

    assert tuple(update.shape) == (2, 2)
    assert tuple(new_rotation.shape) == (2, 2)
    assert row_group in all_reduce_calls
    assert col_group in all_reduce_calls


def test_aro_scqr_normal_path_does_not_gather_rows(monkeypatch):
    class FakeGroup:
        pass

    group = FakeGroup()
    all_reduce_calls = []

    monkeypatch.setattr(aro_kernels, "_group_size", lambda group: 2)
    monkeypatch.setattr(aro_kernels, "_group_key", lambda group: (0, 1))
    monkeypatch.setattr(
        aro_kernels.dist,
        "all_reduce",
        lambda tensor, op, group: all_reduce_calls.append(group),
    )

    def _fail_all_gather(*_args, **_kwargs):
        raise AssertionError("SCQR normal path should not gather rows")

    monkeypatch.setattr(aro_kernels.dist, "all_gather", _fail_all_gather)

    q = shifted_cholesky_qr(
        torch.eye(2, dtype=torch.float32),
        eps=1e-5,
        row_groups=(group,),
        qr_backend="scqr",
    )

    assert all_reduce_calls == [group]
    torch.testing.assert_close(q, torch.eye(2, dtype=torch.float32), rtol=1e-4, atol=1e-4)


def test_aro_qr_fallback_gathers_every_row_group(monkeypatch):
    class FakeGroup:
        pass

    first_group = FakeGroup()
    second_group = FakeGroup()
    gather_calls = []
    all_reduce_calls = []

    monkeypatch.setattr(aro_kernels, "_group_size", lambda group: 2)
    monkeypatch.setattr(
        aro_kernels,
        "_group_key",
        lambda group: (0, 1) if group is first_group else (0, 2),
    )
    monkeypatch.setattr(aro_kernels.dist, "get_rank", lambda group: 0)
    monkeypatch.setattr(
        aro_kernels.dist,
        "all_reduce",
        lambda tensor, op, group: all_reduce_calls.append(group),
    )
    monkeypatch.setattr(
        aro_kernels,
        "_deterministic_qr",
        lambda tensor: tensor,
    )

    def _fake_gather_rows(tensor, group):
        gather_calls.append(group)
        rows = int(tensor.size(0))
        return torch.cat((tensor, tensor), dim=0), (rows, rows)

    monkeypatch.setattr(aro_kernels, "_gather_rows", _fake_gather_rows)

    q = shifted_cholesky_qr(
        torch.ones(1, 2),
        eps=1e-5,
        row_groups=(first_group, second_group),
        qr_backend="qr",
    )

    assert gather_calls == [first_group, second_group]
    assert all_reduce_calls == []
    assert tuple(q.shape) == (1, 2)


def test_aro_checkpoint_metadata_carries_backend_invariant():
    spec = AroBackend().state_spec()
    topology = {
        "data_parallel": (0, 1),
        "fs": (0, 1),
        "tp": (),
        "rp": (),
        "state_replica": (),
    }
    metadata = build_matrix_checkpoint_metadata(
        dp_size=2,
        fs_size=2,
        tp_size=1,
        rp_size=1,
        state_replica_size=1,
        requested_type="matrix_fs_rank_state",
        topology_signature=topology,
        backend_state_spec=spec,
    )

    validate_matrix_checkpoint_metadata(
        metadata,
        dp_size=2,
        fs_size=2,
        tp_size=1,
        rp_size=1,
        state_replica_size=1,
        topology_signature=topology,
        backend_state_spec=spec,
    )
    with pytest.raises(RuntimeError, match="backend checkpoint restore"):
        validate_matrix_checkpoint_metadata(
            metadata,
            dp_size=2,
            fs_size=2,
            tp_size=1,
            rp_size=1,
            state_replica_size=1,
            topology_signature=topology,
            backend_state_spec=MatrixStateSpec(backend="other", state_keys=()),
        )


def test_aro_restored_state_validation_rejects_orientation_mismatch():
    param = torch.nn.Parameter(torch.zeros(2, 6))
    meta = AroDistMeta(
        shape=(2, 6),
        local_shape=(2, 6),
        global_shape=(4, 6),
        fs_shard_dim=0,
        fs_world_size=2,
        is_matrix_param=True,
        is_aro_param=True,
        orientation="normal",
        param_uid=("param", (4, 6)),
    )
    optimizer = type("Optimizer", (), {"state": {param: {
        "momentum": torch.zeros(2, 6),
        "rotation": torch.zeros(2, 4),
        "local_shape": (2, 6),
        "global_shape": (4, 6),
        "orientation": "transpose",
    }}})()
    adapter = object.__new__(DistributedAroOptimizer)
    adapter.dist_metas = {param: meta}
    adapter.optimizer = optimizer

    with pytest.raises(RuntimeError, match="ARO_CHECKPOINT_ORIENTATION_MISMATCH"):
        adapter._validate_restored_aro_state()


def test_aro_restored_state_validation_rejects_rotation_shape_mismatch():
    param = torch.nn.Parameter(torch.zeros(2, 6))
    meta = AroDistMeta(
        shape=(2, 6),
        local_shape=(2, 6),
        global_shape=(4, 6),
        fs_shard_dim=0,
        fs_world_size=2,
        is_matrix_param=True,
        is_aro_param=True,
        orientation="normal",
        param_uid=("param", (4, 6)),
    )
    optimizer = type("Optimizer", (), {"state": {param: {
        "momentum": torch.zeros(2, 6),
        "rotation": torch.zeros(4, 4),
        "local_shape": (2, 6),
        "global_shape": (4, 6),
        "orientation": "normal",
    }}})()
    adapter = object.__new__(DistributedAroOptimizer)
    adapter.dist_metas = {param: meta}
    adapter.optimizer = optimizer

    with pytest.raises(RuntimeError, match="ARO_CHECKPOINT_STATE_SHAPE_MISMATCH"):
        adapter._validate_restored_aro_state()


def test_aro_checkpoint_payload_round_trip_restores_matrix_state():
    param_key = ("param", (2, 6))
    saved_param = torch.nn.Parameter(torch.arange(12, dtype=torch.float32).view(2, 6))
    saved_momentum = torch.full((2, 6), 3.0)
    saved_rotation = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32)
    saved_state = {
        "momentum": saved_momentum,
        "rotation": saved_rotation,
        "local_shape": (2, 6),
        "global_shape": (2, 6),
        "orientation": "normal",
    }
    payload = build_persistent_param_state(
        [{"params": [saved_param]}],
        {saved_param: saved_state},
        lambda param: param_key,
    )

    restored_param = torch.nn.Parameter(torch.zeros(2, 6))
    restored_state = {
        "momentum": torch.zeros(2, 6),
        "rotation": torch.zeros(2, 2),
        "local_shape": (2, 6),
        "global_shape": (2, 6),
        "orientation": "normal",
    }
    optimizer_state = {restored_param: restored_state}

    summary = restore_distributed_checkpoint_state(
        param_state_payload=payload,
        param_groups=[{"params": [restored_param]}],
        optimizer_state=optimizer_state,
        get_param_key=lambda param: param_key,
    )

    assert summary == {"restored": 1, "unnamed": 0, "no_payload_entry": 0}
    torch.testing.assert_close(restored_param, saved_param)
    torch.testing.assert_close(optimizer_state[restored_param]["momentum"], saved_momentum)
    torch.testing.assert_close(optimizer_state[restored_param]["rotation"], saved_rotation)
    assert optimizer_state[restored_param]["orientation"] == "normal"

    meta = AroDistMeta(
        shape=(2, 6),
        local_shape=(2, 6),
        global_shape=(2, 6),
        is_matrix_param=True,
        is_aro_param=True,
        orientation="normal",
        param_uid=param_key,
    )
    adapter = object.__new__(DistributedAroOptimizer)
    adapter.dist_metas = {restored_param: meta}
    adapter.optimizer = type("Optimizer", (), {"state": optimizer_state})()
    adapter._validate_restored_aro_state()


def test_aro_checkpoint_payload_round_trip_preserves_qkv_split_state():
    param_key = ("linear_qkv", (4, 3))
    saved_param = torch.nn.Parameter(torch.arange(12, dtype=torch.float32).view(4, 3))
    saved_state = {
        "momentum": torch.ones(4, 3),
        "local_shape": (4, 3),
        "global_shape": (4, 3),
        "orientation": "transpose",
        "qkv_split_qkv": True,
        "qkv_split_shapes": (2, 1, 1),
        "qkv_q_orientation": "normal",
        "qkv_k_orientation": "normal",
        "qkv_v_orientation": "normal",
        qkv_state_key("rotation", "q"): torch.eye(2),
        qkv_state_key("rotation", "k"): torch.eye(1),
        qkv_state_key("rotation", "v"): torch.eye(1),
    }
    payload = build_persistent_param_state(
        [{"params": [saved_param]}],
        {saved_param: saved_state},
        lambda param: param_key,
    )

    restored_param = torch.nn.Parameter(torch.zeros(4, 3))
    optimizer_state = {restored_param: {"momentum": torch.zeros(4, 3)}}
    summary = restore_distributed_checkpoint_state(
        param_state_payload=payload,
        param_groups=[{"params": [restored_param]}],
        optimizer_state=optimizer_state,
        get_param_key=lambda param: param_key,
    )

    assert summary == {"restored": 1, "unnamed": 0, "no_payload_entry": 0}
    restored_state = optimizer_state[restored_param]
    assert restored_state["qkv_split_shapes"] == (2, 1, 1)
    assert restored_state["qkv_q_orientation"] == "normal"
    torch.testing.assert_close(restored_state[qkv_state_key("rotation", "q")], torch.eye(2))
    torch.testing.assert_close(restored_state[qkv_state_key("rotation", "k")], torch.eye(1))
    torch.testing.assert_close(restored_state[qkv_state_key("rotation", "v")], torch.eye(1))

    meta = AroDistMeta(
        shape=(4, 3),
        local_shape=(4, 3),
        global_shape=(4, 3),
        is_matrix_param=True,
        is_aro_param=True,
        orientation="transpose",
        param_uid=param_key,
    )
    adapter = object.__new__(DistributedAroOptimizer)
    adapter.dist_metas = {restored_param: meta}
    adapter.optimizer = type("Optimizer", (), {"state": optimizer_state})()
    adapter._validate_restored_aro_state()


def test_aro_checkpoint_payload_round_trip_preserves_qkvg_split_state():
    param_key = ("linear_qkv", (6, 3))
    saved_param = torch.nn.Parameter(torch.arange(18, dtype=torch.float32).view(6, 3))
    saved_state = {
        "momentum": torch.ones(6, 3),
        "local_shape": (6, 3),
        "global_shape": (6, 3),
        "orientation": "transpose",
        "qkvg_split_qkvg": True,
        "qkvg_split_shapes": (2, 2, 1, 1),
        "qkvg_q_orientation": "normal",
        "qkvg_gate_orientation": "normal",
        "qkvg_k_orientation": "normal",
        "qkvg_v_orientation": "normal",
        qkvg_state_key("rotation", "q"): torch.eye(2),
        qkvg_state_key("rotation", "gate"): torch.eye(2),
        qkvg_state_key("rotation", "k"): torch.eye(1),
        qkvg_state_key("rotation", "v"): torch.eye(1),
    }
    payload = build_persistent_param_state(
        [{"params": [saved_param]}],
        {saved_param: saved_state},
        lambda param: param_key,
    )

    restored_param = torch.nn.Parameter(torch.zeros(6, 3))
    optimizer_state = {restored_param: {"momentum": torch.zeros(6, 3)}}
    summary = restore_distributed_checkpoint_state(
        param_state_payload=payload,
        param_groups=[{"params": [restored_param]}],
        optimizer_state=optimizer_state,
        get_param_key=lambda param: param_key,
    )

    assert summary == {"restored": 1, "unnamed": 0, "no_payload_entry": 0}
    restored_state = optimizer_state[restored_param]
    assert restored_state["qkvg_split_shapes"] == (2, 2, 1, 1)
    assert restored_state["qkvg_gate_orientation"] == "normal"
    torch.testing.assert_close(restored_state[qkvg_state_key("rotation", "q")], torch.eye(2))
    torch.testing.assert_close(
        restored_state[qkvg_state_key("rotation", "gate")],
        torch.eye(2),
    )
    torch.testing.assert_close(restored_state[qkvg_state_key("rotation", "k")], torch.eye(1))
    torch.testing.assert_close(restored_state[qkvg_state_key("rotation", "v")], torch.eye(1))

    meta = AroDistMeta(
        shape=(6, 3),
        local_shape=(6, 3),
        global_shape=(6, 3),
        is_matrix_param=True,
        is_aro_param=True,
        orientation="transpose",
        param_uid=param_key,
    )
    adapter = object.__new__(DistributedAroOptimizer)
    adapter.dist_metas = {restored_param: meta}
    adapter.optimizer = type("Optimizer", (), {"state": optimizer_state})()
    adapter._validate_restored_aro_state()


def test_aro_checkpoint_payload_round_trip_preserves_linear_split_state():
    param_key = ("linear_fc1", (4, 3))
    saved_param = torch.nn.Parameter(torch.arange(12, dtype=torch.float32).view(4, 3))
    saved_state = {
        "momentum": torch.ones(4, 3),
        "local_shape": (4, 3),
        "global_shape": (4, 3),
        "orientation": "transpose",
        "linear_split_linear": True,
        "linear_split_rows": (2, 2),
        "linear_gate_orientation": "normal",
        "linear_up_orientation": "normal",
        linear_state_key("rotation", "gate"): torch.eye(2),
        linear_state_key("rotation", "up"): torch.eye(2),
    }
    payload = build_persistent_param_state(
        [{"params": [saved_param]}],
        {saved_param: saved_state},
        lambda param: param_key,
    )

    restored_param = torch.nn.Parameter(torch.zeros(4, 3))
    optimizer_state = {restored_param: {"momentum": torch.zeros(4, 3)}}
    summary = restore_distributed_checkpoint_state(
        param_state_payload=payload,
        param_groups=[{"params": [restored_param]}],
        optimizer_state=optimizer_state,
        get_param_key=lambda param: param_key,
    )

    assert summary == {"restored": 1, "unnamed": 0, "no_payload_entry": 0}
    restored_state = optimizer_state[restored_param]
    assert restored_state["linear_split_rows"] == (2, 2)
    assert restored_state["linear_gate_orientation"] == "normal"
    torch.testing.assert_close(
        restored_state[linear_state_key("rotation", "gate")],
        torch.eye(2),
    )
    torch.testing.assert_close(
        restored_state[linear_state_key("rotation", "up")],
        torch.eye(2),
    )

    meta = AroDistMeta(
        shape=(4, 3),
        local_shape=(4, 3),
        global_shape=(4, 3),
        is_matrix_param=True,
        is_aro_param=True,
        orientation="transpose",
        param_uid=param_key,
    )
    adapter = object.__new__(DistributedAroOptimizer)
    adapter.dist_metas = {restored_param: meta}
    adapter.optimizer = type("Optimizer", (), {"state": optimizer_state})()
    adapter._validate_restored_aro_state()


def test_aro_casts_restored_matrix_state_to_configured_dtypes():
    param = torch.nn.Parameter(torch.zeros(4, 3, dtype=torch.float32))
    state = {
        "momentum": torch.ones(4, 3, dtype=torch.float32),
        "local_shape": (4, 3),
        "global_shape": (4, 3),
        "orientation": "transpose",
        "qkv_split_qkv": True,
        "qkv_split_shapes": (2, 1, 1),
        "qkv_q_orientation": "normal",
        "qkv_k_orientation": "normal",
        "qkv_v_orientation": "normal",
        qkv_state_key("rotation", "q"): torch.eye(2, dtype=torch.float32),
        qkv_state_key("rotation", "k"): torch.eye(1, dtype=torch.float32),
        qkv_state_key("rotation", "v"): torch.eye(1, dtype=torch.float32),
    }
    meta = AroDistMeta(
        shape=(4, 3),
        local_shape=(4, 3),
        global_shape=(4, 3),
        is_matrix_param=True,
        is_aro_param=True,
        orientation="transpose",
        param_uid=("linear_qkv", (4, 3)),
    )
    adapter = object.__new__(DistributedAroOptimizer)
    adapter.dist_metas = {param: meta}
    adapter.optimizer = type("Optimizer", (), {"state": {param: state}})()
    adapter._mixed_precision_config = AroMixedPrecisionConfig(
        momentum_dtype=torch.bfloat16,
        rotation_dtype=torch.bfloat16,
    )

    adapter._cast_restored_aro_state_dtypes()

    assert state["momentum"].dtype == torch.bfloat16
    assert state[qkv_state_key("rotation", "q")].dtype == torch.bfloat16
    assert state[qkv_state_key("rotation", "k")].dtype == torch.bfloat16
    assert state[qkv_state_key("rotation", "v")].dtype == torch.bfloat16
    adapter._validate_restored_aro_state()


def test_aro_checkpoint_payload_round_trip_restores_scalar_state():
    param_key = ("scalar", (6,))
    saved_param = torch.nn.Parameter(torch.arange(6, dtype=torch.float32))
    saved_state = {
        "exp_avg": torch.full((6,), 0.25),
        "exp_avg_sq": torch.full((6,), 0.5),
        "step": 7,
    }
    payload = build_persistent_param_state(
        [{"params": [saved_param]}],
        {saved_param: saved_state},
        lambda param: param_key,
    )

    restored_param = torch.nn.Parameter(torch.zeros(6))
    optimizer_state = {
        restored_param: {
            "exp_avg": torch.zeros(6),
            "exp_avg_sq": torch.zeros(6),
        }
    }
    summary = restore_distributed_checkpoint_state(
        param_state_payload=payload,
        param_groups=[{"params": [restored_param]}],
        optimizer_state=optimizer_state,
        get_param_key=lambda param: param_key,
    )

    assert summary == {"restored": 1, "unnamed": 0, "no_payload_entry": 0}
    torch.testing.assert_close(restored_param, saved_param)
    torch.testing.assert_close(optimizer_state[restored_param]["exp_avg"], saved_state["exp_avg"])
    torch.testing.assert_close(
        optimizer_state[restored_param]["exp_avg_sq"],
        saved_state["exp_avg_sq"],
    )
    assert optimizer_state[restored_param]["step"] == 7


def _dist_update_worker(rank, world_size, init_file, case_name):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        shard_case, orientation_case = case_name.split(":")
        generator = torch.Generator().manual_seed(1234)
        full = torch.randn(
            (4, 6) if orientation_case == "normal" else (6, 4),
            dtype=torch.float32,
            generator=generator,
        )
        orientation = choose_orientation(full.shape)
        assert orientation == orientation_case
        oriented_rows, oriented_cols = orient_tensor(full, orientation).shape
        force_qr_cases = {"row2dqr", "rowemptyqr"}
        rms_scale_cases = {"rowempty", "rowemptyqr"}
        config = AroParamConfig(
            sinkhorn_iters=3,
            qr_backend="qr" if shard_case in force_qr_cases else "scqr",
            scqr_eps=1e-5,
            update_rms_scale=1.0 if shard_case in rms_scale_cases else 0.0,
            orientation=orientation,
        )
        full_update, full_rotation = compute_aro_update(
            momentum=full,
            rotation=torch.eye(4, dtype=torch.float32),
            config=config,
            orientation=orientation,
        )

        if shard_case == "row":
            group = dist.new_group(tuple(range(world_size)))
            rows_per_rank = oriented_rows // world_size
            row_start = rank * rows_per_rank
            row_end = row_start + rows_per_rank
            if orientation == "normal":
                local = full[row_start:row_end, :].contiguous()
                expected_update = full_update[row_start:row_end, :]
            else:
                local = full[:, row_start:row_end].contiguous()
                expected_update = full_update[:, row_start:row_end]
            rotation = torch.eye(oriented_rows, dtype=torch.float32)[row_start:row_end, :].contiguous()
            update, new_rotation = compute_aro_update(
                momentum=local,
                rotation=rotation,
                config=config,
                orientation=orientation,
                row_groups=(group,),
            )
            torch.testing.assert_close(update, expected_update, rtol=2e-4, atol=2e-4)
            torch.testing.assert_close(new_rotation, full_rotation[row_start:row_end, :], rtol=2e-4, atol=2e-4)
        elif shard_case == "column":
            group = dist.new_group(tuple(range(world_size)))
            cols_per_rank = oriented_cols // world_size
            col_start = rank * cols_per_rank
            col_end = col_start + cols_per_rank
            if orientation == "normal":
                local = full[:, col_start:col_end].contiguous()
                expected_update = full_update[:, col_start:col_end]
            else:
                local = full[col_start:col_end, :].contiguous()
                expected_update = full_update[col_start:col_end, :]
            update, new_rotation = compute_aro_update(
                momentum=local,
                rotation=torch.eye(oriented_rows, dtype=torch.float32),
                config=config,
                orientation=orientation,
                column_groups=(group,),
            )
            torch.testing.assert_close(update, expected_update, rtol=2e-4, atol=2e-4)
            torch.testing.assert_close(new_rotation, full_rotation, rtol=2e-4, atol=2e-4)
        elif shard_case == "grid":
            row_groups = [dist.new_group((col, 2 + col)) for col in range(2)]
            col_groups = [dist.new_group((2 * row, 2 * row + 1)) for row in range(2)]
            row_rank = rank // 2
            col_rank = rank % 2
            rows_per_rank = oriented_rows // 2
            cols_per_rank = oriented_cols // 2
            row_start = row_rank * rows_per_rank
            row_end = row_start + rows_per_rank
            col_start = col_rank * cols_per_rank
            col_end = col_start + cols_per_rank
            if orientation == "normal":
                local = full[row_start:row_end, col_start:col_end].contiguous()
                expected_update = full_update[row_start:row_end, col_start:col_end]
            else:
                local = full[col_start:col_end, row_start:row_end].contiguous()
                expected_update = full_update[col_start:col_end, row_start:row_end]
            rotation = torch.eye(oriented_rows, dtype=torch.float32)[row_start:row_end, :].contiguous()
            update, new_rotation = compute_aro_update(
                momentum=local,
                rotation=rotation,
                config=config,
                orientation=orientation,
                row_groups=(row_groups[col_rank],),
                column_groups=(col_groups[row_rank],),
            )
            torch.testing.assert_close(
                update,
                expected_update,
                rtol=2e-4,
                atol=2e-4,
            )
            torch.testing.assert_close(new_rotation, full_rotation[row_start:row_end, :], rtol=2e-4, atol=2e-4)
        elif shard_case in ("row2d", "row2dqr"):
            row_groups_a = [dist.new_group((0, 1)), dist.new_group((2, 3))]
            row_groups_b = [dist.new_group((0, 2)), dist.new_group((1, 3))]
            group_a_index = rank // 2
            group_b_index = rank % 2
            row_start = rank
            row_end = rank + 1
            if orientation == "normal":
                local = full[row_start:row_end, :].contiguous()
                expected_update = full_update[row_start:row_end, :]
            else:
                local = full[:, row_start:row_end].contiguous()
                expected_update = full_update[:, row_start:row_end]
            rotation = torch.eye(oriented_rows, dtype=torch.float32)[row_start:row_end, :].contiguous()
            update, new_rotation = compute_aro_update(
                momentum=local,
                rotation=rotation,
                config=config,
                orientation=orientation,
                row_groups=(row_groups_a[group_a_index], row_groups_b[group_b_index]),
            )
            torch.testing.assert_close(
                update,
                expected_update,
                rtol=2e-4,
                atol=2e-4,
            )
            torch.testing.assert_close(new_rotation, full_rotation[row_start:row_end, :], rtol=2e-4, atol=2e-4)
        elif shard_case in ("rowempty", "rowemptyqr"):
            group = dist.new_group(tuple(range(world_size)))
            row_ranges = ((0, 2), (2, 2), (2, oriented_rows))
            row_start, row_end = row_ranges[rank]
            if orientation == "normal":
                local = full[row_start:row_end, :].contiguous()
                expected_update = full_update[row_start:row_end, :]
            else:
                local = full[:, row_start:row_end].contiguous()
                expected_update = full_update[:, row_start:row_end]
            rotation = torch.eye(oriented_rows, dtype=torch.float32)[row_start:row_end, :].contiguous()
            update, new_rotation = compute_aro_update(
                momentum=local,
                rotation=rotation,
                config=config,
                orientation=orientation,
                row_groups=(group,),
            )
            torch.testing.assert_close(
                update,
                expected_update,
                rtol=2e-4,
                atol=2e-4,
            )
            torch.testing.assert_close(
                new_rotation,
                full_rotation[row_start:row_end, :],
                rtol=2e-4,
                atol=2e-4,
            )
        else:
            raise AssertionError(f"unknown case: {case_name}")
    finally:
        dist.destroy_process_group()


def _dist_batched_update_worker(rank, world_size, init_file):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        group = dist.new_group(tuple(range(world_size)))
        generator = torch.Generator().manual_seed(2468)
        full_momentums = [
            torch.randn((4, 6), dtype=torch.float32, generator=generator),
            torch.randn((4, 6), dtype=torch.float32, generator=generator),
        ]
        config = AroParamConfig(
            sinkhorn_iters=3,
            qr_backend="scqr",
            scqr_eps=1e-5,
            update_rms_scale=1.0,
            orientation="normal",
        )
        full_updates = []
        full_rotations = []
        for full in full_momentums:
            update, rotation = compute_aro_update(
                momentum=full,
                rotation=torch.eye(4, dtype=torch.float32),
                config=config,
                orientation="normal",
            )
            full_updates.append(update)
            full_rotations.append(rotation)

        row_ranges = ((0, 2), (2, 2), (2, 4))
        row_start, row_end = row_ranges[rank]
        local_momentums = [full[row_start:row_end, :].contiguous() for full in full_momentums]
        local_rotations = [
            torch.eye(4, dtype=torch.float32)[row_start:row_end, :].contiguous()
            for _ in local_momentums
        ]
        result = compute_aro_updates_batched(
            momentums=local_momentums,
            rotations=local_rotations,
            config=config,
            orientation="normal",
            row_groups=(group,),
        )

        assert result is not None
        updates, rotations = result
        for update, rotation, full_update, full_rotation in zip(
            updates,
            rotations,
            full_updates,
            full_rotations,
        ):
            torch.testing.assert_close(update, full_update[row_start:row_end, :], rtol=2e-4, atol=2e-4)
            torch.testing.assert_close(
                rotation,
                full_rotation[row_start:row_end, :],
                rtol=2e-4,
                atol=2e-4,
            )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is unavailable")
@pytest.mark.parametrize(
    "case_name,world_size",
    [
        ("row:normal", 2),
        ("column:normal", 2),
        ("grid:normal", 4),
        ("row2d:normal", 4),
        ("row2dqr:normal", 4),
        ("row:transpose", 2),
        ("column:transpose", 2),
        ("grid:transpose", 4),
        ("row2d:transpose", 4),
        ("row2dqr:transpose", 4),
        ("rowempty:normal", 3),
        ("rowempty:transpose", 3),
        ("rowemptyqr:normal", 3),
        ("rowemptyqr:transpose", 3),
    ],
)
def test_distributed_aro_update_matches_full_reference(case_name, world_size):
    if not dist.is_gloo_available():
        pytest.skip("gloo backend is unavailable")
    with tempfile.TemporaryDirectory() as tmpdir:
        init_file = os.path.join(tmpdir, f"aro_{case_name}.init")
        mp.spawn(
            _dist_update_worker,
            args=(world_size, init_file, case_name),
            nprocs=world_size,
            join=True,
        )


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is unavailable")
def test_distributed_batched_aro_update_matches_full_reference_with_empty_row_shard():
    if not dist.is_gloo_available():
        pytest.skip("gloo backend is unavailable")
    with tempfile.TemporaryDirectory() as tmpdir:
        init_file = os.path.join(tmpdir, "aro_batched_empty.init")
        mp.spawn(
            _dist_batched_update_worker,
            args=(3, init_file),
            nprocs=3,
            join=True,
        )
