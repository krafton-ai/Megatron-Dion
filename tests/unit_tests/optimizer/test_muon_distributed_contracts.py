import importlib
import inspect
from types import SimpleNamespace

import pytest
import torch

from megatron.core.optimizer.matrix.distrib_optimizer import DistributedMatrixOptimizer


def _import_required(module_name: str):
    try:
        return importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - assertion keeps failure readable.
        raise AssertionError(f"required Muon module is not importable: {module_name}") from exc


def _call_with_supported_kwargs(fn, **kwargs):
    signature = inspect.signature(fn)
    parameters = signature.parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
        return fn(**kwargs)
    return fn(**{key: value for key, value in kwargs.items() if key in parameters})


def _make_param_config(**overrides):
    types = _import_required("megatron.core.optimizer.muon.types")
    config_cls = getattr(types, "MuonParamConfig", None)
    if config_cls is None:
        state = _import_required("megatron.core.optimizer.muon.state")
        config_cls = getattr(state, "MuonParamConfig", None)
    if config_cls is None:
        raise AssertionError("Muon implementation must expose MuonParamConfig")

    kwargs = {
        "has_fs_shard": True,
        "use_fs_shard": True,
        "fs_shard_dim": 0,
        "has_tp_shard": True,
        "use_tp_shard": True,
        "tp_shard_dim": 1,
        "is_transposed": False,
        "fs_mode": "distributed",
        "tp_mode": "distributed",
        "ns_backend": "gram",
        "scale_mode": "spectral",
        "extra_scale_factor": 0.25,
    }
    kwargs.update(overrides)
    return _call_with_supported_kwargs(config_cls, **kwargs)


def test_distributed_muon_optimizer_uses_matrix_base():
    module = _import_required("megatron.core.optimizer.muon.distributed.optimizer")

    assert issubclass(module.DistributedMuonOptimizer, DistributedMatrixOptimizer)


def test_muon_backend_delegates_to_muon_adapter_surface():
    backend_module = _import_required("megatron.core.optimizer.muon.backend")
    backend = backend_module.MuonBackend()
    calls = []

    class FakeAdapter:
        def _refresh_muon_step_metadata(self, **kwargs):
            calls.append(("refresh", kwargs))

        def _should_use_distributed_muon_update(self, *args):
            calls.append(("use_matrix", args))
            return True

        def _expand_split_muon_params(self, **kwargs):
            calls.append(("split_children", kwargs))
            return ["child"]

        def _sync_muon_state(self, matrix_params):
            calls.append(("sync_state", tuple(matrix_params)))

        def _build_muon_batches(self, matrix_params):
            calls.append(("build_batches", tuple(matrix_params)))
            return ["batch"]

    adapter = FakeAdapter()
    param = torch.nn.Parameter(torch.ones(2, 2))
    state = {"momentum": torch.zeros(2, 2)}
    optim_group = {"algorithm": "muon"}
    dist_meta = SimpleNamespace(param_uid=("param",), param_name="param")

    backend.refresh_state(
        adapter,
        param=param,
        state=state,
        optim_group=optim_group,
        dist_meta=dist_meta,
    )
    assert backend.use_matrix(
        adapter,
        param=param,
        state=state,
        optim_group=optim_group,
        dist_meta=dist_meta,
    )
    assert backend.split_children(
        adapter,
        param=param,
        grad=torch.ones_like(param),
        state=state,
        optim_group=optim_group,
        config=SimpleNamespace(),
        dist_meta=dist_meta,
    ) == ["child"]
    backend.sync_state(adapter, ["step"])
    assert backend.build_batches(adapter, ["step"]) == ["batch"]

    assert [name for name, _ in calls] == [
        "refresh",
        "use_matrix",
        "split_children",
        "sync_state",
        "build_batches",
    ]


def test_muon_batch_key_uses_logical_shape_and_named_modes():
    batches = _import_required("megatron.core.optimizer.muon.distributed.batches")
    build_batch_key = getattr(batches, "build_batch_key", None)
    if build_batch_key is None:
        raise AssertionError("Muon distributed batches module must expose build_batch_key")

    key = _call_with_supported_kwargs(
        build_batch_key,
        shape=(2, 3),
        cfg=_make_param_config(),
        dtype=torch.float32,
        global_shape=(8, 6),
        per_expert_global_shape=None,
        tensor_row_shard_sizes=None,
        row_shard_sizes=None,
    )

    key_text = repr(key)
    assert "(8, 6)" in key_text
    assert "distributed" in key_text
    assert "gram" in key_text
    assert "rank_fraction" not in key_text
    assert "low_rank" not in key_text


def test_muon_batch_key_is_tp_rank_invariant_for_fs_distributed_schedule():
    batches = _import_required("megatron.core.optimizer.muon.distributed.batches")
    build_batch_key = getattr(batches, "build_batch_key", None)
    if build_batch_key is None:
        raise AssertionError("Muon distributed batches module must expose build_batch_key")

    tp_group = SimpleNamespace(name="tp")
    fs_group = SimpleNamespace(name="fs")
    common = {
        "shape": (4, 3),
        "cfg": _make_param_config(fs_mode="distributed", tp_mode="distributed"),
        "dtype": torch.float32,
        "global_shape": (8, 6),
        "param_uid": ("layer", 0, "q"),
        "fs_group": fs_group,
        "tp_group": tp_group,
    }

    rank0_key = _call_with_supported_kwargs(
        build_batch_key,
        **common,
        dist_meta=SimpleNamespace(
            tp_rank=0,
            tp_world_size=2,
            fs_rank=1,
            fs_world_size=2,
            param_uid=common["param_uid"],
            tp_group=tp_group,
            fs_group=fs_group,
        ),
    )
    rank1_key = _call_with_supported_kwargs(
        build_batch_key,
        **common,
        dist_meta=SimpleNamespace(
            tp_rank=1,
            tp_world_size=2,
            fs_rank=1,
            fs_world_size=2,
            param_uid=common["param_uid"],
            tp_group=tp_group,
            fs_group=fs_group,
        ),
    )

    assert rank0_key == rank1_key


@pytest.mark.parametrize(
    "fs_mode,tp_mode",
    tuple(
        (fs_mode, tp_mode)
        for fs_mode in ("blockwise", "duplicated", "distributed")
        for tp_mode in ("blockwise", "duplicated", "distributed")
    ),
)
def test_muon_param_config_keeps_fs_and_tp_modes_explicit(fs_mode, tp_mode):
    config = _make_param_config(fs_mode=fs_mode, tp_mode=tp_mode)

    assert getattr(config, "fs_mode") == fs_mode
    assert getattr(config, "tp_mode") == tp_mode


def test_muon_param_config_normalizes_legacy_duplicated_debug_alias():
    config = _make_param_config(fs_mode="duplicated_debug", tp_mode="duplicated_debug")

    assert getattr(config, "fs_mode") == "duplicated"
    assert getattr(config, "tp_mode") == "duplicated"


def test_muon_fs_distributed_path_does_not_use_owner_all_to_all():
    optimizer_module = _import_required("megatron.core.optimizer.muon.distributed.optimizer")
    source = inspect.getsource(optimizer_module.DistributedMuonOptimizer._apply_fs_distributed_batch)

    assert "all_to_all" not in source
    assert "owner" not in source
    assert "_orthogonalize_2d" in source


def test_muon_fs_tp_scheduling_order_is_explicit():
    optimizer_module = _import_required("megatron.core.optimizer.muon.distributed.optimizer")
    fs_distributed = inspect.getsource(
        optimizer_module.DistributedMuonOptimizer._apply_fs_distributed_batch
    )
    fs_duplicated = inspect.getsource(
        optimizer_module.DistributedMuonOptimizer._apply_fs_duplicated_batch
    )

    assert fs_distributed.index("if duplicate_tp") < fs_distributed.index("orthogonalized")
    assert "_gather_axis" in fs_distributed
    assert "_orthogonalize_2d" in fs_distributed
    assert "_orthogonalize_axis" in fs_distributed

    assert "_apply_fs_duplicated_entry(entry)" in fs_duplicated
    assert "_orthogonalize_2d" not in fs_duplicated


def test_muon_axis_gather_handles_uneven_shards():
    optimizer_module = _import_required("megatron.core.optimizer.muon.distributed.optimizer")
    gather_source = inspect.getsource(optimizer_module.DistributedMuonOptimizer._gather_axis)
    slice_source = inspect.getsource(optimizer_module.DistributedMuonOptimizer._slice_axis)

    assert "gathered_sizes" in gather_source
    assert "max_size" in gather_source
    assert "return_sizes" in gather_source
    assert "partition_sizes" in slice_source
    assert "torch.chunk" not in slice_source
    assert "_split_range" in slice_source


def test_matrix_inter_instance_reduce_buffers_include_matrix_and_standard_grads():
    gradients = _import_required("megatron.core.optimizer.matrix.gradients")
    matrix_grad = torch.ones(3)
    standard_grad = torch.ones(2)
    bucket = SimpleNamespace(
        _matrix_use_full_grad_after_sync=False,
        _matrix_grad_transport=SimpleNamespace(
            matrix_grad_shard=matrix_grad,
            standard_grad=standard_grad,
        ),
    )

    buffers = gradients.get_inter_instance_grad_buffers(bucket)

    assert len(buffers) == 2
    assert buffers[0] is matrix_grad
    assert buffers[1] is standard_grad


def test_matrix_inter_instance_reduce_uses_full_grad_after_all_reduce_path():
    gradients = _import_required("megatron.core.optimizer.matrix.gradients")
    full_grad = torch.ones(5)
    bucket = SimpleNamespace(
        _matrix_use_full_grad_after_sync=True,
        grad_data=full_grad,
        _matrix_grad_transport=SimpleNamespace(
            matrix_grad_shard=torch.ones(3),
            standard_grad=torch.ones(2),
        ),
    )

    buffers = gradients.get_inter_instance_grad_buffers(bucket)

    assert len(buffers) == 1
    assert buffers[0] is full_grad


def test_param_and_grad_buffer_uses_generic_matrix_inter_instance_hook():
    buffer_module = _import_required("megatron.core.distributed.param_and_grad_buffer")
    source = inspect.getsource(buffer_module._ParamAndGradBucketGroup.start_grad_sync)

    assert "_get_inter_instance_grad_buffers" in source
    assert "_get_standard_inter_instance_grad_buffer(bucket)" not in source


def test_muon_inter_instance_reduce_hook_is_backend_specific():
    muon_module = _import_required("megatron.core.optimizer.muon.distributed.optimizer")
    matrix_module = _import_required("megatron.core.optimizer.matrix.distrib_optimizer")
    grad_norm = _import_required("megatron.core.optimizer.matrix.grad_norm")

    matrix_source = inspect.getsource(
        matrix_module.DistributedMatrixOptimizer._get_inter_instance_grad_buffers
    )
    muon_source = inspect.getsource(
        muon_module.DistributedMuonOptimizer._get_inter_instance_grad_buffers
    )

    assert "get_standard_inter_instance_grad_buffer" in matrix_source
    assert "get_inter_instance_grad_buffers" in muon_source
    assert matrix_module.DistributedMatrixOptimizer._matrix_grads_are_replicate_synced(
        object.__new__(matrix_module.DistributedMatrixOptimizer)
    ) is False
    assert muon_module.DistributedMuonOptimizer._matrix_grads_are_replicate_synced(
        object.__new__(muon_module.DistributedMuonOptimizer)
    ) is True
    assert "_matrix_grads_are_replicate_synced" in inspect.getsource(
        grad_norm._matrix_grad_norm_sq
    )


def test_muon_gram_dtype_cli_field_reaches_optimizer_config_and_builders():
    optimizer_config = _import_required("megatron.core.optimizer.optimizer_config")
    algorithm = _import_required("megatron.core.optimizer.muon.algorithm")
    optimizer_module = _import_required("megatron.core.optimizer.muon.distributed.optimizer")

    config = optimizer_config.OptimizerConfig(muon_gram_ns_dtype="bfloat16")

    assert config.muon_gram_ns_dtype == "bfloat16"
    assert "gram_dtype" in inspect.signature(algorithm.MegatronMuon).parameters
    assert "muon_gram_ns_dtype" in inspect.getsource(algorithm.build_muon_optimizer)
    assert "muon_gram_ns_dtype" in inspect.getsource(
        optimizer_module.DistributedMuonOptimizer._build_param_config
    )


def test_muon_linear_child_fs_ranges_keep_empty_uneven_slots():
    optimizer_module = _import_required("megatron.core.optimizer.muon.distributed.optimizer")
    optimizer = object.__new__(optimizer_module.DistributedMuonOptimizer)
    entry = SimpleNamespace(
        dist_meta=SimpleNamespace(
            fs_world_size=4,
            fs_shard_dim=0,
            is_linear_child=True,
            linear_split_rows=(4, 4),
            linear_child_kind="gate",
        )
    )

    assert optimizer._fs_rank_ranges(entry, split_size=4) == [
        (0, 2),
        (2, 4),
        (0, 0),
        (0, 0),
    ]


def _make_split_child_optimizer():
    optimizer_module = _import_required("megatron.core.optimizer.muon.distributed.optimizer")
    optimizer = object.__new__(optimizer_module.DistributedMuonOptimizer)
    optimizer._child_param_config = lambda child_meta, child_shape, optim_group: child_meta.param_config
    return optimizer


def _make_muon_dist_meta(**overrides):
    muon_types = _import_required("megatron.core.optimizer.muon.types")
    kwargs = {
        "shape": (1, 2),
        "local_shape": (1, 2),
        "global_shape": (1, 2),
        "fs_start_idx": 0,
        "fs_end_idx": 1,
        "fs_shard_dim": 0,
        "fs_world_size": 1,
        "fs_rank": 0,
        "tp_shard_dim": 1,
        "tp_world_size": 1,
        "tp_rank": 0,
        "is_muon_param": True,
        "param_config": SimpleNamespace(fs_mode="distributed", tp_mode="distributed"),
        "param_uid": ("param",),
        "param_name": "param",
    }
    kwargs.update(overrides)
    return muon_types.MuonDistMeta(**kwargs)


def test_muon_qkv_split_children_keep_child_specific_empty_fs_slots():
    optimizer = _make_split_child_optimizer()
    param = torch.arange(2, dtype=torch.float32).view(1, 2)
    grad = torch.ones_like(param)
    state = {}
    meta = _make_muon_dist_meta(
        shape=(1, 2),
        local_shape=(1, 2),
        global_shape=(4, 2),
        fs_start_idx=3,
        fs_end_idx=4,
        fs_world_size=4,
        fs_rank=3,
        qkv_split_shapes=(2, 1, 1),
        param_uid=("qkv",),
        param_name="qkv",
    )

    children = optimizer._expand_qkv(
        param,
        grad,
        state,
        {},
        meta,
        tuple(meta.shape),
        tuple(meta.global_shape),
        (2, 1, 1),
    )

    assert [child.dist_meta.qkv_child_kind for child in children] == ["q", "k", "v"]
    assert [tuple(child.param.shape) for child in children] == [(0, 2), (0, 2), (1, 2)]
    assert [tuple(child.dist_meta.global_shape) for child in children] == [(2, 2), (1, 2), (1, 2)]
    assert optimizer._fs_rank_ranges(SimpleNamespace(dist_meta=children[0].dist_meta), 2) == [
        (0, 1),
        (1, 2),
        (0, 0),
        (0, 0),
    ]
    assert optimizer._fs_rank_ranges(SimpleNamespace(dist_meta=children[2].dist_meta), 1) == [
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 1),
    ]

    before = param.clone()
    children[0].commit_update(children[0].param, children[0].optimizer_state["momentum_buffer"])
    torch.testing.assert_close(param, before)

    children[2].commit_update(
        torch.full_like(children[2].param, 7.0),
        torch.full_like(children[2].optimizer_state["momentum_buffer"], 9.0),
    )
    torch.testing.assert_close(param, torch.full_like(param, 7.0))
    torch.testing.assert_close(state["momentum_buffer"], torch.full_like(param, 9.0))


def test_muon_qkvg_split_children_keep_child_specific_empty_fs_slots():
    optimizer = _make_split_child_optimizer()
    param = torch.arange(2, dtype=torch.float32).view(1, 2)
    grad = torch.ones_like(param)
    state = {}
    meta = _make_muon_dist_meta(
        shape=(1, 2),
        local_shape=(1, 2),
        global_shape=(6, 2),
        fs_start_idx=5,
        fs_end_idx=6,
        fs_world_size=6,
        fs_rank=5,
        qkvg_split_shapes=(2, 2, 1, 1),
        param_uid=("qkvg",),
        param_name="qkvg",
    )

    children = optimizer._expand_qkvg(
        param,
        grad,
        state,
        {},
        meta,
        tuple(meta.shape),
        tuple(meta.global_shape),
        (2, 2, 1, 1),
    )

    assert [child.dist_meta.qkvg_child_kind for child in children] == ["q", "gate", "k", "v"]
    assert [tuple(child.param.shape) for child in children] == [(0, 2), (0, 2), (0, 2), (1, 2)]
    assert [tuple(child.dist_meta.global_shape) for child in children] == [
        (2, 2),
        (2, 2),
        (1, 2),
        (1, 2),
    ]
    assert optimizer._fs_rank_ranges(SimpleNamespace(dist_meta=children[1].dist_meta), 2) == [
        (0, 0),
        (0, 0),
        (0, 1),
        (1, 2),
        (0, 0),
        (0, 0),
    ]
    assert optimizer._fs_rank_ranges(SimpleNamespace(dist_meta=children[3].dist_meta), 1) == [
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 1),
    ]

    before = param.clone()
    children[1].commit_update(children[1].param, children[1].optimizer_state["momentum_buffer"])
    torch.testing.assert_close(param, before)

    children[3].commit_update(
        torch.full_like(children[3].param, 7.0),
        torch.full_like(children[3].optimizer_state["momentum_buffer"], 9.0),
    )
    torch.testing.assert_close(param, torch.full_like(param, 7.0))
    torch.testing.assert_close(state["momentum_buffer"], torch.full_like(param, 9.0))


def test_muon_linear_split_children_keep_child_specific_empty_fs_slots():
    optimizer = _make_split_child_optimizer()
    param = torch.arange(4, dtype=torch.float32).view(2, 2)
    grad = torch.ones_like(param)
    state = {}
    meta = _make_muon_dist_meta(
        shape=(2, 2),
        local_shape=(2, 2),
        global_shape=(8, 2),
        fs_start_idx=4,
        fs_end_idx=6,
        fs_world_size=4,
        fs_rank=2,
        linear_split_rows=(4, 4),
        param_uid=("linear_fc1",),
        param_name="linear_fc1",
    )

    children = optimizer._expand_linear(
        param,
        grad,
        state,
        {},
        meta,
        tuple(meta.shape),
        tuple(meta.global_shape),
        (4, 4),
    )

    assert [child.dist_meta.linear_child_kind for child in children] == ["gate", "up"]
    assert [tuple(child.param.shape) for child in children] == [(0, 2), (2, 2)]
    assert [tuple(child.dist_meta.global_shape) for child in children] == [(4, 2), (4, 2)]
    assert optimizer._fs_rank_ranges(SimpleNamespace(dist_meta=children[0].dist_meta), 4) == [
        (0, 2),
        (2, 4),
        (0, 0),
        (0, 0),
    ]
    assert optimizer._fs_rank_ranges(SimpleNamespace(dist_meta=children[1].dist_meta), 4) == [
        (0, 0),
        (0, 0),
        (0, 2),
        (2, 4),
    ]

    before = param.clone()
    children[0].commit_update(children[0].param, children[0].optimizer_state["momentum_buffer"])
    torch.testing.assert_close(param, before)

    children[1].commit_update(
        torch.full_like(children[1].param, 7.0),
        torch.full_like(children[1].optimizer_state["momentum_buffer"], 9.0),
    )
    torch.testing.assert_close(param, torch.full_like(param, 7.0))
    torch.testing.assert_close(state["momentum_buffer"], torch.full_like(param, 9.0))


@pytest.mark.parametrize("_method_name", ("_expand_qkv", "_expand_qkvg", "_expand_linear"))
def test_muon_split_children_keep_empty_collective_slots_without_committing_them(_method_name):
    optimizer_module = _import_required("megatron.core.optimizer.muon.distributed.optimizer")
    helper = optimizer_module.DistributedMuonOptimizer._include_empty_split_child
    assert helper(SimpleNamespace(param_config=SimpleNamespace(fs_mode="blockwise", tp_mode="distributed")))
    assert helper(SimpleNamespace(param_config=SimpleNamespace(fs_mode="duplicated", tp_mode="blockwise")))
    assert not helper(SimpleNamespace(param_config=SimpleNamespace(fs_mode="blockwise", tp_mode="blockwise")))

    helper_source = inspect.getsource(helper)
    source = inspect.getsource(getattr(optimizer_module.DistributedMuonOptimizer, _method_name))

    assert "_include_empty_split_child" in source
    assert "\"duplicated\"" in helper_source
    assert "\"distributed\"" in helper_source
    assert "has_local_overlap" in source
    assert "if not has_local_overlap:\n                    return" in source
    assert "new_empty(child_shape)" in source


def test_muon_linear_fc1_split_metadata_is_copied_to_shards_and_dist_meta():
    optimizer_module = _import_required("megatron.core.optimizer.muon.distributed.optimizer")
    copy_source = inspect.getsource(optimizer_module.DistributedMuonOptimizer._copy_param_attrs)
    meta_source = inspect.getsource(optimizer_module.DistributedMuonOptimizer._build_dist_metas)

    assert "is_linear_fc1_param" in copy_source
    assert "linear_split_rows" in copy_source
    assert "get_linear_split_rows(model_param)" in copy_source

    assert "linear_split_rows=linear_split_rows" in meta_source
    assert "linear_partition_stride" in meta_source


def test_matrix_topology_defaults_muon_fs_to_dense_dp_domain():
    topology = _import_required("megatron.core.optimizer.matrix.topology")
    args = SimpleNamespace(
        optimizer="muon",
        world_size=16,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=4,
        expert_tensor_parallel_size=1,
        fully_shard_model_parallel_size=1,
        replicate_model_parallel_size=1,
    )

    assert topology.resolve_fs_rp_topology(args, optimizer_name="Muon optimizer") == (16, 1)

    args.fully_shard_model_parallel_size = 4
    assert topology.resolve_fs_rp_topology(args, optimizer_name="Muon optimizer") == (4, 4)


def test_muon_distributed_integration_resolves_matrix_fs_and_rp_groups():
    integration = _import_required("megatron.core.optimizer.muon.distributed.integration")
    source = inspect.getsource(integration.build_muon_distributed_optimizer)

    assert "resolve_fs_group" in source
    assert "get_matrix_replica_group" in source
    assert "fully_shard_model_parallel_size" in source
    assert "replica_model_parallel_size" in source
    assert "replica_group" in source


def test_muon_tp_group_falls_back_to_runtime_dense_tp_group(monkeypatch):
    optimizer_module = _import_required("megatron.core.optimizer.muon.distributed.optimizer")
    optimizer = object.__new__(optimizer_module.DistributedMuonOptimizer)
    optimizer._muon_tp_group = None
    optimizer._is_expert_muon = False
    calls = []
    group = object()

    optimizer._assert_group_excludes_context_parallel = lambda group, label: calls.append((group, label))
    monkeypatch.setattr(
        optimizer_module.parallel_state,
        "get_tensor_model_parallel_group",
        lambda check_initialized=False: group,
    )
    monkeypatch.setattr(
        optimizer_module.parallel_state,
        "get_expert_tensor_parallel_group",
        lambda check_initialized=False: None,
    )

    assert optimizer._resolve_muon_tp_group() is group
    assert calls == [(group, "muon_tp_group")]


def test_muon_checkpoint_topology_signature_includes_rp_group():
    optimizer_module = _import_required("megatron.core.optimizer.muon.distributed.optimizer")
    source = inspect.getsource(optimizer_module.DistributedMuonOptimizer._muon_checkpoint_topology_signature)

    assert "rp_group" in source


@pytest.mark.parametrize("rp_size", (1, 2))
def test_muon_checkpoint_metadata_carries_matrix_backend_contract(rp_size):
    backend_module = _import_required("megatron.core.optimizer.muon.backend")
    checkpoint_io = _import_required(
        "megatron.core.optimizer.muon.distributed.checkpoint_io"
    )
    build_metadata = getattr(checkpoint_io, "build_muon_checkpoint_metadata", None)
    validate_metadata = getattr(checkpoint_io, "validate_muon_checkpoint_metadata", None)
    if build_metadata is None or validate_metadata is None:
        raise AssertionError(
            "Muon checkpoint_io must expose build_muon_checkpoint_metadata and "
            "validate_muon_checkpoint_metadata"
        )

    spec = backend_module.MuonBackend().state_spec()
    metadata = _call_with_supported_kwargs(
        build_metadata,
        dp_size=4,
        fs_size=2,
        tp_size=2,
        rp_size=rp_size,
        state_replica_size=1,
        requested_type="dp_reshardable",
        topology_signature={
            "data_parallel": (0, 1, 2, 3),
            "fs": (0, 1),
            "tp": (0, 2),
            "state_replica": (),
        },
        backend_state_spec=spec,
    )

    assert metadata["matrix_optimizer"]["backend"] == "muon"
    assert metadata["rp_size"] == rp_size
    assert metadata["matrix_optimizer"]["backend_state_version"] == spec.version
    assert tuple(metadata["matrix_optimizer"]["state_keys"]) == spec.state_keys
    assert "momentum" in metadata["matrix_optimizer"]["state_keys"]
    assert "Q" not in metadata["matrix_optimizer"]["state_keys"]
    assert "r" not in metadata["matrix_optimizer"]["state_keys"]

    _call_with_supported_kwargs(
        validate_metadata,
        metadata=metadata,
        checkpoint_metadata=metadata,
        dp_size=4,
        fs_size=2,
        tp_size=2,
        rp_size=rp_size,
        state_replica_size=1,
        topology_signature={
            "data_parallel": (0, 1, 2, 3),
            "fs": (0, 1),
            "tp": (0, 2),
            "state_replica": (),
        },
        backend_state_spec=spec,
    )
