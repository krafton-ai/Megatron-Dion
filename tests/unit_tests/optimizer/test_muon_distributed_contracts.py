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
        for fs_mode in ("blockwise", "distributed", "duplicated_debug")
        for tp_mode in ("blockwise", "distributed", "duplicated_debug")
    ),
)
def test_muon_param_config_keeps_fs_and_tp_modes_explicit(fs_mode, tp_mode):
    config = _make_param_config(fs_mode=fs_mode, tp_mode=tp_mode)

    assert getattr(config, "fs_mode") == fs_mode
    assert getattr(config, "tp_mode") == tp_mode
    assert getattr(config, "fs_mode") != "duplicated"
    assert getattr(config, "tp_mode") != "duplicated"


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


def test_muon_checkpoint_metadata_carries_matrix_backend_contract():
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
        rp_size=2,
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
    assert metadata["rp_size"] == 2
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
        rp_size=2,
        state_replica_size=1,
        topology_signature={
            "data_parallel": (0, 1, 2, 3),
            "fs": (0, 1),
            "tp": (0, 2),
            "state_replica": (),
        },
        backend_state_spec=spec,
    )
