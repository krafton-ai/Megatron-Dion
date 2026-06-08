import dataclasses
import importlib.util
import re
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest
import torch

from megatron.core.optimizer.aro.backend import AroBackend
from megatron.core.optimizer.aro.distributed import integration as aro_integration
from megatron.core.optimizer.aro.distributed.optimizer import DistributedAroOptimizer
from megatron.core.optimizer.aro.types import (
    AroDistMeta,
    AroStepParam,
)
from megatron.core.optimizer.dion.backend import DionBackend
from megatron.core.optimizer.dion.distributed.optimizer import DistributedDionOptimizer
from megatron.core.optimizer.matrix.gradients import MatrixGradTransport
from megatron.core.optimizer.matrix.grad_norm import grad_norm_inputs
from megatron.core.optimizer.matrix.topology import resolve_fs_rp_topology
from megatron.core.optimizer.matrix.checkpoint_io import (
    MATRIX_SUBSTRATE_FORMAT_VERSION,
    build_matrix_checkpoint_metadata as build_dion_checkpoint_metadata,
    build_matrix_checkpoint_metadata,
    validate_matrix_checkpoint_metadata as validate_dion_checkpoint_metadata,
    validate_matrix_checkpoint_metadata,
)
from megatron.core.optimizer.dion.types import (
    DionDistMeta,
    DionStepParam,
)
from megatron.core.optimizer.matrix.backend import MatrixBackend, MatrixStateSpec
from megatron.core.optimizer.matrix.distrib_optimizer import DistributedMatrixOptimizer
from megatron.core.optimizer.optimizer_config import AroOptimizerConfig
from megatron.core.optimizer.matrix.types import (
    MatrixBucketLayout,
    MatrixDistMeta,
    MatrixShardEntry,
    MatrixShardLayout,
    MatrixStepParam,
)
from megatron.core.optimizer.matrix.splits import linear as matrix_linear
from megatron.core.optimizer.matrix import parameter as matrix_parameter
from megatron.core.optimizer.matrix.splits import qkv as matrix_qkv
from megatron.core.optimizer.matrix.splits import qkvg as matrix_qkvg


def test_dion_distributed_optimizer_uses_matrix_base():
    assert issubclass(DistributedDionOptimizer, DistributedMatrixOptimizer)


def test_dion_backend_supports_current_matrix_features():
    backend = DionBackend()

    assert isinstance(backend, MatrixBackend)
    assert backend.name == "dion"
    assert backend.supports_fs
    assert backend.supports_rp
    assert backend.supports_tp
    assert backend.supports_expert_parallel
    assert backend.supports_split_qkv
    assert backend.supports_split_qkvg
    assert backend.supports_split_linear
    assert isinstance(backend.state_spec(), MatrixStateSpec)

    backend.validate_topology(
        fs_size=2,
        rp_size=2,
        tp_size=2,
        is_expert=True,
        split_qkv=True,
        split_qkvg=True,
        split_linear=True,
    )


def test_matrix_package_has_no_dion_dependency():
    matrix_root = Path(__file__).parents[3] / "megatron" / "core" / "optimizer" / "matrix"
    forbidden = (
        "dion.distributed",
        "..dion",
        "Dion",
        "DION",
        "dion",
        "rank_fraction",
        "use_low_rank_sync",
    )

    offenders = []
    for path in matrix_root.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        text = path.read_text()
        for token in forbidden:
            if token in text:
                offenders.append((path.relative_to(matrix_root), token))

    assert offenders == []


def test_matrix_package_has_no_optimizer_backend_dependency():
    matrix_root = Path(__file__).parents[3] / "megatron" / "core" / "optimizer" / "matrix"
    forbidden = (
        "optimizer.aro",
        "optimizer.dion",
        "optimizer.muon",
        "..aro",
        "..dion",
        "..muon",
        "Aro",
        "ARO",
        "Dion",
        "DION",
        "Muon",
        "MUON",
        "rank_fraction",
        "use_low_rank_sync",
    )

    offenders = []
    for path in matrix_root.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        text = path.read_text()
        for token in forbidden:
            if token in text:
                offenders.append((path.relative_to(matrix_root), token))

    assert offenders == []


def test_matrix_bucket_layout_validation_uses_tensor_collective(monkeypatch):
    group = object()
    calls = []

    def fake_all_gather(outputs, tensor, group=None):
        calls.append(tuple(int(value) for value in tensor.cpu().tolist()))
        for output in outputs:
            output.copy_(tensor)

    monkeypatch.setattr(matrix_parameter.dist, "get_backend", lambda group: "gloo")
    monkeypatch.setattr(matrix_parameter.dist, "all_gather", fake_all_gather)
    monkeypatch.setattr(
        matrix_parameter.dist,
        "all_gather_object",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("object gather")),
    )

    gathered = matrix_parameter._all_gather_layout_tuples((1, 2, 3), group, 2)

    assert gathered == [(1, 2, 3), (1, 2, 3)]
    assert calls == [(3,), (1, 2, 3)]


def test_dion_package_has_no_sharding_reexport_shim():
    shim = (
        Path(__file__).parents[3]
        / "megatron"
        / "core"
        / "optimizer"
        / "dion"
        / "distributed"
        / "sharding.py"
    )

    assert not shim.exists()


def test_aro_package_has_no_reexport_shim():
    aro_root = Path(__file__).parents[3] / "megatron" / "core" / "optimizer" / "aro"
    for relative in ("__init__.py", "distributed/__init__.py"):
        text = (aro_root / relative).read_text().strip()
        assert text in {
            '"""ARO optimizer package."""',
            '"""Distributed ARO optimizer package."""',
        }


def test_aro_package_has_no_cross_backend_or_private_runtime_tokens():
    aro_root = Path(__file__).parents[3] / "megatron" / "core" / "optimizer" / "aro"
    forbidden_tokens = (
        "optimizer.dion",
        "optimizer.muon",
        "..dion",
        "..muon",
        "Dion",
        "DION",
        "Muon",
        "MUON",
    )
    private_runtime_patterns = (
        re.compile(r"/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+){2,}"),
        re.compile(r"\b[a-z]\d{3}-[a-z]{2}\b"),
        re.compile(r"\b[A-Za-z][A-Za-z0-9-]*_\d{6}(?:_[A-Za-z0-9-]+)*\b"),
        re.compile(r"\.(?:s[bc]atch|s" r"qsh)\b"),
    )

    offenders = []
    for path in aro_root.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        text = path.read_text()
        for token in forbidden_tokens:
            if token in text:
                offenders.append((path.relative_to(aro_root), token))
        for pattern in private_runtime_patterns:
            if pattern.search(text):
                offenders.append((path.relative_to(aro_root), pattern.pattern))

    assert offenders == []


def test_matrix_types_are_backend_neutral_invariants():
    assert MatrixDistMeta.__module__.endswith(".matrix.types")
    assert MatrixShardEntry.__module__.endswith(".matrix.types")
    assert MatrixBucketLayout.__module__.endswith(".matrix.types")
    assert MatrixShardLayout.__module__.endswith(".matrix.types")


def test_dion_types_extend_matrix_invariants():
    assert issubclass(DionStepParam, MatrixStepParam)
    assert issubclass(DionDistMeta, MatrixDistMeta)


def test_aro_distributed_optimizer_uses_matrix_base():
    assert issubclass(DistributedAroOptimizer, DistributedMatrixOptimizer)


def test_aro_backend_supports_current_matrix_features():
    backend = AroBackend()

    assert isinstance(backend, MatrixBackend)
    assert backend.name == "aro"
    assert backend.supports_fs
    assert backend.supports_rp
    assert backend.supports_tp
    assert backend.supports_expert_parallel
    assert backend.supports_split_qkv
    assert backend.supports_split_qkvg
    assert backend.supports_split_linear
    assert isinstance(backend.state_spec(), MatrixStateSpec)

    backend.validate_topology(
        fs_size=2,
        rp_size=2,
        tp_size=2,
        is_expert=True,
        split_qkv=True,
        split_qkvg=True,
        split_linear=True,
    )


def test_aro_dense_fs_group_rejects_context_parallel_peers(monkeypatch):
    class FakeGroup:
        pass

    dense_group = FakeGroup()
    pure_dp_group = FakeGroup()
    cp_group = FakeGroup()
    group_ranks = {
        dense_group: (0, 1, 2, 3),
        pure_dp_group: (0, 1, 2, 3),
        cp_group: (0, 1),
    }

    monkeypatch.setattr(aro_integration.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(aro_integration.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(
        aro_integration.dist,
        "get_process_group_ranks",
        lambda group: group_ranks[group],
    )
    monkeypatch.setattr(
        "megatron.core.optimizer.matrix.topology.parallel_state.get_context_parallel_group",
        lambda check_initialized=False: cp_group,
    )

    with pytest.raises(RuntimeError, match="exclude context-parallel peers"):
        aro_integration._resolve_aro_fs_group(
            dense_fs_group=dense_group,
            pure_data_parallel_group=pure_dp_group,
            is_expert_parallel=False,
            requested_fs_size=4,
            requested_rp_size=1,
        )


def test_aro_tp_group_rejects_context_parallel_peers(monkeypatch):
    class FakeGroup:
        pass

    tp_group = FakeGroup()
    cp_group = FakeGroup()
    group_ranks = {
        tp_group: (0, 1),
        cp_group: (0, 1),
    }

    monkeypatch.setattr(
        "megatron.core.optimizer.matrix.distrib_optimizer.dist.is_initialized",
        lambda: True,
    )
    monkeypatch.setattr(
        "megatron.core.optimizer.matrix.distrib_optimizer.dist.get_rank",
        lambda: 0,
    )
    monkeypatch.setattr(
        "megatron.core.optimizer.matrix.distrib_optimizer.dist.get_process_group_ranks",
        lambda group: group_ranks[group],
    )
    monkeypatch.setattr(
        "megatron.core.optimizer.matrix.distrib_optimizer.parallel_state.get_context_parallel_group",
        lambda check_initialized=False: cp_group,
    )
    adapter = object.__new__(DistributedAroOptimizer)
    adapter._aro_tp_group = tp_group
    adapter._is_expert_aro = False

    with pytest.raises(RuntimeError, match="MATRIX_CP_GROUP_LEAK"):
        adapter._resolve_aro_tp_group()


def test_aro_types_extend_matrix_invariants():
    assert issubclass(AroStepParam, MatrixStepParam)
    assert issubclass(AroDistMeta, MatrixDistMeta)


def test_aro_marks_matrix_replica_grads_synced():
    assert DistributedAroOptimizer._matrix_grads_are_replicate_synced(
        object.__new__(DistributedAroOptimizer)
    )


def test_aro_exposes_matrix_owned_inter_instance_grad_buffers():
    matrix_grad = torch.ones(2)
    standard_grad = torch.ones(3)
    bucket = SimpleNamespace(
        _matrix_use_full_grad_after_sync=False,
        _matrix_grad_transport=MatrixGradTransport(
            matrix_grad_shard=matrix_grad,
            standard_grad=standard_grad,
        ),
    )

    assert DistributedAroOptimizer._get_inter_instance_grad_buffers(bucket) == (
        matrix_grad,
        standard_grad,
    )

    bucket._matrix_grad_transport = MatrixGradTransport(
        matrix_grad_shard=torch.empty(0),
        standard_grad=standard_grad,
    )
    assert DistributedAroOptimizer._get_inter_instance_grad_buffers(bucket) == (standard_grad,)

    full_grad = torch.ones(5)
    bucket = SimpleNamespace(
        _matrix_use_full_grad_after_sync=True,
        grad_data=full_grad,
    )
    assert DistributedAroOptimizer._get_inter_instance_grad_buffers(bucket) == (full_grad,)

    bucket.grad_data = torch.empty(0)
    assert DistributedAroOptimizer._get_inter_instance_grad_buffers(bucket) == ()


def test_aro_bucket_fs_group_uses_dense_override_and_expert_bucket_group():
    class FakeGroup:
        def __init__(self, size, rank):
            self._size = size
            self._rank = rank

        def size(self):
            return self._size

        def rank(self):
            return self._rank

    dense_fs_group = FakeGroup(size=4, rank=1)
    bucket_group = FakeGroup(size=2, rank=0)
    buffer = SimpleNamespace(
        aro_fs_group=dense_fs_group,
        aro_fs_size=4,
        aro_fs_rank=1,
        data_parallel_group=bucket_group,
    )

    dense_param = torch.nn.Parameter(torch.zeros(2, 2))
    dense_bucket = SimpleNamespace(bucket_id=0, params=(dense_param,))
    group, size, rank = DistributedAroOptimizer._bucket_fs_get_group_size_rank(
        buffer,
        dense_bucket,
    )
    assert group is dense_fs_group
    assert size == 4
    assert rank == 1

    expert_param = torch.nn.Parameter(torch.zeros(2, 2))
    expert_param.allreduce = False
    expert_bucket = SimpleNamespace(
        bucket_id=1,
        params=(expert_param,),
        intra_distributed_optimizer_instance_group=bucket_group,
        intra_distributed_optimizer_instance_size=2,
        intra_distributed_optimizer_instance_rank=0,
    )
    group, size, rank = DistributedAroOptimizer._bucket_fs_get_group_size_rank(
        buffer,
        expert_bucket,
    )
    assert group is bucket_group
    assert size == 2
    assert rank == 0


def test_aro_dist_meta_uses_bucket_specific_fs_group_for_expert_params(monkeypatch):
    class FakeGroup:
        pass

    dense_group = FakeGroup()
    expert_group = FakeGroup()
    sizes = {dense_group: 16, expert_group: 2}
    ranks = {dense_group: 7, expert_group: 1}

    monkeypatch.setattr(
        "megatron.core.optimizer.aro.distributed.optimizer._group_size",
        lambda group: 1 if group is None else sizes[group],
    )
    monkeypatch.setattr(
        "megatron.core.optimizer.aro.distributed.optimizer._group_rank",
        lambda group: 0 if group is None else ranks[group],
    )

    dense_model = torch.nn.Parameter(torch.zeros(16, 4))
    dense_model._param_name = "dense.weight"
    dense_model.matrix_optimizer_ready = True
    dense_model.use_aro = True
    dense_shard = torch.nn.Parameter(torch.zeros(1, 4))
    dense_shard._model_param = dense_model

    expert_model = torch.nn.Parameter(torch.zeros(2, 4))
    expert_model._param_name = "experts.0.weight"
    expert_model.matrix_optimizer_ready = True
    expert_model.use_aro = True
    expert_model.allreduce = False
    expert_shard = torch.nn.Parameter(torch.zeros(1, 4))
    expert_shard._model_param = expert_model

    adapter = object.__new__(DistributedAroOptimizer)
    adapter.optimizer = SimpleNamespace(param_groups=[{"params": [dense_shard, expert_shard]}])
    adapter.config = SimpleNamespace()
    adapter.fs_group = dense_group
    adapter.fs_size = 16
    adapter.fs_rank = 7
    adapter.aro_tp_group = None
    adapter.tp_size = 1
    adapter.tp_rank = 0
    adapter._global_param_name = lambda param: getattr(param, "_param_name", "")
    adapter._shard_layouts_by_param = {
        dense_model: MatrixShardLayout(
            local_shape=(1, 4),
            global_shape=(16, 4),
            fs_shard_dim=0,
            start_idx=7,
            end_idx=8,
        ),
        expert_model: MatrixShardLayout(
            local_shape=(1, 4),
            global_shape=(2, 4),
            fs_shard_dim=0,
            start_idx=1,
            end_idx=2,
            per_expert_global_shape=(2, 4),
        ),
    }
    adapter._matrix_buckets_by_param = {
        dense_model: SimpleNamespace(matrix_shard_group=dense_group),
        expert_model: SimpleNamespace(matrix_shard_group=expert_group),
    }

    metas = adapter._build_dist_metas()

    dense_meta = metas[dense_shard]
    assert dense_meta.fs_group is dense_group
    assert dense_meta.fs_world_size == 16
    assert dense_meta.fs_rank == 7

    expert_meta = metas[expert_shard]
    assert expert_meta.fs_group is expert_group
    assert expert_meta.fs_world_size == 2
    assert expert_meta.fs_rank == 1
    assert expert_meta.per_expert_global_shape == (2, 4)


def test_aro_checkpoint_metadata_tracks_rp_and_state_replica_topology():
    spec = AroBackend().state_spec()
    topology = {
        "data_parallel": (0, 1, 2, 3),
        "fs": (0, 1),
        "tp": (0,),
        "rp": (0, 2),
        "state_replica": (0, 2),
    }
    metadata = build_matrix_checkpoint_metadata(
        dp_size=4,
        fs_size=2,
        tp_size=1,
        rp_size=2,
        state_replica_size=2,
        requested_type="matrix_fs_rank_state",
        topology_signature=topology,
        backend_state_spec=spec,
    )

    validate_matrix_checkpoint_metadata(
        metadata,
        dp_size=4,
        fs_size=2,
        tp_size=1,
        rp_size=2,
        state_replica_size=2,
        topology_signature=topology,
        backend_state_spec=spec,
    )
    with pytest.raises(RuntimeError, match="RP topology change"):
        validate_matrix_checkpoint_metadata(
            metadata,
            dp_size=4,
            fs_size=2,
            tp_size=1,
            rp_size=1,
            state_replica_size=2,
            topology_signature=topology,
            backend_state_spec=spec,
        )
    with pytest.raises(RuntimeError, match="state-replica topology change"):
        validate_matrix_checkpoint_metadata(
            metadata,
            dp_size=4,
            fs_size=2,
            tp_size=1,
            rp_size=2,
            state_replica_size=1,
            topology_signature=topology,
            backend_state_spec=spec,
        )


def _matrix_topology_args(**overrides):
    values = {
        "optimizer": "aro",
        "world_size": 16,
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "context_parallel_size": 1,
        "expert_model_parallel_size": 8,
        "expert_tensor_parallel_size": 1,
        "fully_shard_model_parallel_size": 1,
        "replicate_model_parallel_size": 1,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_aro_two_node_validation_topologies_resolve_without_hardcoded_sizes():
    cases = (
        (
            _matrix_topology_args(
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                fully_shard_model_parallel_size=1,
            ),
            (16, 1),
        ),
        (
            _matrix_topology_args(
                tensor_model_parallel_size=8,
                pipeline_model_parallel_size=2,
                fully_shard_model_parallel_size=1,
            ),
            (1, 1),
        ),
        (
            _matrix_topology_args(
                tensor_model_parallel_size=4,
                pipeline_model_parallel_size=1,
                fully_shard_model_parallel_size=4,
            ),
            (4, 1),
        ),
    )

    for args, expected in cases:
        assert resolve_fs_rp_topology(args, optimizer_name="ARO optimizer") == expected


def _load_script_module(name):
    repo_root = Path(__file__).parents[3]
    script_path = repo_root / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_aro_two_node_validation_launcher_uses_planned_topologies():
    launcher = _load_script_module("aro_validation_matrix")

    cases = {case.name: case for case in launcher.CASES}
    expected_dense_topology = {
        "fs-only": (16, 1),
        "tp-only": (1, 1),
        "fs-tp": (4, 1),
    }
    expected_expert_fs = {
        "fs-only": 2,
        "tp-only": 1,
        "fs-tp": 2,
    }
    assert set(cases) == {"fs-only", "tp-only", "fs-tp"}
    assert (
        cases["fs-only"].tensor_parallel,
        cases["fs-only"].pipeline_parallel,
        cases["fs-only"].fully_sharded,
    ) == (1, 1, 16)
    assert (
        cases["tp-only"].tensor_parallel,
        cases["tp-only"].pipeline_parallel,
        cases["tp-only"].fully_sharded,
    ) == (8, 2, 1)
    assert (
        cases["fs-tp"].tensor_parallel,
        cases["fs-tp"].pipeline_parallel,
        cases["fs-tp"].fully_sharded,
    ) == (4, 1, 4)

    fs_args = launcher.train_args_for_case(cases["fs-only"])
    tp_args = launcher.train_args_for_case(cases["tp-only"])
    fs_tp_args = launcher.train_args_for_case(cases["fs-tp"])

    for args in (fs_args, tp_args, fs_tp_args):
        assert "--optimizer" in args
        assert args[args.index("--optimizer") + 1] == "aro"
        assert "--use-distributed-optimizer" in args
        assert "--overlap-grad-reduce" in args
        assert "--overlap-param-gather" in args
        assert "--replicate-model-parallel-size" in args
        assert args[args.index("--replicate-model-parallel-size") + 1] == "1"
        assert "--expert-model-parallel-size" in args
        assert args[args.index("--expert-model-parallel-size") + 1] == "8"
        assert "--expert-tensor-parallel-size" in args
        assert args[args.index("--expert-tensor-parallel-size") + 1] == "1"
        assert "--num-experts" in args
        assert args[args.index("--num-experts") + 1] == "80"
        assert "--clip-grad" in args
        assert args[args.index("--clip-grad") + 1] == "1.0"
        assert "--bf16" in args
        assert "--attention-output-gate" in args
        assert "--sliding-window-num-attention-heads" in args
        assert "--sliding-window-num-query-groups" in args
        assert "--aro-split-qkv" in args
        assert "--aro-split-linear" in args
        assert "--train-iters" not in args

    assert "--sequence-parallel" not in fs_args
    assert "--tp-comm-overlap" not in fs_args
    assert "--sequence-parallel" in tp_args
    assert "--tp-comm-overlap" in tp_args
    assert "--sequence-parallel" in fs_tp_args
    assert "--tp-comm-overlap" in fs_tp_args

    for name, case in cases.items():
        args = _matrix_topology_args(
            tensor_model_parallel_size=case.tensor_parallel,
            pipeline_model_parallel_size=case.pipeline_parallel,
            fully_shard_model_parallel_size=case.fully_sharded,
        )
        assert resolve_fs_rp_topology(args, optimizer_name="ARO optimizer") == (
            expected_dense_topology[name]
        )
        expert_data_parallel_size = args.world_size // (
            args.expert_tensor_parallel_size
            * args.expert_model_parallel_size
            * args.pipeline_model_parallel_size
        )
        assert expert_data_parallel_size == expected_expert_fs[name]

    commands = launcher.sbatch_commands(
        base_command="python pretrain_gpt.py",
        case_name="all",
        constraint="gpu-h200",
    )
    assert len(commands) == 3
    for name, command in commands:
        assert name in cases
        assert "--nodes=2" in command
        assert "--gpus-per-node=8" in command
        assert "--ntasks-per-node=8" in command
        assert "--constraint gpu-h200" in command
        assert "--nodelist" not in command

    with pytest.raises(ValueError, match="full-schedule"):
        launcher.train_args_for_case(cases["fs-only"], ["--train-iters", "2"])
    with pytest.raises(ValueError, match="full-schedule"):
        launcher.train_args_for_case(cases["fs-only"], ["--train-iters=2"])
    with pytest.raises(ValueError, match="full-schedule"):
        launcher.train_args_for_case(cases["fs-only"], ["--train-samples", "1024"])
    with pytest.raises(ValueError, match="full-schedule"):
        launcher.train_args_for_case(cases["fs-only"], ["--lr-decay-iters=2"])
    with pytest.raises(ValueError, match="full-schedule"):
        launcher.train_args_for_case(cases["fs-only"], ["--exit-interval", "10"])
    with pytest.raises(ValueError, match="validation-controlled"):
        launcher.train_args_for_case(cases["fs-only"], ["--optimizer", "adam"])
    with pytest.raises(ValueError, match="validation-controlled"):
        launcher.train_args_for_case(cases["fs-only"], ["--tensor-model-parallel-size=2"])
    with pytest.raises(ValueError, match="validation-controlled"):
        launcher.train_args_for_case(cases["fs-only"], ["--no-aro-split-qkv"])
    with pytest.raises(ValueError, match="H200"):
        launcher.sbatch_commands(
            base_command="python pretrain_gpt.py",
            case_name="fs-only",
            partition="generic-gpu",
        )
    with pytest.raises(ValueError, match="H200"):
        launcher.sbatch_commands(
            base_command="python pretrain_gpt.py",
            case_name="fs-only",
            partition="nonh200",
        )


def test_aro_validation_report_extracts_required_runtime_evidence():
    reporter = _load_script_module("aro_validation_report")
    log_lines = [
        "iteration        1/  750 | elapsed time per iteration (ms): 900.0 | "
        "lm loss: 7.9000E+00 | grad norm: 5.0\n",
        "iteration       10/  750 | elapsed time per iteration (ms): 120.5 | "
        "lm loss: 7.677413E+00 | grad norm: 3.3900E-01\n",
        "iteration       20/  750 | elapsed time per iteration (ms): 118.25 | "
        "lm loss: 7.643631E+00 | grad norm: 2.7400E-01\n",
    ]

    summary = reporter.summarize(
        case="fs-only",
        job_id="job-123",
        lines=log_lines,
        min_step=10,
        warmup_step=10,
    )
    assert summary.passed
    assert summary.reached_iteration == 20
    assert summary.metric_at_or_after(10).loss == pytest.approx(7.677413)
    assert summary.metric_at_or_after(20).grad_norm == pytest.approx(0.274)
    assert summary.post_startup_wall_clock().ms_per_iter == pytest.approx(120.5)

    markdown = reporter.format_markdown(summary)
    assert "fs-only" in markdown
    assert "job-123" in markdown
    assert "iteration 10" in markdown
    assert "iteration 20" in markdown
    assert "wall-clock" in markdown
    assert "/wbl" not in markdown


def test_aro_validation_report_merges_duplicate_log_iteration_lines():
    reporter = _load_script_module("aro_validation_report")
    summary = reporter.summarize(
        case="fs-only",
        job_id="job-duplicate-step",
        lines=[
            "iteration       10/  750 | elapsed time per iteration (ms): 120.5 |\n",
            "iteration       10/  750 | lm loss: 7.677413E+00 | grad norm: 3.3900E-01\n",
        ],
        min_step=10,
        warmup_step=10,
    )

    assert summary.passed
    metric = summary.metric_at_or_after(10)
    assert metric.loss == pytest.approx(7.677413)
    assert metric.grad_norm == pytest.approx(0.339)
    assert metric.ms_per_iter == pytest.approx(120.5)


def test_aro_validation_report_merges_tensorboard_scalar_tags():
    reporter = _load_script_module("aro_validation_report")
    scalar_events = [
        ("lm loss", 10, 7.677413),
        ("lm loss vs samples", 1024, 1.0),
        ("grad-norm", 10, 0.339),
        ("iteration-time", 10, 120.5),
        ("lm loss", 20, 7.643631),
        ("grad-norm", 20, 0.274),
        ("iteration-time", 20, 118.25),
    ]

    metrics = reporter.parse_scalar_events(scalar_events)
    summary = reporter.summarize_metrics(
        case="fs-tp",
        job_id="job-456",
        metrics=metrics,
        min_step=10,
        warmup_step=10,
    )

    assert summary.passed
    assert summary.reached_iteration == 20
    assert summary.metric_at_or_after(10).loss == pytest.approx(7.677413)
    assert summary.metric_at_or_after(10).grad_norm == pytest.approx(0.339)
    assert summary.metric_at_or_after(10).ms_per_iter == pytest.approx(120.5)
    assert summary.metric_at_or_after(1024) is None

    markdown = reporter.format_markdown(summary)
    assert "fs-tp" in markdown
    assert "job-456" in markdown
    assert "iteration 20" in markdown


def test_aro_validation_report_requires_loss_grad_norm_and_wall_clock(tmp_path):
    reporter = _load_script_module("aro_validation_report")
    incomplete = reporter.summarize(
        case="tp-only",
        job_id="job-789",
        lines=[
            "iteration       20/  750 | lm loss: 7.643631E+00 |\n",
        ],
        min_step=10,
        warmup_step=10,
    )

    assert not incomplete.passed
    assert incomplete.missing_evidence() == (
        "grad_norm>=iteration10",
        "wall_clock>=iteration10",
    )
    assert "missing evidence" in reporter.format_markdown(incomplete)

    event_dir = tmp_path / "tb"
    nested_dir = event_dir / "nested"
    nested_dir.mkdir(parents=True)
    event_file = nested_dir / "events.out.tfevents.test"
    event_file.write_text("", encoding="utf-8")
    ignored_file = nested_dir / "metrics.log"
    ignored_file.write_text("", encoding="utf-8")

    assert reporter.expand_event_paths([str(event_dir)]) == (event_file,)


def test_aro_validation_report_checks_all_required_topologies():
    reporter = _load_script_module("aro_validation_report")

    def complete_summary(case):
        return reporter.summarize(
            case=case,
            job_id=f"job-{case}",
            lines=[
                "iteration       10/  750 | elapsed time per iteration (ms): 120.5 | "
                "lm loss: 7.677413E+00 | grad norm: 3.3900E-01\n",
            ],
            min_step=10,
            warmup_step=10,
        )

    summaries = [
        complete_summary("fs-only"),
        complete_summary("tp-only"),
        complete_summary("fs-tp"),
    ]
    assert reporter.aggregate_missing_evidence(summaries) == ()
    assert "aggregate: passed" in reporter.format_aggregate_markdown(summaries)
    assert reporter.aggregate_missing_evidence([*summaries, complete_summary("fs-only")]) == (
        "fs-only:duplicate-summary",
    )
    assert reporter.aggregate_missing_evidence([*summaries, complete_summary("extra")]) == (
        "extra:unexpected-summary",
    )

    missing_case = summaries[:2]
    assert reporter.aggregate_missing_evidence(missing_case) == ("fs-tp:missing-summary",)

    incomplete_case = [
        complete_summary("fs-only"),
        complete_summary("tp-only"),
        reporter.summarize(
            case="fs-tp",
            job_id="job-fs-tp",
            lines=[
                "iteration       10/  750 | lm loss: 7.677413E+00 |\n",
            ],
            min_step=10,
            warmup_step=10,
        ),
    ]
    missing = reporter.aggregate_missing_evidence(incomplete_case)
    assert len(missing) == 1
    assert missing[0].startswith("fs-tp:")
    assert "grad_norm>=iteration10" in missing[0]
    assert "wall_clock>=iteration10" in missing[0]


def test_aro_validation_report_aggregate_sources_from_logs(tmp_path):
    reporter = _load_script_module("aro_validation_report")

    def write_log(case):
        path = tmp_path / f"{case}.log"
        path.write_text(
            "iteration       10/  750 | elapsed time per iteration (ms): 120.5 | "
            "lm loss: 7.677413E+00 | grad norm: 3.3900E-01\n"
            "iteration       20/  750 | elapsed time per iteration (ms): 118.25 | "
            "lm loss: 7.643631E+00 | grad norm: 2.7400E-01\n",
            encoding="utf-8",
        )
        return path

    specs = [
        f"fs-only:job-fs:{write_log('fs-only')}",
        f"tp-only:job-tp:{write_log('tp-only')}",
        f"fs-tp:job-grid:{write_log('fs-tp')}",
    ]

    summaries = reporter.summarize_case_sources(log_specs=specs, min_step=10, warmup_step=10)
    assert tuple(summary.case for summary in summaries) == ("fs-only", "fs-tp", "tp-only")
    assert reporter.aggregate_missing_evidence(summaries) == ()
    markdown = reporter.format_aggregate_markdown(summaries)
    assert "aggregate: passed" in markdown
    assert "job-fs" in markdown
    assert "job-grid" in markdown
    assert str(tmp_path) not in markdown

    with pytest.raises(ValueError, match="duplicate summary"):
        reporter.summarize_case_sources(log_specs=[specs[0], specs[0]])
    with pytest.raises(ValueError, match="cannot use both log and TensorBoard"):
        reporter.summarize_case_sources(log_specs=[specs[0]], event_specs=[specs[0]])
    with pytest.raises(ValueError, match="duplicate TensorBoard summary"):
        reporter.summarize_case_sources(
            event_specs=[
                f"fs-only:job-a:{tmp_path / 'events.out.tfevents.a'}",
                f"fs-only:job-b:{tmp_path / 'events.out.tfevents.b'}",
            ]
        )


def test_aro_matrix_topology_supports_rp_greater_than_one():
    args = _matrix_topology_args(
        expert_model_parallel_size=1,
        fully_shard_model_parallel_size=4,
        replicate_model_parallel_size=1,
    )
    assert resolve_fs_rp_topology(args, optimizer_name="ARO optimizer") == (4, 4)

    args = _matrix_topology_args(
        expert_model_parallel_size=1,
        fully_shard_model_parallel_size=1,
        replicate_model_parallel_size=4,
    )
    assert resolve_fs_rp_topology(args, optimizer_name="ARO optimizer") == (4, 4)

    args = _matrix_topology_args(
        expert_model_parallel_size=1,
        fully_shard_model_parallel_size=4,
        replicate_model_parallel_size=4,
    )
    assert resolve_fs_rp_topology(args, optimizer_name="ARO optimizer") == (4, 4)


def test_aro_param_group_tagging_sets_qkv_qkvg_and_linear_split_metadata(monkeypatch):
    from megatron.core.optimizer import _get_param_groups

    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        lambda output, item: output.__setitem__(0, item),
    )

    qkv_weight = torch.nn.Parameter(torch.zeros(4, 2))
    qkvg_weight = torch.nn.Parameter(torch.zeros(6, 2))
    linear_fc1_weight = torch.nn.Parameter(torch.zeros(8, 2))
    linear_fc1_weight.partition_stride = 2

    qkv_chunk = SimpleNamespace(
        config=SimpleNamespace(
            attention_output_gate=False,
            num_attention_heads=2,
            num_query_groups=1,
            kv_channels=1,
        ),
        named_parameters=lambda: (
            ("layers.0.self_attention.linear_qkv.weight", qkv_weight),
            ("layers.0.mlp.linear_fc1.weight", linear_fc1_weight),
        ),
    )
    qkvg_chunk = SimpleNamespace(
        config=SimpleNamespace(
            attention_output_gate=True,
            num_attention_heads=2,
            num_query_groups=1,
            kv_channels=1,
        ),
        named_parameters=lambda: (
            ("layers.1.self_attention.linear_qkv.weight", qkvg_weight),
        ),
    )

    param_groups = _get_param_groups([qkv_chunk, qkvg_chunk], AroOptimizerConfig(), {})

    assert any(qkv_weight in group["params"] for group in param_groups)
    assert qkv_weight.is_qkv is True
    assert qkv_weight.is_qkvg is False
    assert qkv_weight.qkv_split_shapes == (2, 1, 1)
    assert not hasattr(qkv_weight, "qkvg_split_shapes")

    assert qkvg_weight.is_qkvg is True
    assert qkvg_weight.is_qkv is False
    assert qkvg_weight.qkvg_split_shapes == (2, 2, 1, 1)
    assert not hasattr(qkvg_weight, "qkv_split_shapes")

    assert linear_fc1_weight.is_linear_fc1 is True
    assert linear_fc1_weight.linear_split_rows == (4, 4)


def test_shared_split_tagging_uses_backend_neutral_error_tags():
    repo_root = Path(__file__).parents[3]
    optimizer_init_text = (repo_root / "megatron" / "core" / "optimizer" / "__init__.py").read_text()

    assert "[MATRIX_LINEAR_FC1_INVALID_ROWS]" in optimizer_init_text
    assert "[DION_LINEAR_FC1_INVALID_ROWS]" not in optimizer_init_text
    assert "[ARO_LINEAR_FC1_INVALID_ROWS]" not in optimizer_init_text
    assert "[MUON_LINEAR_FC1_INVALID_ROWS]" not in optimizer_init_text


def test_aro_config_surface_matches_public_invariant():
    repo_root = Path(__file__).parents[3]
    required_fields = {
        "optimizer",
        "aro_momentum",
        "aro_base_optimizer",
        "aro_sinkhorn_iters",
        "aro_qr_backend",
        "aro_scqr_eps",
        "aro_update_rms_scale",
        "aro_scalar_optimizer",
        "aro_scalar_lr_scale",
        "aro_beta1",
        "aro_beta2",
        "aro_scalar_eps",
        "aro_split_qkv",
        "aro_split_linear",
        "aro_momentum_dtype",
        "aro_rotation_dtype",
        "fully_shard_model_parallel_size",
        "replicate_model_parallel_size",
    }
    config_fields = {field.name for field in dataclasses.fields(AroOptimizerConfig)}
    assert required_fields <= config_fields

    config = AroOptimizerConfig()
    assert config.optimizer == "aro"
    assert config.aro_base_optimizer == "sinkhorn"
    assert config.aro_qr_backend == "scqr"
    assert config.aro_update_rms_scale == 0.2
    assert config.aro_scalar_optimizer == "adam"

    arguments_text = (repo_root / "megatron" / "training" / "arguments.py").read_text()
    for flag in (
        "--aro-momentum",
        "--aro-base-optimizer",
        "--aro-sinkhorn-iters",
        "--aro-qr-backend",
        "--aro-scqr-eps",
        "--aro-update-rms-scale",
        "--aro-scalar-optimizer",
        "--aro-scalar-lr-scale",
        "--aro-beta1",
        "--aro-beta2",
        "--aro-scalar-eps",
        "--aro-split-qkv",
        "--no-aro-split-qkv",
        "--aro-split-linear",
        "--no-aro-split-linear",
        "--aro-momentum-dtype",
        "--aro-rotation-dtype",
        "--fully-shard-model-parallel-size",
        "--replicate-model-parallel-size",
    ):
        assert flag in arguments_text
    assert "choices=['adam', 'sgd', 'dion', 'muon', 'dist_muon', 'aro']" in arguments_text
    assert 'args.optimizer == "aro" and args.use_distributed_optimizer' in arguments_text
    assert 'args.optimizer == "aro" and not args.use_distributed_optimizer' in arguments_text
    assert "ARO split flags require --use-distributed-optimizer" in arguments_text
    assert '"--ckpt-format torch_dist' in arguments_text

    training_text = (repo_root / "megatron" / "training" / "training.py").read_text()
    assert "AroOptimizerConfig" in training_text
    assert "args.optimizer == 'aro'" in training_text

    optimizer_text = (repo_root / "megatron" / "core" / "optimizer" / "__init__.py").read_text()
    assert "build_aro_optimizer" in optimizer_text
    assert "build_aro_distributed_optimizer" in optimizer_text
    assert "prepare_aro_params" in optimizer_text
    assert "config.optimizer == 'aro'" in optimizer_text

    distrib_optimizer_text = (
        repo_root / "megatron" / "core" / "optimizer" / "distrib_optimizer.py"
    ).read_text()
    assert "MegatronAro" in distrib_optimizer_text


def test_matrix_split_helpers_are_canonical():
    assert matrix_qkv.resolve_qkv_split_shapes.__module__.endswith(".matrix.splits.qkv")
    assert matrix_qkvg.resolve_qkvg_split_shapes.__module__.endswith(".matrix.splits.qkvg")
    assert matrix_linear.resolve_linear_split_rows.__module__.endswith(".matrix.splits.linear")


def test_dion_checkpoint_metadata_carries_matrix_backend_invariant():
    spec = DionBackend().state_spec()
    metadata = build_dion_checkpoint_metadata(
        dp_size=2,
        fs_size=2,
        tp_size=1,
        rp_size=1,
        state_replica_size=1,
        requested_type="dp_reshardable",
        topology_signature={"fs": (0, 1), "tp": (), "rp": ()},
        backend_state_spec=spec,
    )

    assert metadata["matrix_optimizer"] == {
        "backend": "dion",
        "substrate_version": MATRIX_SUBSTRATE_FORMAT_VERSION,
        "backend_state_version": spec.version,
        "state_keys": spec.state_keys,
    }
    validate_dion_checkpoint_metadata(
        metadata,
        dp_size=2,
        fs_size=2,
        tp_size=1,
        rp_size=1,
        state_replica_size=1,
        topology_signature={"fs": (0, 1), "tp": (), "rp": ()},
        backend_state_spec=spec,
    )


def test_dion_checkpoint_topology_allows_singleton_group_identity_changes():
    spec = DionBackend().state_spec()
    metadata = build_dion_checkpoint_metadata(
        dp_size=16,
        fs_size=16,
        tp_size=1,
        rp_size=1,
        state_replica_size=1,
        requested_type="dp_reshardable",
        topology_signature={
            "data_parallel": tuple(range(16)),
            "fs": tuple(range(16)),
            "tp": (0,),
            "rp": (),
            "state_replica": (),
        },
        backend_state_spec=spec,
    )

    validate_dion_checkpoint_metadata(
        metadata,
        dp_size=16,
        fs_size=16,
        tp_size=1,
        rp_size=1,
        state_replica_size=1,
        topology_signature={
            "data_parallel": tuple(range(16)),
            "fs": tuple(range(16)),
            "tp": (13,),
            "rp": (),
            "state_replica": (),
        },
        backend_state_spec=spec,
    )


def test_grad_norm_inputs_use_standard_and_dion_step_surfaces():
    class FakeRange:
        start = 1
        end = 3
        size = 2

    std_model = torch.nn.Parameter(torch.zeros(4))
    std_model.main_grad = torch.arange(4, dtype=torch.float32)
    std_shard = torch.nn.Parameter(torch.zeros(2))
    std_shard.tensor_model_parallel = True

    dion_model = torch.nn.Parameter(torch.zeros(4, 4))
    dion_model.is_dion_param = True
    dion_model.is_matrix_param = True
    dion_shard = torch.nn.Parameter(torch.zeros(2, 4))
    dion_shard._model_param = dion_model
    dion_shard.tensor_model_parallel = True
    dion_local_grad = torch.arange(8, dtype=torch.float32).view(2, 4)

    optimizer = SimpleNamespace(
        config=SimpleNamespace(use_precision_aware_optimizer_no_fp8_or_ds_fp8=False),
        model_float16_groups=[],
        model_fp32_groups=[[std_model, dion_model]],
        shard_fp32_from_float16_groups=[],
        shard_fp32_groups=[[std_shard, dion_shard]],
        _get_model_param_range_map=lambda param: {"param": FakeRange()},
        _get_local_grad=lambda model_param, shard_param: dion_local_grad,
        _shard_param_uid=lambda shard_param: ("dion", 0),
        _resolve_matrix_tp_group=lambda: None,
    )

    grads = grad_norm_inputs(optimizer)

    assert len(grads) == 2
    assert torch.equal(grads[0], std_model.main_grad[1:3])
    assert torch.equal(grads[1], dion_local_grad)


def test_grad_norm_inputs_use_aro_matrix_step_surface():
    aro_model = torch.nn.Parameter(torch.zeros(4, 4))
    aro_model.is_aro_param = True
    aro_model.is_matrix_param = True
    aro_shard = torch.nn.Parameter(torch.zeros(2, 4))
    aro_shard._model_param = aro_model
    aro_shard.tensor_model_parallel = True
    aro_local_grad = torch.arange(8, dtype=torch.float32).view(2, 4)

    optimizer = SimpleNamespace(
        config=SimpleNamespace(use_precision_aware_optimizer_no_fp8_or_ds_fp8=False),
        model_float16_groups=[],
        model_fp32_groups=[[aro_model]],
        shard_fp32_from_float16_groups=[],
        shard_fp32_groups=[[aro_shard]],
        _get_local_grad=lambda model_param, shard_param: aro_local_grad,
        _shard_param_uid=lambda shard_param: ("aro", 0),
        _resolve_matrix_tp_group=lambda: None,
    )

    grads = grad_norm_inputs(optimizer)

    assert len(grads) == 1
    assert torch.equal(grads[0], aro_local_grad)
