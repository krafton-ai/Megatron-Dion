from types import SimpleNamespace
from pathlib import Path

import torch

from megatron.core.optimizer.dion.backend import DionBackend
from megatron.core.optimizer.dion.distributed.optimizer import DistributedDionOptimizer
from megatron.core.optimizer.matrix.grad_norm import grad_norm_inputs
from megatron.core.optimizer.matrix.checkpoint_io import (
    MATRIX_SUBSTRATE_FORMAT_VERSION,
    build_matrix_checkpoint_metadata as build_dion_checkpoint_metadata,
    validate_matrix_checkpoint_metadata as validate_dion_checkpoint_metadata,
)
from megatron.core.optimizer.dion.types import (
    DionDistMeta,
    DionStepParam,
)
from megatron.core.optimizer.matrix.backend import MatrixBackend, MatrixStateSpec
from megatron.core.optimizer.matrix.distrib_optimizer import DistributedMatrixOptimizer
from megatron.core.optimizer.matrix.types import (
    MatrixBucketLayout,
    MatrixDistMeta,
    MatrixShardEntry,
    MatrixShardLayout,
    MatrixStepParam,
)
from megatron.core.optimizer.matrix.splits import linear as matrix_linear
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


def test_matrix_types_are_backend_neutral_contracts():
    assert MatrixDistMeta.__module__.endswith(".matrix.types")
    assert MatrixShardEntry.__module__.endswith(".matrix.types")
    assert MatrixBucketLayout.__module__.endswith(".matrix.types")
    assert MatrixShardLayout.__module__.endswith(".matrix.types")


def test_dion_types_extend_matrix_contracts():
    assert issubclass(DionStepParam, MatrixStepParam)
    assert issubclass(DionDistMeta, MatrixDistMeta)


def test_matrix_split_helpers_are_canonical():
    assert matrix_qkv.resolve_qkv_split_shapes.__module__.endswith(".matrix.splits.qkv")
    assert matrix_qkvg.resolve_qkvg_split_shapes.__module__.endswith(".matrix.splits.qkvg")
    assert matrix_linear.resolve_linear_split_rows.__module__.endswith(".matrix.splits.linear")


def test_dion_checkpoint_metadata_carries_matrix_backend_contract():
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
        _resolve_dion_tp_group=lambda: None,
    )

    grads = grad_norm_inputs(optimizer)

    assert len(grads) == 2
    assert torch.equal(grads[0], std_model.main_grad[1:3])
    assert torch.equal(grads[1], dion_local_grad)
