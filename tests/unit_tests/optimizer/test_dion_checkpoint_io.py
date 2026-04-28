import pytest
import torch

from megatron.core.dist_checkpointing.mapping import ShardedObject
from megatron.core.optimizer.dion.types import DionMixedPrecisionConfig
from megatron.core.optimizer.distrib_dion import checkpoint_io
from megatron.core.optimizer.distrib_dion.checkpoint_io import (
    build_dion_checkpoint_metadata,
    build_distributed_checkpoint_state,
    resolve_dion_checkpoint_sharding_type,
    restore_persistent_param_state_,
    validate_dion_checkpoint_metadata,
)

TOPOLOGY = {
    "data_parallel": (0, 1, 2, 3, 4, 5, 6, 7),
    "fs": (0, 1, 2, 3),
    "tp": (0, 1),
    "rp": (0, 4),
    "state_replica": (0, 4),
}


def test_dion_param_state_uses_dp_rank_object_metadata():
    param = torch.nn.Parameter(torch.empty(2, 3))
    param._dion_param_uid = "layer.weight"
    optimizer_state = {
        param: {
            "momentum": torch.ones(2, 3),
            "Q": torch.ones(3, 1),
            "_q_full_buffer": torch.empty(3, 1),
        }
    }

    state_dict = build_distributed_checkpoint_state(
        common_state={"optimizer": {"param_groups": []}},
        param_groups=[{"params": [param]}],
        optimizer_state=optimizer_state,
        get_param_key=lambda tensor: tensor._dion_param_uid,
        base_key="optimizer.distributed.dp_group_idx_0",
        common_replica_id=(0, 0, 3),
        state_global_shape=(8,),
        state_global_offset=(3,),
        state_replica_id=(0,),
        checkpoint_metadata=build_dion_checkpoint_metadata(
            dp_size=8,
            fs_size=4,
            tp_size=2,
            rp_size=2,
            state_replica_size=2,
            requested_type="dp_reshardable",
            topology_signature=TOPOLOGY,
        ),
        sharded_object_cls=ShardedObject,
    )

    dion_state = state_dict["dion_param_state"]
    assert dion_state.global_shape == (8,)
    assert dion_state.global_offset == (3,)
    assert dion_state.replica_id == (0,)
    assert "layer.weight" in dion_state.data
    assert "param" in dion_state.data["layer.weight"]
    assert "_q_full_buffer" not in dion_state.data["layer.weight"]

    metadata = state_dict["dion_checkpoint_metadata"]
    assert metadata.global_shape == (1,)
    assert metadata.global_offset == (0,)
    assert metadata.replica_id == (0, 0, 3)
    assert metadata.data["param_state_format"] == checkpoint_io.DION_PARAM_STATE_FORMAT
    assert metadata.data["dp_size"] == 8
    assert metadata.data["topology_signature"] == TOPOLOGY


def test_restore_casts_to_current_state_dtype_and_marks_normal_q_for_sync():
    param = torch.nn.Parameter(torch.empty(2, 3, dtype=torch.float32))
    param.data.zero_()
    param._dion_param_uid = "layer.weight"
    optimizer_state = {
        param: {
            "momentum": torch.zeros(2, 3, dtype=torch.float32),
            "Q": torch.zeros(3, 1, dtype=torch.float32),
            "_needs_state_replica_q_sync": False,
        }
    }

    summary = restore_persistent_param_state_(
        param_groups=[{"params": [param]}],
        optimizer_state=optimizer_state,
        get_param_key=lambda tensor: tensor._dion_param_uid,
        key_to_state={
            "layer.weight": {
                "param": torch.full((2, 3), 7.0, dtype=torch.float32),
                "momentum": torch.ones(2, 3, dtype=torch.float16),
                "Q": torch.ones(3, 1, dtype=torch.float16),
                "r": 1,
            }
        },
        mixed_precision_config=DionMixedPrecisionConfig(
            momentum_dtype=torch.float16,
            q_dtype=torch.float16,
        ),
    )

    restored = optimizer_state[param]
    assert summary == {"restored": 1, "unnamed": 0, "no_payload_entry": 0}
    assert restored["momentum"].dtype == torch.float32
    assert restored["Q"].dtype == torch.float32
    assert restored["_needs_state_replica_q_sync"] is True
    assert torch.equal(param.detach(), torch.full((2, 3), 7.0))


def test_restore_rejects_missing_master_param_payload():
    param = torch.nn.Parameter(torch.empty(2, 3, dtype=torch.float32))
    param._dion_param_uid = "layer.weight"

    with pytest.raises(RuntimeError, match="missing optimizer master param"):
        restore_persistent_param_state_(
            param_groups=[{"params": [param]}],
            optimizer_state={param: {"Q": torch.zeros(3, 1), "r": 1}},
            get_param_key=lambda tensor: tensor._dion_param_uid,
            key_to_state={"layer.weight": {"Q": torch.ones(3, 1), "r": 1}},
            mixed_precision_config=DionMixedPrecisionConfig(),
        )


def test_dion_checkpoint_metadata_rejects_unsupported_fs_or_tp_changes():
    metadata = build_dion_checkpoint_metadata(
        dp_size=8,
        fs_size=2,
        tp_size=4,
        rp_size=1,
        state_replica_size=1,
        requested_type="dp_reshardable",
    )

    with pytest.raises(RuntimeError, match="DP topology change"):
        validate_dion_checkpoint_metadata(metadata, dp_size=4, fs_size=2, tp_size=4)
    with pytest.raises(RuntimeError, match="FS topology change"):
        validate_dion_checkpoint_metadata(metadata, dp_size=8, fs_size=3, tp_size=4)
    with pytest.raises(RuntimeError, match="TP topology change"):
        validate_dion_checkpoint_metadata(metadata, dp_size=8, fs_size=2, tp_size=2)


def test_dion_checkpoint_metadata_rejects_unsupported_rp_or_state_replica_changes():
    metadata = build_dion_checkpoint_metadata(
        dp_size=8,
        fs_size=2,
        tp_size=4,
        rp_size=2,
        state_replica_size=2,
        requested_type="dp_reshardable",
    )

    with pytest.raises(RuntimeError, match="RP topology change"):
        validate_dion_checkpoint_metadata(
            metadata,
            dp_size=8,
            fs_size=2,
            tp_size=4,
            rp_size=1,
            state_replica_size=2,
        )
    with pytest.raises(RuntimeError, match="state-replica topology change"):
        validate_dion_checkpoint_metadata(
            metadata,
            dp_size=8,
            fs_size=2,
            tp_size=4,
            rp_size=2,
            state_replica_size=1,
        )


def test_dion_checkpoint_metadata_rejects_same_size_topology_identity_change():
    metadata = build_dion_checkpoint_metadata(
        dp_size=8,
        fs_size=2,
        tp_size=4,
        rp_size=1,
        state_replica_size=1,
        requested_type="dp_reshardable",
        topology_signature={
            "data_parallel": (0, 1, 2, 3, 4, 5, 6, 7),
            "fs": (0, 1),
            "tp": (0, 1, 2, 3),
            "rp": (0,),
            "state_replica": (0,),
        },
    )

    with pytest.raises(RuntimeError, match="topology identity change"):
        validate_dion_checkpoint_metadata(
            metadata,
            dp_size=8,
            fs_size=2,
            tp_size=4,
            rp_size=1,
            state_replica_size=1,
            topology_signature={
                "data_parallel": (0, 1, 2, 3, 4, 5, 6, 7),
                "fs": (1, 0),
                "tp": (0, 1, 2, 3),
                "rp": (0,),
                "state_replica": (0,),
            },
        )


def test_dion_checkpoint_metadata_rejects_missing_metadata_for_replicated_state():
    with pytest.raises(RuntimeError, match="missing Dion optimizer metadata"):
        validate_dion_checkpoint_metadata(
            None,
            fs_size=1,
            tp_size=1,
            state_replica_size=2,
        )


def test_dion_checkpoint_sharding_type_rejects_tensor_resharding_formats():
    with pytest.raises(NotImplementedError, match="tensor-level Dion momentum/Q resharding"):
        resolve_dion_checkpoint_sharding_type(
            None,
            {"distrib_optim_sharding_type": "fully_reshardable"},
        )


def test_empty_flat_checkpoint_shard_still_enters_collectives(monkeypatch):
    calls = []
    expected = torch.arange(4, dtype=torch.float32)

    def fake_all_gather(outputs, input_tensor, group=None):
        del group
        calls.append(input_tensor.clone())
        if input_tensor.dtype == torch.long:
            ranges = ((0, 1), (1, 2), (2, 3), (3, 4), (4, 4))
            for output, range_pair in zip(outputs, ranges):
                output.copy_(torch.tensor(range_pair, dtype=torch.long))
        else:
            for index, output in enumerate(outputs):
                output.zero_()
                if index < expected.numel():
                    output[0].copy_(expected[index])

    monkeypatch.setattr(checkpoint_io.dist, "all_gather", fake_all_gather)

    param_flat = expected.clone()
    checkpoint_io.all_gather_flat_shards_(
        param_flat,
        flat_start=4,
        flat_end=4,
        group=None,
        world_size=5,
    )

    assert len(calls) == 2
    assert torch.equal(param_flat, expected)


def test_restore_full_flat_param_empty_range_still_enters_collectives(monkeypatch):
    calls = []
    expected = torch.arange(4, dtype=torch.float32)

    def fake_all_gather(outputs, input_tensor, group=None):
        del group
        calls.append(input_tensor.clone())
        if input_tensor.dtype == torch.long:
            ranges = ((0, 1), (1, 2), (2, 3), (4, 4))
            for output, range_pair in zip(outputs, ranges):
                output.copy_(torch.tensor(range_pair, dtype=torch.long))
        else:
            for index, output in enumerate(outputs):
                output.zero_()
                if index < 3:
                    output[0].copy_(expected[index])

    param = torch.nn.Parameter(expected.clone())
    monkeypatch.setattr(checkpoint_io.dist, "all_gather", fake_all_gather)

    restored = checkpoint_io.restore_full_model_param_(
        model_param=param,
        param_range=type("Range", (), {"start": 4, "end": 4})(),
        dion_shard_layout=None,
        fs_group=None,
        fs_size=4,
    )

    assert restored is True
    assert len(calls) == 2
    assert torch.equal(param.detach(), expected)


def test_empty_2d_checkpoint_shard_still_enters_collective(monkeypatch):
    calls = []
    expected = torch.arange(6, dtype=torch.float32).view(2, 3)

    def fake_all_gather(outputs, input_tensor, group=None):
        del group
        calls.append(input_tensor.clone())
        outputs[0].copy_(expected[0:1, :])
        outputs[1].copy_(expected[1:2, :])
        outputs[2].zero_()

    param_2d = expected.clone()
    monkeypatch.setattr(checkpoint_io.dist, "all_gather", fake_all_gather)
    checkpoint_io.all_gather_fs_shards_2d_(
        param_2d,
        fs_shard_dim=0,
        start_idx=2,
        end_idx=2,
        fs_group=None,
        fs_size=3,
    )

    assert len(calls) == 1
    assert torch.equal(param_2d, expected)
