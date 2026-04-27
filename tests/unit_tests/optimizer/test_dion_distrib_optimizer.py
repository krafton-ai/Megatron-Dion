from types import SimpleNamespace

import pytest
import torch

from megatron.core.optimizer import dion_distrib_optimizer as dion_do
from megatron.core.optimizer.dion.kernels import scaled_lr_for_shape
from megatron.core.optimizer.dion.ortho import orthogonalize
from megatron.core.optimizer.distrib_dion import bootstrap as dion_bootstrap


def test_enable_distributed_dion_builds_metadata_for_single_rank(monkeypatch):
    """Single-rank distributed optimizer still needs Dion metadata and step routing."""

    class DummyOptimizer:
        def __init__(self):
            self.route_step_params = None

        def enable_distributed_mode(self, *, route_step_params):
            self.route_step_params = route_step_params

    optimizer = DummyOptimizer()
    metadata = {"built": object()}
    route_step_params = object()
    group = object()

    monkeypatch.setattr(dion_bootstrap.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dion_bootstrap.dist, "get_world_size", lambda group=None: 1)
    monkeypatch.setattr(dion_bootstrap.dist, "get_rank", lambda group=None: 0)
    monkeypatch.setattr(dion_bootstrap.dist, "get_process_group_ranks", lambda group: [0])
    monkeypatch.setattr(dion_bootstrap.dist, "all_reduce", lambda *args, **kwargs: None)
    monkeypatch.setattr(dion_bootstrap.torch.cuda, "current_device", lambda: "cpu")

    result = dion_bootstrap.enable_distributed_dion(
        optimizer=optimizer,
        global_rank=0,
        replica_group=None,
        data_parallel_group=group,
        tp_group=None,
        rp_group=None,
        fs_group=None,
        state_replica_group=None,
        expected_rp_size=1,
        build_all_dist_metas=lambda: metadata,
        route_step_params=route_step_params,
        group_size=lambda group: 1,
        group_rank=lambda group: 0,
        use_low_rank_sync=True,
        use_fs_collectives=True,
        validate_enabled_rp_topology=dion_bootstrap.validate_enabled_rp_topology,
        log_error=lambda *args, **kwargs: None,
    )

    assert result is metadata
    assert optimizer.route_step_params is route_step_params


def test_shard_param_uid_recovers_from_canonical_shard_metadata():
    wrapper = dion_do.DionDistributedOptimizer.__new__(dion_do.DionDistributedOptimizer)

    model_param = torch.nn.Parameter(torch.empty(4, 4))
    canonical_shard = model_param.detach().view(-1)[:8]
    replacement_shard = canonical_shard.clone()
    canonical_shard._model_param = model_param
    replacement_shard._model_param = model_param

    param_uid = ("decoder.layers.0.mlp.linear_fc2.weight", (4, 4), True)
    dist_meta = SimpleNamespace(param_uid=param_uid)
    wrapper.dist_metas = {canonical_shard: dist_meta}
    wrapper._dion_dist_meta_by_uid = {param_uid: dist_meta}
    wrapper._shards_by_param = {model_param: (canonical_shard, canonical_shard)}

    assert wrapper._shard_param_uid(replacement_shard) == param_uid
    assert replacement_shard._dion_param_uid == param_uid
    assert wrapper.dist_metas[replacement_shard] is dist_meta


@pytest.mark.skipif(not torch.cuda.is_available(), reason="BF16 QR regression requires CUDA")
def test_orthogonalize_promotes_bfloat16_tall_matrix_qr_inputs_to_fp32():
    """BF16 Dion state must not send BF16 tensors into CUDA QR kernels."""

    torch.manual_seed(1234)
    p = torch.randn(2, 32, 4, device="cuda", dtype=torch.bfloat16)

    q = orthogonalize(p, rcqr_oversample=1.25)

    assert q.dtype is torch.bfloat16
    assert q.shape == p.shape
    assert torch.isfinite(q.float()).all()


def test_scaled_lr_spectral_keeps_rank_fraction_scale():
    scaled_lr = scaled_lr_for_shape(
        lr=2.0,
        m_global=25,
        n_global=9,
        scale_mode="spectral",
        rank_fraction=0.25,
        extra_scale_factor=0.2,
    )

    assert scaled_lr == pytest.approx(2.0 * (0.2 / 0.5) * 5.0)
