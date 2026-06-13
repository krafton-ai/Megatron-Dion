import torch
from types import SimpleNamespace

from megatron.core.optimizer.muon.algorithm import MegatronMuon, build_muon_optimizer


def test_muon_scalar_lion_updates_only_first_moment():
    param = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float32))
    param.grad = torch.tensor([0.25, -0.5], dtype=torch.float32)

    optimizer = MegatronMuon(
        [param],
        lr=0.1,
        weight_decay=0.0,
        scalar_optimizer="lion",
        betas=(0.95, 0.98),
    )
    optimizer.step()

    state = optimizer.state[param]
    assert "exp_avg" in state
    assert "exp_avg_sq" not in state
    assert torch.allclose(param.detach(), torch.tensor([0.9, -1.9]))


def test_build_muon_optimizer_uses_lion_betas_for_scalar_lion():
    param = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
    config = SimpleNamespace(
        lr=0.1,
        muon_momentum=0.95,
        muon_use_nesterov=False,
        weight_decay=0.0,
        lion_beta1=0.95,
        lion_beta2=0.98,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_eps=1e-8,
        muon_split_parameters=True,
        muon_fp32_matmul_prec="medium",
        muon_num_ns_steps=5,
        muon_scale_mode="spectral",
        muon_extra_scale_factor=1.0,
        muon_tp_mode="distributed",
        muon_scalar_optimizer="lion",
    )

    optimizer = build_muon_optimizer(config=config, param_groups=[param])

    assert optimizer.defaults["scalar_optimizer"] == "lion"
    assert optimizer.defaults["betas"] == (0.95, 0.98)
