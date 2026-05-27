import math

import pytest
import torch

import megatron.core.optimizer.muon.kernels as muon_kernels
from megatron.core.optimizer.muon.backend import MuonBackend
from megatron.core.optimizer.muon.kernels import (
    apply_muon_momentum,
    get_muon_scale_factor,
    gram_newton_schulz,
    newton_schulz,
    orthogonalize_muon_update,
    standard_newton_schulz,
)
from megatron.core.optimizer.muon.state import build_param_config, init_param_state
from megatron.core.optimizer.muon.types import (
    MuonBatchEntry,
    MuonDistMeta,
    MuonMixedPrecisionConfig,
    MuonParamConfig,
    MuonStepParam,
)


def _devices():
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    return devices


@pytest.mark.parametrize("device", _devices())
def test_muon_momentum_matches_reference_ema(device):
    grad = torch.tensor([[2.0, -4.0], [6.0, -8.0]], device=device)
    momentum = torch.tensor([[1.0, 3.0], [-5.0, 7.0]], device=device)
    old_momentum = momentum.clone()

    update = apply_muon_momentum(momentum, grad, beta=0.9, nesterov=False)

    expected = 0.9 * old_momentum + 0.1 * grad
    assert update.data_ptr() == momentum.data_ptr()
    torch.testing.assert_close(momentum, expected)
    torch.testing.assert_close(update, expected)


@pytest.mark.parametrize("device", _devices())
def test_muon_nesterov_matches_reference_lerp(device):
    grad = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
    momentum = torch.tensor([[4.0, 3.0], [2.0, 1.0]], device=device)
    old_momentum = momentum.clone()
    beta = 0.75

    update = apply_muon_momentum(momentum, grad, beta=beta, nesterov=True)

    expected_momentum = beta * old_momentum + (1.0 - beta) * grad
    expected_update = (1.0 - beta) * grad + beta * expected_momentum
    torch.testing.assert_close(momentum, expected_momentum)
    torch.testing.assert_close(update, expected_update)


def _manual_standard_ns(x, *, steps, coeff):
    X = x.float()
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT
    X = X / X.square().sum(dim=(-2, -1), keepdim=True).sqrt().clamp_min(1e-7)
    a, b, c = coeff
    for _ in range(steps):
        gram = X @ X.mT
        X = a * X + (b * gram + c * (gram @ gram)) @ X
    if transposed:
        X = X.mT
    return X.to(dtype=x.dtype)


@pytest.mark.parametrize("device", _devices())
def test_standard_newton_schulz_matches_reference_loop(device):
    x = torch.arange(1, 13, dtype=torch.float32, device=device).view(3, 4) / 13.0

    actual = standard_newton_schulz(x, steps=3, coefficient_type="simple")
    expected = _manual_standard_ns(x, steps=3, coeff=(3.4445, -4.7750, 2.0315))

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("shape", [(3, 7), (7, 3)])
@pytest.mark.parametrize("device", _devices())
def test_gram_newton_schulz_matches_standard_backend(shape, device):
    torch.manual_seed(1234)
    x = torch.randn(shape, dtype=torch.float32, device=device)

    standard = standard_newton_schulz(
        x,
        steps=5,
        coefficient_type="polar_express",
        fp32_matmul_prec="highest",
    )
    gram = gram_newton_schulz(
        x,
        steps=5,
        coefficient_type="polar_express",
        restart_iterations=(2,),
        gram_dtype=torch.float32,
        fp32_matmul_prec="highest",
    )

    torch.testing.assert_close(gram, standard, rtol=3e-4, atol=3e-4)


@pytest.mark.parametrize("shape", [(3, 7), (7, 3)])
@pytest.mark.parametrize("device", _devices())
def test_right_gram_newton_schulz_matches_left_gram(shape, device):
    torch.manual_seed(4321)
    x = torch.randn(shape, dtype=torch.float32, device=device)

    left = newton_schulz(
        x,
        steps=4,
        coefficient_type="quintic",
        fp32_matmul_prec="highest",
    )
    right = newton_schulz(
        x,
        steps=4,
        coefficient_type="quintic",
        fp32_matmul_prec="highest",
        gram_side="right",
    )

    torch.testing.assert_close(right, left, rtol=3e-4, atol=3e-4)


@pytest.mark.parametrize("shape", [(3, 7), (7, 3)])
@pytest.mark.parametrize("device", _devices())
def test_right_gram_backend_matches_standard_backend(shape, device):
    torch.manual_seed(5678)
    x = torch.randn(shape, dtype=torch.float32, device=device)

    standard = standard_newton_schulz(
        x,
        steps=5,
        coefficient_type="polar_express",
        fp32_matmul_prec="highest",
    )
    gram = gram_newton_schulz(
        x,
        steps=5,
        coefficient_type="polar_express",
        gram_dtype=torch.float32,
        fp32_matmul_prec="highest",
        gram_side="right",
    )

    torch.testing.assert_close(gram, standard, rtol=3e-4, atol=3e-4)


def test_gram_kernel_policy_dispatches_to_dao_backend(monkeypatch):
    calls = []

    class FakeDaoBackend:
        name = "dao"

        @staticmethod
        def sym_mm(a, b):
            calls.append("sym_mm")
            return a @ b

        @staticmethod
        def sym_baddmm(a, b, *, C, alpha=1.0, beta=1.0):
            calls.append("sym_baddmm")
            return torch.addmm(C, a, b, alpha=alpha, beta=beta)

        @staticmethod
        def mm(a, b):
            calls.append("mm")
            return a @ b

        @staticmethod
        def mm_add(a, b, *, C, beta=1.0):
            calls.append("mm_add")
            return torch.addmm(C, a, b, beta=beta)

    monkeypatch.setattr(muon_kernels, "_dao_gram_eligible", lambda x: True)
    monkeypatch.setattr(muon_kernels, "_DAO_GRAM_BACKEND", FakeDaoBackend())
    monkeypatch.setattr(muon_kernels, "_DAO_GRAM_IMPORT_ERROR", None)

    torch.manual_seed(2468)
    x = torch.randn(4, 8, dtype=torch.float32)
    expected = gram_newton_schulz(
        x,
        steps=3,
        coefficient_type="simple",
        gram_dtype=torch.float32,
        gram_kernel_policy="torch",
    )
    actual = gram_newton_schulz(
        x,
        steps=3,
        coefficient_type="simple",
        gram_dtype=torch.float32,
        gram_kernel_policy="dao",
    )

    assert "sym_baddmm" in calls
    torch.testing.assert_close(actual, expected)


def test_gram_kernel_policy_auto_falls_back_when_dao_unavailable(monkeypatch):
    monkeypatch.setattr(muon_kernels, "_dao_gram_eligible", lambda x: True)
    monkeypatch.setattr(muon_kernels, "_DAO_GRAM_BACKEND", None)
    monkeypatch.setattr(muon_kernels, "_DAO_GRAM_IMPORT_ERROR", ImportError("missing quack"))

    x = torch.randn(4, 8, dtype=torch.float32)
    gram_newton_schulz(
        x,
        steps=1,
        coefficient_type="simple",
        gram_dtype=torch.float32,
        gram_kernel_policy="auto",
    )
    with pytest.raises(RuntimeError, match="MUON_DAO_GRAM_BACKEND_UNAVAILABLE"):
        gram_newton_schulz(
            x,
            steps=1,
            coefficient_type="simple",
            gram_dtype=torch.float32,
            gram_kernel_policy="dao",
        )


def test_muon_scale_factors_match_reference_formulas():
    assert get_muon_scale_factor(4, 16, mode="spectral") == math.sqrt(16)
    assert get_muon_scale_factor(4, 16, mode="unit_rms_norm") == math.sqrt(4 / 16)
    assert get_muon_scale_factor(4, 16, mode="shape_scaling") == 1.0
    assert get_muon_scale_factor(16, 4, mode="shape_scaling") == 2.0

    with pytest.raises(ValueError):
        get_muon_scale_factor(4, 16, mode="bad")


@pytest.mark.parametrize("device", _devices())
def test_orthogonalize_update_uses_logical_global_shape_for_scale(device):
    update = torch.randn(2, 4, dtype=torch.float32, device=device)

    local_scaled = orthogonalize_muon_update(
        update,
        coefficient_type="simple",
        num_ns_steps=1,
        scale_mode="spectral",
        extra_scale_factor=1.0,
        global_shape=(2, 4),
    )
    global_scaled = orthogonalize_muon_update(
        update,
        coefficient_type="simple",
        num_ns_steps=1,
        scale_mode="spectral",
        extra_scale_factor=1.0,
        global_shape=(8, 4),
    )

    torch.testing.assert_close(global_scaled, local_scaled * (math.sqrt(8) / math.sqrt(4)))


def test_muon_dataclasses_are_matrix_native_without_dion_fields():
    config = MuonParamConfig()
    assert not hasattr(config, "rank" + "_fraction")
    assert not hasattr(config, "use_low_" + "rank_sync")

    dist_meta = MuonDistMeta(is_muon_param=True, global_shape=(8, 4), shape=(2, 4))
    assert dist_meta.is_matrix_param

    entry = MuonBatchEntry(
        param=torch.zeros(2, 4),
        grad=torch.ones(2, 4),
        optimizer_state={},
        config=config,
        dist_meta=dist_meta,
        param_shape=(2, 4),
        global_shape=(8, 4),
    )
    batch = MuonBackend().build_batches(
        object(),
        [
            MuonStepParam(
                param=entry.param,
                grad=entry.grad,
                optimizer_state=entry.optimizer_state,
                optim_group={"lr": 0.1},
                config=config,
                dist_meta=dist_meta,
            )
        ],
    )[0]
    assert batch.real_batch_size == 1
    assert batch.params[0] is entry.param
    assert batch.grads[0] is entry.grad
    assert batch.global_shapes == ((8, 4),)


@pytest.mark.parametrize("device", _devices())
def test_state_helpers_initialize_muon_and_scalar_state(device):
    param = torch.zeros(2, 3, dtype=torch.float32, device=device)
    dist_meta = MuonDistMeta(
        is_muon_param=True,
        shape=(2, 3),
        global_shape=(4, 3),
        param_uid=("param",),
    )
    config = build_param_config(
        param_ndim=param.ndim,
        local_shape=(2, 3),
        dist_meta=dist_meta,
        momentum_beta=0.9,
        num_ns_steps=5,
    )
    state = {}

    init_param_state(
        param=param,
        state=state,
        optim_group={"algorithm": "muon"},
        mixed_precision_config=MuonMixedPrecisionConfig(),
        config=config,
        dist_meta=dist_meta,
        is_muon_eligible=True,
        local_shape=(2, 3),
    )

    assert state["local_shape"] == (2, 3)
    assert state["global_shape"] == (4, 3)
    torch.testing.assert_close(state["momentum_buffer"], torch.zeros_like(param))

    scalar = torch.zeros(3, dtype=torch.float32, device=device)
    scalar_state = {}
    init_param_state(
        param=scalar,
        state=scalar_state,
        optim_group={"algorithm": "scalar"},
        mixed_precision_config=MuonMixedPrecisionConfig(),
        config=config,
        dist_meta=None,
        is_muon_eligible=False,
        local_shape=None,
    )
    assert set(("exp_avg", "exp_avg_sq", "step")).issubset(scalar_state)
