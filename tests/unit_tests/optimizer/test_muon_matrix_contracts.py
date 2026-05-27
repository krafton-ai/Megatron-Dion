import ast
import importlib
import inspect
from pathlib import Path

import pytest
import torch

from megatron.core.optimizer.matrix.backend import MatrixBackend, MatrixStateSpec
from megatron.core.optimizer.matrix.splits import qkv as matrix_qkv
from megatron.core.optimizer.matrix.splits import qkvg as matrix_qkvg
from megatron.core.optimizer.matrix.types import MatrixDistMeta, MatrixStepParam


REPO_ROOT = Path(__file__).resolve().parents[3]


def _python_files(root: Path):
    return sorted(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)


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


def _optimizer_class(algorithm_module):
    for name in ("TensorParallelMuon", "MegatronMuon", "Muon"):
        optimizer_cls = getattr(algorithm_module, name, None)
        if optimizer_cls is not None:
            return optimizer_cls
    raise AssertionError(
        "Muon algorithm module must expose a local torch optimizer class "
        "(TensorParallelMuon, MegatronMuon, or Muon)"
    )


def _make_local_muon(param, *, lr, beta, extra_scale_factor=1.0):
    algorithm = _import_required("megatron.core.optimizer.muon.algorithm")
    optimizer_cls = _optimizer_class(algorithm)
    return _call_with_supported_kwargs(
        optimizer_cls,
        params=[param],
        lr=lr,
        momentum_beta=beta,
        momentum=beta,
        mu=beta,
        use_nesterov=False,
        nesterov=False,
        weight_decay=0.0,
        use_decoupled_weight_decay=True,
        split_qkv=False,
        split_linear=False,
        fp32_matmul_prec="highest",
        coefficient_type="quintic",
        num_ns_steps=5,
        scale_mode="spectral",
        extra_scale_factor=extra_scale_factor,
        pg_collection=None,
        mode="blockwise",
        fs_mode="blockwise",
        tp_mode="blockwise",
        ns_backend="standard",
    )


def _state_momentum(state):
    for key in ("momentum", "momentum_buffer"):
        value = state.get(key)
        if isinstance(value, torch.Tensor):
            return value
    raise AssertionError(f"Muon optimizer state is missing a momentum tensor: {sorted(state)}")


def _state_for_param(optimizer, param):
    state = getattr(optimizer, "state", None)
    if state is None:
        raise AssertionError("Muon optimizer must expose torch.optim-style state")
    if param not in state:
        raise AssertionError("Muon optimizer did not initialize state for the stepped parameter")
    return state[param]


def _scaled_orthogonalize(optimizer, param, momentum):
    if hasattr(optimizer, "orthogonalize"):
        return optimizer.orthogonalize(param, momentum.clone())

    kernels = _import_required("megatron.core.optimizer.muon.kernels")
    for name in (
        "scaled_orthogonalize",
        "scaled_newton_schulz",
        "orthogonalize",
        "standard_newton_schulz",
        "newton_schulz",
    ):
        fn = getattr(kernels, name, None)
        if fn is None:
            continue
        return _call_with_supported_kwargs(
            fn,
            tensor=momentum.clone(),
            grad=momentum.clone(),
            update=momentum.clone(),
            steps=5,
            num_steps=5,
            num_ns_steps=5,
            coefficient_type="quintic",
            scale_mode="spectral",
            extra_scale_factor=1.0,
            global_shape=tuple(momentum.shape),
        )

    raise AssertionError("Muon kernels must expose a standard Newton-Schulz update helper")


def _muon_scale_factor(global_shape, *, scale_mode="spectral", extra_scale_factor=1.0):
    kernels = _import_required("megatron.core.optimizer.muon.kernels")
    for name in ("get_muon_scale_factor", "muon_scale_factor", "compute_muon_scale_factor"):
        fn = getattr(kernels, name, None)
        if fn is None:
            continue
        base = _call_with_supported_kwargs(
            fn,
            m=int(global_shape[0]),
            n=int(global_shape[1]),
            rows=int(global_shape[0]),
            cols=int(global_shape[1]),
            shape=tuple(int(dim) for dim in global_shape),
            mode=scale_mode,
            scale_mode=scale_mode,
        )
        return float(base) * float(extra_scale_factor)
    raise AssertionError("Muon kernels must expose a scale-factor helper")


def _config_scale(config):
    for name in ("scale_factor", "muon_scale_factor", "lr_scale", "matrix_scale"):
        if hasattr(config, name):
            return float(getattr(config, name))
    raise AssertionError(f"Muon child config is missing a stored scale factor: {config!r}")


def _make_dist_meta(kind, *, split_shapes, shape):
    types = _import_required("megatron.core.optimizer.muon.types")
    dist_meta_cls = getattr(types, "MuonDistMeta", None)
    if dist_meta_cls is None:
        raise AssertionError("Muon types module must expose MuonDistMeta")

    split_kwargs = {}
    if kind == "qkv":
        split_kwargs["qkv_split_shapes"] = split_shapes
    elif kind == "qkvg":
        split_kwargs["qkvg_split_shapes"] = split_shapes
    else:
        raise AssertionError(f"unsupported split kind: {kind}")

    return _call_with_supported_kwargs(
        dist_meta_cls,
        shape=shape,
        local_shape=shape,
        global_shape=shape,
        fs_shard_dim=1,
        fs_world_size=1,
        fs_rank=-1,
        fs_start_idx=0,
        fs_end_idx=shape[1],
        tp_shard_dim=-1,
        tp_world_size=1,
        tp_rank=-1,
        is_matrix_param=True,
        is_muon_param=True,
        param_uid=(f"{kind}_parent",),
        param_name=f"decoder.layers.0.self_attention.{kind}.weight",
        **split_kwargs,
    )


def _make_distributed_optimizer_stub():
    distributed_optimizer = _import_required(
        "megatron.core.optimizer.muon.distributed.optimizer"
    )
    optimizer_cls = getattr(distributed_optimizer, "DistributedMuonOptimizer", None)
    if optimizer_cls is None:
        raise AssertionError(
            "Muon distributed optimizer module must expose DistributedMuonOptimizer"
        )
    optimizer = object.__new__(optimizer_cls)
    optimizer.optimizer = type(
        "_MuonInner",
        (),
        {
            "defaults": {
                "algorithm": "muon",
                "split_qkv": True,
                "split_linear": False,
                "scale_mode": "spectral",
                "extra_scale_factor": 0.25,
            }
        },
    )()
    return optimizer


def _expand_split_children(optimizer, kind, *, param, grad, state, dist_meta):
    method_names = {
        "qkv": ("_expand_split_qkv_params", "_expand_split_muon_params"),
        "qkvg": ("_expand_split_qkvg_params", "_expand_split_muon_params"),
    }[kind]
    for method_name in method_names:
        method = getattr(optimizer, method_name, None)
        if method is None:
            continue
        children = method(
            param=param,
            grad=grad,
            optimizer_state=state,
            optim_group={"algorithm": "muon"},
            config=None,
            dist_meta=dist_meta,
        )
        if children is not None:
            return children
    raise AssertionError(f"DistributedMuonOptimizer must expand fused {kind.upper()} children")


def _child_config(child):
    if child.config is not None:
        return child.config
    config = getattr(child.dist_meta, "param_config", None)
    if config is not None:
        return config
    raise AssertionError("Muon split child is missing param config")


def _child_kind(child, kind):
    attr = "qkv_child_kind" if kind == "qkv" else "qkvg_child_kind"
    value = getattr(child.dist_meta, attr, "")
    if value:
        return value
    raise AssertionError(f"Muon split child metadata is missing {attr}")


def _expected_child(module, parent, split_shapes, child_kind):
    if module is matrix_qkv:
        return matrix_qkv.extract_qkv_child(parent, split_shapes, child_kind)
    return matrix_qkvg.extract_qkvg_child(parent, split_shapes, child_kind)


def test_matrix_package_has_no_muon_or_dion_dependency():
    matrix_root = REPO_ROOT / "megatron" / "core" / "optimizer" / "matrix"
    forbidden_tokens = (
        "megatron.core.optimizer.dion",
        "megatron.core.optimizer.muon",
        "..dion",
        "..muon",
        ".dion",
        ".muon",
        "Dion",
        "DION",
        "dion",
        "Muon",
        "MUON",
        "muon",
        "rank_fraction",
        "use_low_rank_sync",
    )

    offenders = []
    for path in _python_files(matrix_root):
        text = path.read_text()
        for token in forbidden_tokens:
            if token in text:
                offenders.append((path.relative_to(matrix_root), token))

    assert offenders == []


def test_muon_package_has_no_dion_dependency():
    muon_root = REPO_ROOT / "megatron" / "core" / "optimizer" / "muon"
    assert muon_root.is_dir(), "MCore-native Muon package must live under optimizer/muon/"

    forbidden_imports = []
    forbidden_text = []
    for path in _python_files(muon_root):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if "dion" in alias.name.lower():
                        forbidden_imports.append((path.relative_to(muon_root), alias.name))
            elif isinstance(node, ast.ImportFrom):
                module = "." * int(node.level) + (node.module or "")
                imported_names = [alias.name for alias in node.names]
                if "dion" in module.lower() or any(
                    "dion" in imported.lower() for imported in imported_names
                ):
                    forbidden_imports.append((path.relative_to(muon_root), module))

        text = path.read_text()
        for token in (
            "rank_fraction",
            "use_low_rank_sync",
            "DionBackend",
            "DistributedDion",
            "DionDistMeta",
            "DionStepParam",
        ):
            if token in text:
                forbidden_text.append((path.relative_to(muon_root), token))

    assert forbidden_imports == []
    assert forbidden_text == []


def test_muon_package_does_not_import_dtensor_or_device_mesh():
    muon_root = REPO_ROOT / "megatron" / "core" / "optimizer" / "muon"
    forbidden = {
        "DTensor",
        "DeviceMesh",
        "Placement",
        "Shard",
        "Replicate",
        "Partial",
    }

    offenders = []
    for path in _python_files(muon_root):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if (
                        alias.name == "torch.distributed._tensor"
                        or alias.name.startswith("torch.distributed._tensor.")
                        or alias.name == "torch.distributed.device_mesh"
                    ):
                        offenders.append((path.relative_to(muon_root), alias.name))
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                imported_names = {alias.name for alias in node.names}
                if module.startswith("torch.distributed._tensor") or module == (
                    "torch.distributed.device_mesh"
                ):
                    offenders.append((path.relative_to(muon_root), module))
                for name in imported_names & forbidden:
                    offenders.append((path.relative_to(muon_root), name))

    assert offenders == []


def test_muon_backend_and_types_extend_matrix_contracts():
    backend_module = _import_required("megatron.core.optimizer.muon.backend")
    types_module = _import_required("megatron.core.optimizer.muon.types")

    assert issubclass(backend_module.MuonBackend, MatrixBackend)
    assert issubclass(types_module.MuonStepParam, MatrixStepParam)
    assert issubclass(types_module.MuonDistMeta, MatrixDistMeta)

    backend = backend_module.MuonBackend()
    assert backend.name == "muon"
    assert backend.supports_fs
    assert backend.supports_tp
    assert backend.supports_expert_parallel
    assert backend.supports_split_qkv
    assert backend.supports_split_qkvg
    assert isinstance(backend.state_spec(), MatrixStateSpec)


def test_muon_checkpoint_state_spec_is_momentum_only_for_matrix_params():
    backend_module = _import_required("megatron.core.optimizer.muon.backend")

    spec = backend_module.MuonBackend().state_spec()

    assert spec.backend == "muon"
    assert "momentum" in spec.state_keys
    forbidden = {
        "Q",
        "R",
        "r",
        "rank",
        "rank_fraction",
        "use_low_rank_sync",
        "q_tensor",
        "low_rank",
    }
    assert forbidden.isdisjoint(set(spec.state_keys))


def test_local_unsharded_muon_step_matches_reference_ema_math():
    torch.manual_seed(1234)
    lr = 0.05
    beta = 0.8
    param = torch.nn.Parameter(
        torch.tensor(
            [[0.5, -1.0, 1.5], [2.0, -0.5, 0.25]],
            dtype=torch.float32,
        )
    )
    optimizer = _make_local_muon(param, lr=lr, beta=beta)
    grads = (
        torch.tensor([[1.0, -2.0, 0.5], [0.25, 1.5, -1.0]], dtype=torch.float32),
        torch.tensor([[-0.5, 0.75, 2.0], [1.25, -1.5, 0.0]], dtype=torch.float32),
    )
    momentum = torch.zeros_like(param)

    for grad in grads:
        before = param.detach().clone()
        momentum = beta * momentum + (1.0 - beta) * grad
        expected_update = _scaled_orthogonalize(optimizer, param, momentum)

        param.grad = grad.clone()
        optimizer.step()

        assert torch.allclose(_state_momentum(_state_for_param(optimizer, param)), momentum)
        assert torch.allclose(
            param.detach(),
            before - lr * expected_update,
            atol=1e-5,
            rtol=1e-5,
        )


@pytest.mark.parametrize(
    "kind,module,split_shapes,expected_kinds",
    (
        ("qkv", matrix_qkv, (2, 1, 1), ("q", "k", "v")),
        ("qkvg", matrix_qkvg, (2, 2, 1, 1), ("q", "gate", "k", "v")),
    ),
)
def test_fused_attention_split_children_use_child_shape_and_scale(
    kind,
    module,
    split_shapes,
    expected_kinds,
):
    optimizer = _make_distributed_optimizer_stub()
    rows = 12 if kind == "qkvg" else 8
    shape = (rows, 4)
    parent = torch.nn.Parameter(torch.arange(rows * 4, dtype=torch.float32).view(rows, 4))
    grad = torch.arange(1000, 1000 + rows * 4, dtype=torch.float32).view(rows, 4)
    state = {
        "momentum": torch.arange(2000, 2000 + rows * 4, dtype=torch.float32).view(rows, 4),
    }
    if kind == "qkv":
        state.update({"qkv_split_qkv": True, "qkv_split_shapes": split_shapes})
    else:
        state.update({"qkvg_split_qkvg": True, "qkvg_split_shapes": split_shapes})
    dist_meta = _make_dist_meta(kind, split_shapes=split_shapes, shape=shape)

    children = _expand_split_children(
        optimizer,
        kind,
        param=parent,
        grad=grad,
        state=state,
        dist_meta=dist_meta,
    )

    assert [_child_kind(child, kind) for child in children] == list(expected_kinds)
    for child in children:
        child_kind = _child_kind(child, kind)
        expected_global_shape = getattr(module, f"{kind}_child_global_shape")(
            shape,
            split_shapes,
            child_kind,
        )
        expected_local_shape = getattr(module, f"{kind}_child_local_shape")(
            shape,
            split_shapes,
            child_kind,
            dist_meta=dist_meta,
        )
        assert tuple(child.param.shape) == expected_local_shape
        assert tuple(child.grad.shape) == expected_local_shape
        assert tuple(child.dist_meta.local_shape) == expected_local_shape
        assert tuple(child.dist_meta.global_shape) == expected_global_shape
        assert torch.equal(
            child.param,
            _expected_child(module, parent.detach(), split_shapes, child_kind),
        )
        assert torch.equal(child.grad, _expected_child(module, grad, split_shapes, child_kind))
        assert torch.equal(
            _state_momentum(child.optimizer_state),
            _expected_child(module, state["momentum"], split_shapes, child_kind),
        )
        assert _config_scale(_child_config(child)) == pytest.approx(
            _muon_scale_factor(
                expected_global_shape,
                scale_mode="spectral",
                extra_scale_factor=0.25,
            )
        )

    with torch.no_grad():
        for child in children:
            child.commit_update(
                torch.full_like(child.param, -7.0),
                torch.full_like(_state_momentum(child.optimizer_state), -9.0),
            )

    for child_kind in expected_kinds:
        expected_shape = getattr(module, f"{kind}_child_global_shape")(
            shape,
            split_shapes,
            child_kind,
        )
        assert torch.equal(
            _expected_child(module, parent.detach(), split_shapes, child_kind),
            torch.full(expected_shape, -7.0),
        )
        assert torch.equal(
            _expected_child(module, state["momentum"], split_shapes, child_kind),
            torch.full(expected_shape, -9.0),
        )
