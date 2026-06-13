"""Muon state helpers matching the local MCore backend invariant."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor

from ..matrix.parameter import is_matrix_param, is_vocab_param, prepare_matrix_params
from ..matrix.splits.gdn import (
    get_gdn_split_axis_from_dist_meta,
    get_gdn_split_shapes_from_dist_meta,
)
from ..matrix.splits.linear import (
    get_linear_child_kinds_from_dist_meta,
    get_linear_split_axis_from_dist_meta,
    get_linear_split_rows_from_dist_meta,
)
from ..matrix.splits.qkv import (
    get_qkv_split_axis_from_dist_meta,
    get_qkv_split_shapes_from_dist_meta,
)
from ..matrix.splits.qkvg import (
    get_qkvg_split_axis_from_dist_meta,
    get_qkvg_split_shapes_from_dist_meta,
)
from .kernels import muon_scale_factor
from .types import MuonDistMeta, MuonMixedPrecisionConfig, MuonParamConfig


def str_to_dtype(dtype_val) -> Optional[torch.dtype]:
    """Convert a string dtype name to ``torch.dtype``."""
    if dtype_val is None:
        return None
    if isinstance(dtype_val, torch.dtype):
        return dtype_val
    dtype_map = {
        "float32": torch.float32,
        "float": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if isinstance(dtype_val, str):
        dtype_lower = dtype_val.lower()
        if dtype_lower.startswith("torch."):
            dtype_lower = dtype_lower.split(".", 1)[1]
        if dtype_lower in dtype_map:
            return dtype_map[dtype_lower]
        raise ValueError(f"Unknown dtype string: {dtype_val}")
    return dtype_val


def get_global_shape(
    dist_meta: Optional[MuonDistMeta],
    m_local: int,
    n_local: int,
) -> Tuple[int, int]:
    """Return the logical Muon matrix shape for scaling and orientation."""
    if dist_meta is not None:
        per_expert_shape = getattr(dist_meta, "per_expert_global_shape", None)
        if per_expert_shape is not None:
            return tuple(int(dim) for dim in per_expert_shape)
        global_shape = getattr(dist_meta, "global_shape", None)
        if global_shape is not None:
            return tuple(int(dim) for dim in global_shape)
        if getattr(dist_meta, "is_muon_param", False):
            raise RuntimeError(
                "[MUON_MISSING_GLOBAL_SHAPE] "
                f"local_shape=({m_local}, {n_local}) "
                f"param_uid={getattr(dist_meta, 'param_uid', None)} "
                f"param_name={getattr(dist_meta, 'param_name', '')}"
            )
    return (int(m_local), int(n_local))


def get_local_shape(
    dist_meta: Optional[MuonDistMeta],
    m_local: int,
    n_local: int,
) -> Tuple[int, int]:
    """Return the object-local Muon matrix shape."""
    if dist_meta is not None and getattr(dist_meta, "local_shape", None) is not None:
        return tuple(int(dim) for dim in dist_meta.local_shape)
    return (int(m_local), int(n_local))


def require_2d_local_shape(param: Tensor, dist_meta: Optional[MuonDistMeta]) -> Tuple[int, int]:
    """Return the exact local 2D shard shape from metadata or the tensor shape."""
    if dist_meta is not None and dist_meta.shape is not None:
        if len(dist_meta.shape) != 2:
            raise RuntimeError(
                "[MUON_INVALID_LOCAL_SHAPE] "
                f"dist_meta_shape={dist_meta.shape} "
                f"param_uid={getattr(dist_meta, 'param_uid', None)} "
                f"param_name={getattr(dist_meta, 'param_name', '')}"
            )
        local_shape = tuple(int(dim) for dim in dist_meta.shape)
    elif param.ndim == 2:
        local_shape = tuple(int(dim) for dim in param.shape)
    else:
        raise RuntimeError(
            "[MUON_MISSING_LOCAL_SHAPE] "
            f"param_ndim={param.ndim} "
            f"param_uid={getattr(dist_meta, 'param_uid', None) if dist_meta is not None else None}"
        )

    m_local, n_local = local_shape
    if m_local <= 0 or n_local <= 0:
        raise RuntimeError(f"[MUON_EMPTY_LOCAL_SHAPE] local_shape={local_shape}")
    if int(param.numel()) != m_local * n_local:
        raise RuntimeError(
            "[MUON_LOCAL_SHAPE_NUMEL_MISMATCH] "
            f"local_shape={local_shape} numel={int(param.numel())}"
        )
    return local_shape


def is_muon_matrix_param(param: Tensor, dist_meta: Optional[MuonDistMeta] = None) -> bool:
    """Return whether a tensor should use Muon matrix math."""
    if getattr(param, "use_muon", None) is False:
        return False
    if dist_meta is not None and getattr(dist_meta, "is_muon_param", False):
        return True
    return is_matrix_param(param) and not is_vocab_param(param)


def prepare_muon_params(module: torch.nn.Module) -> None:
    """Prepare local parameters for Muon routing."""
    prepare_matrix_params(module)
    for param in module.parameters():
        param.use_muon = is_muon_matrix_param(param)


def init_matrix_state(
    param: Tensor,
    state: Dict[str, Any],
    mixed_precision_config: Optional[MuonMixedPrecisionConfig] = None,
) -> None:
    """Initialize Muon matrix state for local wrappers."""
    if mixed_precision_config is None:
        mixed_precision_config = MuonMixedPrecisionConfig()
    momentum_dtype = str_to_dtype(mixed_precision_config.momentum_dtype)
    if momentum_dtype is None:
        momentum_dtype = param.dtype
    if "momentum_buffer" not in state:
        state["momentum_buffer"] = torch.zeros_like(param, dtype=momentum_dtype)
    state.setdefault("local_shape", tuple(int(dim) for dim in param.shape))
    state.setdefault("global_shape", tuple(int(dim) for dim in param.shape))


def build_param_config(
    *,
    param_ndim: int,
    local_shape: Optional[Tuple[int, int]],
    dist_meta: Optional[MuonDistMeta],
    tp_world_size: int = 1,
    tp_active: bool = False,
    momentum_beta: float = 0.95,
    use_nesterov: bool = False,
    ns_backend: str = "standard",
    coefficient_type: str = "quintic",
    num_ns_steps: int = 5,
    ns_epsilon: float = 1e-7,
    gram_restart_iterations: Tuple[int, ...] = (2,),
    gram_kernel_policy: str = "torch",
    gram_dtype: Optional[torch.dtype | str] = None,
    scale_mode: str = "spectral",
    extra_scale_factor: float = 0.2,
    fs_mode: str = "blockwise",
    tp_mode: str = "blockwise",
    split_parameters: bool = False,
) -> MuonParamConfig:
    """Build one Muon parameter config from explicit metadata and defaults."""
    config = MuonParamConfig(
        momentum_beta=float(momentum_beta),
        use_nesterov=bool(use_nesterov),
        ns_backend=ns_backend,
        coefficient_type=coefficient_type,
        num_ns_steps=int(num_ns_steps),
        ns_epsilon=float(ns_epsilon),
        gram_restart_iterations=tuple(gram_restart_iterations),
        gram_kernel_policy=gram_kernel_policy,
        gram_dtype=gram_dtype,
        scale_mode=scale_mode,
        extra_scale_factor=float(extra_scale_factor),
        fs_mode=fs_mode,
        tp_mode=tp_mode,
        split_parameters=bool(split_parameters),
    )

    if param_ndim != 2 or local_shape is None:
        return config

    m_local, n_local = (int(local_shape[0]), int(local_shape[1]))
    global_shape = get_global_shape(dist_meta, m_local, n_local)
    config.is_transposed = int(global_shape[0]) > int(global_shape[1])
    config.scale_factor = muon_scale_factor(*global_shape, mode=scale_mode) * float(
        extra_scale_factor
    )

    if dist_meta is not None and getattr(dist_meta, "is_muon_param", False):
        tp_shard_dim = int(getattr(dist_meta, "tp_shard_dim", -1))
        if tp_shard_dim in (0, 1):
            config.has_tp_shard = True
            config.use_tp_shard = bool(tp_active and int(tp_world_size) > 1)
            config.tp_shard_dim = tp_shard_dim

        fs_shard_dim = int(getattr(dist_meta, "fs_shard_dim", -1))
        if fs_shard_dim in (0, 1):
            config.has_fs_shard = True
            config.use_fs_shard = int(getattr(dist_meta, "fs_world_size", 1)) > 1
            config.fs_shard_dim = fs_shard_dim

    return config


def init_param_state(
    *,
    param: Tensor,
    state: Dict[str, Any],
    optim_group: Dict[str, Any],
    mixed_precision_config: Optional[MuonMixedPrecisionConfig],
    config: MuonParamConfig,
    dist_meta: Optional[MuonDistMeta],
    is_muon_eligible: bool,
    local_shape: Optional[Tuple[int, int]],
    split_parameters_default: bool = False,
) -> None:
    """Initialize optimizer state for one matrix or scalar-fallback parameter."""
    if mixed_precision_config is None:
        mixed_precision_config = MuonMixedPrecisionConfig()

    algorithm = optim_group.get("algorithm", "muon")
    if algorithm == "muon" and is_muon_eligible and local_shape is not None:
        momentum_dtype = str_to_dtype(mixed_precision_config.momentum_dtype)
        if momentum_dtype is None:
            momentum_dtype = param.dtype
        state["momentum_buffer"] = torch.zeros_like(param, dtype=momentum_dtype)

        qkvg_split_shapes = get_qkvg_split_shapes_from_dist_meta(dist_meta)
        if bool(split_parameters_default) and qkvg_split_shapes is not None:
            state["qkvg_split_qkvg"] = True
            state["qkvg_split_shapes"] = qkvg_split_shapes
            state["qkvg_split_axis"] = get_qkvg_split_axis_from_dist_meta(dist_meta)
            return

        qkv_split_shapes = get_qkv_split_shapes_from_dist_meta(dist_meta)
        if bool(split_parameters_default) and qkv_split_shapes is not None:
            state["qkv_split_qkv"] = True
            state["qkv_split_shapes"] = qkv_split_shapes
            state["qkv_split_axis"] = get_qkv_split_axis_from_dist_meta(dist_meta)
            return

        gdn_split_shapes = get_gdn_split_shapes_from_dist_meta(dist_meta)
        if bool(split_parameters_default) and gdn_split_shapes is not None:
            state["gdn_split_gdn"] = True
            state["gdn_split_shapes"] = gdn_split_shapes
            state["gdn_split_axis"] = get_gdn_split_axis_from_dist_meta(dist_meta)
            return

        linear_split_rows = get_linear_split_rows_from_dist_meta(dist_meta)
        if bool(split_parameters_default) and linear_split_rows is not None:
            state["linear_split_linear"] = True
            state["linear_split_rows"] = linear_split_rows
            state["linear_child_kinds"] = get_linear_child_kinds_from_dist_meta(dist_meta)
            state["linear_split_axis"] = get_linear_split_axis_from_dist_meta(dist_meta)
            return

        m_local, n_local = (int(local_shape[0]), int(local_shape[1]))
        state["local_shape"] = (m_local, n_local)
        state["global_shape"] = get_global_shape(dist_meta, m_local, n_local)
        per_expert_shape = (
            getattr(dist_meta, "per_expert_global_shape", None) if dist_meta is not None else None
        )
        if per_expert_shape is not None:
            state["per_expert_global_shape"] = tuple(int(dim) for dim in per_expert_shape)
        return

    init_scalar_state(
        param=param,
        state=state,
        mixed_precision_config=mixed_precision_config,
    )


def init_scalar_state(
    param: Tensor,
    state: Dict[str, Any],
    mixed_precision_config: Optional[MuonMixedPrecisionConfig] = None,
    scalar_optimizer: str = "adamw",
) -> None:
    """Initialize scalar fallback state."""
    if mixed_precision_config is None:
        mixed_precision_config = MuonMixedPrecisionConfig()
    scalar_optimizer = str(scalar_optimizer).lower()
    if scalar_optimizer == "adam":
        scalar_optimizer = "adamw"
    momentum_dtype = str_to_dtype(mixed_precision_config.scalar_momentum_dtype)
    if momentum_dtype is None:
        momentum_dtype = param.dtype
    if "exp_avg" not in state:
        state["exp_avg"] = torch.zeros_like(param, dtype=momentum_dtype)
    if scalar_optimizer == "adamw" and "exp_avg_sq" not in state:
        variance_dtype = str_to_dtype(mixed_precision_config.scalar_variance_dtype)
        if variance_dtype is None:
            variance_dtype = param.dtype
        state["exp_avg_sq"] = torch.zeros_like(param, dtype=variance_dtype)
    state.setdefault("step", 0)


def state_backend_keys() -> tuple[str, ...]:
    return (
        "momentum",
        "momentum_buffer",
        "exp_avg",
        "exp_avg_sq",
        "local_shape",
        "global_shape",
        "per_expert_global_shape",
        "qkv_split_shapes",
        "qkv_split_axis",
        "qkvg_split_shapes",
        "qkvg_split_axis",
        "gdn_split_shapes",
        "gdn_split_axis",
        "linear_split_rows",
        "linear_child_kinds",
        "linear_split_axis",
    )


__all__ = [
    "build_param_config",
    "get_global_shape",
    "get_local_shape",
    "init_matrix_state",
    "init_param_state",
    "init_scalar_state",
    "is_muon_matrix_param",
    "prepare_muon_params",
    "require_2d_local_shape",
    "state_backend_keys",
    "str_to_dtype",
]
