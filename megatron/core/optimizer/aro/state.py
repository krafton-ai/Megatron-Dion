"""ARO state and parameter-routing helpers."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor

from ..matrix.parameter import is_matrix_param, mark_matrix_bucket_params, prepare_matrix_params
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
from ..matrix.utils import str_to_dtype
from .kernels import choose_orientation, oriented_shape
from .types import AroDistMeta, AroMixedPrecisionConfig, AroParamConfig


def get_global_shape(
    dist_meta: Optional[AroDistMeta],
    m_local: int,
    n_local: int,
) -> Tuple[int, int]:
    """Return the logical ARO matrix shape for orientation and rotation state."""
    if dist_meta is not None:
        per_expert_shape = getattr(dist_meta, "per_expert_global_shape", None)
        if per_expert_shape is not None:
            return tuple(int(dim) for dim in per_expert_shape)
        global_shape = getattr(dist_meta, "global_shape", None)
        if global_shape is not None:
            return tuple(int(dim) for dim in global_shape)
        if getattr(dist_meta, "is_aro_param", False):
            raise RuntimeError(
                "[ARO_MISSING_GLOBAL_SHAPE] "
                f"local_shape=({m_local}, {n_local}) "
                f"param_uid={getattr(dist_meta, 'param_uid', None)} "
                f"param_name={getattr(dist_meta, 'param_name', '')}"
            )
    return int(m_local), int(n_local)


def require_2d_local_shape(param: Tensor, dist_meta: Optional[AroDistMeta]) -> Tuple[int, int]:
    """Return the exact local 2D shard shape from metadata or the tensor shape."""
    if dist_meta is not None and dist_meta.shape is not None:
        if len(dist_meta.shape) != 2:
            raise RuntimeError(
                "[ARO_INVALID_LOCAL_SHAPE] "
                f"dist_meta_shape={dist_meta.shape} "
                f"param_uid={getattr(dist_meta, 'param_uid', None)} "
                f"param_name={getattr(dist_meta, 'param_name', '')}"
            )
        local_shape = tuple(int(dim) for dim in dist_meta.shape)
    elif param.ndim == 2:
        local_shape = tuple(int(dim) for dim in param.shape)
    else:
        raise RuntimeError(
            "[ARO_MISSING_LOCAL_SHAPE] "
            f"param_ndim={param.ndim} "
            f"param_uid={getattr(dist_meta, 'param_uid', None) if dist_meta is not None else None}"
        )
    if int(param.numel()) != int(local_shape[0]) * int(local_shape[1]):
        raise RuntimeError(
            "[ARO_LOCAL_SHAPE_NUMEL_MISMATCH] "
            f"local_shape={local_shape} numel={int(param.numel())}"
        )
    return local_shape


def is_aro_matrix_param(
    param: Tensor,
    dist_meta: Optional[AroDistMeta] = None,
    *,
    param_name: Optional[str] = None,
) -> bool:
    """Return whether a tensor should use ARO matrix math."""
    if getattr(param, "use_aro", None) is False:
        return False
    if dist_meta is not None and getattr(dist_meta, "is_aro_param", False):
        return True
    return is_matrix_param(param, param_name)


def prepare_aro_params(module: torch.nn.Module) -> None:
    """Prepare local parameters for ARO routing."""
    prepare_matrix_params(module)
    for param in module.parameters():
        param.use_aro = is_aro_matrix_param(param)


def mark_aro_bucket_params(param_map, param_to_name, fs_size: int, *, tp_group=None):
    """Classify bucket params and build static ARO metadata once."""
    aro_param_count, aro_info_by_param = mark_matrix_bucket_params(
        param_map=param_map,
        param_to_name=param_to_name,
        fs_size=fs_size,
        include_vocab=True,
        tp_group=tp_group,
    )
    for param in param_map.keys():
        use_aro = (
            getattr(param, "use_aro", None) is not False
            and bool(getattr(param, "is_matrix_param", False))
        )
        param.is_aro_param = bool(use_aro)
        param.use_aro = bool(use_aro)
        param.is_matrix_param = bool(param.is_aro_param)
        if not param.is_aro_param and param in aro_info_by_param:
            aro_info_by_param.pop(param, None)
            aro_param_count = max(0, int(aro_param_count) - 1)
    return aro_param_count, aro_info_by_param


def _rotation_shard_dim(orientation: str, shard_dim: int) -> int:
    shard_dim = int(shard_dim)
    if shard_dim not in (0, 1):
        return -1
    if orientation == "normal":
        return shard_dim
    if orientation == "transpose":
        return 1 - shard_dim
    raise RuntimeError(f"[ARO_INVALID_ORIENTATION] orientation={orientation!r}")


def rotation_local_rows(
    *,
    local_shape: Tuple[int, int],
    global_shape: Tuple[int, int],
    orientation: str,
    fs_shard_dim: int = -1,
    tp_shard_dim: int = -1,
) -> int:
    """Return the number of local rows held by the rotation state."""
    oriented_local = oriented_shape(local_shape, orientation)
    oriented_global = oriented_shape(global_shape, orientation)
    fs_dim = _rotation_shard_dim(orientation, fs_shard_dim)
    tp_dim = _rotation_shard_dim(orientation, tp_shard_dim)
    if fs_dim == 0 or tp_dim == 0:
        return int(oriented_local[0])
    return int(oriented_global[0])


def rotation_shape(
    *,
    local_shape: Tuple[int, int],
    global_shape: Tuple[int, int],
    orientation: str,
    fs_shard_dim: int = -1,
    tp_shard_dim: int = -1,
) -> Tuple[int, int]:
    global_rows = int(oriented_shape(global_shape, orientation)[0])
    local_rows = rotation_local_rows(
        local_shape=local_shape,
        global_shape=global_shape,
        orientation=orientation,
        fs_shard_dim=fs_shard_dim,
        tp_shard_dim=tp_shard_dim,
    )
    return int(local_rows), int(global_rows)


def _identity_rows(
    *,
    shape: Tuple[int, int],
    device: torch.device,
    dtype: torch.dtype,
    row_start: int = 0,
) -> Tensor:
    local_rows, global_rows = int(shape[0]), int(shape[1])
    result = torch.zeros((local_rows, global_rows), dtype=dtype, device=device)
    if local_rows <= 0:
        return result
    row_ids = torch.arange(local_rows, device=device)
    col_ids = row_ids + int(row_start)
    valid = col_ids < global_rows
    result[row_ids[valid], col_ids[valid]] = 1.0
    return result


def _rotation_row_start(dist_meta: Optional[AroDistMeta], orientation: str) -> int:
    if dist_meta is None:
        return 0
    fs_shard_dim = int(getattr(dist_meta, "fs_shard_dim", -1))
    fs_dim = _rotation_shard_dim(orientation, fs_shard_dim)
    if fs_dim == 0 and int(getattr(dist_meta, "fs_world_size", 1)) > 1:
        if int(getattr(dist_meta, "row_shard_start_idx", -1)) >= 0:
            return int(getattr(dist_meta, "row_shard_start_idx"))
        return int(getattr(dist_meta, "fs_start_idx", 0))
    tp_shard_dim = int(getattr(dist_meta, "tp_shard_dim", -1))
    tp_dim = _rotation_shard_dim(orientation, tp_shard_dim)
    if tp_dim == 0 and int(getattr(dist_meta, "tp_world_size", 1)) > 1:
        if int(getattr(dist_meta, "row_shard_start_idx", -1)) >= 0:
            return int(getattr(dist_meta, "row_shard_start_idx"))
        global_rows = int(oriented_shape(getattr(dist_meta, "global_shape"), orientation)[0])
        world = int(getattr(dist_meta, "tp_world_size", 1))
        rank = int(getattr(dist_meta, "tp_rank", 0))
        base = global_rows // world
        rem = global_rows % world
        return rank * base + min(rank, rem)
    return 0


def init_matrix_state(
    param: Tensor,
    state: Dict[str, Any],
    *,
    dist_meta: Optional[AroDistMeta] = None,
    mixed_precision_config: Optional[AroMixedPrecisionConfig] = None,
    init_rotation: bool = True,
) -> None:
    """Initialize ARO matrix state for a local or distributed parameter shard."""
    if mixed_precision_config is None:
        mixed_precision_config = AroMixedPrecisionConfig()
    local_shape = require_2d_local_shape(param, dist_meta)
    global_shape = get_global_shape(dist_meta, *local_shape)
    orientation = (
        getattr(dist_meta, "orientation", None)
        if dist_meta is not None and getattr(dist_meta, "orientation", None) is not None
        else choose_orientation(global_shape)
    )
    momentum_dtype = str_to_dtype(mixed_precision_config.momentum_dtype)
    if momentum_dtype is None:
        momentum_dtype = param.dtype
    momentum = state.get("momentum")
    if momentum is None or tuple(momentum.shape) != tuple(local_shape):
        state["momentum"] = torch.zeros(local_shape, dtype=momentum_dtype, device=param.device)
    state["local_shape"] = tuple(int(dim) for dim in local_shape)
    state["global_shape"] = tuple(int(dim) for dim in global_shape)
    state["orientation"] = str(orientation)
    if dist_meta is not None:
        split_shapes = get_qkv_split_shapes_from_dist_meta(dist_meta)
        if split_shapes is not None:
            state["qkv_split_shapes"] = tuple(int(dim) for dim in split_shapes)
            state["qkv_split_axis"] = get_qkv_split_axis_from_dist_meta(dist_meta)
        split_shapes = get_qkvg_split_shapes_from_dist_meta(dist_meta)
        if split_shapes is not None:
            state["qkvg_split_shapes"] = tuple(int(dim) for dim in split_shapes)
            state["qkvg_split_axis"] = get_qkvg_split_axis_from_dist_meta(dist_meta)
        split_shapes = get_gdn_split_shapes_from_dist_meta(dist_meta)
        if split_shapes is not None:
            state["gdn_split_shapes"] = tuple(int(dim) for dim in split_shapes)
            state["gdn_split_axis"] = get_gdn_split_axis_from_dist_meta(dist_meta)
        linear_rows = get_linear_split_rows_from_dist_meta(dist_meta)
        if linear_rows is not None:
            state["linear_split_rows"] = tuple(int(dim) for dim in linear_rows)
            state["linear_child_kinds"] = get_linear_child_kinds_from_dist_meta(dist_meta)
            state["linear_split_axis"] = get_linear_split_axis_from_dist_meta(dist_meta)
        per_expert_shape = getattr(dist_meta, "per_expert_global_shape", None)
        if per_expert_shape is not None:
            state["per_expert_global_shape"] = tuple(int(dim) for dim in per_expert_shape)

    if not init_rotation:
        state.pop("rotation", None)
        return
    rotation_dtype = str_to_dtype(mixed_precision_config.rotation_dtype)
    if rotation_dtype is None:
        rotation_dtype = param.dtype
    rot_shape = rotation_shape(
        local_shape=local_shape,
        global_shape=global_shape,
        orientation=orientation,
        fs_shard_dim=getattr(dist_meta, "fs_shard_dim", -1) if dist_meta is not None else -1,
        tp_shard_dim=getattr(dist_meta, "tp_shard_dim", -1) if dist_meta is not None else -1,
    )
    rotation = state.get("rotation")
    if rotation is None or tuple(rotation.shape) != tuple(rot_shape):
        state["rotation"] = _identity_rows(
            shape=rot_shape,
            device=param.device,
            dtype=rotation_dtype,
            row_start=_rotation_row_start(dist_meta, orientation),
        )


def build_param_config(
    *,
    param_ndim: int,
    local_shape: Optional[Tuple[int, int]],
    dist_meta: Optional[AroDistMeta],
    momentum: float = 0.95,
    base_optimizer: str = "sinkhorn",
    sinkhorn_iters: int = 5,
    qr_backend: str = "scqr",
    scqr_eps: float = 1e-6,
    update_rms_scale: float = 0.2,
    scalar_optimizer: str = "adam",
    scalar_lr_scale: float = 1.0,
    beta1: float = 0.9,
    beta2: float = 0.95,
    scalar_eps: float = 1e-8,
    split_parameters: bool = False,
) -> AroParamConfig:
    """Build one ARO parameter config from explicit metadata and defaults."""
    if dist_meta is not None and getattr(dist_meta, "global_shape", None) is not None:
        global_shape = tuple(int(dim) for dim in dist_meta.global_shape)
    elif local_shape is not None:
        global_shape = tuple(int(dim) for dim in local_shape)
    else:
        global_shape = ()
    orientation = (
        getattr(dist_meta, "orientation", None)
        if dist_meta is not None and getattr(dist_meta, "orientation", None) is not None
        else (choose_orientation(global_shape) if len(global_shape) == 2 else "normal")
    )
    return AroParamConfig(
        momentum=float(momentum),
        base_optimizer=base_optimizer,
        sinkhorn_iters=int(sinkhorn_iters),
        qr_backend=qr_backend,
        scqr_eps=float(scqr_eps),
        update_rms_scale=float(update_rms_scale),
        scalar_optimizer=scalar_optimizer,
        scalar_lr_scale=float(scalar_lr_scale),
        beta1=float(beta1),
        beta2=float(beta2),
        scalar_eps=float(scalar_eps),
        split_parameters=bool(split_parameters),
        orientation=str(orientation),
        fs_shard_dim=int(getattr(dist_meta, "fs_shard_dim", -1)) if dist_meta is not None else -1,
        tp_shard_dim=int(getattr(dist_meta, "tp_shard_dim", -1)) if dist_meta is not None else -1,
        has_fs_shard=bool(
            dist_meta is not None
            and int(getattr(dist_meta, "fs_world_size", 1)) > 1
            and int(getattr(dist_meta, "fs_shard_dim", -1)) in (0, 1)
        ),
        has_tp_shard=bool(
            dist_meta is not None
            and int(getattr(dist_meta, "tp_world_size", 1)) > 1
            and int(getattr(dist_meta, "tp_shard_dim", -1)) in (0, 1)
        ),
    )


def state_backend_keys() -> tuple[str, ...]:
    return (
        "momentum",
        "rotation",
        "orientation",
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


def _scalar_optimizer_for_group(opt, group, config=None) -> str:
    defaults = getattr(opt, "defaults", {}) or {}
    scalar_optimizer = group.get(
        "scalar_optimizer",
        group.get(
            "aro_scalar_optimizer",
            defaults.get(
                "scalar_optimizer",
                getattr(config, "aro_scalar_optimizer", "adam") if config is not None else "adam",
            ),
        ),
    )
    return str(scalar_optimizer).lower()


def init_aro_state(opt, config=None):
    """Initialize ARO state for checkpointing wrappers."""
    mixed_precision_config = getattr(opt, "_mixed_precision_config", AroMixedPrecisionConfig())
    for group in opt.param_groups:
        for param in group["params"]:
            state = opt.state[param]
            dist_meta = getattr(opt, "dist_metas", {}).get(param)
            if is_aro_matrix_param(param, dist_meta):
                init_matrix_state(
                    param,
                    state,
                    dist_meta=dist_meta,
                    mixed_precision_config=mixed_precision_config,
                )
                continue
            if "exp_avg" not in state:
                state["exp_avg"] = torch.zeros_like(param)
            scalar_optimizer = _scalar_optimizer_for_group(opt, group, config)
            if scalar_optimizer in ("adam", "adamw"):
                if "exp_avg_sq" not in state:
                    state["exp_avg_sq"] = torch.zeros_like(param)
            elif scalar_optimizer != "lion":
                raise RuntimeError(f"[ARO_INVALID_SCALAR_OPTIMIZER] {scalar_optimizer!r}")


__all__ = [
    "build_param_config",
    "choose_orientation",
    "get_global_shape",
    "init_aro_state",
    "init_matrix_state",
    "is_aro_matrix_param",
    "mark_aro_bucket_params",
    "prepare_aro_params",
    "require_2d_local_shape",
    "rotation_shape",
    "state_backend_keys",
    "str_to_dtype",
]
