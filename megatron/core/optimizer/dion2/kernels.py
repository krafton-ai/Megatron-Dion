"""Dion2 kernel helpers that do not own Megatron runtime state."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
from torch import Tensor

from ..muon import kernels as muon_kernels
from ..muon.kernels import orthogonalize_muon, orthogonalize_muon_2d
from .types import Dion2ParamConfig


DION2_POLAR_EXPRESS_COEFFICIENT_TYPE = "dion2_polar_express"
DION2_POLAR_EXPRESS_COEFFS = (
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
)


def ensure_dion2_coefficients_registered() -> None:
    """Register Dion2-local Polar Express coefficients with Muon NS dispatch."""
    existing = muon_kernels._COEFFICIENT_SETS.get(DION2_POLAR_EXPRESS_COEFFICIENT_TYPE)
    if existing is None:
        muon_kernels._COEFFICIENT_SETS[DION2_POLAR_EXPRESS_COEFFICIENT_TYPE] = (
            DION2_POLAR_EXPRESS_COEFFS
        )
        return
    if tuple(existing) != DION2_POLAR_EXPRESS_COEFFS:
        raise RuntimeError("[DION2_POLAR_EXPRESS_COEFF_MISMATCH]")


def _dist_world_size(group) -> int:
    if group is None or not dist.is_available() or not dist.is_initialized():
        return 1
    return int(dist.get_world_size(group))


def _dist_rank(group) -> int:
    if group is None or not dist.is_available() or not dist.is_initialized():
        return 0
    return int(dist.get_rank(group))


def _split_range(size: int, world_size: int, rank: int) -> tuple[int, int]:
    base = int(size) // int(world_size)
    rem = int(size) % int(world_size)
    start = int(rank) * base + min(int(rank), rem)
    return start, start + base + (1 if int(rank) < rem else 0)


def adjusted_lr_scale(
    *,
    base_lr: float,
    global_shape: tuple[int, int],
    adjust_lr: Optional[str],
) -> float:
    """Return adjusted_lr / base_lr for the Dion2 matrix update."""
    if float(base_lr) == 0.0 or adjust_lr is None:
        return 1.0
    fan_out, fan_in = int(global_shape[0]), int(global_shape[1])
    if fan_out <= 0 or fan_in <= 0:
        raise RuntimeError(f"[DION2_INVALID_GLOBAL_SHAPE] global_shape={global_shape}")
    if adjust_lr == "spectral_norm":
        return math.sqrt(float(fan_out) / float(fan_in))
    if adjust_lr == "rms_norm":
        return 0.2 * math.sqrt(float(max(fan_out, fan_in)))
    raise RuntimeError(f"[DION2_INVALID_ADJUST_LR] adjust_lr={adjust_lr!r}")


def resolve_select_dim(
    *,
    local_shape: tuple[int, int],
    global_shape: tuple[int, int],
    fs_shard_dim: int = -1,
    fs_world_size: int = 1,
    tp_shard_dim: int = -1,
    tp_world_size: int = 1,
    requested: str | int = "auto",
) -> int:
    """Resolve Dion2 selected axis as 0 for rows or 1 for columns."""
    del local_shape
    if requested in ("row", "rows", 0, -2):
        return 0
    if requested in ("col", "cols", "column", "columns", 1, -1):
        return 1
    if requested != "auto":
        raise RuntimeError(f"[DION2_INVALID_SELECT_DIM] select_dim={requested!r}")

    active_axes = []
    if int(fs_world_size) > 1 and int(fs_shard_dim) in (0, 1):
        active_axes.append(int(fs_shard_dim))
    if int(tp_world_size) > 1 and int(tp_shard_dim) in (0, 1):
        active_axes.append(int(tp_shard_dim))
    unique_axes = tuple(sorted(set(active_axes)))
    if len(unique_axes) == 1:
        return int(unique_axes[0])
    return 0 if int(global_shape[0]) <= int(global_shape[1]) else 1


def _axis_group(meta, axis: int):
    """Return the process group that shards one original matrix axis, if any."""
    if meta is None:
        return None, -1, 1, 0
    fs_dim = int(getattr(meta, "fs_shard_dim", -1))
    fs_world = int(getattr(meta, "fs_world_size", 1))
    if fs_dim == int(axis) and fs_world > 1:
        return (
            getattr(meta, "fs_group", None),
            fs_dim,
            fs_world,
            int(getattr(meta, "fs_rank", 0)),
        )
    tp_dim = int(getattr(meta, "tp_shard_dim", -1))
    tp_world = int(getattr(meta, "tp_world_size", 1))
    if tp_dim == int(axis) and tp_world > 1:
        return (
            getattr(meta, "tp_group", None),
            tp_dim,
            tp_world,
            int(getattr(meta, "tp_rank", 0)),
        )
    return None, -1, 1, 0


def _collect_axis_sizes(local_size: int, group) -> Optional[tuple[int, ...]]:
    if group is None or _dist_world_size(group) <= 1:
        return None
    device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
    size_t = torch.tensor([int(local_size)], dtype=torch.int64, device=device)
    gathered = [torch.empty_like(size_t) for _ in range(_dist_world_size(group))]
    dist.all_gather(gathered, size_t, group=group)
    return tuple(int(item.item()) for item in gathered)


def _select_slices(
    momentum: Tensor,
    *,
    select_dim: int,
    fraction: float,
    norm_group=None,
    ef_decay: float,
) -> tuple[Tensor, Tensor]:
    """Select Dion2 slices and decay selected momentum entries in-place."""
    local_select = int(momentum.size(select_dim))
    if local_select == 0:
        empty_shape = (
            (0, int(momentum.size(1))) if select_dim == 0 else (int(momentum.size(0)), 0)
        )
        return momentum.new_empty(empty_shape), torch.empty(0, dtype=torch.long, device=momentum.device)

    norm_dim = 1 if int(select_dim) == 0 else 0
    scores = momentum.abs().sum(dim=norm_dim)
    if norm_group is not None and _dist_world_size(norm_group) > 1:
        dist.all_reduce(scores, op=dist.ReduceOp.SUM, group=norm_group)
    k = max(1, int(math.ceil(float(fraction) * float(local_select))))
    k = min(k, local_select)
    _, indices = torch.topk(scores, k, dim=0, sorted=False)
    indices = torch.sort(indices).values

    if int(select_dim) == 0:
        expanded = indices[:, None].expand(k, int(momentum.size(1)))
        selected = torch.gather(momentum, dim=0, index=expanded).contiguous()
        momentum.scatter_(dim=0, index=expanded, src=selected * float(ef_decay))
    else:
        expanded = indices[None, :].expand(int(momentum.size(0)), k)
        selected = torch.gather(momentum, dim=1, index=expanded).contiguous()
        momentum.scatter_(dim=1, index=expanded, src=selected * float(ef_decay))
    return selected, indices


@dataclass(frozen=True)
class SelectedLayout:
    """Distributed layout of a canonical selected Dion2 submatrix."""

    select_dim: int
    canonical_transposed: bool
    logical_shape: tuple[int, int]
    row_group: object | None
    row_partition_dim: int
    row_mode: str
    row_partition_sizes: Optional[tuple[int, ...]]
    col_group: object | None
    col_partition_dim: int
    col_mode: str
    col_partition_sizes: Optional[tuple[int, ...]]


def build_selected_layout(
    *,
    selected: Tensor,
    indices: Tensor,
    select_dim: int,
    global_shape: tuple[int, int],
    dist_meta,
    config: Dion2ParamConfig,
) -> SelectedLayout:
    """Build the canonical selected-submatrix layout consumed by Muon NS kernels."""
    selected_axis = int(select_dim)
    opposite_axis = 1 - selected_axis
    selected_group, _selected_dim, selected_world, _selected_rank = _axis_group(
        dist_meta,
        selected_axis,
    )
    opposite_group, _opposite_dim, opposite_world, _opposite_rank = _axis_group(
        dist_meta,
        opposite_axis,
    )
    selected_global = int(indices.numel())
    selected_partition_sizes = None
    if selected_group is not None and int(selected_world) > 1:
        selected_partition_sizes = _collect_axis_sizes(int(indices.numel()), selected_group)
        selected_global = int(sum(selected_partition_sizes))
    opposite_global = int(global_shape[opposite_axis])
    opposite_local = int(selected.size(1)) if selected_axis == 0 else int(selected.size(0))
    opposite_partition_sizes = None
    if opposite_group is not None and int(opposite_world) > 1:
        opposite_partition_sizes = _collect_axis_sizes(opposite_local, opposite_group)

    if selected_axis == 0:
        canonical_shape = (selected_global, opposite_global)
        canonical_transposed = False
        row_group = selected_group
        row_partition_dim = 0 if selected_group is not None else -1
        row_mode = config.fs_mode if _selected_dim == getattr(dist_meta, "fs_shard_dim", -2) else config.tp_mode
        row_partition_sizes = selected_partition_sizes
        col_group = opposite_group
        col_partition_dim = 1 if opposite_group is not None else -1
        col_mode = config.fs_mode if _opposite_dim == getattr(dist_meta, "fs_shard_dim", -2) else config.tp_mode
    else:
        canonical_shape = (selected_global, opposite_global)
        canonical_transposed = True
        row_group = selected_group
        row_partition_dim = 0 if selected_group is not None else -1
        row_mode = config.fs_mode if _selected_dim == getattr(dist_meta, "fs_shard_dim", -2) else config.tp_mode
        row_partition_sizes = selected_partition_sizes
        col_group = opposite_group
        col_partition_dim = 1 if opposite_group is not None else -1
        col_mode = config.fs_mode if _opposite_dim == getattr(dist_meta, "fs_shard_dim", -2) else config.tp_mode

    return SelectedLayout(
        select_dim=selected_axis,
        canonical_transposed=canonical_transposed,
        logical_shape=canonical_shape,
        row_group=row_group,
        row_partition_dim=row_partition_dim,
        row_mode=row_mode,
        row_partition_sizes=row_partition_sizes,
        col_group=col_group,
        col_partition_dim=col_partition_dim,
        col_mode=col_mode,
        col_partition_sizes=opposite_partition_sizes,
    )


def orthogonalize_selected(
    canonical: Tensor,
    *,
    layout: SelectedLayout,
    config: Dion2ParamConfig,
) -> Tensor:
    """Orthogonalize a canonical selected Dion2 matrix without Muon scaling."""
    ensure_dion2_coefficients_registered()
    row_world = _dist_world_size(layout.row_group)
    col_world = _dist_world_size(layout.col_group)
    row_active = row_world > 1 and layout.row_partition_dim in (0, 1)
    col_active = col_world > 1 and layout.col_partition_dim in (0, 1)

    if row_active and col_active and layout.row_mode == "distributed" and layout.col_mode == "distributed":
        logical_transposed = int(layout.logical_shape[0]) > int(layout.logical_shape[1])
        oriented_row_partition_sizes = (
            layout.col_partition_sizes if logical_transposed else layout.row_partition_sizes
        )
        return orthogonalize_muon_2d(
            canonical,
            ns_backend=config.ns_backend,
            steps=config.num_ns_steps,
            coefficient_type=config.coefficient_type,
            fs_group=layout.row_group,
            fs_partition_dim=0,
            tp_group=layout.col_group,
            tp_partition_dim=1,
            logical_shape=layout.logical_shape,
            row_partition_sizes=oriented_row_partition_sizes,
            gram_restart_steps=config.gram_restart_iterations,
            gram_dtype=config.gram_dtype,
            gram_kernel_policy=config.gram_kernel_policy,
            eps=config.ns_epsilon,
            fp32_matmul_prec=config.fp32_matmul_prec,
        )

    if row_active:
        return orthogonalize_muon(
            canonical,
            ns_backend=config.ns_backend,
            steps=config.num_ns_steps,
            coefficient_type=config.coefficient_type,
            tp_group=layout.row_group,
            partition_dim=0,
            tp_mode=layout.row_mode,
            logical_shape=layout.logical_shape,
            gram_restart_steps=config.gram_restart_iterations,
            gram_dtype=config.gram_dtype,
            gram_kernel_policy=config.gram_kernel_policy,
            eps=config.ns_epsilon,
            fp32_matmul_prec=config.fp32_matmul_prec,
        )

    if col_active:
        return orthogonalize_muon(
            canonical,
            ns_backend=config.ns_backend,
            steps=config.num_ns_steps,
            coefficient_type=config.coefficient_type,
            tp_group=layout.col_group,
            partition_dim=1,
            tp_mode=layout.col_mode,
            logical_shape=layout.logical_shape,
            gram_restart_steps=config.gram_restart_iterations,
            gram_dtype=config.gram_dtype,
            gram_kernel_policy=config.gram_kernel_policy,
            eps=config.ns_epsilon,
            fp32_matmul_prec=config.fp32_matmul_prec,
        )

    return orthogonalize_muon(
        canonical,
        ns_backend=config.ns_backend,
        steps=config.num_ns_steps,
        coefficient_type=config.coefficient_type,
        logical_shape=layout.logical_shape,
        gram_restart_steps=config.gram_restart_iterations,
        gram_dtype=config.gram_dtype,
        gram_kernel_policy=config.gram_kernel_policy,
        eps=config.ns_epsilon,
        fp32_matmul_prec=config.fp32_matmul_prec,
    )


def dion2_update_tensor(
    *,
    param: Tensor,
    grad: Tensor,
    momentum: Tensor,
    config: Dion2ParamConfig,
    global_shape: tuple[int, int],
    base_lr: float,
    weight_decay: float,
    dist_meta=None,
) -> None:
    """Apply one Dion2 matrix update in-place."""
    update = dion2_compute_update(
        grad=grad,
        momentum=momentum,
        config=config,
        global_shape=global_shape,
        dist_meta=dist_meta,
    )
    if weight_decay != 0.0:
        param.mul_(1.0 - float(base_lr) * float(weight_decay))
    param.add_(update.to(dtype=param.dtype), alpha=-float(base_lr))


def dion2_prepare_selected(
    *,
    grad: Tensor,
    momentum: Tensor,
    config: Dion2ParamConfig,
    global_shape: tuple[int, int],
    dist_meta=None,
) -> tuple[Tensor, Tensor, int, SelectedLayout, float]:
    """Update momentum, select a Dion2 submatrix, and return canonical NS input."""
    if momentum.shape != grad.shape:
        raise RuntimeError(
            "[DION2_MOMENTUM_SHAPE_MISMATCH] "
            f"momentum_shape={tuple(momentum.shape)} grad_shape={tuple(grad.shape)}"
        )

    momentum.add_(grad.to(dtype=momentum.dtype))
    select_dim = resolve_select_dim(
        local_shape=(int(grad.size(0)), int(grad.size(1))),
        global_shape=global_shape,
        fs_shard_dim=int(getattr(dist_meta, "fs_shard_dim", -1)),
        fs_world_size=int(getattr(dist_meta, "fs_world_size", 1)),
        tp_shard_dim=int(getattr(dist_meta, "tp_shard_dim", -1)),
        tp_world_size=int(getattr(dist_meta, "tp_world_size", 1)),
        requested=config.select_dim,
    )
    opposite_group, _, opposite_world, _ = _axis_group(dist_meta, 1 - select_dim)
    norm_group = opposite_group if int(opposite_world) > 1 else None
    selected, indices = _select_slices(
        momentum,
        select_dim=select_dim,
        fraction=config.fraction,
        norm_group=norm_group,
        ef_decay=config.ef_decay,
    )
    canonical = selected if select_dim == 0 else selected.mT.contiguous()
    layout = build_selected_layout(
        selected=selected,
        indices=indices,
        select_dim=select_dim,
        global_shape=global_shape,
        dist_meta=dist_meta,
        config=config,
    )
    lr_scale = adjusted_lr_scale(
        base_lr=1.0,
        global_shape=global_shape,
        adjust_lr=config.adjust_lr,
    )
    return canonical, indices, select_dim, layout, lr_scale


def dion2_scatter_update(
    *,
    target_shape: tuple[int, int],
    orthogonalized_canonical: Tensor,
    indices: Tensor,
    select_dim: int,
    lr_scale: float,
    dtype: torch.dtype,
) -> Tensor:
    """Scatter a canonical orthogonalized Dion2 update back to full local shape."""
    update_full = torch.zeros(
        target_shape,
        dtype=orthogonalized_canonical.dtype,
        device=orthogonalized_canonical.device,
    )
    if int(indices.numel()) == 0:
        return update_full.to(dtype=dtype)
    orth = orthogonalized_canonical if int(select_dim) == 0 else orthogonalized_canonical.mT.contiguous()
    update = orth * float(lr_scale)
    if int(select_dim) == 0:
        index = indices[:, None].expand_as(update)
        update_full.scatter_add_(dim=0, index=index, src=update)
    else:
        index = indices[None, :].expand_as(update)
        update_full.scatter_add_(dim=1, index=index, src=update)
    return update_full.to(dtype=dtype)


def dion2_compute_update(
    *,
    grad: Tensor,
    momentum: Tensor,
    config: Dion2ParamConfig,
    global_shape: tuple[int, int],
    dist_meta=None,
) -> Tensor:
    """Return a full-shape Dion2 update tensor and mutate momentum in-place."""
    canonical, indices, select_dim, layout, lr_scale = dion2_prepare_selected(
        grad=grad,
        momentum=momentum,
        config=config,
        global_shape=global_shape,
        dist_meta=dist_meta,
    )
    if int(indices.numel()) == 0:
        return torch.zeros_like(grad, dtype=momentum.dtype)
    orth = orthogonalize_selected(canonical, layout=layout, config=config)
    return dion2_scatter_update(
        target_shape=tuple(int(dim) for dim in grad.shape),
        orthogonalized_canonical=orth,
        indices=indices,
        select_dim=select_dim,
        lr_scale=lr_scale,
        dtype=momentum.dtype,
    )


__all__ = [
    "SelectedLayout",
    "adjusted_lr_scale",
    "build_selected_layout",
    "dion2_compute_update",
    "dion2_prepare_selected",
    "dion2_scatter_update",
    "dion2_update_tensor",
    "orthogonalize_selected",
    "resolve_select_dim",
]
