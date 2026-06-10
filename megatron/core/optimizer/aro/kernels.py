"""ARO kernel helpers independent of Megatron runtime ownership."""

from __future__ import annotations

import os
from typing import Iterable, Optional, Sequence

import torch
import torch.distributed as dist
from torch import Tensor

_GROUP_KEY_CACHE: dict[int, tuple[object, tuple[int, ...] | None]] = {}
_UNIQUE_GROUPS_CACHE: dict[tuple[int, ...], tuple] = {}
_VALIDATE_SCQR_OUTPUTS = os.getenv("MEGATRON_ARO_VALIDATE_SCQR", "0") == "1"


def _validate_scqr_outputs() -> bool:
    return _VALIDATE_SCQR_OUTPUTS


def choose_orientation(global_shape: Sequence[int]) -> str:
    """Choose the safe one-sided orientation with the smaller rotation dimension."""
    rows, cols = int(global_shape[0]), int(global_shape[1])
    return "normal" if rows <= cols else "transpose"


def oriented_shape(shape: Sequence[int], orientation: str) -> tuple[int, int]:
    rows, cols = int(shape[0]), int(shape[1])
    if orientation == "normal":
        return rows, cols
    if orientation == "transpose":
        return cols, rows
    raise RuntimeError(f"[ARO_INVALID_ORIENTATION] orientation={orientation!r}")


def orient_tensor(tensor: Tensor, orientation: str) -> Tensor:
    if orientation == "normal":
        return tensor
    if orientation == "transpose":
        return tensor.mT
    raise RuntimeError(f"[ARO_INVALID_ORIENTATION] orientation={orientation!r}")


def unorient_tensor(tensor: Tensor, orientation: str) -> Tensor:
    return orient_tensor(tensor, orientation)


def _group_size(group) -> int:
    if group is None or not dist.is_available() or not dist.is_initialized():
        return 1
    return int(dist.get_world_size(group))


def _group_key(group):
    if group is None:
        return None
    cache_key = id(group)
    cached = _GROUP_KEY_CACHE.get(cache_key)
    if cached is not None and cached[0] is group:
        return cached[1]
    if dist.is_available() and dist.is_initialized():
        ranks = tuple(int(rank) for rank in dist.get_process_group_ranks(group))
    else:
        ranks = None
    _GROUP_KEY_CACHE[cache_key] = (group, ranks)
    return ranks


def _unique_groups(groups: Iterable) -> tuple:
    if isinstance(groups, tuple):
        cache_key = tuple(id(group) for group in groups)
        cached = _UNIQUE_GROUPS_CACHE.get(cache_key)
        if cached is not None:
            return cached
    else:
        cache_key = None
    result = []
    seen = set()
    for group in groups:
        if group is None or _group_size(group) <= 1:
            continue
        key = _group_key(group)
        if key in seen:
            continue
        seen.add(key)
        result.append(group)
    unique = tuple(result)
    if cache_key is not None:
        _UNIQUE_GROUPS_CACHE[cache_key] = unique
    return unique


def _all_reduce_sum_unique_(tensor: Tensor, groups: Iterable) -> Tensor:
    for group in groups:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    return tensor


def _all_reduce_sum_(tensor: Tensor, groups: Iterable) -> Tensor:
    return _all_reduce_sum_unique_(tensor, _unique_groups(groups))


def _target_numel_tensor(
    target_numel: float | Tensor | None,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> Optional[Tensor]:
    if target_numel is None:
        return None
    if isinstance(target_numel, Tensor):
        return target_numel.to(dtype=dtype, device=device)
    return torch.tensor(float(target_numel), dtype=dtype, device=device)


def _target_numel(
    *,
    rows: int,
    local_cols: int,
    device: torch.device,
    column_groups: Iterable,
) -> Tensor:
    numel = torch.tensor(float(int(rows) * int(local_cols)), dtype=torch.float32, device=device)
    _all_reduce_sum_unique_(numel, column_groups)
    return numel


def _target_numel_batched(
    *,
    rows: int,
    column_counts: Sequence[int],
    device: torch.device,
    column_groups: Iterable,
) -> Tensor:
    numel = torch.tensor(
        [float(int(rows) * int(cols)) for cols in column_counts],
        dtype=torch.float32,
        device=device,
    )
    _all_reduce_sum_unique_(numel, column_groups)
    return numel


def _gather_rows(tensor: Tensor, group) -> tuple[Tensor, tuple[int, ...]]:
    world_size = _group_size(group)
    if group is None or world_size <= 1:
        return tensor, (int(tensor.size(0)),)
    local_rows = torch.tensor([int(tensor.size(0))], dtype=torch.int64, device=tensor.device)
    row_sizes_t = [torch.empty_like(local_rows) for _ in range(world_size)]
    dist.all_gather(row_sizes_t, local_rows, group=group)
    row_sizes = tuple(int(item.item()) for item in row_sizes_t)
    max_rows = max(row_sizes)
    if int(local_rows.item()) == max_rows:
        send = tensor.contiguous()
    else:
        send = tensor.new_zeros((max_rows, int(tensor.size(1))))
        if int(local_rows.item()) > 0:
            send[: int(local_rows.item()), :].copy_(tensor.contiguous())
    gathered = [torch.empty_like(send) for _ in range(world_size)]
    dist.all_gather(gathered, send, group=group)
    shards = [shard[:rows, :] for shard, rows in zip(gathered, row_sizes)]
    return torch.cat(shards, dim=0), row_sizes


def _slice_gathered_rows(tensor: Tensor, row_sizes: Sequence[int], rank: int) -> Tensor:
    start = sum(int(size) for size in row_sizes[: int(rank)])
    end = start + int(row_sizes[int(rank)])
    return tensor[start:end, :].contiguous()


def _gather_rows_across_groups(tensor: Tensor, groups: Iterable) -> tuple[Tensor, tuple]:
    contexts = []
    gathered = tensor
    for group in _unique_groups(groups):
        rank = dist.get_rank(group)
        gathered, row_sizes = _gather_rows(gathered, group)
        contexts.append((row_sizes, rank))
    return gathered, tuple(contexts)


def _slice_rows_across_groups(tensor: Tensor, contexts: Sequence) -> Tensor:
    local = tensor
    for row_sizes, rank in reversed(tuple(contexts)):
        local = _slice_gathered_rows(local, row_sizes, rank)
    return local.contiguous()


def sinkhorn_project(
    x: Tensor,
    *,
    iters: int = 5,
    eps: float = 1e-8,
    column_groups: Iterable = (),
    target_numel: Optional[Tensor] = None,
) -> Tensor:
    """Apply simultaneous row/column L2 Sinkhorn normalization.

    The input is expected to contain the full oriented row dimension and a
    possibly column-sharded local slice. Row norms are therefore reduced over
    column-shard groups; column norms are local. The proportionality constant
    in the ARO paper's Sinkhorn step is chosen so that the global output RMS is
    one after each iteration.
    """
    y = x
    col_groups = _unique_groups(column_groups)
    if target_numel is None:
        target_numel = _target_numel(
            rows=int(x.size(0)),
            local_cols=int(x.size(1)),
            device=x.device,
            column_groups=col_groups,
        )
    num_iters = int(iters)
    if num_iters <= 0:
        return y
    for _ in range(num_iters):
        y32 = y.float()
        sq = y32.square()
        row_sq = sq.sum(dim=1, keepdim=True)
        _all_reduce_sum_unique_(row_sq, col_groups)
        col_sq = sq.sum(dim=0, keepdim=True)
        y = y * row_sq.clamp_min(float(eps)).rsqrt().to(dtype=y.dtype)
        y = y * col_sq.clamp_min(float(eps)).rsqrt().to(dtype=y.dtype)
    norm_sq = y.float().square().sum()
    _all_reduce_sum_unique_(norm_sq, col_groups)
    factor = target_numel.sqrt() * norm_sq.clamp_min(float(eps) * float(eps)).rsqrt()
    y = y * factor.to(dtype=y.dtype)
    return y


def sinkhorn_project_batched(
    x: Tensor,
    *,
    iters: int = 5,
    eps: float = 1e-8,
    column_groups: Iterable = (),
    column_counts: Optional[Sequence[int]] = None,
    target_numel: Optional[Tensor] = None,
) -> Tensor:
    """Batched row/column L2 Sinkhorn normalization.

    ``x`` has shape ``[batch, rows, local_cols]``. Row norms are reduced over
    column-shard groups; column norms stay local, matching ``sinkhorn_project``.
    """
    if x.ndim != 3:
        raise RuntimeError(f"[ARO_BATCHED_SINKHORN_REQUIRES_3D] shape={tuple(x.shape)}")
    y = x
    col_groups = _unique_groups(column_groups)
    if column_counts is None:
        column_counts = tuple(int(x.size(2)) for _ in range(int(x.size(0))))
    else:
        column_counts = tuple(int(count) for count in column_counts)
    if len(column_counts) != int(x.size(0)):
        raise RuntimeError(
            "[ARO_BATCHED_SINKHORN_COLUMN_COUNT_MISMATCH] "
            f"batch={int(x.size(0))} column_counts={column_counts}"
        )
    if target_numel is None:
        target_numel = _target_numel_batched(
            rows=int(x.size(1)),
            column_counts=column_counts,
            device=x.device,
            column_groups=col_groups,
        )
    num_iters = int(iters)
    if num_iters <= 0:
        return y
    for _ in range(num_iters):
        y32 = y.float()
        sq = y32.square()
        row_sq = sq.sum(dim=2, keepdim=True)
        _all_reduce_sum_unique_(row_sq, col_groups)
        col_sq = sq.sum(dim=1, keepdim=True)
        y = y * row_sq.clamp_min(float(eps)).rsqrt().to(dtype=y.dtype)
        y = y * col_sq.clamp_min(float(eps)).rsqrt().to(dtype=y.dtype)
    norm_sq = y.float().square().sum(dim=(1, 2))
    _all_reduce_sum_unique_(norm_sq, col_groups)
    factors = target_numel.sqrt() * norm_sq.clamp_min(float(eps) * float(eps)).rsqrt()
    y = y * factors.to(dtype=y.dtype).view(-1, 1, 1)
    return y


def _deterministic_qr(a: Tensor) -> Tensor:
    q, r = torch.linalg.qr(a.float(), mode="reduced")
    diag = torch.diagonal(r, 0, dim1=-2, dim2=-1)
    signs = torch.where(diag < 0, -torch.ones_like(diag), torch.ones_like(diag))
    return (q * signs.unsqueeze(-2)).to(dtype=a.dtype)


def shifted_cholesky_qr(
    a: Tensor,
    *,
    eps: float,
    row_groups: Iterable = (),
    fallback_groups: Iterable = (),
    qr_backend: str = "scqr",
) -> Tensor:
    """Return the local rows of the SCQR/QR left factor for a row-sharded square matrix."""
    if a.ndim != 2:
        raise RuntimeError(f"[ARO_SCQR_REQUIRES_2D] shape={tuple(a.shape)}")
    row_groups = _unique_groups(row_groups)
    fallback_groups = _unique_groups(fallback_groups)
    if qr_backend not in ("scqr", "qr"):
        raise RuntimeError(f"[ARO_INVALID_QR_BACKEND] qr_backend={qr_backend!r}")

    force_qr = qr_backend == "qr"
    q_local = None
    failed = force_qr
    if not force_qr:
        try:
            a32 = a.float()
            gram = a32.mT @ a32
            _all_reduce_sum_unique_(gram, row_groups)
            gram.diagonal(dim1=-2, dim2=-1).add_(float(eps))
            chol, info = torch.linalg.cholesky_ex(gram, check_errors=False)
            if _validate_scqr_outputs() and bool((info != 0).any().item()):
                failed = True
                q_local = None
            else:
                q_local = torch.linalg.solve_triangular(
                    chol.mT,
                    a32,
                    upper=True,
                    left=False,
                ).to(dtype=a.dtype)
                if _validate_scqr_outputs():
                    failed = not bool(torch.isfinite(q_local).all().item())
        except RuntimeError:
            failed = True

    if not failed and q_local is not None:
        return q_local

    if row_groups:
        full, row_contexts = _gather_rows_across_groups(a, row_groups)
        q_full = _deterministic_qr(full)
        return _slice_rows_across_groups(q_full, row_contexts).to(dtype=a.dtype)
    return _deterministic_qr(a).to(dtype=a.dtype)


def _rotation_times_x(rotation_rows: Tensor, x_rows: Tensor, row_groups: Iterable) -> Tensor:
    rotated = rotation_rows.float().mT @ x_rows.float()
    _all_reduce_sum_unique_(rotated, row_groups)
    return rotated.to(dtype=x_rows.dtype)


def _rotation_times_x_batched(
    rotation_rows: Tensor,
    x_rows: Tensor,
    row_groups: Iterable,
) -> Tensor:
    rotated = rotation_rows.float().transpose(-2, -1) @ x_rows.float()
    _all_reduce_sum_unique_(rotated, row_groups)
    return rotated.to(dtype=x_rows.dtype)


def _make_rotation_update(
    *,
    x_rows: Tensor,
    rotation_rows: Tensor,
    config,
    row_groups: Iterable,
    column_groups: Iterable,
    target_numel: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor]:
    """Compute local oriented ARO update and new local rotation rows."""
    col_groups = _unique_groups(column_groups)
    if target_numel is None:
        target_numel = _target_numel(
            rows=int(rotation_rows.size(1)),
            local_cols=int(x_rows.size(1)),
            device=x_rows.device,
            column_groups=col_groups,
        )
    z_old = _rotation_times_x(rotation_rows, x_rows, row_groups)
    y_old = sinkhorn_project(
        z_old,
        iters=int(config.sinkhorn_iters),
        eps=float(config.scqr_eps),
        column_groups=col_groups,
        target_numel=target_numel,
    )
    align = x_rows.float() @ y_old.float().mT
    _all_reduce_sum_unique_(align, col_groups)
    new_rotation = shifted_cholesky_qr(
        align,
        eps=float(config.scqr_eps),
        row_groups=row_groups,
        fallback_groups=col_groups,
        qr_backend=str(config.qr_backend),
    )
    z_new = _rotation_times_x(new_rotation, x_rows, row_groups)
    y_new = sinkhorn_project(
        z_new,
        iters=int(config.sinkhorn_iters),
        eps=float(config.scqr_eps),
        column_groups=col_groups,
        target_numel=target_numel,
    )
    update = new_rotation.to(dtype=y_new.dtype) @ y_new
    return update.to(dtype=x_rows.dtype), new_rotation


def _shifted_cholesky_qr_batched(
    a: Tensor,
    *,
    eps: float,
    row_groups: Iterable = (),
    fallback_groups: Iterable = (),
    qr_backend: str = "scqr",
) -> Optional[Tensor]:
    """Return batched SCQR rows, or ``None`` when the caller must use fallback QR."""
    if a.ndim != 3:
        raise RuntimeError(f"[ARO_BATCHED_SCQR_REQUIRES_3D] shape={tuple(a.shape)}")
    row_groups = _unique_groups(row_groups)
    fallback_groups = _unique_groups(fallback_groups)
    if qr_backend != "scqr":
        return None

    a32 = a.float()
    gram = a32.transpose(-2, -1) @ a32
    _all_reduce_sum_unique_(gram, row_groups)
    gram.diagonal(dim1=-2, dim2=-1).add_(float(eps))
    chol, info = torch.linalg.cholesky_ex(gram, check_errors=False)
    if _validate_scqr_outputs() and bool((info != 0).any().item()):
        return None
    q_local = torch.linalg.solve_triangular(
        chol.transpose(-2, -1),
        a32,
        upper=True,
        left=False,
    ).to(dtype=a.dtype)
    if _validate_scqr_outputs() and not bool(torch.isfinite(q_local).all().item()):
        return None
    return q_local


def _make_rotation_update_batched(
    *,
    x_rows: Tensor,
    rotation_rows: Tensor,
    config,
    row_groups: Iterable,
    column_groups: Iterable,
    column_counts: Optional[Sequence[int]] = None,
    target_numel: Optional[Tensor] = None,
) -> Optional[tuple[Tensor, Tensor]]:
    col_groups = _unique_groups(column_groups)
    if column_counts is None:
        resolved_column_counts = tuple(int(x_rows.size(2)) for _ in range(int(x_rows.size(0))))
    else:
        resolved_column_counts = tuple(int(count) for count in column_counts)
    if target_numel is None:
        target_numel = _target_numel_batched(
            rows=int(rotation_rows.size(2)),
            column_counts=resolved_column_counts,
            device=x_rows.device,
            column_groups=col_groups,
        )
    z_old = _rotation_times_x_batched(rotation_rows, x_rows, row_groups)
    y_old = sinkhorn_project_batched(
        z_old,
        iters=int(config.sinkhorn_iters),
        eps=float(config.scqr_eps),
        column_groups=col_groups,
        column_counts=resolved_column_counts,
        target_numel=target_numel,
    )
    align = x_rows.float() @ y_old.float().transpose(-2, -1)
    _all_reduce_sum_unique_(align, col_groups)
    new_rotation = _shifted_cholesky_qr_batched(
        align,
        eps=float(config.scqr_eps),
        row_groups=row_groups,
        fallback_groups=col_groups,
        qr_backend=str(config.qr_backend),
    )
    if new_rotation is None:
        return None
    z_new = _rotation_times_x_batched(new_rotation, x_rows, row_groups)
    y_new = sinkhorn_project_batched(
        z_new,
        iters=int(config.sinkhorn_iters),
        eps=float(config.scqr_eps),
        column_groups=col_groups,
        column_counts=resolved_column_counts,
        target_numel=target_numel,
    )
    update = new_rotation.to(dtype=y_new.dtype) @ y_new
    return update.to(dtype=x_rows.dtype), new_rotation


def _stack_padded(
    tensors: Sequence[Tensor],
    *,
    rows: int,
    cols: int,
    cache: Optional[dict] = None,
    name: str = "",
) -> Tensor:
    if not tensors:
        raise RuntimeError("[ARO_STACK_PADDED_EMPTY_BATCH]")
    rows = int(rows)
    cols = int(cols)
    shape = (len(tensors), rows, cols)
    first = tensors[0]
    result = None
    if cache is not None:
        key = (str(name), shape, first.dtype, first.device)
        result = cache.get(key)
        if result is None:
            result = first.new_empty(shape)
            cache[key] = result
    if all(int(tensor.size(0)) == rows and int(tensor.size(1)) == cols for tensor in tensors):
        if result is None:
            return torch.stack([tensor.contiguous() for tensor in tensors], dim=0)
        return torch.stack([tensor.contiguous() for tensor in tensors], dim=0, out=result)

    if result is None:
        result = first.new_empty(shape)
    result.zero_()
    for index, tensor in enumerate(tensors):
        local_rows = int(tensor.size(0))
        local_cols = int(tensor.size(1))
        if local_rows > 0 and local_cols > 0:
            result[index, :local_rows, :local_cols].copy_(tensor.contiguous())
    return result


def compute_aro_update(
    *,
    momentum: Tensor,
    rotation: Tensor,
    config,
    orientation: str,
    row_groups: Iterable = (),
    column_groups: Iterable = (),
    target_numel: float | Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Compute one ARO update for the local parameter shard."""
    row_groups = _unique_groups(row_groups)
    column_groups = _unique_groups(column_groups)
    x_rows = orient_tensor(momentum, orientation).contiguous()
    target_numel_t = _target_numel_tensor(
        target_numel,
        dtype=torch.float32,
        device=x_rows.device,
    )
    update_oriented, new_rotation = _make_rotation_update(
        x_rows=x_rows,
        rotation_rows=rotation,
        config=config,
        row_groups=row_groups,
        column_groups=column_groups,
        target_numel=target_numel_t,
    )
    update = unorient_tensor(update_oriented, orientation)
    scale = float(getattr(config, "update_rms_scale", 0.2))
    if scale > 0.0:
        norm_sq = update.float().square().sum()
        _all_reduce_sum_unique_(norm_sq, (*row_groups, *column_groups))
        if target_numel_t is None:
            target_numel_t = update.new_tensor(float(update.numel()), dtype=torch.float32)
            _all_reduce_sum_unique_(target_numel_t, (*row_groups, *column_groups))
        denom = norm_sq.sqrt().clamp_min(float(getattr(config, "scqr_eps", 1e-6)))
        update.mul_((scale * target_numel_t.sqrt() / denom).to(dtype=update.dtype))
    return update, new_rotation


def compute_aro_updates_batched(
    *,
    momentums: Sequence[Tensor],
    rotations: Sequence[Tensor],
    config,
    orientation: str,
    row_groups: Iterable = (),
    column_groups: Iterable = (),
    target_numels: Sequence[float] | Tensor | None = None,
    stack_cache: Optional[dict] = None,
) -> Optional[tuple[list[Tensor], list[Tensor]]]:
    """Compute same-invariant ARO updates with one batched Sinkhorn/SCQR path.

    The batch may include entries with different local row/column counts on the
    current rank; tensors are padded locally and sliced back before returning.
    The matrix/global invariant must still be identical across the batch.
    """
    if not momentums:
        return [], []
    if len(momentums) != len(rotations):
        raise RuntimeError(
            f"[ARO_BATCH_SIZE_MISMATCH] momentums={len(momentums)} rotations={len(rotations)}"
        )
    if str(getattr(config, "qr_backend", "scqr")) != "scqr":
        return None

    row_groups = _unique_groups(row_groups)
    column_groups = _unique_groups(column_groups)
    x_rows_list = [orient_tensor(momentum, orientation).contiguous() for momentum in momentums]
    x_shapes = [tuple(int(dim) for dim in x.shape) for x in x_rows_list]
    rotation_shapes = [tuple(int(dim) for dim in rotation.shape) for rotation in rotations]
    if any(len(shape) != 2 for shape in x_shapes) or any(len(shape) != 2 for shape in rotation_shapes):
        raise RuntimeError(
            f"[ARO_BATCH_REQUIRES_2D] x_shapes={x_shapes} rotation_shapes={rotation_shapes}"
        )
    global_rows = {shape[1] for shape in rotation_shapes}
    if len(global_rows) != 1:
        raise RuntimeError(f"[ARO_BATCH_ROTATION_WIDTH_MISMATCH] rotation_shapes={rotation_shapes}")
    for x_shape, rotation_shape in zip(x_shapes, rotation_shapes):
        if int(x_shape[0]) != int(rotation_shape[0]):
            raise RuntimeError(
                "[ARO_BATCH_LOCAL_ROW_MISMATCH] "
                f"x_shape={x_shape} rotation_shape={rotation_shape}"
            )

    max_rows = max(int(shape[0]) for shape in x_shapes)
    max_cols = max(int(shape[1]) for shape in x_shapes)
    rot_cols = next(iter(global_rows))
    x_batch = _stack_padded(x_rows_list, rows=max_rows, cols=max_cols, cache=stack_cache, name="x")
    rotation_batch = _stack_padded(
        rotations,
        rows=max_rows,
        cols=rot_cols,
        cache=stack_cache,
        name="rotation",
    )
    target_numel_t = None
    if target_numels is not None:
        target_numel_t = torch.as_tensor(
            target_numels,
            dtype=torch.float32,
            device=x_batch.device,
        )
    result = _make_rotation_update_batched(
        x_rows=x_batch,
        rotation_rows=rotation_batch,
        config=config,
        row_groups=row_groups,
        column_groups=column_groups,
        column_counts=[shape[1] for shape in x_shapes],
        target_numel=target_numel_t,
    )
    if result is None:
        return None
    update_batch, new_rotation_batch = result
    scale = float(getattr(config, "update_rms_scale", 0.2))
    if scale > 0.0:
        norm_sq = update_batch.float().square().sum(dim=(1, 2))
        _all_reduce_sum_unique_(norm_sq, (*row_groups, *column_groups))
        if target_numel_t is None:
            if len(set(x_shapes)) == 1:
                rows, cols = x_shapes[0]
                target_numel_t = torch.empty(
                    (len(x_shapes),),
                    dtype=torch.float32,
                    device=update_batch.device,
                )
                target_numel_t.fill_(float(int(rows) * int(cols)))
            else:
                target_numel_t = torch.tensor(
                    [float(int(rows) * int(cols)) for rows, cols in x_shapes],
                    dtype=torch.float32,
                    device=update_batch.device,
                )
            _all_reduce_sum_unique_(target_numel_t, (*row_groups, *column_groups))
        denom = norm_sq.sqrt().clamp_min(float(getattr(config, "scqr_eps", 1e-6)))
        factors = (scale * target_numel_t.sqrt() / denom).to(dtype=update_batch.dtype)
        update_batch.mul_(factors.view(-1, 1, 1))

    updates: list[Tensor] = []
    new_rotations: list[Tensor] = []
    for index, ((rows, cols), rotation_shape) in enumerate(zip(x_shapes, rotation_shapes)):
        update_oriented = update_batch[index, :rows, :cols]
        updates.append(unorient_tensor(update_oriented, orientation))
        new_rotations.append(
            new_rotation_batch[
                index,
                : int(rotation_shape[0]),
                : int(rotation_shape[1]),
            ]
        )
    return updates, new_rotations


__all__ = [
    "choose_orientation",
    "compute_aro_update",
    "compute_aro_updates_batched",
    "orient_tensor",
    "oriented_shape",
    "shifted_cholesky_qr",
    "sinkhorn_project_batched",
    "sinkhorn_project",
    "unorient_tensor",
]
