"""Orthogonalization and sketch helpers for Dion."""

import contextlib
import hashlib
import math
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch
import torch.distributed as dist
from torch import Tensor

from .utils import format_meta_id


@contextlib.contextmanager
def _dion_math_precision_context():
    """Disable TF32 for Dion linear algebra kernels.

    Reference Dion expects float32 linear algebra quality for the low-rank
    update path. Keep the model forward/backward settings intact and fence off
    only the optimizer-side Dion math.

    Duplex training enables TF32 globally for throughput, which is acceptable
    for the model forward/backward but too aggressive for Dion matmuls and
    RCQR/Cholesky.
    """
    prev_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    prev_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_matmul_tf32
        torch.backends.cudnn.allow_tf32 = prev_cudnn_tf32


def orthogonalize(
    P: Tensor,
    rcqr_oversample: float = 1.25,
    sketch_fn: Optional[Callable[[Tensor, float], Tensor]] = None,
) -> Tensor:
    """
    Local orthogonalization with Randomized Cholesky QR.

    Reference: dion.py lines 1265-1305

    Args:
        P: Matrix to orthogonalize, shape (..., m, r)
        rcqr_oversample: Oversampling factor for RCQR (default 1.25)
        sketch_fn: Function to generate random sketch matrix.
                   Signature: sketch_fn(P, oversample) -> S
                   If None, uses default random sketch.

    Returns:
        Orthogonalized matrix Q, same shape as P
    """
    with _dion_math_precision_context():
        assert P.ndim >= 2
        original_dtype = P.dtype
        m, r = P.shape[-2:]

        # Case 1: Square or wide matrix - use standard QR
        if m <= r:
            Q, _ = torch.linalg.qr(P.to(dtype=torch.float32))
            return Q.to(original_dtype).contiguous()

        # Case 2: Tall matrix - use Randomized Cholesky QR
        else:
            # Step 1: Generate random sketch matrix
            if sketch_fn is not None:
                S = sketch_fn(P, rcqr_oversample)
            else:
                S = _default_sketch_matrix(P, rcqr_oversample)

            # Step 2: Compute sketch
            SP = S @ P

            # Step 3: QR decomposition of sketch
            SP = SP.to(dtype=torch.float32)
            R = torch.linalg.qr(SP, mode='r')[1]

            # Match the reference RCQR contract: float32 triangular solve.
            R = R.to(dtype=torch.float32)
            P = torch.linalg.solve_triangular(
                R,
                P.to(dtype=torch.float32),
                upper=True,
                left=False,
            )

            # Match the reference RCQR contract: float32 Cholesky QR.
            PP = P.to(dtype=torch.float32).mT @ P.to(dtype=torch.float32)
            R = torch.linalg.cholesky_ex(PP, upper=True)[0]
            R = R.to(dtype=torch.float32)

            P = torch.linalg.solve_triangular(
                R,
                P.to(dtype=torch.float32),
                upper=True,
                left=False,
            )

            return P.to(original_dtype).contiguous()

def _logical_sketch_seed(seed_key: object) -> int:
    """Map one logical sketch key to a deterministic per-slot RNG seed."""
    return int.from_bytes(
        hashlib.blake2b(repr(seed_key).encode("utf-8"), digest_size=8).digest(),
        "little",
    ) & ((1 << 63) - 1)


def _seeded_normal_tensor(
    shape: tuple[int, ...],
    *,
    seed_key: object,
    device: torch.device,
    dtype: torch.dtype,
    std: float,
) -> Tensor:
    """Generate one topology-invariant logical sketch tensor from a logical seed."""
    gen = torch.Generator(device=str(device))
    gen.manual_seed(_logical_sketch_seed(seed_key))
    tensor = torch.empty(shape, device=device, dtype=dtype)
    tensor.normal_(mean=0.0, std=std, generator=gen)
    return tensor


def logical_sketch_keys(
    *,
    dist_metas: Optional[List],
    tag: str,
    step_count: int,
) -> Optional[List[object]]:
    """Return topology-invariant logical sketch ids for one Dion update batch."""
    if not dist_metas:
        return None

    logical_ids: List[object] = []
    for dist_meta in dist_metas:
        if dist_meta is None:
            logical_ids.append((tag, step_count, None))
            continue
        logical_ids.append(
            (
                tag,
                step_count,
                getattr(dist_meta, "param_uid", None),
                getattr(dist_meta, "param_name", ""),
            )
        )
    return logical_ids


def make_seeded_sketch(
    *,
    dist_metas: Optional[List],
    tag: str,
    step_count: int,
    format_meta_id: Callable[[object], Dict[str, Any]],
):
    """Build a topology-independent sketch generator for one logical Dion update batch."""
    if not dist_metas:
        return None

    logical_ids = logical_sketch_keys(
        dist_metas=dist_metas,
        tag=tag,
        step_count=step_count,
    )

    def _make_sketch(P: Tensor, oversample: float) -> Tensor:
        batch_shape = P.shape[:-2]
        if len(batch_shape) == 0:
            batch = 1
        elif len(batch_shape) == 1:
            batch = batch_shape[0]
        else:
            raise RuntimeError(
                "[DION_INVALID_SKETCH_BATCH] "
                f"tag={tag} expected batched 3D tensor, got shape={tuple(P.shape)}"
            )
        if batch != len(logical_ids):
            raise RuntimeError(
                "[DION_SKETCH_META_MISMATCH] "
                f"tag={tag} batch={batch} logical_ids={len(logical_ids)}"
            )
        m = P.size(-2)
        r = P.size(-1)
        k = math.ceil(oversample * r / 128.0) * 128
        if k <= 0:
            raise RuntimeError(
                f"[DION_INVALID_SKETCH_RANK] tag={tag} r={r} oversample={oversample} k={k}"
            )
        std = math.sqrt(1.0 / k)
        if batch == 1 and len(batch_shape) == 0:
            logical_id = logical_ids[0]
            sketch = _seeded_normal_tensor(
                (k, m),
                seed_key=logical_id,
                device=P.device,
                dtype=P.dtype,
                std=std,
            )
            return sketch
        sketch = torch.empty((batch, k, m), device=P.device, dtype=P.dtype)
        for idx, logical_id in enumerate(logical_ids):
            sketch[idx].copy_(
                _seeded_normal_tensor(
                    (k, m),
                    seed_key=logical_id,
                    device=P.device,
                    dtype=P.dtype,
                    std=std,
                )
            )
        return sketch

    setattr(_make_sketch, "_dion_sketch_tag", tag)
    setattr(_make_sketch, "_dion_logical_seed_keys", tuple(logical_ids))
    setattr(
        _make_sketch,
        "_dion_batch_meta_ids",
        tuple(format_meta_id(dist_meta) for dist_meta in dist_metas),
    )

    return _make_sketch


def make_local_sketch(
    *,
    dist_metas: Optional[List],
    step_count: int,
    format_meta_id: Callable[[object], Dict[str, Any]],
):
    """Return the topology-invariant local sketch contract."""
    return make_seeded_sketch(
        dist_metas=dist_metas,
        tag="logical_local",
        step_count=step_count,
        format_meta_id=format_meta_id,
    )


def make_seeded_sketch_for_update(
    optimizer,
    *,
    dist_metas: Optional[List],
    tag: str,
):
    """Bind logical sketch generation to one optimizer step."""
    return make_seeded_sketch(
        dist_metas=dist_metas,
        tag=tag,
        step_count=optimizer._step_count,
        format_meta_id=format_meta_id,
    )


def sketch_keys_for_update(
    optimizer,
    *,
    dist_metas: Optional[List],
    tag: str,
) -> Optional[List[object]]:
    """Bind logical sketch ids to one optimizer step."""
    return logical_sketch_keys(
        dist_metas=dist_metas,
        tag=tag,
        step_count=optimizer._step_count,
    )


def make_local_sketch_for_update(
    optimizer,
    *,
    dist_metas: Optional[List],
):
    """Bind the local sketch contract to one optimizer step."""
    return make_local_sketch(
        dist_metas=dist_metas,
        step_count=optimizer._step_count,
        format_meta_id=format_meta_id,
    )


def make_fs_only_sketch(
    *,
    fs_group: torch.distributed.ProcessGroup,
    fs_rank: int,
    broadcast_replicate_domain: Callable[[Tensor], None],
):
    """Reproduce the fs-only local sketch RNG contract."""
    fs_group_ranks = dist.get_process_group_ranks(fs_group)
    broadcast_src = fs_group_ranks[0]

    def _make_sketch(P: Tensor, oversample: float) -> Tensor:
        if P.ndim != 3:
            raise RuntimeError(
                "[DION_FSONLY_SKETCH_INVALID_RANK] "
                f"expected batched 3D tensor, got shape={tuple(P.shape)}"
            )
        if P.size(0) != 1:
            raise RuntimeError(
                "[DION_FSONLY_SKETCH_INVALID_BATCH] "
                f"expected local fs-only batch size 1, got shape={tuple(P.shape)}"
            )

        m = P.size(-2)
        r = P.size(-1)
        k = math.ceil(oversample * r / 128.0) * 128
        if k <= 0:
            raise RuntimeError(
                f"[DION_INVALID_SKETCH_RANK] r={r} oversample={oversample} k={k}"
            )

        std = math.sqrt(1.0 / k)
        seed_tensor = torch.zeros((), device=P.device, dtype=torch.int64)
        if dist.get_rank() == broadcast_src:
            seed_tensor.random_()
        dist.broadcast(seed_tensor, src=broadcast_src, group=fs_group)
        broadcast_replicate_domain(seed_tensor)

        gen = torch.Generator(device=P.device)
        gen.manual_seed(int(seed_tensor.item()))

        for _ in range(fs_rank):
            torch.empty((1, k, m), device=P.device, dtype=P.dtype).normal_(
                mean=0.0,
                std=std,
                generator=gen,
            )

        local_sketch = torch.empty((1, k, m), device=P.device, dtype=P.dtype)
        local_sketch.normal_(mean=0.0, std=std, generator=gen)
        return local_sketch

    setattr(_make_sketch, "_dion_sketch_tag", "fs_only")

    return _make_sketch


def _split_range(size: int, world_size: int, rank: int) -> tuple[int, int]:
    """Return the canonical contiguous shard range for one rank."""
    if world_size <= 0:
        raise RuntimeError(f"[DION_INVALID_WORLD_SIZE] world_size={world_size}")
    if rank < 0 or rank >= world_size:
        raise RuntimeError(
            f"[DION_INVALID_RANK] rank={rank} world_size={world_size}"
        )
    base = size // world_size
    remainder = size % world_size
    start = rank * base + min(rank, remainder)
    local = base + (1 if rank < remainder else 0)
    return start, start + local


def _canonical_shard_sizes(size: int, world_size: int) -> list[int]:
    """Return the canonical contiguous shard sizes for one logical axis."""
    return [
        _split_range(size, world_size, rank)[1] - _split_range(size, world_size, rank)[0]
        for rank in range(world_size)
    ]


def _resolve_row_sizes_from_dist_meta(
    *,
    dist_metas: Optional[List],
    batch_size: int,
    ortho_world_size: int,
    ortho_rank: int,
    local_rows: int,
) -> tuple[list[int], int]:
    """Reconstruct distributed P row sizes from logical metadata only."""
    if dist_metas is None:
        raise RuntimeError(
            "[DION_MISSING_ORTHO_DIST_META] distributed orthogonalize requires dist_metas"
        )

    active_metas = [dist_meta for dist_meta in dist_metas[:batch_size] if dist_meta is not None]
    if not active_metas:
        raise RuntimeError(
            "[DION_MISSING_ORTHO_ACTIVE_META] distributed orthogonalize requires at least one "
            "non-null dist_meta"
        )

    first_meta = active_metas[0]
    meta_id = format_meta_id(first_meta)
    global_shape = getattr(first_meta, "global_shape", None)
    if global_shape is None or len(global_shape) != 2:
        raise RuntimeError(
            "[DION_MISSING_ORTHO_GLOBAL_SHAPE] distributed orthogonalize requires exact "
            f"global_shape meta_id={meta_id} global_shape={global_shape}"
        )
    param_config = getattr(first_meta, "param_config", None)
    if param_config is None:
        raise RuntimeError(
            "[DION_MISSING_ORTHO_PARAM_CONFIG] distributed orthogonalize requires param_config "
            f"meta_id={meta_id}"
        )

    m_global, n_global = (int(global_shape[0]), int(global_shape[1]))
    is_transposed = bool(getattr(param_config, "is_transposed", False))
    global_rows = n_global if is_transposed else m_global
    if global_rows <= 0:
        raise RuntimeError(
            "[DION_INVALID_ORTHO_GLOBAL_ROWS] distributed orthogonalize requires positive "
            f"global_rows meta_id={meta_id} global_shape={(m_global, n_global)} "
            f"is_transposed={int(is_transposed)} global_rows={global_rows}"
        )

    for index, dist_meta in enumerate(active_metas[1:], start=1):
        other_shape = getattr(dist_meta, "global_shape", None)
        other_config = getattr(dist_meta, "param_config", None)
        if other_shape is None or len(other_shape) != 2:
            raise RuntimeError(
                "[DION_MISSING_ORTHO_GLOBAL_SHAPE] distributed orthogonalize requires exact "
                f"global_shape meta_id={format_meta_id(dist_meta)} global_shape={other_shape}"
            )
        other_shape = (int(other_shape[0]), int(other_shape[1]))
        other_is_transposed = bool(
            getattr(other_config, "is_transposed", False) if other_config is not None else False
        )
        if other_shape != (m_global, n_global) or other_is_transposed != is_transposed:
            raise RuntimeError(
                "[DION_INCONSISTENT_ORTHO_ROW_META] all batch entries routed into one "
                "distributed orthogonalize call must share the same logical Dion row axis "
                f"slot={index} first_meta_id={meta_id} meta_id={format_meta_id(dist_meta)} "
                f"first_global_shape={(m_global, n_global)} other_global_shape={other_shape} "
                f"first_is_transposed={int(is_transposed)} "
                f"other_is_transposed={int(other_is_transposed)}"
            )

    row_sizes = _canonical_shard_sizes(global_rows, ortho_world_size)
    expected_local_rows = int(row_sizes[ortho_rank])
    if expected_local_rows != int(local_rows):
        raise RuntimeError(
            "[DION_ORTHO_ROW_LAYOUT_MISMATCH] local P row shard does not match the canonical "
            "logical split reconstructed from global Dion metadata "
            f"meta_id={meta_id} global_shape={(m_global, n_global)} "
            f"is_transposed={int(is_transposed)} global_rows={global_rows} "
            f"ortho_world_size={ortho_world_size} ortho_rank={ortho_rank} "
            f"expected_local_rows={expected_local_rows} local_rows={int(local_rows)} "
            f"row_sizes={tuple(int(size) for size in row_sizes)}"
        )

    return row_sizes, global_rows


def _all_gather_batch_shards(
    local_tensor: Tensor,
    *,
    batch_sizes: Sequence[int],
    group: torch.distributed.ProcessGroup,
) -> Tensor:
    """All-gather a batch-sharded tensor with variable batch counts."""
    max_batch = max(int(size) for size in batch_sizes)
    padded_shape = (max_batch, *tuple(local_tensor.shape[1:]))
    padded = torch.zeros(
        padded_shape,
        device=local_tensor.device,
        dtype=local_tensor.dtype,
    )
    local_batch = int(local_tensor.size(0))
    if local_batch > 0:
        padded[:local_batch].copy_(local_tensor)
    gathered = [torch.empty_like(padded) for _ in range(dist.get_world_size(group))]
    dist.all_gather(gathered, padded, group=group)
    parts = [
        shard[: int(batch_size)]
        for shard, batch_size in zip(gathered, batch_sizes)
        if int(batch_size) > 0
    ]
    if not parts:
        return torch.empty(
            (0, *tuple(local_tensor.shape[1:])),
            device=local_tensor.device,
            dtype=local_tensor.dtype,
        )
    return torch.cat(parts, dim=0).contiguous()


def _all_gather_row_shards(
    local_tensor: Tensor,
    *,
    row_sizes: Sequence[int],
    group: torch.distributed.ProcessGroup,
) -> Tensor:
    """All-gather a row-sharded tensor with variable local row counts."""
    max_rows = max(int(size) for size in row_sizes)
    padded_shape = (int(local_tensor.size(0)), max_rows, int(local_tensor.size(2)))
    padded = torch.zeros(
        padded_shape,
        device=local_tensor.device,
        dtype=local_tensor.dtype,
    )
    local_rows = int(local_tensor.size(1))
    if local_rows > 0:
        padded[:, :local_rows].copy_(local_tensor)
    gathered = [torch.empty_like(padded) for _ in range(dist.get_world_size(group))]
    dist.all_gather(gathered, padded, group=group)
    parts = [
        shard[:, : int(row_size)]
        for shard, row_size in zip(gathered, row_sizes)
        if int(row_size) > 0
    ]
    if not parts:
        return torch.empty(
            (int(local_tensor.size(0)), 0, int(local_tensor.size(2))),
            device=local_tensor.device,
            dtype=local_tensor.dtype,
        )
    return torch.cat(parts, dim=1).contiguous()


def _row_to_batch(
    local_tensor: Tensor,
    *,
    row_sizes: Sequence[int],
    batch_sizes: Sequence[int],
    group: torch.distributed.ProcessGroup,
) -> Tensor:
    """Exchange row-sharded batches into batch-sharded full matrices."""
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    local_batch = int(batch_sizes[rank])
    r = int(local_tensor.size(2))
    inputs = []
    for dst_rank, batch_size in enumerate(batch_sizes):
        batch_size = int(batch_size)
        if batch_size > 0:
            batch_start, batch_end = _split_range(int(local_tensor.size(0)), world_size, dst_rank)
            inputs.append(local_tensor[batch_start:batch_end].contiguous())
        else:
            inputs.append(
                torch.empty(
                    (0, int(local_tensor.size(1)), r),
                    device=local_tensor.device,
                    dtype=local_tensor.dtype,
                )
            )
    outputs = [
        torch.empty(
            (local_batch, int(row_size), r),
            device=local_tensor.device,
            dtype=local_tensor.dtype,
        )
        for row_size in row_sizes
    ]
    dist.all_to_all(outputs, inputs, group=group)
    parts = [part for part in outputs if int(part.size(1)) > 0]
    if not parts:
        return torch.empty(
            (local_batch, 0, r),
            device=local_tensor.device,
            dtype=local_tensor.dtype,
        )
    return torch.cat(parts, dim=1).contiguous()


def _batch_to_row(
    local_tensor: Tensor,
    *,
    row_sizes: Sequence[int],
    batch_sizes: Sequence[int],
    group: torch.distributed.ProcessGroup,
) -> Tensor:
    """Exchange batch-sharded full matrices back into row-sharded batches."""
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    local_rows = int(row_sizes[rank])
    r = int(local_tensor.size(2))
    inputs = []
    row_cursor = 0
    for row_size in row_sizes:
        row_size = int(row_size)
        next_cursor = row_cursor + row_size
        inputs.append(local_tensor[:, row_cursor:next_cursor].contiguous())
        row_cursor = next_cursor
    outputs = [
        torch.empty(
            (int(batch_size), local_rows, r),
            device=local_tensor.device,
            dtype=local_tensor.dtype,
        )
        for batch_size in batch_sizes
    ]
    dist.all_to_all(outputs, inputs, group=group)
    parts = [part for part in outputs if int(part.size(0)) > 0]
    if not parts:
        return torch.empty(
            (0, local_rows, r),
            device=local_tensor.device,
            dtype=local_tensor.dtype,
        )
    return torch.cat(parts, dim=0).contiguous()


def _make_sharded_sketch(
    *,
    logical_seed_keys: Sequence[object],
    oversample: float,
    row_sizes: Sequence[int],
    row_rank: int,
    device: torch.device,
    dtype: torch.dtype,
    r: int,
) -> Tensor:
    """Return the local row shard of one logical distributed sketch batch."""
    local_rows = int(row_sizes[row_rank])
    row_offset = sum(int(size) for size in row_sizes[:row_rank])
    global_rows = sum(int(size) for size in row_sizes)
    k = math.ceil(oversample * r / 128.0) * 128
    if k <= 0:
        raise RuntimeError(
            f"[DION_INVALID_SKETCH_RANK] r={r} oversample={oversample} k={k}"
        )
    std = math.sqrt(1.0 / k)
    sketch = torch.empty(
        (len(logical_seed_keys), k, local_rows),
        device=device,
        dtype=dtype,
    )
    for index, seed_key in enumerate(logical_seed_keys):
        global_slot = _seeded_normal_tensor(
            (k, global_rows),
            seed_key=seed_key,
            device=device,
            dtype=dtype,
            std=std,
        )
        sketch[index].copy_(global_slot[:, row_offset : row_offset + local_rows])
    return sketch


def generate_random_sketch_matrix(
    P: Tensor,
    oversample: float = 1.25,
    sketch_fn=None,
) -> Tensor:
    """Local sketch generation contract for regular-tensor orthogonalization."""
    assert P.ndim >= 3, "P must have batch dimension"

    batch_size = P.shape[:-2]
    m = P.size(-2)
    r = P.size(-1)
    k = math.ceil(oversample * r / 128.0) * 128
    std = math.sqrt(1.0 / k)

    if sketch_fn is not None:
        return sketch_fn(P, oversample)

    S = torch.empty((*batch_size, k, m), device=P.device, dtype=P.dtype)
    S.normal_(std=std)
    return S


def distributed_orthogonalize(
    optimizer,
    P_batch: torch.Tensor,
    *,
    ortho_group: Optional[torch.distributed.ProcessGroup],
    oversample: float = 1.25,
    dist_metas: Optional[List] = None,
    real_batch_size: Optional[int] = None,
) -> torch.Tensor:
    """Distributed orthogonalization using plain tensors and explicit collectives."""
    del real_batch_size
    batch_size = int(P_batch.size(0))
    original_dtype = P_batch.dtype

    if ortho_group is not None:
        ortho_world_size = dist.get_world_size(ortho_group)
        ortho_rank = dist.get_rank(ortho_group)
    else:
        ortho_world_size = 1
        ortho_rank = 0

    if ortho_group is None or ortho_world_size <= 1:
        result = torch.empty_like(P_batch)
        for index in range(batch_size):
            result[index] = optimizer._orthogonalize(P_batch[index], rcqr_oversample=oversample)
        return result

    logical_seed_keys = sketch_keys_for_update(
        optimizer,
        dist_metas=dist_metas[:batch_size] if dist_metas is not None else None,
        tag="logical_local",
    )
    if logical_seed_keys is None or len(logical_seed_keys) != batch_size:
        raise RuntimeError(
            "[DION_MISSING_DISTRIBUTED_SKETCH_KEYS] "
            f"batch_size={batch_size} logical_seed_keys="
            f"{0 if logical_seed_keys is None else len(logical_seed_keys)}"
        )

    local_rows = int(P_batch.size(1))
    r = int(P_batch.size(2))
    row_sizes, global_rows = _resolve_row_sizes_from_dist_meta(
        dist_metas=dist_metas,
        batch_size=batch_size,
        ortho_world_size=ortho_world_size,
        ortho_rank=ortho_rank,
        local_rows=local_rows,
    )
    batch_sizes = [
        _split_range(batch_size, ortho_world_size, rank)[1]
        - _split_range(batch_size, ortho_world_size, rank)[0]
        for rank in range(ortho_world_size)
    ]
    local_batch_start, local_batch_end = _split_range(batch_size, ortho_world_size, ortho_rank)

    with _dion_math_precision_context():
        P_local = P_batch.to(dtype=torch.float32)

        if global_rows <= r:
            P_owned = _row_to_batch(
                P_local,
                row_sizes=row_sizes,
                batch_sizes=batch_sizes,
                group=ortho_group,
            )
            if int(P_owned.size(0)) > 0:
                Q_owned, _ = torch.linalg.qr(
                    P_owned.to(dtype=torch.float32),
                    mode="reduced",
                )
                P_owned = Q_owned
            P_local = _batch_to_row(
                P_owned.to(dtype=torch.float32),
                row_sizes=row_sizes,
                batch_sizes=batch_sizes,
                group=ortho_group,
            )
            return P_local.to(original_dtype).contiguous()

        S_local = _make_sharded_sketch(
            logical_seed_keys=logical_seed_keys,
            oversample=oversample,
            row_sizes=row_sizes,
            row_rank=ortho_rank,
            device=P_local.device,
            dtype=P_local.dtype,
            r=r,
        )
        SP_local = S_local @ P_local
        dist.all_reduce(SP_local, op=dist.ReduceOp.SUM, group=ortho_group)
        SP_owned = SP_local[local_batch_start:local_batch_end]
        if int(SP_owned.size(0)) > 0:
            SP_owned = SP_owned.to(dtype=torch.float32)
            R_local = torch.linalg.qr(SP_owned, mode="r")[1]
        else:
            R_local = torch.empty(
                (0, r, r),
                device=P_local.device,
                dtype=torch.float32,
            )
        R = _all_gather_batch_shards(R_local, batch_sizes=batch_sizes, group=ortho_group)
        R32 = R.to(dtype=torch.float32)
        P_local32 = P_local.to(dtype=torch.float32)
        P_local = torch.linalg.solve_triangular(
            R32,
            P_local32,
            upper=True,
            left=False,
        ).to(dtype=torch.float32)

        P_local32 = P_local.to(dtype=torch.float32)
        PP_local = P_local32.mT @ P_local32
        dist.all_reduce(PP_local, op=dist.ReduceOp.SUM, group=ortho_group)
        PP_owned = PP_local[local_batch_start:local_batch_end]
        if int(PP_owned.size(0)) > 0:
            R_local = torch.linalg.cholesky_ex(PP_owned, upper=True)[0]
        else:
            R_local = torch.empty(
                (0, r, r),
                device=P_local.device,
                dtype=torch.float32,
            )
        R = _all_gather_batch_shards(R_local, batch_sizes=batch_sizes, group=ortho_group)
        P_local = torch.linalg.solve_triangular(
            R.to(dtype=torch.float32),
            P_local32,
            upper=True,
            left=False,
        ).to(dtype=torch.float32)
        return P_local.to(original_dtype).contiguous()
def orthogonalize_local_slice(
    optimizer,
    P_slice: torch.Tensor,
    dist_metas: Optional[List] = None,
    *,
    tag: str = "local",
    sketch_tag: Optional[str] = None,
    sketch_fn=None,
    use_seeded_sketch: bool = True,
) -> torch.Tensor:
    """Local orthogonalization path for a batched slice."""
    if P_slice.size(0) == 0:
        return P_slice

    if use_seeded_sketch and sketch_fn is None and dist_metas is not None:
        sketch_fn = make_seeded_sketch_for_update(
            optimizer,
            dist_metas=dist_metas,
            tag=sketch_tag if sketch_tag is not None else tag,
        )

    P_ortho_slice = orthogonalize(
        P_slice,
        rcqr_oversample=optimizer.defaults['rcqr_oversample'],
        sketch_fn=sketch_fn,
    ).to(torch.float32)
    return P_ortho_slice.to(P_slice.dtype)


def orthogonalize_local_matrix_batch(
    optimizer,
    P_slice: torch.Tensor,
    dist_metas: Optional[List] = None,
    *,
    tag: str = "local_matrix",
) -> torch.Tensor:
    """Regular-Tensor local orthogonalization for one logical full matrix."""
    return orthogonalize_local_slice(
        optimizer,
        P_slice,
        dist_metas=dist_metas,
        tag=tag,
        sketch_tag="logical_local",
    )


def _default_sketch_matrix(P: Tensor, oversample: float) -> Tensor:
    """Generate default random sketch matrix without synchronization.

    Used when no sketch_fn is provided (single-rank case).

    Args:
        P: Matrix being orthogonalized, shape (..., m, r)
        oversample: Oversampling factor

    Returns:
        Sketch matrix S of shape (..., k, m)
    """

    batch_shape = P.shape[:-2]
    m = P.size(-2)
    r = P.size(-1)

    # Round k to multiple of 128 for efficiency (matches reference)
    k = math.ceil(oversample * r / 128.0) * 128

    std = math.sqrt(1.0 / k)

    S = torch.empty((*batch_shape, k, m), device=P.device, dtype=P.dtype)
    S.normal_(std=std)

    return S


def reshard_q_along_tp(
    Q: Tensor,
    tp_group: Optional[torch.distributed.ProcessGroup],
    tp_rank: int,
) -> Tensor:
    """Re-shard Q matrix along TP dimension after update.

    Reference: dion.py line 1159 - update_Q_matrix_ with Q_sharded_placements

    Args:
        Q: Full Q matrix of shape (n/fs, r)
        tp_group: Tensor parallel process group
        tp_rank: This rank's position in TP group

    Returns:
        TP-sharded Q matrix of shape (n/fs, r/tp)
    """
    if tp_group is None or dist.get_world_size(tp_group) == 1:
        return Q

    # Split Q along column dimension
    tp_size = dist.get_world_size(tp_group)
    n, r_total = Q.shape
    r_per_rank = r_total // tp_size

    # Extract this rank's shard
    start_col = tp_rank * r_per_rank
    end_col = (tp_rank + 1) * r_per_rank if tp_rank < tp_size - 1 else r_total

    Q_shard = Q[:, start_col:end_col].contiguous()

    return Q_shard
