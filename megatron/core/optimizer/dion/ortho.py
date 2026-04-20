"""Orthogonalization and sketch helpers for Dion."""

import contextlib
import hashlib
import math
from typing import Any, Callable, List, Optional, Sequence

import torch
import torch.distributed as dist
from torch import Tensor

from .utils import format_meta_id, get_global_shape


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


@torch.compile(fullgraph=True)
def _qr_r_factor(X: Tensor) -> Tensor:
    return torch.linalg.qr(X, mode="r")[1].to(dtype=torch.float32)


@torch.compile(fullgraph=True)
def _solve_triangular_right(R: Tensor, X: Tensor) -> Tensor:
    return torch.linalg.solve_triangular(
        R.to(dtype=torch.float32),
        X.to(dtype=torch.float32),
        upper=True,
        left=False,
    ).to(dtype=torch.float32)


@torch.compile(fullgraph=True)
def _cholesky_upper_factor(X: Tensor) -> Tensor:
    return torch.linalg.cholesky_ex(
        X.to(dtype=torch.float32),
        upper=True,
    )[0].to(dtype=torch.float32)


@torch.compile()
def orthogonalize(
    P: Tensor,
    rcqr_oversample: float = 1.25,
    make_sketch=None,
) -> Tensor:
    """
    Local orthogonalization with Randomized Cholesky QR.

    Reference: dion.py lines 1265-1305

    Args:
        P: Matrix to orthogonalize, shape (..., m, r)
        rcqr_oversample: Oversampling factor for RCQR (default 1.25)
        make_sketch: Optional explicit sketch generator for the tall-matrix path
    Returns:
        Orthogonalized matrix Q, same shape as P
    """
    assert P.ndim >= 3, "Expected P to have batch dimension"
    original_dtype = P.dtype
    P_local = P

    if P.size(-2) <= P.size(-1):
        P_local = torch.linalg.qr(P_local.to(dtype=torch.float32), mode="reduced")[0]
    else:
        S = generate_random_sketch_matrix(
            P,
            oversample=rcqr_oversample,
            make_sketch=make_sketch,
        )
        R = torch.linalg.qr(
            S @ P_local,
            mode="r",
        )[1].to(dtype=torch.float32)
        P_local = torch.linalg.solve_triangular(
            R,
            P_local.to(dtype=torch.float32),
            upper=True,
            left=False,
        )

        R = torch.linalg.cholesky_ex(P_local.mT @ P_local, upper=True)[0]
        P_local = torch.linalg.solve_triangular(
            R,
            P_local,
            upper=True,
            left=False,
        )

    return P_local.to(original_dtype).contiguous()


def _sketch_seed(seed_key: object) -> int:
    """Map one sketch key to a deterministic per-matrix RNG seed."""
    return int.from_bytes(
        hashlib.blake2b(repr(seed_key).encode("utf-8"), digest_size=8).digest(),
        "little",
    ) & ((1 << 63) - 1)


@torch.compiler.disable
def _seeded_normal_tensor(
    shape: tuple[int, ...],
    *,
    seed: int,
    offset: int = 0,
    device: torch.device,
    dtype: torch.dtype,
    std: float,
) -> Tensor:
    """Generate one topology-invariant sketch tensor from one sketch seed."""
    gen = torch.Generator(device=str(device))
    gen.manual_seed(seed)
    if offset != 0:
        gen.set_offset(offset)
    tensor = torch.empty(shape, device=device, dtype=dtype)
    tensor.normal_(mean=0.0, std=std, generator=gen)
    return tensor


def sketch_keys(
    *,
    dist_metas: Optional[List],
    contract: str,
    step_count: int,
) -> Optional[List[object]]:
    """Return topology-invariant sketch keys for one Dion update batch."""
    if not dist_metas:
        return None

    keys: List[object] = []
    for dist_meta in dist_metas:
        if dist_meta is None:
            keys.append((contract, step_count, None))
            continue
        keys.append(
            (
                contract,
                step_count,
                getattr(dist_meta, "param_uid", None),
                getattr(dist_meta, "param_name", ""),
            )
        )
    return keys


def make_sketch(
    *,
    dist_metas: Optional[List],
    contract: str,
    step_count: int,
):
    """Build a topology-independent sketch generator for one Dion update batch."""
    if not dist_metas:
        return None

    keys = sketch_keys(
        dist_metas=dist_metas,
        contract=contract,
        step_count=step_count,
    )
    seeds = [_sketch_seed(seed_key) for seed_key in keys]

    @torch.compiler.disable
    def build_sketch(P: Tensor, oversample: float) -> Tensor:
        batch_shape = P.shape[:-2]
        if len(batch_shape) == 0:
            batch = 1
        elif len(batch_shape) == 1:
            batch = batch_shape[0]
        else:
            raise RuntimeError(
                "[DION_INVALID_SKETCH_BATCH] "
                f"contract={contract} expected batched 3D tensor, got shape={tuple(P.shape)}"
            )
        if batch != len(seeds):
            raise RuntimeError(
                "[DION_SKETCH_META_MISMATCH] "
                f"contract={contract} batch={batch} sketch_keys={len(seeds)}"
            )
        m = P.size(-2)
        r = P.size(-1)
        k = math.ceil(oversample * r / 128.0) * 128
        if k <= 0:
            raise RuntimeError(
                f"[DION_INVALID_SKETCH_RANK] contract={contract} r={r} oversample={oversample} k={k}"
            )
        std = math.sqrt(1.0 / k)
        if batch == 1 and len(batch_shape) == 0:
            sketch = _seeded_normal_tensor(
                (k, m),
                seed=seeds[0],
                device=P.device,
                dtype=P.dtype,
                std=std,
            )
            return sketch
        sketch = torch.empty((batch, k, m), device=P.device, dtype=P.dtype)
        for idx, seed in enumerate(seeds):
            sketch[idx].copy_(
                _seeded_normal_tensor(
                    (k, m),
                    seed=seed,
                    device=P.device,
                    dtype=P.dtype,
                    std=std,
                )
            )
        return sketch

    return build_sketch


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
    shard_size = base + (1 if rank < remainder else 0)
    return start, start + shard_size


def _canonical_shard_sizes(size: int, world_size: int) -> list[int]:
    """Return the canonical contiguous shard sizes for one axis."""
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
    """Reconstruct distributed P row sizes from metadata only."""
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
    global_shape = get_global_shape(first_meta, 0, 0)
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
        other_shape = get_global_shape(dist_meta, 0, 0)
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
                "distributed orthogonalize call must share the same Dion row axis "
                f"batch_index={index} first_meta_id={meta_id} meta_id={format_meta_id(dist_meta)} "
                f"first_global_shape={(m_global, n_global)} other_global_shape={other_shape} "
                f"first_is_transposed={int(is_transposed)} "
                f"other_is_transposed={int(other_is_transposed)}"
            )

    row_sizes = _canonical_shard_sizes(global_rows, ortho_world_size)
    expected_local_rows = int(row_sizes[ortho_rank])
    if expected_local_rows != int(local_rows):
        raise RuntimeError(
            "[DION_ORTHO_ROW_LAYOUT_MISMATCH] local P row shard does not match the canonical "
            "split reconstructed from global Dion metadata "
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
    world_size = dist.get_world_size(group)
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
    gathered = torch.empty(
        (world_size * max_batch, *tuple(local_tensor.shape[1:])),
        device=local_tensor.device,
        dtype=local_tensor.dtype,
    )
    dist.all_gather_into_tensor(gathered, padded.contiguous(), group=group)
    gathered = gathered.view(world_size, max_batch, *tuple(local_tensor.shape[1:]))
    parts = [
        shard[: int(batch_size)]
        for shard, batch_size in zip(gathered.unbind(dim=0), batch_sizes)
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
    world_size = dist.get_world_size(group)
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
    gathered = torch.empty(
        (world_size * int(local_tensor.size(0)), max_rows, int(local_tensor.size(2))),
        device=local_tensor.device,
        dtype=local_tensor.dtype,
    )
    dist.all_gather_into_tensor(gathered, padded.contiguous(), group=group)
    gathered = gathered.view(
        world_size,
        int(local_tensor.size(0)),
        max_rows,
        int(local_tensor.size(2)),
    )
    parts = [
        shard[:, : int(row_size)]
        for shard, row_size in zip(gathered.unbind(dim=0), row_sizes)
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
    max_rows = max(int(row_size) for row_size in row_sizes)
    padded = torch.zeros(
        (int(local_tensor.size(0)), max_rows, r),
        device=local_tensor.device,
        dtype=local_tensor.dtype,
    )
    local_rows = int(local_tensor.size(1))
    if local_rows > 0:
        padded[:, :local_rows].copy_(local_tensor)
    exchanged = torch.empty(
        (world_size * local_batch, max_rows, r),
        device=local_tensor.device,
        dtype=local_tensor.dtype,
    )
    dist.all_to_all_single(
        exchanged,
        padded.contiguous(),
        output_split_sizes=[local_batch] * world_size,
        input_split_sizes=[int(batch_size) for batch_size in batch_sizes],
        group=group,
    )
    exchanged = exchanged.view(world_size, local_batch, max_rows, r)
    parts = [
        shard[:, : int(row_size)]
        for shard, row_size in zip(exchanged.unbind(dim=0), row_sizes)
        if int(row_size) > 0
    ]
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
    max_rows = max(int(row_size) for row_size in row_sizes)
    local_batch = int(local_tensor.size(0))
    input_padded = torch.zeros(
        (world_size * local_batch, max_rows, r),
        device=local_tensor.device,
        dtype=local_tensor.dtype,
    )
    row_cursor = 0
    for dst_rank, row_size in enumerate(row_sizes):
        row_size = int(row_size)
        next_cursor = row_cursor + row_size
        if row_size > 0 and local_batch > 0:
            dst_start = dst_rank * local_batch
            dst_end = dst_start + local_batch
            input_padded[dst_start:dst_end, :row_size].copy_(
                local_tensor[:, row_cursor:next_cursor]
            )
        row_cursor = next_cursor
    exchanged = torch.empty(
        (sum(int(batch_size) for batch_size in batch_sizes), max_rows, r),
        device=local_tensor.device,
        dtype=local_tensor.dtype,
    )
    dist.all_to_all_single(
        exchanged,
        input_padded.contiguous(),
        output_split_sizes=[int(batch_size) for batch_size in batch_sizes],
        input_split_sizes=[local_batch] * world_size,
        group=group,
    )
    parts = []
    cursor = 0
    for batch_size in batch_sizes:
        batch_size = int(batch_size)
        next_cursor = cursor + batch_size
        if batch_size > 0:
            parts.append(exchanged[cursor:next_cursor, :local_rows])
        cursor = next_cursor
    if not parts:
        return torch.empty(
            (0, local_rows, r),
            device=local_tensor.device,
            dtype=local_tensor.dtype,
        )
    return torch.cat(parts, dim=0).contiguous()


def _reduce_scatter_batch_shards(
    local_tensor: Tensor,
    *,
    batch_sizes: Sequence[int],
    group: torch.distributed.ProcessGroup,
) -> Tensor:
    """Reduce-scatter a batch tensor to one rank-local batch shard."""
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    batch_sizes = [int(size) for size in batch_sizes]
    max_batch = max(batch_sizes)
    total_padded = max_batch * world_size
    padded = torch.zeros(
        (total_padded, *tuple(local_tensor.shape[1:])),
        device=local_tensor.device,
        dtype=local_tensor.dtype,
    )
    batch_size = int(local_tensor.size(0))
    if batch_size != sum(batch_sizes):
        raise RuntimeError(
            "[DION_BATCH_PARTITION_MISMATCH] "
            f"batch_size={batch_size} batch_sizes={tuple(batch_sizes)}"
        )
    cursor = 0
    for shard_rank, shard_batch in enumerate(batch_sizes):
        next_cursor = cursor + shard_batch
        if shard_batch > 0:
            shard_start = shard_rank * max_batch
            shard_end = shard_start + shard_batch
            padded[shard_start:shard_end].copy_(local_tensor[cursor:next_cursor])
        cursor = next_cursor
    reduced = torch.empty(
        (max_batch, *tuple(local_tensor.shape[1:])),
        device=local_tensor.device,
        dtype=local_tensor.dtype,
    )
    dist.reduce_scatter_tensor(
        reduced,
        padded.contiguous(),
        op=dist.ReduceOp.SUM,
        group=group,
    )
    local_batch = int(batch_sizes[rank])
    return reduced[:local_batch].contiguous()


def _make_sharded_sketch(
    *,
    sketch_seeds: Sequence[int],
    oversample: float,
    row_sizes: Sequence[int],
    row_rank: int,
    device: torch.device,
    dtype: torch.dtype,
    r: int,
) -> Tensor:
    """Return the local row shard of one distributed sketch batch."""
    local_rows = int(row_sizes[row_rank])
    k = math.ceil(oversample * r / 128.0) * 128
    if k <= 0:
        raise RuntimeError(
            f"[DION_INVALID_SKETCH_RANK] r={r} oversample={oversample} k={k}"
        )
    std = math.sqrt(1.0 / k)
    rank0_rows = int(row_sizes[0])
    shard_offset = ((row_rank * k * rank0_rows) + 3) // 4 * 4
    sketch = torch.empty(
        (len(sketch_seeds), k, local_rows),
        device=device,
        dtype=dtype,
    )
    for index, seed in enumerate(sketch_seeds):
        sketch[index].copy_(
            _seeded_normal_tensor(
                (k, local_rows),
                seed=seed,
                offset=shard_offset,
                device=device,
                dtype=dtype,
                std=std,
            )
        )
    return sketch


def generate_random_sketch_matrix(
    P: Tensor,
    oversample: float = 1.25,
    make_sketch=None,
) -> Tensor:
    """Local sketch generation contract for regular-tensor orthogonalization."""
    assert P.ndim >= 3, "P must have batch dimension"

    batch_shape = P.shape[:-2]
    m = P.size(-2)
    r = P.size(-1)
    k = math.ceil(oversample * r / 128.0) * 128
    std = math.sqrt(1.0 / k)

    if make_sketch is not None:
        return make_sketch(P, oversample)

    S = torch.empty((*batch_shape, k, m), device=P.device, dtype=P.dtype)
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
        with _dion_math_precision_context():
            P_ortho = orthogonalize(
                P_batch,
                rcqr_oversample=optimizer.defaults["rcqr_oversample"],
            ).to(torch.float32)
        return P_ortho.to(original_dtype).contiguous()

    batch_sketch_keys = sketch_keys(
        dist_metas=dist_metas[:batch_size] if dist_metas is not None else None,
        contract="distributed",
        step_count=optimizer._step_count,
    )
    if batch_sketch_keys is None or len(batch_sketch_keys) != batch_size:
        raise RuntimeError(
            "[DION_MISSING_DISTRIBUTED_SKETCH_KEYS] "
            f"batch_size={batch_size} sketch_keys="
            f"{0 if batch_sketch_keys is None else len(batch_sketch_keys)}"
        )
    batch_sketch_seeds = [_sketch_seed(seed_key) for seed_key in batch_sketch_keys]

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
    with _dion_math_precision_context():
        P_local = P_batch.to(dtype=torch.float32)

        if global_rows <= r:
            P_local = _row_to_batch(
                P_local,
                row_sizes=row_sizes,
                batch_sizes=batch_sizes,
                group=ortho_group,
            )
            if int(P_local.size(0)) > 0:
                P_local = torch.linalg.qr(
                    P_local.to(dtype=torch.float32),
                    mode="reduced",
                )[0].to(dtype=torch.float32)
            P_local = _batch_to_row(
                P_local.to(dtype=torch.float32),
                row_sizes=row_sizes,
                batch_sizes=batch_sizes,
                group=ortho_group,
            )
            return P_local.to(original_dtype).contiguous()

        R_local = _reduce_scatter_batch_shards(
            _make_sharded_sketch(
                sketch_seeds=batch_sketch_seeds,
                oversample=oversample,
                row_sizes=row_sizes,
                row_rank=ortho_rank,
                device=P_local.device,
                dtype=P_local.dtype,
                r=r,
            )
            @ P_local,
            batch_sizes=batch_sizes,
            group=ortho_group,
        )
        if int(R_local.size(0)) > 0:
            R_local = _qr_r_factor(R_local.to(dtype=torch.float32))
        else:
            R_local = torch.empty(
                (0, r, r),
                device=P_local.device,
                dtype=torch.float32,
            )
        P_local = _solve_triangular_right(
            _all_gather_batch_shards(
                R_local,
                batch_sizes=batch_sizes,
                group=ortho_group,
            ),
            P_local,
        )

        R_local = _reduce_scatter_batch_shards(
            P_local.mT @ P_local,
            batch_sizes=batch_sizes,
            group=ortho_group,
        )
        if int(R_local.size(0)) > 0:
            R_local = _cholesky_upper_factor(R_local)
        else:
            R_local = torch.empty(
                (0, r, r),
                device=P_local.device,
                dtype=torch.float32,
            )
        P_local = _solve_triangular_right(
            _all_gather_batch_shards(
                R_local,
                batch_sizes=batch_sizes,
                group=ortho_group,
            ),
            P_local,
        )
        return P_local.to(original_dtype).contiguous()


def _default_sketch_matrix(P: Tensor, oversample: float) -> Tensor:
    """Generate default random sketch matrix without synchronization.

    Used when no explicit sketch generator is provided (single-rank case).

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
