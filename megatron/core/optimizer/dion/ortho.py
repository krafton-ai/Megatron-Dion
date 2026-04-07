"""Orthogonalization functions for Dion optimizer."""

import contextlib
import hashlib
import math
from typing import Callable, Optional, Sequence

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor import randn as dtensor_randn
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset


@contextlib.contextmanager
def _dion_ortho_precision_context():
    """Disable TF32 only for Dion orthogonalization kernels.

    Reference Dion orthogonalization expects float32 linear algebra quality.
    Duplex training enables TF32 globally for throughput, which is acceptable
    for the model forward/backward but too aggressive for RCQR/Cholesky.
    Keep the global setting intact and only fence off the sensitive ortho path.
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
    with _dion_ortho_precision_context():
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
            _, R = torch.linalg.qr(SP.to(dtype=torch.float32), mode='r')

            # Step 4: Solve for orthogonal factor
            P = torch.linalg.solve_triangular(
                R, P.to(dtype=torch.float32), upper=True, left=False
            )

            # Step 5: Cholesky QR for better orthogonalization
            PP = P.mT @ P  # Always do float32 matrix multiply

            R, info = torch.linalg.cholesky_ex(PP, upper=True)

            P = torch.linalg.solve_triangular(
                R, P, upper=True, left=False
            )

            return P.to(original_dtype).contiguous()


def _dtensor_from_local(local_tensor: Tensor, ref: DTensor) -> DTensor:
    """Create a DTensor from a local shard using an existing DTensor as metadata reference."""
    return DTensor.from_local(
        local_tensor,
        device_mesh=ref.device_mesh,
        placements=ref.placements,
    )


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


def _extract_dtensor_local_shard_from_global(
    global_tensor: Tensor,
    *,
    mesh,
    placements,
) -> Tensor:
    """Slice the topology-invariant local DTensor shard from one logical global tensor."""
    local_shape, global_offset = compute_local_shape_and_global_offset(
        tuple(int(dim) for dim in global_tensor.shape),
        mesh,
        placements,
    )
    local_shape = tuple(int(dim) for dim in local_shape)
    global_offset = tuple(int(off) for off in global_offset)
    if any(dim <= 0 for dim in local_shape):
        raise RuntimeError(
            "[DION_INVALID_DISTRIBUTED_SKETCH_LOCAL_SHAPE] "
            f"global_shape={tuple(int(dim) for dim in global_tensor.shape)} "
            f"local_shape={local_shape} global_offset={global_offset} "
            f"placements={tuple(str(p) for p in placements)}"
        )
    slices = tuple(
        slice(offset, offset + size)
        for offset, size in zip(global_offset, local_shape)
    )
    local_tensor = global_tensor[slices].contiguous()
    if tuple(local_tensor.shape) != local_shape:
        raise RuntimeError(
            "[DION_INVALID_DISTRIBUTED_SKETCH_LOCAL_SLICE] "
            f"expected_shape={local_shape} actual_shape={tuple(local_tensor.shape)} "
            f"global_offset={global_offset} placements={tuple(str(p) for p in placements)}"
        )
    return local_tensor


def generate_random_sketch_matrix(
    P: Tensor,
    oversample: float = 1.25,
    shard_mesh_dim: Optional[int] = None,
    sketch_fn=None,
    logical_seed_keys: Optional[Sequence[object]] = None,
) -> Tensor:
    """Exact reference sketch generation contract for local or DTensor orthogonalization."""
    assert P.ndim >= 3, "P must have batch dimension"

    batch_size = P.shape[:-2]
    m = P.size(-2)
    r = P.size(-1)
    k = math.ceil(oversample * r / 128.0) * 128
    std = math.sqrt(1.0 / k)

    if isinstance(P, DTensor):
        s_placements = list(P.placements)
        if shard_mesh_dim is not None:
            s_placements[shard_mesh_dim] = Shard(P.ndim - 1)

        if logical_seed_keys is not None:
            if sketch_fn is not None:
                raise RuntimeError(
                    "[DION_INVALID_SKETCH_CONTRACT] DTensor logical_seed_keys and sketch_fn are mutually exclusive"
                )
            if len(batch_size) != 1:
                raise RuntimeError(
                    "[DION_INVALID_DISTRIBUTED_SKETCH_BATCH] "
                    f"expected single batch dimension, got shape={tuple(P.shape)}"
                )
            global_batch = int(batch_size[0])
            local_batch = int(P.to_local().shape[0])
            batch_sharded = any(p.is_shard(0) for p in s_placements)

            if batch_sharded:
                if local_batch != len(logical_seed_keys):
                    raise RuntimeError(
                        "[DION_DISTRIBUTED_SKETCH_META_MISMATCH] "
                        f"global_batch={global_batch} local_batch={local_batch} "
                        f"logical_seed_keys={len(logical_seed_keys)}"
                    )

                # For exact DTensor orthogonalization, the matrix dimensions are unsharded.
                # If the batch dimension is sharded, each rank only materializes its local
                # batch slots. Each local slot must still come from the same logical-seeded
                # global sketch tensor, which here is identical to the local slot because
                # only the batch axis is sharded.
                local_slots = []
                for seed_key in logical_seed_keys:
                    local_slots.append(
                        _seeded_normal_tensor(
                            (k, m),
                            seed_key=seed_key,
                            device=P.device,
                            dtype=P.dtype,
                            std=std,
                        )
                    )

                local_S = torch.stack(local_slots, dim=0)
                return DTensor.from_local(
                    local_S,
                    device_mesh=P.device_mesh,
                    placements=tuple(s_placements),
                )

            if global_batch != len(logical_seed_keys):
                raise RuntimeError(
                    "[DION_DISTRIBUTED_SKETCH_META_MISMATCH] "
                    f"global_batch={global_batch} logical_seed_keys={len(logical_seed_keys)}"
                )

            slot_placements = []
            for placement in s_placements:
                if placement.is_shard():
                    shard_dim = int(placement.dim)
                    if shard_dim <= 0:
                        raise RuntimeError(
                            "[DION_INVALID_SLOT_SHARD_DIM] "
                            f"slot sketch cannot shard removed batch dim: dim={shard_dim}"
                        )
                    slot_placements.append(Shard(shard_dim - 1))
                else:
                    slot_placements.append(placement)

            local_slots = []
            slot_mesh = P.device_mesh
            slot_placements = tuple(slot_placements)
            for seed_key in logical_seed_keys:
                # Build the logical global sketch slot from the logical seed, then
                # take this rank's DTensor shard explicitly. Relying on
                # `dtensor_randn()` here makes the seeded sketch depend on the
                # current mesh embedding, which breaks TP topology invariance.
                slot_global = _seeded_normal_tensor(
                    (k, m),
                    seed_key=seed_key,
                    device=P.device,
                    dtype=P.dtype,
                    std=std,
                )
                local_slots.append(
                    _extract_dtensor_local_shard_from_global(
                        slot_global,
                        mesh=slot_mesh,
                        placements=slot_placements,
                    )
                )

            local_S = torch.stack(local_slots, dim=0)
            return DTensor.from_local(
                local_S,
                device_mesh=P.device_mesh,
                placements=tuple(s_placements),
            )

        if sketch_fn is not None:
            local_P = P.to_local()
            local_S = sketch_fn(local_P, oversample)
            return DTensor.from_local(
                local_S,
                device_mesh=P.device_mesh,
                placements=tuple(s_placements),
            )

        S = dtensor_randn(
            (*batch_size, k, m),
            device_mesh=P.device_mesh,
            dtype=P.dtype,
            placements=s_placements,
        )
        return S * std

    if sketch_fn is not None:
        return sketch_fn(P, oversample)

    if shard_mesh_dim is not None:
        raise TypeError("Must use DTensor parameters for sharded random sketch.")

    S = torch.empty((*batch_size, k, m), device=P.device, dtype=P.dtype)
    S.normal_(std=std)
    return S


@torch.compile()
def _orthogonalize_dtensor_exact_compiled(
    P: Tensor,
    oversample: float = 1.25,
    sketch_fn=None,
    logical_seed_keys: Optional[Sequence[object]] = None,
    batch_meta_ids: Optional[Sequence[str]] = None,
) -> Tensor:
    """Compiled exact `dion_reference.py::orthogonalize()` contract."""
    assert P.ndim >= 3, "Expected P to have batch dimension"
    original_dtype = P.dtype
    P_local = P.to_local() if isinstance(P, DTensor) else P

    if isinstance(P, DTensor):
        assert not any(p.is_shard(P.ndim - 2) for p in P.placements)
        assert not any(p.is_shard(P.ndim - 1) for p in P.placements)

    if P.size(-2) <= P.size(-1):
        P_local, _ = torch.linalg.qr(P_local.to(dtype=torch.float32))
    else:
        S = generate_random_sketch_matrix(
            P,
            oversample,
            sketch_fn=sketch_fn,
            logical_seed_keys=logical_seed_keys,
        )
        S_local = S.to_local() if isinstance(S, DTensor) else S

        SP = S_local @ P_local
        _, R = torch.linalg.qr(SP.to(dtype=torch.float32), mode="r")
        P_local = torch.linalg.solve_triangular(
            R, P_local.to(dtype=torch.float32), upper=True, left=False
        )

        PP = P_local.mT @ P_local
        R, _ = torch.linalg.cholesky_ex(PP, upper=True)
        P_local = torch.linalg.solve_triangular(R, P_local, upper=True, left=False)

    if isinstance(P, DTensor):
        return _dtensor_from_local(P_local.to(original_dtype).contiguous(), ref=P)
    return P_local.to(original_dtype).contiguous()


def _orthogonalize_dtensor_exact_debug(
    P: Tensor,
    oversample: float = 1.25,
    sketch_fn=None,
    logical_seed_keys: Optional[Sequence[object]] = None,
    batch_meta_ids: Optional[Sequence[str]] = None,
) -> Tensor:
    """Debuggable exact `dion_reference.py::orthogonalize()` contract."""
    assert P.ndim >= 3, "Expected P to have batch dimension"
    original_dtype = P.dtype
    P_local = P.to_local() if isinstance(P, DTensor) else P

    if isinstance(P, DTensor):
        assert not any(p.is_shard(P.ndim - 2) for p in P.placements)
        assert not any(p.is_shard(P.ndim - 1) for p in P.placements)
    if P.size(-2) <= P.size(-1):
        P_local, _ = torch.linalg.qr(P_local.to(dtype=torch.float32))
    else:
        S = generate_random_sketch_matrix(
            P,
            oversample,
            sketch_fn=sketch_fn,
            logical_seed_keys=logical_seed_keys,
        )
        S_local = S.to_local() if isinstance(S, DTensor) else S

        SP = S_local @ P_local
        _, R = torch.linalg.qr(SP.to(dtype=torch.float32), mode="r")
        P_local = torch.linalg.solve_triangular(
            R, P_local.to(dtype=torch.float32), upper=True, left=False
        )

        PP = P_local.mT @ P_local
        R, _ = torch.linalg.cholesky_ex(PP, upper=True)
        P_local = torch.linalg.solve_triangular(R, P_local, upper=True, left=False)

    if isinstance(P, DTensor):
        return _dtensor_from_local(P_local.to(original_dtype).contiguous(), ref=P)
    return P_local.to(original_dtype).contiguous()


def orthogonalize_dtensor_exact(
    P: Tensor,
    oversample: float = 1.25,
    sketch_fn=None,
    logical_seed_keys: Optional[Sequence[object]] = None,
    batch_meta_ids: Optional[Sequence[str]] = None,
) -> Tensor:
    """Exact `dion_reference.py::orthogonalize()` contract."""
    with _dion_ortho_precision_context():
        return _orthogonalize_dtensor_exact_debug(
            P,
            oversample=oversample,
            sketch_fn=sketch_fn,
            logical_seed_keys=logical_seed_keys,
            batch_meta_ids=batch_meta_ids,
        )


@torch.compile()
def _distributed_orthogonalize_dtensor_exact_compiled(
    P: DTensor,
    oversample: float = 1.25,
    shard_mesh_dim: Optional[int] = None,
    sketch_fn=None,
    logical_seed_keys: Optional[Sequence[object]] = None,
) -> DTensor:
    """Compiled exact `dion_reference.py::distributed_orthogonalize()` contract."""
    assert isinstance(P, DTensor)
    assert not any(p.is_partial() for p in P.placements)
    assert not any(p.is_shard(P.ndim - 1) for p in P.placements)
    assert P.ndim >= 3, "Expected P to have batch dimension"
    original_dtype = P.dtype
    original_placements = P.placements

    fully_replicated_placements = [Replicate() for _ in P.placements]
    batch_sharded_placements = fully_replicated_placements.copy()
    if shard_mesh_dim is not None:
        batch_sharded_placements[shard_mesh_dim] = Shard(0)

    if P.size(-2) <= P.size(-1):
        P_single = P.redistribute(placements=batch_sharded_placements)
        Q_local, _ = torch.linalg.qr(
            P_single.to_local().to(dtype=torch.float32), mode="reduced"
        )
        P = _dtensor_from_local(
            Q_local.to(original_dtype).contiguous(),
            ref=P_single,
        ).redistribute(placements=original_placements)
    else:
        S = generate_random_sketch_matrix(
            P,
            oversample,
            shard_mesh_dim=shard_mesh_dim,
            sketch_fn=sketch_fn,
            logical_seed_keys=logical_seed_keys,
        )
        SP: DTensor = S @ P

        SP_single = SP.redistribute(placements=batch_sharded_placements)
        _, R_local = torch.linalg.qr(SP_single.to_local().to(dtype=torch.float32), mode="r")
        R = _dtensor_from_local(R_local, ref=SP_single).redistribute(
            placements=fully_replicated_placements
        )

        P_local = torch.linalg.solve_triangular(
            R.to_local(),
            P.to_local().to(dtype=torch.float32),
            upper=True,
            left=False,
        )
        P = _dtensor_from_local(P_local, ref=P)

        PP: DTensor = P.mT @ P
        PP_single = PP.redistribute(placements=batch_sharded_placements)
        R_local, _ = torch.linalg.cholesky_ex(PP_single.to_local(), upper=True)
        R = _dtensor_from_local(R_local, ref=PP_single).redistribute(
            placements=fully_replicated_placements
        )

        P_local = torch.linalg.solve_triangular(
            R.to_local(),
            P.to_local(),
            upper=True,
            left=False,
        )
        P = _dtensor_from_local(P_local.to(original_dtype).contiguous(), ref=P)

    assert P.dtype == original_dtype, "Output dtype mismatch"
    assert P.placements == original_placements, "Output placements mismatch"
    return P


def _distributed_orthogonalize_dtensor_exact_debug(
    P: DTensor,
    oversample: float = 1.25,
    shard_mesh_dim: Optional[int] = None,
    sketch_fn=None,
    logical_seed_keys: Optional[Sequence[object]] = None,
    batch_meta_ids: Optional[Sequence[str]] = None,
) -> DTensor:
    """Debuggable exact `dion_reference.py::distributed_orthogonalize()` contract."""
    assert isinstance(P, DTensor)
    assert not any(p.is_partial() for p in P.placements)
    assert not any(p.is_shard(P.ndim - 1) for p in P.placements)
    assert P.ndim >= 3, "Expected P to have batch dimension"
    original_dtype = P.dtype
    original_placements = P.placements

    fully_replicated_placements = [Replicate() for _ in P.placements]
    batch_sharded_placements = fully_replicated_placements.copy()
    if shard_mesh_dim is not None:
        batch_sharded_placements[shard_mesh_dim] = Shard(0)

    if P.size(-2) <= P.size(-1):
        P_single = P.redistribute(placements=batch_sharded_placements)
        Q_local, _ = torch.linalg.qr(
            P_single.to_local().to(dtype=torch.float32), mode="reduced"
        )
        P = _dtensor_from_local(
            Q_local.to(original_dtype).contiguous(),
            ref=P_single,
        ).redistribute(placements=original_placements)
    else:
        S = generate_random_sketch_matrix(
            P,
            oversample,
            shard_mesh_dim=shard_mesh_dim,
            sketch_fn=sketch_fn,
            logical_seed_keys=logical_seed_keys,
        )
        SP: DTensor = S @ P

        P_single = SP.redistribute(placements=batch_sharded_placements)
        _, R_local = torch.linalg.qr(P_single.to_local().to(dtype=torch.float32), mode="r")
        R = _dtensor_from_local(R_local, ref=P_single).redistribute(
            placements=fully_replicated_placements
        )

        P_local = torch.linalg.solve_triangular(
            R.to_local(),
            P.to_local().to(dtype=torch.float32),
            upper=True,
            left=False,
        )
        P = _dtensor_from_local(P_local, ref=P)

        PP: DTensor = P.mT @ P
        PP_single = PP.redistribute(placements=batch_sharded_placements)
        R_local, _ = torch.linalg.cholesky_ex(PP_single.to_local(), upper=True)
        R = _dtensor_from_local(R_local, ref=PP_single).redistribute(
            placements=fully_replicated_placements
        )

        P_local = torch.linalg.solve_triangular(
            R.to_local(),
            P.to_local(),
            upper=True,
            left=False,
        )
        P = _dtensor_from_local(P_local.to(original_dtype).contiguous(), ref=P)

    assert P.dtype == original_dtype, "Output dtype mismatch"
    assert P.placements == original_placements, "Output placements mismatch"
    return P


def distributed_orthogonalize_dtensor_exact(
    P: DTensor,
    oversample: float = 1.25,
    shard_mesh_dim: Optional[int] = None,
    sketch_fn=None,
    logical_seed_keys: Optional[Sequence[object]] = None,
    batch_meta_ids: Optional[Sequence[str]] = None,
) -> DTensor:
    """Exact `dion_reference.py::distributed_orthogonalize()` contract."""
    with _dion_ortho_precision_context():
        return _distributed_orthogonalize_dtensor_exact_debug(
            P,
            oversample=oversample,
            shard_mesh_dim=shard_mesh_dim,
            sketch_fn=sketch_fn,
            logical_seed_keys=logical_seed_keys,
            batch_meta_ids=batch_meta_ids,
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
