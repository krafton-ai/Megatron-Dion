"""Orthogonalization functions for Dion optimizer."""

import contextlib
import hashlib
import logging
import math
import os
from pathlib import Path
from typing import Callable, Optional, Sequence

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor import randn as dtensor_randn


logger = logging.getLogger(__name__)
_SKETCH_DUMP_COUNTER = 0
_DISTRIBUTED_SKETCH_TRACE_COUNTER = 0
_DISTRIBUTED_ORTHO_TRACE_COUNTER = 0
_ORTHO_TENSOR_DUMP_COUNTER = 0


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


def _should_log_large_r_probe(P: Tensor, r: int) -> bool:
    if os.getenv("DION_DEBUG_LARGE_R", "0") != "1":
        return False
    try:
        min_r = int(os.getenv("DION_DEBUG_LARGE_R_MIN_R", "512"))
    except ValueError as exc:
        raise RuntimeError(
            f"[DION_INVALID_ENV] DION_DEBUG_LARGE_R_MIN_R={os.getenv('DION_DEBUG_LARGE_R_MIN_R')!r} is not an int"
        ) from exc
    return r >= min_r


def _should_log_ortho_stage_probe(r: int) -> bool:
    if os.getenv("DION_DEBUG_ORTHO_STAGES", "0") != "1":
        return False
    try:
        min_r = int(os.getenv("DION_DEBUG_LARGE_R_MIN_R", "512"))
    except ValueError as exc:
        raise RuntimeError(
            f"[DION_INVALID_ENV] DION_DEBUG_LARGE_R_MIN_R={os.getenv('DION_DEBUG_LARGE_R_MIN_R')!r} is not an int"
        ) from exc
    return r >= min_r


def _ortho_stage_debug_enabled() -> bool:
    return os.getenv("DION_DEBUG_ORTHO_STAGES", "0") == "1"


def _tensor_stage_stats(X: Tensor) -> tuple[bool, float, float]:
    X_fp32 = X.to(torch.float32)
    if not torch.isfinite(X_fp32).all():
        return False, float("inf"), float("inf")
    max_abs = float(X_fp32.abs().max().item())
    fro_norm = float(torch.linalg.vector_norm(X_fp32).item())
    return True, max_abs, fro_norm


def _p_orthogonality_stats(P: Tensor) -> tuple[bool, float, float]:
    with _dion_ortho_precision_context():
        P_fp32 = P.to(torch.float32)
        gram = P_fp32.mT @ P_fp32
    if not torch.isfinite(gram).all():
        return False, float("inf"), float("inf")
    r = gram.size(-1)
    diff = gram - torch.eye(r, device=gram.device, dtype=gram.dtype)
    max_err = float(diff.abs().max().item())
    fro_norm = float(torch.linalg.matrix_norm(diff, ord="fro").item())
    return True, max_err, fro_norm


def _matrix_condition_stats(X: Tensor) -> tuple[bool, float, float, float]:
    X_fp32 = X.to(torch.float32)
    if not torch.isfinite(X_fp32).all():
        return False, float("inf"), float("inf"), float("inf")
    sv = torch.linalg.svdvals(X_fp32)
    if not torch.isfinite(sv).all():
        return False, float("inf"), float("inf"), float("inf")
    sigma_max = float(sv.max().item())
    sigma_min = float(sv.min().item())
    cond = float("inf") if sigma_min == 0.0 else sigma_max / sigma_min
    return True, sigma_min, sigma_max, cond


def _log_ortho_stage(
    *,
    stage: str,
    tensor: Tensor,
    r: int,
    check_orthogonality: bool = False,
    check_conditioning: bool = False,
    extra: Optional[dict] = None,
) -> None:
    if not _should_log_ortho_stage_probe(r):
        return
    finite, max_abs, fro_norm = _tensor_stage_stats(tensor)
    if check_orthogonality:
        ortho_finite, ortho_max_err, ortho_fro_norm = _p_orthogonality_stats(tensor)
    else:
        ortho_finite, ortho_max_err, ortho_fro_norm = True, 0.0, 0.0
    if check_conditioning:
        cond_finite, sigma_min, sigma_max, cond = _matrix_condition_stats(tensor)
    else:
        cond_finite, sigma_min, sigma_max, cond = True, 0.0, 0.0, 0.0
    logger.info(
        "[DION_DEBUG_ORTHO_STAGE] stage=%s shape=%s dtype=%s finite=%s max_abs=%.6e fro_norm=%.6e "
        "ortho_finite=%s ortho_max_err=%.6e ortho_fro_norm=%.6e "
        "cond_finite=%s sigma_min=%.6e sigma_max=%.6e cond=%.6e extra=%s",
        stage,
        tuple(tensor.shape),
        str(tensor.dtype),
        finite,
        max_abs,
        fro_norm,
        ortho_finite,
        ortho_max_err,
        ortho_fro_norm,
        cond_finite,
        sigma_min,
        sigma_max,
        cond,
        extra or {},
    )
    if not finite:
        raise RuntimeError(
            "[DION_DEBUG_ORTHO_STAGE_NONFINITE] "
            f"stage={stage} shape={tuple(tensor.shape)} dtype={tensor.dtype} extra={extra or {}}"
        )
    if check_orthogonality and not ortho_finite:
        raise RuntimeError(
            "[DION_DEBUG_ORTHO_STAGE_NONFINITE_ORTHO] "
            f"stage={stage} shape={tuple(tensor.shape)} dtype={tensor.dtype} extra={extra or {}}"
        )


def _maybe_dump_sketch_tensor(
    *,
    tag: str,
    S: Tensor,
    logical_seed_keys: Optional[Sequence[object]] = None,
    batch_meta_ids: Optional[Sequence[str]] = None,
) -> None:
    if os.getenv("DION_DEBUG_DUMP_SKETCH", "0") != "1":
        return
    dump_dir_env = os.getenv("DION_DEBUG_DUMP_SKETCH_DIR", "").strip()
    if not dump_dir_env:
        raise RuntimeError(
            "[DION_INVALID_ENV] DION_DEBUG_DUMP_SKETCH=1 requires DION_DEBUG_DUMP_SKETCH_DIR"
        )
    dump_dir = Path(dump_dir_env)
    dump_dir.mkdir(parents=True, exist_ok=True)
    rank = dist.get_rank() if dist.is_initialized() else 0
    global _SKETCH_DUMP_COUNTER
    _SKETCH_DUMP_COUNTER += 1
    payload = {
        "rank": rank,
        "counter": _SKETCH_DUMP_COUNTER,
        "tag": tag,
        "is_dtensor": isinstance(S, DTensor),
        "shape": tuple(S.shape),
        "dtype": str(S.dtype),
        "logical_seed_keys": [repr(key) for key in logical_seed_keys] if logical_seed_keys is not None else None,
        "batch_meta_ids": list(batch_meta_ids) if batch_meta_ids is not None else None,
    }
    if isinstance(S, DTensor):
        payload["placements"] = tuple(str(p) for p in S.placements)
        payload["mesh"] = S.device_mesh.mesh.detach().cpu()
        payload["local"] = S.to_local().detach().cpu()
    else:
        payload["local"] = S.detach().cpu()
    dump_path = dump_dir / f"sketch_rank{rank:03d}_{_SKETCH_DUMP_COUNTER:04d}_{tag}.pt"
    logger.info(
        "[DION_DEBUG_DUMP_SKETCH] rank=%d counter=%d tag=%s path=%s is_dtensor=%s shape=%s",
        rank,
        _SKETCH_DUMP_COUNTER,
        tag,
        str(dump_path),
        isinstance(S, DTensor),
        tuple(S.shape),
    )
    torch.save(payload, dump_path)


def _maybe_dump_ortho_tensor(*, tag: str, X: Tensor, batch_meta_ids: Optional[Sequence[str]] = None) -> None:
    if os.getenv("DION_DEBUG_DUMP_ORTHO_TENSOR", "0") != "1":
        return
    dump_dir_env = os.getenv("DION_DEBUG_DUMP_ORTHO_TENSOR_DIR", "").strip()
    if not dump_dir_env:
        raise RuntimeError(
            "[DION_INVALID_ENV] DION_DEBUG_DUMP_ORTHO_TENSOR=1 requires DION_DEBUG_DUMP_ORTHO_TENSOR_DIR"
        )
    dump_dir = Path(dump_dir_env)
    dump_dir.mkdir(parents=True, exist_ok=True)
    rank = dist.get_rank() if dist.is_initialized() else 0
    global _ORTHO_TENSOR_DUMP_COUNTER
    _ORTHO_TENSOR_DUMP_COUNTER += 1
    payload = {
        "rank": rank,
        "counter": _ORTHO_TENSOR_DUMP_COUNTER,
        "tag": tag,
        "is_dtensor": isinstance(X, DTensor),
        "shape": tuple(X.shape),
        "dtype": str(X.dtype),
        "batch_meta_ids": list(batch_meta_ids) if batch_meta_ids is not None else None,
    }
    if isinstance(X, DTensor):
        payload["placements"] = tuple(str(p) for p in X.placements)
        payload["mesh"] = X.device_mesh.mesh.detach().cpu()
        payload["local"] = X.to_local().detach().cpu()
    else:
        payload["local"] = X.detach().cpu()
    dump_path = dump_dir / f"ortho_rank{rank:03d}_{_ORTHO_TENSOR_DUMP_COUNTER:04d}_{tag}.pt"
    logger.info(
        "[DION_DEBUG_DUMP_ORTHO_TENSOR] rank=%d counter=%d tag=%s path=%s is_dtensor=%s shape=%s",
        rank,
        _ORTHO_TENSOR_DUMP_COUNTER,
        tag,
        str(dump_path),
        isinstance(X, DTensor),
        tuple(X.shape),
    )
    torch.save(payload, dump_path)


def _maybe_log_distributed_sketch_trace(
    *,
    stage: str,
    S: Optional[Tensor],
    shard_mesh_dim: Optional[int],
) -> None:
    if os.getenv("DION_DEBUG_DISTRIBUTED_SKETCH_TRACE", "0") != "1":
        return
    global _DISTRIBUTED_SKETCH_TRACE_COUNTER
    _DISTRIBUTED_SKETCH_TRACE_COUNTER += 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    if S is None:
        local = None
        placements = ()
    elif isinstance(S, DTensor):
        local = S.to_local()
        placements = tuple(str(p) for p in S.placements)
    else:
        local = S
        placements = ()
    if local is not None:
        local_cpu = local.detach().to(torch.float32).cpu().contiguous()
        sketch_hash = hashlib.blake2b(local_cpu.numpy().tobytes(), digest_size=16).hexdigest()
        local_shape = tuple(local.shape)
        local_norm = float(torch.linalg.vector_norm(local_cpu).item())
        local_sum = float(local_cpu.sum().item())
        device = local.device
    else:
        local_shape = ()
        local_norm = float("nan")
        local_sum = float("nan")
        sketch_hash = ""
        device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        rng_state = torch.cuda.get_rng_state(device)
    else:
        rng_state = torch.get_rng_state()
    rng_hash = hashlib.blake2b(rng_state.detach().cpu().numpy().tobytes(), digest_size=16).hexdigest()
    logger.info(
        "[DION_DEBUG_DISTRIBUTED_SKETCH_TRACE] rank=%d counter=%d stage=%s shard_mesh_dim=%s placements=%s local_shape=%s local_norm=%.6e local_sum=%.6e sketch_hash=%s rng_hash=%s",
        rank,
        _DISTRIBUTED_SKETCH_TRACE_COUNTER,
        stage,
        shard_mesh_dim,
        placements,
        local_shape,
        local_norm,
        local_sum,
        sketch_hash,
        rng_hash,
    )


def _maybe_log_distributed_ortho_trace(
    *,
    stage: str,
    X: Tensor,
    shard_mesh_dim: Optional[int],
) -> None:
    if os.getenv("DION_DEBUG_DISTRIBUTED_ORTHO_TRACE", "0") != "1":
        return
    global _DISTRIBUTED_ORTHO_TRACE_COUNTER
    _DISTRIBUTED_ORTHO_TRACE_COUNTER += 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    if isinstance(X, DTensor):
        local = X.to_local()
        placements = tuple(str(p) for p in X.placements)
    else:
        local = X
        placements = ()
    local_cpu = local.detach().to(torch.float32).cpu().contiguous()
    tensor_hash = hashlib.blake2b(local_cpu.numpy().tobytes(), digest_size=16).hexdigest()
    logger.info(
        "[DION_DEBUG_DISTRIBUTED_ORTHO_TRACE] rank=%d counter=%d stage=%s shard_mesh_dim=%s placements=%s local_shape=%s local_norm=%.6e local_sum=%.6e tensor_hash=%s",
        rank,
        _DISTRIBUTED_ORTHO_TRACE_COUNTER,
        stage,
        shard_mesh_dim,
        placements,
        tuple(local.shape),
        float(torch.linalg.vector_norm(local_cpu).item()),
        float(local_cpu.sum().item()),
        tensor_hash,
    )


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
            if _should_log_large_r_probe(P, r):
                logger.info(
                    "[DION_DEBUG_LARGE_R_ORTHO] stage=wide_qr shape=%s dtype=%s r=%d",
                    tuple(P.shape),
                    str(P.dtype),
                    r,
                )
            Q, _ = torch.linalg.qr(P.to(dtype=torch.float32))
            return Q.to(original_dtype).contiguous()

        # Case 2: Tall matrix - use Randomized Cholesky QR
        else:
            # Step 1: Generate random sketch matrix
            if sketch_fn is not None:
                S = sketch_fn(P, rcqr_oversample)
            else:
                S = _default_sketch_matrix(P, rcqr_oversample)
            _maybe_dump_sketch_tensor(
                tag=getattr(sketch_fn, "_dion_sketch_tag", "local_orthogonalize"),
                S=S,
                logical_seed_keys=getattr(sketch_fn, "_dion_logical_seed_keys", None),
                batch_meta_ids=getattr(sketch_fn, "_dion_batch_meta_ids", None),
            )
            if _should_log_large_r_probe(P, r):
                logger.info(
                    "[DION_DEBUG_LARGE_R_ORTHO] stage=rcqr_start shape=%s dtype=%s r=%d k=%d sketch_dtype=%s",
                    tuple(P.shape),
                    str(P.dtype),
                    r,
                    int(S.size(-2)),
                    str(S.dtype),
                )

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
            if _should_log_large_r_probe(P, r) and (info > 0).any():
                logger.warning(
                    "[DION_DEBUG_LARGE_R_ORTHO] stage=cholesky_info shape=%s dtype=%s r=%d info=%s",
                    tuple(P.shape),
                    str(P.dtype),
                    r,
                    info.detach().cpu().tolist(),
                )

            P = torch.linalg.solve_triangular(
                R, P, upper=True, left=False
            )
            if _should_log_large_r_probe(P, r):
                logger.info(
                    "[DION_DEBUG_LARGE_R_ORTHO] stage=rcqr_done shape=%s dtype=%s r=%d",
                    tuple(P.shape),
                    str(original_dtype),
                    r,
                )

            return P.to(original_dtype).contiguous()


def _dtensor_from_local(local_tensor: Tensor, ref: DTensor) -> DTensor:
    """Create a DTensor from a local shard using an existing DTensor as metadata reference."""
    return DTensor.from_local(
        local_tensor,
        device_mesh=ref.device_mesh,
        placements=ref.placements,
    )


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
                # batch slots; the logical global sketch is therefore reconstructed by
                # generating the keyed local shard for those slots only and wrapping it with
                # the original batch-sharded placements.
                fork_devices = [torch.cuda.current_device()] if P.device.type == "cuda" else []
                local_slots = []
                local_m = int(P.to_local().size(-2))
                for seed_key in logical_seed_keys:
                    seed = int.from_bytes(
                        hashlib.blake2b(repr(seed_key).encode("utf-8"), digest_size=8).digest(),
                        "little",
                    ) & ((1 << 63) - 1)
                    with torch.random.fork_rng(devices=fork_devices):
                        torch.manual_seed(seed)
                        slot_local = torch.empty((k, local_m), device=P.device, dtype=P.dtype)
                        slot_local.normal_(mean=0.0, std=std)
                    local_slots.append(slot_local)

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

            fork_devices = [torch.cuda.current_device()] if P.device.type == "cuda" else []
            local_slots = []
            for seed_key in logical_seed_keys:
                seed = int.from_bytes(
                    hashlib.blake2b(repr(seed_key).encode("utf-8"), digest_size=8).digest(),
                    "little",
                ) & ((1 << 63) - 1)
                with torch.random.fork_rng(devices=fork_devices):
                    torch.manual_seed(seed)
                    slot_dtensor = dtensor_randn(
                        (k, m),
                        device_mesh=P.device_mesh,
                        dtype=P.dtype,
                        placements=tuple(slot_placements),
                    )
                local_slots.append(slot_dtensor.to_local())

            local_S = torch.stack(local_slots, dim=0) * std
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
        _maybe_dump_sketch_tensor(
            tag="exact",
            S=S,
            logical_seed_keys=logical_seed_keys,
            batch_meta_ids=batch_meta_ids,
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
    _log_ortho_stage(
        stage="exact_input",
        tensor=P_local,
        r=P.size(-1),
        check_orthogonality=False,
        extra={"is_dtensor": isinstance(P, DTensor)},
    )

    if P.size(-2) <= P.size(-1):
        P_local, _ = torch.linalg.qr(P_local.to(dtype=torch.float32))
        _log_ortho_stage(
            stage="exact_wide_qr_output",
            tensor=P_local,
            r=P.size(-1),
            check_orthogonality=True,
        )
    else:
        S = generate_random_sketch_matrix(
            P,
            oversample,
            sketch_fn=sketch_fn,
            logical_seed_keys=logical_seed_keys,
        )
        _maybe_dump_sketch_tensor(
            tag="exact",
            S=S,
            logical_seed_keys=logical_seed_keys,
            batch_meta_ids=batch_meta_ids,
        )
        S_local = S.to_local() if isinstance(S, DTensor) else S
        _log_ortho_stage(
            stage="exact_sketch",
            tensor=S_local,
            r=P.size(-1),
            check_orthogonality=False,
            extra={"oversample": oversample},
        )

        SP = S_local @ P_local
        _log_ortho_stage(
            stage="exact_sp",
            tensor=SP,
            r=P.size(-1),
            check_orthogonality=False,
        )
        _, R = torch.linalg.qr(SP.to(dtype=torch.float32), mode="r")
        _log_ortho_stage(
            stage="exact_qr_r",
            tensor=R,
            r=P.size(-1),
            check_orthogonality=False,
            check_conditioning=True,
        )
        P_local = torch.linalg.solve_triangular(
            R, P_local.to(dtype=torch.float32), upper=True, left=False
        )
        _log_ortho_stage(
            stage="exact_after_qr_solve",
            tensor=P_local,
            r=P.size(-1),
            check_orthogonality=True,
            check_conditioning=True,
        )

        PP = P_local.mT @ P_local
        _log_ortho_stage(
            stage="exact_pp",
            tensor=PP,
            r=P.size(-1),
            check_orthogonality=False,
            check_conditioning=True,
        )
        R, info = torch.linalg.cholesky_ex(PP, upper=True)
        _log_ortho_stage(
            stage="exact_cholesky_r",
            tensor=R,
            r=P.size(-1),
            check_orthogonality=False,
            check_conditioning=True,
            extra={"cholesky_info": info.detach().cpu().tolist()},
        )
        P_local = torch.linalg.solve_triangular(R, P_local, upper=True, left=False)
        _log_ortho_stage(
            stage="exact_after_cholesky_solve",
            tensor=P_local,
            r=P.size(-1),
            check_orthogonality=True,
            check_conditioning=True,
        )

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
        _maybe_dump_sketch_tensor(
            tag="distributed_exact",
            S=S,
            logical_seed_keys=logical_seed_keys,
            batch_meta_ids=batch_meta_ids,
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
        _maybe_dump_ortho_tensor(tag="p_input", X=P, batch_meta_ids=batch_meta_ids)
        _maybe_log_distributed_sketch_trace(
            stage="pre_generate",
            S=None,
            shard_mesh_dim=shard_mesh_dim,
        )
        S = generate_random_sketch_matrix(
            P,
            oversample,
            shard_mesh_dim=shard_mesh_dim,
            sketch_fn=sketch_fn,
            logical_seed_keys=logical_seed_keys,
        )
        _maybe_log_distributed_sketch_trace(
            stage="post_generate",
            S=S,
            shard_mesh_dim=shard_mesh_dim,
        )
        _maybe_dump_sketch_tensor(
            tag="distributed_trace",
            S=S,
            logical_seed_keys=logical_seed_keys,
            batch_meta_ids=batch_meta_ids,
        )
        SP: DTensor = S @ P
        _maybe_dump_ortho_tensor(tag="sp_pre_redistribute", X=SP, batch_meta_ids=batch_meta_ids)

        P_single = SP.redistribute(placements=batch_sharded_placements)
        _maybe_dump_ortho_tensor(tag="sp_single", X=P_single, batch_meta_ids=batch_meta_ids)
        _maybe_log_distributed_ortho_trace(
            stage="sp_single",
            X=P_single,
            shard_mesh_dim=shard_mesh_dim,
        )
        _, R_local = torch.linalg.qr(P_single.to_local().to(dtype=torch.float32), mode="r")
        _maybe_log_distributed_ortho_trace(
            stage="r_local_qr",
            X=R_local,
            shard_mesh_dim=shard_mesh_dim,
        )
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
