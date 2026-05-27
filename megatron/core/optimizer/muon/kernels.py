"""Muon kernel helpers that do not own Megatron runtime state."""

from __future__ import annotations

import itertools
import math
import os
from contextlib import contextmanager
from functools import lru_cache
from types import SimpleNamespace
from typing import Iterable, Iterator, Optional, Sequence

import torch
import torch.distributed as dist
from torch import Tensor


_COEFFICIENT_SETS: dict[str, tuple[tuple[float, float, float], ...]] = {
    "simple": ((3.4445, -4.7750, 2.0315),),
    "quintic": (
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ),
    "polar_express": (
        (8.2051, -22.9019, 16.4607),
        (4.0664, -2.8612, 0.5184),
        (3.9096, -2.8234, 0.5250),
        (3.2856, -2.4153, 0.4853),
        (2.2779, -1.6198, 0.3985),
        (1.8726, -1.2307, 0.3585),
        (1.8564, -1.2132, 0.3568),
        (1.8750, -1.2500, 0.3750),
    ),
    "aol": (
        (4.0098, -7.0585, 2.4635),
        (3.4585, -5.5479, 2.5959),
        (2.7573, -3.2939, 1.4254),
        (2.7215, -3.0494, 1.3169),
    ),
}

_DAO_GRAM_TILE_SIZE = 256
_DAO_GRAM_BACKEND = None
_DAO_GRAM_IMPORT_ERROR = None
_DAO_GRAM_LOGGED = False
_DAO_SYM_MM = None
_DAO_SYM_BADDMM = None
_DAO_MM = None
_DAO_MM_ADD = None

_GRAM_PROFILE_KEYS = (
    "dao_groups",
    "torch_groups",
    "dao_matrices",
    "torch_matrices",
    "fallback_policy_torch_groups",
    "fallback_device_groups",
    "fallback_dtype_groups",
    "fallback_tile_groups",
    "gram_all_reduce_calls",
    "gram_all_reduce_numel",
)
_GRAM_PROFILE = {key: 0 for key in _GRAM_PROFILE_KEYS}


def _gram_profile_enabled() -> bool:
    return os.getenv("MEGATRON_MUON_PROFILE_NS", "0") == "1"


def _matrix_count(x: Tensor) -> int:
    if x.ndim <= 2:
        return 1
    matrix_numel = max(1, int(x.size(-2)) * int(x.size(-1)))
    return int(x.numel() // matrix_numel)


def _dao_gram_fallback_reason(x: Tensor) -> Optional[str]:
    if not x.is_cuda:
        return "device"
    if x.dtype not in (torch.float16, torch.bfloat16):
        return "dtype"
    if min(int(x.size(-2)), int(x.size(-1))) <= _DAO_GRAM_TILE_SIZE:
        return "tile"
    return None


def _record_gram_backend(name: str, x: Tensor, reason: Optional[str] = None) -> None:
    if not _gram_profile_enabled():
        return
    matrices = _matrix_count(x)
    if name == "dao":
        _GRAM_PROFILE["dao_groups"] += 1
        _GRAM_PROFILE["dao_matrices"] += matrices
        return
    _GRAM_PROFILE["torch_groups"] += 1
    _GRAM_PROFILE["torch_matrices"] += matrices
    if reason == "policy_torch":
        _GRAM_PROFILE["fallback_policy_torch_groups"] += 1
    elif reason == "device":
        _GRAM_PROFILE["fallback_device_groups"] += 1
    elif reason == "dtype":
        _GRAM_PROFILE["fallback_dtype_groups"] += 1
    elif reason == "tile":
        _GRAM_PROFILE["fallback_tile_groups"] += 1


def _record_gram_all_reduce(gram: Tensor) -> None:
    if not _gram_profile_enabled():
        return
    _GRAM_PROFILE["gram_all_reduce_calls"] += 1
    _GRAM_PROFILE["gram_all_reduce_numel"] += int(gram.numel())


def get_and_reset_gram_profile() -> dict[str, int]:
    stats = dict(_GRAM_PROFILE)
    for key in _GRAM_PROFILE_KEYS:
        _GRAM_PROFILE[key] = 0
    return stats


def _torch_baddmm(a: Tensor, b: Tensor, *, C: Tensor, alpha: float = 1.0, beta: float = 1.0) -> Tensor:
    if a.ndim == 2:
        return torch.addmm(C, a, b, alpha=alpha, beta=beta)
    if a.ndim == 3:
        return torch.baddbmm(C, a, b, alpha=alpha, beta=beta)
    return (a @ b).mul(float(alpha)).add(C, alpha=float(beta))


def _torch_mm_add(a: Tensor, b: Tensor, *, C: Tensor, beta: float = 1.0) -> Tensor:
    return _torch_baddmm(a, b, C=C, beta=beta)


_TORCH_GRAM_BACKEND = SimpleNamespace(
    name="torch",
    sym_mm=lambda a, b: a @ b,
    sym_baddmm=_torch_baddmm,
    mm=lambda a, b: a @ b,
    mm_add=_torch_mm_add,
)


@contextmanager
def _matmul_precision(precision: str):
    if not hasattr(torch, "get_float32_matmul_precision"):
        yield
        return
    old_precision = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision(precision)
        yield
    finally:
        torch.set_float32_matmul_precision(old_precision)


def _str_to_dtype(dtype_val) -> Optional[torch.dtype]:
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


def _dist_world_size(group) -> int:
    if group is None or not dist.is_available() or not dist.is_initialized():
        return 1
    return int(dist.get_world_size(group))


def _make_dao_gram_backend():
    global _DAO_GRAM_BACKEND, _DAO_GRAM_IMPORT_ERROR
    if _DAO_GRAM_BACKEND is not None:
        return _DAO_GRAM_BACKEND
    if _DAO_GRAM_IMPORT_ERROR is not None:
        raise _DAO_GRAM_IMPORT_ERROR
    sym_mm, sym_baddmm, mm, mm_add = _load_dao_gemm_ops()
    _DAO_GRAM_BACKEND = SimpleNamespace(
        name="dao",
        sym_mm=sym_mm,
        sym_baddmm=sym_baddmm,
        mm=mm,
        mm_add=mm_add,
    )
    return _DAO_GRAM_BACKEND


def _load_dao_gemm_ops():
    global _DAO_GRAM_IMPORT_ERROR, _DAO_SYM_MM, _DAO_SYM_BADDMM, _DAO_MM, _DAO_MM_ADD
    if _DAO_SYM_MM is not None:
        return _DAO_SYM_MM, _DAO_SYM_BADDMM, _DAO_MM, _DAO_MM_ADD
    if _DAO_GRAM_IMPORT_ERROR is not None:
        raise _DAO_GRAM_IMPORT_ERROR
    try:
        from .gram_newton_schulz.gram_newton_schulz import _make_kernel_backend

        backend = _make_kernel_backend()
    except Exception as exc:  # pragma: no cover - depends on optional package.
        _DAO_GRAM_IMPORT_ERROR = exc
        raise
    _DAO_SYM_MM = backend.sym_mm
    _DAO_SYM_BADDMM = lambda a, b, *, C, alpha=1.0, beta=1.0: backend.sym_baddbmm(
        a,
        b,
        C=C,
        alpha=alpha,
        beta=beta,
    )
    _DAO_MM = backend.mm
    _DAO_MM_ADD = lambda a, b, *, C, beta=1.0: backend.mm_add(a, b, C=C, beta=beta)
    return _DAO_SYM_MM, _DAO_SYM_BADDMM, _DAO_MM, _DAO_MM_ADD


def _normalize_gram_kernel_policy(policy: Optional[str]) -> str:
    policy = "torch" if policy is None else str(policy).lower()
    if policy == "eager":
        return "torch"
    if policy == "disabled":
        return "torch"
    if policy == "quack":
        return "dao"
    if policy in ("torch", "auto", "dao", "compile"):
        return policy
    raise RuntimeError(f"[MUON_INVALID_GRAM_KERNEL_POLICY] policy={policy!r}")


def _dao_gram_eligible(x: Tensor) -> bool:
    return _dao_gram_fallback_reason(x) is None


def _select_gram_backend(x: Tensor, *, policy: Optional[str]):
    global _DAO_GRAM_LOGGED
    policy = _normalize_gram_kernel_policy(policy)
    if policy == "torch":
        _record_gram_backend("torch", x, "policy_torch")
        return _TORCH_GRAM_BACKEND
    fallback_reason = _dao_gram_fallback_reason(x)
    if fallback_reason is not None:
        _record_gram_backend("torch", x, fallback_reason)
        return _TORCH_GRAM_BACKEND
    try:
        backend = _make_dao_gram_backend()
    except Exception as exc:
        if policy == "auto":
            return _TORCH_GRAM_BACKEND
        raise RuntimeError(
            "[MUON_DAO_GRAM_BACKEND_UNAVAILABLE] "
            "Install optional Gram NS dependencies with "
            "`source scripts/setup_muon_gram_deps.sh`, or use "
            "--muon-gram-ns-kernel-policy=auto/torch."
        ) from exc
    if os.getenv("MEGATRON_MUON_GRAM_LOG", "0") == "1" and not _DAO_GRAM_LOGGED:
        print(
            "[Muon] Gram Newton-Schulz using Dao/Quack kernels "
            f"for shape={tuple(int(dim) for dim in x.shape[-2:])} dtype={x.dtype}",
            flush=True,
        )
        _DAO_GRAM_LOGGED = True
    _record_gram_backend("dao", x)
    return backend


def _dao_sym_mm(a: Tensor, b: Tensor) -> Tensor:
    sym_mm, _, _, _ = _load_dao_gemm_ops()
    return sym_mm(a, b)


def _dao_sym_baddmm(
    a: Tensor,
    b: Tensor,
    *,
    C: Tensor,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> Tensor:
    _, sym_baddmm, _, _ = _load_dao_gemm_ops()
    return sym_baddmm(a, b, C=C, alpha=alpha, beta=beta)


def _dao_mm(a: Tensor, b: Tensor) -> Tensor:
    _, _, mm, _ = _load_dao_gemm_ops()
    return mm(a, b)


def _dao_mm_add(a: Tensor, b: Tensor, *, C: Tensor, beta: float = 1.0) -> Tensor:
    _, _, _, mm_add = _load_dao_gemm_ops()
    return mm_add(a, b, C=C, beta=beta)


def _gram_ns_core(
    x_work: Tensor,
    *,
    coeffs: tuple[tuple[float, float, float], ...],
    restarts: tuple[int, ...],
    sym_mm,
    sym_baddmm,
    mm,
) -> Tensor:
    r = sym_mm(x_work, x_work.mT)
    eye = torch.eye(r.size(-1), device=x_work.device, dtype=x_work.dtype)
    eye = eye.expand(*r.shape[:-2], r.size(-1), r.size(-1)).contiguous()
    q: Optional[Tensor] = None
    for step, (a, b, c) in enumerate(coeffs):
        if step in restarts and step != 0:
            x_work = mm(q, x_work)
            r = sym_mm(x_work, x_work.mT)
            q = None

        z = sym_baddmm(r, r, C=r, alpha=c, beta=b)
        q = z + a * eye if q is None else sym_baddmm(q, z, C=q, beta=a)

        if step < len(coeffs) - 1 and (step + 1) not in restarts:
            rz = sym_baddmm(r, z, C=r, beta=a)
            r = sym_baddmm(z, rz, C=rz, beta=a)
    return mm(q, x_work)


def _standard_ns_core(
    x_work: Tensor,
    *,
    coeffs: tuple[tuple[float, float, float], ...],
    sym_mm,
    sym_baddmm,
    mm_add,
) -> Tensor:
    for a, b, c in coeffs:
        gram = sym_mm(x_work, x_work.mT)
        poly = sym_baddmm(gram, gram, C=gram, alpha=c, beta=b)
        x_work = mm_add(poly, x_work, C=x_work, beta=a)
    return x_work


@lru_cache(maxsize=None)
def _compiled_local_gram_runner(
    *,
    backend_name: str,
    coeffs: tuple[tuple[float, float, float], ...],
    restarts: tuple[int, ...],
    gram_dtype: Optional[torch.dtype],
):
    if backend_name == "dao":
        _load_dao_gemm_ops()
        sym_mm = _dao_sym_mm
        sym_baddmm = _dao_sym_baddmm
        mm = _dao_mm
        mm_add = _dao_mm_add
    elif backend_name == "torch":
        sym_mm = lambda a, b: a @ b
        sym_baddmm = _torch_baddmm
        mm = lambda a, b: a @ b
        mm_add = _torch_mm_add
    else:
        raise RuntimeError(f"[MUON_INVALID_COMPILED_GRAM_BACKEND] backend={backend_name!r}")

    def fn(x: Tensor) -> Tensor:
        original_shape = x.shape
        original_dtype = x.dtype
        squeezed = False
        if x.ndim == 2:
            x_work = x.unsqueeze(0)
            squeezed = True
        elif x.ndim > 3:
            x_work = x.view(-1, x.size(-2), x.size(-1))
        else:
            x_work = x

        transposed = x_work.size(-2) > x_work.size(-1)
        if transposed:
            x_work = x_work.mT
        x_work = x_work.to(dtype=torch.float32)
        x_work = x_work / x_work.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-7)
        if gram_dtype is not None:
            x_work = x_work.to(dtype=gram_dtype)

        if x_work.size(-2) == x_work.size(-1):
            x_work = _standard_ns_core(
                x_work,
                coeffs=coeffs,
                sym_mm=sym_mm,
                sym_baddmm=sym_baddmm,
                mm_add=mm_add,
            )
        else:
            x_work = _gram_ns_core(
                x_work,
                coeffs=coeffs,
                restarts=restarts,
                sym_mm=sym_mm,
                sym_baddmm=sym_baddmm,
                mm=mm,
            )

        x_work = x_work.to(dtype=torch.float32 if original_dtype != torch.float64 else torch.float64)
        if transposed:
            x_work = x_work.mT
        if squeezed:
            x_work = x_work.squeeze(0)
        return x_work.to(dtype=original_dtype).view(original_shape)

    return torch.compile(fn, fullgraph=True, mode="reduce-overhead")


def _compiled_gram_newton_schulz(
    x: Tensor,
    *,
    steps: int,
    coefficient_type: str,
    restart_steps: tuple[int, ...],
    custom_coefficient_sets: Optional[Sequence[Sequence[float]]] = None,
    gram_dtype: Optional[torch.dtype | str] = None,
    fp32_matmul_prec: str = "medium",
    backend_name: str = "dao",
) -> Tensor:
    dtype = _str_to_dtype(gram_dtype)
    if dtype is None and x.is_cuda:
        dtype = torch.float16
    coeffs = tuple(
        _coefficients(
            steps,
            coefficient_type=coefficient_type,
            custom_coefficient_sets=custom_coefficient_sets,
        )
    )
    runner = _compiled_local_gram_runner(
        backend_name=backend_name,
        coeffs=coeffs,
        restarts=tuple(int(step) for step in restart_steps),
        gram_dtype=dtype,
    )
    _record_gram_backend(
        backend_name,
        x,
        None if backend_name == "dao" else "policy_torch",
    )
    with _matmul_precision(fp32_matmul_prec):
        return runner(x)


def _coefficients(
    steps: int,
    *,
    coefficient_type: str = "quintic",
    custom_coefficient_sets: Optional[Sequence[Sequence[float]]] = None,
) -> Iterator[tuple[float, float, float]]:
    steps = int(steps)
    if steps < 1:
        raise RuntimeError(f"[MUON_INVALID_NS_STEPS] steps={steps}")
    if coefficient_type == "custom":
        if custom_coefficient_sets is None:
            raise RuntimeError("[MUON_CUSTOM_COEFFICIENTS_MISSING]")
        coefficient_sets = tuple(
            tuple(float(value) for value in coeff) for coeff in custom_coefficient_sets
        )
    elif coefficient_type in _COEFFICIENT_SETS:
        coefficient_sets = _COEFFICIENT_SETS[coefficient_type]
    else:
        raise RuntimeError(
            f"[MUON_UNSUPPORTED_NS_COEFFICIENT] coefficient_type={coefficient_type!r}"
        )
    if not coefficient_sets:
        raise RuntimeError("[MUON_EMPTY_NS_COEFFICIENTS]")
    for coeff in coefficient_sets:
        if len(coeff) != 3:
            raise RuntimeError(f"[MUON_INVALID_NS_COEFFICIENT] coeff={coeff}")
    if coefficient_type == "polar_express":
        base: Iterable[tuple[float, float, float]] = itertools.chain(
            coefficient_sets, itertools.repeat(coefficient_sets[-1])
        )
    else:
        base = itertools.cycle(coefficient_sets)
    return itertools.islice(base, steps)


def _normalize_frobenius(x: Tensor, *, eps: float, group=None) -> Tensor:
    norm_sq = x.square().sum(dim=(-2, -1), keepdim=True)
    if _dist_world_size(group) > 1:
        dist.all_reduce(norm_sq, op=dist.ReduceOp.SUM, group=group)
    return x / norm_sq.sqrt().clamp_min(float(eps))


def _normalize_ns_input(x: Tensor, *, eps: float = 1e-7, group=None) -> tuple[Tensor, bool]:
    transposed = x.size(-2) > x.size(-1)
    x_work = x.mT if transposed else x
    dtype = torch.float64 if x_work.dtype == torch.float64 else torch.float32
    x_work = x_work.to(dtype=dtype)
    return _normalize_frobenius(x_work, eps=eps, group=group), transposed


def muon_scale_factor(m: int, n: int, mode: str = "spectral") -> float:
    """Return the reference Muon shape scale."""
    m = int(m)
    n = int(n)
    if m <= 0 or n <= 0:
        raise RuntimeError(f"[MUON_INVALID_SCALE_SHAPE] m={m} n={n}")
    if mode == "spectral":
        return math.sqrt(float(max(m, n)))
    if mode == "unit_rms_norm":
        return math.sqrt(float(m) / float(n))
    if mode == "shape_scaling":
        return math.sqrt(max(1.0, float(m) / float(n)))
    raise ValueError(f"[MUON_INVALID_SCALE_MODE] scale_mode={mode!r}")


def get_muon_scale_factor(m: int, n: int, mode: str = "spectral") -> float:
    """Compatibility alias for the reference scale helper name."""
    return muon_scale_factor(m, n, mode=mode)


def scaled_lr_for_shape(
    *,
    lr: float,
    m_global: int,
    n_global: int,
    scale_mode: str = "spectral",
    extra_scale_factor: float = 1.0,
) -> float:
    """Return ``lr`` multiplied by the reference Muon logical-shape scale."""
    return (
        float(lr)
        * muon_scale_factor(int(m_global), int(n_global), mode=scale_mode)
        * float(extra_scale_factor)
    )


def logical_shape_for_tp(
    tensor: Tensor,
    *,
    partition_dim: Optional[int],
    tp_group: Optional[dist.ProcessGroup],
) -> tuple[int, int]:
    """Return logical matrix shape after accounting for a TP shard."""
    m, n = int(tensor.size(-2)), int(tensor.size(-1))
    if partition_dim is None or partition_dim < 0 or tp_group is None:
        return m, n
    tp_size = _dist_world_size(tp_group)
    if int(partition_dim) == 0:
        m *= int(tp_size)
    elif int(partition_dim) == 1:
        n *= int(tp_size)
    else:
        raise RuntimeError(f"[MUON_INVALID_TP_PARTITION_DIM] partition_dim={partition_dim}")
    return m, n


def newton_schulz(
    x: Tensor,
    *,
    steps: int = 5,
    coefficient_type: str = "quintic",
    custom_coefficient_sets: Optional[Sequence[Sequence[float]]] = None,
    eps: float = 1e-7,
    fp32_matmul_prec: str = "medium",
    transpose: Optional[bool] = None,
    tp_group: Optional[dist.ProcessGroup] = None,
    gram_side: str = "left",
) -> Tensor:
    """Local Newton-Schulz orthogonalization used by Muon."""
    if x.ndim < 2:
        raise RuntimeError(f"[MUON_NS_REQUIRES_MATRIX] ndim={x.ndim}")
    if gram_side not in ("left", "right"):
        raise RuntimeError(f"[MUON_INVALID_GRAM_SIDE] gram_side={gram_side!r}")

    original_dtype = x.dtype
    if transpose is None:
        x_work, transposed = _normalize_ns_input(x, eps=eps, group=tp_group)
    else:
        x_oriented = x.mT if transpose else x
        dtype = torch.float64 if x_oriented.dtype == torch.float64 else torch.float32
        x_work = _normalize_frobenius(x_oriented.to(dtype=dtype), eps=eps, group=tp_group)
        transposed = bool(transpose)

    with _matmul_precision(fp32_matmul_prec):
        for a, b, c in _coefficients(
            steps,
            coefficient_type=coefficient_type,
            custom_coefficient_sets=custom_coefficient_sets,
        ):
            gram = x_work @ x_work.mT if gram_side == "left" else x_work.mT @ x_work
            if _dist_world_size(tp_group) > 1:
                dist.all_reduce(gram, op=dist.ReduceOp.SUM, group=tp_group)
            gram2 = gram @ gram
            poly = b * gram + c * gram2
            if gram_side == "left":
                x_work = a * x_work + poly @ x_work
            else:
                x_work = a * x_work + x_work @ poly
    if transposed:
        x_work = x_work.mT
    return x_work.to(dtype=original_dtype)


def _all_gather_matrix(
    x: Tensor,
    *,
    partition_dim: int,
    tp_group: dist.ProcessGroup,
) -> Tensor:
    world_size = dist.get_world_size(tp_group)
    shards = [torch.empty_like(x) for _ in range(world_size)]
    dist.all_gather(shards, x.contiguous(), group=tp_group)
    return torch.cat(shards, dim=int(partition_dim))


def _slice_tp_matrix(
    x: Tensor,
    *,
    partition_dim: int,
    tp_group: dist.ProcessGroup,
) -> Tensor:
    world_size = dist.get_world_size(tp_group)
    rank = dist.get_rank(tp_group)
    chunks = torch.chunk(x, world_size, dim=int(partition_dim))
    return chunks[rank].contiguous()


def newton_schulz_tp(
    x: Tensor,
    *,
    steps: int = 5,
    coefficient_type: str = "quintic",
    tp_group: Optional[dist.ProcessGroup] = None,
    partition_dim: Optional[int] = None,
    mode: Optional[str] = None,
    tp_mode: Optional[str] = None,
    eps: float = 1e-7,
    fp32_matmul_prec: str = "medium",
) -> Tensor:
    """TP-aware Newton-Schulz using explicit collectives."""
    mode = tp_mode if tp_mode is not None else ("blockwise" if mode is None else mode)
    if mode == "duplicated":
        mode = "duplicated_debug"
    if tp_group is None or partition_dim is None or int(partition_dim) < 0:
        return newton_schulz(
            x,
            steps=steps,
            coefficient_type=coefficient_type,
            eps=eps,
            fp32_matmul_prec=fp32_matmul_prec,
        )
    if _dist_world_size(tp_group) <= 1 or mode == "blockwise":
        return newton_schulz(
            x,
            steps=steps,
            coefficient_type=coefficient_type,
            eps=eps,
            fp32_matmul_prec=fp32_matmul_prec,
        )

    if mode == "duplicated_debug":
        full = _all_gather_matrix(x, partition_dim=int(partition_dim), tp_group=tp_group)
        full_update = newton_schulz(
            full,
            steps=steps,
            coefficient_type=coefficient_type,
            eps=eps,
            fp32_matmul_prec=fp32_matmul_prec,
        )
        return _slice_tp_matrix(full_update, partition_dim=int(partition_dim), tp_group=tp_group)

    if mode != "distributed":
        raise RuntimeError(f"[MUON_INVALID_TP_MODE] mode={mode!r}")

    transposed, oriented_partition_dim = _tp_oriented_partition_dim(x, int(partition_dim))
    return newton_schulz(
        x,
        steps=steps,
        coefficient_type=coefficient_type,
        eps=eps,
        fp32_matmul_prec=fp32_matmul_prec,
        transpose=transposed,
        tp_group=tp_group,
        gram_side="left" if oriented_partition_dim == 1 else "right",
    )


def _gram_eye(r: Tensor) -> Tensor:
    eye = torch.eye(r.size(-1), device=r.device, dtype=r.dtype)
    return eye.expand(*r.shape[:-2], r.size(-1), r.size(-1))


def _form_gram(x: Tensor) -> Tensor:
    return x @ x.mT


def _form_right_gram(x: Tensor) -> Tensor:
    return x.mT @ x


def _form_distributed_gram(
    x: Tensor,
    *,
    group=None,
    side: str = "left",
    ops=_TORCH_GRAM_BACKEND,
) -> Tensor:
    if side not in ("left", "right"):
        raise RuntimeError(f"[MUON_INVALID_GRAM_SIDE] gram_side={side!r}")
    gram = ops.sym_mm(x, x.mT) if side == "left" else ops.sym_mm(x.mT, x)
    if _dist_world_size(group) > 1:
        dist.all_reduce(gram, op=dist.ReduceOp.SUM, group=group)
        _record_gram_all_reduce(gram)
    return gram


def _tp_oriented_partition_dim(x: Tensor, partition_dim: int) -> tuple[bool, int]:
    transposed = x.size(-2) > x.size(-1)
    oriented_partition_dim = 1 - int(partition_dim) if transposed else int(partition_dim)
    return transposed, oriented_partition_dim


def _gram_newton_schulz_impl(
    x: Tensor,
    *,
    steps: int = 5,
    coefficient_type: str = "quintic",
    restart_steps: tuple[int, ...] = (2,),
    restart_iterations: Optional[tuple[int, ...]] = None,
    gram_dtype: Optional[torch.dtype | str] = None,
    eps: float = 1e-7,
    fp32_matmul_prec: str = "medium",
    tp_group: Optional[dist.ProcessGroup] = None,
    gram_side: str = "left",
    gram_kernel_policy: str = "torch",
) -> Tensor:
    """Gram Newton-Schulz backend with optional Dao/Quack kernels."""
    if x.ndim < 2:
        raise RuntimeError(f"[MUON_GRAM_NS_REQUIRES_MATRIX] ndim={x.ndim}")
    if gram_side not in ("left", "right"):
        raise RuntimeError(f"[MUON_INVALID_GRAM_SIDE] gram_side={gram_side!r}")
    if restart_iterations is not None:
        restart_steps = restart_iterations
    original_dtype = x.dtype
    x_work, transposed = _normalize_ns_input(x, eps=eps, group=tp_group)
    dtype = _str_to_dtype(gram_dtype)
    if dtype is None and x_work.is_cuda:
        dtype = torch.float16
    if dtype is not None:
        x_work = x_work.to(dtype=dtype)
    ops = _select_gram_backend(x_work, policy=gram_kernel_policy)

    coeffs = tuple(_coefficients(steps, coefficient_type=coefficient_type))
    if gram_side == "right":
        with _matmul_precision(fp32_matmul_prec):
            for a, b, c in coeffs:
                gram = _form_distributed_gram(
                    x_work,
                    group=tp_group,
                    side="right",
                    ops=ops,
                )
                poly = ops.sym_baddmm(gram, gram, C=gram, alpha=c, beta=b)
                x_work = ops.mm_add(x_work, poly, C=x_work, beta=a)
    elif x_work.size(-2) == x_work.size(-1):
        with _matmul_precision(fp32_matmul_prec):
            for a, b, c in coeffs:
                gram = _form_distributed_gram(x_work, group=tp_group, ops=ops)
                poly = ops.sym_baddmm(gram, gram, C=gram, alpha=c, beta=b)
                x_work = ops.mm_add(poly, x_work, C=x_work, beta=a)
    else:
        restarts = {int(step) for step in restart_steps}
        if any(step < 0 or step >= len(coeffs) for step in restarts):
            raise RuntimeError(f"[MUON_INVALID_GRAM_RESTART_STEPS] steps={tuple(restarts)}")
        with _matmul_precision(fp32_matmul_prec):
            r = _form_distributed_gram(x_work, group=tp_group, ops=ops)
            eye = _gram_eye(r)
            q: Optional[Tensor] = None
            for step, (a, b, c) in enumerate(coeffs):
                if step in restarts and step != 0:
                    if q is None:
                        raise RuntimeError("[MUON_GRAM_NS_MISSING_Q_AT_RESTART]")
                    x_work = ops.mm(q, x_work)
                    r = _form_distributed_gram(x_work, group=tp_group, ops=ops)
                    q = None

                z = ops.sym_baddmm(r, r, C=r, alpha=c, beta=b)
                q = z + a * eye if q is None else ops.sym_baddmm(q, z, C=q, beta=a)

                if step < len(coeffs) - 1 and (step + 1) not in restarts:
                    rz = ops.sym_baddmm(r, z, C=r, beta=a)
                    r = ops.sym_baddmm(z, rz, C=rz, beta=a)
            if q is None:
                raise RuntimeError("[MUON_GRAM_NS_MISSING_FINAL_Q]")
            x_work = ops.mm(q, x_work)

    x_work = x_work.to(dtype=torch.float32 if original_dtype != torch.float64 else torch.float64)
    if transposed:
        x_work = x_work.mT
    return x_work.to(dtype=original_dtype)


def gram_newton_schulz(
    x: Tensor,
    *,
    steps: int = 5,
    coefficient_type: str = "quintic",
    restart_steps: tuple[int, ...] = (2,),
    restart_iterations: Optional[tuple[int, ...]] = None,
    gram_dtype: Optional[torch.dtype | str] = None,
    eps: float = 1e-7,
    fp32_matmul_prec: str = "medium",
    tp_group: Optional[dist.ProcessGroup] = None,
    gram_side: str = "left",
    gram_kernel_policy: str = "torch",
) -> Tensor:
    if restart_iterations is not None:
        restart_steps = restart_iterations
    policy = _normalize_gram_kernel_policy(gram_kernel_policy)
    compile_requested = policy == "compile" or os.getenv("MEGATRON_MUON_COMPILE_GRAM_NS", "0") == "1"
    if compile_requested and tp_group is None and gram_side == "left":
        backend_name = "torch" if policy == "torch" else "dao"
        return _compiled_gram_newton_schulz(
            x,
            steps=steps,
            coefficient_type=coefficient_type,
            restart_steps=restart_steps,
            gram_dtype=gram_dtype,
            fp32_matmul_prec=fp32_matmul_prec,
            backend_name=backend_name,
        )
    if policy == "compile":
        policy = "dao"
    return _gram_newton_schulz_impl(
        x,
        steps=steps,
        coefficient_type=coefficient_type,
        restart_steps=restart_steps,
        gram_dtype=gram_dtype,
        eps=eps,
        fp32_matmul_prec=fp32_matmul_prec,
        tp_group=tp_group,
        gram_side=gram_side,
        gram_kernel_policy=policy,
    )


def gram_newton_schulz_tp(
    x: Tensor,
    *,
    steps: int = 5,
    coefficient_type: str = "quintic",
    restart_steps: tuple[int, ...] = (2,),
    restart_iterations: Optional[tuple[int, ...]] = None,
    gram_dtype: Optional[torch.dtype | str] = None,
    tp_group: Optional[dist.ProcessGroup] = None,
    partition_dim: Optional[int] = None,
    mode: str = "blockwise",
    eps: float = 1e-7,
    fp32_matmul_prec: str = "medium",
    gram_kernel_policy: str = "torch",
) -> Tensor:
    """TP-aware Gram Newton-Schulz using no full gather outside debug mode."""
    if mode == "duplicated":
        mode = "duplicated_debug"
    if tp_group is None or partition_dim is None or int(partition_dim) < 0:
        return gram_newton_schulz(
            x,
            steps=steps,
            coefficient_type=coefficient_type,
            restart_steps=restart_steps,
            restart_iterations=restart_iterations,
            gram_dtype=gram_dtype,
            eps=eps,
            fp32_matmul_prec=fp32_matmul_prec,
            gram_kernel_policy=gram_kernel_policy,
        )
    if _dist_world_size(tp_group) <= 1 or mode == "blockwise":
        return gram_newton_schulz(
            x,
            steps=steps,
            coefficient_type=coefficient_type,
            restart_steps=restart_steps,
            restart_iterations=restart_iterations,
            gram_dtype=gram_dtype,
            eps=eps,
            fp32_matmul_prec=fp32_matmul_prec,
            gram_kernel_policy=gram_kernel_policy,
        )
    if mode == "duplicated_debug":
        full = _all_gather_matrix(x, partition_dim=int(partition_dim), tp_group=tp_group)
        full_update = gram_newton_schulz(
            full,
            steps=steps,
            coefficient_type=coefficient_type,
            restart_steps=restart_steps,
            restart_iterations=restart_iterations,
            gram_dtype=gram_dtype,
            eps=eps,
            fp32_matmul_prec=fp32_matmul_prec,
            gram_kernel_policy=gram_kernel_policy,
        )
        return _slice_tp_matrix(full_update, partition_dim=int(partition_dim), tp_group=tp_group)
    if mode != "distributed":
        raise RuntimeError(f"[MUON_INVALID_TP_MODE] mode={mode!r}")

    _, oriented_partition_dim = _tp_oriented_partition_dim(x, int(partition_dim))
    return gram_newton_schulz(
        x,
        steps=steps,
        coefficient_type=coefficient_type,
        restart_steps=restart_steps,
        restart_iterations=restart_iterations,
        gram_dtype=gram_dtype,
        eps=eps,
        fp32_matmul_prec=fp32_matmul_prec,
        tp_group=tp_group,
        gram_side="left" if oriented_partition_dim == 1 else "right",
        gram_kernel_policy=gram_kernel_policy,
    )


def standard_newton_schulz(
    x: Tensor,
    *,
    steps: int = 5,
    coefficient_type: str = "quintic",
    eps: float = 1e-7,
    fp32_matmul_prec: str = "medium",
) -> Tensor:
    """Compatibility alias for the standard local Newton-Schulz backend."""
    return newton_schulz(
        x,
        steps=steps,
        coefficient_type=coefficient_type,
        eps=eps,
        fp32_matmul_prec=fp32_matmul_prec,
    )


def orthogonalize_muon(
    x: Tensor,
    *,
    ns_backend: str = "standard",
    steps: int = 5,
    coefficient_type: str = "quintic",
    tp_group: Optional[dist.ProcessGroup] = None,
    partition_dim: Optional[int] = None,
    tp_mode: str = "blockwise",
    gram_restart_steps: tuple[int, ...] = (2,),
    gram_dtype: Optional[torch.dtype | str] = None,
    gram_kernel_policy: str = "torch",
    eps: float = 1e-7,
    fp32_matmul_prec: str = "medium",
) -> Tensor:
    """Dispatch Muon orthogonalization backend."""
    if ns_backend == "standard":
        return newton_schulz_tp(
            x,
            steps=steps,
            coefficient_type=coefficient_type,
            tp_group=tp_group,
            partition_dim=partition_dim,
            mode=tp_mode,
            eps=eps,
            fp32_matmul_prec=fp32_matmul_prec,
        )
    if ns_backend == "gram":
        return gram_newton_schulz_tp(
            x,
            steps=steps,
            coefficient_type=coefficient_type,
            restart_steps=gram_restart_steps,
            gram_dtype=gram_dtype,
            tp_group=tp_group,
            partition_dim=partition_dim,
            mode=tp_mode,
            eps=eps,
            fp32_matmul_prec=fp32_matmul_prec,
            gram_kernel_policy=gram_kernel_policy,
        )
    raise RuntimeError(f"[MUON_INVALID_NS_BACKEND] ns_backend={ns_backend!r}")


def update_momentum(
    momentum: Tensor,
    grad: Tensor,
    *,
    beta: float,
) -> None:
    """Reference Muon EMA momentum update."""
    beta = float(beta)
    if beta < 0.0 or beta >= 1.0:
        raise RuntimeError(f"[MUON_INVALID_MOMENTUM_BETA] beta={beta}")
    if momentum.shape != grad.shape:
        raise RuntimeError(
            "[MUON_MOMENTUM_SHAPE_MISMATCH] "
            f"momentum_shape={tuple(momentum.shape)} grad_shape={tuple(grad.shape)}"
        )
    momentum.lerp_(grad.to(momentum.dtype), 1.0 - beta)


def nesterov_update(momentum: Tensor, grad: Tensor, *, beta: float) -> Tensor:
    """Reference Nesterov-style Muon update input."""
    return grad.to(momentum.dtype).lerp(momentum, float(beta))


def apply_muon_momentum(
    momentum: Tensor,
    grad: Tensor,
    *,
    beta: float,
    nesterov: bool = False,
) -> Tensor:
    """Apply reference EMA momentum and return the Muon update input."""
    update_momentum(momentum, grad, beta=beta)
    if nesterov:
        return nesterov_update(momentum, grad, beta=beta)
    return momentum


def orthogonalize_muon_update(
    update: Tensor,
    *,
    ns_backend: str = "standard",
    coefficient_type: str = "quintic",
    num_ns_steps: int = 5,
    eps: float = 1e-7,
    scale_mode: str = "spectral",
    extra_scale_factor: float = 1.0,
    global_shape: Optional[tuple[int, int]] = None,
    tp_group: Optional[dist.ProcessGroup] = None,
    partition_dim: Optional[int] = None,
    tp_mode: str = "blockwise",
    gram_restart_iterations: tuple[int, ...] = (2,),
    gram_dtype: Optional[torch.dtype | str] = None,
    gram_kernel_policy: str = "torch",
    fp32_matmul_prec: str = "medium",
) -> Tensor:
    """Orthogonalize and scale one Muon update tensor."""
    orth_update = orthogonalize_muon(
        update,
        ns_backend=ns_backend,
        steps=num_ns_steps,
        coefficient_type=coefficient_type,
        tp_group=tp_group,
        partition_dim=partition_dim,
        tp_mode=tp_mode,
        gram_restart_steps=gram_restart_iterations,
        gram_dtype=gram_dtype,
        gram_kernel_policy=gram_kernel_policy,
        eps=eps,
        fp32_matmul_prec=fp32_matmul_prec,
    )
    if global_shape is None:
        global_shape = tuple(int(dim) for dim in update.shape[-2:])
    scale = get_muon_scale_factor(int(global_shape[0]), int(global_shape[1]), mode=scale_mode)
    return orth_update * float(scale) * float(extra_scale_factor)


def compute_muon_update(
    grad: Tensor,
    momentum: Tensor,
    *,
    beta: float,
    nesterov: bool,
    ns_backend: str = "standard",
    coefficient_type: str = "quintic",
    num_ns_steps: int = 5,
    eps: float = 1e-7,
    scale_mode: str = "spectral",
    extra_scale_factor: float = 1.0,
    global_shape: Optional[tuple[int, int]] = None,
    tp_group: Optional[dist.ProcessGroup] = None,
    partition_dim: Optional[int] = None,
    tp_mode: str = "blockwise",
    gram_restart_iterations: tuple[int, ...] = (2,),
    gram_dtype: Optional[torch.dtype | str] = None,
    gram_kernel_policy: str = "torch",
    fp32_matmul_prec: str = "medium",
) -> Tensor:
    """Compute a complete Muon matrix update from grad and momentum state."""
    update = apply_muon_momentum(momentum, grad, beta=beta, nesterov=nesterov)
    return orthogonalize_muon_update(
        update,
        ns_backend=ns_backend,
        coefficient_type=coefficient_type,
        num_ns_steps=num_ns_steps,
        eps=eps,
        scale_mode=scale_mode,
        extra_scale_factor=extra_scale_factor,
        global_shape=global_shape,
        tp_group=tp_group,
        partition_dim=partition_dim,
        tp_mode=tp_mode,
        gram_restart_iterations=gram_restart_iterations,
        gram_dtype=gram_dtype,
        gram_kernel_policy=gram_kernel_policy,
        fp32_matmul_prec=fp32_matmul_prec,
    )


__all__ = [
    "apply_muon_momentum",
    "compute_muon_update",
    "get_muon_scale_factor",
    "get_and_reset_gram_profile",
    "gram_newton_schulz",
    "gram_newton_schulz_tp",
    "logical_shape_for_tp",
    "muon_scale_factor",
    "nesterov_update",
    "newton_schulz",
    "newton_schulz_tp",
    "orthogonalize_muon",
    "orthogonalize_muon_update",
    "scaled_lr_for_shape",
    "standard_newton_schulz",
    "update_momentum",
]
