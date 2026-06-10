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


def _split_range(size: int, world_size: int, rank: int) -> tuple[int, int]:
    base = int(size) // int(world_size)
    rem = int(size) % int(world_size)
    start = int(rank) * base + min(int(rank), rem)
    return start, start + base + (1 if int(rank) < rem else 0)


def _normalize_shard_mode(mode: Optional[str]) -> str:
    mode = "blockwise" if mode is None else str(mode)
    if mode == "duplicated_debug":
        mode = "duplicated"
    if mode not in ("blockwise", "duplicated", "distributed"):
        raise RuntimeError(f"[MUON_INVALID_SHARD_MODE] mode={mode!r}")
    return mode


def _normalize_gram_restart_steps(restart_steps: tuple[int, ...], num_steps: int) -> set[int]:
    restarts = {int(step) for step in restart_steps}
    if any(step < 0 for step in restarts):
        raise RuntimeError(f"[MUON_INVALID_GRAM_RESTART_STEPS] steps={tuple(restarts)}")
    return {step for step in restarts if step < int(num_steps)}


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
    # Keep Dao/Quack on the symmetric Gram kernels, where the custom kernel is
    # useful. Generic GEMM goes through Quack autotuning per shape; full MCore
    # runs hit many Muon matrix shapes in one optimizer step and can retain
    # enough autotune workspace to OOM before the next forward.
    _DAO_MM = _TORCH_GRAM_BACKEND.mm
    _DAO_MM_ADD = _TORCH_GRAM_BACKEND.mm_add
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


def _default_gram_dtype(x: Tensor, *, policy: str) -> Optional[torch.dtype]:
    if not x.is_cuda or x.dtype == torch.float64:
        return None
    policy = _normalize_gram_kernel_policy(policy)
    if policy not in ("dao", "compile"):
        return None
    if min(int(x.size(-2)), int(x.size(-1))) <= _DAO_GRAM_TILE_SIZE:
        return None
    if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


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
            "[MUON_GRAM_DAO_UNAVAILABLE] "
            "Dao/Quack Gram NS backend was requested, but optional dependencies "
            "are not installed. Install them with `uv sync --extra muon-gram`, "
            "or use --muon-gram-ns-kernel-policy=auto/torch."
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
    q: Optional[Tensor] = None
    for step, (a, b, c) in enumerate(coeffs):
        if step in restarts and step != 0:
            x_work = mm(q, x_work)
            r = sym_mm(x_work, x_work.mT)
            q = None

        z = sym_baddmm(r, r, C=r, alpha=c, beta=b)
        q = _add_identity_scaled_(z, a, alias_source=r) if q is None else sym_baddmm(q, z, C=q, beta=a)

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

    compile_mode = os.getenv("MEGATRON_MUON_COMPILE_MODE", "default").strip() or "default"
    return torch.compile(fn, fullgraph=True, mode=compile_mode)


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
    if dtype is None:
        dtype = _default_gram_dtype(x, policy=backend_name)
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
    norm_sq = torch.linalg.vector_norm(x, ord=2, dim=(-2, -1), keepdim=True).square()
    if _dist_world_size(group) > 1:
        dist.all_reduce(norm_sq, op=dist.ReduceOp.SUM, group=group)
    return x * norm_sq.clamp_min(float(eps) * float(eps)).rsqrt()


def _normalize_ns_input(x: Tensor, *, eps: float = 1e-7, group=None) -> tuple[Tensor, bool]:
    transposed = x.size(-2) > x.size(-1)
    x_work = x.mT if transposed else x
    dtype = torch.float64 if x_work.dtype == torch.float64 else torch.float32
    x_work = x_work.to(dtype=dtype)
    return _normalize_frobenius(x_work, eps=eps, group=group), transposed


def _logical_transpose(x: Tensor, logical_shape: Optional[tuple[int, int]]) -> bool:
    if logical_shape is None:
        return bool(x.size(-2) > x.size(-1))
    return int(logical_shape[0]) > int(logical_shape[1])


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
    first_gram = None
    fused_norm = _dist_world_size(tp_group) > 1
    if transpose is None:
        transposed = x.size(-2) > x.size(-1)
        x_oriented = x.mT if transposed else x
        dtype = torch.float64 if x_oriented.dtype == torch.float64 else torch.float32
        x_work = x_oriented.to(dtype=dtype)
    else:
        x_oriented = x.mT if transpose else x
        dtype = torch.float64 if x_oriented.dtype == torch.float64 else torch.float32
        x_work = x_oriented.to(dtype=dtype)
        transposed = bool(transpose)

    with _matmul_precision(fp32_matmul_prec):
        if fused_norm:
            first_gram = x_work @ x_work.mT if gram_side == "left" else x_work.mT @ x_work
            dist.all_reduce(first_gram, op=dist.ReduceOp.SUM, group=tp_group)
            norm_sq = torch.diagonal(first_gram.float(), dim1=-2, dim2=-1).sum(dim=-1)
            norm_sq = norm_sq.view(*norm_sq.shape, 1, 1)
            inv_norm = norm_sq.sqrt().clamp_min(float(eps)).reciprocal()
            x_work = x_work * inv_norm
            first_gram = first_gram * inv_norm.square()
        else:
            x_work = _normalize_frobenius(x_work, eps=eps, group=tp_group)
        for a, b, c in _coefficients(
            steps,
            coefficient_type=coefficient_type,
            custom_coefficient_sets=custom_coefficient_sets,
        ):
            use_first_gram = first_gram is not None
            if use_first_gram:
                gram = first_gram
                first_gram = None
            else:
                gram = x_work @ x_work.mT if gram_side == "left" else x_work.mT @ x_work
            if _dist_world_size(tp_group) > 1 and not use_first_gram:
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
    return_sizes: bool = False,
    partition_sizes: Optional[Sequence[int]] = None,
) -> Tensor | tuple[Tensor, tuple[int, ...]]:
    world_size = dist.get_world_size(tp_group)
    matrix_dim = x.ndim - 2 + int(partition_dim)
    local_size = int(x.size(matrix_dim))
    if partition_sizes is None:
        local_size_t = torch.tensor([local_size], device=x.device, dtype=torch.int64)
        gathered_sizes = [torch.empty_like(local_size_t) for _ in range(world_size)]
        dist.all_gather(gathered_sizes, local_size_t, group=tp_group)
        sizes = [int(size.item()) for size in gathered_sizes]
    else:
        sizes = [int(size) for size in partition_sizes]
        if len(sizes) != int(world_size):
            raise RuntimeError(
                "[MUON_INVALID_GATHER_PARTITION_SIZES] "
                f"num_sizes={len(sizes)} world_size={int(world_size)}"
            )
        rank = dist.get_rank(tp_group)
        if int(sizes[int(rank)]) != local_size:
            raise RuntimeError(
                "[MUON_GATHER_PARTITION_SIZE_MISMATCH] "
                f"rank={int(rank)} expected={int(sizes[int(rank)])} actual={local_size}"
            )
    max_size = max(sizes)
    if max_size == local_size:
        send = x.contiguous()
    else:
        padded_shape = list(x.shape)
        padded_shape[matrix_dim] = max_size
        send = x.new_zeros(tuple(padded_shape))
        index = [slice(None)] * x.ndim
        index[matrix_dim] = slice(0, local_size)
        send[tuple(index)].copy_(x.contiguous())
    gathered = [torch.empty_like(send) for _ in range(world_size)]
    dist.all_gather(gathered, send, group=tp_group)
    shards = []
    for shard, size in zip(gathered, sizes):
        index = [slice(None)] * x.ndim
        index[matrix_dim] = slice(0, int(size))
        shards.append(shard[tuple(index)])
    gathered = torch.cat(shards, dim=matrix_dim)
    return (gathered, tuple(sizes)) if return_sizes else gathered


def _slice_tp_matrix(
    x: Tensor,
    *,
    partition_dim: int,
    tp_group: dist.ProcessGroup,
    partition_sizes: Optional[tuple[int, ...]] = None,
) -> Tensor:
    world_size = dist.get_world_size(tp_group)
    rank = dist.get_rank(tp_group)
    matrix_dim = x.ndim - 2 + int(partition_dim)
    if partition_sizes is None:
        start, end = _split_range(int(x.size(matrix_dim)), int(world_size), int(rank))
    else:
        if len(partition_sizes) != int(world_size):
            raise RuntimeError(
                "[MUON_INVALID_PARTITION_SIZES] "
                f"num_sizes={len(partition_sizes)} world_size={int(world_size)}"
            )
        start = sum(int(size) for size in partition_sizes[: int(rank)])
        end = start + int(partition_sizes[int(rank)])
    index = [slice(None)] * x.ndim
    index[matrix_dim] = slice(start, end)
    return x[tuple(index)].contiguous()


def _slice_matrix_dim(
    x: Tensor,
    *,
    matrix_dim: int,
    start: int,
    end: int,
) -> Tensor:
    index = [slice(None)] * x.ndim
    index[int(matrix_dim)] = slice(int(start), int(end))
    return x[tuple(index)]


def _scatter_duplicated_update(
    full: Tensor,
    *,
    partition_dim: int,
    tp_group: dist.ProcessGroup,
    partition_sizes: Sequence[int],
    compute_update,
) -> Tensor:
    world_size = int(dist.get_world_size(tp_group))
    rank = int(dist.get_rank(tp_group))
    sizes = tuple(int(size) for size in partition_sizes)
    if len(sizes) != world_size:
        raise RuntimeError(
            "[MUON_DUPLICATED_SCATTER_PARTITION_SIZE_MISMATCH] "
            f"sizes={len(sizes)} world_size={world_size}"
        )
    local_size = int(sizes[rank])
    max_size = max(sizes)
    matrix_dim = full.ndim - 2 + int(partition_dim)
    group_ranks = tuple(dist.get_process_group_ranks(tp_group))
    if len(group_ranks) != world_size:
        raise RuntimeError(
            "[MUON_DUPLICATED_SCATTER_GROUP_RANK_MISMATCH] "
            f"group_ranks={len(group_ranks)} world_size={world_size}"
        )

    def _padded_shape(tensor: Tensor) -> list[int]:
        shape = list(tensor.shape)
        shape[matrix_dim] = max_size
        return shape

    if full.ndim <= 2:
        recv = full.new_empty(_padded_shape(full))
        if rank == 0:
            update = compute_update(full)
            scatter_list = []
            offset = 0
            for size in sizes:
                send_update = update.new_zeros(_padded_shape(update))
                shard = _slice_matrix_dim(
                    update,
                    matrix_dim=matrix_dim,
                    start=offset,
                    end=offset + int(size),
                )
                _slice_matrix_dim(send_update, matrix_dim=matrix_dim, start=0, end=int(size)).copy_(
                    shard
                )
                scatter_list.append(send_update)
                offset += int(size)
        else:
            scatter_list = None
        dist.scatter(recv, scatter_list=scatter_list, src=int(group_ranks[0]), group=tp_group)
        return _slice_matrix_dim(recv, matrix_dim=matrix_dim, start=0, end=local_size).contiguous()

    batch_size = int(full.size(0))
    local_shape = list(full.shape)
    local_shape[matrix_dim] = local_size
    result = full.new_empty(local_shape)
    for owner_rank in range(world_size):
        owned_indices = [index for index in range(batch_size) if index % world_size == owner_rank]
        if not owned_indices:
            continue
        recv_shape = list(full.shape)
        recv_shape[0] = len(owned_indices)
        recv_shape[matrix_dim] = max_size
        recv = full.new_empty(recv_shape)
        if rank == owner_rank:
            index_tensor = torch.tensor(owned_indices, dtype=torch.long, device=full.device)
            owned_full = full.index_select(0, index_tensor)
            owned_update = compute_update(owned_full)
            scatter_list = []
            offset = 0
            for size in sizes:
                send_update = owned_update.new_zeros(recv_shape)
                shard = _slice_matrix_dim(
                    owned_update,
                    matrix_dim=matrix_dim,
                    start=offset,
                    end=offset + int(size),
                )
                _slice_matrix_dim(send_update, matrix_dim=matrix_dim, start=0, end=int(size)).copy_(
                    shard
                )
                scatter_list.append(send_update)
                offset += int(size)
        else:
            scatter_list = None
        dist.scatter(
            recv,
            scatter_list=scatter_list,
            src=int(group_ranks[owner_rank]),
            group=tp_group,
        )
        local = _slice_matrix_dim(recv, matrix_dim=matrix_dim, start=0, end=local_size)
        result.index_copy_(
            0,
            torch.tensor(owned_indices, dtype=torch.long, device=full.device),
            local,
        )
    return result.contiguous()


def _partition_sizes(size: int, world_size: int) -> tuple[int, ...]:
    return tuple(
        int(end) - int(start)
        for start, end in (
            _split_range(int(size), int(world_size), rank)
            for rank in range(int(world_size))
        )
    )


def _known_partition_sizes_for_local(
    *,
    global_size: Optional[int],
    local_size: int,
    group,
) -> Optional[tuple[int, ...]]:
    if global_size is None or group is None or _dist_world_size(group) <= 1:
        return None
    sizes = _partition_sizes(int(global_size), int(_dist_world_size(group)))
    rank = int(dist.get_rank(group))
    if int(sizes[rank]) != int(local_size):
        return None
    return sizes


def newton_schulz_1d(
    x: Tensor,
    *,
    steps: int = 5,
    coefficient_type: str = "quintic",
    tp_group: Optional[dist.ProcessGroup] = None,
    partition_dim: Optional[int] = None,
    mode: Optional[str] = None,
    tp_mode: Optional[str] = None,
    logical_shape: Optional[tuple[int, int]] = None,
    eps: float = 1e-7,
    fp32_matmul_prec: str = "medium",
) -> Tensor:
    """1D-shard-aware Newton-Schulz using explicit collectives."""
    mode = _normalize_shard_mode(tp_mode if tp_mode is not None else mode)
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

    if mode == "duplicated":
        known_partition_sizes = None
        if logical_shape is not None:
            known_partition_sizes = _partition_sizes(
                int(logical_shape[int(partition_dim)]),
                dist.get_world_size(tp_group),
            )
        full, partition_sizes = _all_gather_matrix(
            x,
            partition_dim=int(partition_dim),
            tp_group=tp_group,
            return_sizes=True,
            partition_sizes=known_partition_sizes,
        )
        return _scatter_duplicated_update(
            full,
            partition_dim=int(partition_dim),
            tp_group=tp_group,
            partition_sizes=partition_sizes,
            compute_update=lambda tensor: newton_schulz(
                tensor,
                steps=steps,
                coefficient_type=coefficient_type,
                eps=eps,
                fp32_matmul_prec=fp32_matmul_prec,
            ),
        )

    transposed, oriented_partition_dim = _oriented_partition_dim(
        x,
        int(partition_dim),
        logical_shape=logical_shape,
    )
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


def _add_identity_scaled_(tensor: Tensor, scale: float, *, alias_source: Optional[Tensor] = None) -> Tensor:
    if alias_source is not None and tensor.untyped_storage().data_ptr() == alias_source.untyped_storage().data_ptr():
        tensor = tensor.clone()
    tensor.diagonal(dim1=-2, dim2=-1).add_(scale)
    return tensor


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


def _oriented_partition_dim(
    x: Tensor,
    partition_dim: int,
    *,
    logical_shape: Optional[tuple[int, int]] = None,
) -> tuple[bool, int]:
    transposed = _logical_transpose(x, logical_shape)
    oriented_partition_dim = 1 - int(partition_dim) if transposed else int(partition_dim)
    return transposed, oriented_partition_dim


def _tp_oriented_partition_dim(x: Tensor, partition_dim: int) -> tuple[bool, int]:
    return _oriented_partition_dim(x, partition_dim)


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
    transpose: Optional[bool] = None,
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
    policy = _normalize_gram_kernel_policy(gram_kernel_policy)
    if transpose is None:
        transposed = x.size(-2) > x.size(-1)
        x_oriented = x.mT if transposed else x
        dtype = torch.float64 if x_oriented.dtype == torch.float64 else torch.float32
        x_work = x_oriented.to(dtype=dtype)
    else:
        x_oriented = x.mT if bool(transpose) else x
        dtype = torch.float64 if x_oriented.dtype == torch.float64 else torch.float32
        x_work = x_oriented.to(dtype=dtype)
        transposed = bool(transpose)
    dtype = _str_to_dtype(gram_dtype)
    if dtype is None:
        dtype = (
            None
            if _dist_world_size(tp_group) > 1
            else _default_gram_dtype(x_work, policy=policy)
        )
    if dtype is not None:
        x_work = x_work.to(dtype=dtype)
    ops = _select_gram_backend(x_work, policy=policy)
    first_gram = None
    if _dist_world_size(tp_group) > 1:
        first_gram = _form_distributed_gram(
            x_work,
            group=tp_group,
            side=gram_side,
            ops=ops,
        )
        norm_sq = torch.diagonal(first_gram.float(), dim1=-2, dim2=-1).sum(dim=-1)
        norm_sq = norm_sq.view(*norm_sq.shape, 1, 1)
        inv_norm = norm_sq.sqrt().clamp_min(float(eps)).reciprocal()
        x_work = x_work * inv_norm
        first_gram = first_gram * inv_norm.square()
    else:
        x_work = _normalize_frobenius(x_work, eps=eps)

    coeffs = tuple(_coefficients(steps, coefficient_type=coefficient_type))
    if gram_side == "right":
        restarts = _normalize_gram_restart_steps(restart_steps, len(coeffs))
        with _matmul_precision(fp32_matmul_prec):
            r = first_gram
            first_gram = None
            if r is None:
                r = _form_distributed_gram(x_work, group=tp_group, side="right", ops=ops)
            q: Optional[Tensor] = None
            for step, (a, b, c) in enumerate(coeffs):
                if step in restarts and step != 0:
                    if q is None:
                        raise RuntimeError("[MUON_GRAM_NS_MISSING_Q_AT_RESTART]")
                    x_work = ops.mm(x_work, q)
                    r = _form_distributed_gram(x_work, group=tp_group, side="right", ops=ops)
                    q = None

                z = ops.sym_baddmm(r, r, C=r, alpha=c, beta=b)
                q = _add_identity_scaled_(z, a, alias_source=r) if q is None else ops.sym_baddmm(q, z, C=q, beta=a)

                if step < len(coeffs) - 1 and (step + 1) not in restarts:
                    rz = ops.sym_baddmm(r, z, C=r, beta=a)
                    r = ops.sym_baddmm(z, rz, C=rz, beta=a)
            if q is None:
                raise RuntimeError("[MUON_GRAM_NS_MISSING_FINAL_Q]")
            x_work = ops.mm(x_work, q)
    elif x_work.size(-2) == x_work.size(-1):
        with _matmul_precision(fp32_matmul_prec):
            for a, b, c in coeffs:
                if first_gram is not None:
                    gram = first_gram
                    first_gram = None
                else:
                    gram = _form_distributed_gram(x_work, group=tp_group, ops=ops)
                poly = ops.sym_baddmm(gram, gram, C=gram, alpha=c, beta=b)
                x_work = ops.mm_add(poly, x_work, C=x_work, beta=a)
    else:
        restarts = _normalize_gram_restart_steps(restart_steps, len(coeffs))
        with _matmul_precision(fp32_matmul_prec):
            r = first_gram
            first_gram = None
            if r is None:
                r = _form_distributed_gram(x_work, group=tp_group, ops=ops)
            q: Optional[Tensor] = None
            for step, (a, b, c) in enumerate(coeffs):
                if step in restarts and step != 0:
                    if q is None:
                        raise RuntimeError("[MUON_GRAM_NS_MISSING_Q_AT_RESTART]")
                    x_work = ops.mm(q, x_work)
                    r = _form_distributed_gram(x_work, group=tp_group, ops=ops)
                    q = None

                z = ops.sym_baddmm(r, r, C=r, alpha=c, beta=b)
                q = _add_identity_scaled_(z, a, alias_source=r) if q is None else ops.sym_baddmm(q, z, C=q, beta=a)

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
    transpose: Optional[bool] = None,
    gram_side: str = "left",
    gram_kernel_policy: str = "torch",
) -> Tensor:
    if restart_iterations is not None:
        restart_steps = restart_iterations
    policy = _normalize_gram_kernel_policy(gram_kernel_policy)
    compile_requested = policy == "compile" or os.getenv("MEGATRON_MUON_COMPILE_GRAM_NS", "0") == "1"
    if compile_requested and tp_group is None and transpose is None and gram_side == "left":
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
        transpose=transpose,
        gram_side=gram_side,
        gram_kernel_policy=policy,
    )


def gram_newton_schulz_1d(
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
    logical_shape: Optional[tuple[int, int]] = None,
    eps: float = 1e-7,
    fp32_matmul_prec: str = "medium",
    gram_kernel_policy: str = "torch",
) -> Tensor:
    """1D-shard-aware Gram Newton-Schulz using no full gather outside duplicated mode."""
    mode = _normalize_shard_mode(mode)
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
    if mode == "duplicated":
        known_partition_sizes = None
        if logical_shape is not None:
            known_partition_sizes = _partition_sizes(
                int(logical_shape[int(partition_dim)]),
                dist.get_world_size(tp_group),
            )
        full, partition_sizes = _all_gather_matrix(
            x,
            partition_dim=int(partition_dim),
            tp_group=tp_group,
            return_sizes=True,
            partition_sizes=known_partition_sizes,
        )
        return _scatter_duplicated_update(
            full,
            partition_dim=int(partition_dim),
            tp_group=tp_group,
            partition_sizes=partition_sizes,
            compute_update=lambda tensor: gram_newton_schulz(
                tensor,
                steps=steps,
                coefficient_type=coefficient_type,
                restart_steps=restart_steps,
                restart_iterations=restart_iterations,
                gram_dtype=gram_dtype,
                eps=eps,
                fp32_matmul_prec=fp32_matmul_prec,
                gram_kernel_policy=gram_kernel_policy,
            ),
        )
    transposed, oriented_partition_dim = _oriented_partition_dim(
        x,
        int(partition_dim),
        logical_shape=logical_shape,
    )
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
        transpose=transposed,
        gram_side="left" if oriented_partition_dim == 1 else "right",
        gram_kernel_policy=gram_kernel_policy,
    )


def _oriented_2d_groups(
    x: Tensor,
    *,
    logical_shape: Optional[tuple[int, int]],
    fs_group: Optional[dist.ProcessGroup],
    fs_partition_dim: int,
    tp_group: Optional[dist.ProcessGroup],
    tp_partition_dim: int,
) -> tuple[Tensor, bool, dist.ProcessGroup, dist.ProcessGroup]:
    if fs_partition_dim not in (0, 1) or tp_partition_dim not in (0, 1):
        raise RuntimeError(
            "[MUON_2D_DISTRIBUTED_INVALID_DIMS] "
            f"fs_partition_dim={fs_partition_dim} tp_partition_dim={tp_partition_dim}"
        )
    transposed = _logical_transpose(x, logical_shape)
    fs_dim = 1 - int(fs_partition_dim) if transposed else int(fs_partition_dim)
    tp_dim = 1 - int(tp_partition_dim) if transposed else int(tp_partition_dim)
    if fs_dim == tp_dim:
        raise RuntimeError(
            "[MUON_2D_DISTRIBUTED_NON_ORTHOGONAL_DIMS] "
            f"fs_partition_dim={fs_partition_dim} tp_partition_dim={tp_partition_dim} "
            f"transposed={int(transposed)}"
        )
    if fs_group is None or tp_group is None:
        raise RuntimeError("[MUON_2D_DISTRIBUTED_REQUIRES_FS_AND_TP_GROUPS]")

    x_work = x.mT if transposed else x
    row_group = fs_group if fs_dim == 0 else tp_group
    col_group = fs_group if fs_dim == 1 else tp_group
    return x_work, transposed, row_group, col_group


def _form_2d_left_gram(slab: Tensor, *, col_group=None, ops=_TORCH_GRAM_BACKEND) -> Tensor:
    gram = ops.sym_mm(slab, slab.mT)
    if _dist_world_size(col_group) > 1:
        dist.all_reduce(gram, op=dist.ReduceOp.SUM, group=col_group)
        _record_gram_all_reduce(gram)
    return gram


def _finish_2d_slab(
    slab: Tensor,
    *,
    row_group,
    transposed: bool,
    row_partition_sizes: Optional[tuple[int, ...]] = None,
) -> Tensor:
    local = _slice_tp_matrix(
        slab,
        partition_dim=0,
        tp_group=row_group,
        partition_sizes=row_partition_sizes,
    )
    return local.mT.contiguous() if transposed else local


def newton_schulz_2d(
    x: Tensor,
    *,
    steps: int = 5,
    coefficient_type: str = "quintic",
    fs_group: Optional[dist.ProcessGroup] = None,
    fs_partition_dim: int = -1,
    tp_group: Optional[dist.ProcessGroup] = None,
    tp_partition_dim: int = -1,
    logical_shape: Optional[tuple[int, int]] = None,
    row_partition_sizes: Optional[tuple[int, ...]] = None,
    eps: float = 1e-7,
    fp32_matmul_prec: str = "medium",
) -> Tensor:
    """Exact 2D-sharded standard NS without materializing the full matrix."""
    if x.ndim < 2:
        raise RuntimeError(f"[MUON_NS_REQUIRES_MATRIX] ndim={x.ndim}")
    original_dtype = x.dtype
    x_work, transposed, row_group, col_group = _oriented_2d_groups(
        x,
        logical_shape=logical_shape,
        fs_group=fs_group,
        fs_partition_dim=fs_partition_dim,
        tp_group=tp_group,
        tp_partition_dim=tp_partition_dim,
    )
    dtype = torch.float64 if x_work.dtype == torch.float64 else torch.float32
    if row_partition_sizes is None and logical_shape is not None:
        global_row_size = int(logical_shape[1] if transposed else logical_shape[0])
        row_partition_sizes = _known_partition_sizes_for_local(
            global_size=global_row_size,
            local_size=int(x_work.size(-2)),
            group=row_group,
        )
    slab, row_partition_sizes = _all_gather_matrix(
        x_work.to(dtype=dtype),
        partition_dim=0,
        tp_group=row_group,
        return_sizes=True,
        partition_sizes=row_partition_sizes,
    )

    first_gram = None
    with _matmul_precision(fp32_matmul_prec):
        first_gram = _form_2d_left_gram(slab, col_group=col_group)
        norm_sq = torch.diagonal(first_gram.float(), dim1=-2, dim2=-1).sum(dim=-1)
        norm_sq = norm_sq.view(*norm_sq.shape, 1, 1)
        inv_norm = norm_sq.sqrt().clamp_min(float(eps)).reciprocal()
        slab = slab * inv_norm
        first_gram = first_gram * inv_norm.square()
        for a, b, c in _coefficients(steps, coefficient_type=coefficient_type):
            if first_gram is not None:
                gram = first_gram
                first_gram = None
            else:
                gram = _form_2d_left_gram(slab, col_group=col_group)
            gram2 = gram @ gram
            poly = b * gram + c * gram2
            slab = a * slab + poly @ slab

    result = _finish_2d_slab(
        slab,
        row_group=row_group,
        transposed=transposed,
        row_partition_sizes=row_partition_sizes,
    )
    return result.to(dtype=original_dtype)


def gram_newton_schulz_2d(
    x: Tensor,
    *,
    steps: int = 5,
    coefficient_type: str = "quintic",
    restart_steps: tuple[int, ...] = (2,),
    restart_iterations: Optional[tuple[int, ...]] = None,
    gram_dtype: Optional[torch.dtype | str] = None,
    fs_group: Optional[dist.ProcessGroup] = None,
    fs_partition_dim: int = -1,
    tp_group: Optional[dist.ProcessGroup] = None,
    tp_partition_dim: int = -1,
    logical_shape: Optional[tuple[int, int]] = None,
    row_partition_sizes: Optional[tuple[int, ...]] = None,
    eps: float = 1e-7,
    fp32_matmul_prec: str = "medium",
    gram_kernel_policy: str = "torch",
) -> Tensor:
    """Exact 2D-sharded Gram NS without materializing the full matrix."""
    if restart_iterations is not None:
        restart_steps = restart_iterations
    if x.ndim < 2:
        raise RuntimeError(f"[MUON_GRAM_NS_REQUIRES_MATRIX] ndim={x.ndim}")
    original_dtype = x.dtype
    policy = _normalize_gram_kernel_policy(gram_kernel_policy)
    x_work, transposed, row_group, col_group = _oriented_2d_groups(
        x,
        logical_shape=logical_shape,
        fs_group=fs_group,
        fs_partition_dim=fs_partition_dim,
        tp_group=tp_group,
        tp_partition_dim=tp_partition_dim,
    )
    dtype = _str_to_dtype(gram_dtype)
    if dtype is None:
        dtype = torch.float64 if x_work.dtype == torch.float64 else torch.float32
        collective = _dist_world_size(fs_group) > 1 or _dist_world_size(tp_group) > 1
        if dtype == torch.float32 and not collective:
            default_dtype = _default_gram_dtype(x_work, policy=policy)
            if default_dtype is not None:
                dtype = default_dtype
    if row_partition_sizes is None and logical_shape is not None:
        global_row_size = int(logical_shape[1] if transposed else logical_shape[0])
        row_partition_sizes = _known_partition_sizes_for_local(
            global_size=global_row_size,
            local_size=int(x_work.size(-2)),
            group=row_group,
        )
    slab, row_partition_sizes = _all_gather_matrix(
        x_work.to(dtype=dtype),
        partition_dim=0,
        tp_group=row_group,
        return_sizes=True,
        partition_sizes=row_partition_sizes,
    )
    ops = _select_gram_backend(slab, policy=policy)
    coeffs = tuple(_coefficients(steps, coefficient_type=coefficient_type))
    restarts = _normalize_gram_restart_steps(restart_steps, len(coeffs))

    with _matmul_precision(fp32_matmul_prec):
        r = _form_2d_left_gram(slab, col_group=col_group, ops=ops)
        norm_sq = torch.diagonal(r.float(), dim1=-2, dim2=-1).sum(dim=-1)
        norm_sq = norm_sq.view(*norm_sq.shape, 1, 1)
        inv_norm = norm_sq.sqrt().clamp_min(float(eps)).reciprocal()
        slab = slab * inv_norm
        r = r * inv_norm.square()
        q: Optional[Tensor] = None
        for step, (a, b, c) in enumerate(coeffs):
            if step in restarts and step != 0:
                if q is None:
                    raise RuntimeError("[MUON_GRAM_NS_MISSING_Q_AT_RESTART]")
                slab = ops.mm(q, slab)
                r = _form_2d_left_gram(slab, col_group=col_group, ops=ops)
                q = None

            z = ops.sym_baddmm(r, r, C=r, alpha=c, beta=b)
            q = _add_identity_scaled_(z, a, alias_source=r) if q is None else ops.sym_baddmm(q, z, C=q, beta=a)

            if step < len(coeffs) - 1 and (step + 1) not in restarts:
                rz = ops.sym_baddmm(r, z, C=r, beta=a)
                r = ops.sym_baddmm(z, rz, C=rz, beta=a)
        if q is None:
            raise RuntimeError("[MUON_GRAM_NS_MISSING_FINAL_Q]")
        slab = ops.mm(q, slab)

    slab = slab.to(dtype=torch.float32 if original_dtype != torch.float64 else torch.float64)
    result = _finish_2d_slab(
        slab,
        row_group=row_group,
        transposed=transposed,
        row_partition_sizes=row_partition_sizes,
    )
    return result.to(dtype=original_dtype)


def orthogonalize_muon(
    x: Tensor,
    *,
    ns_backend: str = "standard",
    steps: int = 5,
    coefficient_type: str = "quintic",
    tp_group: Optional[dist.ProcessGroup] = None,
    partition_dim: Optional[int] = None,
    tp_mode: str = "blockwise",
    logical_shape: Optional[tuple[int, int]] = None,
    gram_restart_steps: tuple[int, ...] = (2,),
    gram_dtype: Optional[torch.dtype | str] = None,
    gram_kernel_policy: str = "torch",
    eps: float = 1e-7,
    fp32_matmul_prec: str = "medium",
) -> Tensor:
    """Dispatch Muon orthogonalization backend."""
    if ns_backend == "standard":
        return newton_schulz_1d(
            x,
            steps=steps,
            coefficient_type=coefficient_type,
            tp_group=tp_group,
            partition_dim=partition_dim,
            mode=tp_mode,
            logical_shape=logical_shape,
            eps=eps,
            fp32_matmul_prec=fp32_matmul_prec,
        )
    if ns_backend == "gram":
        return gram_newton_schulz_1d(
            x,
            steps=steps,
            coefficient_type=coefficient_type,
            restart_steps=gram_restart_steps,
            gram_dtype=gram_dtype,
            tp_group=tp_group,
            partition_dim=partition_dim,
            mode=tp_mode,
            logical_shape=logical_shape,
            eps=eps,
            fp32_matmul_prec=fp32_matmul_prec,
            gram_kernel_policy=gram_kernel_policy,
        )
    raise RuntimeError(f"[MUON_INVALID_NS_BACKEND] ns_backend={ns_backend!r}")


def orthogonalize_muon_2d(
    x: Tensor,
    *,
    ns_backend: str = "standard",
    steps: int = 5,
    coefficient_type: str = "quintic",
    fs_group: Optional[dist.ProcessGroup] = None,
    fs_partition_dim: int = -1,
    tp_group: Optional[dist.ProcessGroup] = None,
    tp_partition_dim: int = -1,
    logical_shape: Optional[tuple[int, int]] = None,
    row_partition_sizes: Optional[tuple[int, ...]] = None,
    gram_restart_steps: tuple[int, ...] = (2,),
    gram_dtype: Optional[torch.dtype | str] = None,
    gram_kernel_policy: str = "torch",
    eps: float = 1e-7,
    fp32_matmul_prec: str = "medium",
) -> Tensor:
    """Dispatch exact 2D-sharded Muon orthogonalization."""
    if ns_backend == "standard":
        return newton_schulz_2d(
            x,
            steps=steps,
            coefficient_type=coefficient_type,
            fs_group=fs_group,
            fs_partition_dim=fs_partition_dim,
            tp_group=tp_group,
            tp_partition_dim=tp_partition_dim,
            logical_shape=logical_shape,
            row_partition_sizes=row_partition_sizes,
            eps=eps,
            fp32_matmul_prec=fp32_matmul_prec,
        )
    if ns_backend == "gram":
        return gram_newton_schulz_2d(
            x,
            steps=steps,
            coefficient_type=coefficient_type,
            restart_steps=gram_restart_steps,
            gram_dtype=gram_dtype,
            fs_group=fs_group,
            fs_partition_dim=fs_partition_dim,
            tp_group=tp_group,
            tp_partition_dim=tp_partition_dim,
            logical_shape=logical_shape,
            row_partition_sizes=row_partition_sizes,
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
    logical_shape: Optional[tuple[int, int]] = None,
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
        logical_shape=logical_shape,
        gram_restart_steps=gram_restart_iterations,
        gram_dtype=gram_dtype,
        gram_kernel_policy=gram_kernel_policy,
        eps=eps,
        fp32_matmul_prec=fp32_matmul_prec,
    )
    if global_shape is None:
        global_shape = tuple(int(dim) for dim in update.shape[-2:])
    scale = muon_scale_factor(int(global_shape[0]), int(global_shape[1]), mode=scale_mode)
    return orth_update * float(scale) * float(extra_scale_factor)


def orthogonalize_muon_update_2d(
    update: Tensor,
    *,
    ns_backend: str = "standard",
    coefficient_type: str = "quintic",
    num_ns_steps: int = 5,
    eps: float = 1e-7,
    scale_mode: str = "spectral",
    extra_scale_factor: float = 1.0,
    global_shape: Optional[tuple[int, int]] = None,
    logical_shape: Optional[tuple[int, int]] = None,
    fs_group: Optional[dist.ProcessGroup] = None,
    fs_partition_dim: int = -1,
    tp_group: Optional[dist.ProcessGroup] = None,
    tp_partition_dim: int = -1,
    row_partition_sizes: Optional[tuple[int, ...]] = None,
    gram_restart_iterations: tuple[int, ...] = (2,),
    gram_dtype: Optional[torch.dtype | str] = None,
    gram_kernel_policy: str = "torch",
    fp32_matmul_prec: str = "medium",
) -> Tensor:
    """Orthogonalize and scale one 2D-sharded Muon update tensor."""
    if logical_shape is None:
        logical_shape = global_shape
    orth_update = orthogonalize_muon_2d(
        update,
        ns_backend=ns_backend,
        steps=num_ns_steps,
        coefficient_type=coefficient_type,
        fs_group=fs_group,
        fs_partition_dim=fs_partition_dim,
        tp_group=tp_group,
        tp_partition_dim=tp_partition_dim,
        logical_shape=logical_shape,
        row_partition_sizes=row_partition_sizes,
        gram_restart_steps=gram_restart_iterations,
        gram_dtype=gram_dtype,
        gram_kernel_policy=gram_kernel_policy,
        eps=eps,
        fp32_matmul_prec=fp32_matmul_prec,
    )
    if global_shape is None:
        global_shape = tuple(int(dim) for dim in update.shape[-2:])
    scale = muon_scale_factor(int(global_shape[0]), int(global_shape[1]), mode=scale_mode)
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
    logical_shape: Optional[tuple[int, int]] = None,
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
        logical_shape=logical_shape,
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
    "get_and_reset_gram_profile",
    "gram_newton_schulz",
    "gram_newton_schulz_1d",
    "gram_newton_schulz_2d",
    "logical_shape_for_tp",
    "muon_scale_factor",
    "nesterov_update",
    "newton_schulz",
    "newton_schulz_1d",
    "newton_schulz_2d",
    "orthogonalize_muon",
    "orthogonalize_muon_2d",
    "orthogonalize_muon_update",
    "orthogonalize_muon_update_2d",
    "scaled_lr_for_shape",
    "update_momentum",
]
