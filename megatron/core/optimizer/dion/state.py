"""Dion state helpers matching the dion_reference.py contract.

This module isolates Dion-local state-init semantics from Megatron runtime.
The caller remains responsible for:

- local shard / metadata routing
- process-group discovery
- topology validation
- runtime ordering
"""

import hashlib
import math
from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor

from .linear import get_linear_split_rows_from_dist_meta
from .qkv import get_qkv_split_shapes_from_dist_meta
from .types import (
    DionMixedPrecisionConfig,
    DionParamConfig,
    DionQLayout,
    DionQInit,
    DionDistMeta,
)
from .utils import get_global_shape, str_to_dtype


def _split_range(size: int, world_size: int, rank: int) -> Tuple[int, int]:
    """Return the canonical contiguous range for one rank."""
    size = int(size)
    world_size = int(world_size)
    rank = int(rank)
    if size < 0:
        raise RuntimeError(f"[DION_INVALID_RANGE_SIZE] size={size}")
    if world_size <= 0:
        raise RuntimeError(f"[DION_INVALID_WORLD_SIZE] world_size={world_size}")
    if rank < 0 or rank >= world_size:
        raise RuntimeError(f"[DION_INVALID_RANK] rank={rank} world_size={world_size}")
    base = size // world_size
    remainder = size % world_size
    start = rank * base + min(rank, remainder)
    end = start + base + (1 if rank < remainder else 0)
    return int(start), int(end)


def _normal_q_submatrix(
    *,
    q_global_shape: Tuple[int, int],
    row_start: int,
    row_end: int,
    col_start: int,
    col_end: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Generate one local Q block from the seeded full-Q random stream."""
    rows, cols = (int(q_global_shape[0]), int(q_global_shape[1]))
    row_start = int(row_start)
    row_end = int(row_end)
    col_start = int(col_start)
    col_end = int(col_end)
    if rows <= 0 or cols <= 0:
        raise RuntimeError(f"[DION_INVALID_Q_GLOBAL_SHAPE] q_global_shape={q_global_shape}")
    if row_start < 0 or row_end < row_start or row_end > rows:
        raise RuntimeError(
            "[DION_INVALID_Q_ROW_RANGE] "
            f"q_global_shape={q_global_shape} row_start={row_start} row_end={row_end}"
        )
    if col_start < 0 or col_end < col_start or col_end > cols:
        raise RuntimeError(
            "[DION_INVALID_Q_COL_RANGE] "
            f"q_global_shape={q_global_shape} col_start={col_start} col_end={col_end}"
        )

    local_rows = row_end - row_start
    local_cols = col_end - col_start
    if local_rows <= 0 or local_cols <= 0:
        raise RuntimeError(
            "[DION_EMPTY_Q_SHARD] "
            f"q_global_shape={q_global_shape} row_range=({row_start}, {row_end}) "
            f"col_range=({col_start}, {col_end})"
        )
    q_state = torch.empty((local_rows, local_cols), device=device, dtype=dtype)

    gen = torch.Generator(device=str(device))
    gen.manual_seed(int(seed))

    if device.type != "cuda":
        q_full = torch.randn(q_global_shape, device=device, dtype=dtype, generator=gen)
        return q_full[row_start:row_end, col_start:col_end].contiguous()

    for local_row, global_row in enumerate(range(row_start, row_end)):
        element_start = int(global_row) * cols + col_start
        aligned_start = (element_start // 4) * 4
        prefix = element_start - aligned_start
        gen.set_offset(aligned_start)
        row_values = torch.empty(
            prefix + local_cols,
            device=device,
            dtype=dtype,
        )
        row_values.normal_(mean=0.0, std=1.0, generator=gen)
        q_state[local_row].copy_(row_values[prefix:])
    return q_state


def require_2d_local_shape(
    param: Tensor, dist_meta: DionDistMeta
) -> Tuple[int, int]:
    """Return the exact local 2D shard shape from distributed metadata."""
    if dist_meta is None or dist_meta.shape is None or len(dist_meta.shape) != 2:
        raise RuntimeError(
            "[Dion] distributed param is missing exact local 2D shape metadata "
            f"param_uid={getattr(dist_meta, 'param_uid', None)} "
            f"param_name={getattr(dist_meta, 'param_name', '')} "
            f"dist_meta_shape={getattr(dist_meta, 'shape', None)}"
        )
    local_shape = tuple(int(dim) for dim in dist_meta.shape)
    m_local, n_local = local_shape
    if m_local <= 0 or n_local <= 0:
        raise RuntimeError(
            "[Dion] invalid empty local 2D shape metadata "
            f"param_uid={getattr(dist_meta, 'param_uid', None)} "
            f"param_name={getattr(dist_meta, 'param_name', '')} "
            f"local_shape={local_shape}"
        )
    if int(param.numel()) != m_local * n_local:
        raise RuntimeError(
            "[Dion] local 2D shape metadata does not match shard numel "
            f"param_uid={getattr(dist_meta, 'param_uid', None)} "
            f"param_name={getattr(dist_meta, 'param_name', '')} "
            f"local_shape={local_shape} numel={int(param.numel())}"
        )
    return local_shape


def restore_tp_shape(
    m_local: int,
    n_local: int,
    config: DionParamConfig,
    *,
    tp_world_size: int,
) -> Tuple[int, int]:
    """Restore TP-sharded global shape from a local 2D shard shape."""
    if not config.has_tp_shard or tp_world_size <= 1:
        return m_local, n_local
    if config.tp_shard_dim == 0:
        return m_local * tp_world_size, n_local
    if config.tp_shard_dim == 1:
        return m_local, n_local * tp_world_size
    return m_local, n_local


def resolve_q_state_layout(
    m_local: int,
    n_local: int,
    config: DionParamConfig,
    *,
    tp_world_size: int,
    tp_rank: int = 0,
    use_q_unshard: bool,
    global_shape: Optional[Tuple[int, int]] = None,
    rank_fraction: float = 1.0,
    rank_multiple_of: int = 1,
) -> DionQLayout:
    """Resolve global/local Q layout and rank dimensions for a 2D Dion param."""
    if global_shape is None:
        m_global, n_global = restore_tp_shape(
            m_local,
            n_local,
            config,
            tp_world_size=tp_world_size,
        )
    else:
        m_global, n_global = global_shape

    q_base_local = m_local if config.is_transposed else n_local
    q_base_global = m_global if config.is_transposed else n_global

    r_global = rank_fraction * min(m_global, n_global)
    r_global = rank_multiple_of * math.ceil(r_global / rank_multiple_of)
    r_global = min(r_global, m_global, n_global)
    r_global = max(1, int(r_global))

    if use_q_unshard:
        r_start, r_end = _split_range(r_global, tp_world_size, tp_rank)
        r_local = r_end - r_start
    else:
        r_local = r_global
    if q_base_local <= 0 or r_local <= 0:
        raise RuntimeError(
            "[DION_EMPTY_Q_SHARD] "
            f"local_shape=({m_local}, {n_local}) global_shape=({m_global}, {n_global}) "
            f"q_base_local={q_base_local} r_global={r_global} "
            f"r_local={r_local} tp_world_size={tp_world_size} tp_rank={tp_rank}"
        )

    q_local_layout = (
        "shard(0)" if bool(getattr(config, "use_fs_shard", False)) else "replicate",
        "shard(1)" if use_q_unshard and tp_world_size > 1 else "replicate",
    )
    return DionQLayout(
        q_global_shape=(int(q_base_global), int(r_global)),
        q_local_shape=(int(q_base_local), int(r_local)),
        q_gathered_shape=(int(q_base_local), int(r_global)),
        q_base_global=int(q_base_global),
        q_base_local=int(q_base_local),
        r_global=int(r_global),
        r_local=int(r_local),
        q_local_layout=q_local_layout,
        q_gathered_layout=(q_local_layout[0], "replicate"),
    )


def should_use_low_rank_sync(
    *,
    global_shape: Tuple[int, int],
    r_global: int,
    rank_fraction: float,
) -> bool:
    """Return whether the low-rank replica-sync path is cheaper than dense sync."""
    m_global, n_global = int(global_shape[0]), int(global_shape[1])
    if rank_fraction >= 1.0:
        return False
    return (m_global + n_global) * int(r_global) < m_global * n_global


def q_seed_from_param_key(
    *,
    base_seed: int,
    dist_meta: Optional[DionDistMeta],
    q_global_shape: Tuple[int, int],
    is_transposed: bool,
) -> int:
    """Return the topology-invariant Q-init seed for one Dion parameter."""
    param_uid = getattr(dist_meta, "param_uid", None) if dist_meta is not None else None
    param_name = getattr(dist_meta, "param_name", "") if dist_meta is not None else ""
    if param_uid is None and not param_name:
        raise RuntimeError(
            "[DION_Q_INIT_SEED_ID_MISSING] Dion Q init requires param_uid or param_name"
        )

    seed_key = repr(
        (
            "dion_q_init",
            int(base_seed),
            param_uid if param_uid is not None else param_name,
            tuple(int(dim) for dim in q_global_shape),
            bool(is_transposed),
        )
    ).encode("utf-8")
    return int.from_bytes(
        hashlib.blake2b(seed_key, digest_size=8).digest(),
        "little",
    ) & ((1 << 63) - 1)


def build_param_config(
    *,
    param_ndim: int,
    local_shape: Optional[Tuple[int, int]],
    dist_meta: Optional[DionDistMeta],
    use_low_rank_sync: bool,
    r_global_override: Optional[int],
    rank_fraction_default: float,
    rank_multiple_of_default: int,
    tp_world_size: int,
    tp_active: bool,
) -> DionParamConfig:
    """Build one Dion parameter config from explicit metadata and runtime inputs."""
    config = DionParamConfig()
    config.has_fs_shard = False
    config.use_fs_shard = False
    config.has_tp_shard = False
    config.use_tp_shard = False

    if (
        dist_meta is not None
        and dist_meta.global_shape is not None
        and len(dist_meta.global_shape) == 2
        and bool(dist_meta.is_dion_param)
        and param_ndim == 2
        and local_shape is not None
    ):
        m_local, n_local = local_shape

        tp_shard_dim = getattr(dist_meta, "tp_shard_dim", -1)
        if tp_shard_dim in (0, 1):
            config.has_tp_shard = True
            config.use_tp_shard = bool(tp_active)
            config.tp_shard_dim = tp_shard_dim

        fs_shard_dim = getattr(dist_meta, "fs_shard_dim", -1)
        if fs_shard_dim in (0, 1):
            config.has_fs_shard = True
            config.use_fs_shard = getattr(dist_meta, "fs_world_size", 1) > 1
            config.fs_shard_dim = fs_shard_dim

        inner = config.tp_shard_dim
        outer = config.fs_shard_dim
        config.is_transposed = m_local < n_local
        if inner == 0 or outer == 1:
            config.is_transposed = False
        if outer == 0 or inner == 1:
            config.is_transposed = True

        if use_low_rank_sync:
            global_shape = get_global_shape(dist_meta, m_local, n_local)
            layout = resolve_q_state_layout(
                m_local,
                n_local,
                config,
                tp_world_size=tp_world_size,
                use_q_unshard=bool(config.has_tp_shard and tp_active),
                global_shape=global_shape,
                rank_fraction=dist_meta.rank_fraction,
                rank_multiple_of=rank_multiple_of_default,
            )
            r_global = (
                int(r_global_override)
                if r_global_override is not None
                else int(layout.r_global)
            )
            m_global, n_global = global_shape
            config.use_low_rank_sync = should_use_low_rank_sync(
                global_shape=(m_global, n_global),
                r_global=r_global,
                rank_fraction=dist_meta.rank_fraction,
            )
        else:
            config.use_low_rank_sync = False

        return config

    if param_ndim == 2 and local_shape is not None:
        m_local, n_local = local_shape
        config.is_transposed = m_local < n_local
        if use_low_rank_sync:
            layout = resolve_q_state_layout(
                m_local,
                n_local,
                config,
                tp_world_size=tp_world_size,
                use_q_unshard=False,
                global_shape=(m_local, n_local),
                rank_fraction=rank_fraction_default,
                rank_multiple_of=rank_multiple_of_default,
            )
            r_global = (
                int(r_global_override)
                if r_global_override is not None
                else int(layout.r_global)
            )
            m_global, n_global = m_local, n_local
            config.use_low_rank_sync = should_use_low_rank_sync(
                global_shape=(m_global, n_global),
                r_global=r_global,
                rank_fraction=rank_fraction_default,
            )
        else:
            config.use_low_rank_sync = False

    return config


def is_p_fs_sharded(config: DionParamConfig) -> bool:
    """Return whether P is FS-sharded from config-local topology only."""
    return bool(config.use_fs_shard) and (
        (not config.is_transposed and config.fs_shard_dim == 0)
        or (config.is_transposed and config.fs_shard_dim == 1)
    )


def is_tp_active(config: DionParamConfig) -> bool:
    """Return whether TP sharding is configured and active."""
    return bool(getattr(config, "use_tp_shard", False))


def is_fs_active(config: DionParamConfig) -> bool:
    """Return whether FS sharding is configured and active."""
    return bool(config.has_fs_shard and getattr(config, "use_fs_shard", False))


def use_q_unshard(config: DionParamConfig) -> bool:
    """Return whether STEP 2 must all-gather Q across TP ranks."""
    return is_tp_active(config)


def is_fs_without_tp(config: DionParamConfig) -> bool:
    """Return whether the config follows dion_reference.py fs-only semantics."""
    return is_fs_active(config) and not is_tp_active(config)


def should_reduce_p_over_fs(config: DionParamConfig) -> bool:
    """Return whether STEP 3.5 must reduce P across FS shards."""
    return bool(config.use_fs_shard) and (
        (not config.is_transposed and config.fs_shard_dim == 1)
        or (config.is_transposed and config.fs_shard_dim == 0)
    )


def is_p_tp_sharded(
    config: DionParamConfig,
    *,
    tp_active: bool,
) -> bool:
    """Return whether P is TP-sharded from config-local topology plus explicit TP activity."""
    return bool(config.has_tp_shard and tp_active) and (
        (not config.is_transposed and config.tp_shard_dim == 0)
        or (config.is_transposed and config.tp_shard_dim == 1)
    )


def should_reduce_r_over_tp(
    config: DionParamConfig,
    *,
    tp_active: bool,
) -> bool:
    """Return whether STEP 5 must all-reduce R across TP ranks."""
    return is_p_tp_sharded(config, tp_active=tp_active)


def init_q_state(
    *,
    param: Tensor,
    mixed_precision_config: DionMixedPrecisionConfig,
    config: DionParamConfig,
    dist_meta: Optional[DionDistMeta],
    q_layout,
    q_seed: Optional[int],
    tp_world_size: int,
    tp_rank: int,
    use_q_unshard: bool,
) -> Tensor:
    """Initialize one local Q state shard from explicit Q layout metadata."""
    if q_seed is None:
        raise RuntimeError(
            "[DION_MISSING_Q_INIT_SEED] "
            f"param_uid={getattr(dist_meta, 'param_uid', None)} "
            f"param_name={getattr(dist_meta, 'param_name', '') if dist_meta is not None else ''}"
        )

    q_global_shape = tuple(int(dim) for dim in (q_layout.q_global_shape or ()))
    q_local_shape = tuple(int(dim) for dim in (q_layout.q_local_shape or ()))
    if len(q_global_shape) != 2 or len(q_local_shape) != 2:
        raise RuntimeError(
            "[DION_INVALID_Q_LAYOUT] Q init requires exact q_global_shape and q_local_shape "
            f"param_uid={getattr(dist_meta, 'param_uid', None)} "
            f"param_name={getattr(dist_meta, 'param_name', '') if dist_meta is not None else ''}"
        )

    q_base_local = int(q_layout.q_base_local)
    r_local = int(q_layout.r_local)
    if q_base_local <= 0 or r_local <= 0:
        raise RuntimeError(
            "[DION_EMPTY_Q_SHARD] Q init requires non-empty local Q dimensions "
            f"param_uid={getattr(dist_meta, 'param_uid', None)} "
            f"param_name={getattr(dist_meta, 'param_name', '') if dist_meta is not None else ''} "
            f"q_base_local={q_base_local} r_local={r_local}"
        )

    if config.has_fs_shard and getattr(dist_meta, "fs_world_size", 1) > 1:
        fs_rank = int(getattr(dist_meta, "fs_rank", -1))
        fs_start = int(getattr(dist_meta, "fs_start_idx", -1))
        fs_end = int(getattr(dist_meta, "fs_end_idx", -1))
        if fs_rank < 0:
            raise RuntimeError(
                "[DION_MISSING_FS_RANK_FOR_Q_INIT] "
                f"param_uid={getattr(dist_meta, 'param_uid', None)} "
                f"param_name={getattr(dist_meta, 'param_name', '') if dist_meta is not None else ''}"
            )
        if fs_start < 0 or fs_end < fs_start:
            raise RuntimeError(
                "[DION_MISSING_FS_RANGE_FOR_Q_INIT] "
                f"param_uid={getattr(dist_meta, 'param_uid', None)} "
                f"param_name={getattr(dist_meta, 'param_name', '') if dist_meta is not None else ''} "
                f"fs_start_idx={fs_start} fs_end_idx={fs_end}"
            )
    else:
        fs_rank = 0
        fs_start = 0
        fs_end = int(q_local_shape[0])

    q_tp_world_size = int(tp_world_size) if use_q_unshard and int(tp_world_size) > 1 else 1
    q_tp_rank = int(tp_rank) if q_tp_world_size > 1 else 0
    tp_start, tp_end = _split_range(q_global_shape[1], q_tp_world_size, q_tp_rank)
    if int(tp_end - tp_start) != int(q_local_shape[1]):
        raise RuntimeError(
            "[DION_Q_TP_RANGE_MISMATCH] "
            f"param_uid={getattr(dist_meta, 'param_uid', None)} "
            f"param_name={getattr(dist_meta, 'param_name', '') if dist_meta is not None else ''} "
            f"q_global_shape={q_global_shape} q_local_shape={q_local_shape} "
            f"tp_world_size={tp_world_size} tp_rank={q_tp_rank} "
            f"tp_range=({tp_start}, {tp_end})"
        )

    q_dtype = str_to_dtype(mixed_precision_config.q_dtype)
    if q_dtype is None:
        q_dtype = param.dtype

    q_state = _normal_q_submatrix(
        q_global_shape=q_global_shape,
        row_start=fs_start,
        row_end=fs_end,
        col_start=tp_start,
        col_end=tp_end,
        seed=int(q_seed),
        device=param.device,
        dtype=q_dtype,
    ).contiguous()

    if tuple(q_state.shape) != q_local_shape:
        raise RuntimeError(
            "[DION_Q_INIT_SHAPE_MISMATCH] "
            f"param_uid={getattr(dist_meta, 'param_uid', None)} "
            f"param_name={getattr(dist_meta, 'param_name', '') if dist_meta is not None else ''} "
            f"expected={q_local_shape} got={tuple(q_state.shape)}"
        )
    return q_state


def init_param_state(
    *,
    param: Tensor,
    state: Dict[str, Any],
    optim_group: Dict[str, Any],
    mixed_precision_config: DionMixedPrecisionConfig,
    config: DionParamConfig,
    dist_meta: Optional[DionDistMeta],
    is_dion_eligible: bool,
    local_shape: Optional[Tuple[int, int]],
    rank_fraction_default: float,
    rank_multiple_of_default: int,
    split_qkv_default: bool = False,
    split_linear_default: bool = False,
    q_init: Optional[DionQInit] = None,
) -> None:
    """Initialize optimizer state for one param using adapter-authored metadata."""
    momentum_dtype = str_to_dtype(mixed_precision_config.momentum_dtype)
    if momentum_dtype is None:
        momentum_dtype = param.dtype
    state["momentum"] = torch.zeros_like(param, dtype=momentum_dtype)

    algorithm = optim_group.get("algorithm", "dion")
    if algorithm != "dion" or not is_dion_eligible or local_shape is None:
        return
    split_shapes = get_qkv_split_shapes_from_dist_meta(dist_meta)
    if bool(split_qkv_default) and split_shapes is not None:
        state["qkv_split_qkv"] = True
        state["qkv_split_shapes"] = split_shapes
        return
    linear_split_rows = get_linear_split_rows_from_dist_meta(dist_meta)
    if bool(split_linear_default) and linear_split_rows is not None:
        state["linear_split_linear"] = True
        state["linear_split_rows"] = linear_split_rows
        return

    m_local, n_local = local_shape

    if q_init is None:
        raise RuntimeError(
            "[DION_MISSING_Q_INIT] "
            f"param_uid={getattr(dist_meta, 'param_uid', None)} "
            f"param_name={getattr(dist_meta, 'param_name', '') if dist_meta is not None else ''}"
        )
    tp_world_size = int(q_init.tp_world_size)
    tp_rank = int(q_init.tp_rank)
    use_q_unshard = bool(q_init.use_q_unshard)
    q_seed = q_init.q_seed
    broadcast_q = q_init.broadcast_q
    q_layout = q_init.q_layout
    if broadcast_q is None:
        raise RuntimeError(
            "[DION_MISSING_Q_INIT_BROADCAST] "
            f"param_uid={getattr(dist_meta, 'param_uid', None)} "
            f"param_name={getattr(dist_meta, 'param_name', '') if dist_meta is not None else ''}"
        )
    if q_layout is None:
        raise RuntimeError(
            "[DION_MISSING_Q_LAYOUT] "
            f"param_uid={getattr(dist_meta, 'param_uid', None)} "
            f"param_name={getattr(dist_meta, 'param_name', '') if dist_meta is not None else ''}"
        )

    m_global, n_global = get_global_shape(dist_meta, m_local, n_local)
    q_local_shape = tuple(int(dim) for dim in (q_layout.q_local_shape or ()))
    q_gathered_shape = tuple(
        int(dim) for dim in (q_layout.q_gathered_shape or ())
    )
    q_global_shape = tuple(int(dim) for dim in (q_layout.q_global_shape or ()))
    q_local_layout = tuple(str(axis_name) for axis_name in q_layout.q_local_layout)
    q_gathered_layout = tuple(
        str(axis_name) for axis_name in q_layout.q_gathered_layout
    )
    if (
        len(q_local_shape) != 2
        or len(q_gathered_shape) != 2
        or len(q_global_shape) != 2
        or len(q_local_layout) != 2
        or len(q_gathered_layout) != 2
    ):
        raise RuntimeError(
            "[DION_INVALID_Q_LAYOUT] "
            f"param_uid={getattr(dist_meta, 'param_uid', None)} "
            f"param_name={getattr(dist_meta, 'param_name', '') if dist_meta is not None else ''}"
        )
    if use_q_unshard and q_gathered_layout[1] != "replicate":
        raise RuntimeError(
            "[DION_INVALID_Q_INNER_UNSHARDED_LAYOUT] "
            f"param_uid={getattr(dist_meta, 'param_uid', None)} "
            f"param_name={getattr(dist_meta, 'param_name', '') if dist_meta is not None else ''} "
            f"placements={q_gathered_layout}"
        )

    q_state = init_q_state(
        param=param,
        mixed_precision_config=mixed_precision_config,
        config=config,
        dist_meta=dist_meta,
        q_layout=q_layout,
        q_seed=q_seed,
        tp_world_size=tp_world_size,
        tp_rank=tp_rank,
        use_q_unshard=use_q_unshard,
    )
    if tuple(q_state.shape) != q_local_shape:
        raise RuntimeError(
            f"Dion Q init shape mismatch for {getattr(dist_meta, 'param_name', '')}: "
            f"expected {q_local_shape}, got {tuple(q_state.shape)}"
        )

    broadcast_q(q_state)

    state["Q"] = q_state
    state["_needs_state_replica_q_sync"] = True
    state["r"] = int(q_layout.r_global)
    state["local_shape"] = (m_local, n_local)
    state["global_shape"] = (m_global, n_global)

    per_expert = (
        getattr(dist_meta, "per_expert_global_shape", None) if dist_meta is not None else None
    )
    if per_expert is not None:
        state["per_expert_global_shape"] = per_expert
