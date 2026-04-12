"""Dion optimizer batching helpers aligned with the reference implementation."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterator, List, Sequence

import torch
from torch import Tensor

from .types import DionParamConfig
from .utils import str_to_dtype


def _normalize_shard_dim(dim, has_axis: bool) -> int:
    if dim is None and not has_axis:
        return -1
    if dim is None:
        raise RuntimeError(
            "[Dion] missing shard tensor dim for active sharded axis in batch key construction"
        )
    return int(dim)


def build_batch_key(shape, cfg, dtype: torch.dtype) -> tuple:
    """Build the canonical Dion batch key used by local/distributed runtimes."""
    if len(shape) == 2:
        resolved_shape = (int(shape[0]), int(shape[1]))
    else:
        resolved_shape = tuple(int(dim) for dim in shape)
    return (
        resolved_shape,
        bool(cfg.has_fs_shard),
        bool(getattr(cfg, "use_fs_shard", cfg.has_fs_shard)),
        bool(cfg.has_tp_shard),
        bool(getattr(cfg, "use_tp_shard", cfg.has_tp_shard)),
        bool(cfg.is_transposed),
        bool(cfg.compressed_all_reduce),
        _normalize_shard_dim(cfg.tp_shard_dim, cfg.has_tp_shard),
        _normalize_shard_dim(cfg.fs_shard_dim, cfg.has_fs_shard),
        dtype,
    )


def config_from_batch_key(batch_key: tuple) -> DionParamConfig:
    """Decode a synchronized batch key back into its Dion parameter config."""
    return DionParamConfig(
        has_fs_shard=bool(batch_key[1]),
        use_fs_shard=bool(batch_key[2]),
        has_tp_shard=bool(batch_key[3]),
        use_tp_shard=bool(batch_key[4]),
        is_transposed=bool(batch_key[5]),
        compressed_all_reduce=bool(batch_key[6]),
        tp_shard_dim=batch_key[7] if batch_key[7] != -1 else None,
        fs_shard_dim=batch_key[8] if batch_key[8] != -1 else None,
    )


def serialize_batch_key(batch_key: tuple) -> tuple:
    """Serialize a canonical batch key for cross-rank object collectives."""
    return (
        tuple(batch_key[0]) if isinstance(batch_key[0], (tuple, list)) else batch_key[0],
        bool(batch_key[1]),
        bool(batch_key[2]),
        bool(batch_key[3]),
        bool(batch_key[4]),
        bool(batch_key[5]),
        bool(batch_key[6]),
        int(batch_key[7]) if batch_key[7] is not None else -1,
        int(batch_key[8]) if batch_key[8] is not None else -1,
        str(batch_key[9]),
    )


def deserialize_batch_key(serialized_batch_key: tuple) -> tuple:
    """Deserialize a synchronized batch-key payload back into runtime form."""
    return (
        serialized_batch_key[0],
        serialized_batch_key[1],
        serialized_batch_key[2],
        serialized_batch_key[3],
        serialized_batch_key[4],
        serialized_batch_key[5],
        serialized_batch_key[6],
        serialized_batch_key[7],
        serialized_batch_key[8],
        str_to_dtype(serialized_batch_key[9]),
    )


def canonicalize_gathered_sequences(
    gathered_sequences: Sequence[Sequence[Any]],
) -> tuple[tuple[Any, ...], list[int]]:
    """Return the rank0 canonical sequence and ranks with mismatched multiset content."""
    canonical_items = tuple(gathered_sequences[0]) if gathered_sequences else tuple()
    canonical_counter = Counter(canonical_items)
    mismatch_ranks = [
        idx
        for idx, rank_items in enumerate(gathered_sequences)
        if Counter(rank_items) != canonical_counter
    ]
    return canonical_items, mismatch_ranks


def batch_member_id(dist_meta) -> tuple[str, Any]:
    """Return the logical identifier used for canonical batch ordering."""
    if dist_meta is None:
        return "", None
    return getattr(dist_meta, "param_name", "") or "", getattr(dist_meta, "param_uid", None)


def canonical_reorder_indices(
    local_items: Sequence[Any],
    canonical_items: Sequence[Any],
    *,
    error_tag: str,
) -> List[int]:
    """Return indices that reorder local items into canonical order."""
    local_positions: dict[Any, list[int]] = {}
    for idx, item in enumerate(local_items):
        local_positions.setdefault(item, []).append(idx)

    reorder_indices: List[int] = []
    for item in canonical_items:
        positions = local_positions.get(item)
        if not positions:
            raise RuntimeError(
                f"[{error_tag}] missing_item={item} "
                f"local_items={list(local_items)} canonical_items={list(canonical_items)}"
            )
        reorder_indices.append(positions.pop(0))

    leftover_items = [item for item, positions in local_positions.items() if positions]
    if leftover_items:
        raise RuntimeError(
            f"[{error_tag}] leftover_items={leftover_items} "
            f"local_items={list(local_items)} canonical_items={list(canonical_items)}"
        )

    return reorder_indices


def unique_preserve_order(items: Sequence[Any]) -> list[Any]:
    """Deduplicate while preserving first-seen order."""
    return list(dict.fromkeys(items))


def group_items_by_batch_key(
    items: Sequence[Any],
    batch_keys: Sequence[tuple],
) -> list[tuple[tuple, list[Any]]]:
    """Group items by canonical batch key while preserving first-seen key order."""
    if len(items) != len(batch_keys):
        raise RuntimeError(
            f"[Dion] item/key length mismatch for batching: items={len(items)} keys={len(batch_keys)}"
        )
    grouped: dict[tuple, list[Any]] = {}
    for item, batch_key in zip(items, batch_keys):
        grouped.setdefault(batch_key, []).append(item)
    return list(grouped.items())


def pad_batch(batch: List[Tensor], batch_size: int) -> List[Tensor]:
    """Pad with inert zero tensors so partial distributed batches remain numerically stable."""
    assert len(batch) > 0
    assert len(batch) <= batch_size
    while len(batch) < batch_size:
        batch.append(torch.zeros_like(batch[0]))
    return batch
