"""Dense RP grad-reduction cache helpers for Matrix."""

from __future__ import annotations

from typing import Iterable, Literal

import torch


DENSE_GRAD_REDUCTION_CACHE = "_matrix_dense_grad_reduction_cache"
DenseGradCacheState = Literal["missing", "match", "mismatch"]
_ENTRY_INDEX_BY_LIST_ID: dict[int, dict[tuple, list[tuple[int, dict]]]] = {}


def tensor_region(tensor: torch.Tensor):
    if int(tensor.numel()) <= 0:
        return None
    start = int(tensor.storage_offset())
    end = start
    for size, stride in zip(tensor.shape, tensor.stride()):
        extent = (int(size) - 1) * int(stride)
        if extent < 0:
            start += extent
        else:
            end += extent
    return {
        "storage_data_ptr": int(tensor.untyped_storage().data_ptr()),
        "start": int(start),
        "end": int(end) + 1,
        "dtype": tensor.dtype,
        "device": tensor.device,
    }


def region_contains(entry, region) -> bool:
    return (
        entry["storage_data_ptr"] == region["storage_data_ptr"]
        and entry["dtype"] == region["dtype"]
        and entry["device"] == region["device"]
        and int(entry["start"]) <= int(region["start"])
        and int(region["end"]) <= int(entry["end"])
    )


def _region_key(region) -> tuple:
    return (region["storage_data_ptr"], region["dtype"], region["device"])


def _rebuild_index(cache) -> None:
    index = {}
    entries = cache.get("entries", [])
    for entry_index, entry in enumerate(entries):
        index.setdefault(_region_key(entry), []).append((entry_index, entry))
    cache["index"] = index
    _ENTRY_INDEX_BY_LIST_ID[id(entries)] = index


def dense_cache_entries(
    owner,
    before_step: int,
    *,
    create: bool,
    delete_empty: bool = False,
) -> list[dict] | None:
    cache = getattr(owner, DENSE_GRAD_REDUCTION_CACHE, None)
    if cache is None:
        if not create:
            return None
        cache = {"entries": []}
        setattr(owner, DENSE_GRAD_REDUCTION_CACHE, cache)
        _rebuild_index(cache)
        return cache["entries"]

    old_entries = cache.get("entries", [])
    _ENTRY_INDEX_BY_LIST_ID.pop(id(old_entries), None)
    entries = [
        entry
        for entry in old_entries
        if int(entry.get("before_step", -1)) == int(before_step)
    ]
    cache["entries"] = entries
    if not entries and delete_empty:
        delattr(owner, DENSE_GRAD_REDUCTION_CACHE)
        return None
    _rebuild_index(cache)
    return entries


def find_dense_grad_entry(
    entries: list[dict],
    tensor: torch.Tensor,
    *,
    replica_group,
    op,
    before_step: int,
) -> tuple[DenseGradCacheState, int | None]:
    region = tensor_region(tensor)
    if region is None:
        return "missing", None
    state: DenseGradCacheState = "missing"
    index_by_key = _ENTRY_INDEX_BY_LIST_ID.get(id(entries))
    candidates = (
        index_by_key.get(_region_key(region), [])
        if index_by_key is not None
        else enumerate(entries)
    )
    for index, entry in candidates:
        if not region_contains(entry, region):
            continue
        if (
            int(entry.get("before_step", -1)) == int(before_step)
            and entry.get("group") is replica_group
            and entry.get("op") == op
        ):
            return "match", index
        state = "mismatch"
    return state, None


def dense_cache_state(
    owner,
    tensor: torch.Tensor,
    *,
    replica_group,
    op,
    before_step: int,
) -> DenseGradCacheState:
    entries = dense_cache_entries(owner, before_step, create=False)
    if entries is None:
        return "missing"
    state, _ = find_dense_grad_entry(
        entries,
        tensor,
        replica_group=replica_group,
        op=op,
        before_step=before_step,
    )
    return state


def clear_dense_grad_cache(owner) -> None:
    cache = getattr(owner, DENSE_GRAD_REDUCTION_CACHE, None)
    if cache is None:
        return
    entries = cache.get("entries", [])
    _ENTRY_INDEX_BY_LIST_ID.pop(id(entries), None)
    delattr(owner, DENSE_GRAD_REDUCTION_CACHE)


def mark_dense_grad_reduced(
    owner,
    tensor: torch.Tensor,
    *,
    replica_group,
    op,
    before_step: int,
) -> None:
    region = tensor_region(tensor)
    if region is None:
        return
    entries = dense_cache_entries(owner, before_step, create=True)
    entry = dict(region)
    entry["group"] = replica_group
    entry["op"] = op
    entry["before_step"] = int(before_step)
    entries.append(entry)
    cache = getattr(owner, DENSE_GRAD_REDUCTION_CACHE, None)
    if cache is not None:
        index = cache.setdefault("index", {})
        index.setdefault(_region_key(entry), []).append(
            (len(entries) - 1, entry)
        )
        _ENTRY_INDEX_BY_LIST_ID[id(entries)] = index


def delete_dense_grad_entries(owner, indices: Iterable[int]) -> list[dict]:
    cache = getattr(owner, DENSE_GRAD_REDUCTION_CACHE, None)
    if cache is None:
        return []
    entries = cache.get("entries", [])
    for index in sorted(set(int(index) for index in indices), reverse=True):
        del entries[index]
    if not entries:
        delattr(owner, DENSE_GRAD_REDUCTION_CACHE)
        _ENTRY_INDEX_BY_LIST_ID.pop(id(entries), None)
        return []
    cache["entries"] = entries
    _rebuild_index(cache)
    return entries
