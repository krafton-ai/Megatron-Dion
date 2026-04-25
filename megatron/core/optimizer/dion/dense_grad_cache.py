"""Dense RP grad-reduction cache helpers for Dion."""

from __future__ import annotations

from typing import Iterable, Literal

import torch


DENSE_GRAD_REDUCTION_CACHE = "_dion_dense_grad_reduction_cache"
DenseGradCacheState = Literal["missing", "match", "mismatch"]


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
        return cache["entries"]

    entries = [
        entry
        for entry in cache.get("entries", [])
        if int(entry.get("before_step", -1)) == int(before_step)
    ]
    cache["entries"] = entries
    if not entries and delete_empty:
        delattr(owner, DENSE_GRAD_REDUCTION_CACHE)
        return None
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
    for index, entry in enumerate(entries):
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


def delete_dense_grad_entries(owner, indices: Iterable[int]) -> list[dict]:
    cache = getattr(owner, DENSE_GRAD_REDUCTION_CACHE, None)
    if cache is None:
        return []
    entries = cache.get("entries", [])
    for index in sorted(set(int(index) for index in indices), reverse=True):
        del entries[index]
    if not entries:
        delattr(owner, DENSE_GRAD_REDUCTION_CACHE)
        return []
    cache["entries"] = entries
    return entries
