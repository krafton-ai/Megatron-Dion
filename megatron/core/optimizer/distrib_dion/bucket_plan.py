"""Bucket-level pack/unpack plan types for Dion FS all-gather."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Iterator, List, Tuple

import torch
import torch.distributed as dist

from ..distrib_optimizer import Range
from ...fp8_utils import is_float8tensor
from .fs_layout import compute_fs_shard_range

logger = logging.getLogger(__name__)


def check_bucket_dion_flags_(
    *,
    bucket_index: int,
    dp_group,
    param_map,
    param_to_name=None,
) -> None:
    """Validate per-param Dion classification is identical across the DP group.

    PP>1 hangs can happen when ranks in the same stage build different
    `(param_name, is_dion)` tables for the same bucket. Catch that at init time
    before any optimizer task scheduling happens.
    """
    local_flags = []
    for param in param_map.keys():
        if param_to_name is not None:
            param_name = param_to_name.get(param, "")
        else:
            param_name = ""
        local_flags.append(
            {
                "name": param_name or f"<id_{id(param)}>",
                "shape": tuple(param.shape),
                "ndim": int(param.ndim),
                "is_dion": bool(getattr(param, "is_dion_param", False)),
                "use_dion": getattr(param, "use_dion", None),
                "is_fp8": bool(is_float8tensor(param)),
            }
        )

    gathered = [None] * dist.get_world_size(dp_group)
    dist.all_gather_object(gathered, local_flags, group=dp_group)

    reference = gathered[0]
    if all(flags == reference for flags in gathered[1:]):
        return

    logger.error("[Dion] Bucket %s has inconsistent Dion classification across DP ranks", bucket_index)
    for rank_idx, flags in enumerate(gathered):
        if flags == reference:
            continue
        logger.error("  DP rank %s classification differs from DP rank 0", rank_idx)
        ref_map = {entry["name"]: entry for entry in reference}
        cur_map = {entry["name"]: entry for entry in flags}
        all_names = sorted(set(ref_map) | set(cur_map))
        for name in all_names:
            ref_entry = ref_map.get(name)
            cur_entry = cur_map.get(name)
            if ref_entry != cur_entry:
                logger.error("    name=%s ref=%s cur=%s", name, ref_entry, cur_entry)
    raise RuntimeError(
        f"[Dion] Bucket {bucket_index}: per-rank Dion classification mismatch; "
        "same PP/DP stage must build identical (param_name, is_dion) tables"
    )


@dataclass(frozen=True)
class DionLayoutEntry:
    """Single entry in a Dion FS all-gather pack plan."""

    param: torch.nn.Parameter
    global_shape: Tuple[int, int]
    local_shape: Tuple[int, int]
    fs_split_dim: int
    fs_rank: int
    start_idx: int
    end_idx: int
    size_per_rank: int
    segment_size: int
    pack_offset: int

    @property
    def numel(self) -> int:
        return self.local_shape[0] * self.local_shape[1]

    # Dict-like access for backward compatibility during migration.
    def __getitem__(self, key: str):
        return getattr(self, key)

    def get(self, key: str, default=None):
        return getattr(self, key, default)

    def copy(self) -> dict:
        # Some callsites expect a mutable mapping; keep the shape stable.
        return {
            "param": self.param,
            "global_shape": self.global_shape,
            "local_shape": self.local_shape,
            "fs_split_dim": self.fs_split_dim,
            "fs_rank": self.fs_rank,
            "start_idx": self.start_idx,
            "end_idx": self.end_idx,
            "size_per_rank": self.size_per_rank,
            "segment_size": self.segment_size,
            "pack_offset": self.pack_offset,
        }


@dataclass
class DionParamLayout:
    """Container for DionLayoutEntry with minimal helpers.

    This is intentionally small and keeps list-like behavior for existing code.
    """

    entries: List[DionLayoutEntry] = field(default_factory=list)

    def __iter__(self) -> Iterator[DionLayoutEntry]:
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    def __bool__(self) -> bool:
        return len(self.entries) > 0

    def append(self, entry: DionLayoutEntry) -> None:
        self.entries.append(entry)

    def extend(self, entries) -> None:
        if isinstance(entries, DionParamLayout):
            self.entries.extend(entries.entries)
        else:
            self.entries.extend(entries)


@dataclass(frozen=True)
class NonDionLayoutEntry:
    """Single entry in a mixed-bucket non-Dion shard plan."""

    param: torch.nn.Parameter
    full_start: int
    full_end: int
    pack_offset: int
    input_offset: int
    segment_size: int
    param_numel: int
    section_start: int
    section_end: int
    local_start: int = 0
    local_end: int = 0
    local_bucket_start: int = 0
    local_bucket_end: int = 0
    rank_bucket_ranges: Tuple[Tuple[int, int], ...] = ()
    rank_param_ranges: Tuple[Tuple[int, int], ...] = ()

    def __getitem__(self, key: str):
        return getattr(self, key)

    def get(self, key: str, default=None):
        return getattr(self, key, default)

    def copy(self) -> dict:
        return {
            "param": self.param,
            "full_start": self.full_start,
            "full_end": self.full_end,
            "pack_offset": self.pack_offset,
            "input_offset": self.input_offset,
            "segment_size": self.segment_size,
            "param_numel": self.param_numel,
            "section_start": self.section_start,
            "section_end": self.section_end,
            "local_start": self.local_start,
            "local_end": self.local_end,
            "local_bucket_start": self.local_bucket_start,
            "local_bucket_end": self.local_bucket_end,
            "rank_bucket_ranges": self.rank_bucket_ranges,
            "rank_param_ranges": self.rank_param_ranges,
        }


def build_stock_bucket_non_dion_plan_(
    *,
    bucket,
    params,
    dp_size: int,
    dp_rank: int,
) -> tuple[list[NonDionLayoutEntry], dict, int, int]:
    """Build a mixed non-Dion plan that matches stock DO local-shard layout exactly.

    Stock Megatron-Core DO semantics:
    - pre-RS input is the full bucket.grad_data flat buffer
    - reduce-scatter output is one local shard of that same full buffer
    - inter-instance all-reduce happens on that same local shard

    For mixed buckets we preserve that contract by storing non-Dion grads/params in
    the same flat bucket coordinates and masking out Dion-owned regions rather than
    compacting the non-Dion section.
    """
    if dp_size <= 0:
        raise RuntimeError(f"Invalid dp_size for stock mixed non-Dion plan: {dp_size}")
    bucket_size = int(bucket.grad_data.numel())
    if bucket_size % dp_size != 0:
        raise RuntimeError(
            f"Mixed bucket size must be divisible by dp_size: bucket_size={bucket_size} dp_size={dp_size}"
        )
    if not hasattr(bucket, "param_to_index") or bucket.param_to_index is None:
        raise RuntimeError(
            "[Dion] mixed non-Dion stock plan requires bucket.param_to_index for canonical bucket offsets."
        )

    local_shard_size = bucket_size // dp_size
    local_bucket_abs_start = dp_rank * local_shard_size
    local_bucket_abs_end = local_bucket_abs_start + local_shard_size

    pack_plan = []
    param_ranges = {}
    for param in params:
        if param not in bucket.param_to_index:
            raise RuntimeError(
                "[Dion] mixed non-Dion stock plan could not find canonical bucket offset "
                f"for param shape={tuple(param.shape)}"
            )
        full_start, full_end = bucket.param_to_index[param]
        full_start = int(full_start)
        full_end = int(full_end)
        param_numel = full_end - full_start

        local_abs_start = max(full_start, local_bucket_abs_start)
        local_abs_end = min(full_end, local_bucket_abs_end)
        local_size = max(0, local_abs_end - local_abs_start)
        if local_size == 0:
            local_param_start = 0
            local_param_end = 0
            local_bucket_start = 0
            local_bucket_end = 0
        else:
            local_param_start = local_abs_start - full_start
            local_param_end = local_param_start + local_size
            local_bucket_start = local_abs_start - local_bucket_abs_start
            local_bucket_end = local_bucket_start + local_size

        rank_bucket_ranges = []
        rank_param_ranges = []
        for rank_j in range(dp_size):
            shard_abs_start = rank_j * local_shard_size
            shard_abs_end = shard_abs_start + local_shard_size
            shard_param_abs_start = max(full_start, shard_abs_start)
            shard_param_abs_end = min(full_end, shard_abs_end)
            shard_size = max(0, shard_param_abs_end - shard_param_abs_start)
            if shard_size == 0:
                rank_bucket_ranges.append((0, 0))
                rank_param_ranges.append((0, 0))
                continue
            rank_bucket_ranges.append(
                (shard_param_abs_start - shard_abs_start, shard_param_abs_end - shard_abs_start)
            )
            rank_param_ranges.append(
                (shard_param_abs_start - full_start, shard_param_abs_end - full_start)
            )

        entry = NonDionLayoutEntry(
            param=param,
            full_start=full_start,
            full_end=full_end,
            pack_offset=local_bucket_start,
            input_offset=full_start,
            segment_size=local_shard_size,
            param_numel=param_numel,
            section_start=full_start,
            section_end=full_end,
            local_start=local_param_start,
            local_end=local_param_end,
            local_bucket_start=local_bucket_start,
            local_bucket_end=local_bucket_end,
            rank_bucket_ranges=tuple(rank_bucket_ranges),
            rank_param_ranges=tuple(rank_param_ranges),
        )
        pack_plan.append(entry)
        param_ranges[param] = (full_start, full_end)

    return pack_plan, param_ranges, local_shard_size, bucket_size


def build_dion_bucket_layout_(
    *,
    param_map,
    param_to_name,
    fs_size: int,
    fs_rank: int,
    bucket_index: int,
) -> tuple[DionParamLayout, dict, int, int]:
    """Build `DionParamLayout` for Dion params and update `param_map` ranges in-place.

    This is a mechanical refactor of the Dion path in `_build_model_gbuf_range()`:
    - compute FS shard start/end and local shape for each Dion param
    - write shard metadata into `param_map[param]["dion_info"]`
    - overwrite `param_map[param]` ranges to reflect the packed FS layout

    Returns:
        layout: DionParamLayout
        shard_range: dict[param -> (start,end)] for main_grad binding
        section_size: total packed size (including padding) for Dion section
        dion_param_count: number of Dion params in this bucket
    """
    layout = DionParamLayout()
    shard_range = {}

    local_offset = 0
    pack_offset = 0

    dion_params = [
        (param, info) for param, info in param_map.items() if getattr(param, "is_dion_param", False)
    ]

    def _sort_key(param_item):
        param = param_item[0]
        param_name = ""
        try:
            param_name = param_to_name.get(param, "") if param_to_name else ""
        except Exception:
            param_name = ""
        return (tuple(param.shape), param_name)

    dion_params.sort(key=_sort_key)

    dion_param_count = 0
    for param, param_info in dion_params:
        dion_info = param_info.get("dion_info")
        if not dion_info:
            continue

        global_shape = dion_info["global_shape"]
        fs_split_dim = dion_info["fs_split_dim"]
        m, n = param.shape  # TP-sharded shape

        split_size = m if fs_split_dim == 0 else n
        start_idx, end_idx = compute_fs_shard_range(split_size, fs_size, fs_rank)
        local_split_size = end_idx - start_idx
        size_per_rank = math.ceil(split_size / fs_size)

        if fs_split_dim == 0:
            local_shape = (local_split_size, n)
            local_size = local_split_size * n
        else:
            local_shape = (m, local_split_size)
            local_size = m * local_split_size

        segment_size = size_per_rank * (n if fs_split_dim == 0 else m)

        stock_param_range = param_info["param"]
        updated_dion_info = dict(param_map[param].get("dion_info", {}) or {})
        updated_dion_info.update(
            {
                "shape": local_shape,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "fs_owner_ranks": tuple(range(fs_size)),
                "bucket_idx": bucket_index,
                # Preserve the parent stock-DO optimizer shard range. Dion pack/layout
                # metadata must not replace the canonical local-shard contract.
                "stock_param_start": int(stock_param_range.start),
                "stock_param_end": int(stock_param_range.end),
            }
        )
        param_map[param]["dion_info"] = updated_dion_info

        local_range = Range(pack_offset, pack_offset + local_size)
        param_map[param]["param"] = local_range
        param_map[param]["gbuf_world"] = local_range
        param_map[param]["gbuf_local"] = local_range

        shard_range[param] = (pack_offset, pack_offset + local_size)

        layout.append(
            DionLayoutEntry(
                param=param,
                global_shape=global_shape,
                local_shape=local_shape,
                fs_split_dim=fs_split_dim,
                fs_rank=fs_rank,
                start_idx=start_idx,
                end_idx=end_idx,
                size_per_rank=size_per_rank,
                segment_size=segment_size,
                pack_offset=pack_offset,
            )
        )

        local_offset += segment_size
        pack_offset += segment_size
        dion_param_count += 1

    section_size = local_offset
    return layout, shard_range, section_size, dion_param_count


def select_non_dion_bucket_params_(
    *,
    bucket_params,
    dion_param_layout,
    param_map=None,
):
    """Return non-Dion params in main-grad binding order for a bucket."""
    dion_param_ids = {id(entry["param"]) for entry in dion_param_layout or []}
    ordered_params = param_map.keys() if param_map is not None else bucket_params
    return [param for param in ordered_params if id(param) not in dion_param_ids]


def build_standard_non_dion_pack_plan_(
    *,
    params,
    param_map,
    param_index_map,
    bucket_offset: int,
    bucket_size: int,
    dion_section_size: int,
    dp_size: int,
) -> tuple[list[NonDionLayoutEntry], dict, int, int]:
    """Build mixed-bucket non-Dion plan from standard DO bucket shard ranges.

    This follows the parent DistributedOptimizer bucket partition exactly:
    - local optimizer shard sizes come from the current rank's parent `param_map`
    - per-rank reconstruction ranges are computed from the same bucket/world partition
    - communication slots use `max(local_size over ranks)` so all-gather / RS inputs stay
      fixed-size across ranks
    """
    from ..distrib_optimizer import DistributedOptimizer

    if dp_size <= 0:
        raise RuntimeError(f"Invalid dp_size for mixed non-Dion plan: {dp_size}")
    if bucket_size % dp_size != 0:
        raise RuntimeError(
            f"Standard DO bucket size must be divisible by dp_size: bucket_size={bucket_size} dp_size={dp_size}"
        )

    max_gbuf_range_size = bucket_size // dp_size
    rank_param_maps = []
    for rank_i in range(dp_size):
        gbuf_world_start = rank_i * max_gbuf_range_size
        gbuf_world_end = min(bucket_size, gbuf_world_start + max_gbuf_range_size)
        gbuf_world_range = Range(
            gbuf_world_start + bucket_offset,
            gbuf_world_end + bucket_offset,
        )
        rank_param_maps.append(
            DistributedOptimizer._build_model_gbuf_param_range_map(
                param_index_map,
                gbuf_world_range,
                bucket_offset,
            )
        )

    pack_plan = []
    param_ranges = {}
    pack_offset = 0
    input_offset = 0

    for param in params:
        local_range = param_map[param]["param"]
        rank_param_ranges = []
        segment_size = 0

        for rank_map in rank_param_maps:
            if param in rank_map:
                rank_range = rank_map[param]["param"]
                start = int(rank_range.start)
                end = int(rank_range.end)
            else:
                start = 0
                end = 0
            rank_param_ranges.append((start, end))
            segment_size = max(segment_size, end - start)

        section_start = dion_section_size + pack_offset
        section_end = section_start + segment_size

        entry = NonDionLayoutEntry(
            param=param,
            pack_offset=pack_offset,
            input_offset=input_offset,
            segment_size=segment_size,
            param_numel=param.numel(),
            section_start=section_start,
            section_end=section_end,
            local_start=int(local_range.start),
            local_end=int(local_range.end),
            rank_param_ranges=tuple(rank_param_ranges),
        )
        pack_plan.append(entry)
        param_ranges[param] = (section_start, section_end)

        pack_offset += segment_size
        input_offset += param.numel()

    shard_size_total = pack_offset
    full_grad_total = shard_size_total * dp_size
    return pack_plan, param_ranges, shard_size_total, full_grad_total


def check_mixed_bucket_ranges_(
    *,
    bucket_id: int,
    grad_buffer_numel: int,
    dion_section_size: int,
    dion_param_shard_range,
    non_dion_param_ranges,
    non_dion_shard_size: int,
    param_to_name=None,
) -> None:
    """Validate mixed-bucket Dion/non-Dion RS sections are contiguous."""
    expected_total = int(dion_section_size) + int(non_dion_shard_size)
    if expected_total > int(grad_buffer_numel):
        raise RuntimeError(
            f"bucket.grad_data too small: need {expected_total}, have {grad_buffer_numel}"
        )

    if dion_param_shard_range:
        for param, (range_start, range_end) in dion_param_shard_range.items():
            if range_start < 0 or range_end < range_start or range_end > int(dion_section_size):
                param_name = ""
                if param_to_name is not None:
                    param_name = param_to_name.get(param, "")
                raise RuntimeError(
                    f"Dion rs_range out of bounds: {param_name} range=({range_start},{range_end}) "
                    f"dion_section={int(dion_section_size)} bucket={bucket_id}"
                )

    sorted_ranges = sorted(non_dion_param_ranges.items(), key=lambda item: item[1][0])
    expected_start = int(dion_section_size)
    for param, (range_start, range_end) in sorted_ranges:
        if range_start != expected_start:
            param_name = ""
            if param_to_name is not None:
                param_name = param_to_name.get(param, "")
            raise RuntimeError(
                f"Non-Dion range hole/overlap: {param_name} got_start={range_start} "
                f"expected_start={expected_start} bucket={bucket_id}"
            )
        segment_size = range_end - range_start
        if segment_size <= 0:
            raise RuntimeError(
                f"Non-Dion range invalid: start={range_start} end={range_end} bucket={bucket_id}"
            )
        expected_start = range_end

    expected_end = int(dion_section_size) + int(non_dion_shard_size)
    if expected_start != expected_end:
        raise RuntimeError(
            f"Non-Dion ranges do not cover section: covered_end={expected_start} "
            f"expected_end={expected_end} bucket={bucket_id}"
        )


def check_local_dion_bucket_(
    *,
    parent_result,
    buckets,
    dion_param_layout,
    fs_size: int,
) -> None:
    """Validate local Dion bucket metadata built during init."""
    if not dion_param_layout:
        return

    total_local_size = sum(
        entry["local_shape"][0] * entry["local_shape"][1] for entry in dion_param_layout
    )
    expected_full_grad_total = total_local_size * fs_size
    parent_result["fs_full_grad_total"] = expected_full_grad_total

    for bucket in buckets:
        if not (hasattr(bucket, "dion_param_layout") and bucket.dion_param_layout):
            continue

        shard_size = sum(
            entry["local_shape"][0] * entry["local_shape"][1] for entry in bucket.dion_param_layout
        )
        if shard_size * fs_size != expected_full_grad_total:
            logger.error(
                "[Dion] bucket=%s, shard_size=%s, fs_size=%s, expected_full=%s",
                bucket.bucket_id,
                shard_size,
                fs_size,
                expected_full_grad_total,
            )
            raise RuntimeError("dion_grad_buffer size mismatch (shard_size*fs_size != full_total)")

        missing_param_ids = [
            id(entry["param"])
            for entry in bucket.dion_param_layout
            if id(entry["param"]) not in bucket.fs_param_id_to_full_offset
        ]
        if missing_param_ids:
            logger.error(
                "[Dion] bucket=%s, missing_param_ids=%s",
                bucket.bucket_id,
                missing_param_ids,
            )
            raise RuntimeError("fs_param_id_to_full_offset missing entries for Dion params")


def check_shared_dion_layout_(
    *,
    dp_group,
    replica_group,
    bucket_index: int,
    dp_rank: int,
    tp_rank: int,
    fs_rank: int,
    dion_param_layout,
    dion_param_count: int,
    non_dion_count: int,
    local_total: int,
    param_to_name=None,
) -> None:
    """Validate that every DP rank builds the same Dion layout for a bucket."""
    layout_summary = {
        "bucket_idx": bucket_index,
        "global_rank": dist.get_rank(),
        "dp_rank": dp_rank,
        "tp_rank": tp_rank,
        "fs_rank": fs_rank,
        "layout_len": len(dion_param_layout),
        "pack_total": max(
            (entry["pack_offset"] + entry["segment_size"] for entry in dion_param_layout),
            default=0,
        )
        if dion_param_layout
        else 0,
        "dion_count": dion_param_count,
        "non_dion_count": non_dion_count,
        "local_total": local_total,
    }

    all_layout_summaries = [None] * dp_group.size()
    dist.all_gather_object(all_layout_summaries, layout_summary, group=dp_group)

    tp_ranks = [summary["tp_rank"] for summary in all_layout_summaries]
    layout_lens = [summary["layout_len"] for summary in all_layout_summaries]
    pack_totals = [summary["pack_total"] for summary in all_layout_summaries]

    if len(set(tp_ranks)) > 1:
        logger.error("[Dion] Bucket %s: Same DP group has DIFFERENT TP ranks!", bucket_index)
        for rank_index, summary in enumerate(all_layout_summaries):
            logger.error(
                "  DP rank %s (global %s): TP rank=%s, FS rank=%s",
                rank_index,
                summary["global_rank"],
                summary["tp_rank"],
                summary["fs_rank"],
            )
        raise RuntimeError("Same DP group has different TP ranks - violates FS×TP orthogonality!")

    if len(set(layout_lens)) > 1 or len(set(pack_totals)) > 1:
        logger.error("[Dion] Bucket %s, DP rank %s:", bucket_index, dp_rank)
        for rank_index, summary in enumerate(all_layout_summaries):
            logger.error(
                "  DP rank %s (global %s, TP %s): fs_pack_len=%s, pack_total=%s, dion=%s, non_dion=%s, local_total=%s",
                rank_index,
                summary["global_rank"],
                summary["tp_rank"],
                summary["layout_len"],
                summary["pack_total"],
                summary["dion_count"],
                summary["non_dion_count"],
                summary["local_total"],
            )

        layout_entries = []
        for entry in dion_param_layout:
            param = entry["param"]
            param_name = (
                param_to_name.get(param, f"<id_{id(param)}>")
                if param_to_name is not None
                else f"<id_{id(param)}>"
            )
            layout_entries.append(
                {
                    "name": param_name,
                    "shape": tuple(param.shape),
                    "global_shape": entry["global_shape"],
                    "fs_split_dim": entry["fs_split_dim"],
                    "segment_size": entry["segment_size"],
                    "pack_offset": entry["pack_offset"],
                }
            )

        detail_summary = {
            "global_rank": dist.get_rank(),
            "dp_rank": dp_rank,
            "tp_rank": tp_rank,
            "layout_entries": layout_entries,
        }

        all_detail_summaries = [None] * dp_group.size()
        dist.all_gather_object(all_detail_summaries, detail_summary, group=dp_group)

        logger.error("[Dion] Bucket %s detailed comparison:", bucket_index)
        for detail in all_detail_summaries:
            logger.error(
                "  Global rank %s (DP %s, TP %s): %s Dion params",
                detail["global_rank"],
                detail["dp_rank"],
                detail["tp_rank"],
                len(detail["layout_entries"]),
            )
            for entry_index, entry in enumerate(detail["layout_entries"]):
                logger.error(
                    "    [%s] %s: shape=%s, global=%s, fs_split_dim=%s, segment_size=%s",
                    entry_index,
                    entry["name"],
                    entry["shape"],
                    entry["global_shape"],
                    entry["fs_split_dim"],
                    entry["segment_size"],
                )

        raise RuntimeError(f"dion_param_layout mismatch across DP ranks in bucket {bucket_index}")

    if replica_group is None or dist.get_world_size(replica_group) <= 1:
        return

    local_signature = []
    for entry in dion_param_layout:
        param = entry["param"]
        param_name = (
            param_to_name.get(param, f"<id_{id(param)}>")
            if param_to_name is not None
            else f"<id_{id(param)}>"
        )
        local_signature.append(
            (
                param_name,
                tuple(entry["local_shape"]),
                tuple(entry["global_shape"]),
                int(entry["fs_split_dim"]),
                int(entry["segment_size"]),
                int(entry["pack_offset"]),
            )
        )

    replica_summary = {
        "global_rank": dist.get_rank(),
        "tp_rank": tp_rank,
        "fs_rank": fs_rank,
        "layout_signature": local_signature,
    }
    all_replica_summaries = [None] * dist.get_world_size(replica_group)
    dist.all_gather_object(all_replica_summaries, replica_summary, group=replica_group)

    canonical_signature = all_replica_summaries[0]["layout_signature"]
    mismatches = [
        summary["global_rank"]
        for summary in all_replica_summaries
        if summary["layout_signature"] != canonical_signature
    ]
    if mismatches:
        logger.error(
            "[Dion] Bucket %s: inter-instance Dion layout mismatch across replica group %s",
            bucket_index,
            dist.get_process_group_ranks(replica_group),
        )
        for summary in all_replica_summaries:
            logger.error(
                "  Global rank %s (TP %s, FS %s): %s",
                summary["global_rank"],
                summary["tp_rank"],
                summary["fs_rank"],
                summary["layout_signature"],
            )
        raise RuntimeError(
            f"dion_param_layout mismatch across optimizer-state replicas in bucket {bucket_index}"
        )
