"""
Distributed optimizer wrapper for Dion optimizer in Megatron-LM.

Supports orthogonal TP × FS sharding:
- tp_split_dim=0 (ColumnParallel): FS shards cols
- tp_split_dim=1 (RowParallel): FS shards rows
"""

import logging
import math
import os
import traceback
import warnings
import torch
import torch.distributed as dist
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Iterator

from .distrib_optimizer import DistributedOptimizer, Range
from .megatron_dion import MegatronDion, MegatronDionDistMeta
from .. import tensor_parallel
from ..fp8_utils import is_float8tensor

logger = logging.getLogger(__name__)


# Keywords that exclude a 2D parameter from Dion classification
DION_EXCLUDE_KEYWORDS = [
    'embedding', 'word_embeddings', 'position_embeddings',
    'output_layer', 'lm_head', 'vocab',
    'norm', 'layernorm', 'rmsnorm', 'groupnorm', 'batchnorm'
]


def is_dion_param(param: torch.Tensor, param_name: Optional[str] = None) -> bool:
    """Classify whether a parameter should use Dion (2D FS sharding).

    Dion params: 2D tensors excluding embedding/norm/lm_head/output_layer.
    """
    # Rule 1: Must be 2D
    if param.ndim != 2:
        return False

    # Rule 2: Check exclusions by parameter name
    if param_name:
        name_lower = param_name.lower()
        for keyword in DION_EXCLUDE_KEYWORDS:
            if keyword in name_lower:
                return False

    return True


# Dion Layout Data Structures

@dataclass(frozen=True)
class DionLayoutEntry:
    """Single entry in DionParamLayout.

    Represents metadata for one Dion parameter's FS shard.
    Immutable (frozen=True) to prevent accidental modification.

    Attributes:
        param: The model parameter tensor
        global_shape: Shape before any parallelism (m, n)
        local_shape: Shape after TP partitioning (m_local, n) or (m, n_local)
        fs_split_dim: FS split dimension (orthogonal to TP split)
        fs_rank: This rank's position in FS group
        start_idx: Start index for FS shard in split dimension
        end_idx: End index for FS shard in split dimension
        size_per_rank: Elements per rank in split dimension
        segment_size: Total elements in this shard (numel)
        pack_offset: Offset in packed buffer for all-gather
    """
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
        """Total number of elements in this shard."""
        return self.local_shape[0] * self.local_shape[1]

    # Dict-like access for backward compatibility during migration
    def __getitem__(self, key: str):
        return getattr(self, key)

    def get(self, key: str, default=None):
        return getattr(self, key, default)

    def copy(self) -> dict:
        """Return dict copy for backward compatibility with code that modifies entries."""
        return {
            'param': self.param,
            'global_shape': self.global_shape,
            'local_shape': self.local_shape,
            'fs_split_dim': self.fs_split_dim,
            'fs_rank': self.fs_rank,
            'start_idx': self.start_idx,
            'end_idx': self.end_idx,
            'size_per_rank': self.size_per_rank,
            'segment_size': self.segment_size,
            'pack_offset': self.pack_offset,
        }


@dataclass
class DionParamLayout:
    """Container for DionLayoutEntry list with cached aggregate properties.

    Supports dict-like iteration for backward compatibility.

    Usage:
        layout = DionParamLayout()
        layout.append(DionLayoutEntry(...))
        for entry in layout:
            print(entry.numel)
    """
    entries: List[DionLayoutEntry] = field(default_factory=list)

    def __iter__(self) -> Iterator[DionLayoutEntry]:
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    def __bool__(self) -> bool:
        return len(self.entries) > 0

    def append(self, entry: DionLayoutEntry) -> None:
        """Add entry to layout."""
        self.entries.append(entry)

    def extend(self, entries) -> None:
        """Extend with entries from another layout or list."""
        if isinstance(entries, DionParamLayout):
            self.entries.extend(entries.entries)
        else:
            self.entries.extend(entries)


# Dion Shard Info

@dataclass
class DionShardInfo:
    """Unified shard information for a Dion parameter.

    Stored in _dion_shard_info dict, keyed by model_param.
    Provides access to both data_shard (FP16) and opt_shard (FP32) along with metadata.

    Attributes:
        data_shard: FP16 shard tensor (all-gather source, same dtype as model)
        opt_shard: FP32 shard tensor (optimizer state), equals data_shard for FP32 params
        local_shape: Shape after TP partitioning (m_local, n) or (m, n_local)
        global_shape: Shape before any parallelism (m, n)
        start_idx: Start index for FS shard in split dimension
        end_idx: End index for FS shard in split dimension
        fs_split_dim: FS split dimension (orthogonal to TP split)
        gbuf_index: Gradient buffer index
        bucket_index: Bucket index within gradient buffer
        param_range_info: Parameter range info dict from gbuf_ranges
    """
    data_shard: torch.Tensor
    opt_shard: torch.Tensor
    local_shape: Tuple[int, int]
    global_shape: Tuple[int, int]
    start_idx: int
    end_idx: int
    fs_split_dim: int
    gbuf_index: int
    bucket_index: int
    param_range_info: dict


# FS Sharding Helper Functions

def compute_fs_shard_range(global_size: int, fs_size: int, fs_rank: int) -> Tuple[int, int]:
    """
    Compute (start_idx, end_idx) for FS sharding.

    Handles uneven division (remainder) correctly:
    - First `remainder` ranks get `size_per_rank + 1` elements
    - Remaining ranks get `size_per_rank` elements

    Args:
        global_size: Total size along the split dimension
        fs_size: Number of FS ranks
        fs_rank: Current FS rank (0-indexed)

    Returns:
        (start_idx, end_idx): Range [start_idx, end_idx) for this FS rank

    Example:
        >>> compute_fs_shard_range(1536, 4, 0)  # rank 0
        (0, 384)
        >>> compute_fs_shard_range(1536, 4, 1)  # rank 1
        (384, 768)
        >>> compute_fs_shard_range(10, 4, 0)   # uneven: rank 0 gets 3
        (0, 3)
        >>> compute_fs_shard_range(10, 4, 2)   # uneven: rank 2 gets 2
        (6, 8)
    """
    if fs_size <= 0:
        raise ValueError(f"fs_size must be positive, got {fs_size}")
    if fs_rank < 0 or fs_rank >= fs_size:
        raise ValueError(f"fs_rank must be in [0, {fs_size}), got {fs_rank}")

    size_per_rank = global_size // fs_size
    remainder = global_size % fs_size

    if fs_rank < remainder:
        # First `remainder` ranks get one extra element
        start_idx = fs_rank * (size_per_rank + 1)
        end_idx = start_idx + size_per_rank + 1
    else:
        # Remaining ranks get base size
        start_idx = remainder * (size_per_rank + 1) + (fs_rank - remainder) * size_per_rank
        end_idx = start_idx + size_per_rank

    return start_idx, end_idx


def get_fs_split_dim(tp_split_dim: int) -> int:
    """
    Determine FS split dimension based on TP split dimension.

    TP and FS split on ORTHOGONAL dimensions:
    - tp_split_dim=0 (ColumnParallel): TP splits rows → FS splits cols (dim=1)
    - tp_split_dim=1 (RowParallel): TP splits cols → FS splits rows (dim=0)
    - tp_split_dim=-1 (No TP): default to row split (dim=0)

    Args:
        tp_split_dim: TP split dimension (0, 1, or -1 for no TP)
                      Maps from Megatron's 'partition_dim' attribute

    Returns:
        fs_split_dim: Dimension to split for FS (orthogonal to TP)
    """
    if tp_split_dim == 0:
        return 1  # ColumnParallel: FS splits cols
    elif tp_split_dim == 1:
        return 0  # RowParallel: FS splits rows
    else:
        return 0  # No TP: default to row split


def compute_local_shape(
    m: int, n: int,
    start_idx: int, end_idx: int,
    fs_split_dim: int
) -> Tuple[int, int]:
    """
    Compute local shape after FS sharding.

    Args:
        m, n: Full shape dimensions
        start_idx, end_idx: FS shard range
        fs_split_dim: Dimension split by FS (0=row, 1=col)

    Returns:
        (local_m, local_n): Local shape for this FS shard
    """
    local_split_size = end_idx - start_idx
    if fs_split_dim == 0:
        return (local_split_size, n)
    else:
        return (m, local_split_size)


def get_param_name(param: torch.Tensor, fallback_style: str = "shape") -> str:
    """
    Get parameter name consistently.

    Args:
        param: Parameter tensor
        fallback_style: "shape", "id", or "both" for fallback format

    Returns:
        Parameter name string
    """
    name = getattr(param, '_param_name', None)
    if name is not None:
        return name

    if fallback_style == "id":
        return f"id_{id(param)}"
    elif fallback_style == "both":
        return f"shape={param.shape}"
    else:  # "shape"
        return str(param.shape)


def get_tp_split_dim(param: torch.Tensor) -> int:
    """
    Get TP split dimension from parameter.

    Converts Megatron's 'partition_dim' attribute to Dion's naming convention.
    This provides consistent naming with fs_split_dim:
      - tp_split_dim: dimension split by Tensor Parallelism
      - fs_split_dim: dimension split by Full Sharding (Dion)

    Args:
        param: Parameter tensor with optional 'partition_dim' attribute

    Returns:
        tp_split_dim: 0 (row-split, ColumnParallel), 1 (col-split, RowParallel), or -1 (no TP)
    """
    return getattr(param, 'partition_dim', -1)


def is_tp_enabled(param: torch.Tensor) -> bool:
    """
    Check if TP is enabled for this parameter.

    Args:
        param: Parameter tensor

    Returns:
        True if tensor_model_parallel attribute is set and True
    """
    return hasattr(param, 'tensor_model_parallel') and param.tensor_model_parallel


class DistributedOptimizerForDion(DistributedOptimizer):
    """
    Distributed optimizer for MegatronDion with true parameter sharding.

    Architecture:
    - True parameter sharding: Only local shards stored on GPU
    - Buffer sizes reduced to shard size (not full size)
    - Bucket-wise all-gather/reduce-scatter via standard Megatron-Core DO
    - FSDP-style memory efficiency: Full params only during forward/backward

    Extends DistributedOptimizer to support Dion's 2D parallelism model:
    - RP (Replicate Process): Gradient averaging across replicas with same FS shard
    - FS (Fully Shard): True row-wise sharding for 2D params (GPU memory saved)
    - TP (Tensor Parallel): Column-wise tensor sharding

    Provides automatic configuration of 2D process groups and FS-aware annotation.
    """

    # Class variable for _build_model_gbuf_range (classmethod workaround)
    _current_init_fs_config = None

    @classmethod
    def _build_model_gbuf_range(cls, param_and_grad_buffer, bucket_index):
        """
        Build model grad buffer ranges using parent DO standard, then add Dion metadata.

        Architecture:
        - Uses parent DistributedOptimizer's standard bucket structure
        - Adds Dion metadata only for 2D weight matrices
        - Non-Dion params follow standard DO path entirely

        Returns: Parent DO structure + optional "dion_info" per param
        """
        # STEP 1: Call parent to get DO standard bucket structure
        from .distrib_optimizer import DistributedOptimizer
        parent_result = DistributedOptimizer._build_model_gbuf_range(
            param_and_grad_buffer, bucket_index
        )

        # STEP 2: Verify parent DO param_map consistency
        bucket = param_and_grad_buffer.buckets[bucket_index]
        dp_group = param_and_grad_buffer.data_parallel_group
        dp_rank = dp_group.rank()
        dp_world_size = dp_group.size()

        # Cache param names for fallback
        if not hasattr(cls, "_param_name_bank"):
            cls._param_name_bank = dict(param_and_grad_buffer.param_to_name) if hasattr(param_and_grad_buffer, 'param_to_name') else {}

        from collections import OrderedDict

        # Preserve original name map
        original_name_map = dict(param_and_grad_buffer.param_to_name) if hasattr(param_and_grad_buffer, 'param_to_name') else {}

        # Sync name map across ranks (use id-based serialization)
        if original_name_map:
            local_names = {id(p): name for p, name in original_name_map.items()}
        else:
            local_names = {}

        all_names = [None] * dp_group.size()
        dist.all_gather_object(all_names, local_names, group=dp_group)

        merged_names_by_id = {}
        for m in all_names:
            if m:
                merged_names_by_id.update(m)

        # Rebuild param mapping for bucket params
        rebuilt_param_to_name = {}
        for p in bucket.params:
            p_id = id(p)
            if p_id in merged_names_by_id:
                rebuilt_param_to_name[p] = merged_names_by_id[p_id]

        param_and_grad_buffer.param_to_name = rebuilt_param_to_name


        # Broadcast DP rank0's name list for canonical ordering
        local_names_list = []
        for p in bucket.params:
            name = original_name_map.get(p, "")
            if not name and cls._param_name_bank:
                name = cls._param_name_bank.get(p, "")
            local_names_list.append(name)

        all_names_lists = [None] * dp_group.size()
        dist.all_gather_object(all_names_lists, local_names_list, group=dp_group)

        names_list = all_names_lists[0] if all_names_lists and all_names_lists[0] is not None else []

        if len(names_list) != len(bucket.params):
            logger.error(
                f"[Dion] len(names_list)={len(names_list)} "
                f"!= len(bucket.params)={len(bucket.params)}; all_lengths={[len(x) if x else None for x in all_names_lists]}"
            )
            raise RuntimeError("param_to_name broadcast length mismatch")

        if any(not n for n in names_list):
            bad_idx = [i for i, n in enumerate(names_list) if not n]
            logger.error(f"[Dion] Empty names at indices {bad_idx} in bucket={bucket_index}, dp_rank={dp_rank}")
            raise RuntimeError("param_to_name contains empty names")

        # Use rank0's name order
        names_list = all_names_lists[0]

        # Build name -> param mapping with duplicate check
        name_to_param = {}
        for p in bucket.params:
            name = param_and_grad_buffer.param_to_name.get(p, "")
            if not name:
                continue
            if name in name_to_param:
                logger.error(f"[Dion] Duplicate param name detected locally: {name}")
                raise RuntimeError("Duplicate parameter names found in bucket.params")
            name_to_param[name] = p

        ordered_params = []
        for name in names_list:
            if name not in name_to_param:
                logger.error(f"[Dion] Canonical name '{name}' not found in local bucket.params (bucket={bucket_index}, dp_rank={dp_rank})")
                raise RuntimeError("Canonical param order mismatch across DP ranks")
            ordered_params.append(name_to_param[name])

        # Verify names consistency
        seen_names = set()
        for name in names_list:
            if not name:
                continue
            if name in seen_names:
                logger.error(f"[Dion] Duplicate param name detected: {name}")
                raise RuntimeError("Duplicate parameter names found in bucket.params")
            seen_names.add(name)

        new_name_map = dict(original_name_map)
        for i, p in enumerate(ordered_params):
            new_name_map[p] = names_list[i]

        param_and_grad_buffer.param_to_name = new_name_map
        cls._param_name_bank.update(new_name_map)

        # 2-2) Compare parent param_map length (log only for DP sharding status)
        parent_param_map = parent_result["param_map"]
        # Save parent shard sizes for non-Dion shard fallback
        parent_param_shard_size = {p: parent_param_map[p]["param"].size for p in parent_param_map}
        parent_param_summary = {
            "bucket_idx": bucket_index,
            "dp_rank": dp_rank,
            "param_count": len(parent_param_map),
        }
        all_param_summaries = [None] * dp_group.size()
        dist.all_gather_object(all_param_summaries, parent_param_summary, group=dp_group)
        param_counts = [s["param_count"] for s in all_param_summaries]
        if len(set(param_counts)) > 1:
            if bucket_index == 0 and dp_rank == 0:
                logger.warning(f"[Dion] Bucket {bucket_index}: param_map sizes differ (expected with optim_grads_params). Using bucket.params as canonical.")
        # 2-3) Reconstruct canonical param_map with bucket global params (deterministic sort)
        canonical_param_map = OrderedDict()
        for p in ordered_params:
            if p in parent_param_map:
                entry = parent_param_map[p]
            else:
                entry = {
                    "param": Range(0, 0),
                    "gbuf_world": Range(0, 0),
                    "gbuf_local": Range(0, 0),
                    "gbuf_world_in_bucket": Range(0, 0),
                }
            canonical_param_map[p] = entry

        parent_result["param_map"] = canonical_param_map
        param_map = canonical_param_map
        params_in_order = list(canonical_param_map.keys())

        # Get FS config
        fs_config = cls._current_init_fs_config
        fs_size = fs_config.get('fs_size', 1) if fs_config else 1

        # When CP > 1, use pure DP group rank for fs_rank (FS RS uses pure DP, not CP-combined group)
        from megatron.core import parallel_state as ps
        cp_world_size = ps.get_context_parallel_world_size() if ps.is_initialized() else 1
        if cp_world_size > 1 and fs_size > 1:
            # CP > 1: Get rank from pure DP group (excludes CP)
            pure_dp_group = ps.get_data_parallel_group(with_context_parallel=False)
            pure_dp_rank = pure_dp_group.rank()
            fs_rank = pure_dp_rank % fs_size
        else:
            # CP = 1: dp_rank is already correct (same as pure DP rank)
            fs_rank = dp_rank % fs_size if fs_size > 1 else 0

        # Get TP info helper
        from ..parallel_state import get_tensor_model_parallel_world_size

        classified_as_dion = 0
        classified_as_non_dion = 0

        # Counter must be initialized before classification; otherwise UnboundLocalError
        dion_param_count = 0

        for param in param_map.keys():
            # Get parameter name for classification
            param_name = None
            if hasattr(param_and_grad_buffer, 'param_to_name') and param in param_and_grad_buffer.param_to_name:
                param_name = param_and_grad_buffer.param_to_name[param]

            # Classify param for Dion vs non-Dion handling
            param.is_dion_param = is_dion_param(param, param_name)

            if not param.is_dion_param:
                classified_as_non_dion += 1
                continue

            # This is a Dion param!
            classified_as_dion += 1
            dion_param_count += 1
            m, n = param.shape

            # Get TP info
            tp_split_dim = get_tp_split_dim(param)
            has_tp = is_tp_enabled(param)
            # CRITICAL: Expert params use Expert TP world size, not Dense TP world size
            is_expert = not getattr(param, 'allreduce', True)
            if is_expert and has_tp:
                from megatron.core import parallel_state
                tp_world_size = parallel_state.get_expert_tensor_parallel_world_size()
            else:
                tp_world_size = get_tensor_model_parallel_world_size() if has_tp else 1

            # FS split dimension is orthogonal to TP split dimension
            fs_split_dim = get_fs_split_dim(tp_split_dim)

            # Calculate global shape (before TP sharding)
            # When has_tp=False, tp_world_size=1, so multiplication is a no-op
            if tp_split_dim == 0:
                global_m = m * tp_world_size
                global_n = n
            elif tp_split_dim == 1:
                global_m = m
                global_n = n * tp_world_size
            else:
                global_m = m
                global_n = n

            # Add Dion metadata (start_idx/end_idx set later in _create_fs_aware_shard)
            param_map[param]["dion_info"] = {
                "is_dion": True,
                "global_shape": (global_m, global_n),
                "fs_split_dim": fs_split_dim,
                "tp_split_dim": tp_split_dim,  # Consistent naming with fs_split_dim
            }

        # STEP 3: Recalculate buffer ranges for FS × TP hybrid sharding
        # Dion params use FS shard, non-Dion params use DP shard

        dion_param_layout = DionParamLayout()
        dion_param_shard_range = {}  # Map param → (start, end) for main_grad binding
        local_offset = 0
        pack_offset = 0

        # First pass: Dion params (FS sharding), sorted for deterministic order
        dion_params = [(p, info) for p, info in param_map.items()
                      if hasattr(p, 'is_dion_param') and p.is_dion_param]
        def get_sort_key(param_info_tuple):
            param = param_info_tuple[0]
            param_name = param_and_grad_buffer.param_to_name.get(param, "") if hasattr(param_and_grad_buffer, 'param_to_name') else ""
            return (tuple(param.shape), param_name)

        dion_params.sort(key=get_sort_key)

        dion_param_count = 0
        for param, param_info in dion_params:
            dion_info = param_info.get("dion_info")
            if not dion_info:
                continue

            global_shape = dion_info["global_shape"]
            fs_split_dim = dion_info["fs_split_dim"]
            m, n = param.shape  # Current TP-sharded shape

            # Calculate FS shard range
            split_size = m if fs_split_dim == 0 else n
            start_idx, end_idx = compute_fs_shard_range(split_size, fs_size, fs_rank)
            local_split_size = end_idx - start_idx
            size_per_rank = math.ceil(split_size / fs_size)  # For segment_size padding

            # Local shape after FS sharding
            if fs_split_dim == 0:
                local_shape = (local_split_size, n)
                local_size = local_split_size * n
            else:
                local_shape = (m, local_split_size)
                local_size = m * local_split_size

            # Calculate segment_size FIRST (needed for local_range)
            # segment_size is padded to ensure uniform layout across FS ranks
            segment_size = size_per_rank * (n if fs_split_dim == 0 else m)

            # Update param_map ranges using pack_offset
            local_range = Range(pack_offset, pack_offset + local_size)
            param_map[param]["param"] = local_range
            param_map[param]["gbuf_world"] = local_range
            param_map[param]["gbuf_local"] = local_range

            # Store FS shard range for main_grad binding (bucket-relative)
            # Use pack_offset to match RS output layout
            dion_param_shard_range[param] = (pack_offset, pack_offset + local_size)

            # Add to dion_param_layout for FS all-gather
            dion_param_layout.append(DionLayoutEntry(
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
            ))

            # local_offset must match pack_offset for consistent layout
            # Both use segment_size to ensure RS output position == main_grad binding position
            local_offset += segment_size
            pack_offset += segment_size
            dion_param_count += 1

        # Store Dion section size
        dion_section_size = local_offset

        # Second pass: Non-Dion params (DP shard, standard DO)
        non_dion_params = [(p, info) for p, info in param_map.items()
                          if not (hasattr(p, 'is_dion_param') and p.is_dion_param)]

        # When CP > 1, non-Dion params use pure DP for RS
        from megatron.core import parallel_state as ps
        cp_size_check = ps.get_context_parallel_world_size()
        if cp_size_check > 1:
            # CP > 1: use fs_size (pure DP) for non-Dion params
            non_dion_dp_size = fs_size
            non_dion_dp_rank = fs_rank
        else:
            # CP = 1: use standard DP size (existing behavior)
            non_dion_dp_size = dp_world_size
            non_dion_dp_rank = dp_rank

        # Non-Dion params: uniform shard size = ceil(numel / dp_size)
        for param, param_info in non_dion_params:
            param_numel = param.numel()
            shard_size = (param_numel + non_dion_dp_size - 1) // non_dion_dp_size

            # Param shard range for this DP rank
            param_shard_start = non_dion_dp_rank * shard_size
            param_shard_end = min(param_shard_start + shard_size, param_numel)
            param_relative_range = Range(param_shard_start, param_shard_end)

            # Bucket-relative range (for main_grad binding after RS)
            bucket_relative_range = Range(local_offset, local_offset + shard_size)

            param_map[param]["param"] = param_relative_range  # For model_param slicing
            param_map[param]["gbuf_world"] = bucket_relative_range  # For bucket operations
            param_map[param]["gbuf_local"] = bucket_relative_range  # For bucket operations

            local_offset += shard_size

        # Update local_total for buffer resize
        parent_result["local_total"] = local_offset

        # Calculate param counts for summary
        total_params = len(param_map)
        non_dion_count = total_params - dion_param_count

        # Add dion_param_layout and dion_param_shard_range to result
        parent_result["dion_param_layout"] = dion_param_layout
        parent_result["dion_param_shard_range"] = dion_param_shard_range

        # Verify dion_param_layout consistency across FS/DP group
        # Same FS group must have identical dion_param_layout to avoid all-gather size mismatch
        dp_group = param_and_grad_buffer.data_parallel_group

        # Get TP rank for this rank
        from ..parallel_state import get_tensor_model_parallel_rank
        tp_rank = get_tensor_model_parallel_rank()

        summary = {
            "bucket_idx": bucket_index,
            "global_rank": dist.get_rank(),
            "dp_rank": dp_rank,
            "tp_rank": tp_rank,
            "fs_rank": fs_rank,
            "len": len(dion_param_layout),
            "pack_total": max((e["pack_offset"] + e["segment_size"] for e in dion_param_layout), default=0) if dion_param_layout else 0,
            "dion_count": dion_param_count,
            "non_dion_count": non_dion_count,
            "local_total": local_offset,
        }

        # Gather summaries from all DP ranks
        all_summaries = [None] * dp_group.size()
        dist.all_gather_object(all_summaries, summary, group=dp_group)

        # Check consistency
        pack_lens = [s["len"] for s in all_summaries]
        pack_totals = [s["pack_total"] for s in all_summaries]
        local_totals = [s["local_total"] for s in all_summaries]
        tp_ranks = [s["tp_rank"] for s in all_summaries]

        # Local consistency checks for FS grad buffers (single rank, init time)
        if dion_param_layout:
            total_local_size = sum(entry['local_shape'][0] * entry['local_shape'][1] for entry in dion_param_layout)
            expected_full_grad_total = total_local_size * fs_size
            parent_result["fs_full_grad_total"] = expected_full_grad_total

            # Verify per-bucket mappings exist and sizes match
            for bucket in param_and_grad_buffer.buckets:
                if hasattr(bucket, 'dion_param_layout') and bucket.dion_param_layout:
                    shard_size = sum(e['local_shape'][0] * e['local_shape'][1] for e in bucket.dion_param_layout)
                    if shard_size * fs_size != expected_full_grad_total:
                        logger.error(f"[Dion] bucket={bucket.bucket_id}, shard_size={shard_size}, fs_size={fs_size}, expected_full={expected_full_grad_total}")
                        raise RuntimeError("dion_grad_buffer size mismatch (shard_size*fs_size != full_total)")
                    missing = [id(e['param']) for e in bucket.dion_param_layout if id(e['param']) not in bucket.fs_param_id_to_full_offset]
                    if missing:
                        logger.error(f"[Dion] bucket={bucket.bucket_id}, missing_param_ids={missing}")
                        raise RuntimeError("fs_param_id_to_full_offset missing entries for Dion params")

        # Check if same DP group has different TP ranks (critical issue!)
        if len(set(tp_ranks)) > 1:
            logger.error(f"[Dion] Bucket {bucket_index}: Same DP group has DIFFERENT TP ranks!")
            for rank_i, s in enumerate(all_summaries):
                logger.error(f"  DP rank {rank_i} (global {s['global_rank']}): TP rank={s['tp_rank']}, FS rank={s['fs_rank']}")
            raise RuntimeError(f"Same DP group has different TP ranks - violates FS×TP orthogonality!")

        # Verify FS plan consistency across ranks
        if len(set(pack_lens)) > 1 or len(set(pack_totals)) > 1:
            logger.error(f"[Dion] Bucket {bucket_index}, DP rank {dp_rank}:")
            for rank_i, s in enumerate(all_summaries):
                logger.error(f"  DP rank {rank_i} (global {s['global_rank']}, TP {s['tp_rank']}): "
                           f"fs_pack_len={s['len']}, pack_total={s['pack_total']}, "
                           f"dion={s['dion_count']}, non_dion={s['non_dion_count']}, local_total={s['local_total']}")

            # Collect detailed dion_param_layout entries to identify which params differ
            plan_details = []
            for entry in dion_param_layout:
                param = entry['param']
                param_name = param_and_grad_buffer.param_to_name.get(param, f"<id_{id(param)}>") if hasattr(param_and_grad_buffer, 'param_to_name') else f"<id_{id(param)}>"
                plan_details.append({
                    "name": param_name,
                    "shape": tuple(param.shape),
                    "global_shape": entry['global_shape'],
                    "fs_split_dim": entry['fs_split_dim'],
                    "segment_size": entry['segment_size'],
                    "pack_offset": entry['pack_offset'],
                })

            detail_summary = {
                "global_rank": dist.get_rank(),
                "dp_rank": dp_rank,
                "tp_rank": tp_rank,
                "entries": plan_details
            }

            all_details = [None] * dp_group.size()
            dist.all_gather_object(all_details, detail_summary, group=dp_group)

            logger.error(f"[Dion] Bucket {bucket_index} detailed comparison:")
            for det in all_details:
                logger.error(f"  Global rank {det['global_rank']} (DP {det['dp_rank']}, TP {det['tp_rank']}): {len(det['entries'])} Dion params")
                for idx, e in enumerate(det['entries']):
                    logger.error(f"    [{idx}] {e['name']}: shape={e['shape']}, global={e['global_shape']}, "
                               f"fs_split_dim={e['fs_split_dim']}, segment_size={e['segment_size']}")

            raise RuntimeError(f"dion_param_layout mismatch across DP ranks in bucket {bucket_index}")

        # Return hybrid sharding structure (parent ranges + Dion metadata)
        return parent_result
    def __init__(self, *args, **kwargs):
        """Initialize with improved Dion support."""
        # Initialize Dion param info before parent init
        self._dion_param_info = {}
        self._global_owner_tuples = set()

        # Batch processing configuration
        self._batch_size = kwargs.pop('dion_batch_size', 8)

        # 2D parallelism configuration
        # RP = Replicate Process (replicas with same shard)
        # FS = Fully Shard (shards within same replica)
        self._rp_size = kwargs.pop('rp_size', None) or kwargs.pop('replica_model_parallel_size', None)
        self._fs_size = kwargs.pop('fs_size', None) or kwargs.pop('fully_shard_model_parallel_size', None)

        # Enforce RP=1 for DO overlap compatibility
        if self._rp_size is not None and self._rp_size != 1:
            raise ValueError(
                f"MegatronDion currently requires RP=1 for DO overlap compatibility. "
                f"Got rp_size={self._rp_size}. "
                f"This simplification allows direct use of DO bucket groups."
            )
        self._rp_size = 1  # Force RP=1

        # Set class variable for _build_model_gbuf_range (accessed during super().__init__())
        DistributedOptimizerForDion._current_init_fs_config = {
            'fs_size': self._fs_size if self._fs_size else 1,
            'rp_size': self._rp_size if self._rp_size else 1,
        }

        # Call parent initialization with full DP group (RP × FS)
        # DistributedOptimizer will do uniform sharding across all DP ranks
        # Dion will handle 2D topology (RP × FS) at optimizer state level
        super().__init__(*args, **kwargs)

        # Clean up class variable after super().__init__()
        DistributedOptimizerForDion._current_init_fs_config = None

        # Cache global rank for logging (avoid repeated dist.get_rank() calls)
        self._global_rank = dist.get_rank() if dist.is_initialized() else 0

        # FS Group: use optimizer.fs_group if available, else data_parallel_group
        use_optimizer_fs_group = (
            hasattr(self, 'optimizer') and
            hasattr(self.optimizer, 'fs_group') and
            self.optimizer.fs_group is not None
        )

        if use_optimizer_fs_group:
            self.fs_group = self.optimizer.fs_group
            self.fs_size = self.fs_group.size()
            self.fs_rank = self.fs_group.rank()
            # Verify consistency: FS size should match configured _fs_size
            if hasattr(self, '_fs_size') and self._fs_size is not None and self._fs_size != self.fs_size:
                logger.warning(
                    f"[Dion] Global rank {self._global_rank}: "
                    f"FS size mismatch! configured={self._fs_size}, optimizer.fs_group.size()={self.fs_size}"
                )
        else:
            # Fallback: When CP=1, data_parallel_group equals pure DP group
            self.fs_group = self.data_parallel_group
            self.fs_size = self.data_parallel_group.size()
            self.fs_rank = self.data_parallel_group.rank()
        self.rp_group = None  # No replicas

        # FS shard-alloc mode (bucket-wise all-gather/reduce-scatter via standard DO)
        self.use_fs_shard_alloc = True

        # Unified shard info mapping
        # Consolidates model_param → DionShardInfo (data_shard, opt_shard, metadata)
        self._dion_shard_info: Dict[torch.nn.Parameter, DionShardInfo] = {}

        # Buffers already sized in _build_model_gbuf_range during super().__init__()
        if self.use_fs_shard_alloc and hasattr(self, 'buffers'):

                # Store dion_param_layout metadata on buffers for FS all-gather operations
                if hasattr(self, 'gbuf_ranges'):
                    for gbuf_idx, gbuf_range_maps in enumerate(self.gbuf_ranges):
                        buffer = self.buffers[gbuf_idx]
                        buffer.dion_param_layouts_by_bucket = []
                        buffer.fs_bucket_plan_map = {}
                        total_entries = 0
                        cumulative_bucket_id = 0

                        # Build bucket_id → plan map with cumulative counter (global bucket ids)
                        for dtype, bucket_list in gbuf_range_maps.items():
                            for dtype_local_idx, gbuf_range_map in enumerate(bucket_list):
                                global_bucket_id = cumulative_bucket_id

                                dion_shard_range = gbuf_range_map.get("dion_param_shard_range", None)
                                buffer.fs_bucket_plan_map[global_bucket_id] = {
                                    "dtype": dtype,
                                    "local_total": gbuf_range_map.get("local_total", 0),
                                    "dion_param_layout": gbuf_range_map.get("dion_param_layout", []),
                                    "dion_param_shard_range": dion_shard_range,
                                }
                                if 'dion_param_layout' in gbuf_range_map and gbuf_range_map['dion_param_layout']:
                                    buffer.dion_param_layouts_by_bucket.append({
                                        'bucket_idx': global_bucket_id,
                                        'dtype': dtype,
                                        'dion_param_layout': gbuf_range_map['dion_param_layout']
                                    })
                                    total_entries += len(gbuf_range_map['dion_param_layout'])

                                cumulative_bucket_id += 1

                        # Sanity: bucket ids must match DDP buckets exactly
                        expected_bucket_ids = {b.bucket_id for b in buffer.buckets}
                        if set(buffer.fs_bucket_plan_map.keys()) != expected_bucket_ids:
                            raise RuntimeError(
                                f"[DDP/OPT BUCKET MISMATCH] buffer={gbuf_idx} expected bucket_ids={sorted(expected_bucket_ids)} "
                                f"but gbuf_ranges provided {sorted(buffer.fs_bucket_plan_map.keys())}"
                            )

                        # Also keep combined plan for backward compatibility
                        buffer.dion_param_layout = DionParamLayout()
                        for bucket_info in buffer.dion_param_layouts_by_bucket:
                            buffer.dion_param_layout.extend(bucket_info['dion_param_layout'])

                        # Update buffer.param_index_map to match new [Dion shards][non-Dion shards] layout
                        for bucket in buffer.buckets:
                            bucket_plan = buffer.fs_bucket_plan_map.get(bucket.bucket_id)
                            if bucket_plan is None:
                                continue

                            # Get param_map from gbuf_ranges for this bucket
                            dtype_key = bucket_plan["dtype"]
                            gbuf_range_map = self.gbuf_ranges[gbuf_idx][dtype_key][bucket.bucket_id]
                            param_map = gbuf_range_map.get("param_map", {})

                            # Update param_index_map for each param in this bucket
                            for param, range_info in param_map.items():
                                local_range = range_info["param"]
                                # Convert bucket-relative offset to buffer-global offset
                                global_start = bucket.offset + local_range.start
                                global_end = bucket.offset + local_range.end
                                buffer.param_index_map[param] = (global_start, global_end, bucket.bucket_id)


                        # Allocate dion_grad_buffer for shape-aware FS reduce-scatter
                        fs_group, fs_size, fs_rank = self.fs_group, self.fs_size, self.fs_rank

                        # Process each bucket in this buffer
                        for bucket in buffer.buckets:
                            plan_info = buffer.fs_bucket_plan_map.get(bucket.bucket_id, None)
                            bucket_dion_param_layout = None
                            dion_param_shard_range_map = None

                            # Use buffer's param_dtype and grad_dtype to match parent DO's dtype_key
                            dtype_key = (buffer.param_dtype, buffer.grad_dtype)

                            if plan_info is None:
                                raise RuntimeError(f"[DDP/OPT BUCKET MISMATCH] buffer={gbuf_idx} missing plan for bucket_id={bucket.bucket_id}")
                            if plan_info["dtype"] != dtype_key:
                                raise RuntimeError(
                                    f"[DDP/OPT BUCKET MISMATCH] buffer={gbuf_idx} bucket={bucket.bucket_id} dtype mismatch: "
                                    f"DDP bucket dtype_key={dtype_key} vs plan dtype_key={plan_info['dtype']}"
                                )

                            bucket_dion_param_layout = plan_info.get('dion_param_layout', None)
                            dion_param_shard_range_map = plan_info.get('dion_param_shard_range', None)

                            dtype_bucket_idx = bucket.bucket_id

                            if bucket_dion_param_layout and len(bucket_dion_param_layout) > 0:
                                # This bucket has Dion params - allocate dion_grad_buffer
                                bucket.dion_param_layout = bucket_dion_param_layout

                                # Also set dion_param_shard_range from gbuf_range_map
                                if dion_param_shard_range_map:
                                    bucket.dion_param_shard_range = dion_param_shard_range_map

                                # Calculate full gradient buffer size for RS input:
                                # sum(local_size) * fs_size (local_size is TP-sharded, FS-shard per rank)
                                bucket_local_full_size = sum(
                                    entry['local_shape'][0] * entry['local_shape'][1]
                                    for entry in bucket_dion_param_layout
                                ) * fs_size
                                # Store total size, allocate on first backward
                                bucket.fs_full_grad_total = bucket_local_full_size
                                bucket.dion_grad_buffer = None

                                # Create param ID → offset mapping for backward hook
                                # Rebind dion_param_layout params to bucket.params; handle new param objects after checkpoint
                                stale_name_map = getattr(self, "param_to_name", {}) or getattr(buffer, "param_to_name", {})
                                # Build runtime name map from current module params (up-to-date objects)
                                runtime_name_map = {}
                                if hasattr(self, "model_chunks"):
                                    for i, model in enumerate(self.model_chunks):
                                        try:
                                            # Check if named_parameters exists
                                            named_params_method = getattr(model, 'named_parameters', None)
                                            if named_params_method is None or not callable(named_params_method):
                                                continue

                                            try:
                                                params_list = list(named_params_method())
                                            except Exception as e_list:
                                                logger.warning(f"[Dion] model_chunks[{i}] list() failed: {e_list}")
                                                continue

                                            for j, (n, p) in enumerate(params_list):
                                                try:
                                                    # Use id(p) as key, NOT p directly (avoid Tensor comparison in dict)
                                                    runtime_name_map[id(p)] = n
                                                except Exception as e_inner:
                                                    if j < 3:
                                                        logger.warning(f"[Dion] param {j} assignment failed: {e_inner}")

                                        except Exception as e:
                                            logger.warning(f"[Dion] model_chunks[{i}] iteration failed: {e}")
                                            continue
                                elif hasattr(self, "module") and isinstance(self.module, torch.nn.Module):
                                    # module is a single module; safe to iterate
                                    runtime_name_map = {p: n for n, p in self.module.named_parameters()}
                                    runtime_name_map.update({id(p): n for n, p in self.module.named_parameters()})
                                active_name_map = runtime_name_map or stale_name_map or {}
                                if not active_name_map:
                                    raise RuntimeError(
                                        f"[Dion] buffer={gbuf_idx} bucket={bucket.bucket_id} "
                                        f"no active name map (model_chunks/module missing or empty)"
                                    )

                                bucket_param_by_name = {}
                                # First pass: direct lookup with active_name_map (runtime preferred)
                                for p in bucket.params:
                                    n = active_name_map.get(id(p), None)  # Use id(p) only!
                                    if n and n not in bucket_param_by_name:
                                        bucket_param_by_name[n] = p
                                # Second pass: use stale_name_map if still missing
                                matched_ids = {id(v) for v in bucket_param_by_name.values()}
                                for p in bucket.params:
                                    if id(p) in matched_ids:
                                        continue
                                    n = stale_name_map.get(id(p), None)  # Use id(p) only!
                                    if n and n not in bucket_param_by_name:
                                        bucket_param_by_name[n] = p
                                # Third pass: direct search over model_chunks/module if still unnamed
                                matched_ids = {id(v) for v in bucket_param_by_name.values()}
                                for p in bucket.params:
                                    if id(p) in matched_ids:
                                        continue
                                    found_name = None
                                    if hasattr(self, "model_chunks"):
                                        for model in self.model_chunks:
                                            try:
                                                for n, param_obj in model.named_parameters():
                                                    if id(param_obj) == id(p):
                                                        found_name = n
                                                        break
                                                if found_name:
                                                    break
                                            except Exception:
                                                continue
                                    if found_name is None and hasattr(self, "module") and isinstance(self.module, torch.nn.Module):
                                        for n, param_obj in self.module.named_parameters():
                                            if id(param_obj) == id(p):
                                                found_name = n
                                                break
                                    if found_name and found_name not in bucket_param_by_name:
                                        bucket_param_by_name[found_name] = p

                                # Shape-based fallback using old entries (ensure unique match)
                                matched_ids = {id(v) for v in bucket_param_by_name.values()}
                                for p in bucket.params:
                                    if id(p) in matched_ids:
                                        continue
                                    candidates = []
                                    for entry in bucket_dion_param_layout:
                                        if tuple(p.shape) == tuple(entry["param"].shape):
                                            candidates.append(entry)
                                    if len(candidates) == 1:
                                        entry = candidates[0]
                                        n_old = (
                                            active_name_map.get(id(entry["param"]), None)
                                            or stale_name_map.get(id(entry["param"]), None)
                                        )
                                        if n_old and n_old not in bucket_param_by_name:
                                            bucket_param_by_name[n_old] = p
                                    elif len(candidates) > 1:
                                        logger.warning(
                                            f"[Dion] buffer={gbuf_idx} bucket={bucket.bucket_id} "
                                            f"param shape={tuple(p.shape)} has multiple dion_param_layout matches; skipping fallback"
                                        )

                                rebuilt_entries = []
                                rebuilt_dion_param_shard_range = {}  # Rebuild with current param objects for PP/VPP
                                bucket.fs_param_id_to_full_offset = {}
                                offset = 0
                                for entry in bucket_dion_param_layout:
                                    n = None
                                    if active_name_map:
                                        n = active_name_map.get(id(entry["param"]), None)
                                    if n is None and stale_name_map:
                                        n = stale_name_map.get(id(entry["param"]), None)
                                    target_param = bucket_param_by_name.get(n, None)
                                    if target_param is None:
                                        raise RuntimeError(
                                            f"[Dion] buffer={gbuf_idx} bucket={bucket.bucket_id} "
                                            f"name={n} shape={entry['param'].shape} not found in current bucket params"
                                        )
                                    new_entry = entry.copy()
                                    new_entry["param"] = target_param
                                    rebuilt_entries.append(new_entry)
                                    # Use padded segment_size for consistent RS chunk
                                    seg_size = entry["segment_size"]
                                    bucket.fs_param_id_to_full_offset[id(target_param)] = offset

                                    # Rebuild dion_param_shard_range with current param objects
                                    pack_offset = entry.get('pack_offset', 0)
                                    local_size = entry['local_shape'][0] * entry['local_shape'][1]
                                    rebuilt_dion_param_shard_range[target_param] = (pack_offset, pack_offset + local_size)

                                    offset += seg_size

                                bucket.dion_param_layout = rebuilt_entries
                                bucket.dion_param_shard_range = rebuilt_dion_param_shard_range
                                # FS shard size uses padded segment_size sum
                                bucket.dion_shard_size = sum(entry["segment_size"] for entry in bucket.dion_param_layout)

                                # Override full size with padded shard*fs_size to match RS buffers
                                bucket_full_size = bucket.dion_shard_size * fs_size
                                bucket.fs_full_grad_total = bucket_full_size

                                # Store FS group for reduce-scatter
                                bucket.dion_comm_group = fs_group

                                # Store CP group for post-RS all-reduce
                                from megatron.core import parallel_state as ps
                                cp_size = ps.get_context_parallel_world_size()
                                if cp_size > 1:
                                    bucket.cp_group = ps.get_context_parallel_group()
                                else:
                                    bucket.cp_group = None

                                # Build name-to-entry lookup for DDP hook
                                # DDP hook can't match by object identity, so we use param name
                                dion_param_name_to_entry = {}
                                for entry in bucket.dion_param_layout:
                                    entry_param = entry.get("param")
                                    if entry_param is not None:
                                        # Try to get name from buffer's param_to_name
                                        param_name = None
                                        if hasattr(buffer, 'param_to_name'):
                                            param_name = buffer.param_to_name.get(entry_param)
                                        # Fallback to name_map
                                        if param_name is None:
                                            param_name = active_name_map.get(id(entry_param)) or stale_name_map.get(id(entry_param))
                                        if param_name:
                                            dion_param_name_to_entry[param_name] = entry
                                bucket.dion_param_name_to_entry = dion_param_name_to_entry

                                # Pack total for lazy AG buffer allocation
                                bucket_pack_total = max(
                                    entry['pack_offset'] + entry['segment_size']
                                    for entry in bucket_dion_param_layout
                                )
                                bucket.fs_pack_total = bucket_pack_total
                                bucket.fs_pack_buffer = None
                                bucket.fs_gathered_buffer = None

                                # Register fs_all_gather_fn for forward pass
                                # start_param_sync() will call this to restore FS × TP sharding
                                # This function handles Dion params only
                                bucket.fs_all_gather_fn = lambda async_op=False, b=buffer, bkt=bucket, plan=bucket.dion_param_layout: self._all_gather_params_bucket(b, bkt, plan, async_op=async_op)

                                # Mixed buckets use fs_all_gather_fn, set param_data=None to skip DO path
                                bucket.param_data = None

                            else:
                                # non-Dion bucket or missing FS metadata
                                has_dion = any(getattr(p, "is_dion_param", False) for p in bucket.params)
                                if has_dion:
                                    name_map = getattr(self, "param_to_name", {}) or getattr(buffer, "param_to_name", {})
                                    dion_params = []
                                    for p in bucket.params:
                                        if getattr(p, "is_dion_param", False):
                                            n = None
                                            if name_map:
                                                n = name_map.get(p, None) or name_map.get(id(p), None)
                                            dion_params.append((id(p), n or f"id_{id(p)}", tuple(p.shape)))
                                    logger.error(f"[Dion] Buffer {gbuf_idx} Bucket {bucket.bucket_id} has Dion params but no dion_param_layout. params={dion_params}")
                                    raise RuntimeError(f"[Dion] Buffer {gbuf_idx} Bucket {bucket.bucket_id} has Dion params but no dion_param_layout")

                                bucket.dion_param_layout = DionParamLayout()
                                bucket.dion_grad_buffer = None
                                bucket.fs_full_grad_total = 0
                                bucket.fs_param_id_to_full_offset = {}
                                bucket.dion_shard_size = 0
                                bucket.dion_comm_group = fs_group
                                # Store CP group
                                from megatron.core import parallel_state as ps
                                cp_size = ps.get_context_parallel_world_size()
                                bucket.cp_group = ps.get_context_parallel_group() if cp_size > 1 else None
                                bucket.fs_pack_total = 0
                                bucket.fs_pack_buffer = None
                                bucket.fs_gathered_buffer = None
                                bucket.fs_all_gather_fn = None

                        # Setup non-Dion params in mixed buckets
                        for bucket in buffer.buckets:
                            # Check if this is a mixed bucket
                            has_dion = hasattr(bucket, 'dion_param_layout') and bucket.dion_param_layout and len(bucket.dion_param_layout) > 0
                            fs_param_ids = {id(e['param']) for e in (bucket.dion_param_layout if has_dion else [])}

                            if has_dion:
                                # Use gbuf_ranges param_map order to match main_grad binding
                                bucket_plan = buffer.fs_bucket_plan_map.get(bucket.bucket_id)
                                if bucket_plan:
                                    dtype_key = bucket_plan["dtype"]
                                    gbuf_range_map = self.gbuf_ranges[gbuf_idx][dtype_key][bucket.bucket_id]
                                    param_map = gbuf_range_map.get("param_map", {})

                                    # Get non-Dion params in param_map order (same order as main_grad binding)
                                    non_dion_params_in_bucket = [
                                        p for p in param_map.keys()
                                        if id(p) not in fs_param_ids
                                    ]
                                else:
                                    # Fallback to bucket.params order
                                    non_dion_params_in_bucket = [p for p in bucket.params if id(p) not in fs_param_ids]
                            else:
                                non_dion_params_in_bucket = [p for p in bucket.params if id(p) not in fs_param_ids]

                            if has_dion and non_dion_params_in_bucket:
                                dp_group = self.fs_group
                                dp_size = dp_group.size()

                                non_dion_pack_plan = []
                                non_dion_param_ranges = {}
                                pack_offset = 0  # For RS output layout (shard-based)
                                input_offset = 0  # For input buffer layout (full param-based)
                                dion_shard_size = getattr(bucket, 'dion_shard_size', 0)

                                for param in non_dion_params_in_bucket:
                                    param_numel = param.numel()
                                    shard_size = math.ceil(param_numel / dp_size)  # DP flat sharding

                                    # Calculate absolute bucket offsets (RS output position)
                                    # RS writes to: bucket.grad_data[dion_shard_size + pack_offset]
                                    section_start = dion_shard_size + pack_offset
                                    section_end = section_start + shard_size

                                    non_dion_pack_plan.append({
                                        'param': param,
                                        'pack_offset': pack_offset,  # RS output offset (shard-based)
                                        'input_offset': input_offset,  # Input buffer offset (full param-based)
                                        'segment_size': shard_size,
                                        'param_numel': param_numel,
                                        'section_start': section_start,  # Absolute bucket offset
                                        'section_end': section_end,
                                    })

                                    # Store in dict for O(1) lookup (like dion_param_shard_range)
                                    non_dion_param_ranges[param] = (section_start, section_end)

                                    pack_offset += shard_size
                                    input_offset += param_numel

                                bucket.non_dion_pack_plan = non_dion_pack_plan
                                bucket.non_dion_param_ranges = non_dion_param_ranges  # For main_grad binding
                                bucket.non_dion_pack_total = pack_offset
                                bucket.non_dion_shard_size = pack_offset  # Total shard size for all non-Dion params
                                bucket.non_dion_full_grad_total = pack_offset * dp_size
                                bucket.non_dion_full_grad_buffer = torch.zeros(
                                    bucket.non_dion_full_grad_total,
                                    dtype=buffer.grad_dtype,
                                    device=torch.cuda.current_device(),
                                )
                                # Store buffer size at creation time for reliable offload/reload
                                bucket._non_dion_full_grad_buffer_size = bucket.non_dion_full_grad_total
                                # Store DP group for RS/AG
                                bucket.non_dion_dp_group = dp_group
                                bucket.non_dion_all_gather_fn = lambda async_op=False, b=buffer, bkt=bucket: self._all_gather_non_dion_params(bkt, b, async_op)

                                # Build name-to-entry lookup for DDP hook
                                # DDP hook can't match by object identity, so we use param name
                                non_dion_param_name_to_entry = {}
                                for entry in non_dion_pack_plan:
                                    entry_param = entry.get("param")
                                    if entry_param is not None:
                                        param_name = None
                                        if hasattr(buffer, 'param_to_name'):
                                            param_name = buffer.param_to_name.get(entry_param)
                                        if param_name is None:
                                            param_name = active_name_map.get(id(entry_param)) or stale_name_map.get(id(entry_param))
                                        if param_name:
                                            non_dion_param_name_to_entry[param_name] = entry
                                bucket.non_dion_param_name_to_entry = non_dion_param_name_to_entry

                                # Verify RS output offset matches main_grad binding offset
                                dion_shard_size = getattr(bucket, 'dion_shard_size', 0)
                                mismatch_found = False
                                for entry in non_dion_pack_plan:
                                    param = entry['param']
                                    pack_offset_entry = entry['pack_offset']
                                    expected_gbuf_local_start = dion_shard_size + pack_offset_entry

                                    # Get gbuf_local from param_map
                                    if param in param_map:
                                        gbuf_local = param_map[param].get("gbuf_local")
                                        if gbuf_local:
                                            actual_gbuf_local_start = gbuf_local.start
                                            if actual_gbuf_local_start != expected_gbuf_local_start:
                                                mismatch_found = True
                                                logger.error(
                                                    f"[RS/MAIN_GRAD MISMATCH] Bucket {bucket.bucket_id} param {param.shape}: "
                                                    f"RS writes to offset {expected_gbuf_local_start} "
                                                    f"(dion_shard_size={dion_shard_size} + pack_offset={pack_offset_entry}), "
                                                    f"but main_grad binds to {actual_gbuf_local_start}"
                                                )
                                # RS/MAIN_GRAD verified if not mismatch_found
                            else:
                                # Pure bucket: no non-Dion handling needed
                                bucket.non_dion_pack_plan = []
                                bucket.non_dion_full_grad_buffer = None
                                bucket.non_dion_full_grad_total = 0
                                bucket.non_dion_shard_size = 0
                                bucket.non_dion_pack_total = 0
                                bucket.non_dion_all_gather_fn = None
                                bucket.non_dion_dp_group = None

                            # dion_grad_buffer lazy allocated in backward hook


        # After parent init, setup Dion if applicable
        if hasattr(self, 'optimizer') and isinstance(self.optimizer, (MegatronDion,)):
            if hasattr(self, 'gbuf_ranges') and hasattr(self, 'buffers'):
                # Annotate and setup Dion parameters
                self._annotate_dion_parameters_batch()

                # Rebuild parameter groups with batch support
                if hasattr(self, 'opt_group_ranges'):
                    (
                        self.model_float16_groups,
                        self.model_fp32_groups,
                        self.shard_float16_groups,
                        self.shard_fp32_groups,
                        self.shard_fp32_from_float16_groups,
                    ) = self._build_model_and_main_param_groups_batch(
                        self.gbuf_ranges,
                        self.model_param_gbuf_map,
                        self.opt_group_ranges,
                        self.config
                    )

                    # Update optimizer param groups
                    self._update_optimizer_param_groups()

                    # Rebuild _dion_shard_info opt_shard to point to actual tensors in optimizer.param_groups
                    rebuild_count = 0
                    new_mapping_count = 0
                    log_count = 0
                    for group in self.optimizer.param_groups:
                        for p in group['params']:
                            if hasattr(p, '_model_param'):
                                model_param = p._model_param
                                old_opt_shard = self._get_opt_shard(model_param)

                                # Update _dion_shard_info with new opt_shard
                                if old_opt_shard is None:
                                    # First time: log but cannot create new entry (need full shard info)
                                    new_mapping_count += 1
                                    pass  # New mapping detected
                                elif old_opt_shard is not p:
                                    # Remapping: optimizer tensor changed
                                    # Copy _dion_param_uid from old tensor to new tensor
                                    if hasattr(old_opt_shard, '_dion_param_uid'):
                                        p._dion_param_uid = old_opt_shard._dion_param_uid
                                    self._update_opt_shard(model_param, p)
                                    rebuild_count += 1

                    # Rebuild opt_shard for consistent object identity
                    if hasattr(self, 'shard_fp32_from_float16_groups'):
                        fallback_count = 0
                        for model_group, shard_group in zip(self.model_float16_groups, self.shard_fp32_from_float16_groups):
                            for model_param, shard_param in zip(model_group, shard_group):
                                if shard_param is not None:
                                    self._update_opt_shard(model_param, shard_param)
                                    fallback_count += 1

                    # Pass _dion_shard_info to Dion optimizer for consistent shard references
                    self.optimizer._distrib_dion_shard_info = self._dion_shard_info

                    # Rebind main_grad after _update_optimizer_param_groups() replaces param_groups

                    # Rebind main_grad for ALL parameters in model_float16_groups
                    total_rebound = 0
                    total_not_found = 0
                    for group in self.model_float16_groups:
                        for model_param in group:
                            # Use model_param_gbuf_map to get buffer location
                            if model_param in self.model_param_gbuf_map:
                                gbuf_idx, dtype, bucket_idx = self.model_param_gbuf_map[model_param]
                                buffer = self.buffers[gbuf_idx]

                                # Get param range from gbuf_ranges using the indices
                                gbuf_range_map = self.gbuf_ranges[gbuf_idx][dtype][bucket_idx]

                                if model_param in gbuf_range_map['param_map']:
                                    range_info = gbuf_range_map['param_map'][model_param]
                                    dion_info = range_info.get("dion_info", {})

                                    bucket = buffer.buckets[bucket_idx]

                                    # Skip main_grad rebind - keep FULL shape for TE GEMM compatibility
                                    is_dion_param = hasattr(bucket, 'dion_param_shard_range') and model_param in bucket.dion_param_shard_range
                                    is_non_dion_param = hasattr(bucket, 'non_dion_param_ranges') and model_param in bucket.non_dion_param_ranges

                                    if is_dion_param or is_non_dion_param:
                                        total_rebound += 1
                                        continue
                                    else:
                                        # Skip uncategorized params to preserve FULL main_grad shape
                                        if total_not_found < 5:
                                            logger.warning(f"[Dion] Uncategorized param {model_param.shape}, skipping rebind")
                                        total_not_found += 1
                                        continue
                                else:
                                    logger.warning(f"[Dion] Model param in model_param_gbuf_map but NOT in gbuf_range_map['param_map']!")
                                    total_not_found += 1
                            else:
                                total_not_found += 1

                    if total_not_found > 0:
                        logger.warning(f"[Dion] {total_not_found} params in model_float16_groups NOT found in model_param_gbuf_map!")

                    # Also rebind main_grad for FP32 params (bias, layernorm)

                    total_rebound_fp32 = 0
                    total_not_found_fp32 = 0
                    for group in self.model_fp32_groups:
                        for model_param in group:
                            # Use model_param_gbuf_map to get buffer location
                            if model_param in self.model_param_gbuf_map:
                                gbuf_idx, dtype, bucket_idx = self.model_param_gbuf_map[model_param]
                                buffer = self.buffers[gbuf_idx]

                                # Get param range from gbuf_ranges using the indices
                                gbuf_range_map = self.gbuf_ranges[gbuf_idx][dtype][bucket_idx]

                                if model_param in gbuf_range_map['param_map']:
                                    range_info = gbuf_range_map['param_map'][model_param]
                                    dion_info = range_info.get("dion_info", {})

                                    bucket = buffer.buckets[bucket_idx]

                                    # Keep main_grad FULL for TE GEMM compatibility

                                    is_dion_param = hasattr(bucket, 'dion_param_shard_range') and model_param in bucket.dion_param_shard_range
                                    is_non_dion_param = hasattr(bucket, 'non_dion_param_ranges') and model_param in bucket.non_dion_param_ranges

                                    if is_dion_param or is_non_dion_param:
                                        total_rebound_fp32 += 1
                                        continue
                                    else:
                                        # Skip uncategorized params to preserve FULL main_grad shape
                                        if total_not_found_fp32 < 5:
                                            logger.warning(f"[Dion] Uncategorized param {model_param.shape}, skipping rebind")
                                        total_not_found_fp32 += 1
                                        continue
                                else:
                                    total_not_found_fp32 += 1
                            else:
                                total_not_found_fp32 += 1

                    if total_not_found_fp32 > 0:
                        logger.warning(f"[Dion] {total_not_found_fp32} params in model_fp32_groups NOT found in model_param_gbuf_map!")

                    # Initialize overlap flags before _enable_dion_distributed_mode_improved()
                    self._enable_overlap_grad_reduce = bool(getattr(self.ddp_config, "overlap_grad_reduce", False))
                    self._enable_overlap_param_gather = bool(getattr(self.ddp_config, "overlap_param_gather", False))

                    # Enable distributed mode with improved configuration
                    try:
                        self._enable_dion_distributed_mode_improved()
                    except Exception as e:
                        logger.error(f"[Dion] Global rank {self._global_rank}: Failed in _enable_dion_distributed_mode_improved: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
                        raise

                    # Mixed bucket adapters registered in _enable_dion() after buffer_indices populated

        # param.data stays FULL, bucket.param_data=None for mixed buckets

        # DP barrier to ensure all ranks finish initialization together
        if dist.is_initialized() and hasattr(self, 'data_parallel_group'):
            dist.barrier(group=self.data_parallel_group)

    def _all_gather_params_bucket(self, buffer, bucket, dion_param_layout, async_op=False):
        """
        All-gather Dion parameters for one bucket from FS shards to full parameters.

        Uses pack/unpack approach like Megatron-Core DO standard:
        1. Pack all local shards into single pack_buffer
        2. Single all_gather_into_tensor call (async-capable, no shard_list allocation)
        3. Unpack full parameters from gathered_buffer

        This eliminates per-parameter shard_list allocation, preventing memory leaks.

        Args:
            buffer: ParamAndGradBuffer for this bucket group
            bucket: Specific bucket with cached workspace
            dion_param_layout: List of pack entries with Dion parameter metadata
            async_op: If True, launch async all-gather and return handle dict

        Returns:
            Dict with handle and unpack info if async_op=True, None otherwise
        """
        global_rank = self._global_rank

        if not dion_param_layout:
            return None

        # Use cached instance attributes for FS group info
        fs_group, fs_size, fs_rank = self.fs_group, self.fs_size, self.fs_rank
        if fs_group is None or fs_size == 1:
            return None


        # Verify layout consistency across FS group
        plan_len = len(dion_param_layout)
        plan_lens = [None] * fs_size
        dist.all_gather_object(plan_lens, plan_len, group=fs_group)
        if len(set(plan_lens)) != 1:
            logger.error(f"[Dion] Rank {global_rank} (FS rank {fs_rank}): local_len={plan_len}, all_lens={plan_lens}")
            raise RuntimeError(f"dion_param_layout length mismatch across FS ranks in bucket {getattr(bucket, 'bucket_id', -1)}")

        # Allocate cached workspace on first use
        pack_total = getattr(bucket, "fs_pack_total", None)
        if pack_total is None:
            raise RuntimeError(f"[Dion] Rank {global_rank}: fs_pack_total is None for bucket {bucket.bucket_id}")

        if bucket.fs_pack_buffer is None or bucket.fs_pack_buffer.numel() != pack_total:
            # Delete old buffer before reallocating to prevent memory leak
            if bucket.fs_pack_buffer is not None:
                logger.warning(f"[Dion] Rank {global_rank}: fs_pack_buffer size changed! Old={bucket.fs_pack_buffer.numel()}, New={pack_total}")
                del bucket.fs_pack_buffer
            bucket.fs_pack_buffer = torch.zeros(
                pack_total,
                dtype=buffer.param_dtype,
                device=torch.cuda.current_device(),
            )
        if bucket.fs_gathered_buffer is None or bucket.fs_gathered_buffer.numel() != pack_total * fs_size:
            # Delete old buffer before reallocating to prevent memory leak
            if bucket.fs_gathered_buffer is not None:
                logger.warning(f"[Dion] Rank {global_rank}: fs_gathered_buffer size changed! Old={bucket.fs_gathered_buffer.numel()}, New={pack_total * fs_size}")
                del bucket.fs_gathered_buffer
            bucket.fs_gathered_buffer = torch.zeros(
                pack_total * fs_size,
                dtype=buffer.param_dtype,
                device=torch.cuda.current_device(),
            )

        pack_buffer = bucket.fs_pack_buffer
        gathered_buffer = bucket.fs_gathered_buffer

        # Verify pack_buffer size consistency across FS group
        sizes = [None] * fs_size
        dist.all_gather_object(sizes, pack_total, group=fs_group)
        if len(set(sizes)) != 1:
            logger.error(f"[Dion] Rank {global_rank} (FS rank {fs_rank}):")
            logger.error(f"  Local pack_buffer.numel(): {pack_total}")
            logger.error(f"  All FS group sizes: {sizes}")
            logger.error(f"  dion_param_layout_len: {len(dion_param_layout)}")
            raise RuntimeError(f"FS all-gather input size mismatch at rank {global_rank}")

        # Pack local shards into pack_buffer
        for entry in dion_param_layout:
            param = entry['param']
            local_shape = entry['local_shape']
            pack_offset = entry['pack_offset']
            segment_size = entry['segment_size']
            fs_split_dim = entry['fs_split_dim']
            start_idx = entry['start_idx']
            end_idx = entry['end_idx']

            # Get param name for logging
            pname = self.param_to_name.get(param, f"id_{id(param)}") if hasattr(self, "param_to_name") else f"id_{id(param)}"

            # Validate segment_size (padding allowed, reject if too small)
            expected_numel = local_shape[0] * local_shape[1]
            if segment_size < expected_numel:
                raise RuntimeError(
                    f"[Dion] segment_size too small: seg={segment_size} expected>={expected_numel} "
                    f"shape={local_shape} param={pname}"
                )
            # Get local FS shard (reuse existing tensor to prevent memory leak)
            expected_shape = tuple(local_shape)
            existing_shard = self._get_data_shard(param)

            if existing_shard is not None and existing_shard.shape == expected_shape:
                # Reuse existing tensor - copy fresh data in-place (no new allocation)
                # Use .data.copy_() to completely bypass autograd (existing_shard may be a leaf variable with requires_grad)
                if param.data.shape == expected_shape:
                    existing_shard.data.copy_(param.data)
                elif fs_split_dim == 0:
                    existing_shard.data.copy_(param.data[start_idx:end_idx, :])
                else:
                    existing_shard.data.copy_(param.data[:, start_idx:end_idx])
                local_shard_2d = existing_shard
            else:
                # First allocation - use detach().clone() for separate memory
                if param.data.shape == expected_shape:
                    local_shard_2d = param.data.detach().clone()
                elif fs_split_dim == 0:
                    local_shard_2d = param.data[start_idx:end_idx, :].detach().clone()
                else:
                    local_shard_2d = param.data[:, start_idx:end_idx].detach().clone()
            # Flatten and pack
            shard_numel = local_shape[0] * local_shape[1]
            local_shard = local_shard_2d.view(-1).to(pack_buffer.device, non_blocking=True)

            # No separate workspace - param.data is the only source
            # Fail-fast if shard is zero (indicates initialization issue)
            if local_shard.numel() > 0 and local_shard.abs().max().item() == 0:
                raise RuntimeError(
                    f"[Dion] rank={fs_rank} bucket={bucket.bucket_id} param={pname}\n"
                    f"  start={start_idx} end={end_idx} fs_dim={fs_split_dim}\n"
                    f"  param.data.shape={tuple(param.data.shape)}, param.data.max={param.data.abs().max().item():.4e}\n"
                    f"  param.data shard is zero - check initialization!"
                )

            # Update mappings only for new allocations
            if existing_shard is None or existing_shard.shape != expected_shape:
                # New allocation - update mappings
                self._update_data_shard(param, local_shard_2d)
                param._fs_shard = local_shard_2d

            # Pack into buffer
            pack_buffer[pack_offset:pack_offset + shard_numel].copy_(local_shard[:shard_numel])

            # Zero padding if needed
            if shard_numel < segment_size:
                pack_buffer[pack_offset + shard_numel:pack_offset + segment_size].zero_()

        # All-gather using cached gathered_buffer

        # Use standard DO API for async support
        handle = torch.distributed.all_gather_into_tensor(
            output_tensor=gathered_buffer,
            input_tensor=pack_buffer,
            group=fs_group,
            async_op=async_op
        )

        if async_op:
            # Non-Dion params: AG only in pure non-Dion buckets, mixed buckets skip

            # Return handle and unpack metadata for finish_param_sync()
            return {
                'handle': handle,
                'gathered_buffer': gathered_buffer,
                'pack_buffer': pack_buffer,  # Keep alive until wait()
                'dion_param_layout': dion_param_layout,
                'pack_total': pack_total,
                'fs_size': fs_size,
                'buffer': buffer,
                'bucket': bucket,
                'optimizer': self,  # Store optimizer reference for unpack!
            }
        else:
            # Synchronous mode: unpack immediately
            self._unpack_all_gathered_params(
                gathered_buffer=gathered_buffer,
                dion_param_layout=dion_param_layout,
                pack_total=pack_total,
                fs_size=fs_size,
                buffer=buffer,
            )

            # Release AG buffers (use = None, not resize_(0), to avoid view issues)
            if hasattr(bucket, 'fs_pack_buffer') and bucket.fs_pack_buffer is not None:
                bucket.fs_pack_buffer = None
            if hasattr(bucket, 'fs_gathered_buffer') and bucket.fs_gathered_buffer is not None:
                bucket.fs_gathered_buffer = None

            # Mixed buckets skip non-Dion AG here; pure non-Dion use standard DO AG

            return None

    def _all_gather_non_dion_params(self, bucket, buffer, async_op=False):
        """
        All-gather Non-Dion parameters for Mixed buckets.

        Non-Dion params (bias, layernorm, etc.) use standard DP sharding.
        In forward pass, they need DP all-gather to restore full size.

        Args:
            bucket: _ParamAndGradBucket with Non-Dion params
            buffer: ParamAndGradBuffer
            async_op: If True, return async handles; if False, execute synchronously

        Returns:
            Dict with handles if async_op=True, None otherwise
        """
        global_rank = self._global_rank

        # Use non_dion_pack_plan (populated after is_dion_param annotation)
        if not hasattr(bucket, 'non_dion_pack_plan') or not bucket.non_dion_pack_plan:
            return None

        dp_group = getattr(bucket, 'non_dion_dp_group', None) or buffer.data_parallel_group
        dp_size = dp_group.size()
        dp_rank = dp_group.rank()


        if dp_size == 1:
            # No all-gather needed for single rank
            return None

        # Non-Dion params stay FULL (only gradient is DP-sharded), just verify
        for idx, entry in enumerate(bucket.non_dion_pack_plan):
            param = entry['param']
            expected_full_numel = entry['param_numel']
            actual_numel = param.data.numel()

            # Non-Dion params should always be FULL - fail fast if not
            if actual_numel != expected_full_numel:
                raise RuntimeError(
                    f"[NON-Dion AG ERROR] Rank {global_rank}: param {idx} is unexpectedly SHARD. "
                    f"actual_numel={actual_numel}, expected_full_numel={expected_full_numel}. "
                    f"Non-Dion params should always have FULL param.data shape."
                )

        return None

    def _unpack_all_gathered_params(self, gathered_buffer, dion_param_layout, pack_total, fs_size, buffer):
        """
        Unpack full parameters from gathered_buffer and rebind param.data.

        This is called either:
        - Immediately after sync all-gather (_all_gather_params_bucket with async_op=False)
        - In finish_param_sync() after async all-gather completes

        Args:
            gathered_buffer: (pack_total * fs_size,) buffer with all gathered shards
            dion_param_layout: List of pack entries with metadata
            pack_total: Size of pack_buffer per rank
            fs_size: FS group size
            buffer: ParamAndGradBuffer
        """
        global_rank = self._global_rank
        # Use instance attributes directly
        fs_group, fs_rank = self.fs_group, self.fs_rank

        for entry_idx, entry in enumerate(dion_param_layout):
            param = entry['param']
            pack_offset = entry['pack_offset']
            size_per_rank = entry['size_per_rank']
            fs_split_dim = entry['fs_split_dim']
            global_shape = entry['global_shape']
            # FS restores one dimension; other stays TP-local
            if fs_split_dim == 0:
                # Row split: rows become full, cols stay TP-sharded
                m_full = global_shape[0]
                n_full = param.shape[1]
            else:
                # Col split: cols become full, rows stay TP-sharded
                m_full = param.shape[0]
                n_full = global_shape[1]

            # Use param.data directly (FULL shape bucket view)
            full_param_flat = param.data.view(-1)

            split_dim_size = m_full if fs_split_dim == 0 else n_full

            for rank_i in range(fs_size):
                # This rank's segment in gathered buffer
                rank_pack_offset = rank_i * pack_total + pack_offset
                segment_size = entry['segment_size']
                rank_segment = gathered_buffer[rank_pack_offset:rank_pack_offset + segment_size]

                start_idx, end_idx = compute_fs_shard_range(split_dim_size, fs_size, rank_i)
                actual_size = end_idx - start_idx

                # Copy based on split dimension
                if fs_split_dim == 0:
                    # Row split: copy to rows [start_idx:end_idx]
                    full_param_flat[start_idx * n_full:end_idx * n_full].copy_(rank_segment[:actual_size * n_full])
                else:
                    # Col split: copy to cols [start_idx:end_idx] (vectorized)
                    rank_segment_2d = rank_segment[:actual_size * m_full].view(m_full, actual_size)
                    full_param_2d = full_param_flat.view(m_full, n_full)
                    full_param_2d[:, start_idx:end_idx].copy_(rank_segment_2d)

            # param.data already updated in-place (no copy needed)

            # Update param._fs_shard after all-gather to keep it in sync with param.data
            fs_group, fs_size, fs_rank = self.fs_group, self.fs_size, self.fs_rank

            # Calculate current rank's shard range
            full_m, full_n = param.data.shape
            split_dim_size = full_m if fs_split_dim == 0 else full_n
            my_start_idx, my_end_idx = compute_fs_shard_range(split_dim_size, fs_size, fs_rank)

            # Extract current rank's shard from full parameter (now in param.data)
            full_param_2d = param.data  # Already in 2D shape (full_m, full_n)

            # Reuse existing _fs_shard tensor to prevent memory leak
            if fs_split_dim == 0:
                shard_view = full_param_2d[my_start_idx:my_end_idx, :]
                expected_shape = (my_end_idx - my_start_idx, full_n)
            else:
                shard_view = full_param_2d[:, my_start_idx:my_end_idx]
                expected_shape = (full_m, my_end_idx - my_start_idx)

            if hasattr(param, '_fs_shard') and param._fs_shard is not None and param._fs_shard.shape == expected_shape:
                # Reuse existing tensor (in-place copy, no new allocation)
                # Use .data.copy_() to bypass autograd
                param._fs_shard.data.copy_(shard_view)
            else:
                # First allocation - use detach().clone() for separate memory
                param._fs_shard = shard_view.detach().clone()

            # Also reuse data_shard tensor in _dion_shard_info
            existing_data_shard = self._get_data_shard(param)
            if existing_data_shard is not None and existing_data_shard.shape == expected_shape:
                # Reuse existing (in-place copy) - use .data.copy_() to bypass autograd
                existing_data_shard.data.copy_(shard_view)
            else:
                # First allocation - share with _fs_shard (same tensor)
                self._update_data_shard(param, param._fs_shard)

    def _annotate_dion_parameters_batch(self):
        """
        Improved annotation with batch processing for efficiency.
        Uses original model parameters directly (independent of FS sharding).
        """
        from ..parallel_state import get_tensor_model_parallel_rank, get_data_parallel_rank

        # Use FS group for annotation (each FS group operates independently)
        # Parameters are sharded at annotation phase
        # Each FS group will split parameters uniformly across FS ranks
        # RP groups replicate the same FS sharding pattern

        # Get FS and RP info from optimizer
        fs_group = self.optimizer.fs_group if hasattr(self.optimizer, 'fs_group') else None
        rp_group = self.optimizer.rp_group if hasattr(self.optimizer, 'rp_group') else None

        if fs_group is not None:
            annotation_group = fs_group  # FS group (size=2)
            fs_rank = dist.get_rank(fs_group)
            fs_size = dist.get_world_size(fs_group)
        else:
            # Fallback to DP group if FS not configured
            annotation_group = self.data_parallel_group
            fs_rank = annotation_group.rank()
            fs_size = dist.get_world_size(annotation_group)

        device = torch.cuda.current_device()
        tp_rank = get_tensor_model_parallel_rank()
        global_rank = self._global_rank
        dp_rank = self.data_parallel_group.rank()  # Original DP rank for logging


        self._param_owner_ranks = {}
        self._fs_rank = fs_rank
        self._fs_size = fs_size

        # Process all parameters and create FS-aware shards
        # Parameters are sharded row-wise across FS group
        total_params = 0
        eligible_params = 0
        total_2d_params = 0

        for gbuf_idx, gbuf_range_maps in enumerate(self.gbuf_ranges):
            buffer = self.buffers[gbuf_idx]

            for dtype in sorted(gbuf_range_maps.keys(), key=lambda dt: str(dt)):
                gbuf_range_map_list = gbuf_range_maps[dtype]

                for bucket_idx, gbuf_range_map in enumerate(gbuf_range_map_list):
                    # Annotate with FS-aware sharding
                    for param in gbuf_range_map["param_map"].keys():
                        total_params += 1
                        pri = gbuf_range_map["param_map"][param]

                        # Get parameter name from model_chunks for eligibility check
                        param_name = None
                        for model_chunk in self.model_chunks:
                            for name, p in model_chunk.named_parameters():
                                if p is param:
                                    param_name = name
                                    break
                            if param_name:
                                break
                        # Store param_name in pri for eligibility check
                        pri["param_name"] = param_name

                        # Preserve existing dion_info from _build_gbuf_range_map if set
                        # Only initialize if not already set
                        if "dion_info" not in pri:
                            pri["dion_info"] = {"is_dion": False}

                        # Check 2D and eligibility
                        if param.ndim == 2:
                            total_2d_params += 1
                            if self._check_param_eligibility(param, pri, buffer):
                                # _create_fs_aware_shard completes dion_info with start_idx/end_idx
                                self._create_fs_aware_shard(param, pri, buffer, fs_rank, fs_size, gbuf_idx, bucket_idx)
                                eligible_params += 1

        # Convert FS group members to DP rank indices for owner_tuples
        # owner_tuples must be in DP rank space for later DP→world conversion
        if fs_group is not None and fs_size > 1:
            # Get FS group's world ranks
            fs_world_ranks = dist.get_process_group_ranks(fs_group)

            # Get DP group's world ranks to create world→DP mapping
            dp_world_ranks = dist.get_process_group_ranks(self.data_parallel_group)
            world_to_dp = {w: i for i, w in enumerate(dp_world_ranks)}

            # Verify all FS ranks are in DP group (sanity check)
            for w in fs_world_ranks:
                if w not in world_to_dp:
                    raise RuntimeError(
                        f"Global rank {global_rank}: FS group contains world rank {w} "
                        f"which is not in my DP group {dp_world_ranks}! "
                        f"This indicates incorrect group topology."
                    )

            # Convert FS world ranks to DP indices
            fs_dp_ranks = tuple(world_to_dp[w] for w in fs_world_ranks)
            self._global_owner_tuples.add(fs_dp_ranks)


        # Collect all unique owner_tuples across DP group (union across TP slices)
        local_owner_tuples = sorted(list(self._global_owner_tuples))
        dp_world_ranks = dist.get_process_group_ranks(self.data_parallel_group)
        dp_size = len(dp_world_ranks)

        gathered_tuples = [None] * dp_size
        dist.all_gather_object(gathered_tuples, local_owner_tuples, group=self.data_parallel_group)

        # Collect all unique tuples from all ranks (union instead of enforcing uniformity)
        all_unique_tuples = set()
        for tuples in gathered_tuples:
            if tuples:
                all_unique_tuples.update(tuples)

        # Use union of all tuples
        self._global_owner_tuples = all_unique_tuples

        # No barrier needed - each FS group operates independently
        # TP slices have different FS groups, barrier would hang

    def _create_fs_aware_shard(self, param, pri, buffer, fs_rank, fs_size, gbuf_idx, bucket_idx):
        """
        Create FS-aware shard metadata with tp_split_dim-aware orthogonal TP×FS sharding.

        All FS ranks have the full parameter (replicated).
        FS split dimension depends on tp_split_dim for orthogonal sharding:
        - tp_split_dim=0 (ColumnParallel): TP splits rows → FS splits cols → (m/tp, n/fs)
        - tp_split_dim=1 (RowParallel): TP splits cols → FS splits rows → (m/fs, n/tp)

        Args:
            param: Original 2D parameter
            pri: Parameter range info from gbuf_range_map
            buffer: Gradient buffer
            fs_rank: This rank's position in FS group (0, 1, ...)
            fs_size: FS group size (e.g., 2)
            gbuf_idx: Gradient buffer index
            bucket_idx: Bucket index within buffer
        """
        from ..parallel_state import get_tensor_model_parallel_world_size
        from megatron.core import parallel_state

        m, n = param.shape

        # Get TP info
        tp_split_dim = get_tp_split_dim(param)
        has_tp = is_tp_enabled(param)
        # Expert params use Expert TP world size
        is_expert = not getattr(param, 'allreduce', True)
        if is_expert and has_tp:
            tp_world_size = parallel_state.get_expert_tensor_parallel_world_size()
        else:
            tp_world_size = get_tensor_model_parallel_world_size() if has_tp else 1

        # FS split dimension is orthogonal to TP
        fs_split_dim = get_fs_split_dim(tp_split_dim)

        # Get split size along FS dimension (param.shape is FULL at this point)
        split_size = n if fs_split_dim == 1 else m

        start_idx, end_idx = compute_fs_shard_range(split_size, fs_size, fs_rank)

        # TP and FS shard orthogonal dimensions, so no adjustment needed

        local_shape = compute_local_shape(m, n, start_idx, end_idx, fs_split_dim)

        # Get buffer offset
        p_start, _, _ = buffer.param_index_map[param] if param in buffer.param_index_map else (0, 0, 0)

        # Use existing global_shape from _build_gbuf_range_map or fallback to param.shape
        existing_global_shape = pri.get("dion_info", {}).get("global_shape")
        if existing_global_shape:
            global_shape = existing_global_shape
        else:
            # Fallback: param.shape is FULL (TP-partitioned, not FS-partitioned)
            global_shape = param.shape

        # Mark as Dion with tp_split_dim-aware FS sharding metadata
        pri["dion_info"] = {
            "is_dion": True,
            "shape": local_shape,
            "global_shape": global_shape,  # FS-restored shape, TP still partitioned (CORRECT!)
            "start_idx": start_idx,  # General start index (for any split dimension)
            "end_idx": end_idx,      # General end index (for any split dimension)
            "fs_split_dim": fs_split_dim,    # FS split dimension (orthogonal to TP)
            "tp_split_dim": tp_split_dim,     # TP split dimension (consistent naming with fs_split_dim)
            "tp_world_size": tp_world_size,   # TP size for reference
            "param_start_in_buffer": p_start,
            "fs_owner_ranks": tuple(range(fs_size)),  # All FS ranks own this (different shards)
            "buffer_idx": gbuf_idx,  # For reverse lookup via buffer_indices
            "bucket_idx": bucket_idx,  # For reverse lookup via buffer_indices
        }

        # Store in dion_param_info
        self._dion_param_info[param] = pri["dion_info"]
        self._param_owner_ranks[param] = tuple(range(fs_size))


    def _check_param_eligibility(self, param, pri, buffer):
        """Check if parameter meets basic eligibility criteria for Dion.

        Dion Parameter Selection Rule:
        - **Dion**: 2D network weight parameters (Linear layer weights)
        - **non-Dion**:
          - Embedding layer (word_embeddings, position_embeddings)
          - Output layer (output_layer, lm_head)
          - 1D parameters (bias, scale, normalization)
          - Normalization layers (LayerNorm, RMSNorm - usually 1D)
        """
        # Track rejection stats
        if not hasattr(self, '_eligibility_stats'):
            self._eligibility_stats = {
                'manual_override': 0,
                'not_2d': 0,
                'float8': 0,
                'not_in_buffer': 0,
                'embedding_or_output': 0,
                'accepted': 0
            }

        # Manual override check
        manual = getattr(param, "use_dion", None)
        if manual is False:
            self._eligibility_stats['manual_override'] += 1
            return False

        # Rule 1: Only 2D parameters (1D parameters are non-Dion)
        # This excludes: bias, scale, normalization layer params
        if param.ndim != 2:
            self._eligibility_stats['not_2d'] += 1
            return False

        # Rule 2: Exclude embedding, output, and normalization layers
        # Check parameter name for keywords
        # Try to get param_name from pri (set during annotation), fallback to param_to_name
        param_name = pri.get("param_name", None)
        if param_name is None and hasattr(self, 'param_to_name') and param in self.param_to_name:
            param_name = self.param_to_name[param]

        if param_name:
            # Exclude embedding/output/norm layers
            exclude_keywords = ['embedding', 'word_embeddings', 'position_embeddings',
                               'output_layer', 'lm_head', 'vocab',
                               'norm', 'layernorm', 'rmsnorm', 'groupnorm', 'batchnorm']
            for keyword in exclude_keywords:
                if keyword in param_name.lower():
                    self._eligibility_stats['embedding_or_output'] += 1
                    return False

        # FP8 tensors not supported
        if is_float8tensor(param):
            self._eligibility_stats['float8'] += 1
            return False

        # Rule 4: Must be in gradient buffer
        if param not in buffer.param_index_map:
            self._eligibility_stats['not_in_buffer'] += 1
            return False

        # All checks passed → This is a network weight parameter → Dion
        self._eligibility_stats['accepted'] += 1
        return True

    def _build_model_and_main_param_groups_batch(
        self,
        gbuf_ranges: List,
        param_gbuf_map: Dict,
        opt_group_ranges: List,
        config,
    ):
        """
        Build parameter groups with batch processing for 2D views.
        """
        # Initialize parameter groups
        model_float16_groups = []
        model_fp32_groups = []
        shard_float16_groups = []
        shard_fp32_groups = []
        shard_fp32_from_float16_groups = []

        # Process each optimizer group
        for group_range in opt_group_ranges:
            # Initialize group lists
            model_float16_params_this_group = []
            model_fp32_params_this_group = []
            shard_float16_params_this_group = []
            shard_fp32_params_this_group = []
            shard_fp32_from_float16_params_this_group = []

            # Add to main groups
            model_float16_groups.append(model_float16_params_this_group)
            model_fp32_groups.append(model_fp32_params_this_group)
            shard_float16_groups.append(shard_float16_params_this_group)
            shard_fp32_groups.append(shard_fp32_params_this_group)
            shard_fp32_from_float16_groups.append(shard_fp32_from_float16_params_this_group)

            # Batch process parameters in this group
            param_batch = []
            for model_param in group_range["params"]:
                param_batch.append(model_param)

                # Process batch when full or at end
                if len(param_batch) >= self._batch_size or model_param is group_range["params"][-1]:
                    try:
                        self._process_param_batch(
                            param_batch,
                            gbuf_ranges,
                            param_gbuf_map,
                            config,
                            model_float16_params_this_group,
                            model_fp32_params_this_group,
                            shard_float16_params_this_group,
                            shard_fp32_params_this_group,
                            shard_fp32_from_float16_params_this_group
                        )
                    except Exception as e:
                        global_rank = self._global_rank
                        logger.error(f"[Dion] Global rank {global_rank}: Failed in _process_param_batch for batch of {len(param_batch)} params: {e}")
                        for i, p in enumerate(param_batch):
                            logger.error(f"  Param {i}: shape={p.shape}, ndim={p.ndim}, requires_grad={p.requires_grad}")
                        import traceback
                        logger.error(traceback.format_exc())
                        raise
                    param_batch = []

            # Update optimizer's params
            if not config.use_precision_aware_optimizer_no_fp8_or_ds_fp8:
                group_range["orig_group"]["params"] = [
                    *shard_fp32_params_this_group,
                    *shard_fp32_from_float16_params_this_group,
                ]
            else:
                group_range["orig_group"]["params"] = [
                    *shard_fp32_params_this_group,
                    *shard_float16_params_this_group,
                ]

        return (
            model_float16_groups,
            model_fp32_groups,
            shard_float16_groups,
            shard_fp32_groups,
            shard_fp32_from_float16_groups,
        )

    def _process_param_batch(self, param_batch, gbuf_ranges, param_gbuf_map, config,
                            model_float16_params, model_fp32_params,
                            shard_float16_params, shard_fp32_params,
                            shard_fp32_from_float16_params):
        """Process a batch of parameters efficiently."""
        for model_param in param_batch:
            assert model_param.requires_grad

            gbuf_index, dtype, bucket_index = param_gbuf_map[model_param]
            gbuf_range = gbuf_ranges[gbuf_index][dtype][bucket_index]
            param_range_info = gbuf_range["param_map"][model_param]
            param_range = param_range_info["param"]
            dion_info = param_range_info.get("dion_info", {})

            # Handle different parameter types
            if model_param.type() in ['torch.cuda.HalfTensor', 'torch.cuda.BFloat16Tensor']:
                self._process_float16_param(
                    model_param, param_range, param_range_info, dion_info,
                    config, gbuf_index, bucket_index,
                    model_float16_params, shard_float16_params,
                    shard_fp32_from_float16_params
                )
            elif model_param.type() == 'torch.cuda.FloatTensor':
                self._process_float32_param(
                    model_param, param_range, param_range_info, dion_info,
                    config, gbuf_index, bucket_index,
                    model_fp32_params, shard_fp32_params
                )
            else:
                raise TypeError(f'Unsupported parameter type: {model_param.type()}')

    def _create_fs_shard(self, model_param, dion_info):
        """Create FS shard from model parameter based on dion_info.

        Args:
            model_param: The model parameter tensor (2D, TP-partitioned)
            dion_info: Dict containing 'start_idx', 'end_idx', 'fs_split_dim'

        Returns:
            shard: The FS shard tensor (cloned, not a view)
        """
        start_idx = dion_info['start_idx']
        end_idx = dion_info['end_idx']
        fs_split_dim = dion_info['fs_split_dim']

        # Extract FS shard (use .clone() to avoid copy-to-self bug)
        if fs_split_dim == 0:
            shard = model_param[start_idx:end_idx, :].clone()
        else:
            shard = model_param[:, start_idx:end_idx].clone()

        shard._model_param = model_param
        return shard

    def _prepare_fs_shard(self, model_param, shard):
        """Attach FS shard to model_param for optimizer state.

        param.data stays as FS-full, TP-sharded for forward pass.
        _fs_shard is the local shard used for optimizer state operations.

        Args:
            model_param: The model parameter tensor (FS-full, TP-sharded; Megatron-Core standard)
            shard: The FS shard tensor
        """
        model_param._fs_shard = shard

    # Unified Shard Registration/Query Helpers

    def _register_dion_shard(
        self,
        model_param: torch.nn.Parameter,
        data_shard: torch.Tensor,
        opt_shard: torch.Tensor,
        dion_info: dict,
        gbuf_index: int,
        bucket_index: int,
        param_range_info: dict,
    ) -> None:
        """Register all shard info for a Dion parameter in one call.

        Creates DionShardInfo and stores in _dion_shard_info[model_param].

        Args:
            model_param: The model parameter tensor
            data_shard: FP16 shard tensor (all-gather source)
            opt_shard: FP32 shard tensor (optimizer state)
            dion_info: Dict containing Dion metadata (shape, global_shape, etc.)
            gbuf_index: Gradient buffer index
            bucket_index: Bucket index
            param_range_info: Parameter range info dict
        """
        # Create unified shard info
        shard_info = DionShardInfo(
            data_shard=data_shard,
            opt_shard=opt_shard,
            local_shape=dion_info['shape'],
            global_shape=dion_info['global_shape'],
            start_idx=dion_info['start_idx'],
            end_idx=dion_info['end_idx'],
            fs_split_dim=dion_info['fs_split_dim'],
            gbuf_index=gbuf_index,
            bucket_index=bucket_index,
            param_range_info=param_range_info,
        )
        self._dion_shard_info[model_param] = shard_info

    def _get_data_shard(self, model_param: torch.nn.Parameter) -> Optional[torch.Tensor]:
        """Get data shard (FP16) for a model parameter.

        Args:
            model_param: The model parameter tensor

        Returns:
            Data shard tensor or None if not found
        """
        info = self._dion_shard_info.get(model_param)
        return info.data_shard if info else None

    def _get_opt_shard(self, model_param: torch.nn.Parameter) -> Optional[torch.Tensor]:
        """Get optimizer shard (FP32) for a model parameter.

        Args:
            model_param: The model parameter tensor

        Returns:
            Optimizer shard tensor or None if not found
        """
        info = self._dion_shard_info.get(model_param)
        return info.opt_shard if info else None

    def _update_data_shard(self, model_param: torch.nn.Parameter, new_data_shard: torch.Tensor) -> None:
        """Update data shard for a model parameter.

        This is used when the data_shard tensor is replaced (e.g., during all-gather operations).

        Args:
            model_param: The model parameter tensor
            new_data_shard: The new data shard tensor
        """
        info = self._dion_shard_info.get(model_param)
        if info is not None:
            info.data_shard = new_data_shard

    def _update_opt_shard(self, model_param: torch.nn.Parameter, new_opt_shard: torch.Tensor) -> None:
        """Update optimizer shard for a model parameter.

        This is used when the opt_shard tensor is replaced (e.g., during checkpoint restoration).

        Args:
            model_param: The model parameter tensor
            new_opt_shard: The new optimizer shard tensor
        """
        info = self._dion_shard_info.get(model_param)
        if info is not None:
            info.opt_shard = new_opt_shard

    def _get_dion_info_for_param(self, model_param: torch.nn.Parameter) -> dict:
        """Get dion_info-like dict from DionShardInfo for runtime use.

        Args:
            model_param: The model parameter tensor

        Returns:
            Dict with dion info fields, or {"is_dion": False} if not a Dion param
        """
        shard_info = self._dion_shard_info.get(model_param)
        if shard_info is None:
            return {"is_dion": False}
        return {
            "is_dion": True,
            "fs_split_dim": shard_info.fs_split_dim,
            "start_idx": shard_info.start_idx,
            "end_idx": shard_info.end_idx,
            "shape": shard_info.local_shape,
            "global_shape": shard_info.global_shape,
            "buffer_idx": shard_info.gbuf_index,
            "bucket_idx": shard_info.bucket_index,
        }

    def _verify_dion_flag(self, model_param, context=""):
        """Verify is_dion_param flag is set correctly.

        Args:
            model_param: The model parameter tensor
            context: Context string for logging (e.g., "FP16", "FP32")

        Raises:
            RuntimeError: If is_dion_param flag is missing or inconsistent
        """
        if not hasattr(model_param, 'is_dion_param'):
            raise RuntimeError(
                f"[Dion] {context} param missing is_dion_param flag! "
                f"Must be set in _build_model_gbuf_range. "
                f"param shape={model_param.shape}"
            )
        elif not model_param.is_dion_param:
            raise RuntimeError(
                f"[Dion] {context} param has is_dion_param=False but entered "
                f"Dion processing path. param shape={model_param.shape}"
            )

    def _process_float16_param(self, model_param, param_range, param_range_info, dion_info,
                              config, gbuf_index, bucket_index,
                              model_float16_params, shard_float16_params,
                              shard_fp32_from_float16_params):
        """Process float16/bfloat16 parameters."""
        if dion_info.get('is_dion', False):
            try:
                # Create FS shard using helper
                shard_model_param = self._create_fs_shard(model_param, dion_info)

                # Prepare FS shard (attach to model_param)
                self._prepare_fs_shard(model_param, shard_model_param)

                # Verify is_dion_param flag
                self._verify_dion_flag(model_param, "FP16")
            except Exception as e:
                global_rank = self._global_rank
                logger.error(f"[Dion] Global rank {global_rank}: Failed for Dion param")
                logger.error(f"  model_param.shape: {model_param.shape}")
                logger.error(f"  dion_info: {dion_info}")
                import traceback
                logger.error(traceback.format_exc())
                raise

            # Copy metadata
            tensor_parallel.copy_tensor_model_parallel_attributes(
                shard_model_param, model_param
            )
            if hasattr(model_param, 'shared'):
                shard_model_param.shared = model_param.shared

            # Create fp32 main param
            if not config.use_precision_aware_optimizer_no_fp8_or_ds_fp8:
                shard_main_param = shard_model_param.clone().float()
                shard_main_param._model_param = model_param
            else:
                shard_main_param = None

            # Register shard info using unified helper (Phase 2)
            opt_shard = shard_main_param if shard_main_param is not None else shard_model_param
            self._register_dion_shard(
                model_param=model_param,
                data_shard=shard_model_param,
                opt_shard=opt_shard,
                dion_info=dion_info,
                gbuf_index=gbuf_index,
                bucket_index=bucket_index,
                param_range_info=param_range_info,
            )
        else:
            # Standard 1D handling (non-Dion params)
            # Non-Dion params are ALWAYS DP-sharded via reduce-scatter.

            if is_float8tensor(model_param) and config.fp8_recipe != "delayed":
                shard_model_param = None
            else:
                # Always use DP shard for non-Dion params
                # param_range contains the DP shard range in the resized bucket layout
                shard_model_param = model_param.detach().view(-1)[
                    param_range.start : param_range.end
                ]

                shard_model_param._model_param = model_param
                tensor_parallel.copy_tensor_model_parallel_attributes(
                    shard_model_param, model_param
                )
                if hasattr(model_param, 'shared'):
                    shard_model_param.shared = model_param.shared

            # Generate main param
            if not config.use_precision_aware_optimizer_no_fp8_or_ds_fp8:
                if is_float8tensor(model_param):
                    # Handle FP8 tensors - always use DP shard
                    if hasattr(model_param, 'get_high_precision_init_val'):
                        shard_main_param = (
                            model_param.get_high_precision_init_val()
                            .view(-1)[param_range.start : param_range.end]
                            .clone()
                            .to(model_param.device)
                            .float()
                        )
                        model_param.clear_high_precision_init_val()
                    else:
                        shard_main_param = model_param.float().view(-1)[
                            param_range.start : param_range.end
                        ]
                    shard_main_param._model_param = model_param
                else:
                    shard_main_param = shard_model_param.clone().float()
                    shard_main_param._model_param = model_param
            else:
                shard_main_param = None

        # Copy metadata to main param
        if shard_main_param is not None:
            tensor_parallel.copy_tensor_model_parallel_attributes(
                shard_main_param, model_param
            )
            if hasattr(model_param, 'shared'):
                shard_main_param.shared = model_param.shared

        # Store handle
        model_param.main_param = shard_main_param
        model_param.main_param_sharded = True

        # Note: Non-Dion params use standard DO path, not registered in _dion_shard_info

        # Add to groups
        model_float16_params.append(model_param)
        shard_float16_params.append(shard_model_param)
        shard_fp32_from_float16_params.append(shard_main_param)

    def _process_float32_param(self, model_param, param_range, param_range_info, dion_info,
                              config, gbuf_index, bucket_index,
                              model_fp32_params, shard_fp32_params):
        """Process float32 parameters."""
        if dion_info.get('is_dion', False):
            # Create FS shard using helper
            shard_model_param = self._create_fs_shard(model_param, dion_info)

            # Prepare FS shard (attach to model_param)
            self._prepare_fs_shard(model_param, shard_model_param)

            # Copy metadata
            tensor_parallel.copy_tensor_model_parallel_attributes(
                shard_model_param, model_param
            )
            if hasattr(model_param, 'shared'):
                shard_model_param.shared = model_param.shared

            # Add forward backlink for bucket-wise communication
            model_param.main_param = shard_model_param
            model_param.main_param_sharded = True

            # Verify is_dion_param flag
            self._verify_dion_flag(model_param, "FP32")

            # Register shard info using unified helper (Phase 2)
            # FP32 params: data_shard == opt_shard == shard_model_param
            self._register_dion_shard(
                model_param=model_param,
                data_shard=shard_model_param,
                opt_shard=shard_model_param,
                dion_info=dion_info,
                gbuf_index=gbuf_index,
                bucket_index=bucket_index,
                param_range_info=param_range_info,
            )
        else:
            # Standard 1D handling
            shard_model_param = model_param.view(-1)[param_range.start : param_range.end]
            shard_model_param._model_param = model_param

            tensor_parallel.copy_tensor_model_parallel_attributes(
                shard_model_param, model_param
            )
            if hasattr(model_param, 'shared'):
                shard_model_param.shared = model_param.shared

            # Note: Non-Dion params use standard DO path, not registered in _dion_shard_info

        model_fp32_params.append(model_param)
        shard_fp32_params.append(shard_model_param)

    def _enable_dion_distributed_mode_improved(self):
        """Enable distributed mode with improved batch processing."""
        global_rank = self._global_rank

        if not isinstance(self.optimizer, (MegatronDion,)):
            return

        from ..parallel_state import (
            get_tensor_model_parallel_group,
            get_tensor_model_parallel_world_size,
            get_data_parallel_world_size,
        )

        if get_data_parallel_world_size() == 1 and get_tensor_model_parallel_world_size() == 1:
            return

        # Collect buffer sizes
        gbuf_sizes = [
            [(bucket.grad_data.numel(), bucket.offset) for bucket in buffer.buckets]
            for buffer in self.buffers
        ]

        # Store groups for dist_meta creation
        self.my_rp_group = self.optimizer.rp_group
        self.my_fs_group = self.optimizer.fs_group

        # Create replica groups with improved batching (will be no-op if groups already set)
        self._create_replica_groups_batch()

        # Create true RP and FS groups for 2D parallelism (will be no-op if groups already set)
        self._create_rp_fs_groups()

        # Create dist_metas with batch processing
        try:
            dist_metas_sharded = self._create_dist_metas_batch()
        except Exception as e:
            logger.error(f"[Dion] Global rank {global_rank}: Failed in _create_dist_metas_batch: {e}")
            logger.error(traceback.format_exc())
            raise

        # Enable distributed mode with 2D parallelism support
        # Use original DP group for dist_group (not FS group used for sharding)
        dist_group_to_use = self._original_dp_group if hasattr(self, '_original_dp_group') and self._original_dp_group else self.data_parallel_group

        enable_args = {
            'global_buffer_sizes': gbuf_sizes,
            'dist_group': dist_group_to_use,  # Original full DP = RP × FS
            'tp_group': get_tensor_model_parallel_group(),
            'dist_metas': dist_metas_sharded,
            'rp_group': self.my_rp_group,  # Same handle as optimizer.rp_group
            'fs_group': self.my_fs_group,  # Same handle as optimizer.fs_group
        }


        self.optimizer.enable_distributed_mode(**enable_args)

        # Re-sync groups after enable_distributed_mode()
        # enable_distributed_mode() may have updated optimizer's rp_group/fs_group
        self.my_rp_group = self.optimizer.rp_group
        self.my_fs_group = self.optimizer.fs_group

        # Register mixed bucket adapters (buffer_indices populated in enable_distributed_mode)
        if self._enable_overlap_param_gather or self._enable_overlap_grad_reduce:
            overlap_features = []
            if self._enable_overlap_grad_reduce:
                overlap_features.append("gradient RS")
            if self._enable_overlap_param_gather:
                overlap_features.append("parameter AG")
            overlap_msg = " + ".join(overlap_features)

        # Verify all ranks have rp_group before RP check
        # Prevent partial participation in RP all_gather
        have_rp = torch.tensor(
            [1 if self.my_rp_group is not None else 0],
            device=torch.cuda.current_device(), dtype=torch.int64
        )
        dist.all_reduce(have_rp, op=dist.ReduceOp.MIN, group=self.data_parallel_group)
        all_have_rp = int(have_rp.item()) == 1

        if not all_have_rp:
            # Not all ranks have rp_group - MUST skip collectively
            # Cannot proceed with RP collective operations if only some ranks have the group
            logger.warning(f"[Dion] Global rank {global_rank}: "
                          f"Not all DP ranks have rp_group (MIN={have_rp.item()}), skipping RP consistency check collectively")
            # Barrier to ensure all ranks skip together
            dist.barrier(group=self.data_parallel_group)
            # Skip to next section without RP collective operations
        else:
            # All ranks should have rp_group when all_have_rp == True
            # Double-check to catch bugs
            if self.my_rp_group is None:
                raise RuntimeError(
                    f"Global rank {global_rank}: all_have_rp=True but my_rp_group is None! "
                    f"This indicates a bug in group creation or collective voting logic."
                )

            # Verify Dion eligibility consistency within RP group
            # All ranks in same RP group must have identical Dion param counts
            # Now ALL ranks will execute this block (no partial participation)
            my_dion_count = sum(1 for meta in dist_metas_sharded.values() if meta.is_dion_param)
            my_cnt_tensor = torch.tensor([my_dion_count], device=torch.cuda.current_device(), dtype=torch.int64)

            rp_world_size = dist.get_world_size(self.my_rp_group)
            gathered = [torch.zeros_like(my_cnt_tensor) for _ in range(rp_world_size)]
            dist.all_gather(gathered, my_cnt_tensor, group=self.my_rp_group)

            gathered_counts = [int(t.item()) for t in gathered]
            if not all(count == my_dion_count for count in gathered_counts):
                # Collect Dion param offsets for error diagnosis
                my_dion_offsets = sorted([
                    meta.param_uid for meta in dist_metas_sharded.values() if meta.is_dion_param
                ])
                gathered_offsets = [None] * rp_world_size
                dist.all_gather_object(gathered_offsets, my_dion_offsets, group=self.my_rp_group)

                # Find symmetric difference
                for rp_rank, offsets in enumerate(gathered_offsets):
                    if offsets != my_dion_offsets:
                        my_set = set(my_dion_offsets)
                        other_set = set(offsets)
                        only_mine = my_set - other_set
                        only_other = other_set - my_set
                        logger.error(f"[Dion] RP rank {rp_rank} differs from me: "
                                   f"Only in mine: {sorted(only_mine)[:5]}, "
                                   f"Only in theirs: {sorted(only_other)[:5]}")

                raise RuntimeError(
                    f"CRITICAL: Dion eligibility mismatch within RP group! "
                    f"My Dion count: {my_dion_count}, RP group counts: {gathered_counts}. "
                    f"This will cause collective operation hangs. "
                    f"DistributedOptimizer did uniform sharding across DP (RP×FS), "
                    f"so RP group members have different param chunks. "
                    f"Consider disabling DistributedOptimizer sharding or implementing custom FS-aware sharding."
                )

    def _create_replica_groups_batch(self):
        """
        DISABLED: No-op - RP/FS groups already created in __init__.py

        This function previously created additional subgroups based on owner_tuples,
        but these groups are not actually used (dist_meta.replica_group uses self.my_rp_group instead).
        Creating unnecessary groups causes new_group() call count mismatch across ranks,
        leading to NCCL hang.

        Solution: Skip all new_group() calls. Use collective voting to ensure all ranks
        take the same path (critical for PyTorch new_group() synchronization).
        """
        global_rank = self._global_rank
        dp_group = self.data_parallel_group
        dp_world_size = dist.get_world_size(dp_group)

        # Check if all ranks have rp_group
        have_rp = torch.tensor(
            [1 if hasattr(self, 'my_rp_group') and self.my_rp_group is not None else 0],
            device=torch.cuda.current_device(), dtype=torch.int64
        )
        dist.all_reduce(have_rp, op=dist.ReduceOp.MIN, group=dp_group)
        all_have = int(have_rp.item()) == 1

        if all_have:
            # All ranks have groups - skip collectively with barrier
            self.owner_tuple_to_subgroup = {}
            dist.barrier(group=dp_group)
            return
        elif hasattr(self, 'my_rp_group') and self.my_rp_group is not None:
            # Only some ranks have groups - FATAL ERROR
            raise RuntimeError(
                f"Global rank {global_rank}: RP/FS groups exist only on subset of ranks! "
                f"Ensure rp_group/fs_group are provided uniformly to prevent new_group() mismatch."
            )

        # If we reach here, no ranks have groups - would create them
        # But this path should never execute since __init__.py always creates groups
        raise RuntimeError(
            f"Global rank {global_rank}: No RP/FS groups found! "
            f"Groups should be created in __init__.py before DistributedOptimizerForDion."
        )

    def _create_rp_fs_groups(self):
        """
        DISABLED: No-op - RP/FS groups already created in __init__.py

        This function previously created RP/FS groups, but they are already created
        in __init__.py with deterministic world-level synchronization.
        Attempting to create groups again here causes new_group() call count mismatch
        across ranks (especially with TP>1), leading to NCCL hang.

        Solution: Skip all new_group() calls. Use collective voting to ensure all ranks
        take the same path (critical for PyTorch new_group() synchronization).
        """
        global_rank = self._global_rank
        dp_group = self.data_parallel_group
        dp_world_size = dist.get_world_size(dp_group)


        # Check if all ranks have both rp_group and fs_group
        have_both = torch.tensor(
            [1 if hasattr(self, 'my_rp_group') and self.my_rp_group is not None and
                  hasattr(self, 'my_fs_group') and self.my_fs_group is not None else 0],
            device=torch.cuda.current_device(), dtype=torch.int64
        )
        dist.all_reduce(have_both, op=dist.ReduceOp.MIN, group=dp_group)
        all_have_groups = int(have_both.item()) == 1

        if all_have_groups:
            # All ranks have groups - skip collectively with barrier

            # Initialize compatibility variables
            self.shard_index_to_rp_group = {}
            self.replica_index_to_fs_group = {}
            self.my_replica_idx = -1
            self.my_shard_idx = -1
            self.num_replicas = dist.get_world_size(self.my_rp_group)
            self.num_shards = dist.get_world_size(self.my_fs_group)

            # Barrier to ensure all ranks finish together
            dist.barrier(group=dp_group)
            return

        elif hasattr(self, 'my_rp_group') and self.my_rp_group is not None:
            # Only some ranks have groups - FATAL ERROR
            raise RuntimeError(
                f"Global rank {global_rank}: RP/FS groups exist only on subset of ranks! "
                f"Ensure rp_group/fs_group are provided uniformly to prevent new_group() mismatch."
            )

        # If we reach here, no ranks have groups - should never happen
        raise RuntimeError(
            f"Global rank {global_rank}: No RP/FS groups found! "
            f"Groups should be created in __init__.py before DistributedOptimizerForDion."
        )

    def _create_dist_metas_batch(self):
        """Create dist_metas with batch processing."""
        from ..parallel_state import get_tensor_model_parallel_world_size

        dist_metas_sharded = {}
        dp_rank = self.data_parallel_group.rank()

        # Batch process Dion shard mappings (Phase 3: using unified _dion_shard_info)
        for model_param, shard_info in self._dion_shard_info.items():
            shard_param = shard_info.opt_shard  # Key for dist_metas_sharded
            param_range_info = shard_info.param_range_info

            global_shape = shard_info.global_shape

            # Get TP split dimension
            if is_tp_enabled(model_param):
                tp_split_dim = get_tp_split_dim(model_param)
                # global_shape already restored in _create_fs_aware_shard, no need to restore again
            else:
                tp_split_dim = -1

            # Get global range
            global_range = (
                param_range_info["gbuf_world"].start,
                param_range_info["gbuf_world"].end
            )

            # Get param name and is_expert for shard_group selection
            param_name = ""
            if hasattr(self, 'param_to_name'):
                param_name = self.param_to_name.get(model_param, "")
            is_expert = not getattr(model_param, 'allreduce', True)

            # Use different shard_group for expert vs dense params
            # Expert params use expert_data_parallel_group (EP-internal FS)
            # Dense params use global fs_group
            replica_group = self.my_rp_group
            if is_expert:
                from megatron.core import parallel_state
                try:
                    shard_group = parallel_state.get_expert_data_parallel_group(
                        partial_expert_data_parallel=True
                    )
                except Exception:
                    shard_group = self.my_fs_group
            else:
                shard_group = self.my_fs_group

            # Get group info from actual groups
            if replica_group is not None:
                replica_group_world_size = dist.get_world_size(replica_group)
                replica_group_rank = dist.get_rank(replica_group)
            else:
                replica_group_world_size = 1
                replica_group_rank = -1

            if shard_group is not None:
                shard_group_world_size = dist.get_world_size(shard_group)
                shard_group_rank = dist.get_rank(shard_group)
            else:
                shard_group_world_size = 1
                shard_group_rank = -1

            # Create dist_meta with 2D parallelism fields
            # Get fs_split_dim from shard_info (set in _process_float16_param)
            fs_split_dim = shard_info.fs_split_dim
            dist_meta = MegatronDionDistMeta(
                buffer_idx=shard_info.gbuf_index,
                bucket_idx=shard_info.bucket_index,
                shape=shard_param.shape,
                global_shape=global_shape,
                global_range=global_range,
                tp_split_dim=tp_split_dim,
                fs_split_dim=fs_split_dim,  # Pass FS split dimension for correct axis detection
                rank_fraction=self.optimizer.defaults.get('rank_fraction', 0.25),
                is_dion_param=getattr(model_param, 'is_dion_param', True),
                is_expert=is_expert,
                param_name=param_name,
                # Use tuple (buffer_idx, bucket_idx, start) for unique param_uid
                # gbuf_world.start alone can collide across different buffers
                param_uid=(shard_info.gbuf_index, shard_info.bucket_index, param_range_info["gbuf_world"].start),
            )
            # Set 2D parallelism fields - use actual group handles
            dist_meta.replica_group = replica_group
            dist_meta.replica_group_world_size = replica_group_world_size
            dist_meta.replica_group_rank = replica_group_rank
            dist_meta.shard_group = shard_group
            dist_meta.shard_group_world_size = shard_group_world_size
            dist_meta.shard_group_rank = shard_group_rank

            # Set stable replica_group_id for batching
            # Use shard_group_rank as stable ID (all params on same rank have same FS rank)
            dist_meta.replica_group_id = shard_group_rank if shard_group else 0

            dist_metas_sharded[shard_param] = dist_meta
            # Store param_uid directly on param for lookup after offload/reload
            # This allows _get_or_initialize_state to find UID even when dist_metas lookup fails
            shard_param._dion_param_uid = dist_meta.param_uid

        # Add non-Dion parameters
        self._add_non_dion_dist_metas(dist_metas_sharded)

        return dist_metas_sharded

    def _add_non_dion_dist_metas(self, dist_metas_sharded):
        """Add dist_metas for non-Dion parameters."""
        non_dion_count = 0
        skipped_count = 0
        total_param_groups = sum(len(g['params']) for g in self.optimizer.param_groups)


        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p not in dist_metas_sharded:
                    # Find model param
                    model_param = getattr(p, '_model_param', None)

                    if model_param is None:
                        # Slow scan fallback
                        model_param = self._find_model_param_for_shard(p)

                    if model_param is None:
                        logger.warning(f"[Dion] Could not find model_param for shard shape={p.shape}")
                        skipped_count += 1
                        continue

                    # Create basic dist_meta
                    gbuf_index, dtype, bucket_index = self.model_param_gbuf_map[model_param]
                    gbuf_range = self.gbuf_ranges[gbuf_index][dtype][bucket_index]
                    pr = gbuf_range["param_map"][model_param]["gbuf_world"]

                    param_uid = (gbuf_index, bucket_index, pr.start)
                    dist_metas_sharded[p] = MegatronDionDistMeta(
                        buffer_idx=gbuf_index,
                        bucket_idx=bucket_index,
                        shape=p.shape,
                        global_shape=None,
                        global_range=(pr.start, pr.end),
                        tp_split_dim=-1,
                        rank_fraction=self.optimizer.defaults.get('rank_fraction', 0.25),
                        is_dion_param=getattr(model_param, 'is_dion_param', False),
                        # Use tuple (buffer_idx, bucket_idx, start) for unique param_uid
                        param_uid=param_uid,
                    )
                    # Store param_uid directly on param for lookup after offload/reload
                    p._dion_param_uid = param_uid
                    non_dion_count += 1


    def _find_model_param_for_shard(self, shard_param):
        """Find model parameter for a given shard."""
        for gbuf_range_maps in self.gbuf_ranges:
            for gbuf_range_map_list in gbuf_range_maps.values():
                for gbuf_range_map in gbuf_range_map_list:
                    for mp, param_range_info in gbuf_range_map["param_map"].items():
                        if hasattr(mp, 'main_param') and mp.main_param is shard_param:
                            return mp
        return None

    def _copy_model_grads_to_main_grads(self):
        """Copy gradients from model params to main params with main_grad priority.

        When RP=1 with DO overlap:
        - DO already performed gradient RS into grad buffers (during backward)
        - FS group = DP group, so DO's DP RS = FS RS
        - Here we just map buffer views to optimizer shard params
        - No manual FS reduce-scatter needed

        Note: Dion parameters have gradients in model_float16_groups[].main_grad
        (buffer view, FS×TP sharded), but optimizer receives
        shard_fp32_from_float16_groups[] which are different FP32 tensor objects.
        We MUST copy main_grad reference to shard params so optimizer can access it!

        Root cause: model_param (BF16) and shard_main_param (FP32) are different objects.
        - model_param.main_grad = buffer.grad_data[range].view() ✓ (set in __init__)
        - shard_main_param.main_grad = ??? ✗ (never set!)
        - optimizer receives shard_main_param but reads .main_grad → None!

        Note: We override parent completely to avoid double-processing Dion params.
        Parent's copy_group_grads assumes all params have valid main_grad, but we need
        to handle Dion params separately. If we let parent process Dion params, it causes
        tensor memory conflicts during TensorImpl destruction.
        """
        # Match parent's logging pattern (distrib_optimizer.py:2392)

        if self.is_stub_optimizer:
            return

        if self.ddp_config.use_megatron_fsdp:
            return


        # Copy gradients for all params (Dion and non-Dion)
        copied_dion = 0
        copied_non_dion = 0
        skipped_no_grad = 0

        def copy_group_grads(model_groups, shard_main_groups):
            nonlocal copied_dion, copied_non_dion, skipped_no_grad

            for group_idx, (model_group, shard_main_group) in enumerate(zip(model_groups, shard_main_groups)):
                for param_idx, (model_param, shard_main_param) in enumerate(zip(model_group, shard_main_group)):

                    # Dion params: special handling
                    if getattr(model_param, 'is_dion_param', False):
                        # Dion params: bind RS output to shard_main_param only
                        # Do NOT rebind model_param.main_grad - keep it FULL for next backward's TE GEMM
                        param_range_map = self._get_model_param_range_map(model_param)
                        param_range = param_range_map["param"]
                        dion_info = self._get_dion_info_for_param(model_param)
                        buffer_idx = dion_info.get("buffer_idx")
                        bucket_idx = dion_info.get("bucket_idx")
                        local_shape = dion_info.get("shape", model_param.shape)

                        # Fallback: if dion_info missing buffer/bucket, derive from model_param_gbuf_map
                        if buffer_idx is None or bucket_idx is None:
                            try:
                                gbuf_index, _, bidx = self.model_param_gbuf_map[model_param]
                                buffer_idx = gbuf_index if buffer_idx is None else buffer_idx
                                bucket_idx = bidx if bucket_idx is None else bucket_idx
                                logger.warning(f"[Dion] Filled buffer_idx={buffer_idx}, bucket_idx={bucket_idx} for param id={id(model_param)}")
                            except Exception as e:
                                logger.error(f"[Dion] Could not infer buffer/bucket for param id={id(model_param)}: {e}")

                        bucket_slice = None
                        try:
                            bucket = self.buffers[buffer_idx].buckets[bucket_idx]
                            # Use dion_param_shard_range for RS output position
                            expected_len = local_shape[0] * local_shape[1] if len(local_shape) == 2 else local_shape[0]

                            if hasattr(bucket, 'dion_param_shard_range') and model_param in bucket.dion_param_shard_range:
                                # Use RS output layout from dion_param_shard_range
                                rs_start, rs_end = bucket.dion_param_shard_range[model_param]
                                actual_len = rs_end - rs_start
                                if actual_len != expected_len:
                                    pname = self.param_to_name.get(model_param, f"id_{id(model_param)}") if hasattr(self, "param_to_name") else f"id_{id(model_param)}"
                                    logger.error(
                                        f"[Dion] GRAD RANGE MISMATCH param={pname} rs_range=({rs_start},{rs_end}) "
                                        f"actual_len={actual_len} expected={expected_len}"
                                    )
                                bucket_slice = bucket.grad_data[rs_start:rs_end].view(local_shape)
                            else:
                                # Bug fallback: param should be in dion_param_shard_range
                                pname = self.param_to_name.get(model_param, f"id_{id(model_param)}") if hasattr(self, "param_to_name") else f"id_{id(model_param)}"
                                has_attr = hasattr(bucket, 'dion_param_shard_range')
                                range_keys = len(bucket.dion_param_shard_range) if has_attr else 0
                                logger.warning(
                                    f"[Dion] Group {group_idx}, Param {param_idx} ({pname}): "
                                    f"model_param id={id(model_param)} NOT in dion_param_shard_range! "
                                    f"has_attr={has_attr}, range_keys={range_keys}, bucket_id={bucket.bucket_id}"
                                )
                                slice_end = param_range.start + expected_len
                                bucket_slice = bucket.grad_data[param_range.start:slice_end].view(local_shape)
                        except Exception as e:
                            logger.error(f"[Dion] Failed to fetch bucket slice for param id={id(model_param)}: {e}")

                        if bucket_slice is not None:
                            # Bind to shard_main_param only (keep model_param.main_grad FULL for TE GEMM)
                            shard_main_param.main_grad = bucket_slice.float()

                            # Log gradient stats (use bucket_slice, not model_param.main_grad)
                            grad_max = bucket_slice.abs().max().item()
                            grad_mean = bucket_slice.abs().mean().item()
                            grad_nonzero = (bucket_slice.abs() > 1e-8).sum().item()
                            grad_numel = bucket_slice.numel()
                        else:
                            # If bucket_slice is None, bind zeros
                            skipped_no_grad += 1
                            zero_view = torch.zeros(local_shape, device=model_param.device, dtype=torch.float32)
                            shard_main_param.main_grad = zero_view
                            pname = self.param_to_name.get(model_param, f"id_{id(model_param)}") if hasattr(self, "param_to_name") else f"id_{id(model_param)}"
                            logger.warning(f"  [Dion] Group {group_idx}, Param {param_idx}: no grad slice; binding zeros for {pname}")
                            grad_max = 0.0
                            grad_mean = 0.0
                            grad_nonzero = 0
                            grad_numel = local_shape[0] * local_shape[1] if len(local_shape) == 2 else local_shape[0]

                        # Copy is_dion_param flag to shard_main_param!
                        # Without this, optimizer step() sees is_dion_param=False → AdamW fallback
                        shard_main_param.is_dion_param = True
                        shard_main_param.grad = None  # Clear .grad to force main_grad usage

                        copied_dion += 1

                        # Log warning for zero grads (indicates potential RS issue)
                        if grad_max < 1e-8:
                            pname = self.param_to_name.get(model_param, f"id_{id(model_param)}") if hasattr(self, "param_to_name") else f"id_{id(model_param)}"
                            logger.warning(f"  [ZERO GRAD] Param {pname}: shape={shard_main_param.main_grad.shape}")
                        continue  # Skip parent's logic for Dion params

                    # Non-Dion params in mixed bucket read from RS output

                    shard_model_grad = None

                    # Try to get bucket info for this param
                    try:
                        gbuf_index, _, bucket_idx = self.model_param_gbuf_map[model_param]
                        bucket = self.buffers[gbuf_index].buckets[bucket_idx]

                        # Check if this param is in a mixed bucket with non_dion_param_ranges
                        if (hasattr(bucket, 'non_dion_param_ranges') and
                            model_param in bucket.non_dion_param_ranges):
                            # Mixed bucket: read from RS output using non_dion_param_ranges
                            rs_start, rs_end = bucket.non_dion_param_ranges[model_param]
                            shard_slice = bucket.grad_data[rs_start:rs_end]

                            if shard_slice.numel() == shard_main_param.nelement():
                                shard_model_grad = shard_slice.view(shard_main_param.shape)
                            else:
                                # Size mismatch - pad if needed
                                padded = torch.zeros_like(shard_main_param.view(-1))
                                padded[:min(padded.numel(), shard_slice.numel())].copy_(shard_slice.view(-1))
                                shard_model_grad = padded.view(shard_main_param.shape)
                                logger.warning(f"[NonDion] Size mismatch: rs_slice={shard_slice.numel()}, "
                                             f"shard_param={shard_main_param.nelement()}, padded")
                    except Exception:
                        pass  # Fall through to fallback

                    # Fallback: use model_param.main_grad (pure non-Dion bucket or lookup failed)
                    if shard_model_grad is None:
                        param_range_map = self._get_model_param_range_map(model_param)
                        param_range = param_range_map["param"]
                        model_grad = model_param.main_grad

                        # If model_grad already matches shard size, use directly.
                        if model_grad.numel() == shard_main_param.nelement():
                            shard_model_grad = model_grad.reshape(shard_main_param.shape)
                        else:
                            # Use precise offset from param_range (DP shard view) to avoid padding.
                            flat_grad = model_grad.view(-1)
                            start = param_range.start
                            end = min(param_range.end, flat_grad.numel())
                            shard_view = flat_grad[start:end]
                            if shard_view.numel() != shard_main_param.nelement():
                                # If still short (should be rare), pad once locally without warning spam.
                                padded = torch.zeros_like(shard_main_param.view(-1))
                                padded[:min(padded.numel(), shard_view.numel())].copy_(shard_view)
                                shard_view = padded
                            shard_model_grad = shard_view.view(shard_main_param.shape)

                    if self.config.use_precision_aware_optimizer_no_fp8_or_ds_fp8:
                        shard_main_param.decoupled_grad = shard_model_grad
                    else:
                        # Set main_grad for all params (Dion expects main_grad as gradient source)
                        shard_main_param.main_grad = shard_model_grad.float()
                        shard_main_param.grad = None

                        # Explicitly mark as non-Dion param
                        # Optimizer step() checks is_dion_param flag to decide Dion vs AdamW
                        shard_main_param.is_dion_param = False

                    copied_non_dion += 1

        # Copy model groups to shard groups.
        if self.config.use_precision_aware_optimizer_no_fp8_or_ds_fp8:
            copy_group_grads(self.model_float16_groups, self.shard_float16_groups)
            copy_group_grads(self.model_fp32_groups, self.shard_fp32_groups)
        else:
            copy_group_grads(self.model_float16_groups, self.shard_fp32_from_float16_groups)
            copy_group_grads(self.model_fp32_groups, self.shard_fp32_groups)


        # Propagate is_dion_param flag to optimizer.param_groups
        if hasattr(self, 'optimizer') and hasattr(self.optimizer, 'param_groups'):
            propagated_count = 0
            total_count = 0
            identity_check_count = 0
            for group_idx, group in enumerate(self.optimizer.param_groups):
                for param_idx, optimizer_param in enumerate(group['params']):
                    total_count += 1
                    # Find corresponding model_param by checking if this is a shard of it
                    found_flag = False

                    # Check model_float16_groups
                    if group_idx < len(self.model_float16_groups):
                        model_group = self.model_float16_groups[group_idx]
                        shard_group = self.shard_fp32_from_float16_groups[group_idx]

                        if param_idx < len(shard_group) and optimizer_param is shard_group[param_idx]:
                            model_param = model_group[param_idx]
                            if hasattr(model_param, 'is_dion_param'):
                                optimizer_param.is_dion_param = model_param.is_dion_param
                                propagated_count += 1
                                found_flag = True

                    # If not found, check model_fp32_groups
                    if not found_flag:
                        fp32_group_idx = group_idx - len(self.model_float16_groups)
                        if 0 <= fp32_group_idx < len(self.model_fp32_groups):
                            model_group = self.model_fp32_groups[fp32_group_idx]
                            shard_group = self.shard_fp32_groups[fp32_group_idx]
                            if param_idx < len(shard_group) and optimizer_param is shard_group[param_idx]:
                                model_param = model_group[param_idx]
                                if hasattr(model_param, 'is_dion_param'):
                                    optimizer_param.is_dion_param = model_param.is_dion_param
                                    propagated_count += 1


    def get_main_grads_for_grad_norm(self):
        """
        Override to use correct gradient sources for grad norm computation.

        Note: Handle two different sharding schemes:
          - Dion params: FS × TP 2D sharding → use model_param.main_grad
          - Non-Dion params: standard DO flat DP sharding → use shard_main_param.grad

        For Dion params:
          - model_param.main_grad contains FS-sharded gradient
          - Each FS rank has DIFFERENT shard
          - All-reduce across world group (FS × TP)

        For Non-Dion params:
          - shard_main_param.grad contains DP-sharded gradient (copied from model_param.main_grad)
          - Each DP rank has DIFFERENT shard
          - All-reduce across world group

        Both are combined and all-reduced across the world group to get global grad_norm.
        """
        from megatron.core.transformer.module import param_is_not_shared
        from megatron.core import tensor_parallel

        grads_for_norm = []

        # Use shard_main_param.main_grad for all params (contains RS output)

        # Build mapping from model params to shard params
        model_to_shard = {}
        model_groups_flat = []
        if hasattr(self, 'model_float16_groups'):
            model_groups_flat.extend(self.model_float16_groups)
        if hasattr(self, 'model_fp32_groups'):
            model_groups_flat.extend(self.model_fp32_groups)

        shard_groups_flat = []
        if hasattr(self, 'shard_fp32_from_float16_groups'):
            shard_groups_flat.extend(self.shard_fp32_from_float16_groups)
        if hasattr(self, 'shard_fp32_groups'):
            shard_groups_flat.extend(self.shard_fp32_groups)

        for model_group, shard_group in zip(model_groups_flat, shard_groups_flat):
            for model_param, shard_param in zip(model_group, shard_group):
                model_to_shard[id(model_param)] = shard_param

        # Part 1: Dion params - use shard_main_param.main_grad (RS output shard)
        for model_group in model_groups_flat:
            for model_param in model_group:
                # Check if this is a Dion param
                is_dion_param = getattr(model_param, 'is_dion_param', False)

                if is_dion_param:
                    # Dion param: use shard_main_param.main_grad (RS output)
                    shard_param = model_to_shard.get(id(model_param), None)
                    if shard_param is None:
                        continue

                    grad = getattr(shard_param, 'main_grad', None)
                    if grad is None:
                        continue

                    # Apply standard filters
                    is_not_shared = param_is_not_shared(model_param)
                    is_not_tp_duplicate = tensor_parallel.param_is_not_tensor_parallel_duplicate(model_param)

                    if is_not_shared and is_not_tp_duplicate:
                        # Use shard_main_param.main_grad - it's the RS output shard
                        grads_for_norm.append(grad.view(-1))

        # Part 2: Non-Dion params - use shard_main_param.grad (standard DO)
        shard_param_groups = []
        if hasattr(self, 'shard_fp32_from_float16_groups'):
            shard_param_groups.extend(self.shard_fp32_from_float16_groups)
        if hasattr(self, 'shard_fp32_groups'):
            shard_param_groups.extend(self.shard_fp32_groups)

        # Build mapping from shard params to model params for is_dion_param check
        shard_to_model = {}
        model_groups_flat = []
        if hasattr(self, 'model_float16_groups'):
            model_groups_flat.extend(self.model_float16_groups)
        if hasattr(self, 'model_fp32_groups'):
            model_groups_flat.extend(self.model_fp32_groups)

        shard_groups_flat = []
        if hasattr(self, 'shard_fp32_from_float16_groups'):
            shard_groups_flat.extend(self.shard_fp32_from_float16_groups)
        if hasattr(self, 'shard_fp32_groups'):
            shard_groups_flat.extend(self.shard_fp32_groups)

        # Match by index (model groups and shard groups have same structure)
        for model_group, shard_group in zip(model_groups_flat, shard_groups_flat):
            for model_param, shard_param in zip(model_group, shard_group):
                shard_to_model[id(shard_param)] = model_param

        for shard_group in shard_param_groups:
            for shard_param in shard_group:
                # Get corresponding model param to check is_dion_param
                model_param = shard_to_model.get(id(shard_param), None)
                if model_param is None:
                    continue

                is_dion_param = getattr(model_param, 'is_dion_param', False)
                if is_dion_param:
                    # Already handled in Part 1
                    continue

                # Non-Dion param: use shard_param.grad (or main_grad if available)
                grad = getattr(shard_param, 'main_grad', None)
                if grad is None:
                    grad = shard_param.grad
                if grad is None:
                    continue

                # Apply standard filters using model_param attributes
                is_not_shared = param_is_not_shared(model_param)
                is_not_tp_duplicate = tensor_parallel.param_is_not_tensor_parallel_duplicate(model_param)

                if is_not_shared and is_not_tp_duplicate:
                    grads_for_norm.append(grad)


        return grads_for_norm

    def clip_grad_norm(self, clip_grad: float) -> float:
        """
        Override to apply gradient clipping with correct sharding schemes.

        Note: Handle two different sharding schemes:
          - Dion params: clip model_param.main_grad (FS × TP sharding)
          - Non-Dion params: clip shard_main_param.grad (standard DO flat DP sharding)

        The parent's clip_grad_norm() only clips shard_main_param.grad, which doesn't
        affect Dion's model_param.main_grad (because _copy_model_grads_to_main_grads
        creates a COPY via .float()).
        """
        from megatron.core.transformer.module import param_is_not_shared
        from megatron.core import tensor_parallel
        from megatron.core.optimizer.clip_grads import get_grad_norm_fp32

        # Compute global grad norm using all shards (Dion + non-Dion params)
        grads_for_norm = self.get_main_grads_for_grad_norm()

        grad_norm = get_grad_norm_fp32(
            grads_for_norm, grad_stats_parallel_group=self.get_grad_stats_parallel_group()
        )

        # Apply clipping
        if clip_grad > 0.0 and grad_norm > 0.0:
            clip_coeff = clip_grad / (grad_norm + 1.0e-6)
            if clip_coeff < 1.0:
                # Part 1: Clip Dion params (both BF16 main_grad and FP32 copy)
                model_param_groups = []
                shard_param_groups_for_dion = []
                if hasattr(self, 'model_float16_groups'):
                    model_param_groups.extend(self.model_float16_groups)
                if hasattr(self, 'model_fp32_groups'):
                    model_param_groups.extend(self.model_fp32_groups)
                if hasattr(self, 'shard_fp32_from_float16_groups'):
                    shard_param_groups_for_dion.extend(self.shard_fp32_from_float16_groups)
                if hasattr(self, 'shard_fp32_groups'):
                    shard_param_groups_for_dion.extend(self.shard_fp32_groups)

                for model_group, shard_group in zip(model_param_groups, shard_param_groups_for_dion):
                    for model_param, shard_param in zip(model_group, shard_group):
                        is_dion_param = getattr(model_param, 'is_dion_param', False)
                        if not is_dion_param:
                            continue

                        # Apply standard filters
                        is_not_shared = param_is_not_shared(model_param)
                        is_not_tp_duplicate = tensor_parallel.param_is_not_tensor_parallel_duplicate(model_param)

                        if is_not_shared and is_not_tp_duplicate:
                            # Only clip shard_main_param.main_grad (FP32 RS output)
                            shard_grad = getattr(shard_param, 'main_grad', None)
                            if shard_grad is not None:
                                shard_grad.mul_(clip_coeff)

                # Part 2: Clip Non-Dion params' shard_main_param.grad
                shard_to_model = {}
                model_groups_flat = []
                if hasattr(self, 'model_float16_groups'):
                    model_groups_flat.extend(self.model_float16_groups)
                if hasattr(self, 'model_fp32_groups'):
                    model_groups_flat.extend(self.model_fp32_groups)

                shard_groups_flat = []
                if hasattr(self, 'shard_fp32_from_float16_groups'):
                    shard_groups_flat.extend(self.shard_fp32_from_float16_groups)
                if hasattr(self, 'shard_fp32_groups'):
                    shard_groups_flat.extend(self.shard_fp32_groups)

                for model_group, shard_group in zip(model_groups_flat, shard_groups_flat):
                    for model_param, shard_param in zip(model_group, shard_group):
                        shard_to_model[id(shard_param)] = model_param

                shard_param_groups = []
                if hasattr(self, 'shard_fp32_from_float16_groups'):
                    shard_param_groups.extend(self.shard_fp32_from_float16_groups)
                if hasattr(self, 'shard_fp32_groups'):
                    shard_param_groups.extend(self.shard_fp32_groups)

                for shard_group in shard_param_groups:
                    for shard_param in shard_group:
                        model_param = shard_to_model.get(id(shard_param), None)
                        if model_param is None:
                            continue

                        is_dion_param = getattr(model_param, 'is_dion_param', False)
                        if is_dion_param:
                            # Already handled in Part 1
                            continue

                        # Non-Dion param: clip shard_param.grad
                        grad = getattr(shard_param, 'main_grad', None)
                        if grad is None:
                            grad = shard_param.grad
                        if grad is None:
                            continue

                        # Apply standard filters using model_param attributes
                        is_not_shared = param_is_not_shared(model_param)
                        is_not_tp_duplicate = tensor_parallel.param_is_not_tensor_parallel_duplicate(model_param)

                        if is_not_shared and is_not_tp_duplicate:
                            grad.mul_(clip_coeff)

        return grad_norm

    def prepare_grads(self) -> bool:
        """
        Override to ensure async reduce-scatter completes before gradient copy.

        Note: When overlap_grad_reduce=true, reduce-scatter is async.
        We MUST wait for completion before copying gradients from buffer to param.main_grad.

        Root cause: Without this, _copy_model_grads_to_main_grads() reads buffer
        BEFORE reduce-scatter completes → all gradients appear as zero!

        This follows Megatron's standard pattern: finalize_model_grads() calls
        finish_grad_sync() before optimizer step.
        """

        # Wait for async reduce-scatter to complete (Megatron standard pattern)
        # We manually wait on bucket groups to avoid AssertionError in finish_grad_sync()
        # when some buckets haven't started RS yet
        if not self.is_stub_optimizer and hasattr(self, 'model_chunks'):

            for chunk_idx, model_chunk in enumerate(self.model_chunks):
                # Access bucket groups directly to manually wait on handles
                if hasattr(model_chunk, 'bucket_groups'):
                    # Wait for any pending reduce-scatter handles (usually already done by finish_grad_sync)
                    pending_count = 0
                    for bg_idx, bucket_group in enumerate(model_chunk.bucket_groups):
                        if hasattr(bucket_group, 'grad_reduce_handle') and bucket_group.grad_reduce_handle is not None:
                            bucket_group.grad_reduce_handle.wait()
                            bucket_group.grad_reduce_handle = None
                            pending_count += 1
                # Also handle expert_parallel_bucket_groups if they exist
                if hasattr(model_chunk, 'expert_parallel_bucket_groups') and model_chunk.expert_parallel_bucket_groups:
                    for bg_idx, bucket_group in enumerate(model_chunk.expert_parallel_bucket_groups):
                        if hasattr(bucket_group, 'grad_reduce_handle') and bucket_group.grad_reduce_handle is not None:
                            bucket_group.grad_reduce_handle.wait()
                            bucket_group.grad_reduce_handle = None


        # Optional sync for debugging (DION_FORCE_SYNC=1)
        if os.environ.get('DION_FORCE_SYNC', '0') == '1':
            torch.cuda.synchronize()

        # Call parent's prepare_grads (includes _copy_model_grads_to_main_grads)
        found_inf_flag = super().prepare_grads()

        return found_inf_flag

    def _copy_model_params_to_main_params(self, state_dict=None):
        """Copy model params to main params with FS sharding awareness.

        Handles FS-sharded parameters correctly when loading from checkpoint.
        """
        from .cpu_offloading import HybridDeviceOptimizer
        if isinstance(self.optimizer, HybridDeviceOptimizer):
            self.optimizer.update_fp32_param_by_new_param()
            return

        if self.ddp_config.use_megatron_fsdp:
            return

        # Precision-aware optimizer early return
        if self.config.use_precision_aware_optimizer_no_fp8_or_ds_fp8:
            return

        if state_dict is not None:
            # Build mapping for state dict params
            model_param_to_state_dict_param_map = self._build_model_param_to_state_dict_param_map(
                state_dict
            )

        # Utility method for copying group params
        def copy_group_params(model_groups, shard_main_groups):
            for model_group, shard_main_group in zip(model_groups, shard_main_groups):
                for model_param, shard_main_param in zip(model_group, shard_main_group):
                    # Skip None params (can happen for excluded params)
                    if shard_main_param is None:
                        continue

                    param_range_map = self._get_model_param_range_map(model_param)
                    param_range = param_range_map["param"]
                    dion_info = self._get_dion_info_for_param(model_param)

                    # Get source param
                    if state_dict is not None:
                        source_param = model_param_to_state_dict_param_map[model_param]
                    else:
                        source_param = model_param

                    # Handle 2D Dion params (checkpoint loading with tp_split_dim-aware FS sharding)
                    if dion_info.get('is_dion', False):
                        # Source param is full parameter (m, n) from checkpoint
                        # Need to extract FS shard based on fs_split_dim
                        start_idx = dion_info['start_idx']
                        end_idx = dion_info['end_idx']
                        fs_split_dim = dion_info['fs_split_dim']

                        from ..fp8_utils import is_float8tensor, dequantize_fp8_tensor

                        # Handle FP8 tensors
                        if is_float8tensor(source_param):
                            source_param_fp32 = dequantize_fp8_tensor(source_param)
                        else:
                            source_param_fp32 = source_param

                        # Use checkpoint shape directly (already TP-partitioned)
                        if source_param_fp32.ndim == 2:
                            source_2d = source_param_fp32
                        else:
                            # Fallback: reshape to match shard_main_param
                            source_2d = source_param_fp32.view(shard_main_param.shape)

                        # Extract FS shard (use .clone() to avoid same-memory issues)
                        if fs_split_dim == 0:
                            # FS splits rows
                            shard_model_param = source_2d[start_idx:end_idx, :].clone().view(-1)
                        else:
                            # FS splits cols
                            shard_model_param = source_2d[:, start_idx:end_idx].clone().view(-1)

                        # Verify sizes match
                        assert shard_model_param.numel() == shard_main_param.numel(), \
                            f"FS shard size mismatch: shard_model_param={shard_model_param.numel()}, " \
                            f"shard_main_param={shard_main_param.numel()}, " \
                            f"fs_split_dim={fs_split_dim}, start_idx={start_idx}, end_idx={end_idx}, " \
                            f"source_shape={source_param_fp32.shape}, " \
                            f"global_shape={dion_info['global_shape']}"

                        # Copy to main param (reshape to 2D)
                        shard_main_param.data.copy_(shard_model_param.reshape(shard_main_param.shape))

                        # model_param.data unchanged (hooks handle all-gather during fwd/bwd)
                    else:
                        # 1D params: copy entire param (not sharded)
                        from ..fp8_utils import is_float8tensor, dequantize_fp8_tensor

                        if is_float8tensor(source_param):
                            source_param_fp32 = dequantize_fp8_tensor(source_param)
                        else:
                            source_param_fp32 = source_param

                        # Non-Dion params use DP sharding (bias, layernorm, embedding, output_layer)
                        shard_model_param = source_param_fp32.view(-1)[
                            param_range.start : param_range.end
                        ]
                        assert param_range.size == shard_main_param.nelement(), \
                            f"Non-Dion param size mismatch: param_range.size={param_range.size}, " \
                            f"shard_main={shard_main_param.nelement()}, " \
                            f"source_shape={source_param.shape}, " \
                            f"param_range=[{param_range.start}:{param_range.end}], " \
                            f"is_2D={source_param.dim() >= 2}"

                        # Ensure shapes are compatible before copying
                        if shard_model_param.shape != shard_main_param.shape:
                            shard_model_param = shard_model_param.view_as(shard_main_param)
                        shard_main_param.data.copy_(shard_model_param)

        # Copy model groups to shard groups
        copy_group_params(self.model_float16_groups, self.shard_fp32_from_float16_groups)
        copy_group_params(self.model_fp32_groups, self.shard_fp32_groups)

    def _copy_main_params_to_model_params(self):
        """Copy parameters with efficient batch flattening for 2D params.

        MANUAL mode (no hooks): Copy updated shard params to model params.
        Model params will be all-gathered before next forward.
        """
        if self.is_stub_optimizer:
            return

        # FSDP early return
        if self.ddp_config.use_megatron_fsdp:
            for model_chunk in self.model_chunks:
                model_chunk.param_and_grad_buffer.copy_main_weights_to_model_weights()
            return

        # Precision-aware optimizer early return
        if self.config.use_precision_aware_optimizer_no_fp8_or_ds_fp8:
            return

        # Copy updated optimizer shards (FP32) to model params (BF16)
        # Use model_float16_groups/shard_fp32_from_float16_groups directly
        # to avoid object identity mismatch in shard lookups

        param_count = 0

        # Identity check: verify shard_fp32_from_float16_groups matches optimizer.param_groups
        if not hasattr(self, '_identity_check_done'):
            self._identity_check_done = False

        if not self._identity_check_done and hasattr(self, 'optimizer'):
            self._identity_check_done = True
            mismatch_count = 0
            match_count = 0

            # Build data_ptr → param mapping from optimizer.param_groups
            opt_param_by_ptr = {}
            for pg in self.optimizer.param_groups:
                for p in pg['params']:
                    opt_param_by_ptr[p.data_ptr()] = p

            # Check each param in shard_fp32_from_float16_groups
            for gi, (model_group, shard_group) in enumerate(zip(self.model_float16_groups, self.shard_fp32_from_float16_groups)):
                for pi, (model_param, shard_param) in enumerate(zip(model_group, shard_group)):
                    if shard_param is None:
                        continue

                    shard_ptr = shard_param.data_ptr()
                    opt_p = opt_param_by_ptr.get(shard_ptr)

                    if opt_p is None:
                        # NOT IN optimizer.param_groups!
                        if mismatch_count < 5:
                            logger.error(
                                f"[IDENTITY MISMATCH] shard_fp32_from_float16_groups[{gi}][{pi}] "
                                f"(shape={shard_param.shape}, ptr={shard_ptr}) "
                                f"NOT FOUND in optimizer.param_groups! Dion updates will be LOST!"
                            )
                        mismatch_count += 1
                    elif opt_p is not shard_param:
                        # Same data_ptr but different object (unexpected)
                        if mismatch_count < 5:
                            logger.warning(
                                f"[Dion] shard_fp32_from_float16_groups[{gi}][{pi}] object mismatch: "
                                f"same data_ptr={shard_ptr} but different object. "
                                f"id(shard)={id(shard_param)}, id(opt)={id(opt_p)}"
                            )
                        mismatch_count += 1
                    else:
                        match_count += 1

            if mismatch_count > 0:
                logger.error(
                    f"[Dion] {mismatch_count} params in shard_fp32_from_float16_groups "
                    f"not in optimizer.param_groups ({match_count} matched). "
                    f"Dion updates may be overwritten."
                )
            # Identity check passed (match_count params matched)

        # Iterate over paired groups (model, shard_fp32, shard_fp16)
        for model_group, shard_group, shard16_group in zip(
            self.model_float16_groups,
            self.shard_fp32_from_float16_groups,
            self.shard_float16_groups
        ):
            for model_param, shard_param, shard16_param in zip(model_group, shard_group, shard16_group):
                if shard_param is None:
                    continue

                # shard_param is the FP32 optimizer shard that Dion updated
                opt_shard = shard_param

                # Dion: use registered data_shard; non-Dion: use FP16 shard
                data_shard = self._get_data_shard(model_param)
                if data_shard is None:
                    # Non-Dion param: use FP16 shard from shard_float16_groups
                    data_shard = shard16_param

                # Copy updated optimizer shard to data_shard buffer
                # Use .data.copy_() to bypass autograd
                data_shard.data.copy_(opt_shard.to(data_shard.dtype))

                # Update _fs_shard reference
                model_param._fs_shard = data_shard

                # Also copy to param.data's local shard range
                # This keeps param.data (FULL) up-to-date with optimizer results
                dion_info = self._get_dion_info_for_param(model_param)

                if dion_info.get('is_dion', False):
                    fs_split_dim = dion_info['fs_split_dim']
                    start_idx = dion_info['start_idx']
                    end_idx = dion_info['end_idx']

                    # Verify start_idx/end_idx != 0/0 (otherwise no update copied)
                    if start_idx == 0 and end_idx == 0:
                        if not hasattr(self, '_zero_range_warned'):
                            self._zero_range_warned = 0
                        if self._zero_range_warned < 5:
                            param_name = getattr(model_param, '_param_name', f'shape={model_param.shape}')
                            logger.error(
                                f"[ZERO RANGE] start_idx=0, end_idx=0 for param {param_name}! "
                                f"No data will be copied to model_param.data! "
                                f"dion_info={dion_info}"
                            )
                            self._zero_range_warned += 1

                    # Copy data_shard to param.data's local range
                    if fs_split_dim == 0:
                        # Row split: data_shard → (rows, cols)
                        expected_rows = end_idx - start_idx
                        expected_cols = model_param.data.shape[1]

                        if data_shard.numel() != expected_rows * expected_cols:
                            param_name = getattr(model_param, '_param_name', f'id_{id(model_param)}')
                            raise RuntimeError(
                                f"[Dion] Row split shape mismatch: param={param_name}, "
                                f"data_shard.numel()={data_shard.numel()}, "
                                f"expected={expected_rows}x{expected_cols}={expected_rows*expected_cols}, "
                                f"fs_split_dim={fs_split_dim}, range=[{start_idx}:{end_idx}]"
                            )

                        # Clone if same-storage to ensure copy_() works
                        src_view = data_shard.view(expected_rows, expected_cols)
                        target_slice = model_param.data[start_idx:end_idx, :]
                        if src_view.data_ptr() == target_slice.data_ptr():
                            # Same storage detected - clone to avoid no-op copy
                            src_view = src_view.clone()
                        target_slice.copy_(src_view)
                    else:
                        # Column split: data_shard → (rows, cols)
                        expected_rows = model_param.data.shape[0]
                        expected_cols = end_idx - start_idx

                        if data_shard.numel() != expected_rows * expected_cols:
                            param_name = getattr(model_param, '_param_name', f'id_{id(model_param)}')
                            raise RuntimeError(
                                f"[Dion] Column split shape mismatch: param={param_name}, "
                                f"data_shard.numel()={data_shard.numel()}, "
                                f"expected={expected_rows}x{expected_cols}={expected_rows*expected_cols}, "
                                f"fs_split_dim={fs_split_dim}, range=[{start_idx}:{end_idx}]"
                            )

                        # Clone if same-storage to ensure copy_() works
                        src_view = data_shard.view(expected_rows, expected_cols)
                        target_slice = model_param.data[:, start_idx:end_idx]
                        if src_view.data_ptr() == target_slice.data_ptr():
                            src_view = src_view.clone()
                        target_slice.copy_(src_view)

                    param_count += 1

                else:
                    # Non-Dion params: copy only if not already sharing storage
                    same_storage = (data_shard.storage().data_ptr() == model_param.data.storage().data_ptr())

                    if same_storage:
                        # View relationship - already updated, no action needed
                        pass
                    elif data_shard.numel() == model_param.data.numel():
                        # Different storage but same size - copy needed
                        model_param.data.copy_(data_shard.view(model_param.data.shape))
                    else:
                        # Different storage AND different size - unexpected, warn
                        param_name = get_param_name(model_param, fallback_style="shape")
                        logger.warning(
                            f"[NON-Dion COPY SKIP] param={param_name}, "
                            f"data_shard.numel()={data_shard.numel()} != model_param.numel()={model_param.data.numel()}, "
                            f"same_storage={same_storage}, skipping copy to model_param.data"
                        )

                    param_count += 1

        # FS All-Gather: restore full param.data for VLLM export and checkpoint saving
        if self.fs_size > 1 and self.fs_group is not None:
            dion_ag_count = 0
            non_dion_ag_count = 0

            for model_group, shard_group in zip(self.model_float16_groups, self.shard_fp32_from_float16_groups):
                for model_param, shard_param in zip(model_group, shard_group):
                    if shard_param is None:
                        continue

                    param_range_map = self._get_model_param_range_map(model_param)
                    dion_info = self._get_dion_info_for_param(model_param)

                    if not dion_info.get('is_dion', False):
                        # Non-Dion params: use flat DP sharding, all-gather to restore full param
                        param_range = param_range_map.get("param")
                        if param_range is None:
                            continue

                        # Flat shard range for this FS rank
                        flat_start = param_range.start
                        flat_end = param_range.end
                        local_shard_size = flat_end - flat_start

                        if local_shard_size <= 0:
                            continue

                        # Extract local shard from flat view
                        param_flat = model_param.data.view(-1)
                        local_shard = param_flat[flat_start:flat_end].contiguous()

                        # All-gather: each rank contributes its flat shard
                        # Use max shard size for padding (ceiling division)
                        total_numel = model_param.data.numel()
                        max_shard_size = (total_numel + self.fs_size - 1) // self.fs_size

                        # Pad local shard to max_shard_size
                        padded_shard = torch.zeros(
                            max_shard_size,
                            dtype=model_param.dtype, device=model_param.device
                        )
                        padded_shard[:local_shard_size].copy_(local_shard)

                        # All-gather (all ranks have same size now)
                        gathered_shards = [torch.empty_like(padded_shard) for _ in range(self.fs_size)]
                        dist.all_gather(gathered_shards, padded_shard, group=self.fs_group)

                        # Also all-gather each rank's actual range (flat_start, flat_end)
                        # This handles cases where param spans multiple bucket shards unevenly
                        local_range_info = torch.tensor(
                            [flat_start, flat_end], device=model_param.device, dtype=torch.long
                        )
                        all_range_infos = [torch.empty_like(local_range_info) for _ in range(self.fs_size)]
                        dist.all_gather(all_range_infos, local_range_info, group=self.fs_group)

                        # Unpack: copy each rank's shard to its actual flat range
                        for rank_i in range(self.fs_size):
                            r_start, r_end = all_range_infos[rank_i].tolist()
                            r_size = r_end - r_start
                            if r_size > 0:
                                param_flat[r_start:r_end].copy_(gathered_shards[rank_i][:r_size])

                        non_dion_ag_count += 1
                        continue

                    # Get FS shard info (Dion params only from here)
                    fs_split_dim = dion_info['fs_split_dim']
                    start_idx = dion_info['start_idx']
                    end_idx = dion_info['end_idx']

                    if start_idx == end_idx:
                        continue  # Skip invalid range

                    # Extract local shard (already updated in param.data)
                    if fs_split_dim == 0:
                        local_shard = model_param.data[start_idx:end_idx, :].contiguous()
                        local_shard_size = end_idx - start_idx
                        other_dim_size = model_param.shape[1]
                    else:
                        local_shard = model_param.data[:, start_idx:end_idx].contiguous()
                        local_shard_size = end_idx - start_idx
                        other_dim_size = model_param.shape[0]

                    # All-gather from all FS ranks
                    # dist.all_gather requires all tensors to have same size!
                    # Use max_shard_size (ceiling division) and pad smaller shards
                    split_dim_size = model_param.shape[fs_split_dim]
                    max_shard_size = (split_dim_size + self.fs_size - 1) // self.fs_size

                    # Pad local shard to max_shard_size
                    if fs_split_dim == 0:
                        padded_shard = torch.zeros(
                            max_shard_size, other_dim_size,
                            dtype=model_param.dtype, device=model_param.device
                        )
                        padded_shard[:local_shard_size, :].copy_(local_shard)
                    else:
                        padded_shard = torch.zeros(
                            other_dim_size, max_shard_size,
                            dtype=model_param.dtype, device=model_param.device
                        )
                        padded_shard[:, :local_shard_size].copy_(local_shard)

                    # All-gather (all ranks have same size now)
                    gathered_shards = [torch.empty_like(padded_shard) for _ in range(self.fs_size)]
                    dist.all_gather(gathered_shards, padded_shard, group=self.fs_group)

                    # Unpack: extract actual shard from each rank's padded buffer and copy to param.data
                    for rank_i in range(self.fs_size):
                        r_start, r_end = compute_fs_shard_range(split_dim_size, self.fs_size, rank_i)
                        actual_size = r_end - r_start

                        if fs_split_dim == 0:
                            model_param.data[r_start:r_end, :].copy_(gathered_shards[rank_i][:actual_size, :])
                        else:
                            model_param.data[:, r_start:r_end].copy_(gathered_shards[rank_i][:, :actual_size])

                    dion_ag_count += 1

            # Also process model_fp32_groups (FP32 params like LayerNorm, biases)
            fp32_ag_count = 0
            for model_group, shard_group in zip(self.model_fp32_groups, self.shard_fp32_groups):
                for model_param, shard_param in zip(model_group, shard_group):
                    if shard_param is None:
                        continue

                    param_range_map = self._get_model_param_range_map(model_param)
                    param_range = param_range_map.get("param")
                    if param_range is None:
                        continue

                    flat_start = param_range.start
                    flat_end = param_range.end
                    local_shard_size = flat_end - flat_start

                    if local_shard_size <= 0:
                        continue

                    # Extract local shard from flat view
                    param_flat = model_param.data.view(-1)
                    local_shard = param_flat[flat_start:flat_end].contiguous()

                    # All-gather
                    total_numel = model_param.data.numel()
                    max_shard_size = (total_numel + self.fs_size - 1) // self.fs_size

                    padded_shard = torch.zeros(
                        max_shard_size,
                        dtype=model_param.dtype, device=model_param.device
                    )
                    padded_shard[:local_shard_size].copy_(local_shard)

                    gathered_shards = [torch.empty_like(padded_shard) for _ in range(self.fs_size)]
                    dist.all_gather(gathered_shards, padded_shard, group=self.fs_group)

                    # Also all-gather each rank's actual range (flat_start, flat_end)
                    local_range_info = torch.tensor(
                        [flat_start, flat_end], device=model_param.device, dtype=torch.long
                    )
                    all_range_infos = [torch.empty_like(local_range_info) for _ in range(self.fs_size)]
                    dist.all_gather(all_range_infos, local_range_info, group=self.fs_group)

                    # Unpack using actual ranges from each rank
                    for rank_i in range(self.fs_size):
                        r_start, r_end = all_range_infos[rank_i].tolist()
                        r_size = r_end - r_start
                        if r_size > 0:
                            param_flat[r_start:r_end].copy_(gathered_shards[rank_i][:r_size])

                    fp32_ag_count += 1



    def _update_optimizer_param_groups(self):
        """Update optimizer param groups after rebuilding."""
        from .cpu_offloading import HybridDeviceOptimizer

        if isinstance(self.optimizer, HybridDeviceOptimizer):
            self.optimizer = HybridDeviceOptimizer(
                params=[g["orig_group"] for g in self.opt_group_ranges],
                **self.optimizer.defaults
            )
        else:
            self.optimizer.param_groups = [g["orig_group"] for g in self.opt_group_ranges]
            self.optimizer.load_state_dict(self.optimizer.state_dict())

    # DO Overlap Integration

    def finish_grad_sync(self):
        """
        Wait for async gradient reduce-scatter (DO bucket groups + FS adapters).

        This is called before optimizer.step() to ensure all gradients
        are properly reduced and sharded.
        """
        # 1) Wait DO bucket groups (if exists)
        if not hasattr(self, 'per_model_bucket_groups'):
            return  # No bucket groups

        for bucket_groups in self.per_model_bucket_groups.values():
            for bg in bucket_groups:
                if hasattr(bg, "finish_grad_sync"):
                    bg.finish_grad_sync()

        # Release RS buffers after grad sync completes
        self._release_rs_buffers()

    def _release_rs_buffers(self):
        """
        Release RS buffers after reduce-scatter completes.
        Buffers will be lazily reallocated on next backward pass.
        Note: Using = None only (not storage().resize_(0)) because the buffer
        may have views that would cause illegal memory access if storage is resized.
        """
        if not hasattr(self, 'buffers'):
            return

        for buffer in self.buffers:
            for bucket in buffer.buckets:
                # Release dion_grad_buffer
                if hasattr(bucket, 'dion_grad_buffer') and bucket.dion_grad_buffer is not None:
                    bucket.dion_grad_buffer = None
                # Release _cached_rs_pack_buffer (bucket level only)
                if hasattr(bucket, '_cached_rs_pack_buffer') and bucket._cached_rs_pack_buffer is not None:
                    bucket._cached_rs_pack_buffer = None

    def start_param_gather_after_step(self):
        """
        DEPRECATED: No longer used with bucket-wise async overlap.

        FS parameter all-gather is now handled by the standard Megatron-Core DO mechanism:
        - Forward pre-hooks call finish_param_sync() for each bucket
        - finish_param_sync() waits for async FS all-gather (bucket-wise)
        - Chaining happens automatically via next_param_gather_bucket_group

        This follows the standard DO pattern exactly, with FS all-gathers
        integrated into the bucket-wise async dispatch and wait mechanism.
        """
        # No-op: FS all-gather now handled by standard bucket-wise mechanism
        pass

    @torch.no_grad()
    def step_with_ready_grads(self):
        """
        Override to integrate DO overlap.

        1. Restore param.data to shard BEFORE optimizer step (backward-post, as per guide)
        2. Call parent (DO handles grad sync + optimizer update on SHARD)
        3. Start param AG for next iteration
        """
        ok = super().step_with_ready_grads()

        # param.data stays FULL, update _fs_shard from local range
        if self.use_fs_shard_alloc:
            update_count = 0
            for gbuf_idx, buffer in enumerate(self.buffers):
                if not hasattr(buffer, 'dion_param_layout') or not buffer.dion_param_layout:
                    continue

                for entry in buffer.dion_param_layout:
                    param = entry['param']
                    if hasattr(param, '_fs_shard'):
                        # Do NOT change param.data
                        # param.data stays FULL for DDP hooks
                        # _fs_shard will be updated via _copy_main_params_to_model_params
                        update_count += 1

        # Immediately start overlapped param all-gather for next forward
        self.start_param_gather_after_step()

        return ok

    def sharded_state_dict(
        self,
        model_sharded_state_dict=None,
        is_loading: bool = False,
        sharding_type=None,
        metadata=None,
    ):
        """Override to handle Dion's 2D FS sharding for checkpoint save/load.

        Dion uses 2D FS sharding (row/col split) instead of standard DO's flat sharding.
        The parent DO's dp_reshardable format assumes flat sharding and fails with
        negative tensor dimensions for Dion params.

        Solution: Use non-reshardable format (ShardedObject) which saves each DP rank's
        state as-is. Requires same parallelism config (FS, TP, EP) for save and load.

        Saves two parts:
        1. Common state (step, param_groups) - same format as parent DO
        2. MegatronDion's inner parameter state (momentum, Q, exp_avg, etc.)
           as a single ShardedObject per DP rank
        """
        from ..dist_checkpointing.mapping import ShardedObject

        if model_sharded_state_dict is None:
            model_sharded_state_dict = {}

        dp_rank = self.data_parallel_group.rank()
        base_key = f'optimizer.distributed.dp_group_idx_{self.data_parallel_group_idx}'
        replica_id = (self.distributed_optimizer_instance_id, 0, dp_rank)

        # Part 1: Common state (step, param_groups) - same format as parent
        common_state = self.state_dict()
        state_dict = {
            k: ShardedObject(f'{base_key}.{k}', v, (1,), (0,), replica_id=replica_id)
            for k, v in common_state.items()
        }

        # Part 2: MegatronDion's inner parameter state (momentum, Q, exp_avg, etc.)
        # Use model param NAME as key (deterministic across runs) instead of integer
        # index (param ordering in param_groups can differ between save and load runs)
        name_to_state = {}
        for p, s in self.optimizer.state.items():
            name = self._get_shard_param_name(p)
            if name is not None:
                name_to_state[name] = s
        state_dict['param_state'] = ShardedObject(
            f'{base_key}.dion_param_state', name_to_state, (1,), (0,),
            replica_id=replica_id,
        )
        state_dict['param_state_sharding_type'] = 'dion_non_reshardable'

        # NOTE: Do NOT call self.load_state_dict() during is_loading.
        # The parent's load_state_dict allocates dummy flat 1D tensors (exp_avg, exp_avg_sq)
        # that overwrite MegatronDion's proper state (momentum, Q, etc.) and cause shape
        # mismatches. Our load_state_dict override below handles restoration correctly.

        return state_dict

    def load_state_dict(self, state_dict):
        """Override to handle Dion-specific checkpoint loading.

        When param_state_sharding_type is 'dion_non_reshardable':
        - Restores MegatronDion's full state (momentum, Q, exp_avg, etc.)
        - Restores param_groups hyperparameters (lr, step, mu, etc.)
        - Restores grad_scaler if present

        Otherwise falls back to parent DO's load_state_dict.
        """
        if state_dict.get('param_state_sharding_type') != 'dion_non_reshardable':
            super().load_state_dict(state_dict)
            return

        # Restore MegatronDion's parameter state from checkpoint
        # State is keyed by model param name (deterministic across runs)
        name_to_state = state_dict.get('param_state')
        if name_to_state and isinstance(name_to_state, dict) and len(name_to_state) > 0:
            restored = 0
            missing = 0
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    name = self._get_shard_param_name(p)
                    if name is None:
                        missing += 1
                        continue
                    if name in name_to_state:
                        saved = name_to_state[name]
                        # Validate momentum shape matches param (topology must be same)
                        saved_momentum = saved.get('momentum')
                        if (isinstance(saved_momentum, torch.Tensor)
                                and saved_momentum.shape != p.shape):
                            logger.warning(
                                f'[Dion] Shape mismatch for {name}: '
                                f'saved momentum={tuple(saved_momentum.shape)}, '
                                f'current param={tuple(p.shape)}. '
                                f'Skipping restore (will re-init).'
                            )
                            missing += 1
                            continue
                        # Cast tensors to match param device
                        restored_state = {}
                        for k, v in saved.items():
                            if isinstance(v, torch.Tensor):
                                restored_state[k] = v.to(device=p.device)
                            else:
                                restored_state[k] = v
                        self.optimizer.state[p] = restored_state
                        restored += 1
                    else:
                        missing += 1
            logger.info(
                f'[Dion] Restored {restored} param states from checkpoint '
                f'({missing} missing, will re-init on first step)'
            )
        else:
            logger.info('[Dion] No param state in checkpoint, will re-initialize on first step')

        # Restore param_groups hyperparameters (lr, step, mu, etc.)
        if 'optimizer' in state_dict:
            saved_opt = state_dict['optimizer']
            if isinstance(saved_opt, dict) and 'param_groups' in saved_opt:
                for current_pg, saved_pg in zip(
                    self.optimizer.param_groups, saved_opt['param_groups']
                ):
                    for key, value in saved_pg.items():
                        if key != 'params':
                            current_pg[key] = value

        # Restore grad_scaler
        if 'grad_scaler' in state_dict and self.grad_scaler:
            self.grad_scaler.load_state_dict(state_dict['grad_scaler'])

        # Sync fp32 main params from bf16 model params.
        # The fp32 copies still hold stale values from __init__ (cloned before checkpoint
        # loading). Without this, the first optimizer step would corrupt model params.
        # bf16->fp32 cast has ~1e-3 precision loss, acceptable for training resumption.
        self._copy_model_params_to_main_params()

    def _get_shard_param_name(self, shard_param):
        """Get deterministic model param name for a shard param.

        Traces shard_param -> model_param -> param_to_name mapping.
        Returns None if name cannot be found.
        """
        model_param = getattr(shard_param, '_model_param', None)
        if model_param is None:
            return None
        for buffer in self.buffers:
            if hasattr(buffer, 'param_to_name') and model_param in buffer.param_to_name:
                return buffer.param_to_name[model_param]
        return None

    def offload_to_cpu(self):
        """Clean up Dion-specific buffers during offload.

        Standard DO pattern handles most cleanup via:
        1. Buffer level: param_and_grad_buffer.offload_to_cpu()
        2. Optimizer state: move_optimizer("cpu")
        3. Q buffers: Auto-cleared when device changes
        """
        if hasattr(self, 'optimizer') and hasattr(self.optimizer, 'offload_to_cpu'):
            self.optimizer.offload_to_cpu()

