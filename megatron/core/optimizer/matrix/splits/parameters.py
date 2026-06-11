"""General split-parameter metadata helpers.

This module intentionally stops at policy/metadata routing.  Layout math stays
in qkv/qkvg/gdn/linear helpers, and optimizer state stays in Muon/Dion/ARO.
"""

from __future__ import annotations

from typing import Optional

import torch

from .axis import clear_split_metadata
from .gdn import copy_gdn_split_metadata, is_gdn_param
from .linear import copy_linear_split_metadata, is_linear_split_param
from .qkv import copy_qkv_split_metadata, is_qkv_param
from .qkvg import copy_qkvg_split_metadata, is_qkvg_param


def copy_parameter_split_metadata(
    destination_tensor: torch.Tensor,
    source_tensor: torch.Tensor,
) -> None:
    """Copy optimizer-only split metadata from a model tensor to a shard tensor."""
    clear_split_metadata(destination_tensor)
    split_tags = (
        bool(is_qkvg_param(source_tensor) or hasattr(source_tensor, "qkvg_split_shapes")),
        bool(is_qkv_param(source_tensor) or hasattr(source_tensor, "qkv_split_shapes")),
        bool(is_gdn_param(source_tensor) or hasattr(source_tensor, "gdn_split_shapes")),
        bool(is_linear_split_param(source_tensor) or hasattr(source_tensor, "linear_split_rows")),
    )
    if sum(int(tag) for tag in split_tags) > 1:
        raise RuntimeError(
            "[MATRIX_SPLIT_PARAMETER_AMBIGUOUS] "
            f"param={getattr(source_tensor, '_param_name', '') or id(source_tensor)}"
        )

    if split_tags[0]:
        copy_qkvg_split_metadata(destination_tensor, source_tensor)
        return
    if split_tags[1]:
        copy_qkv_split_metadata(destination_tensor, source_tensor)
        return
    if split_tags[2]:
        copy_gdn_split_metadata(destination_tensor, source_tensor)
        return
    if split_tags[3]:
        copy_linear_split_metadata(destination_tensor, source_tensor)


def split_parameter_state_kind(state: Optional[dict]) -> Optional[str]:
    """Return the persisted split layout marker, if any."""
    if not state:
        return None
    markers = (
        ("qkvg", "qkvg_split_qkvg"),
        ("qkv", "qkv_split_qkv"),
        ("gdn", "gdn_split_gdn"),
        ("linear", "linear_split_linear"),
    )
    active = [kind for kind, key in markers if bool(state.get(key, False))]
    if len(active) > 1:
        raise RuntimeError(f"[MATRIX_SPLIT_STATE_AMBIGUOUS] active={active}")
    return active[0] if active else None


def has_parameter_split_metadata(dist_meta) -> bool:
    """Return whether distributed metadata describes an optimizer-only split parent."""
    if dist_meta is None:
        return False
    return any(
        getattr(dist_meta, attr, None) is not None
        for attr in (
            "qkvg_split_shapes",
            "qkv_split_shapes",
            "gdn_split_shapes",
            "linear_split_rows",
        )
    )
