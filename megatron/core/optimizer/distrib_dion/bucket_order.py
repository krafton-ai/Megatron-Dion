"""Deterministic bucket parameter reordering across DP ranks.

The Dion distributed optimizer relies on consistent parameter ordering across
data-parallel ranks when building per-bucket layouts.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import torch.distributed as dist

logger = logging.getLogger(__name__)


def reorder_bucket_params_(
    *,
    param_and_grad_buffer,
    bucket,
    dp_group,
    dp_rank: int,
    name_bank: Dict,
) -> List:
    """Reorder `bucket.params` deterministically across DP ranks.

    The final ordering is defined by DP-rank0's `param_to_name` name list for
    this bucket, broadcast to all DP ranks. This updates
    `param_and_grad_buffer.param_to_name` to match that shared ordering and
    returns `ordered_params`.
    """
    original_name_map = (
        dict(param_and_grad_buffer.param_to_name)
        if hasattr(param_and_grad_buffer, "param_to_name")
        else {}
    )

    # Broadcast DP rank0's name list for the shared bucket ordering.
    local_param_names = []
    for param in bucket.params:
        name = original_name_map.get(param, "")
        if not name and name_bank:
            name = name_bank.get(param, "")
        local_param_names.append(name)

    ordered_names = local_param_names if dp_rank == 0 else None
    broadcast_payload = [ordered_names]
    # `src` is interpreted as a global rank when `group` is provided. Use the
    # actual global rank for group-rank0 so PP/TP subgroup layouts remain valid.
    group_src = dist.get_global_rank(dp_group, 0)
    dist.broadcast_object_list(broadcast_payload, src=group_src, group=dp_group)
    ordered_names = (
        broadcast_payload[0] if broadcast_payload and broadcast_payload[0] is not None else []
    )

    if len(ordered_names) != len(bucket.params):
        logger.error(
            "[Dion] len(ordered_names)=%s != len(bucket.params)=%s",
            len(ordered_names),
            len(bucket.params),
        )
        raise RuntimeError("param_to_name broadcast length mismatch")

    if any(not name for name in ordered_names):
        bad_idx = [i for i, name in enumerate(ordered_names) if not name]
        logger.error("[Dion] Empty names at indices %s", bad_idx)
        raise RuntimeError("param_to_name contains empty names")

    # Build name -> param mapping with duplicate check.
    name_to_param = {}
    for param in bucket.params:
        name = original_name_map.get(param, "")
        if not name:
            continue
        if name in name_to_param:
            logger.error("[Dion] Duplicate param name detected locally: %s", name)
            raise RuntimeError("Duplicate parameter names found in bucket.params")
        name_to_param[name] = param

    ordered_params = []
    for name in ordered_names:
        if name not in name_to_param:
            logger.error("[Dion] Ordered name '%s' not found in local bucket.params", name)
            raise RuntimeError("Canonical param order mismatch across DP ranks")
        ordered_params.append(name_to_param[name])

    # Verify names consistency.
    seen_names = set()
    for name in ordered_names:
        if not name:
            continue
        if name in seen_names:
            logger.error("[Dion] Duplicate param name detected: %s", name)
            raise RuntimeError("Duplicate parameter names found in bucket.params")
        seen_names.add(name)

    new_name_map = dict(original_name_map)
    for index, param in enumerate(ordered_params):
        new_name_map[param] = ordered_names[index]

    param_and_grad_buffer.param_to_name = new_name_map
    if name_bank is not None:
        name_bank.update(new_name_map)

    return ordered_params
