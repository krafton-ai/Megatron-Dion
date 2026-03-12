"""Diagnostics helpers for Dion grad wiring.

These helpers keep logging behavior and debug formatting stable while removing
diagnostic scaffolding from the main optimizer wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from typing import Callable, Optional

import torch
import torch.distributed as dist

from .param_naming import get_optimizer_param_name


logger = logging.getLogger(__name__)


def get_opt_step(optimizer) -> Optional[int]:
    """Best-effort optimizer step lookup for diagnostics."""
    try:
        param_groups = getattr(optimizer, "param_groups", None)
        group0 = param_groups[0] if param_groups else None
        if isinstance(group0, dict) and "step" in group0:
            return int(group0["step"])
    except Exception:
        pass
    return None


@dataclass
class GradIssueLogger:
    """Stateful helper for structured grad-issue logging."""

    optimizer: object
    data_parallel_group: Optional[object]
    fs_rank: int
    fs_size: int
    primary_name_fn: Optional[Callable[[torch.Tensor], str]]
    param_to_name: object
    buffers: object
    seen: set

    def param_name(self, param: torch.Tensor) -> str:
        return get_optimizer_param_name(
            param,
            primary_name_fn=self.primary_name_fn,
            param_to_name=self.param_to_name,
            buffers=self.buffers,
        )

    def log(
        self,
        kind: str,
        model_param: torch.nn.Parameter,
        shard_param: Optional[torch.nn.Parameter] = None,
        **extra,
    ) -> None:
        """Emit a one-line structured grad issue log, deduplicated per process."""
        try:
            key = (
                kind,
                id(model_param),
                extra.get("buffer_idx", None),
                extra.get("bucket_idx", None),
                extra.get("bucket_id", None),
            )
            if key in self.seen:
                return
            self.seen.add(key)

            global_rank = dist.get_rank() if dist.is_initialized() else 0
            dp_rank = self.data_parallel_group.rank() if self.data_parallel_group is not None else -1
            dp_size = self.data_parallel_group.size() if self.data_parallel_group is not None else -1

            payload = {
                "kind": kind,
                "step": get_opt_step(self.optimizer),
                "global_rank": global_rank,
                "dp_rank": dp_rank,
                "dp_size": dp_size,
                "fs_rank": self.fs_rank,
                "fs_size": self.fs_size,
                "param": self.param_name(model_param),
                "is_dion": bool(getattr(model_param, "is_dion_param", False)),
                "grad_added_to_main_grad": bool(
                    getattr(model_param, "grad_added_to_main_grad", False)
                ),
                "model_shape": tuple(model_param.shape),
                "model_has_main_grad": bool(getattr(model_param, "main_grad", None) is not None),
                "shard_shape": tuple(shard_param.shape) if shard_param is not None else None,
                "shard_has_main_grad": bool(
                    getattr(shard_param, "main_grad", None) is not None
                )
                if shard_param is not None
                else False,
            }
            payload.update(extra)
            logger.error("[DION_GRAD_ISSUE] %s", json.dumps(payload, sort_keys=True))
        except Exception as error:
            logger.error("[DION_GRAD_ISSUE] logging failed: %s", error)


def log_dion_copy_debug_(
    *,
    model_param: torch.nn.Parameter,
    bucket,
    grad_result,
    bucket_slice: Optional[torch.Tensor],
    name_fn: Callable[[torch.nn.Parameter], str],
    copy_count: int,
) -> int:
    """Emit the existing DEBUG_PERPARAM Dion copy log for the first few params."""
    if not (
        bucket_slice is not None
        and grad_result.used_direct_range
        and copy_count < 3
    ):
        return copy_count

    if not (dist.get_rank() == 0):
        return copy_count

    param_name = name_fn(model_param)
    try:
        prefix = bucket.grad_data[
            grad_result.rs_start : grad_result.rs_start
            + min(10, grad_result.rs_end - grad_result.rs_start)
        ].float().tolist()
    except Exception:
        prefix = None

    logger.info(
        "[DION_COPY] %s: slice_norm=%.6f numel=%d rs=(%d,%d) grad_data_prefix=%s",
        param_name,
        float(bucket_slice.float().norm()),
        int(bucket_slice.numel()),
        int(grad_result.rs_start),
        int(grad_result.rs_end),
        prefix,
    )
    return copy_count + 1
