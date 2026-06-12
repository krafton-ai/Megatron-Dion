"""Factory helpers for distributed MCore Dion2."""

from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist

from megatron.core.optimizer.matrix.topology import get_matrix_replica_group, resolve_fs_group
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from megatron.core.optimizer_param_scheduler import ParamGroupOverride


def get_dion2_param_override(
    config: OptimizerConfig,
    param: torch.nn.Parameter,
    param_override: Optional[ParamGroupOverride],
    param_name: Optional[str] = None,
) -> Optional[ParamGroupOverride]:
    """Return Dion2-specific param group overrides."""
    del config, param, param_override, param_name
    return None


def _resolve_dion2_fs_group(
    *,
    dense_fs_group: Optional[torch.distributed.ProcessGroup],
    pure_data_parallel_group: Optional[torch.distributed.ProcessGroup],
    is_expert_parallel: bool,
    requested_fs_size: int,
    requested_rp_size: int,
) -> Optional[torch.distributed.ProcessGroup]:
    fs_group = resolve_fs_group(
        dense_fs_group=dense_fs_group,
        pure_data_parallel_group=pure_data_parallel_group,
        is_expert_parallel=is_expert_parallel,
        requested_fs_world_size=requested_fs_size,
        requested_rp_world_size=requested_rp_size,
        optimizer_name="Dion2 optimizer",
    )
    if fs_group is not None:
        fs_size = len(dist.get_process_group_ranks(fs_group))
        if not is_expert_parallel and fs_size != int(requested_fs_size):
            raise RuntimeError(
                "Dion2 FS topology mismatch while constructing DistributedDion2Optimizer. "
                f"requested_fs={requested_fs_size} actual_fs_group_size={fs_size} "
                f"(fs_group_ranks={dist.get_process_group_ranks(fs_group)})."
            )
    return fs_group


def build_dion2_optimizer(
    *,
    config,
    param_groups,
    data_parallel_group=None,
    pure_data_parallel_group=None,
    dense_fs_group=None,
    dion2_tp_group=None,
    pg_collection=None,
    is_expert_parallel: bool = False,
):
    """Build the inner MegatronDion2 optimizer used by local and distributed paths."""
    del data_parallel_group, pure_data_parallel_group, dense_fs_group, dion2_tp_group
    del is_expert_parallel
    from megatron.core.optimizer.dion2.algorithm import build_dion2_optimizer as _build

    use_distributed = bool(getattr(config, "use_distributed_optimizer", False))
    if use_distributed and isinstance(param_groups, (list, tuple)):
        groups = []
        for group in param_groups:
            if isinstance(group, dict):
                group = dict(group)
                if "split_parameters" in group:
                    group.setdefault("dion2_split_parameters", bool(group.pop("split_parameters")))
                else:
                    group.setdefault(
                        "dion2_split_parameters",
                        bool(getattr(config, "dion2_split_parameters", True)),
                    )
            groups.append(group)
        param_groups = groups
    return _build(config=config, param_groups=param_groups, pg_collection=pg_collection)


def build_dion2_distributed_optimizer(
    *,
    optimizer_args,
    dense_fs_group=None,
    dion2_tp_group=None,
    **kwargs,
):
    """Build a DistributedDion2Optimizer with MCore distributed-optimizer ownership."""
    from .optimizer import DistributedDion2Optimizer

    config = kwargs.pop("config", None)
    pure_data_parallel_group = kwargs.pop("pure_data_parallel_group", None)
    replica_dp_group = kwargs.pop("replica_dp_group", pure_data_parallel_group)
    pg_collection = kwargs.pop("pg_collection", None)
    is_expert_parallel = bool(kwargs.pop("is_expert_parallel", False))
    requested_fs_size = 1
    requested_rp_size = 1
    if config is not None:
        requested_fs_size = int(getattr(config, "fully_shard_model_parallel_size", 1) or 1)
        requested_rp_size = int(getattr(config, "replicate_model_parallel_size", 1) or 1)
        fs_group = _resolve_dion2_fs_group(
            dense_fs_group=dense_fs_group,
            pure_data_parallel_group=pure_data_parallel_group,
            is_expert_parallel=is_expert_parallel,
            requested_fs_size=requested_fs_size,
            requested_rp_size=requested_rp_size,
        )
        fs_size = 1 if fs_group is None else len(dist.get_process_group_ranks(fs_group))
        replica_group = get_matrix_replica_group(
            pg_collection,
            pure_data_parallel_group,
            requested_fs_size,
            requested_rp_size,
            is_expert_parallel,
            replica_dp_group=replica_dp_group,
            optimizer_name="Dion2 optimizer",
        )
        kwargs.setdefault("fully_shard_model_parallel_size", fs_size)
        kwargs.setdefault("replica_model_parallel_size", requested_rp_size)
        kwargs.setdefault("replica_group", replica_group)
        kwargs.setdefault("dion2_fs_mode", getattr(config, "dion2_fs_mode", "distributed"))
        kwargs.setdefault("dion2_tp_mode", getattr(config, "dion2_tp_mode", "distributed"))
        kwargs.setdefault("dion2_ns_backend", getattr(config, "dion2_ns_backend", "standard"))
    else:
        fs_group = dense_fs_group
    kwargs.setdefault("is_expert_dion2", is_expert_parallel)
    return DistributedDion2Optimizer(
        *optimizer_args,
        dion2_fs_group=fs_group,
        dion2_tp_group=dion2_tp_group,
        pure_data_parallel_group=pure_data_parallel_group,
        **kwargs,
    )


__all__ = [
    "build_dion2_distributed_optimizer",
    "build_dion2_optimizer",
    "get_dion2_param_override",
]
