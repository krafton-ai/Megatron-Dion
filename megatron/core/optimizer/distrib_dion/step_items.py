"""Step routing helpers for distributed Dion execution."""

from __future__ import annotations

from typing import Callable

from ..dion.types import DionStepParam, ScalarStepParam


def route_step_params_(
    *,
    param_groups,
    dist_metas,
    get_step_param_grad_fn: Callable,
    get_or_initialize_optimizer_state_fn: Callable,
    require_param_config_fn: Callable,
    use_distributed_dion_update_fn: Callable,
    sync_q_replicas_fn: Callable,
    build_dion_batches_fn: Callable,
):
    """Route one optimizer step into Dion batches and scalar updates."""
    scalar_params: list[ScalarStepParam] = []
    dion_params: list[DionStepParam] = []

    for optim_group in param_groups:
        for param in optim_group['params']:
            grad = get_step_param_grad_fn(param)
            if grad is None:
                continue

            optimizer_state = get_or_initialize_optimizer_state_fn(param, optim_group)
            dist_meta = dist_metas.get(param, None)
            config = require_param_config_fn(param, dist_meta)

            if use_distributed_dion_update_fn(param, optimizer_state, optim_group, dist_meta):
                dion_params.append(
                    DionStepParam(
                        param=param,
                        grad=grad,
                        optimizer_state=optimizer_state,
                        optim_group=optim_group,
                        config=config,
                        dist_meta=dist_meta,
                    )
                )
                continue

            scalar_params.append(
                ScalarStepParam(
                    param=param,
                    grad=grad,
                    optimizer_state=optimizer_state,
                    optim_group=optim_group,
                )
            )

    dion_batches = []
    if dion_params:
        ordered_dion_params = []
        for step_param in dion_params:
            param = step_param.param
            dist_meta = step_param.dist_meta
            param_uid = getattr(dist_meta, "param_uid", None) if dist_meta is not None else None
            if param_uid is None:
                raise RuntimeError(
                    "[DION_MISSING_PARAM_UID] distributed Dion param is missing param_uid: "
                    f"name={getattr(dist_meta, 'param_name', '') if dist_meta is not None else ''} "
                    f"shape={tuple(param.shape)}"
                )
            ordered_dion_params.append((param_uid, step_param))
        ordered_dion_params.sort(key=lambda entry: entry[0])
        dion_params = [step_param for _, step_param in ordered_dion_params]
        sync_q_replicas_fn(dion_params)
        dion_batches = build_dion_batches_fn(dion_params)

    return dion_batches, scalar_params
