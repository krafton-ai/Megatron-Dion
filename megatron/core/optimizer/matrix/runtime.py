"""Backend-neutral matrix step routing."""

from typing import Callable

from .types import MatrixStepParam, ScalarStepParam


def route_step_params(
    *,
    param_groups,
    dist_metas,
    get_step_param_grad: Callable,
    ensure_optimizer_state: Callable,
    require_param_config: Callable,
    use_matrix: Callable,
    split_children: Callable,
    refresh_state: Callable | None = None,
    sync_state: Callable,
    build_batches: Callable,
):
    """Route one optimizer step into matrix batches and scalar fallback params."""
    scalar_params: list[ScalarStepParam] = []
    matrix_params: list[MatrixStepParam] = []

    for optim_group in param_groups:
        for param in optim_group["params"]:
            grad = get_step_param_grad(param)
            if grad is None:
                continue

            state = ensure_optimizer_state(param, optim_group)
            dist_meta = dist_metas.get(param, None)
            if refresh_state is not None:
                refresh_state(
                    param=param,
                    state=state,
                    optim_group=optim_group,
                    dist_meta=dist_meta,
                )
            config = require_param_config(param, dist_meta)

            children = split_children(
                param=param,
                grad=grad,
                state=state,
                optim_group=optim_group,
                config=config,
                dist_meta=dist_meta,
            )
            if children is not None:
                matrix_params.extend(children)
                continue

            if use_matrix(param=param, state=state, optim_group=optim_group, dist_meta=dist_meta):
                matrix_params.append(
                    MatrixStepParam(
                        param=param,
                        grad=grad,
                        optimizer_state=state,
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
                    optimizer_state=state,
                    optim_group=optim_group,
                )
            )

    batches = []
    if matrix_params:
        ordered = []
        for step_param in matrix_params:
            param = step_param.param
            dist_meta = step_param.dist_meta
            param_uid = getattr(dist_meta, "param_uid", None) if dist_meta is not None else None
            if param_uid is None:
                raise RuntimeError(
                    "[MATRIX_MISSING_PARAM_UID] matrix param is missing param_uid: "
                    f"name={getattr(dist_meta, 'param_name', '') if dist_meta is not None else ''} "
                    f"shape={tuple(param.shape)}"
                )
            ordered.append((param_uid, step_param))
        ordered.sort(key=lambda entry: entry[0])
        matrix_params = [step_param for _, step_param in ordered]
        sync_state(matrix_params)
        batches = build_batches(matrix_params)

    return batches, scalar_params
