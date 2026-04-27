"""Distributed Dion bootstrap helpers."""

from __future__ import annotations

import traceback
from typing import Callable, List

import torch
import torch.distributed as dist

from ... import parallel_state
from ..dion.state import (
    get_global_shape,
    q_seed_from_param_key,
    resolve_q_state_layout,
    should_use_low_rank_sync,
)
from ..dion.utils import get_local_shape
from ..dion.types import DionQInit, DionStepParam, ElementwiseStepParam


def make_group_broadcast(process_group, *, group_size: Callable):
    """Build a no-op or process-group broadcast callback."""
    if process_group is None:
        return lambda tensor: None
    if group_size(process_group) <= 1:
        return lambda tensor: None

    group_ranks = dist.get_process_group_ranks(process_group)
    src_rank = int(group_ranks[0])

    def _broadcast_tensor(tensor):
        dist.broadcast(tensor, src=src_rank, group=process_group)

    return _broadcast_tensor


def resolve_base_training_seed() -> int:
    """Return the topology-invariant base training seed used for Dion Q init."""
    try:
        from megatron.training.global_vars import get_args

        args = get_args()
        return int(args.seed)
    except Exception:
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        return (int(torch.initial_seed()) - 100 * int(pp_rank)) % (2**63 - 1)


def build_q_init(
    *,
    param,
    optim_group,
    dist_meta,
    rank_fraction_default: float,
    rank_multiple_of_default: int,
    use_low_rank_sync: bool,
    base_training_seed: int,
    get_replicate_group: Callable,
    make_group_broadcast: Callable,
) -> DionQInit:
    """Return the adapter-authored Q-init contract for one Dion param."""
    if dist_meta is None:
        raise RuntimeError(
            "[DION_MISSING_STATE_INIT_META] "
            f"rank={dist.get_rank()} param={getattr(param, '_param_name', '')}"
        )

    config = getattr(dist_meta, "param_config", None)
    if config is None:
        raise RuntimeError(
            "[DION_MISSING_STATE_INIT_PARAM_CONFIG] "
            f"rank={dist.get_rank()} param={getattr(dist_meta, 'param_name', '')} "
            f"param_uid={getattr(dist_meta, 'param_uid', None)}"
        )

    physical_local_shape = tuple(int(dim) for dim in getattr(dist_meta, "shape", ()))
    local_shape = get_local_shape(
        dist_meta,
        int(physical_local_shape[0]) if len(physical_local_shape) >= 1 else 0,
        int(physical_local_shape[1]) if len(physical_local_shape) >= 2 else 0,
    )
    if len(local_shape) != 2:
        raise RuntimeError(
            "[DION_INVALID_STATE_INIT_LOCAL_SHAPE] "
            f"rank={dist.get_rank()} param={getattr(dist_meta, 'param_name', '')} "
            f"param_uid={getattr(dist_meta, 'param_uid', None)} shape={local_shape}"
        )
    m_local, n_local = local_shape

    rank_fraction = float(optim_group.get("rank_fraction", rank_fraction_default))
    rank_multiple_of = int(optim_group.get("rank_multiple_of", rank_multiple_of_default))
    tp_world_size = int(getattr(dist_meta, "tp_world_size", 1))
    tp_rank = int(getattr(dist_meta, "tp_rank", 0))
    use_q_unshard = bool(getattr(config, "use_tp_shard", False))

    q_layout = resolve_q_state_layout(
        m_local,
        n_local,
        config,
        tp_world_size=tp_world_size,
        tp_rank=tp_rank,
        use_q_unshard=use_q_unshard,
        global_shape=get_global_shape(dist_meta, m_local, n_local),
        rank_fraction=rank_fraction,
        rank_multiple_of=rank_multiple_of,
    )
    q_seed = q_seed_from_param_key(
        base_seed=base_training_seed,
        dist_meta=dist_meta,
        q_global_shape=tuple(int(dim) for dim in q_layout.q_global_shape),
        is_transposed=config.is_transposed,
    )
    dist_meta.rank_fraction = float(rank_fraction)
    config.use_low_rank_sync = bool(use_low_rank_sync) and should_use_low_rank_sync(
        global_shape=get_global_shape(dist_meta, m_local, n_local),
        r_global=int(q_layout.r_global),
        rank_fraction=float(rank_fraction),
    )
    replicate_group = get_replicate_group()
    return DionQInit(
        tp_world_size=tp_world_size,
        tp_rank=tp_rank,
        use_q_unshard=use_q_unshard,
        q_seed=int(q_seed),
        q_layout=q_layout,
        broadcast_q=make_group_broadcast(replicate_group),
    )


def sync_q_replicas(
    *,
    dion_params: List[DionStepParam],
    state_replica_group,
    group_size: Callable,
    make_group_broadcast: Callable,
) -> None:
    """Synchronize freshly initialized Q across standard DO state replicas."""
    if state_replica_group is None or group_size(state_replica_group) <= 1:
        for step_param in dion_params:
            state = step_param.optimizer_state
            if state is None:
                continue
            if state.get("_needs_state_replica_q_sync", False):
                state["_needs_state_replica_q_sync"] = False
            post_q_sync = step_param.post_q_sync
            if callable(post_q_sync):
                post_q_sync()
        return

    broadcast_q = make_group_broadcast(state_replica_group)
    for step_param in dion_params:
        state = step_param.optimizer_state
        dist_meta = step_param.dist_meta
        if state is None:
            raise RuntimeError(
                "[DION_STATE_REPLICA_Q_SYNC_MISSING_STATE] "
                f"param={getattr(dist_meta, 'param_name', '') if dist_meta is not None else ''}"
            )
        if not state.get("_needs_state_replica_q_sync", False):
            continue
        q_state = state.get("Q", None)
        if q_state is None:
            raise RuntimeError(
                "[DION_STATE_REPLICA_Q_SYNC_MISSING_Q] "
                f"param={getattr(dist_meta, 'param_name', '') if dist_meta is not None else ''}"
            )
        broadcast_q(q_state)
        state["_needs_state_replica_q_sync"] = False
        post_q_sync = step_param.post_q_sync
        if callable(post_q_sync):
            post_q_sync()


def should_use_distributed_dion_update(
    *,
    param,
    state,
    optim_group,
    dist_meta,
    global_rank: int,
) -> bool:
    """Return whether this param should use the distributed Dion update."""
    if dist_meta is None:
        raise RuntimeError(
            "[DION_MISSING_DIST_META_FOR_UPDATE_ROUTING] "
            f"rank={global_rank} "
            f"param_name={getattr(param, '_param_name', '')} param_shape={tuple(param.shape)}"
        )

    is_dion_marked = bool(dist_meta.is_dion_param)
    is_2d_global = bool(dist_meta.global_shape is not None and len(dist_meta.global_shape) == 2)
    if is_dion_marked and not is_2d_global:
        raise RuntimeError(
            "[DION_INVALID_DIST_META_FOR_DION_ROUTING] "
            f"rank={global_rank} "
            f"param_uid={getattr(dist_meta, 'param_uid', None)} "
            f"param_name={getattr(dist_meta, 'param_name', '')} "
            f"global_shape={getattr(dist_meta, 'global_shape', None)}"
        )

    return (
        optim_group.get("algorithm", "dion") == "dion"
        and is_dion_marked
        and is_2d_global
        and "Q" in state
    )


def ensure_optimizer_state(
    *,
    optimizer,
    param,
    optim_group,
    dion_state_param_by_uid,
    dion_dist_meta_by_uid,
    init_optimizer_state,
):
    """Own distributed state remap and metadata recovery at the adapter boundary."""
    dist_meta = optimizer.dist_metas.get(param, None)
    param_uid = getattr(param, "_dion_param_uid", None)
    if param_uid is None and dist_meta is not None:
        param_uid = dist_meta.param_uid

    if param not in optimizer.state:
        if param_uid is not None:
            old_param = dion_state_param_by_uid.get(param_uid)
            if old_param is not None and old_param is not param and old_param in optimizer.state:
                optimizer.state[param] = optimizer.state.pop(old_param)
            else:
                optimizer.state[param] = {}
            dion_state_param_by_uid[param_uid] = param
            if param not in optimizer.dist_metas and param_uid in dion_dist_meta_by_uid:
                optimizer.dist_metas[param] = dion_dist_meta_by_uid[param_uid]
        else:
            optimizer.state[param] = {}

    state = optimizer.state[param]
    if len(state) == 0:
        init_optimizer_state(param, state, optim_group)
    return state


def _collective_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def validate_step_groups(
    *,
    replica_group,
    tp_group,
    rp_group,
    fs_group,
    state_replica_group,
    route_step_params: Callable,
    group_size: Callable,
    group_rank: Callable,
    use_low_rank_sync: bool,
    use_fs_collectives: bool,
) -> Callable:
    """Validate distributed bootstrap inputs and return the step-routing callback."""
    global_rank = dist.get_rank()

    if dist.is_initialized() and replica_group is not None:
        device = _collective_device()
        have_rp_arg = torch.tensor(
            [1 if rp_group is not None else 0],
            device=device,
            dtype=torch.int64,
        )
        have_fs_arg = torch.tensor(
            [1 if fs_group is not None else 0],
            device=device,
            dtype=torch.int64,
        )
        have_state_replica_arg = torch.tensor(
            [1 if state_replica_group is not None else 0],
            device=device,
            dtype=torch.int64,
        )
        dist.all_reduce(have_rp_arg, op=dist.ReduceOp.MIN, group=replica_group)
        dist.all_reduce(have_fs_arg, op=dist.ReduceOp.MIN, group=replica_group)
        dist.all_reduce(have_state_replica_arg, op=dist.ReduceOp.MIN, group=replica_group)

        if int(have_rp_arg.item()) != 1 and rp_group is not None:
            raise RuntimeError(
                f"Global rank {global_rank}: Inconsistent rp_group provision! "
                f"This rank received rp_group, but some replica ranks did not (MIN=0)."
            )
        if int(have_fs_arg.item()) != 1 and fs_group is not None:
            raise RuntimeError(
                f"Global rank {global_rank}: Inconsistent fs_group provision! "
                f"This rank received fs_group, but some replica ranks did not (MIN=0)."
            )
        if int(have_state_replica_arg.item()) != 1 and state_replica_group is not None:
            raise RuntimeError(
                f"Global rank {global_rank}: Inconsistent state_replica_group provision! "
                f"This rank received state_replica_group, but some replica ranks did not (MIN=0)."
            )

    def _validate_group_membership(group, label):
        if group is None:
            return 1, 0
        world_size = group_size(group)
        rank = group_rank(group)
        group_ranks = dist.get_process_group_ranks(group)
        if len(group_ranks) != world_size or dist.get_rank() not in group_ranks:
            raise RuntimeError(
                f"Global rank {global_rank}: invalid {label} membership: "
                f"size={world_size} ranks={group_ranks}"
            )
        return world_size, rank

    fs_world_size, _ = _validate_group_membership(fs_group, "fs_group")
    rp_world_size, _ = _validate_group_membership(rp_group, "rp_group")
    _validate_group_membership(state_replica_group, "state_replica_group")
    _validate_group_membership(tp_group, "tp_group")

    if dist.is_initialized() and replica_group is not None:
        world_size = dist.get_world_size(replica_group)
        if world_size > 1:
            local_config = {
                "use_low_rank_sync": bool(use_low_rank_sync),
                "use_fs_collectives": bool(use_fs_collectives),
                "rp_group_size": rp_world_size if rp_group is not None else 0,
                "fs_group_size": fs_world_size if fs_group is not None else 0,
            }
            del local_config

    return route_step_params


def _rp_param_signature(dist_metas_sharded) -> tuple:
    rows = []
    for dist_meta in dist_metas_sharded.values():
        if not bool(getattr(dist_meta, "is_dion_param", False)):
            continue
        param_uid = getattr(dist_meta, "param_uid", None)
        if param_uid is None:
            raise RuntimeError(
                "[DION_RP_PARAM_SIGNATURE_MISSING_UID] "
                f"rank={dist.get_rank()} param={getattr(dist_meta, 'param_name', '')}"
            )
        config = getattr(dist_meta, "param_config", None)
        rows.append(
            (
                param_uid,
                str(getattr(dist_meta, "param_name", "")),
                tuple(int(dim) for dim in (getattr(dist_meta, "global_shape", None) or ())),
                tuple(
                    int(dim)
                    for dim in (getattr(dist_meta, "per_expert_global_shape", None) or ())
                ),
                tuple(int(dim) for dim in (getattr(dist_meta, "shape", None) or ())),
                tuple(int(dim) for dim in (getattr(dist_meta, "local_shape", None) or ())),
                int(getattr(dist_meta, "fs_shard_dim", -1)),
                int(getattr(dist_meta, "tp_shard_dim", -1)),
                int(getattr(dist_meta, "fs_start_idx", -1)),
                int(getattr(dist_meta, "fs_end_idx", -1)),
                int(getattr(dist_meta, "fs_world_size", 1)),
                int(getattr(dist_meta, "fs_rank", -1)),
                int(getattr(dist_meta, "tp_world_size", 1)),
                int(getattr(dist_meta, "tp_rank", -1)),
                bool(getattr(config, "use_fs_shard", False)) if config is not None else False,
                bool(getattr(config, "use_tp_shard", False)) if config is not None else False,
                bool(getattr(config, "is_transposed", False)) if config is not None else False,
                bool(getattr(config, "use_low_rank_sync", False))
                if config is not None
                else False,
            )
        )
    return tuple(sorted(rows, key=lambda row: repr(row[0])))


def validate_enabled_rp_topology(
    *,
    expected_rp_size: int,
    rp_group,
    data_parallel_group,
    global_rank: int,
    dist_metas_sharded,
    log_error,
) -> None:
    """Fail fast unless RP topology and Dion eligibility are uniform across RP groups."""
    if expected_rp_size <= 1:
        return

    have_rp = torch.tensor(
        [1 if rp_group is not None else 0],
        device=torch.cuda.current_device(),
        dtype=torch.int64,
    )
    dist.all_reduce(have_rp, op=dist.ReduceOp.MIN, group=data_parallel_group)
    all_have_rp = int(have_rp.item()) == 1
    if not all_have_rp:
        raise RuntimeError(
            f"[Dion] Global rank {global_rank}: not all ranks in the CP-excluded "
            f"topology validation group have rp_group (MIN={have_rp.item()}); "
            "RP topology must be identical across the Dion validation domain"
        )
    if rp_group is None:
        raise RuntimeError(
            f"Global rank {global_rank}: all_have_rp=True but rp_group is None! "
            f"This indicates a bug in group creation or collective voting logic."
        )

    my_signature = _rp_param_signature(dist_metas_sharded)
    my_dion_count = len(my_signature)
    my_cnt_tensor = torch.tensor(
        [my_dion_count],
        device=torch.cuda.current_device(),
        dtype=torch.int64,
    )

    rp_world_size = dist.get_world_size(rp_group)
    gathered = [torch.zeros_like(my_cnt_tensor) for _ in range(rp_world_size)]
    dist.all_gather(gathered, my_cnt_tensor, group=rp_group)

    gathered_counts = [int(t.item()) for t in gathered]
    if not all(count == my_dion_count for count in gathered_counts):
        raise RuntimeError(
            f"CRITICAL: Dion eligibility mismatch within RP group! "
            f"My Dion count: {my_dion_count}, RP group counts: {gathered_counts}. "
            f"This will cause collective operation hangs. "
            f"RP group members have different Dion param chunks. "
            f"Consider disabling DistributedOptimizer sharding or implementing FS-aware sharding."
        )

    gathered_signatures = [None for _ in range(rp_world_size)]
    dist.all_gather_object(gathered_signatures, my_signature, group=rp_group)
    if all(signature == my_signature for signature in gathered_signatures):
        return

    rp_group_ranks = tuple(int(rank) for rank in dist.get_process_group_ranks(rp_group))
    first_mismatch = next(
        (
            index
            for index, signature in enumerate(gathered_signatures)
            if signature != my_signature
        ),
        -1,
    )
    mismatch_rank = rp_group_ranks[first_mismatch] if first_mismatch >= 0 else None
    log_error(
        "[Dion] RP param signature mismatch rank=%s mismatch_rank=%s local=%s remote=%s",
        global_rank,
        mismatch_rank,
        my_signature,
        gathered_signatures[first_mismatch] if first_mismatch >= 0 else None,
    )
    raise RuntimeError(
        "[DION_RP_PARAM_SIGNATURE_MISMATCH] "
        f"rank={global_rank} rp_group_ranks={rp_group_ranks} "
        f"local_count={my_dion_count} counts={gathered_counts} "
        f"mismatch_rank={mismatch_rank}"
    )


def enable_distributed_dion(
    *,
    optimizer,
    global_rank: int,
    replica_group,
    data_parallel_group,
    tp_group,
    rp_group,
    fs_group,
    state_replica_group,
    expected_rp_size: int,
    build_all_dist_metas: Callable,
    route_step_params: Callable,
    group_size: Callable,
    group_rank: Callable,
    use_low_rank_sync: bool,
    use_fs_collectives: bool,
    validate_enabled_rp_topology: Callable,
    log_error: Callable,
):
    """Enable distributed Dion mode and return the final distributed metadata."""
    if dist.is_initialized() and dist.get_world_size() == 1:
        return None

    try:
        dist_metas_sharded = build_all_dist_metas()
    except Exception as exc:
        log_error("[Dion] Global rank %s: Failed in _build_all_dist_metas: %s", global_rank, exc)
        log_error(traceback.format_exc())
        raise

    replica_group = replica_group or data_parallel_group
    optimizer.enable_distributed_mode(
        route_step_params=validate_step_groups(
            replica_group=replica_group,
            tp_group=tp_group,
            rp_group=rp_group,
            fs_group=fs_group,
            state_replica_group=state_replica_group,
            route_step_params=route_step_params,
            group_size=group_size,
            group_rank=group_rank,
            use_low_rank_sync=use_low_rank_sync,
            use_fs_collectives=use_fs_collectives,
        ),
    )
    validate_enabled_rp_topology(
        expected_rp_size=expected_rp_size,
        rp_group=rp_group,
        data_parallel_group=data_parallel_group,
        global_rank=global_rank,
        dist_metas_sharded=dist_metas_sharded,
        log_error=log_error,
    )
    return dist_metas_sharded


def route_step_params(
    *,
    param_groups,
    dist_metas,
    get_step_param_grad: Callable,
    ensure_optimizer_state: Callable,
    require_param_config: Callable,
    should_use_distributed_dion_update: Callable,
    expand_split_dion_params: Callable,
    refresh_dion_step_metadata: Callable | None = None,
    sync_q_replicas: Callable,
    build_dion_batches: Callable,
):
    """Route one optimizer step into Dion batches and elementwise updates."""
    elementwise_params: list[ElementwiseStepParam] = []
    dion_params: list[DionStepParam] = []

    for optim_group in param_groups:
        for param in optim_group["params"]:
            grad = get_step_param_grad(param)
            if grad is None:
                continue

            optimizer_state = ensure_optimizer_state(param, optim_group)
            dist_meta = dist_metas.get(param, None)
            if refresh_dion_step_metadata is not None:
                refresh_dion_step_metadata(
                    param=param,
                    optimizer_state=optimizer_state,
                    optim_group=optim_group,
                    dist_meta=dist_meta,
                )
            config = require_param_config(param, dist_meta)

            split_child_params = expand_split_dion_params(
                param=param,
                grad=grad,
                optimizer_state=optimizer_state,
                optim_group=optim_group,
                config=config,
                dist_meta=dist_meta,
            )
            if split_child_params is not None:
                dion_params.extend(split_child_params)
                continue

            if should_use_distributed_dion_update(param, optimizer_state, optim_group, dist_meta):
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

            elementwise_params.append(
                ElementwiseStepParam(
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
        sync_q_replicas(dion_params)
        dion_batches = build_dion_batches(dion_params)

    return dion_batches, elementwise_params
