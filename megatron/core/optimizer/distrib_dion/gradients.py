"""Dion gradient-path helpers for the distributed optimizer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.distributed as dist

from .sharding import compute_fs_shard_range, fs_shard_view_2d


@dataclass
class DionBucketGradSync:
    """Bucket-local Dion grad sync payload kept through reduce-scatter completion."""

    local_grad_shard: torch.Tensor
    local_bucket_grad: torch.Tensor | None = None
    non_dion_grad_shard: torch.Tensor | None = None
    non_dion_local_segments: tuple[tuple[int, int, int], ...] = ()
    group_grad_shards: torch.Tensor | None = None
    group_ranks: tuple[int, ...] | None = None
    group_rank: int | None = None
    reduce_scatter_handle: object | None = None


@dataclass(frozen=True)
class DionBucketGradRoute:
    """Bucket reduce-scatter layout for Dion and mixed non-Dion grads."""

    group_size: int
    group_rank: int
    standard_shard_size: int
    dion_size: int
    non_dion_size: int
    payload_size: int
    non_dion_rank_segments: tuple[tuple[tuple[int, int, int, int], ...], ...]
    non_dion_local_segments: tuple[tuple[int, int, int], ...]


def _build_model_grad(*, bucket) -> torch.Tensor:
    """Build this rank's canonical Dion local grad shard directly from model_param.main_grad."""
    dion_layout = getattr(bucket, "dion_layout", None)
    if dion_layout is None or not dion_layout.has_params:
        raise RuntimeError(
            f"[Dion] missing Dion layout for bucket local grad bucket={getattr(bucket, 'bucket_id', -1)}"
        )

    local_grad_shard = torch.zeros(
        int(dion_layout.shard_size),
        dtype=bucket.grad_data.dtype,
        device=bucket.grad_data.device,
    )

    for entry in dion_layout.entries:
        model_grad = getattr(entry.param, "main_grad", None)
        if model_grad is None:
            raise RuntimeError(
                "[Dion] missing model_param.main_grad while building Dion local grad "
                f"bucket={getattr(bucket, 'bucket_id', -1)} "
                f"param={getattr(entry.param, '_param_name', f'id_{id(entry.param)}')}"
            )
        canonical_local = fs_shard_view_2d(
            model_grad,
            int(entry.fs_shard_dim),
            int(entry.start_idx),
            int(entry.end_idx),
        )
        local_numel = int(entry.local_numel)
        if canonical_local.numel() != local_numel:
            raise RuntimeError(
                "[Dion] canonical local grad size mismatch while building bucket local grad "
                f"bucket={getattr(bucket, 'bucket_id', -1)} "
                f"param={getattr(entry.param, '_param_name', f'id_{id(entry.param)}')} "
                f"source={int(canonical_local.numel())} expected={local_numel}"
            )
        shard_start = int(entry.shard_offset)
        shard_end = shard_start + int(entry.shard_capacity)
        local_grad_shard[shard_start : shard_start + local_numel].copy_(
            canonical_local.reshape(-1)[:local_numel]
        )

    return local_grad_shard


def _build_full_bucket_grad(*, bucket) -> torch.Tensor:
    """Build this rank's Dion local grad shard from a synchronized full bucket grad."""
    dion_layout = getattr(bucket, "dion_layout", None)
    if dion_layout is None or not dion_layout.has_params:
        raise RuntimeError(
            f"[Dion] missing Dion layout for full-bucket grad bucket={getattr(bucket, 'bucket_id', -1)}"
        )

    local_grad_shard = torch.zeros(
        int(dion_layout.shard_size),
        dtype=bucket.grad_data.dtype,
        device=bucket.grad_data.device,
    )

    for entry in dion_layout.entries:
        full_view_2d = bucket.grad_data[
            int(entry.canonical_bucket_start) : int(entry.canonical_bucket_end)
        ].view(tuple(entry.param.shape))
        local_source = fs_shard_view_2d(
            full_view_2d,
            int(entry.fs_shard_dim),
            int(entry.start_idx),
            int(entry.end_idx),
        )
        local_numel = int(local_source.numel())
        if local_numel > int(entry.shard_capacity):
            raise RuntimeError(
                "[Dion] invalid full-bucket Dion grad size "
                f"bucket={getattr(bucket, 'bucket_id', -1)} "
                f"param={getattr(entry.param, '_param_name', f'id_{id(entry.param)}')} "
                f"local_numel={local_numel} shard_capacity={int(entry.shard_capacity)}"
        )
        shard_start = int(entry.shard_offset)
        if local_numel > 0:
            local_grad_shard[shard_start : shard_start + local_numel].copy_(
                local_source.reshape(-1)
            )
        padding_start = shard_start + local_numel
        padding_end = shard_start + int(entry.shard_capacity)
        if padding_end > padding_start:
            local_grad_shard[padding_start:padding_end].zero_()

    return local_grad_shard


def _build_non_dion_rank_segments(
    *,
    bucket,
    group_size: int,
    shard_size: int,
) -> tuple[
    tuple[tuple[tuple[int, int, int, int], ...], ...],
    int,
]:
    """Return compact per-rank source ranges for non-Dion params in a mixed bucket."""
    dion_param_ids = bucket.dion_param_ids
    rank_segments = []
    max_rank_numel = 0
    for group_rank in range(int(group_size)):
        rank_start = int(group_rank) * int(shard_size)
        rank_end = rank_start + int(shard_size)
        cursor = 0
        segments = []
        for param in bucket.params_list:
            if id(param) in dion_param_ids:
                continue
            param_start, param_end = bucket.param_to_index[param]
            source_start = max(int(param_start), rank_start)
            source_end = min(int(param_end), rank_end)
            if source_end <= source_start:
                continue
            target_start = cursor
            local_start = source_start - rank_start
            cursor += source_end - source_start
            segments.append(
                (
                    int(source_start),
                    int(source_end),
                    int(target_start),
                    int(local_start),
                )
            )
        max_rank_numel = max(max_rank_numel, cursor)
        rank_segments.append(tuple(segments))
    return tuple(rank_segments), int(max_rank_numel)


def _get_bucket_grad_route(
    *,
    bucket,
    local_data_view: torch.Tensor | None,
    communication_group,
) -> DionBucketGradRoute:
    dion_layout = getattr(bucket, "dion_layout", None)
    if dion_layout is None or not dion_layout.has_params:
        raise RuntimeError(
            f"[Dion] missing Dion layout for bucket grad sync bucket={getattr(bucket, 'bucket_id', -1)}"
        )

    group_size = 1 if communication_group is None else int(dist.get_world_size(communication_group))
    group_rank = 0 if communication_group is None else int(dist.get_rank(communication_group))
    if group_size <= 0:
        raise RuntimeError(
            "[Dion] invalid Dion grad sync group size "
            f"bucket={getattr(bucket, 'bucket_id', -1)} group_size={group_size}"
        )
    if group_rank < 0 or group_rank >= group_size:
        raise RuntimeError(
            "[Dion] invalid Dion grad sync rank "
            f"bucket={getattr(bucket, 'bucket_id', -1)} rank={group_rank} size={group_size}"
        )

    standard_shard_size = 0
    if local_data_view is not None and local_data_view.numel() > 0:
        standard_shard_size = int(local_data_view.numel())
        expected_bucket_numel = standard_shard_size * group_size
        if expected_bucket_numel != int(bucket.grad_data.numel()):
            raise RuntimeError(
                "[Dion] mixed bucket standard shard size mismatch "
                f"bucket={getattr(bucket, 'bucket_id', -1)} "
                f"bucket_numel={int(bucket.grad_data.numel())} "
                f"shard_size={standard_shard_size} group_size={group_size}"
            )

    cached = getattr(bucket, "_dion_grad_route", None)
    if (
        cached is not None
        and int(cached.group_size) == group_size
        and int(cached.group_rank) == group_rank
        and int(cached.standard_shard_size) == standard_shard_size
    ):
        return cached

    non_dion_rank_segments = tuple(() for _ in range(group_size))
    non_dion_size = 0
    if bucket.has_non_dion_params:
        if local_data_view is None or local_data_view.numel() == 0:
            raise RuntimeError(
                "[Dion] mixed bucket grad sync requires a standard local grad view "
                f"bucket={getattr(bucket, 'bucket_id', -1)}"
            )
        non_dion_rank_segments, non_dion_size = _build_non_dion_rank_segments(
            bucket=bucket,
            group_size=group_size,
            shard_size=standard_shard_size,
        )

    non_dion_local_segments = []
    for source_start, source_end, target_start, local_start in non_dion_rank_segments[group_rank]:
        length = int(source_end) - int(source_start)
        if length <= 0:
            raise RuntimeError(
                "[Dion] invalid non-Dion grad route segment "
                f"bucket={getattr(bucket, 'bucket_id', -1)} target_start={target_start}"
            )
        non_dion_local_segments.append(
            (int(target_start), int(target_start) + int(length), int(local_start))
        )

    dion_size = int(dion_layout.shard_size)
    payload_size = dion_size + int(non_dion_size)
    if payload_size <= 0:
        raise RuntimeError(
            "[Dion] empty bucket grad route "
            f"bucket={getattr(bucket, 'bucket_id', -1)}"
        )

    route = DionBucketGradRoute(
        group_size=group_size,
        group_rank=group_rank,
        standard_shard_size=standard_shard_size,
        dion_size=dion_size,
        non_dion_size=int(non_dion_size),
        payload_size=payload_size,
        non_dion_rank_segments=non_dion_rank_segments,
        non_dion_local_segments=tuple(non_dion_local_segments),
    )
    bucket._dion_grad_route = route
    return route


def _entry_grad_split_range(entry, group_rank: int) -> tuple[int, int]:
    rank_segments = entry.grad_rank_flat_segments[int(group_rank)]
    if not rank_segments:
        return 0, 0

    if int(entry.fs_shard_dim) == 0:
        if len(rank_segments) != 1:
            raise RuntimeError(
                "[Dion] row-sharded Dion grad range must be contiguous "
                f"param={getattr(entry.param, '_param_name', f'id_{id(entry.param)}')} "
                f"group_rank={int(group_rank)} ranges={len(rank_segments)}"
            )
        source_start, source_end = rank_segments[0]
        n = int(entry.param.shape[1])
        start_offset = int(source_start) - int(entry.canonical_bucket_start)
        end_offset = int(source_end) - int(entry.canonical_bucket_start)
        if start_offset % n != 0 or end_offset % n != 0:
            raise RuntimeError(
                "[Dion] row-sharded Dion grad range is not row aligned "
                f"param={getattr(entry.param, '_param_name', f'id_{id(entry.param)}')} "
                f"group_rank={int(group_rank)}"
            )
        return start_offset // n, end_offset // n

    first_start, first_end = rank_segments[0]
    start_idx = int(first_start) - int(entry.canonical_bucket_start)
    end_idx = start_idx + int(first_end) - int(first_start)
    return start_idx, end_idx


def _build_grad_reduce_input(*, bucket, route: DionBucketGradRoute) -> torch.Tensor:
    """Build Dion reduce-scatter input from the MCore bucket grad buffer."""
    dion_layout = getattr(bucket, "dion_layout", None)
    if dion_layout is None or not dion_layout.has_params:
        raise RuntimeError(
            f"[Dion] missing Dion layout for bucket grad sync bucket={getattr(bucket, 'bucket_id', -1)}"
        )

    reduce_input = torch.empty(
        int(route.group_size) * int(route.payload_size),
        dtype=bucket.grad_data.dtype,
        device=bucket.grad_data.device,
    )

    for group_rank in range(int(route.group_size)):
        group_rank_start = int(group_rank) * int(route.payload_size)
        for entry in dion_layout.entries:
            if len(entry.grad_rank_flat_segments) != int(route.group_size):
                raise RuntimeError(
                    "[Dion] grad sync rank mapping size mismatch "
                    f"bucket={getattr(bucket, 'bucket_id', -1)} "
                    f"param={getattr(entry.param, '_param_name', f'id_{id(entry.param)}')} "
                    f"group_size={int(route.group_size)} "
                    f"mapping_size={len(entry.grad_rank_flat_segments)}"
                )
            target_start = group_rank_start + int(entry.shard_offset)
            target_end = target_start + int(entry.shard_capacity)

            full_view_2d = bucket.grad_data[
                int(entry.canonical_bucket_start) : int(entry.canonical_bucket_end)
            ].view(tuple(entry.param.shape))
            rank_start, rank_end = _entry_grad_split_range(entry, group_rank)
            if int(entry.fs_shard_dim) == 0:
                source_2d = full_view_2d[int(rank_start) : int(rank_end), :]
            else:
                source_2d = full_view_2d[:, int(rank_start) : int(rank_end)]
            source_numel = int(source_2d.numel())
            if source_numel > int(entry.shard_capacity):
                raise RuntimeError(
                    "[Dion] Dion grad source exceeds shard capacity "
                    f"bucket={getattr(bucket, 'bucket_id', -1)} "
                    f"param={getattr(entry.param, '_param_name', f'id_{id(entry.param)}')} "
                    f"source_numel={source_numel} "
                    f"shard_capacity={int(entry.shard_capacity)}"
                )
            if source_numel > 0:
                reduce_input[target_start : target_start + source_numel].view(
                    source_2d.shape
                ).copy_(source_2d)
            padding_start = target_start + source_numel
            if target_end > padding_start:
                reduce_input[padding_start:target_end].zero_()

        if route.non_dion_size > 0:
            non_dion_start = group_rank_start + int(route.dion_size)
            non_dion_end = non_dion_start + int(route.non_dion_size)
            reduce_input[non_dion_start:non_dion_end].zero_()
            for source_start, source_end, target_start, _ in route.non_dion_rank_segments[
                group_rank
            ]:
                source_start = int(source_start)
                source_end = int(source_end)
                target_start = non_dion_start + int(target_start)
                target_end = target_start + source_end - source_start
                if target_end > non_dion_end:
                    raise RuntimeError(
                        "[Dion] non-Dion grad route exceeds payload "
                        f"bucket={getattr(bucket, 'bucket_id', -1)} "
                        f"target_end={target_end} payload_end={non_dion_end}"
                    )
                reduce_input[target_start:target_end].copy_(
                    bucket.grad_data[source_start:source_end]
                )

    return reduce_input


def _set_bucket_local_grads(
    *,
    bucket,
    set_local_grad: Callable[[torch.nn.Parameter, torch.Tensor], None],
    grad_sync: DionBucketGradSync | None,
) -> None:
    """Set Dion local grads from the reduced bucket tensor without full-bucket reconstruction."""
    dion_layout = getattr(bucket, "dion_layout", None)
    if dion_layout is None or not dion_layout.has_params:
        return

    if grad_sync is None:
        raise RuntimeError(
            "[Dion] missing Dion bucket grad sync "
            f"for bucket={getattr(bucket, 'bucket_id', -1)}"
        )
    local_grad_shard = grad_sync.local_grad_shard
    expected_shard_size = int(dion_layout.shard_size)
    if local_grad_shard.ndim != 1 or local_grad_shard.numel() != expected_shard_size:
        raise RuntimeError(
            "[Dion] invalid Dion bucket grad sync output "
            f"bucket={getattr(bucket, 'bucket_id', -1)} "
            f"shape={tuple(local_grad_shard.shape)} expected_numel={expected_shard_size}"
        )
    for entry in dion_layout.entries:
        local_numel = int(entry.local_numel)
        shard_capacity = int(entry.shard_capacity)
        if local_numel > shard_capacity:
            raise RuntimeError(
                "[Dion] invalid Dion grad shard metadata "
                f"bucket={getattr(bucket, 'bucket_id', -1)} "
                f"param={getattr(entry.param, '_param_name', id(entry.param))} "
                f"local_numel={local_numel} shard_capacity={shard_capacity}"
            )
        shard_start = int(entry.shard_offset)
        local_shard = local_grad_shard[shard_start : shard_start + local_numel].view(
            entry.local_shape
        )
        set_local_grad(entry.param, local_shard)


def _set_non_dion_local_grads(
    *,
    bucket,
    local_data_view: torch.Tensor | None,
    grad_sync: DionBucketGradSync,
) -> None:
    """Restore mixed-bucket non-Dion reduced grads into the standard MCore view."""
    non_dion_grad_shard = grad_sync.non_dion_grad_shard
    if non_dion_grad_shard is None:
        return
    if local_data_view is None or local_data_view.numel() == 0:
        raise RuntimeError(
            "[Dion] mixed bucket non-Dion grad sync missing standard local view "
            f"bucket={getattr(bucket, 'bucket_id', -1)}"
        )
    for source_start, source_end, local_start in grad_sync.non_dion_local_segments:
        source_start = int(source_start)
        source_end = int(source_end)
        local_start = int(local_start)
        if source_end <= source_start:
            raise RuntimeError(
                "[Dion] invalid mixed bucket non-Dion grad segment "
                f"bucket={getattr(bucket, 'bucket_id', -1)} "
                f"source_start={source_start} source_end={source_end}"
            )
        local_end = local_start + source_end - source_start
        if local_end > int(local_data_view.numel()):
            raise RuntimeError(
                "[Dion] mixed bucket non-Dion grad segment exceeds standard local view "
                f"bucket={getattr(bucket, 'bucket_id', -1)} "
                f"local_end={local_end} view_numel={int(local_data_view.numel())}"
            )
        local_data_view[local_start:local_end].copy_(
            non_dion_grad_shard[source_start:source_end]
        )


def validate_dion_local_shard_grad_(
    *,
    model_param: torch.nn.Parameter,
    shard_param: torch.nn.Parameter,
    shard_view: torch.Tensor | None,
    log_grad_issue: Callable,
) -> torch.Tensor:
    """Validate one adapter-stored Dion local grad shard before optimizer use."""
    if shard_view is None:
        log_grad_issue("DION_LOCAL_GRAD_NONE", model_param, shard_param)
        raise RuntimeError(
            "[Dion] Dion grad set requires stored local grad shard "
            f"param_shape={tuple(model_param.shape)} shard_shape={tuple(shard_param.shape)}"
        )

    if shard_view.ndim != shard_param.ndim:
        raise RuntimeError(
            "[Dion] Dion canonical shard grad ndim mismatch "
            f"shard_view_ndim={int(shard_view.ndim)} shard_param_ndim={int(shard_param.ndim)}"
        )

    if shard_view.nelement() != shard_param.nelement():
        log_grad_issue(
            "DION_MAIN_GRAD_NUMEL_MISMATCH",
            model_param,
            shard_param,
            shard_view_numel=int(shard_view.nelement()),
            shard_param_numel=int(shard_param.nelement()),
        )
        raise RuntimeError(
            "[Dion] Dion canonical shard grad numel mismatch "
            f"shard_view_numel={int(shard_view.nelement())} shard_param_numel={int(shard_param.nelement())}"
        )
    if tuple(shard_view.shape) != tuple(shard_param.shape):
        log_grad_issue(
            "DION_MAIN_GRAD_SHAPE_MISMATCH",
            model_param,
            shard_param,
            shard_view_shape=tuple(shard_view.shape),
            shard_param_shape=tuple(shard_param.shape),
        )
        raise RuntimeError(
            "[Dion] Dion canonical shard grad shape mismatch "
            f"shard_view_shape={tuple(shard_view.shape)} shard_param_shape={tuple(shard_param.shape)}"
        )
    return shard_view


def optimizer_shard_grad_view_(
    *,
    model_param: torch.nn.Parameter,
    shard_main_param: torch.nn.Parameter,
    param_range,
    log_grad_issue: Callable,
) -> torch.Tensor:
    """Return one optimizer shard grad view from canonical `model_param.main_grad`."""
    model_grad = model_param.main_grad
    if model_grad is None:
        log_grad_issue("NON_DION_MODEL_MAIN_GRAD_NONE", model_param, shard_main_param)
        raise RuntimeError(
            "[Dion] optimizer shard grad requires canonical model_param.main_grad "
            f"param_shape={tuple(model_param.shape)} shard_shape={tuple(shard_main_param.shape)}"
        )

    flat_grad = model_grad.view(-1)
    start = int(param_range.start)
    end = int(param_range.end)
    if end > flat_grad.numel():
        log_grad_issue(
            "NON_DION_STOCK_SLICE_OOB",
            model_param,
            shard_main_param,
            grad_numel=int(flat_grad.numel()),
            start=int(start),
            end=int(end),
        )
        raise RuntimeError(
            "[Dion] optimizer shard grad view exceeded canonical main_grad "
            f"grad_numel={int(flat_grad.numel())} start={int(start)} end={int(end)}"
        )
    shard_view = flat_grad[start:end]
    if shard_view.numel() != shard_main_param.nelement():
        log_grad_issue(
            "NON_DION_STOCK_SLICE_SIZE_MISMATCH",
            model_param,
            shard_main_param,
            expected_numel=int(shard_main_param.nelement()),
            got_numel=int(shard_view.numel()),
            start=int(start),
            end=int(end),
        )
        raise RuntimeError(
            "[Dion] optimizer shard grad size mismatch "
            f"expected={int(shard_main_param.nelement())} got={int(shard_view.numel())}"
        )
    return shard_view.view(shard_main_param.shape)


def set_dion_shard_grad_(
    *,
    model_param: torch.nn.Parameter,
    shard_param: torch.nn.Parameter,
    get_local_grad: Callable[[torch.nn.Parameter, torch.nn.Parameter], torch.Tensor],
    log_grad_issue: Callable,
    use_precision_aware_optimizer: bool,
) -> None:
    """Set one Dion optimizer shard grad onto an optimizer shard param."""
    optimizer_shard_grad = validate_dion_local_shard_grad_(
        model_param=model_param,
        shard_param=shard_param,
        shard_view=get_local_grad(model_param, shard_param),
        log_grad_issue=log_grad_issue,
    )

    shard_param.is_dion_param = True
    shard_param.main_grad = None
    if use_precision_aware_optimizer:
        shard_param.decoupled_grad = optimizer_shard_grad
        shard_param.grad = None
    else:
        shard_param.decoupled_grad = None
        shard_param.grad = optimizer_shard_grad.float()


def set_non_dion_shard_grad_(
    *,
    model_param: torch.nn.Parameter,
    shard_param: torch.nn.Parameter,
    get_param_range: Callable[[torch.nn.Parameter], object],
    log_grad_issue: Callable,
    use_precision_aware_optimizer: bool,
) -> None:
    """Set one non-Dion grad shard onto an optimizer shard param."""
    param_range = get_param_range(model_param)["param"]

    model_grad = getattr(model_param, "main_grad", None)
    if model_grad is None:
        raise RuntimeError(
            "[Dion] non-Dion optimizer grad set requires canonical model_param.main_grad "
            f"param={getattr(model_param, '_param_name', f'id_{id(model_param)}')} "
            f"shard_shape={tuple(shard_param.shape)}"
        )

    shard_grad = optimizer_shard_grad_view_(
        model_param=model_param,
        shard_main_param=shard_param,
        param_range=param_range,
        log_grad_issue=log_grad_issue,
    )

    if use_precision_aware_optimizer:
        shard_param.main_grad = None
        shard_param.decoupled_grad = shard_grad
        shard_param.grad = None
        shard_param.is_dion_param = False
        return

    shard_param.main_grad = None
    shard_param.decoupled_grad = None
    shard_param.grad = shard_grad.float()
    shard_param.is_dion_param = False


def set_non_dion_optimizer_shard_grads_(
    *,
    model_groups,
    shard_groups,
    get_param_range: Callable[[torch.nn.Parameter], object],
    log_grad_issue: Callable,
    use_precision_aware_optimizer: bool,
) -> None:
    """Project canonical model-side non-Dion grads onto optimizer shards."""
    for model_group, shard_param_group in zip(model_groups, shard_groups):
        if len(model_group) != len(shard_param_group):
            raise RuntimeError(
                "[Dion] non-Dion grad copy requires equal model/shard group lengths "
                f"model_len={len(model_group)} shard_len={len(shard_param_group)}"
            )
        for model_param, shard_param in zip(model_group, shard_param_group):
            if shard_param is None or getattr(model_param, "is_dion_param", False):
                continue
            set_non_dion_shard_grad_(
                model_param=model_param,
                shard_param=shard_param,
                get_param_range=get_param_range,
                log_grad_issue=log_grad_issue,
                use_precision_aware_optimizer=use_precision_aware_optimizer,
            )


def set_dion_optimizer_shard_grads_(
    *,
    model_groups,
    shard_groups,
    get_local_grad: Callable[[torch.nn.Parameter, torch.nn.Parameter], torch.Tensor],
    log_grad_issue: Callable,
    use_precision_aware_optimizer: bool,
) -> None:
    """Set Dion local shard grads onto optimizer shard params for one or more param groups."""
    for model_group, shard_param_group in zip(model_groups, shard_groups):
        if len(model_group) != len(shard_param_group):
            raise RuntimeError(
                "[Dion] Dion grad copy requires equal model/shard group lengths "
                f"model_len={len(model_group)} shard_len={len(shard_param_group)}"
            )
        for model_param, shard_param in zip(model_group, shard_param_group):
            if not getattr(model_param, "is_dion_param", False):
                continue
            set_dion_shard_grad_(
                model_param=model_param,
                shard_param=shard_param,
                get_local_grad=get_local_grad,
                log_grad_issue=log_grad_issue,
                use_precision_aware_optimizer=use_precision_aware_optimizer,
            )


def clear_dion_local_grads(optimizer, params: list[torch.nn.Parameter] | None = None) -> None:
    """Clear the active adapter-stored Dion local grad surface."""
    if params is None:
        optimizer._dion_local_grads.clear()
        return
    for param in params:
        optimizer._dion_local_grads.pop(param, None)


def scale_dion_local_grads(
    optimizer,
    params: list[torch.nn.Parameter] | None,
    scaling_factor: float,
) -> None:
    """Scale adapter-stored Dion grads by the same factor as MCore grad buffers."""
    if params is None:
        local_grads = tuple(optimizer._dion_local_grads.values())
    else:
        seen_param_ids = set()
        local_grads = []
        for param in params:
            if id(param) in seen_param_ids:
                continue
            seen_param_ids.add(id(param))
            local_grad = optimizer._dion_local_grads.get(param)
            if local_grad is None:
                raise RuntimeError(
                    "[Dion] cannot scale missing Dion local grad "
                    f"param={optimizer._param_name(param) or f'id_{id(param)}'}"
                )
            local_grads.append(local_grad)

    for local_grad in local_grads:
        local_grad.mul_(scaling_factor)


def gather_fs_grad(
    optimizer,
    grad: torch.Tensor,
    *,
    shard_layout,
    fs_group,
) -> torch.Tensor:
    """Rebuild one FS-sharded 2D grad into the same local tensor across FS configs."""
    if (
        shard_layout is None
        or fs_group is None
        or optimizer._group_size(fs_group) <= 1
        or grad.ndim != 2
        or int(shard_layout.fs_shard_dim) not in (0, 1)
    ):
        return grad

    fs_world = optimizer._group_size(fs_group)
    fs_shard_dim = int(shard_layout.fs_shard_dim)
    split_dim_size = int(shard_layout.global_shape[fs_shard_dim])
    rank_ranges = [
        compute_fs_shard_range(split_dim_size, fs_world, rank)
        for rank in range(fs_world)
    ]
    rank_extents = [int(end) - int(start) for start, end in rank_ranges]
    max_extent = max(rank_extents, default=0)
    if fs_shard_dim == 0:
        padded_shape = (max_extent, int(grad.size(1)))
        padded = torch.zeros(padded_shape, dtype=grad.dtype, device=grad.device)
        local_extent = int(grad.size(0))
        if local_extent > 0:
            padded[:local_extent, :].copy_(grad)
        gathered = torch.empty(
            (fs_world, *padded_shape),
            dtype=grad.dtype,
            device=grad.device,
        )
    else:
        padded_shape = (int(grad.size(0)), max_extent)
        padded = torch.zeros(padded_shape, dtype=grad.dtype, device=grad.device)
        local_extent = int(grad.size(1))
        if local_extent > 0:
            padded[:, :local_extent].copy_(grad)
        gathered = torch.empty(
            (fs_world, *padded_shape),
            dtype=grad.dtype,
            device=grad.device,
        )
    dist.all_gather_into_tensor(
        gathered.view(-1),
        padded.contiguous().view(-1),
        group=fs_group,
    )

    if fs_shard_dim == 0:
        full_shape = (split_dim_size, int(grad.size(1)))
    else:
        full_shape = (int(grad.size(0)), split_dim_size)

    full_grad = torch.empty(full_shape, dtype=grad.dtype, device=grad.device)
    for rank, (rank_start, rank_end) in enumerate(rank_ranges):
        rank_extent = int(rank_end) - int(rank_start)
        if rank_extent <= 0:
            continue
        if fs_shard_dim == 0:
            full_grad[int(rank_start) : int(rank_end), :].copy_(
                gathered[rank, :rank_extent, :]
            )
        else:
            full_grad[:, int(rank_start) : int(rank_end)].copy_(
                gathered[rank, :, :rank_extent]
            )
    return full_grad


def set_local_grad(optimizer, model_param: torch.nn.Parameter, local_grad: torch.Tensor) -> None:
    """Store one Dion-local grad shard in stable adapter storage."""
    shard_layout = optimizer._param_shard_layout(model_param)
    if shard_layout is None:
        raise RuntimeError(
            "[Dion] cannot store Dion local grad without shard layout "
            f"param={optimizer._param_name(model_param) or f'id_{id(model_param)}'}"
        )
    expected_shape = tuple(int(dim) for dim in shard_layout.local_shape)
    if tuple(local_grad.shape) != expected_shape:
        raise RuntimeError(
            "[Dion] stored Dion local grad shape mismatch "
            f"param={optimizer._param_name(model_param) or f'id_{id(model_param)}'} "
            f"stored_shape={tuple(local_grad.shape)} expected_shape={expected_shape}"
        )
    use_precision_aware_optimizer = bool(
        getattr(
            getattr(optimizer, "config", None),
            "use_precision_aware_optimizer_no_fp8_or_ds_fp8",
            False,
        )
    )
    stored_source = (
        local_grad
        if use_precision_aware_optimizer or local_grad.dtype == torch.float32
        else local_grad.float()
    )
    stored_local_grad = optimizer._dion_local_grads.get(model_param)
    if (
        stored_local_grad is None
        or tuple(stored_local_grad.shape) != tuple(stored_source.shape)
        or stored_local_grad.dtype != stored_source.dtype
        or stored_local_grad.device != stored_source.device
    ):
        stored_local_grad = torch.empty_like(stored_source)
        optimizer._dion_local_grads[model_param] = stored_local_grad
    stored_local_grad.copy_(stored_source)


def get_local_grad(
    optimizer,
    model_param: torch.nn.Parameter,
    shard_param: torch.nn.Parameter,
) -> torch.Tensor:
    """Return the canonical Dion local grad shard for one optimizer param."""
    local_grad = optimizer._dion_local_grads.get(model_param)
    if local_grad is None:
        raise RuntimeError(
            "[Dion] missing stored Dion local grad shard "
            f"param={optimizer._param_name(model_param) or f'id_{id(model_param)}'} "
            f"shard_shape={tuple(shard_param.shape)}"
        )
    return local_grad


def get_main_grad_local(
    optimizer,
    model_param: torch.nn.Parameter,
    shard_param: torch.nn.Parameter,
) -> torch.Tensor | None:
    """Return the canonical Dion local grad view directly from model_param.main_grad."""
    model_grad = getattr(model_param, "main_grad", None)
    if model_grad is None:
        return None

    shard_layout = optimizer._param_shard_layout(model_param)
    if shard_layout is None:
        raise RuntimeError(
            "[Dion] missing shard layout while reading main_grad "
            f"param={optimizer._param_name(model_param) or f'id_{id(model_param)}'}"
        )

    local_grad = fs_shard_view_2d(
        model_grad,
        int(shard_layout.fs_shard_dim),
        int(shard_layout.start_idx),
        int(shard_layout.end_idx),
    )
    if tuple(local_grad.shape) != tuple(shard_param.shape):
        raise RuntimeError(
            "[Dion] main_grad local shape mismatch "
            f"param={optimizer._param_name(model_param) or f'id_{id(model_param)}'} "
            f"main_grad_shape={tuple(model_grad.shape)} "
            f"local_shape={tuple(local_grad.shape)} shard_shape={tuple(shard_param.shape)}"
        )
    return local_grad


def apply_bucket_grads(
    optimizer,
    *,
    bucket,
    local_data_view: torch.Tensor | None,
    communication_group,
) -> None:
    """Set Dion local grads from the completed Dion bucket grad sync."""
    del communication_group
    dion_layout = getattr(bucket, "dion_layout", None)
    if dion_layout is None or not dion_layout.has_params:
        return
    grad_sync = pop_bucket_grad_sync(bucket)
    if grad_sync is None and bool(getattr(bucket, "_dion_use_full_grad_after_sync", False)):
        grad_sync = DionBucketGradSync(local_grad_shard=_build_full_bucket_grad(bucket=bucket))
    bucket._dion_use_full_grad_after_sync = False
    if grad_sync is not None:
        _set_non_dion_local_grads(
            bucket=bucket,
            local_data_view=local_data_view,
            grad_sync=grad_sync,
        )
    _set_bucket_local_grads(
        bucket=bucket,
        set_local_grad=optimizer._set_local_grad,
        grad_sync=grad_sync,
    )


def stash_bucket_grad_sync(optimizer, bucket, grad_sync: DionBucketGradSync) -> None:
    """Keep one Dion grad sync payload alive until local grads are set."""
    del optimizer
    bucket._dion_grad_sync = grad_sync


def wait_bucket_grad_sync(bucket) -> DionBucketGradSync | None:
    """Ensure one bucket's Dion RS payload is ready before any consumer reads it."""
    grad_sync = getattr(bucket, "_dion_grad_sync", None)
    if grad_sync is None:
        return None
    reduce_scatter_handle = getattr(grad_sync, "reduce_scatter_handle", None)
    if reduce_scatter_handle is not None:
        try:
            wait = reduce_scatter_handle.wait
            wait()
        except AttributeError:
            pass
        except ValueError as exc:
            # PyTorch coalescing_manager returns _IllegalWork for per-collective
            # handles; the owning coalescing manager is waited by the bucket group.
            if "IllegalWork" not in str(exc):
                raise
        grad_sync.reduce_scatter_handle = None
    return grad_sync


def pop_bucket_grad_sync(bucket) -> DionBucketGradSync | None:
    """Return and clear one bucket's pending Dion grad sync."""
    grad_sync = wait_bucket_grad_sync(bucket)
    bucket._dion_grad_sync = None
    return grad_sync


def get_dion_bucket_inter_instance_grad_buffer(bucket) -> torch.Tensor | None:
    """Return None because Dion runtime performs replica-domain grad/P/R sync."""
    del bucket
    return None


def get_dion_bucket_inter_instance_combined_grad_buffer(bucket) -> torch.Tensor | None:
    """Return None so mixed buckets do not full-sync Dion regions."""
    del bucket
    return None


def get_dion_bucket_inter_instance_non_dion_grad_buffer(bucket) -> torch.Tensor | None:
    """Return mixed-bucket non-Dion grads that need inter-instance reduction."""
    grad_sync = getattr(bucket, "_dion_grad_sync", None)
    if grad_sync is None:
        return None
    return grad_sync.non_dion_grad_shard


def clear_bucket_grad_sync(bucket) -> None:
    """Drop any stale Dion grad-sync payload cached on a bucket."""
    bucket._dion_grad_sync = None
    bucket._dion_use_full_grad_after_sync = False


def start_dion_bucket_grad_sync(
    optimizer,
    *,
    bucket,
    local_data_view: torch.Tensor | None,
    communication_group,
    reduce_op,
    async_op: bool,
    reduce_scatter,
):
    """Launch stock and Dion bucket grad sync from the MCore bucket grad buffer."""
    clear_bucket_grad_sync(bucket)
    route = _get_bucket_grad_route(
        bucket=bucket,
        local_data_view=local_data_view,
        communication_group=communication_group,
    )
    local_bucket_grad = torch.empty(
        int(route.payload_size),
        dtype=bucket.grad_data.dtype,
        device=bucket.grad_data.device,
    )
    local_grad_shard = local_bucket_grad[: int(route.dion_size)]
    non_dion_grad_shard = None
    if route.non_dion_size > 0:
        non_dion_grad_shard = local_bucket_grad[
            int(route.dion_size) : int(route.payload_size)
        ]
    reduce_input = _build_grad_reduce_input(bucket=bucket, route=route)
    dion_handle = reduce_scatter(
        local_bucket_grad,
        reduce_input,
        op=reduce_op,
        group=communication_group,
        async_op=async_op,
    )
    stash_bucket_grad_sync(
        optimizer,
        bucket,
        DionBucketGradSync(
            local_grad_shard=local_grad_shard,
            local_bucket_grad=local_bucket_grad,
            non_dion_grad_shard=non_dion_grad_shard,
            non_dion_local_segments=route.non_dion_local_segments,
            group_grad_shards=reduce_input,
            reduce_scatter_handle=dion_handle,
        ),
    )
    return dion_handle


def build_grad_route_(
    *,
    use_precision_aware_optimizer: bool,
    model_float16_groups,
    shard_float16_groups,
    model_fp32_groups,
    shard_fp32_groups,
    shard_fp32_from_float16_groups,
    log_grad_issue: Callable,
) -> dict:
    """Build the model-surface to optimizer-shard grad route."""
    if use_precision_aware_optimizer:
        grad_group_pairs = (
            (model_float16_groups, shard_float16_groups),
            (model_fp32_groups, shard_fp32_groups),
        )
    else:
        grad_group_pairs = (
            (model_float16_groups, shard_fp32_from_float16_groups),
            (model_fp32_groups, shard_fp32_groups),
        )
    return {
        "use_precision_aware_optimizer": use_precision_aware_optimizer,
        "grad_group_pairs": grad_group_pairs,
        "log_grad_issue": log_grad_issue,
    }


def set_optimizer_shard_grads(
    *,
    is_stub_optimizer: bool,
    use_megatron_fsdp: bool,
    use_precision_aware_optimizer: bool,
    model_float16_groups,
    shard_float16_groups,
    model_fp32_groups,
    shard_fp32_groups,
    shard_fp32_from_float16_groups,
    get_param_range: Callable[[torch.nn.Parameter], object],
    get_local_grad: Callable[[torch.nn.Parameter, torch.nn.Parameter], torch.Tensor],
    log_grad_issue: Callable,
    release_rs_buffers: Callable,
) -> None:
    """Set model-side grad surfaces onto optimizer shards and release RS buffers."""
    if is_stub_optimizer or use_megatron_fsdp:
        return

    grad_route = build_grad_route_(
        use_precision_aware_optimizer=use_precision_aware_optimizer,
        model_float16_groups=model_float16_groups,
        shard_float16_groups=shard_float16_groups,
        model_fp32_groups=model_fp32_groups,
        shard_fp32_groups=shard_fp32_groups,
        shard_fp32_from_float16_groups=shard_fp32_from_float16_groups,
        log_grad_issue=log_grad_issue,
    )
    use_precision_aware_optimizer = grad_route["use_precision_aware_optimizer"]
    log_grad_issue = grad_route["log_grad_issue"]
    grad_group_pairs = grad_route["grad_group_pairs"]

    for model_groups, shard_groups in grad_group_pairs:
        set_non_dion_optimizer_shard_grads_(
            model_groups=model_groups,
            shard_groups=shard_groups,
            get_param_range=get_param_range,
            log_grad_issue=log_grad_issue,
            use_precision_aware_optimizer=use_precision_aware_optimizer,
        )
        set_dion_optimizer_shard_grads_(
            model_groups=model_groups,
            shard_groups=shard_groups,
            get_local_grad=get_local_grad,
            log_grad_issue=log_grad_issue,
            use_precision_aware_optimizer=use_precision_aware_optimizer,
        )

    release_rs_buffers()


def wait_for_pending_grad_reduce_handles_(model_chunks) -> None:
    """Wait any pending grad-reduce handles on regular and expert bucket groups."""
    for model_chunk in model_chunks or []:
        for attr_name in ("bucket_groups", "expert_parallel_bucket_groups"):
            bucket_groups = getattr(model_chunk, attr_name, None)
            if not bucket_groups:
                continue
            for bucket_group in bucket_groups:
                grad_reduce_handle = getattr(bucket_group, "grad_reduce_handle", None)
                if grad_reduce_handle is None:
                    continue
                grad_reduce_handle.wait()
                bucket_group.grad_reduce_handle = None


def finish_bucket_group_grad_sync(per_model_bucket_groups) -> None:
    """Finish grad sync on all registered bucket groups."""
    for bucket_groups in per_model_bucket_groups.values():
        for bucket_group in bucket_groups:
            finish_grad_sync = getattr(bucket_group, "finish_grad_sync", None)
            if finish_grad_sync is not None:
                finish_grad_sync()


def release_rs_buffers(buffers) -> None:
    """Release cached RS buffers after async grad sync completes."""
    for buffer in buffers or []:
        for bucket in getattr(buffer, "buckets", []):
            if hasattr(bucket, "dion_state"):
                bucket.dion_state = None
