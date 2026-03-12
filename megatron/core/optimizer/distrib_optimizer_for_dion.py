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
import torch
import torch.distributed as dist
from typing import Dict, List, Tuple, Optional

from .distrib_optimizer import DistributedOptimizer, Range
from .megatron_dion import MegatronDion, MegatronDionDistMeta
from .distrib_dion.bucket_order import reorder_bucket_params_
from .distrib_dion.bucket_plan import (
    DionParamLayout,
    build_dion_bucket_layout_,
    build_stock_bucket_non_dion_plan_,
    check_bucket_dion_flags_,
    check_local_dion_bucket_,
    check_mixed_bucket_ranges_,
    check_shared_dion_layout_,
    select_non_dion_bucket_params_,
)
from .distrib_dion.checkpoint_io import (
    apply_stock_non_dion_shards_,
    apply_group_shards_to_model_params_,
    build_named_dion_param_state,
    copy_group_params_to_main_shards_,
    restore_group_params_,
    restore_named_dion_param_state_,
)
from .distrib_dion.fs_all_gather import (
    pack_fs_shards_,
    unpack_fs_shards_,
)
from .distrib_dion.grad_wiring import (
    copy_stock_non_dion_grads_,
    copy_grad_groups_,
    fix_zero_dion_grads_,
)
from .distrib_dion.grad_diag import GradIssueLogger
from .distrib_dion.grad_norm import (
    append_precomputed_norm_grads_,
    build_grad_norm_entries_,
    clip_precomputed_grad_groups_,
    debug_validate_precomputed_norm_grads_,
    get_optimizer_grad_,
    log_grad_norm_contributors_,
    log_zero_global_grad_norm_,
)
from .distrib_dion.fs_layout import (
    compute_fs_shard_range,
    compute_local_shape,
    get_fs_split_dim,
    slice_fs_shard_2d,
    write_fs_shard_2d_,
)
from .distrib_dion.overlap_sync import (
    finish_bucket_group_grad_sync_,
    release_rs_buffers_,
)
from .distrib_dion.param_selection import is_dion_param, is_moe_expert_param
from .distrib_dion.param_update import check_shard_identity_
from .distrib_dion.param_utils import get_tp_split_dim, is_tp_enabled
from .distrib_dion.shard_info import DionShardInfo
from .. import parallel_state, tensor_parallel
from ..fp8_utils import is_float8tensor

logger = logging.getLogger(__name__)


def _slice_expected_dion_shard_from_main_grad(model_param, opt_grad) -> Optional[torch.Tensor]:
    """Derive the expected Dion logical local shard directly from canonical model_param.main_grad."""
    dion_info = getattr(model_param, "dion_info", None)
    model_grad = getattr(model_param, "main_grad", None)
    if dion_info is None or model_grad is None or model_grad.ndim != 2:
        return None

    start_idx = dion_info.get("start_idx")
    end_idx = dion_info.get("end_idx")
    fs_split_dim = dion_info.get("fs_split_dim")
    if start_idx is None or end_idx is None or fs_split_dim not in (0, 1):
        return None

    if fs_split_dim == 0:
        expected = model_grad[start_idx:end_idx, :]
    else:
        expected = model_grad[:, start_idx:end_idx]

    if opt_grad is not None and tuple(expected.shape) != tuple(opt_grad.shape):
        return None
    return expected


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

    def _maybe_log_target_grad_fingerprints(self) -> None:
        """Env-gated focused grad fingerprint for one or more target params.

        This is for proving exact contract mismatches, not for normal logging.
        It logs:
        - model-side canonical grad (`model_param.main_grad`) gathered over `dp_cp_group`
        - optimizer-side canonical grad (`shard_param.grad/decoupled_grad`) gathered over
          `grad_stats_parallel_group`
        - both tensors gathered over `state_replica_group` when available
        """
        targets_raw = os.getenv("DION_PARAM_GRAD_FP_NAMES", "").strip()
        if not targets_raw:
            return

        try:
            every_n = int(os.getenv("DION_PARAM_GRAD_FP_EVERY", "1"))
        except ValueError:
            every_n = 1
        try:
            max_logs = int(os.getenv("DION_PARAM_GRAD_FP_MAX_LOGS", "8"))
        except ValueError:
            max_logs = 8

        call_idx = int(getattr(self, "_dion_param_grad_fp_call_idx", 0)) + 1
        self._dion_param_grad_fp_call_idx = call_idx
        if every_n > 1 and call_idx % every_n != 0:
            return
        if int(getattr(self, "_dion_param_grad_fp_logged", 0)) >= max_logs:
            return

        targets = [token.strip() for token in targets_raw.split(",") if token.strip()]
        if not targets:
            return

        model_groups, shard_groups, _, _ = self._grad_norm_maps()
        dp_cp_group = getattr(self, "dp_cp_group", None)
        grad_group = self.get_grad_stats_parallel_group()
        state_group = getattr(self, "state_replica_group", None)

        def _fingerprint(tensor: Optional[torch.Tensor]):
            if tensor is None:
                return None
            data = tensor.detach().float()
            flat = data.view(-1)
            return {
                "shape": tuple(data.shape),
                "norm": float(data.norm().item()),
                "sum": float(data.sum().item()),
                "amax": float(data.abs().max().item()) if flat.numel() > 0 else 0.0,
                "sample": flat[: min(8, flat.numel())].cpu().tolist(),
                "ptr": int(data.data_ptr()),
            }

        for model_group, shard_group in zip(model_groups, shard_groups):
            for model_param, shard_param in zip(model_group, shard_group):
                if shard_param is None:
                    continue
                param_name = self._find_param_name(model_param) or getattr(
                    model_param, "_param_name", ""
                )
                if not param_name:
                    continue
                if not any(target in param_name for target in targets):
                    continue

                model_grad = getattr(model_param, "main_grad", None)
                opt_grad = get_optimizer_grad_(shard_param)
                model_fp = _fingerprint(model_grad)
                opt_fp = _fingerprint(opt_grad)
                pack_pre_inter_fp = None
                pack_expected_fp = None
                pack_diff = None
                pack_group_ranks = None
                if getattr(model_param, "is_dion_param", False):
                    dion_info = self._param_dion_info(model_param)
                    buffer_idx = dion_info.get("buffer_idx")
                    bucket_idx = dion_info.get("bucket_idx")
                    if buffer_idx is not None and bucket_idx is not None:
                        bucket = self.buffers[buffer_idx].buckets[bucket_idx]
                        pack_debug_inputs = getattr(bucket, "dion_pack_debug_inputs", None) or {}
                        local_pack_info = pack_debug_inputs.get(id(model_param))
                        pack_group = getattr(bucket, "dion_pack_debug_group", None)
                        pack_is_avg = bool(
                            getattr(bucket, "dion_pack_debug_is_avg", False)
                        )
                        if local_pack_info is not None:
                            if pack_group is not None and dist.get_world_size(pack_group) > 1:
                                gathered_pack_inputs = [None] * dist.get_world_size(pack_group)
                                dist.all_gather_object(
                                    gathered_pack_inputs, local_pack_info, group=pack_group
                                )
                                pack_group_ranks = dist.get_process_group_ranks(pack_group)
                            else:
                                gathered_pack_inputs = [local_pack_info]
                                pack_group_ranks = None

                            pack_target_rank = (
                                dist.get_rank(group=pack_group)
                                if pack_group is not None and dist.get_world_size(pack_group) > 1
                                else 0
                            )
                            pack_tensors = []
                            for item in gathered_pack_inputs:
                                if item is None:
                                    continue
                                all_scaled = item.get("all_scaled", None)
                                if all_scaled is None:
                                    continue
                                if not (0 <= pack_target_rank < len(all_scaled)):
                                    continue
                                pack_tensors.append(all_scaled[pack_target_rank].float())
                            if pack_tensors:
                                pack_expected = pack_tensors[0].clone()
                                for item in pack_tensors[1:]:
                                    pack_expected.add_(item)
                                if pack_is_avg:
                                    pack_expected.div_(len(pack_tensors))
                                pack_pre_inter_fp = _fingerprint(pack_expected)
                                final_expected = pack_expected
                                if (
                                    state_group is not None
                                    and dist.get_world_size(state_group) > 1
                                ):
                                    gathered_expected = [None] * dist.get_world_size(state_group)
                                    dist.all_gather_object(
                                        gathered_expected,
                                        pack_expected.cpu(),
                                        group=state_group,
                                    )
                                    state_tensors = [
                                        item.float() for item in gathered_expected if item is not None
                                    ]
                                    if state_tensors:
                                        final_expected = state_tensors[0].clone()
                                        for item in state_tensors[1:]:
                                            final_expected.add_(item)
                                        if pack_is_avg:
                                            final_expected.div_(len(state_tensors))
                                pack_expected_fp = _fingerprint(final_expected)
                                if opt_grad is not None and tuple(final_expected.shape) == tuple(opt_grad.shape):
                                    diff = (final_expected - opt_grad.detach().float().cpu()).abs()
                                    pack_diff = {
                                        "amax": float(diff.max().item()) if diff.numel() > 0 else 0.0,
                                        "sum": float(diff.sum().item()),
                                        "sq_sum": float((diff ** 2).sum().item()),
                                    }

                if dp_cp_group is not None and dist.get_world_size(dp_cp_group) > 1:
                    gathered_model = [None] * dist.get_world_size(dp_cp_group)
                    dist.all_gather_object(gathered_model, model_fp, group=dp_cp_group)
                else:
                    gathered_model = [model_fp]

                if grad_group is not None and dist.get_world_size(grad_group) > 1:
                    gathered_opt = [None] * dist.get_world_size(grad_group)
                    dist.all_gather_object(gathered_opt, opt_fp, group=grad_group)
                else:
                    gathered_opt = [opt_fp]

                if state_group is not None and dist.get_world_size(state_group) > 1:
                    gathered_model_state = [None] * dist.get_world_size(state_group)
                    gathered_opt_state = [None] * dist.get_world_size(state_group)
                    dist.all_gather_object(gathered_model_state, model_fp, group=state_group)
                    dist.all_gather_object(gathered_opt_state, opt_fp, group=state_group)
                else:
                    gathered_model_state = [model_fp]
                    gathered_opt_state = [opt_fp]

                logger.info(
                    "[DION_PARAM_GRAD_FP] step=%s rank=%s param=%s state_group=%s pack_group=%s model_gather=%s opt_gather=%s pack_pre_inter=%s pack_expected=%s pack_diff=%s model_state=%s opt_state=%s",
                    call_idx,
                    self._global_rank,
                    param_name,
                    dist.get_process_group_ranks(state_group) if state_group is not None else None,
                    pack_group_ranks,
                    gathered_model,
                    gathered_opt,
                    pack_pre_inter_fp,
                    pack_expected_fp,
                    pack_diff,
                    gathered_model_state,
                    gathered_opt_state,
                )

        self._dion_param_grad_fp_logged = int(
            getattr(self, "_dion_param_grad_fp_logged", 0)
        ) + 1

    @classmethod
    def _resolve_fs_rank(cls, dp_group, dp_rank: int) -> Tuple[int, int]:
        """Resolve FS shard rank from the actual fs_group captured before parent init."""
        fs_config = cls._current_init_fs_config or {}
        fs_size = fs_config.get('fs_size', 1)
        fs_group = fs_config.get('fs_group', None)

        def _group_size(pg):
            return pg.size() if hasattr(pg, "size") else dist.get_world_size(pg)

        def _group_rank(pg):
            return pg.rank() if hasattr(pg, "rank") else dist.get_rank(pg)

        if fs_group is not None:
            fs_size_pg = _group_size(fs_group)
            fs_rank = _group_rank(fs_group)
            if fs_size_pg != fs_size:
                logger.warning(
                    f"[Dion] FS size mismatch between init config and optimizer.fs_group: "
                    f"config_fs_size={fs_size} fs_group.size={fs_size_pg}. "
                    f"Using fs_group.size for layout computation."
                )
            return fs_size_pg, fs_rank

        if fs_size > 1:
            raise RuntimeError(
                "Dion requires a concrete fs_group when fs_size > 1. "
                "Refusing to infer fs_rank from dp_rank because that is not robust "
                "for multi-node or CP>1 topologies."
            )

        return fs_size, 0

    def _stash_fs_init(self, opt) -> None:
        """Stash init-time FS topology for `_build_model_gbuf_range()`."""
        fs_group = getattr(opt, "fs_group", None) if opt is not None else None
        DistributedOptimizerForDion._current_init_fs_config = {
            'fs_size': self._fs_size if self._fs_size else 1,
            'rp_size': self._rp_size if self._rp_size else 1,
            'fs_group': fs_group,
        }

    def _clear_fs_init(self) -> None:
        """Clear the class-level init FS topology stash."""
        DistributedOptimizerForDion._current_init_fs_config = None

    @classmethod
    def _build_bucket_param_map(
        cls,
        parent_result,
        ordered_params,
        dp_group,
        dp_rank,
        bucket_index,
        param_index_map,
        bucket_offset: int,
        bucket_size: int,
    ):
        """Rebuild the parent DO param_map in canonical bucket-param order."""
        from collections import OrderedDict
        from .distrib_optimizer import DistributedOptimizer

        parent_param_map = parent_result["param_map"]
        dp_world_size = dp_group.size()
        if bucket_size % dp_world_size != 0:
            raise RuntimeError(
                f"[Dion] bucket_size must be divisible by dp_size for canonical param_map "
                f"(bucket={bucket_index}, bucket_size={bucket_size}, dp_size={dp_world_size})"
            )

        max_gbuf_range_size = bucket_size // dp_world_size
        gbuf_world_start = dp_rank * max_gbuf_range_size
        gbuf_world_end = min(bucket_size, gbuf_world_start + max_gbuf_range_size)
        gbuf_world_range = Range(
            gbuf_world_start + bucket_offset,
            gbuf_world_end + bucket_offset,
        )
        reconstructed_param_map = DistributedOptimizer._build_model_gbuf_param_range_map(
            param_index_map,
            gbuf_world_range,
            bucket_offset,
        )

        parent_param_summary = {
            "bucket_idx": bucket_index,
            "dp_rank": dp_rank,
            "param_count": len(parent_param_map),
        }
        all_param_summaries = [None] * dp_group.size()
        dist.all_gather_object(all_param_summaries, parent_param_summary, group=dp_group)
        param_counts = [summary["param_count"] for summary in all_param_summaries]
        if len(set(param_counts)) > 1 and bucket_index == 0 and dp_rank == 0:
            logger.info(
                f"[Dion] Bucket {bucket_index}: param_map sizes differ under optim_grads_params; "
                "canonical runtime order follows bucket.params, and missing local entries are "
                "represented as zero-length standard-DO shard ranges."
            )

        canonical_param_map = OrderedDict()
        for param in ordered_params:
            canonical_param_map[param] = (
                parent_param_map.get(param)
                or reconstructed_param_map.get(
                    param,
                    {
                        "param": Range(0, 0),
                        "gbuf_world": Range(0, 0),
                        "gbuf_local": Range(0, 0),
                        "gbuf_world_in_bucket": Range(0, 0),
                    },
                )
            )

        parent_result["param_map"] = canonical_param_map
        return canonical_param_map

    @classmethod
    def _mark_bucket_dion_params(cls, param_map, param_to_name, fs_size):
        """Classify bucket params and attach Dion metadata once."""
        from ..parallel_state import get_tensor_model_parallel_world_size

        dion_param_count = 0

        for param in param_map.keys():
            param_name = None
            if param_to_name is not None and param in param_to_name:
                param_name = param_to_name[param]
            if param_name:
                param._param_name = param_name

            param.is_dion_param = is_dion_param(param, param_name)

            is_expert = is_moe_expert_param(param, param_name)
            tp_split_dim = get_tp_split_dim(param)
            has_tp = is_tp_enabled(param)
            if is_expert and has_tp:
                from megatron.core import parallel_state

                tp_world_size = parallel_state.get_expert_tensor_parallel_world_size()
            else:
                tp_world_size = get_tensor_model_parallel_world_size() if has_tp else 1

            if not param.is_dion_param:
                param_map[param]["dion_info"] = {"is_dion": False}
                continue

            dion_param_count += 1
            m, n = param.shape
            fs_split_dim = get_fs_split_dim(tp_split_dim)

            if tp_split_dim == 0:
                global_m = m * tp_world_size
                global_n = n
            elif tp_split_dim == 1:
                global_m = m
                global_n = n * tp_world_size
            else:
                global_m = m
                global_n = n

            num_local_experts = getattr(param, "num_local_experts", None)
            if num_local_experts is not None and num_local_experts > 1:
                if tp_split_dim == 0:
                    per_expert_global_shape = (global_m // num_local_experts, global_n)
                elif tp_split_dim == 1:
                    per_expert_global_shape = (global_m, global_n // num_local_experts)
                else:
                    per_expert_global_shape = (global_m, global_n)
            else:
                per_expert_global_shape = None

            param_map[param]["dion_info"] = {
                "is_dion": True,
                "global_shape": (global_m, global_n),
                "fs_split_dim": fs_split_dim,
                "tp_split_dim": tp_split_dim,
                "per_expert_global_shape": per_expert_global_shape,
            }

            if dist.is_initialized() and dist.get_rank() == 0:
                logger.info(
                    f"[PARAM] DION=True, name={param_name}, local=({m},{n}), tp_global=({global_m},{global_n}), "
                    f"fs_split_dim={fs_split_dim}, fs_size={fs_size}, tp_split_dim={tp_split_dim}, "
                    f"tp_world_size={tp_world_size}, is_expert={is_expert}"
                )

        return dion_param_count

    def _resolve_fs_group(self) -> None:
        """Resolve runtime FS group/size/rank after parent init."""
        use_optimizer_fs_group = (
            hasattr(self, 'optimizer') and
            hasattr(self.optimizer, 'fs_group') and
            self.optimizer.fs_group is not None
        )

        if use_optimizer_fs_group:
            self.fs_group = self.optimizer.fs_group
            self.fs_size = self.fs_group.size()
            self.fs_rank = self.fs_group.rank()
            if hasattr(self, '_fs_size') and self._fs_size is not None and self._fs_size != self.fs_size:
                logger.warning(
                    f"[Dion] Global rank {self._global_rank}: "
                    f"FS size mismatch! configured={self._fs_size}, optimizer.fs_group.size()={self.fs_size}"
                )
        else:
            self.fs_group = self.data_parallel_group
            self.fs_size = self.data_parallel_group.size()
            self.fs_rank = self.data_parallel_group.rank()

    def _resolve_rp_group(self) -> None:
        """Resolve runtime RP group/size/rank after parent init."""
        configured_rp_size = int(self._rp_size) if self._rp_size is not None else 1
        optimizer_rp_group = (
            getattr(self.optimizer, 'rp_group', None)
            if hasattr(self, 'optimizer')
            else None
        )

        if configured_rp_size <= 1:
            self.rp_group = None
            return

        if optimizer_rp_group is None:
            raise RuntimeError(
                "[Dion] RP>1 requires optimizer.rp_group to be initialized "
                f"(configured_rp_size={configured_rp_size})"
            )

        rp_size = dist.get_world_size(optimizer_rp_group)
        if rp_size != configured_rp_size:
            group_ranks = dist.get_process_group_ranks(optimizer_rp_group)
            raise RuntimeError(
                "[Dion] RP topology mismatch after parent init: "
                f"configured_rp_size={configured_rp_size} actual_rp_group_size={rp_size} "
                f"rp_group_ranks={group_ranks}"
            )

        self.rp_group = optimizer_rp_group

    def _resolve_state_replica_group(self) -> None:
        """Resolve the stock partial-DO optimizer-state replica group.

        CP is not a Dion compressed replicate mesh, but with multiple distributed
        optimizer instances the same local shard exists in multiple optimizer
        instances. Those replicas must keep optimizer state and task ordering
        identical.
        """
        from megatron.core import parallel_state

        try:
            state_group = parallel_state.get_inter_distributed_optimizer_instance_group(
                check_initialized=False
            )
        except Exception:
            state_group = None
        self.state_replica_group = state_group

    @staticmethod
    def _param_name(name_map, param) -> Optional[str]:
        """Best-effort name lookup for maps keyed by either param or id(param)."""
        if not name_map:
            return None
        try:
            name = name_map.get(id(param))
            if name is not None:
                return name
        except Exception:
            pass
        try:
            return name_map.get(param)
        except Exception:
            return None

    def _find_param_name(self, param) -> Optional[str]:
        """Slow-path runtime param-name lookup for current module objects."""
        if hasattr(self, "model_chunks"):
            for model in self.model_chunks:
                try:
                    for name, model_param in model.named_parameters():
                        if id(model_param) == id(param):
                            return name
                except Exception:
                    continue
        elif hasattr(self, "module") and isinstance(self.module, torch.nn.Module):
            for name, model_param in self.module.named_parameters():
                if id(model_param) == id(param):
                    return name
        return None

    def _build_name_map(self) -> Dict[int, str]:
        """Build id(param) -> name mapping for the current runtime module params."""
        runtime_name_map: Dict[int, str] = {}
        if hasattr(self, "model_chunks"):
            for model_idx, model in enumerate(self.model_chunks):
                try:
                    for name, param in model.named_parameters():
                        runtime_name_map[id(param)] = name
                except Exception as error:
                    logger.warning(
                        f"[Dion] model_chunks[{model_idx}] named_parameters() failed: {error}"
                    )
        elif hasattr(self, "module") and isinstance(self.module, torch.nn.Module):
            for name, param in self.module.named_parameters():
                runtime_name_map[id(param)] = name
        return runtime_name_map

    @staticmethod
    def _bucket_param_view(bucket, param) -> Optional[torch.Tensor]:
        """Return the canonical full-param view for one bucket param."""
        if bucket is None or getattr(bucket, "param_data", None) is None:
            return None
        if not hasattr(bucket, "param_to_index") or param not in bucket.param_to_index:
            return None
        start, end = bucket.param_to_index[param]
        return bucket.param_data.view(-1)[start:end].view(param.data.shape)

    def _bind_bucket_param_views(self, bucket, *, copy_data: bool) -> None:
        """Ensure each param in `bucket` aliases the bucket's canonical param buffer."""
        if bucket is None or getattr(bucket, "param_data", None) is None:
            return
        for param in bucket.params:
            expected_view = self._bucket_param_view(bucket, param)
            if expected_view is None:
                continue
            if (
                param.data.shape == expected_view.shape
                and param.data.data_ptr() == expected_view.data_ptr()
            ):
                continue
            if copy_data and param.data.numel() == expected_view.numel():
                expected_view.copy_(param.data.view(expected_view.shape))
            param.data = expected_view

    def _check_bucket_param_views(self, bucket, *, context: str) -> None:
        """Verify forward-visible param views still alias the canonical bucket buffer."""
        if bucket is None or not getattr(bucket, "_dion_requires_param_sync_check", False):
            return
        if getattr(bucket, "param_data", None) is None:
            raise RuntimeError(
                f"[Dion] {context}: bucket {getattr(bucket, 'bucket_id', -1)} missing bucket.param_data"
            )
        for param in bucket.params:
            expected_view = self._bucket_param_view(bucket, param)
            if expected_view is None:
                continue
            if (
                param.data.shape != expected_view.shape
                or param.data.data_ptr() != expected_view.data_ptr()
            ):
                param_name = self._param_name(getattr(self, "param_to_name", None), param)
                if param_name is None:
                    param_name = self._find_param_name(param) or f"id_{id(param)}"
                logger.error(
                    "[DION_PARAM_BUFFER_VIEW_MISMATCH] context=%s bucket_id=%s param=%s param_ptr=%s bucket_ptr=%s",
                    context,
                    getattr(bucket, "bucket_id", -1),
                    param_name,
                    param.data.data_ptr(),
                    expected_view.data_ptr(),
                )
                raise RuntimeError(
                    f"[Dion] {context}: param.data no longer aliases bucket.param_data for {param_name}"
                )

    def _mark_buckets_full_param_ready(self, ready: bool) -> None:
        """Track whether full param views are ready for forward."""
        if not hasattr(self, "buffers"):
            return
        for buffer in self.buffers:
            for bucket in buffer.buckets:
                if hasattr(bucket, "_dion_full_param_ready"):
                    bucket._dion_full_param_ready = ready

    def _build_fs_bucket_plans(self, gbuf_idx, buffer, gbuf_range_maps) -> None:
        """Populate buffer-level bucket plan metadata from `gbuf_ranges`."""
        buffer.dion_param_layouts_by_bucket = []
        buffer.fs_bucket_plan_map = {}
        cumulative_bucket_id = 0

        for dtype, bucket_list in gbuf_range_maps.items():
            for gbuf_range_map in bucket_list:
                global_bucket_id = cumulative_bucket_id
                dion_shard_range = gbuf_range_map.get("dion_param_shard_range", None)
                buffer.fs_bucket_plan_map[global_bucket_id] = {
                    "dtype": dtype,
                    "local_total": gbuf_range_map.get("local_total", 0),
                    "dion_param_layout": gbuf_range_map.get("dion_param_layout", []),
                    "dion_param_shard_range": dion_shard_range,
                }
                if gbuf_range_map.get("dion_param_layout"):
                    buffer.dion_param_layouts_by_bucket.append(
                        {
                            "bucket_idx": global_bucket_id,
                            "dtype": dtype,
                            "dion_param_layout": gbuf_range_map["dion_param_layout"],
                        }
                    )
                cumulative_bucket_id += 1

        expected_bucket_ids = {bucket.bucket_id for bucket in buffer.buckets}
        if set(buffer.fs_bucket_plan_map.keys()) != expected_bucket_ids:
            raise RuntimeError(
                f"[DDP/OPT BUCKET MISMATCH] buffer={gbuf_idx} expected bucket_ids={sorted(expected_bucket_ids)} "
                f"but gbuf_ranges provided {sorted(buffer.fs_bucket_plan_map.keys())}"
            )

        buffer.dion_param_layout = DionParamLayout()
        for bucket_info in buffer.dion_param_layouts_by_bucket:
            buffer.dion_param_layout.extend(bucket_info["dion_param_layout"])

    def _index_buffer_params(self, gbuf_idx, buffer) -> None:
        """Refresh param_index_map after the Dion shard/non-shard layout rewrite."""
        for bucket in buffer.buckets:
            bucket_plan = buffer.fs_bucket_plan_map.get(bucket.bucket_id)
            if bucket_plan is None:
                continue

            dtype_key = bucket_plan["dtype"]
            gbuf_range_map = self.gbuf_ranges[gbuf_idx][dtype_key][bucket.bucket_id]
            param_map = gbuf_range_map.get("param_map", {})
            for param, range_info in param_map.items():
                local_range = range_info["param"]
                global_start = bucket.offset + local_range.start
                global_end = bucket.offset + local_range.end
                buffer.param_index_map[param] = (global_start, global_end, bucket.bucket_id)

    def _bucket_range_map(self, gbuf_idx, buffer, bucket):
        """Resolve the `gbuf_range_map` matching one runtime bucket."""
        bucket_plan = buffer.fs_bucket_plan_map.get(bucket.bucket_id, None)
        dtype_key = (buffer.param_dtype, buffer.grad_dtype)
        if bucket_plan is None:
            raise RuntimeError(
                f"[DDP/OPT BUCKET MISMATCH] buffer={gbuf_idx} missing plan for bucket_id={bucket.bucket_id}"
            )
        if bucket_plan["dtype"] != dtype_key:
            raise RuntimeError(
                f"[DDP/OPT BUCKET MISMATCH] buffer={gbuf_idx} bucket={bucket.bucket_id} dtype mismatch: "
                f"DDP bucket dtype_key={dtype_key} vs plan dtype_key={bucket_plan['dtype']}"
            )
        return bucket_plan, self.gbuf_ranges[gbuf_idx][bucket_plan["dtype"]][bucket.bucket_id]

    def _fs_positions(self, fs_group, fs_size: int):
        """Build full dp_cp-rank -> FS-shard-position mapping used by bucket packing."""
        dp_cp_group = self._original_dp_group if self._original_dp_group is not None else self.data_parallel_group
        dp_cp_size = dp_cp_group.size()
        if dp_cp_size > fs_size:
            my_fs_rank_val = fs_group.rank()
            all_fs_ranks = [None] * dp_cp_size
            dist.all_gather_object(all_fs_ranks, my_fs_rank_val, group=dp_cp_group)
            return all_fs_ranks
        return list(range(fs_size))

    def _init_bucket_comm(self, bucket, fs_group, fs_size: int) -> None:
        """Attach common FS/DP/CP communication metadata to a bucket."""
        dp_cp_group = self._original_dp_group if self._original_dp_group is not None else self.data_parallel_group
        bucket.dion_comm_group = fs_group
        bucket.dion_dp_cp_group = dp_cp_group
        bucket.dp_cp_size = dp_cp_group.size()
        bucket.dp_cp_to_fs_position = self._fs_positions(fs_group, fs_size)

        from megatron.core import parallel_state as ps

        cp_size = ps.get_context_parallel_world_size()
        expert_flags = [not getattr(p, "allreduce", True) for p in bucket.params]
        is_expert_bucket = any(expert_flags)
        if is_expert_bucket and not all(expert_flags):
            raise RuntimeError(
                f"[Dion][EP] mixed dense/expert bucket is invalid for Dion EP: "
                f"bucket_id={bucket.bucket_id} param_count={len(bucket.params)}"
            )
        if is_expert_bucket:
            expert_group = self._expected_expert_shard_group()
            self._assert_group_matches(
                label=f"expert bucket {bucket.bucket_id}",
                actual_group=fs_group,
                expected_group=expert_group,
                extra="Megatron-Core EP local-shard ownership must stay on intra_expt_dp_group.",
            )
        bucket.cp_group = ps.get_context_parallel_group() if (cp_size > 1 and not is_expert_bucket) else None

    def _build_bucket_name_map(
        self,
        *,
        bucket,
        bucket_dion_param_layout,
        active_name_map,
        stale_name_map,
        gbuf_idx: int,
    ) -> Dict[str, torch.nn.Parameter]:
        """Resolve current runtime bucket params by canonical name."""
        bucket_param_by_name: Dict[str, torch.nn.Parameter] = {}

        for param in bucket.params:
            param_name = self._param_name(active_name_map, param)
            if param_name and param_name not in bucket_param_by_name:
                bucket_param_by_name[param_name] = param

        for param in bucket.params:
            if any(id(param) == id(found) for found in bucket_param_by_name.values()):
                continue
            param_name = self._param_name(stale_name_map, param)
            if param_name and param_name not in bucket_param_by_name:
                bucket_param_by_name[param_name] = param

        for param in bucket.params:
            if any(id(param) == id(found) for found in bucket_param_by_name.values()):
                continue
            found_name = self._find_param_name(param)
            if found_name and found_name not in bucket_param_by_name:
                bucket_param_by_name[found_name] = param

        for param in bucket.params:
            if any(id(param) == id(found) for found in bucket_param_by_name.values()):
                continue
            candidates = [
                entry for entry in bucket_dion_param_layout if tuple(param.shape) == tuple(entry["param"].shape)
            ]
            if len(candidates) == 1:
                entry = candidates[0]
                param_name = (
                    self._param_name(active_name_map, entry["param"])
                    or self._param_name(stale_name_map, entry["param"])
                )
                if param_name and param_name not in bucket_param_by_name:
                    bucket_param_by_name[param_name] = param
            elif len(candidates) > 1:
                logger.warning(
                    f"[Dion] buffer={gbuf_idx} bucket={bucket.bucket_id} "
                    f"param shape={tuple(param.shape)} has multiple dion_param_layout matches; skipping fallback"
                )

        return bucket_param_by_name

    def _rebind_bucket_layout(
        self,
        *,
        gbuf_idx: int,
        buffer,
        bucket,
        bucket_dion_param_layout,
        stale_name_map,
        active_name_map,
    ) -> None:
        """Rebind persisted Dion layout entries onto current runtime param objects."""
        if not active_name_map:
            raise RuntimeError(
                f"[Dion] buffer={gbuf_idx} bucket={bucket.bucket_id} "
                f"no active name map (model_chunks/module missing or empty)"
            )

        bucket_param_by_name = self._build_bucket_name_map(
            bucket=bucket,
            bucket_dion_param_layout=bucket_dion_param_layout,
            active_name_map=active_name_map,
            stale_name_map=stale_name_map,
            gbuf_idx=gbuf_idx,
        )

        rebuilt_entries = []
        rebuilt_dion_param_shard_range = {}
        bucket.fs_param_id_to_full_offset = {}
        offset = 0

        for entry in bucket_dion_param_layout:
            param_name = (
                self._param_name(active_name_map, entry["param"])
                or self._param_name(stale_name_map, entry["param"])
            )
            target_param = bucket_param_by_name.get(param_name, None)
            if target_param is None:
                raise RuntimeError(
                    f"[Dion] buffer={gbuf_idx} bucket={bucket.bucket_id} "
                    f"name={param_name} shape={entry['param'].shape} not found in current bucket params"
                )

            new_entry = entry.copy()
            new_entry["param"] = target_param
            rebuilt_entries.append(new_entry)

            seg_size = entry["segment_size"]
            bucket.fs_param_id_to_full_offset[id(target_param)] = offset

            pack_offset = entry.get("pack_offset", 0)
            local_size = entry["local_shape"][0] * entry["local_shape"][1]
            rebuilt_dion_param_shard_range[target_param] = (pack_offset, pack_offset + local_size)
            offset += seg_size

        bucket.dion_param_layout = rebuilt_entries
        bucket.dion_param_shard_range = rebuilt_dion_param_shard_range
        bucket.dion_shard_size = sum(entry["segment_size"] for entry in rebuilt_entries)

        dion_param_name_to_entry = {}
        for entry in rebuilt_entries:
            entry_param = entry.get("param")
            if entry_param is None:
                continue
            param_name = None
            if hasattr(buffer, "param_to_name"):
                param_name = self._param_name(buffer.param_to_name, entry_param)
            if param_name is None:
                param_name = (
                    self._param_name(active_name_map, entry_param)
                    or self._param_name(stale_name_map, entry_param)
                )
            if param_name:
                dion_param_name_to_entry[param_name] = entry
        bucket.dion_param_name_to_entry = dion_param_name_to_entry

    def _init_dion_bucket(
        self,
        *,
        gbuf_idx: int,
        buffer,
        bucket,
        bucket_dion_param_layout,
        dion_param_shard_range_map,
        fs_group,
        fs_size: int,
        stale_name_map,
        active_name_map,
    ) -> None:
        """Configure one bucket that contains at least one Dion param."""
        bucket.dion_param_layout = bucket_dion_param_layout
        if dion_param_shard_range_map:
            bucket.dion_param_shard_range = dion_param_shard_range_map
        bucket.dion_grad_buffer = None
        bucket.dion_grad_local_view = None
        bucket.dion_rs_local_buffer = None

        self._rebind_bucket_layout(
            gbuf_idx=gbuf_idx,
            buffer=buffer,
            bucket=bucket,
            bucket_dion_param_layout=bucket_dion_param_layout,
            stale_name_map=stale_name_map,
            active_name_map=active_name_map,
        )
        self._init_bucket_comm(bucket, fs_group, fs_size)

        bucket.fs_full_grad_total = bucket.dion_shard_size * fs_size
        bucket.fs_pack_total = max(
            entry["pack_offset"] + entry["segment_size"] for entry in bucket_dion_param_layout
        )
        bucket.fs_pack_buffer = None
        bucket.fs_gathered_buffer = None
        bucket.fs_all_gather_fn = (
            lambda async_op=False, b=buffer, bkt=bucket, plan=bucket.dion_param_layout:
            self._all_gather_params_bucket(b, bkt, plan, async_op=async_op)
        )
        bucket.dion_optimizer = self
        bucket._dion_requires_param_sync_check = True
        bucket._dion_full_param_ready = True
        self._bind_bucket_param_views(bucket, copy_data=False)

    def _init_non_dion_bucket(self, *, gbuf_idx: int, buffer, bucket, fs_group, fs_size: int) -> None:
        """Configure one bucket that has no Dion layout entries."""
        has_dion = any(getattr(param, "is_dion_param", False) for param in bucket.params)
        if has_dion:
            name_map = getattr(self, "param_to_name", {}) or getattr(buffer, "param_to_name", {})
            dion_params = []
            for param in bucket.params:
                if getattr(param, "is_dion_param", False):
                    param_name = self._param_name(name_map, param)
                    dion_params.append((id(param), param_name or f"id_{id(param)}", tuple(param.shape)))
            logger.error(
                f"[Dion] Buffer {gbuf_idx} Bucket {bucket.bucket_id} has Dion params but no dion_param_layout. params={dion_params}"
            )
            raise RuntimeError(
                f"[Dion] Buffer {gbuf_idx} Bucket {bucket.bucket_id} has Dion params but no dion_param_layout"
            )

        bucket.dion_param_layout = DionParamLayout()
        bucket.dion_grad_buffer = None
        bucket.dion_grad_local_view = None
        bucket.dion_rs_local_buffer = None
        bucket.fs_full_grad_total = 0
        bucket.fs_param_id_to_full_offset = {}
        bucket.dion_shard_size = 0
        self._init_bucket_comm(bucket, fs_group, fs_size)
        bucket.fs_pack_total = 0
        bucket.fs_pack_buffer = None
        bucket.fs_gathered_buffer = None
        bucket.fs_all_gather_fn = None
        bucket.is_pure_non_dion_bucket = True
        bucket.dion_optimizer = self
        bucket._dion_requires_param_sync_check = False
        bucket._dion_full_param_ready = True

    def _init_mixed_non_dion(
        self,
        *,
        gbuf_idx: int,
        buffer,
        bucket,
        active_name_map,
        stale_name_map,
    ) -> None:
        """Configure the non-Dion section for one mixed bucket."""
        has_dion = bool(getattr(bucket, "dion_param_layout", None))
        if has_dion:
            _, gbuf_range_map = self._bucket_range_map(gbuf_idx, buffer, bucket)
            param_map = gbuf_range_map.get("param_map", {})
        else:
            param_map = None

        non_dion_params_in_bucket = select_non_dion_bucket_params_(
            bucket_params=bucket.params,
            dion_param_layout=bucket.dion_param_layout if has_dion else None,
            param_map=param_map,
        )

        if has_dion and non_dion_params_in_bucket:
            # Match stock Megatron-Core DO: _ParamAndGradBuffer.data_parallel_group is the
            # full DPxCP group used for canonical bucket ownership, while the actual stock
            # local optimizer shard contract is defined by the bucket group's intra
            # distributed-optimizer-instance group.
            dp_group = getattr(bucket, "intra_distributed_optimizer_instance_group", None)
            if dp_group is None:
                raise RuntimeError(
                    "[Dion] mixed non-Dion path requires "
                    "bucket.intra_distributed_optimizer_instance_group to match "
                    "stock Megatron-Core DO local-shard semantics."
                )
            (
                non_dion_pack_plan,
                non_dion_param_ranges,
                non_dion_shard_size,
                non_dion_full_grad_total,
            ) = build_stock_bucket_non_dion_plan_(
                bucket=bucket,
                params=non_dion_params_in_bucket,
                dp_size=dp_group.size(),
                dp_rank=dp_group.rank(),
            )

            bucket.non_dion_pack_plan = non_dion_pack_plan
            bucket.non_dion_param_ranges = non_dion_param_ranges
            bucket.non_dion_pack_total = non_dion_shard_size
            bucket.non_dion_shard_size = non_dion_shard_size
            bucket.non_dion_full_grad_total = non_dion_full_grad_total

            bucket.non_dion_dp_group = dp_group
            bucket.non_dion_all_gather_fn = (
                lambda async_op=False, b=buffer, bkt=bucket: self._all_gather_non_dion(bkt, b, async_op)
            )
            bucket.dion_optimizer = self
            bucket._dion_requires_param_sync_check = True
            bucket._dion_full_param_ready = True

            non_dion_param_name_to_entry = {}
            non_dion_param_id_to_entry = {}
            for entry in non_dion_pack_plan:
                entry_param = entry.get("param")
                if entry_param is None:
                    continue
                non_dion_param_id_to_entry[id(entry_param)] = entry
                param_name = None
                if hasattr(buffer, "param_to_name"):
                    param_name = self._param_name(buffer.param_to_name, entry_param)
                if param_name is None:
                    param_name = (
                        self._param_name(active_name_map, entry_param)
                        or self._param_name(stale_name_map, entry_param)
                    )
                if param_name:
                    non_dion_param_name_to_entry[param_name] = entry
            bucket.non_dion_param_id_to_entry = non_dion_param_id_to_entry
            bucket.non_dion_param_name_to_entry = non_dion_param_name_to_entry
            mixed_dion_full_bucket_ranges = []
            if hasattr(bucket, "param_to_index") and bucket.param_to_index is not None:
                for dion_entry in bucket.dion_param_layout:
                    dion_param = dion_entry.get("param")
                    if dion_param is None or dion_param not in bucket.param_to_index:
                        continue
                    start, end = bucket.param_to_index[dion_param]
                    mixed_dion_full_bucket_ranges.append((int(start), int(end)))
            bucket.mixed_dion_full_bucket_ranges = tuple(mixed_dion_full_bucket_ranges)

            if dist.is_initialized() and dp_group is not None and dp_group.size() > 1:
                def _entry_name(_entry):
                    _param = _entry["param"]
                    _name = self._param_name(buffer.param_to_name, _param)
                    return _name if _name is not None else f"id_{id(_param)}"

                # Mixed non-Dion stock plan contains both:
                # - rank-invariant metadata (`full_*`, `rank_*_ranges`)
                # - rank-local views (`local_*`, `local_bucket_*`)
                # Only the rank-invariant part should match across DP ranks.
                local_signature = (
                    int(bucket.bucket_id),
                    int(bucket.non_dion_pack_total),
                    int(bucket.non_dion_full_grad_total),
                    tuple(
                        (
                            _entry_name(entry),
                            int(entry["full_start"]),
                            int(entry["full_end"]),
                            tuple(entry.get("rank_bucket_ranges", ())),
                            tuple(entry.get("rank_param_ranges", ())),
                        )
                        for entry in non_dion_pack_plan
                    ),
                )
                for entry in non_dion_pack_plan:
                    rank_bucket_ranges = entry.get("rank_bucket_ranges", ())
                    rank_param_ranges = entry.get("rank_param_ranges", ())
                    if len(rank_bucket_ranges) != dp_group.size() or len(rank_param_ranges) != dp_group.size():
                        raise RuntimeError(
                            "[Dion] mixed non-Dion stock plan has invalid per-rank range metadata: "
                            f"bucket={bucket.bucket_id} param={_entry_name(entry)} "
                            f"bucket_ranges={len(rank_bucket_ranges)} param_ranges={len(rank_param_ranges)} "
                            f"dp_size={dp_group.size()}"
                        )
                    expected_bucket_start, expected_bucket_end = rank_bucket_ranges[dp_group.rank()]
                    expected_param_start, expected_param_end = rank_param_ranges[dp_group.rank()]
                    if (
                        int(entry["local_bucket_start"]) != int(expected_bucket_start)
                        or int(entry["local_bucket_end"]) != int(expected_bucket_end)
                        or int(entry["local_start"]) != int(expected_param_start)
                        or int(entry["local_end"]) != int(expected_param_end)
                    ):
                        raise RuntimeError(
                            "[Dion] mixed non-Dion stock plan local-range mismatch: "
                            f"bucket={bucket.bucket_id} param={_entry_name(entry)} "
                            f"local_bucket=({int(entry['local_bucket_start'])},{int(entry['local_bucket_end'])}) "
                            f"expected_bucket=({int(expected_bucket_start)},{int(expected_bucket_end)}) "
                            f"local_param=({int(entry['local_start'])},{int(entry['local_end'])}) "
                            f"expected_param=({int(expected_param_start)},{int(expected_param_end)})"
                        )
                gathered_signatures = [None] * dp_group.size()
                dist.all_gather_object(gathered_signatures, local_signature, group=dp_group)
                if any(sig != local_signature for sig in gathered_signatures):
                    raise RuntimeError(
                        "[Dion] mixed non-Dion pack plan differs across DP ranks: "
                        f"bucket={bucket.bucket_id} local={local_signature} gathered={gathered_signatures}"
                    )

            for entry in non_dion_pack_plan:
                param = entry["param"]
                local_start = int(entry.get("local_start", 0))
                local_end = int(entry.get("local_end", local_start))
                expected_numel = max(0, local_end - local_start)
                model_grad = getattr(param, "main_grad", None)
                if model_grad is not None and model_grad.numel() < local_end:
                    logger.error(
                        "[NON_DION_MAIN_GRAD_RANGE_MISMATCH] Bucket %s param %s local_range=(%s,%s) "
                        "entry_numel=%s main_grad_numel=%s",
                        bucket.bucket_id,
                        tuple(param.shape),
                        local_start,
                        local_end,
                        expected_numel,
                        model_grad.numel(),
                    )
        else:
            bucket.non_dion_pack_plan = []
            bucket.non_dion_full_grad_total = 0
            bucket.non_dion_shard_size = 0
            bucket.non_dion_pack_total = 0
            bucket.non_dion_all_gather_fn = None
            bucket.non_dion_dp_group = None
            bucket.non_dion_param_id_to_entry = {}
            bucket.non_dion_param_name_to_entry = {}
            bucket.mixed_dion_full_bucket_ranges = ()

    def _init_dion_buffers(self) -> None:
        """Configure buffer-level Dion bucket plans after parent optimizer init."""
        if not hasattr(self, "gbuf_ranges") or not hasattr(self, "buffers"):
            return

        fs_group, fs_size = self.fs_group, self.fs_size
        runtime_name_map = self._build_name_map()

        for gbuf_idx, gbuf_range_maps in enumerate(self.gbuf_ranges):
            buffer = self.buffers[gbuf_idx]
            stale_name_map = getattr(self, "param_to_name", {}) or getattr(buffer, "param_to_name", {})
            active_name_map = runtime_name_map or stale_name_map or {}

            self._build_fs_bucket_plans(gbuf_idx, buffer, gbuf_range_maps)
            self._index_buffer_params(gbuf_idx, buffer)

            for bucket in buffer.buckets:
                bucket_plan, _ = self._bucket_range_map(gbuf_idx, buffer, bucket)
                bucket_dion_param_layout = bucket_plan.get("dion_param_layout", None)
                dion_param_shard_range_map = bucket_plan.get("dion_param_shard_range", None)

                if bucket_dion_param_layout:
                    self._init_dion_bucket(
                        gbuf_idx=gbuf_idx,
                        buffer=buffer,
                        bucket=bucket,
                        bucket_dion_param_layout=bucket_dion_param_layout,
                        dion_param_shard_range_map=dion_param_shard_range_map,
                        fs_group=fs_group,
                        fs_size=fs_size,
                        stale_name_map=stale_name_map,
                        active_name_map=active_name_map,
                    )
                else:
                    self._init_non_dion_bucket(
                        gbuf_idx=gbuf_idx,
                        buffer=buffer,
                        bucket=bucket,
                        fs_group=fs_group,
                        fs_size=fs_size,
                    )

            for bucket in buffer.buckets:
                self._init_mixed_non_dion(
                    gbuf_idx=gbuf_idx,
                    buffer=buffer,
                    bucket=bucket,
                    active_name_map=active_name_map,
                    stale_name_map=stale_name_map,
                )

    def _log_unmapped_main_grads(self, model_groups, label: str) -> None:
        """Log params that are truly missing from Dion bookkeeping.

        Pure non-Dion buckets are handled by the parent DO path and should not be
        flagged as uncategorized.
        """
        missing_gbuf_map = 0
        missing_param_map = 0
        uncategorized = 0
        for group in model_groups:
            for model_param in group:
                if model_param not in self.model_param_gbuf_map:
                    missing_gbuf_map += 1
                    continue

                gbuf_idx, dtype, bucket_idx = self.model_param_gbuf_map[model_param]
                buffer = self.buffers[gbuf_idx]
                gbuf_range_map = self.gbuf_ranges[gbuf_idx][dtype][bucket_idx]
                if model_param not in gbuf_range_map["param_map"]:
                    missing_param_map += 1
                    continue

                bucket = buffer.buckets[bucket_idx]
                if getattr(bucket, "is_pure_non_dion_bucket", False) and model_param in bucket.params:
                    continue
                is_dion_param = hasattr(bucket, "dion_param_shard_range") and model_param in bucket.dion_param_shard_range
                is_non_dion_param = hasattr(bucket, "non_dion_param_ranges") and model_param in bucket.non_dion_param_ranges
                if not (is_dion_param or is_non_dion_param):
                    if uncategorized < 5:
                        logger.warning(f"[Dion] Uncategorized param {model_param.shape}, skipping rebind")
                    uncategorized += 1

        if missing_gbuf_map > 0:
            logger.warning(
                f"[Dion] {missing_gbuf_map} params in {label} are missing from model_param_gbuf_map"
            )
        if missing_param_map > 0:
            logger.warning(
                f"[Dion] {missing_param_map} params in {label} are missing from bucket param_map"
            )
        if uncategorized > 0:
            logger.warning(
                f"[Dion] {uncategorized} params in {label} are not categorized by Dion bucket bookkeeping"
            )

    def _rebind_dion_shards(self) -> None:
        """Rebind runtime optimizer param objects back into the canonical Dion shard map."""
        for group in self.optimizer.param_groups:
            for shard_param in group["params"]:
                if not hasattr(shard_param, "_model_param"):
                    continue
                model_param = shard_param._model_param
                old_opt_shard = self._get_opt_shard(model_param)
                if old_opt_shard is None:
                    continue
                if old_opt_shard is not shard_param:
                    if hasattr(old_opt_shard, "_dion_param_uid"):
                        shard_param._dion_param_uid = old_opt_shard._dion_param_uid
                    self._update_opt_shard(model_param, shard_param)

        if hasattr(self, "shard_fp32_from_float16_groups"):
            for model_group, shard_group in zip(
                self.model_float16_groups, self.shard_fp32_from_float16_groups
            ):
                for model_param, shard_param in zip(model_group, shard_group):
                    if shard_param is not None:
                        self._update_opt_shard(model_param, shard_param)

        self.optimizer._distrib_dion_shard_info = self._dion_shard_info
        self._log_unmapped_main_grads(self.model_float16_groups, "model_float16_groups")
        self._log_unmapped_main_grads(self.model_fp32_groups, "model_fp32_groups")

    def _setup_dion_after_init(self) -> None:
        """Run Dion-specific setup after parent DistributedOptimizer init."""
        self.use_fs_shard_alloc = True
        self._dion_shard_info = {}

        if self.use_fs_shard_alloc and hasattr(self, 'buffers'):
            self._init_dion_buffers()

        if hasattr(self, 'optimizer') and isinstance(self.optimizer, (MegatronDion,)):
            if hasattr(self, 'gbuf_ranges') and hasattr(self, 'buffers'):
                self._check_dion_params()

                if hasattr(self, 'opt_group_ranges'):
                    (
                        self.model_float16_groups,
                        self.model_fp32_groups,
                        self.shard_float16_groups,
                        self.shard_fp32_groups,
                        self.shard_fp32_from_float16_groups,
                    ) = self._build_param_groups(
                        self.gbuf_ranges,
                        self.model_param_gbuf_map,
                        self.opt_group_ranges,
                        self.config
                    )

                    self._refresh_param_groups()
                    self._rebind_dion_shards()
                    self._enable_overlap_grad_reduce = bool(getattr(self.ddp_config, "overlap_grad_reduce", False))
                    self._enable_overlap_param_gather = bool(getattr(self.ddp_config, "overlap_param_gather", False))
                    try:
                        self._enable_dion_mode()
                    except Exception as error:
                        logger.error(
                            f"[Dion] Global rank {self._global_rank}: Failed in _enable_dion_mode: {error}"
                        )
                        logger.error(traceback.format_exc())
                        raise

        if dist.is_initialized() and hasattr(self, 'data_parallel_group'):
            dist.barrier(group=self.data_parallel_group)

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
        ordering_group = getattr(param_and_grad_buffer, "dp_cp_group", None)
        if ordering_group is None:
            ordering_group = dp_group
        ordering_rank = ordering_group.rank()

        # Cache param names for fallback
        if not hasattr(cls, "_param_name_bank"):
            cls._param_name_bank = dict(param_and_grad_buffer.param_to_name) if hasattr(param_and_grad_buffer, 'param_to_name') else {}

        ordered_params = reorder_bucket_params_(
            param_and_grad_buffer=param_and_grad_buffer,
            bucket=bucket,
            dp_group=ordering_group,
            dp_rank=ordering_rank,
            name_bank=cls._param_name_bank,
        )

        param_map = cls._build_bucket_param_map(
            parent_result=parent_result,
            ordered_params=ordered_params,
            dp_group=dp_group,
            dp_rank=dp_rank,
            bucket_index=bucket_index,
            param_index_map=param_and_grad_buffer.param_index_map,
            bucket_offset=bucket.offset,
            bucket_size=bucket.grad_data.numel(),
        )

        fs_size, fs_rank = cls._resolve_fs_rank(dp_group, dp_rank)
        dion_param_count = cls._mark_bucket_dion_params(
            param_map=param_map,
            param_to_name=getattr(param_and_grad_buffer, "param_to_name", None),
            fs_size=fs_size,
        )
        check_bucket_dion_flags_(
            bucket_index=bucket_index,
            dp_group=dp_group,
            param_map=param_map,
            param_to_name=getattr(param_and_grad_buffer, "param_to_name", None),
        )

        # STEP 3: Recalculate buffer ranges for FS × TP hybrid sharding
        # Dion params use FS shard, non-Dion params use DP shard

        dion_param_layout, dion_param_shard_range, dion_section_size, dion_param_count = (
            build_dion_bucket_layout_(
                param_map=param_map,
                param_to_name=getattr(param_and_grad_buffer, "param_to_name", None),
                fs_size=fs_size,
                fs_rank=fs_rank,
                bucket_index=bucket_index,
            )
        )
        # `local_total` is the Dion-owned local section size. Mixed non-Dion params now follow
        # stock DO full-bucket local-shard layout and use separate runtime buffers, so they must
        # not inflate the Dion-local section metadata here.
        local_offset = dion_section_size

        # Second pass: Non-Dion params in MIXED buckets.
        # Keep parent standard-DO `param_map[param]["param"]` intact for optimizer shards.
        # Only build a mixed-bucket communication plan for the extra Dion-managed section.
        if dion_param_count > 0:
            non_dion_params_in_bucket = select_non_dion_bucket_params_(
                bucket_params=ordered_params,
                dion_param_layout=dion_param_layout,
                param_map=param_map,
            )
            if non_dion_params_in_bucket:
                (
                    non_dion_pack_plan,
                    non_dion_param_ranges,
                    non_dion_shard_size,
                    non_dion_full_grad_total,
                ) = build_stock_bucket_non_dion_plan_(
                    bucket=bucket,
                    params=non_dion_params_in_bucket,
                    dp_size=dp_world_size,
                    dp_rank=dp_rank,
                )
                parent_result["non_dion_pack_plan"] = non_dion_pack_plan
                parent_result["non_dion_param_ranges"] = non_dion_param_ranges
                parent_result["non_dion_shard_size"] = non_dion_shard_size
                parent_result["non_dion_full_grad_total"] = non_dion_full_grad_total
            else:
                parent_result["non_dion_pack_plan"] = []
                parent_result["non_dion_param_ranges"] = {}
                parent_result["non_dion_shard_size"] = 0
                parent_result["non_dion_full_grad_total"] = 0

            # Update local_total for buffer resize (used by Dion-only paths).
            parent_result["local_total"] = local_offset

        # Calculate param counts for summary
        total_params = len(param_map)
        non_dion_count = total_params - dion_param_count

        # Add dion_param_layout and dion_param_shard_range to result
        parent_result["dion_param_layout"] = dion_param_layout
        parent_result["dion_param_shard_range"] = dion_param_shard_range

        # Get TP rank for this rank
        from ..parallel_state import get_tensor_model_parallel_rank
        tp_rank = get_tensor_model_parallel_rank()

        check_local_dion_bucket_(
            parent_result=parent_result,
            buckets=param_and_grad_buffer.buckets,
            dion_param_layout=dion_param_layout,
            fs_size=fs_size,
        )
        from .. import parallel_state

        check_shared_dion_layout_(
            dp_group=param_and_grad_buffer.data_parallel_group,
            replica_group=parallel_state.get_inter_distributed_optimizer_instance_group(
                check_initialized=False
            ),
            bucket_index=bucket_index,
            dp_rank=dp_rank,
            tp_rank=tp_rank,
            fs_rank=fs_rank,
            dion_param_layout=dion_param_layout,
            dion_param_count=dion_param_count,
            non_dion_count=non_dion_count,
            local_total=local_offset,
            param_to_name=(
                param_and_grad_buffer.param_to_name
                if hasattr(param_and_grad_buffer, "param_to_name")
                else None
            ),
        )

        # Return hybrid sharding structure (parent ranges + Dion metadata)
        return parent_result

    @classmethod
    def _build_model_param_gbuf_map(
        cls, gbuf_ranges: List[Dict]
    ) -> Dict[torch.nn.Parameter, Tuple]:
        """Create the reverse mapping for locally owned shards only.

        Dion rebuilds each bucket param map in canonical bucket order and inserts
        zero-length placeholder entries for params that are not locally owned on this
        rank. Those placeholders are useful for stable bucket bookkeeping, but they
        must never participate in optimizer ownership. Parent DO only maps params with
        a non-empty local `param` range; Dion must preserve that contract.
        """
        param_gbuf_map = {}
        for gbuf_index, gbuf_range_map in enumerate(gbuf_ranges):
            for dtype, gbuf_range_map_for_all_buckets in gbuf_range_map.items():
                for bucket_index, bucket_range_map in enumerate(gbuf_range_map_for_all_buckets):
                    for param, range_info in bucket_range_map["param_map"].items():
                        param_range = range_info.get("param", None)
                        if param_range is None or param_range.size <= 0:
                            continue
                        assert param not in param_gbuf_map, (
                            "Param should not appear in model_param_gbuf_map more than once; "
                            "only locally owned optimizer shards belong in the map."
                        )
                        param_gbuf_map[param] = (gbuf_index, dtype, bucket_index)
        return param_gbuf_map

    @classmethod
    def _build_optimizer_group_ranges(cls, param_groups: List[Dict], gbuf_ranges: List[Dict]):
        """Build optimizer groups from locally owned shard ranges only.

        Canonical zero-length placeholder entries exist only to stabilize bucket param
        ordering. They must not leak into optimizer param groups, otherwise the optimizer
        starts owning params whose local shard is empty and later violates the parent DO
        local-shard write-back contract.
        """
        world_param_group_map = {}
        for group_index, group in enumerate(param_groups):
            for param in group["params"]:
                assert param.requires_grad
                world_param_group_map[param] = group_index

        local_param_group_map = {}
        group_ranges = [{"params": []} for _ in param_groups]
        for gbuf_range_map in gbuf_ranges:
            for _, bucket_range_maps in gbuf_range_map.items():
                for bucket_range_map in bucket_range_maps:
                    for param, range_info in bucket_range_map["param_map"].items():
                        param_range = range_info.get("param", None)
                        if param_range is None or param_range.size <= 0:
                            continue
                        group_index = world_param_group_map[param]
                        group_range = group_ranges[group_index]
                        group_range["params"].append(param)
                        local_param_group_map[param] = (
                            group_index,
                            len(group_range["params"]) - 1,
                        )

        for group_index, group_range in enumerate(group_ranges):
            group_range["orig_group"] = param_groups[group_index]
            group_range["orig_group_idx"] = param_groups[group_index]

        return local_param_group_map, group_ranges

    def __init__(self, *args, **kwargs):
        """Initialize with improved Dion support."""
        # Initialize Dion param info before parent init
        self._dion_param_info = {}
        self._global_owner_tuples = set()

        # Batch processing configuration
        self._batch_size = kwargs.pop('dion_batch_size', 8)
        self._original_dp_group = kwargs.pop('full_data_parallel_group', None)

        # 2D parallelism configuration
        # RP = Replicate Process (replicas with same shard)
        # FS = Fully Shard (shards within same replica)
        self._rp_size = kwargs.pop('rp_size', None) or kwargs.pop('replica_model_parallel_size', None)
        self._fs_size = kwargs.pop('fs_size', None) or kwargs.pop('fully_shard_model_parallel_size', None)

        if self._rp_size is None:
            self._rp_size = 1

        # Set class variable for _build_model_gbuf_range (accessed during super().__init__()).
        #
        # NOTE: _build_model_gbuf_range is a classmethod invoked inside the parent
        # DistributedOptimizer.__init__(). We stash any FS topology we can infer here
        # (especially the actual fs_group) so range/layout calculations are robust in
        # multi-node and CP>1 settings where dp_rank % fs_size can be wrong.
        opt = args[0] if len(args) > 0 else kwargs.get("optimizer", None)
        self._stash_fs_init(opt)

        # Call parent initialization with full DP group (RP × FS)
        # DistributedOptimizer will do uniform sharding across all DP ranks
        # Dion will handle 2D topology (RP × FS) at optimizer state level
        super().__init__(*args, **kwargs)

        # Clean up class variable after super().__init__()
        self._clear_fs_init()

        # Cache global rank for logging (avoid repeated dist.get_rank() calls)
        self._global_rank = dist.get_rank() if dist.is_initialized() else 0

        self._resolve_fs_group()
        self._resolve_rp_group()
        self._resolve_state_replica_group()

        # Standard-DO-aligned training path: keep post-step full restore disabled.
        # Full-parameter visibility is provided by standard DO param-gather lifecycle.

        # FS shard-alloc mode (bucket-wise all-gather/reduce-scatter via standard DO)
        self.use_fs_shard_alloc = True

        # Unified shard info mapping
        # Consolidates model_param → DionShardInfo (data_shard, opt_shard, metadata)
        self._dion_shard_info: Dict[torch.nn.Parameter, DionShardInfo] = {}
        self._setup_dion_after_init()

        if os.environ.get("DION_DEBUG_GROUPS"):
            try:
                grad_group = self.get_grad_stats_parallel_group()
                logger.info(
                    "[DION_GROUPS] rank=%s tp=%s pp=%s cp=%s fs=%s data_parallel_group_size=%s fs_group_size=%s grad_stats_group_size=%s",
                    self._global_rank,
                    getattr(self.config, "tensor_model_parallel_size", None),
                    getattr(self.config, "pipeline_model_parallel_size", None),
                    getattr(self.config, "context_parallel_size", None),
                    self.fs_size,
                    self.data_parallel_group.size() if self.data_parallel_group is not None else None,
                    self.fs_group.size() if getattr(self, "fs_group", None) is not None else None,
                    grad_group.size() if grad_group is not None else None,
                )
            except Exception as error:
                logger.info("[DION_GROUPS_FAILED] err=%r", error)

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


        # Verify layout invariants across FS group once per bucket.
        # Avoid per-iteration object collectives in the training hot path.
        plan_len = len(dion_param_layout)

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

        if not getattr(bucket, "_dion_ag_invariants_verified", False):
            device = torch.cuda.current_device()
            local = torch.tensor([plan_len, int(pack_total)], device=device, dtype=torch.int64)
            local_min = local.clone()
            local_max = local.clone()
            dist.all_reduce(local_min, op=dist.ReduceOp.MIN, group=fs_group)
            dist.all_reduce(local_max, op=dist.ReduceOp.MAX, group=fs_group)
            if not torch.equal(local_min, local_max):
                # Slow-path details for debugging only.
                try:
                    all_vals = [None] * fs_size
                    dist.all_gather_object(all_vals, (plan_len, int(pack_total)), group=fs_group)
                except Exception:
                    all_vals = None
                logger.error(
                    "[Dion] FS AG invariants mismatch (bucket_id=%s): local(plan_len=%s, pack_total=%s) "
                    "min=%s max=%s all=%s",
                    getattr(bucket, "bucket_id", -1),
                    plan_len,
                    int(pack_total),
                    tuple(int(x) for x in local_min.tolist()),
                    tuple(int(x) for x in local_max.tolist()),
                    all_vals,
                )
                raise RuntimeError("Dion FS all-gather invariants mismatch across FS ranks")

            bucket._dion_ag_invariants_verified = True
            bucket._dion_ag_verified_plan_len = int(plan_len)
            bucket._dion_ag_verified_pack_total = int(pack_total)

        pack_fs_shards_(
            optimizer=self,
            buffer=buffer,
            pack_buffer=pack_buffer,
            dion_param_layout=dion_param_layout,
        )

        # All-gather using cached gathered_buffer

        # Use standard DO API for async support
        handle = torch.distributed.all_gather_into_tensor(
            output_tensor=gathered_buffer,
            input_tensor=pack_buffer,
            group=fs_group,
            async_op=async_op
        )

        # One-time diagnostic: detect if the packed shard buffer is entirely zero.
        # This avoids per-param host sync in the hot loop above.
        if not getattr(bucket, "_dion_ag_pack_nonzero_checked", False):
            bucket._dion_ag_pack_nonzero_checked = True
            try:
                if pack_buffer.numel() > 0 and not bool((pack_buffer != 0).any().item()):
                    logger.error(
                        "[DION_ZERO_PACKED_SHARDS] global_rank=%s fs_rank=%s bucket_id=%s plan_len=%s pack_total=%s",
                        global_rank,
                        fs_rank,
                        getattr(bucket, "bucket_id", -1),
                        plan_len,
                        int(pack_total),
                    )
            except Exception:
                pass

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
            self._unpack_gathered_params(
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

    def _all_gather_non_dion(self, bucket, buffer, async_op=False):
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

        dp_group = getattr(bucket, 'non_dion_dp_group', None)
        if dp_group is None:
            raise RuntimeError(
                "[Dion] mixed non-Dion all-gather requires canonical non_dion_dp_group."
            )
        dp_size = dp_group.size()
        dp_rank = dp_group.rank()
        if dp_size == 1:
            # No all-gather needed for single rank
            return None

        pack_total = getattr(bucket, "non_dion_pack_total", 0)
        if pack_total <= 0:
            return None

        if (
            not hasattr(bucket, "non_dion_pack_buffer")
            or bucket.non_dion_pack_buffer is None
            or bucket.non_dion_pack_buffer.numel() != pack_total
        ):
            bucket.non_dion_pack_buffer = torch.zeros(
                pack_total,
                dtype=buffer.param_dtype,
                device=torch.cuda.current_device(),
            )
        if (
            not hasattr(bucket, "non_dion_gathered_buffer")
            or bucket.non_dion_gathered_buffer is None
            or bucket.non_dion_gathered_buffer.numel() != pack_total * dp_size
        ):
            bucket.non_dion_gathered_buffer = torch.zeros(
                pack_total * dp_size,
                dtype=buffer.param_dtype,
                device=torch.cuda.current_device(),
            )

        pack_buffer = bucket.non_dion_pack_buffer
        gathered_buffer = bucket.non_dion_gathered_buffer
        pack_buffer.zero_()

        for entry in bucket.non_dion_pack_plan:
            param = entry["param"]
            pack_offset = int(entry.get("local_bucket_start", 0))
            local_start = int(entry.get("local_start", 0))
            local_end = int(entry.get("local_end", local_start))
            actual_size = max(0, local_end - local_start)
            if actual_size <= 0:
                continue

            full_view = self._bucket_param_view(bucket, param)
            if full_view is None:
                param_name = self._param_name(getattr(buffer, "param_to_name", None), param)
                raise RuntimeError(
                    "[Dion] mixed non-Dion all-gather requires canonical bucket.param_data view "
                    f"for param={param_name or f'id_{id(param)}'} "
                    f"bucket={getattr(bucket, 'bucket_id', -1)}"
                )
            param_flat = full_view.view(-1)
            local_shard = param_flat[local_start:local_end]
            pack_buffer[pack_offset : pack_offset + actual_size].copy_(local_shard)

        handle = torch.distributed.all_gather_into_tensor(
            output_tensor=gathered_buffer,
            input_tensor=pack_buffer,
            group=dp_group,
            async_op=async_op,
        )

        if async_op:
            return {
                "handle": handle,
                "gathered_buffer": gathered_buffer,
                "pack_plan": bucket.non_dion_pack_plan,
                "pack_total": pack_total,
                "dp_size": dp_size,
                "buffer": buffer,
                "optimizer": self,
            }

        self._unpack_non_dion_params(
            gathered_buffer=gathered_buffer,
            pack_plan=bucket.non_dion_pack_plan,
            pack_total=pack_total,
            dp_size=dp_size,
            buffer=buffer,
        )
        bucket.non_dion_pack_buffer = None
        bucket.non_dion_gathered_buffer = None
        return None

    def _unpack_non_dion_params(self, gathered_buffer, pack_plan, pack_total, dp_size, buffer):
        """Unpack stock local-shard mixed non-Dion all-gather results into full `param.data`."""
        if not pack_plan:
            return
        for entry in pack_plan:
            param = entry["param"]
            bucket = None
            if hasattr(buffer, "param_to_bucket"):
                bucket = buffer.param_to_bucket.get(param)
            pack_offset = entry["pack_offset"]
            full_view = self._bucket_param_view(bucket, param)
            if full_view is None:
                param_name = self._param_name(getattr(buffer, "param_to_name", None), param)
                raise RuntimeError(
                    "[Dion] mixed non-Dion unpack requires canonical bucket.param_data view "
                    f"for param={param_name or f'id_{id(param)}'} "
                    f"bucket={getattr(bucket, 'bucket_id', -1)}"
                )
            param_flat = full_view.view(-1)
            rank_param_ranges = entry.get("rank_param_ranges", ())
            rank_bucket_ranges = entry.get("rank_bucket_ranges", ())

            for rank_i in range(dp_size):
                if rank_param_ranges:
                    shard_start, shard_end = rank_param_ranges[rank_i]
                else:
                    continue
                if rank_bucket_ranges:
                    bucket_start, bucket_end = rank_bucket_ranges[rank_i]
                else:
                    continue
                actual_size = max(0, shard_end - shard_start)
                if actual_size <= 0:
                    continue
                rank_pack_offset = rank_i * pack_total + int(bucket_start)
                rank_segment = gathered_buffer[
                    rank_pack_offset : rank_pack_offset + actual_size
                ]
                param_flat[shard_start:shard_end].copy_(rank_segment)

    def _unpack_gathered_params(self, gathered_buffer, dion_param_layout, pack_total, fs_size, buffer):
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
        unpack_fs_shards_(
            optimizer=self,
            gathered_buffer=gathered_buffer,
            dion_param_layout=dion_param_layout,
            pack_total=pack_total,
            fs_size=fs_size,
            fs_rank=self.fs_rank,
            buffer=buffer,
        )

    # Backward-compatible alias for param_and_grad_buffer async param-gather path.
    def _unpack_all_gathered_params(
        self, gathered_buffer, dion_param_layout, pack_total, fs_size, buffer
    ):
        self._unpack_gathered_params(
            gathered_buffer=gathered_buffer,
            dion_param_layout=dion_param_layout,
            pack_total=pack_total,
            fs_size=fs_size,
            buffer=buffer,
        )

    def _check_dion_params(self):
        """
        Improved annotation with batch processing for efficiency.
        Uses original model parameters directly (independent of FS sharding).
        """
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

        global_rank = self._global_rank

        self._param_owner_ranks = {}
        self._fs_rank = fs_rank
        self._fs_size = fs_size

        # Validate metadata created by `_build_model_gbuf_range()` and build
        # cheap lookup tables used later in the optimizer.
        total_params = 0
        total_dion_params = 0
        mismatch_dion_flag = 0
        repaired_missing_fields = 0

        for gbuf_idx, gbuf_range_maps in enumerate(self.gbuf_ranges):
            buffer = self.buffers[gbuf_idx]

            for dtype in sorted(gbuf_range_maps.keys(), key=lambda dt: str(dt)):
                gbuf_range_map_list = gbuf_range_maps[dtype]

                for bucket_idx, gbuf_range_map in enumerate(gbuf_range_map_list):
                    # Validate per-bucket Dion metadata.
                    for param in gbuf_range_map["param_map"].keys():
                        total_params += 1
                        pri = gbuf_range_map["param_map"][param]

                        dion_info = pri.get("dion_info", None) or {"is_dion": False}
                        pri["dion_info"] = dion_info  # Ensure key exists (behavior-preserving)

                        is_dion = bool(dion_info.get("is_dion", False))
                        flag_is_dion = bool(getattr(param, "is_dion_param", False))
                        if flag_is_dion != is_dion:
                            mismatch_dion_flag += 1
                            pname = ""
                            try:
                                pname = buffer.param_to_name.get(param, "")
                            except Exception:
                                pname = ""
                            logger.warning(
                                f"[Dion] dion flag mismatch (will trust range-map dion_info): "
                                f"name={pname or f'id_{id(param)}'} is_dion_param={flag_is_dion} dion_info.is_dion={is_dion} "
                                f"(gbuf={gbuf_idx} bucket={bucket_idx})"
                            )

                        if not is_dion:
                            continue

                        total_dion_params += 1

                        # `_build_model_gbuf_range()` should have fully populated these.
                        # Keep a slow-path repair (log-only) for robustness in case of
                        # stale checkpoints or upstream changes.
                        missing = [k for k in ("shape", "start_idx", "end_idx", "fs_split_dim") if k not in dion_info]
                        if missing:
                            repaired_missing_fields += 1
                            tp_split_dim = dion_info.get("tp_split_dim", get_tp_split_dim(param))
                            fs_split_dim = dion_info.get("fs_split_dim", get_fs_split_dim(tp_split_dim))
                            m, n = param.shape
                            split_size = m if fs_split_dim == 0 else n
                            start_idx, end_idx = compute_fs_shard_range(split_size, fs_size, fs_rank)
                            local_shape = compute_local_shape(m, n, start_idx, end_idx, fs_split_dim)
                            dion_info.update(
                                {
                                    "shape": local_shape,
                                    "start_idx": start_idx,
                                    "end_idx": end_idx,
                                    "fs_split_dim": fs_split_dim,
                                    "bucket_idx": dion_info.get("bucket_idx", bucket_idx),
                                    "fs_owner_ranks": dion_info.get("fs_owner_ranks", tuple(range(fs_size))),
                                }
                            )
                            pname = ""
                            try:
                                pname = buffer.param_to_name.get(param, "")
                            except Exception:
                                pname = ""
                            logger.warning(
                                f"[Dion] Repaired missing dion_info fields {missing} for "
                                f"name={pname or f'id_{id(param)}'} (gbuf={gbuf_idx} bucket={bucket_idx})"
                            )

                        # Cache the dict for any later lookups/debug.
                        self._dion_param_info[param] = dion_info
                        param.dion_info = dion_info
                        self._param_owner_ranks[param] = tuple(range(fs_size))

        if mismatch_dion_flag and self._global_rank == 0:
            logger.warning(f"[Dion] dion flag mismatches observed: {mismatch_dion_flag}")
        if repaired_missing_fields and self._global_rank == 0:
            logger.warning(f"[Dion] repaired dion_info missing fields for {repaired_missing_fields} params")

        # Convert FS group members to DP rank indices for owner_tuples
        # owner_tuples must be in DP rank space for later DP→world conversion
        if fs_group is not None and fs_size > 1:
            # Get FS group's world ranks
            fs_world_ranks = dist.get_process_group_ranks(fs_group)

            # Get full stock dp_cp world's ranks to create world→DP mapping
            full_dp_group = self._original_dp_group if self._original_dp_group is not None else self.data_parallel_group
            dp_world_ranks = dist.get_process_group_ranks(full_dp_group)
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
        full_dp_group = self._original_dp_group if self._original_dp_group is not None else self.data_parallel_group
        dp_world_ranks = dist.get_process_group_ranks(full_dp_group)
        dp_size = len(dp_world_ranks)

        gathered_tuples = [None] * dp_size
        dist.all_gather_object(gathered_tuples, local_owner_tuples, group=full_dp_group)

        # Collect all unique tuples from all ranks (union instead of enforcing uniformity)
        all_unique_tuples = set()
        for tuples in gathered_tuples:
            if tuples:
                all_unique_tuples.update(tuples)

        # Use union of all tuples
        self._global_owner_tuples = all_unique_tuples

        # No barrier needed - each FS group operates independently
        # TP slices have different FS groups, barrier would hang

    def _build_param_groups(
        self,
        gbuf_ranges: List,
        param_gbuf_map: Dict,
        opt_group_ranges: List,
        config,
    ):
        """
        Build parameter groups with batch processing for 2D views.
        """
        use_prec_opt = config.use_precision_aware_optimizer_no_fp8_or_ds_fp8
        # Initialize parameter groups
        model_fp16_groups = []
        model_fp32_groups = []
        shard_float16_groups = []
        shard_fp32_groups = []
        main_shard_groups = []

        # Process each optimizer group
        for group_range in opt_group_ranges:
            # Initialize group lists
            model_fp16_params = []
            model_fp32_params_this_group = []
            shard_float16_params_this_group = []
            shard_fp32_params_this_group = []
            main_shard_params = []

            # Add to main groups
            model_fp16_groups.append(model_fp16_params)
            model_fp32_groups.append(model_fp32_params_this_group)
            shard_float16_groups.append(shard_float16_params_this_group)
            shard_fp32_groups.append(shard_fp32_params_this_group)
            main_shard_groups.append(main_shard_params)

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
                            model_fp16_params,
                            model_fp32_params_this_group,
                            shard_float16_params_this_group,
                            shard_fp32_params_this_group,
                            main_shard_params
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
            if not use_prec_opt:
                group_range["orig_group"]["params"] = [
                    *shard_fp32_params_this_group,
                    *main_shard_params,
                ]
            else:
                group_range["orig_group"]["params"] = [
                    *shard_fp32_params_this_group,
                    *shard_float16_params_this_group,
                ]

        return (
            model_fp16_groups,
            model_fp32_groups,
            shard_float16_groups,
            shard_fp32_groups,
            main_shard_groups,
        )

    def _process_param_batch(self, param_batch, gbuf_ranges, param_gbuf_map, config,
                            model_fp16_params, model_fp32_params,
                            shard_float16_params, shard_fp32_params,
                            main_shard_params):
        """Process a batch of parameters efficiently."""
        for model_param in param_batch:
            assert model_param.requires_grad

            gbuf_index, dtype, bucket_index = param_gbuf_map[model_param]
            gbuf_range = gbuf_ranges[gbuf_index][dtype][bucket_index]
            param_range_info = gbuf_range["param_map"][model_param]
            param_range = param_range_info["param"]
            dion_info = param_range_info.get("dion_info", {})
            if dion_info:
                model_param.dion_info = dion_info

            # Handle different parameter types
            if model_param.type() in ['torch.cuda.HalfTensor', 'torch.cuda.BFloat16Tensor']:
                self._process_float16_param(
                    model_param, param_range, param_range_info, dion_info,
                    config, gbuf_index, bucket_index,
                    model_fp16_params, shard_float16_params,
                    main_shard_params
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
        """Create the local FS shard view from `model_param`.

        Args:
            model_param: The model parameter tensor (2D, TP-partitioned)
            dion_info: Dict containing 'start_idx', 'end_idx', 'fs_split_dim'

        Returns:
            shard: The local FS shard tensor view into `model_param.data`
        """
        start_idx = dion_info['start_idx']
        end_idx = dion_info['end_idx']
        fs_split_dim = dion_info['fs_split_dim']

        # Keep the FP16 shard on the same storage graph as the forward-visible param.
        # This matches standard Megatron-Core more closely: local shard state is a view
        # into the full param buffer, not a clone-owned shadow tensor.
        shard = slice_fs_shard_2d(model_param.detach(), fs_split_dim, start_idx, end_idx)

        shard._model_param = model_param
        return shard

    def _prepare_fs_shard(self, model_param, shard):
        """Attach FS shard to model_param for optimizer state.

        `param.data` stays as the forward-visible full/TP-local tensor.
        `_fs_shard` is the local FS view into that same storage.

        Args:
            model_param: The model parameter tensor (FS-full, TP-sharded; Megatron-Core standard)
            shard: The FS shard tensor
        """
        dion_info = getattr(model_param, 'dion_info', None)
        if dion_info is not None:
            expected_view = slice_fs_shard_2d(
                model_param.data,
                dion_info['fs_split_dim'],
                dion_info['start_idx'],
                dion_info['end_idx'],
            )
            if shard.shape != expected_view.shape or shard.data_ptr() != expected_view.data_ptr():
                param_name = getattr(model_param, '_param_name', f'id_{id(model_param)}')
                logger.error(
                    "[DION_FS_ALIAS_MISMATCH] param=%s shard_shape=%s expected_shape=%s shard_ptr=%s expected_ptr=%s",
                    param_name,
                    tuple(shard.shape),
                    tuple(expected_view.shape),
                    shard.data_ptr(),
                    expected_view.data_ptr(),
                )
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
        # Dion packed RS remains a transient communication buffer. The canonical
        # optimizer-side local shard contract stays aligned with stock DO:
        # the final local optimizer shard is projected back into the parent
        # bucket.grad_data / model_param.main_grad local shard range.
        rs_start = None
        rs_end = None
        try:
            bucket = self.buffers[gbuf_index].buckets[bucket_index]
            dion_shard_range = getattr(bucket, "dion_param_shard_range", None)
            if dion_shard_range is not None and model_param in dion_shard_range:
                rs_start, rs_end = dion_shard_range[model_param]
        except Exception:
            rs_start = None
            rs_end = None

        if rs_start is None or rs_end is None:
            param_name = getattr(model_param, "_param_name", f"id_{id(model_param)}")
            raise RuntimeError(
                "[Dion] Missing canonical dion_param_shard_range for optimizer grad source: "
                f"param={param_name} buffer={gbuf_index} bucket={bucket_index}"
            )

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
            rs_start=int(rs_start),
            rs_end=int(rs_end),
            stock_param_start=int(dion_info["stock_param_start"]),
            stock_param_end=int(dion_info["stock_param_end"]),
            per_expert_global_shape=dion_info.get('per_expert_global_shape'),
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

    def _param_dion_info(self, model_param: torch.nn.Parameter) -> dict:
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
            "rs_start": shard_info.rs_start,
            "rs_end": shard_info.rs_end,
            "stock_param_start": shard_info.stock_param_start,
            "stock_param_end": shard_info.stock_param_end,
            "per_expert_global_shape": shard_info.per_expert_global_shape,
        }

    def _check_is_dion(self, model_param, context=""):
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
                              model_fp16_params, shard_float16_params,
                              main_shard_params):
        """Process float16/bfloat16 parameters."""
        use_prec_opt = config.use_precision_aware_optimizer_no_fp8_or_ds_fp8
        if dion_info.get('is_dion', False):
            try:
                # Create FS shard using helper
                shard_model_param = self._create_fs_shard(model_param, dion_info)

                # Prepare FS shard (attach to model_param)
                self._prepare_fs_shard(model_param, shard_model_param)

                # Verify is_dion_param flag
                self._check_is_dion(model_param, "FP16")
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
            if not use_prec_opt:
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
            if not use_prec_opt:
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

            # Preserve the exact parent-DO local shard contract used to build the
            # non-Dion optimizer shard. Runtime write-back must target this same
            # bucket-local range, not a later re-derived bookkeeping entry.
            stock_param_range = param_range_info["param"]
            stock_world_range = param_range_info["gbuf_world_in_bucket"]
            model_param._stock_param_start = int(stock_param_range.start)
            model_param._stock_param_end = int(stock_param_range.end)
            model_param._stock_world_bucket_start = int(stock_world_range.start)
            model_param._stock_world_bucket_end = int(stock_world_range.end)
            model_param._stock_bucket_index = int(bucket_index)
            model_param._stock_gbuf_index = int(gbuf_index)

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
        model_fp16_params.append(model_param)
        shard_float16_params.append(shard_model_param)
        main_shard_params.append(shard_main_param)

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
            self._check_is_dion(model_param, "FP32")

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

    def _enable_dion_mode(self):
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

        # Create replica groups with improved batching (will be no-op if groups already set)
        self._validate_prebuilt_topology_groups()

        # Create true RP and FS groups for 2D parallelism (will be no-op if groups already set)
        self._init_topology_cache()

        # Create dist_metas with batch processing
        try:
            dist_metas_sharded = self._build_dist_metas()
        except Exception as e:
            logger.error(f"[Dion] Global rank {global_rank}: Failed in _build_dist_metas: {e}")
            logger.error(traceback.format_exc())
            raise

        # Enable distributed mode with 2D parallelism support
        # Use original DP group for full_data_parallel_group (not fs_group used for sharding)
        full_data_parallel_group = (
            self._original_dp_group
            if hasattr(self, '_original_dp_group') and self._original_dp_group
            else self.data_parallel_group
        )

        enable_args = {
            'global_buffer_sizes': gbuf_sizes,
            'full_data_parallel_group': full_data_parallel_group,  # Original full DP = RP × FS
            'tp_group': get_tensor_model_parallel_group(),
            'dist_metas': dist_metas_sharded,
            'rp_group': self.rp_group,
            'fs_group': self.fs_group,
            'state_replica_group': self.state_replica_group,
        }


        self.optimizer.enable_distributed_mode(**enable_args)

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
            [1 if self.rp_group is not None else 0],
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
            if self.rp_group is None:
                raise RuntimeError(
                    f"Global rank {global_rank}: all_have_rp=True but rp_group is None! "
                    f"This indicates a bug in group creation or collective voting logic."
                )

            # Verify Dion eligibility consistency within RP group
            # All ranks in same RP group must have identical Dion param counts
            # Now ALL ranks will execute this block (no partial participation)
            my_dion_count = sum(1 for meta in dist_metas_sharded.values() if meta.is_dion_param)
            my_cnt_tensor = torch.tensor([my_dion_count], device=torch.cuda.current_device(), dtype=torch.int64)

            rp_world_size = dist.get_world_size(self.rp_group)
            gathered = [torch.zeros_like(my_cnt_tensor) for _ in range(rp_world_size)]
            dist.all_gather(gathered, my_cnt_tensor, group=self.rp_group)

            gathered_counts = [int(t.item()) for t in gathered]
            if not all(count == my_dion_count for count in gathered_counts):
                # Collect Dion param offsets for error diagnosis
                my_dion_offsets = sorted([
                    meta.param_uid for meta in dist_metas_sharded.values() if meta.is_dion_param
                ])
                gathered_offsets = [None] * rp_world_size
                dist.all_gather_object(gathered_offsets, my_dion_offsets, group=self.rp_group)

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

    def _validate_prebuilt_topology_groups(self):
        """
        DISABLED: No-op - RP/FS groups already created in __init__.py

        This function previously created additional subgroups based on owner_tuples,
        but these groups are not actually used (dist_meta.replica_group uses self.rp_group instead).
        Creating unnecessary groups causes new_group() call count mismatch across ranks,
        leading to NCCL hang.

        Solution: Skip all new_group() calls. Use collective voting to ensure all ranks
        take the same path (critical for PyTorch new_group() synchronization).
        """
        global_rank = self._global_rank
        dp_group = self.data_parallel_group
        dp_world_size = dist.get_world_size(dp_group)

        # RP=1 is valid and represented as rp_group=None. What must be uniform is
        # the availability of fs_group, while rp_group may be absent on all ranks.
        have_fs = torch.tensor(
            [1 if hasattr(self, 'fs_group') and self.fs_group is not None else 0],
            device=torch.cuda.current_device(), dtype=torch.int64
        )
        dist.all_reduce(have_fs, op=dist.ReduceOp.MIN, group=dp_group)
        all_have_fs = int(have_fs.item()) == 1

        if all_have_fs:
            # All ranks have the required FS group. RP may legitimately be None
            # on every rank when the stock runtime has a single optimizer instance.
            self.owner_tuple_to_subgroup = {}
            dist.barrier(group=dp_group)
            return
        elif hasattr(self, 'fs_group') and self.fs_group is not None:
            # Only some ranks have fs_group - FATAL ERROR
            raise RuntimeError(
                f"Global rank {global_rank}: FS groups exist only on subset of ranks! "
                f"Ensure rp_group/fs_group are provided uniformly to prevent new_group() mismatch."
            )

        # If we reach here, no ranks have fs_group - this is a real topology bug.
        raise RuntimeError(
            f"Global rank {global_rank}: No FS group found! "
            f"FS group must be provided from the stock Megatron-Core optimizer topology."
        )

    def _init_topology_cache(self):
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


        # FS group is required. RP group may legitimately be absent on every rank
        # when the stock runtime has a single optimizer instance.
        have_fs = torch.tensor(
            [1 if hasattr(self, 'fs_group') and self.fs_group is not None else 0],
            device=torch.cuda.current_device(), dtype=torch.int64
        )
        dist.all_reduce(have_fs, op=dist.ReduceOp.MIN, group=dp_group)
        all_have_fs = int(have_fs.item()) == 1

        if all_have_fs:
            # All ranks have FS group. RP may be None on all ranks for RP=1.

            # Initialize compatibility variables
            self.shard_index_to_rp_group = {}
            self.replica_index_to_fs_group = {}
            self.my_replica_idx = -1
            self.my_shard_idx = -1
            self.num_replicas = (
                dist.get_world_size(self.rp_group) if self.rp_group is not None else 1
            )
            self.num_shards = dist.get_world_size(self.fs_group)

            # Barrier to ensure all ranks finish together
            dist.barrier(group=dp_group)
            return

        elif hasattr(self, 'fs_group') and self.fs_group is not None:
            # Only some ranks have fs_group - FATAL ERROR
            raise RuntimeError(
                f"Global rank {global_rank}: FS groups exist only on subset of ranks! "
                f"Ensure rp_group/fs_group are provided uniformly to prevent new_group() mismatch."
            )

        # If we reach here, no ranks have fs_group - should never happen.
        raise RuntimeError(
            f"Global rank {global_rank}: No FS group found! "
            f"FS group must be provided from the stock Megatron-Core optimizer topology."
        )

    def _model_name_cache(self) -> Dict[int, str]:
        """Return cached model-param-id -> canonical name mapping."""
        key = tuple(
            id(getattr(buffer, "param_to_name", None))
            for buffer in getattr(self, "buffers", [])
        )
        cached_key = getattr(self, "_model_param_name_cache_key", None)
        cached = getattr(self, "_model_param_name_cache", None)
        if cached is not None and cached_key == key:
            return cached

        model_name_by_id: Dict[int, str] = {}
        for buffer in getattr(self, "buffers", []):
            param_to_name = getattr(buffer, "param_to_name", None)
            if not param_to_name:
                continue
            for model_param, name in param_to_name.items():
                model_name_by_id[id(model_param)] = name

        self._model_param_name_cache = model_name_by_id
        self._model_param_name_cache_key = key
        return model_name_by_id

    def _model_by_shard(self) -> Dict[int, torch.nn.Parameter]:
        """Return cached shard-param-id -> model-param mapping."""
        key = id(getattr(self, "gbuf_ranges", None))
        cached_key = getattr(self, "_model_param_by_shard_cache_key", None)
        cached = getattr(self, "_model_param_by_shard_cache", None)
        if cached is not None and cached_key == key:
            return cached

        model_param_by_shard_id: Dict[int, torch.nn.Parameter] = {}
        for gbuf_range_maps in getattr(self, "gbuf_ranges", []):
            for gbuf_range_map_list in gbuf_range_maps.values():
                for gbuf_range_map in gbuf_range_map_list:
                    for model_param in gbuf_range_map["param_map"].keys():
                        shard_param = getattr(model_param, "main_param", None)
                        if shard_param is not None:
                            model_param_by_shard_id[id(shard_param)] = model_param

        self._model_param_by_shard_cache = model_param_by_shard_id
        self._model_param_by_shard_cache_key = key
        return model_param_by_shard_id

    def _shard_name_cache(self) -> Dict[int, str]:
        """Return cached shard-param-id -> canonical name mapping."""
        model_name_by_id = self._model_name_cache()
        _, _, _, shard_to_model = self._grad_norm_maps()
        key = (
            id(model_name_by_id),
            len(shard_to_model),
            getattr(self, "_grad_norm_param_maps_key", None),
        )
        cached_key = getattr(self, "_shard_param_name_cache_key", None)
        cached = getattr(self, "_shard_param_name_cache", None)
        if cached is not None and cached_key == key:
            return cached

        shard_name_by_id = {}
        for shard_id, model_param in shard_to_model.items():
            param_name = model_name_by_id.get(id(model_param))
            if param_name is not None:
                shard_name_by_id[shard_id] = param_name

        self._shard_param_name_cache = shard_name_by_id
        self._shard_param_name_cache_key = key
        return shard_name_by_id

    @staticmethod
    def _group_info(group) -> Tuple[int, int]:
        """Return (world_size, rank) for a process group, or (1, -1) if absent."""
        if group is None:
            return 1, -1
        return dist.get_world_size(group), dist.get_rank(group)

    @staticmethod
    def _group_ranks(group):
        if group is None:
            return None
        return tuple(dist.get_process_group_ranks(group))

    def _expected_expert_shard_group(self):
        group = parallel_state.get_expert_data_parallel_group(
            check_initialized=False,
            partial_expert_data_parallel=True,
        )
        if group is None:
            raise RuntimeError(
                "[Dion][EP] missing expert-local shard group "
                "(expected intra_expt_dp_group for expert param/bucket)."
            )
        return group

    def _assert_group_matches(self, *, label: str, actual_group, expected_group, extra: str = "") -> None:
        actual_ranks = self._group_ranks(actual_group)
        expected_ranks = self._group_ranks(expected_group)
        if actual_ranks != expected_ranks:
            raise RuntimeError(
                f"[Dion][EP] {label} group mismatch: "
                f"actual={actual_ranks} expected={expected_ranks}. {extra}".strip()
            )

    def _select_shard_group(self, model_param):
        """Return the FS shard group for a model param (dense vs expert-parallel)."""
        if not getattr(model_param, 'allreduce', True):
            expert_group = self._expected_expert_shard_group()
            self._assert_group_matches(
                label="expert param shard_group",
                actual_group=self.fs_group,
                expected_group=expert_group,
                extra=(
                    f"param={getattr(model_param, '_param_name', '') or id(model_param)} "
                    "Dion EP must keep expert params on the stock expert-local DO shard group."
                ),
            )
            return expert_group
        return self.fs_group

    @staticmethod
    def _make_param_uid(
        *,
        param_name: str,
        logical_global_shape=None,
        is_dion_param: bool,
    ):
        """Build a topology-independent logical param identity for optimizer state.

        The same logical parameter must keep the same state identity across
        different FS/TP/CP shard layouts. Runtime shard ranges are not part of the
        logical identity.
        """
        if not param_name:
            raise RuntimeError(
                "[Dion] Missing param_name while building logical param_uid"
            )
        return (
            str(param_name or ""),
            tuple(int(dim) for dim in logical_global_shape) if logical_global_shape is not None else (),
            bool(is_dion_param),
        )

    def _build_dist_meta(
        self,
        *,
        model_param,
        shard_param,
        shard_info: DionShardInfo,
    ) -> MegatronDionDistMeta:
        """Build one MegatronDionDistMeta from cached shard info."""
        param_range_info = shard_info.param_range_info
        if is_tp_enabled(model_param):
            tp_split_dim = get_tp_split_dim(model_param)
        else:
            tp_split_dim = -1

        model_name_by_id = self._model_name_cache()
        param_name = model_name_by_id.get(id(model_param), "")
        is_expert = is_moe_expert_param(model_param, param_name)

        replica_group = self.rp_group
        shard_group = self._select_shard_group(model_param)
        replica_group_world_size, replica_group_rank = self._group_info(replica_group)
        shard_group_world_size, shard_group_rank = self._group_info(shard_group)

        dist_meta = MegatronDionDistMeta(
            buffer_idx=shard_info.gbuf_index,
            bucket_idx=shard_info.bucket_index,
            shape=shard_param.shape,
            global_shape=shard_info.global_shape,
            global_range=(
                param_range_info["gbuf_world"].start,
                param_range_info["gbuf_world"].end,
            ),
            tp_split_dim=tp_split_dim,
            fs_split_dim=shard_info.fs_split_dim,
            rank_fraction=self.optimizer.defaults.get('rank_fraction', 0.25),
            is_dion_param=getattr(model_param, 'is_dion_param', True),
            param_uid=self._make_param_uid(
                param_name=param_name,
                logical_global_shape=(
                    shard_info.per_expert_global_shape
                    if shard_info.per_expert_global_shape is not None
                    else shard_info.global_shape
                ),
                is_dion_param=True,
            ),
            is_expert=is_expert,
            param_name=param_name,
        )
        dist_meta.replica_group = replica_group
        dist_meta.replica_group_world_size = replica_group_world_size
        dist_meta.replica_group_rank = replica_group_rank
        dist_meta.shard_group = shard_group
        dist_meta.shard_group_world_size = shard_group_world_size
        dist_meta.shard_group_rank = shard_group_rank
        dist_meta.replica_group_id = shard_group_rank if shard_group else 0
        dist_meta.per_expert_global_shape = shard_info.per_expert_global_shape

        if os.environ.get("DION_TOPO_DIAG", "0") == "1":
            try:
                target = os.environ.get("DION_TOPO_DIAG_PARAM")
                if not target or target in param_name:
                    shard_ranks = (
                        dist.get_process_group_ranks(shard_group) if shard_group is not None else None
                    )
                    replica_ranks = (
                        dist.get_process_group_ranks(replica_group)
                        if replica_group is not None
                        else None
                    )
                    state_ranks = (
                        dist.get_process_group_ranks(self.state_replica_group)
                        if getattr(self, "state_replica_group", None) is not None
                        else None
                    )
                    logger.info(
                        "[DION_TOPO_DIAG] param=%s rank=%s shard_group=%s shard_rank=%s "
                        "replica_group=%s replica_rank=%s state_group=%s local_shape=%s "
                        "global_shape=%s fs_split=%s tp_split=%s",
                        param_name,
                        self._global_rank,
                        shard_ranks,
                        shard_group_rank,
                        replica_ranks,
                        replica_group_rank,
                        state_ranks,
                        tuple(shard_param.shape),
                        tuple(shard_info.global_shape) if shard_info.global_shape is not None else None,
                        shard_info.fs_split_dim,
                        tp_split_dim,
                    )
            except Exception as error:
                logger.error("[DION_TOPO_DIAG_FAILED] param=%s err=%r", param_name, error)
        return dist_meta

    def _build_dist_metas(self):
        """Create dist_metas with batch processing."""
        dist_metas_sharded = {}

        # Batch process Dion shard mappings (Phase 3: using unified _dion_shard_info)
        for model_param, shard_info in self._dion_shard_info.items():
            shard_param = shard_info.opt_shard  # Key for dist_metas_sharded
            dist_meta = self._build_dist_meta(
                model_param=model_param,
                shard_param=shard_param,
                shard_info=shard_info,
            )
            dist_metas_sharded[shard_param] = dist_meta
            # Store param_uid directly on param for lookup after offload/reload
            # This allows _get_or_initialize_state to find UID even when dist_metas lookup fails
            shard_param._dion_param_uid = dist_meta.param_uid

        # Add non-Dion parameters
        self._add_non_dion_metas(dist_metas_sharded)

        uid_to_entries = {}
        for shard_param, meta in dist_metas_sharded.items():
            uid_to_entries.setdefault(meta.param_uid, []).append(
                {
                    "name": getattr(meta, "param_name", "") or self._shard_name(shard_param),
                    "shape": tuple(meta.shape) if meta.shape is not None else tuple(shard_param.shape),
                    "bucket": (meta.buffer_idx, meta.bucket_idx),
                    "is_dion": bool(getattr(meta, "is_dion_param", False)),
                }
            )
        duplicate_uids = {uid: entries for uid, entries in uid_to_entries.items() if len(entries) > 1}
        if duplicate_uids:
            for uid, entries in sorted(duplicate_uids.items(), key=lambda item: str(item[0])):
                logger.error("[Dion] Duplicate param_uid=%s entries=%s", uid, entries)
            raise RuntimeError(
                "[Dion] Duplicate param_uid detected in dist_metas; "
                "optimizer state identity would be ambiguous"
            )

        return dist_metas_sharded

    def _add_non_dion_metas(self, dist_metas_sharded):
        """Add dist_metas for non-Dion parameters."""
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p not in dist_metas_sharded:
                    # Find model param
                    model_param = getattr(p, '_model_param', None)

                    if model_param is None:
                        # Cached reverse lookup fallback
                        model_param = self._find_model_param_for_shard(p)

                    if model_param is None:
                        logger.warning(f"[Dion] Could not find model_param for shard shape={p.shape}")
                        continue

                    # Create basic dist_meta
                    gbuf_index, dtype, bucket_index = self.model_param_gbuf_map[model_param]
                    gbuf_range = self.gbuf_ranges[gbuf_index][dtype][bucket_index]
                    pr = gbuf_range["param_map"][model_param]["gbuf_world"]

                    param_name = self._model_name_cache().get(id(model_param), "") or self._shard_name(p)
                    param_uid = self._make_param_uid(
                        param_name=param_name,
                        logical_global_shape=None,
                        is_dion_param=getattr(model_param, 'is_dion_param', False),
                    )
                    dist_metas_sharded[p] = MegatronDionDistMeta(
                        buffer_idx=gbuf_index,
                        bucket_idx=bucket_index,
                        shape=p.shape,
                        global_shape=None,
                        global_range=(pr.start, pr.end),
                        tp_split_dim=-1,
                        rank_fraction=self.optimizer.defaults.get('rank_fraction', 0.25),
                        is_dion_param=getattr(model_param, 'is_dion_param', False),
                        param_uid=param_uid,
                        param_name=param_name,
                    )
                    # Store param_uid directly on param for lookup after offload/reload
                    p._dion_param_uid = param_uid


    def _find_model_param_for_shard(self, shard_param):
        """Find model parameter for a given shard."""
        return self._model_by_shard().get(id(shard_param))

    def _build_grad_copy_context(self):
        """Build cached logging and group-pair context for grad wiring."""
        use_prec_opt = bool(
            getattr(self.config, "use_precision_aware_optimizer_no_fp8_or_ds_fp8", False)
        )
        issue_seen = getattr(self, "_dion_grad_issue_seen", None)
        if issue_seen is None:
            issue_seen = set()
            self._dion_grad_issue_seen = issue_seen

        issue_logger = GradIssueLogger(
            optimizer=getattr(self, "optimizer", None),
            data_parallel_group=getattr(self, "data_parallel_group", None),
            fs_rank=getattr(self, "fs_rank", -1),
            fs_size=getattr(self, "fs_size", -1),
            primary_name_fn=getattr(self, "_param_name", None),
            param_to_name=getattr(self, "param_to_name", None),
            buffers=getattr(self, "buffers", None),
            seen=issue_seen,
        )

        if use_prec_opt:
            grad_group_pairs = (
                (self.model_float16_groups, self.shard_float16_groups),
                (self.model_fp32_groups, self.shard_fp32_groups),
            )
        else:
            grad_group_pairs = (
                (self.model_float16_groups, self.shard_fp32_from_float16_groups),
                (self.model_fp32_groups, self.shard_fp32_groups),
            )

        return {
            "use_prec_opt": use_prec_opt,
            "grad_group_pairs": grad_group_pairs,
            "issue_logger": issue_logger,
            "state_replica_group": getattr(self, "state_replica_group", None),
        }

    def _build_param_copy_context(self):
        """Build shared setup used by model<->main param copy paths."""
        return {
            "use_prec_opt": bool(
                getattr(self.config, "use_precision_aware_optimizer_no_fp8_or_ds_fp8", False)
            ),
            "main_shard_groups": getattr(self, "shard_fp32_from_float16_groups", None),
        }

    def _bucket_param_data(self, model_param: torch.nn.Parameter):
        """Return the canonical bucket.param_data buffer for a model param."""
        if not getattr(model_param, "is_dion_param", False):
            stock_gbuf = getattr(model_param, "_stock_gbuf_index", None)
            stock_bucket = getattr(model_param, "_stock_bucket_index", None)
            if stock_gbuf is not None and stock_bucket is not None:
                return self.buffers[int(stock_gbuf)].buckets[int(stock_bucket)].param_data
        gbuf_index, _, bucket_index = self.model_param_gbuf_map[model_param]
        return self.buffers[gbuf_index].buckets[bucket_index].param_data

    def _check_main_shards(self, main_shard_groups) -> None:
        """Run the one-time shard identity check against optimizer param groups."""
        if not hasattr(self, '_identity_check_done'):
            self._identity_check_done = False

        if not self._identity_check_done and hasattr(self, 'optimizer'):
            self._identity_check_done = True
            check_shard_identity_(
                optimizer=self.optimizer,
                model_float16_groups=self.model_float16_groups,
                main_shard_groups=main_shard_groups,
            )

    def _copy_model_grads_to_main_grads(self):
        """Copy gradients from model params to main params with main_grad priority."""
        dion_copied = 0  # Debug counter

        """Map canonical model-side grads onto optimizer-side shard grads.

        When RP=1 with DO overlap:
        - DO already finished reduced-grad production into canonical model-side buffers.
        - FS group = the current distributed-optimizer shard group.
        - This method should only bridge those canonical reduced grads onto optimizer
          shard params.

        Contract:
        - model side canonical grad: `model_param.main_grad`
        - optimizer side canonical grad: `shard_param.grad` or `shard_param.decoupled_grad`

        Dion parameters still need a custom bridge because the optimizer shard objects are
        separate FP32 tensors and Dion reads custom FS/TP-local grad slices. Non-Dion should
        match stock Megatron-Core DO as closely as possible.
        """
        # Match parent's logging pattern (distrib_optimizer.py:2392)

        if self.is_stub_optimizer:
            return

        if self.ddp_config.use_megatron_fsdp:
            return


        # Copy non-Dion gradients through the standard DO local-shard contract first.
        # Dion params use the custom logical local-shard bridge below.
        max_zero_grad_warnings = 8
        # Avoid per-param host sync in the hot loop: collect max-abs scalars on GPU and
        # evaluate/log in a post-pass once per step.
        dion_grad_max_infos = []  # (model_param, shard_main_param, buffer_idx, bucket_idx, local_shape, expected_len)
        dion_grad_max_tensors = []  # max-abs scalar per Dion param (GPU)

        grad_copy_ctx = self._build_grad_copy_context()
        use_prec_opt = grad_copy_ctx["use_prec_opt"]
        issue_logger = grad_copy_ctx["issue_logger"]
        state_replica_group = grad_copy_ctx["state_replica_group"]
        name_of = issue_logger.param_name
        log_grad_issue = issue_logger.log
        grad_group_pairs = grad_copy_ctx["grad_group_pairs"]
        copy_step = int(getattr(self, "_debug_grad_copy_call_idx", 0)) + 1
        self._debug_grad_copy_call_idx = copy_step

        for model_groups, shard_groups in grad_group_pairs:
            copy_stock_non_dion_grads_(
                model_groups=model_groups,
                shard_groups=shard_groups,
                get_param_range_fn=self._get_model_param_range_map,
                log_grad_issue_fn=log_grad_issue,
                use_prec_opt=use_prec_opt,
                step=copy_step,
            )
            dion_copied = copy_grad_groups_(
                model_groups=model_groups,
                shard_groups=shard_groups,
                get_param_range_fn=self._get_model_param_range_map,
                get_dion_info_fn=self._param_dion_info,
                model_param_gbuf_map=self.model_param_gbuf_map,
                buffers=self.buffers,
                name_fn=name_of,
                log_grad_issue_fn=log_grad_issue,
                use_prec_opt=use_prec_opt,
                state_replica_group=state_replica_group,
                debug_copy_count=dion_copied,
                max_infos=dion_grad_max_infos,
                max_tensors=dion_grad_max_tensors,
                step=copy_step,
            )

        # Post-pass: exact-zero Dion grads + best-effort fallback (log-only, no crash).
        #
        # We avoid per-param host sync (`.item()`) in the hot loop by collecting scalar max-abs
        # tensors on GPU and evaluating them here once per step.
        fix_zero_dion_grads_(
            max_tensors=dion_grad_max_tensors,
            max_infos=dion_grad_max_infos,
            max_zero_grad_warnings=max_zero_grad_warnings,
            name_fn=name_of,
            get_dion_info_fn=self._param_dion_info,
            buffers=self.buffers,
            use_prec_opt=use_prec_opt,
        )

        # Ensure optimizer params have stable classification flags.
        # This is cheap (cached) and avoids relying on per-step grad-copy touching every param.
        self._sync_dion_flags()
        self._maybe_log_target_grad_fingerprints()
        self._release_post_copy_grad_buffers()

    def _release_post_copy_grad_buffers(self) -> None:
        """Release temporary Dion/mixed RS buffers only after optimizer grad projection.

        Stock Megatron-Core DO keeps the canonical local shard available through the
        model->optimizer grad projection stage. Dion custom RS buffers must follow the
        same timing: finish_grad_sync() may complete communication, but the temporary
        local shard buffers cannot be cleared until _copy_model_grads_to_main_grads()
        finishes consuming them.
        """
        if not hasattr(self, "buffers"):
            return

        for buffer in self.buffers:
            for bucket in getattr(buffer, "buckets", []):
                if hasattr(bucket, "dion_grad_buffer"):
                    bucket.dion_grad_buffer = None
                if hasattr(bucket, "dion_rs_local_buffer"):
                    bucket.dion_rs_local_buffer = None
                if hasattr(bucket, "dion_grad_local_view"):
                    bucket.dion_grad_local_view = None
                if hasattr(bucket, "non_dion_grad_local_view"):
                    bucket.non_dion_grad_local_view = None
                if hasattr(bucket, "_memory_reused"):
                    bucket._memory_reused = False

    def _grad_norm_maps(self):
        """Return cached (flat_groups, maps) used by grad-norm/clip and flag propagation.

        The model/shard param group structure is fixed after optimizer construction, so
        we can cache id-based maps to avoid rebuilding every step.
        """
        use_prec_opt = bool(
            getattr(self.config, "use_precision_aware_optimizer_no_fp8_or_ds_fp8", False)
        )
        key = (
            id(getattr(self, "model_float16_groups", None)),
            id(getattr(self, "model_fp32_groups", None)),
            id(getattr(self, "shard_fp32_from_float16_groups", None)),
            id(getattr(self, "shard_float16_groups", None)),
            id(getattr(self, "shard_fp32_groups", None)),
            use_prec_opt,
        )
        cached_key = getattr(self, "_grad_norm_param_maps_key", None)
        cached = getattr(self, "_grad_norm_param_maps", None)
        if cached is not None and cached_key == key:
            return cached

        model_groups = []
        if hasattr(self, "model_float16_groups"):
            model_groups.extend(self.model_float16_groups)
        if hasattr(self, "model_fp32_groups"):
            model_groups.extend(self.model_fp32_groups)

        shard_groups = []
        if use_prec_opt:
            if hasattr(self, "shard_float16_groups"):
                shard_groups.extend(self.shard_float16_groups)
        else:
            if hasattr(self, "shard_fp32_from_float16_groups"):
                shard_groups.extend(self.shard_fp32_from_float16_groups)
        if hasattr(self, "shard_fp32_groups"):
            shard_groups.extend(self.shard_fp32_groups)

        # Build both directions from the actual shard->model backlink.
        # Standard DO semantics are model-side `main_grad` and optimizer-side shard
        # params. The shard object itself is the reliable source of ownership.
        model_to_shard: Dict[int, torch.nn.Parameter] = {}
        shard_to_model: Dict[int, torch.nn.Parameter] = {}
        for model_group, shard_group in zip(model_groups, shard_groups):
            for zipped_model_param, shard_param in zip(model_group, shard_group):
                if shard_param is None:
                    continue
                model_param = getattr(shard_param, "_model_param", None)
                if model_param is None:
                    raise RuntimeError(
                        "[Dion] grad-norm map build requires shard_param._model_param "
                        f"for shard shape={tuple(shard_param.shape)}"
                    )
                if model_param is not zipped_model_param:
                    raise RuntimeError(
                        "[Dion] grad-norm map zip/backlink mismatch: "
                        f"zip_id={id(zipped_model_param)} backlink_id={id(model_param)} "
                        f"shard_shape={tuple(shard_param.shape)}"
                    )
                model_to_shard[id(model_param)] = shard_param
                shard_to_model[id(shard_param)] = model_param

        cached = (model_groups, shard_groups, model_to_shard, shard_to_model)
        self._grad_norm_param_maps = cached
        self._grad_norm_param_maps_key = key
        return cached

    def _grad_norm_items(self):
        """Return cached pre-filtered grad-norm/clip contribution entries."""
        from megatron.core.transformer.module import param_is_not_shared
        from megatron.core import tensor_parallel, parallel_state

        cp_size = parallel_state.get_context_parallel_world_size()
        model_groups, shard_groups, _, _ = self._grad_norm_maps()
        key = (
            self._grad_norm_param_maps_key,
            cp_size,
        )
        cached_key = getattr(self, "_grad_norm_entries_key", None)
        cached = getattr(self, "_grad_norm_entries", None)
        if cached is not None and cached_key == key:
            return cached

        dion_entries, non_dion_entries = build_grad_norm_entries_(
            model_groups=model_groups,
            shard_groups=shard_groups,
            is_unshared=param_is_not_shared,
            is_tp_unique=tensor_parallel.param_is_not_tensor_parallel_duplicate,
        )
        cached = {
            "cp_size": cp_size,
            "dion_entries": dion_entries,
            "non_dion_entries": non_dion_entries,
        }
        self._grad_norm_entries = cached
        self._grad_norm_entries_key = key
        return cached

    def _sync_dion_flags(self) -> None:
        """Ensure optimizer param objects have `is_dion_param` attribute set.

        Most code paths set `shard_param.is_dion_param` during grad-copy, but this
        makes the classification robust even if a param is skipped due to missing grad.
        """
        opt = getattr(self, "optimizer", None)
        if opt is None:
            return
        param_groups = getattr(opt, "param_groups", None)
        if not param_groups:
            return

        key = (id(opt), len(param_groups))
        if getattr(self, "_dion_flag_cache_key", None) == key:
            return

        _, _, _, shard_to_model = self._grad_norm_maps()
        for group in param_groups:
            for p in group.get("params", []):
                mp = shard_to_model.get(id(p))
                if mp is None:
                    continue
                if hasattr(mp, "is_dion_param"):
                    p.is_dion_param = mp.is_dion_param

        self._dion_flag_cache_key = key


    def get_main_grads_for_grad_norm(self):
        """Override to use correct gradient sources for grad norm computation.

        Dion params: use optimizer-side shard grads (`decoupled_grad` / `.grad`) produced
        by the Dion bridge.
        Non-Dion params: use stock optimizer-side shard grads (`decoupled_grad` / `.grad`)
        sliced from canonical `model_param.main_grad`.

        Grad-norm/clip should match the standard global norm semantics.
        Stock Megatron-Core does not apply an extra CP-only correction factor in the
        grad-norm path. CP handling must already be reflected in the reduced grad
        storage before this method runs.
        """
        grads_for_norm = []

        if os.environ.get("DION_DEBUG_GROUPS") and not getattr(self, "_logged_grad_stats_group", False):
            try:
                from megatron.core import parallel_state

                grad_group = self.get_grad_stats_parallel_group()
                cp_group = parallel_state.get_context_parallel_group()
                logger.info(
                    "[DION_GRAD_STATS_GROUP] rank=%s size=%s fs_group_size=%s cp_group_size=%s fs_group_ranks=%s cp_group_ranks=%s",
                    self._global_rank,
                    grad_group.size() if grad_group is not None else None,
                    self.fs_group.size() if getattr(self, "fs_group", None) is not None else None,
                    cp_group.size() if cp_group is not None else None,
                    dist.get_process_group_ranks(self.fs_group) if getattr(self, "fs_group", None) is not None else None,
                    dist.get_process_group_ranks(cp_group) if cp_group is not None else None,
                )
            except Exception as error:
                logger.info("[DION_GRAD_STATS_GROUP_FAILED] err=%r", error)
            self._logged_grad_stats_group = True

        # Debug: log accumulated copy norm from _copy_model_grads_to_main_grads (no file I/O)
        if os.environ.get("DEBUG_PERPARAM") and dist.get_rank() == 0 and hasattr(self, "_debug_copy_norm2"):
            logger.info(
                "[COPY_TOTAL] %d dion params copied, total_norm=%.6f",
                int(getattr(self, "_debug_copy_count", 0)),
                float(getattr(self, "_debug_copy_norm2", 0.0) ** 0.5),
            )
            self._debug_copy_norm2 = 0.0
            self._debug_copy_count = 0

        grad_norm_entries = self._grad_norm_items()
        cp_size = grad_norm_entries["cp_size"]

        if cp_size > 1:
            cp_debug_count = getattr(self, "_cp_gradnorm_debug_count", 0)
            if cp_debug_count < 1:
                setattr(self, "_cp_gradnorm_debug_count", cp_debug_count + 1)
                debug_validate_precomputed_norm_grads_(
                    dion_entries=grad_norm_entries["dion_entries"],
                    non_dion_entries=grad_norm_entries["non_dion_entries"],
                    name_fn=lambda p: self._find_param_name(p) or getattr(p, "_param_name", f"id_{id(p)}"),
                )

        debug_gradnorm_every = os.environ.get("DION_DEBUG_GRADNORM_EVERY")
        if debug_gradnorm_every:
            try:
                every_n = int(debug_gradnorm_every)
            except ValueError:
                every_n = -1
            call_idx = int(getattr(self, "_debug_gradnorm_call_idx", 0)) + 1
            self._debug_gradnorm_call_idx = call_idx
            if every_n > 0 and call_idx % every_n == 0:
                log_grad_norm_contributors_(
                    dion_entries=grad_norm_entries["dion_entries"],
                    non_dion_entries=grad_norm_entries["non_dion_entries"],
                    name_fn=lambda p: self._find_param_name(p)
                    or getattr(p, "_param_name", f"id_{id(p)}"),
                    step=call_idx,
                    topk=int(os.environ.get("DION_DEBUG_GRADNORM_TOPK", "12")),
                )

        append_precomputed_norm_grads_(
            grads_for_norm=grads_for_norm,
            dion_entries=grad_norm_entries["dion_entries"],
            non_dion_entries=grad_norm_entries["non_dion_entries"],
            name_fn=lambda p: self._find_param_name(p)
            or getattr(p, "_param_name", f"id_{id(p)}"),
            step=int(getattr(self, "_debug_gradnorm_call_idx", 0)) or None,
        )

        if os.environ.get("DION_DEBUG_GRADPTR_RANGES"):
            try:
                ptr_to_names = {}
                for entry in grad_norm_entries["dion_entries"]:
                    grad = get_optimizer_grad_(entry.shard_param)
                    if grad is None or grad.numel() == 0:
                        continue
                    ptr_to_names.setdefault(grad.data_ptr(), []).append(
                        ("dion", entry.model_param, entry.shard_param)
                    )
                for entry in grad_norm_entries["non_dion_entries"]:
                    grad = get_optimizer_grad_(entry.shard_param)
                    if grad is None or grad.numel() == 0:
                        continue
                    ptr_to_names.setdefault(grad.data_ptr(), []).append(
                        ("non_dion", entry.model_param, entry.shard_param)
                    )
                for ptr, items in ptr_to_names.items():
                    if len(items) < 2:
                        continue
                    details = []
                    for kind, model_param, shard_param in items:
                        try:
                            range_map = self._get_model_param_range_map(model_param)
                            local_range = range_map["param"]
                            world_range = range_map["gbuf_world"]
                        except Exception:
                            local_range = None
                            world_range = None
                        try:
                            gbuf_index, _, bucket_index = self.model_param_gbuf_map[model_param]
                        except Exception:
                            gbuf_index, bucket_index = None, None
                        details.append(
                            {
                                "kind": kind,
                                "param": self._find_param_name(model_param) or getattr(model_param, "_param_name", f"id_{id(model_param)}"),
                                "shard_shape": tuple(shard_param.shape),
                                "local_range": None if local_range is None else (local_range.start, local_range.end),
                                "world_range": None if world_range is None else (world_range.start, world_range.end),
                                "bucket": (gbuf_index, bucket_index),
                            }
                        )
                    logger.error("[DION_GRADPTR_RANGE_DUP] ptr=%s details=%s", ptr, details)
            except Exception as exc:
                logger.error("[DION_GRADPTR_RANGE_DUP_FAILED] exc=%r", exc)

        return grads_for_norm

    def clip_grad_norm(self, clip_grad: float) -> float:
        """
        Override to apply gradient clipping with correct sharding schemes.

        Both Dion and non-Dion params should clip the optimizer-side shard grads
        (`decoupled_grad` / `.grad`), matching standard Megatron-Core DO semantics
        as closely as possible. Dion-specific communication is only in how those
        shard grads are produced, not in how clip-grad consumes them.
        """
        from megatron.core.optimizer.clip_grads import get_grad_norm_fp32

        # Compute global grad norm using all shards (Dion + non-Dion params)
        grads_for_norm = self.get_main_grads_for_grad_norm()

        grad_norm = get_grad_norm_fp32(
            grads_for_norm, grad_stats_parallel_group=self.get_grad_stats_parallel_group()
        )

        # Always-on diagnostics: exact zero global grad norm is a strong signal that
        # gradients were not wired correctly for this step.
        if grad_norm == 0.0:
            log_zero_global_grad_norm_(self.optimizer, len(grads_for_norm))

        # Apply clipping
        if clip_grad > 0.0 and grad_norm > 0.0:
            clip_coeff = clip_grad / (grad_norm + 1.0e-6)
            if clip_coeff < 1.0:
                grad_norm_entries = self._grad_norm_items()
                clip_precomputed_grad_groups_(
                    dion_entries=grad_norm_entries["dion_entries"],
                    non_dion_entries=grad_norm_entries["non_dion_entries"],
                    clip_coeff=clip_coeff,
                )

        return grad_norm

    def prepare_grads(self) -> bool:
        """
        Match stock Megatron-Core prepare_grads semantics.

        The standard training loop already calls `finalize_model_grads()`, which in turn
        calls `model_chunk.finish_grad_sync()` before `optimizer.step()`. Repeating
        `finish_grad_sync()` here is incorrect for the multi-instance path because mixed
        non-Dion RS-local buffers have already been flushed and cleared by the first call.
        """
        return super().prepare_grads()

    def _copy_model_params_to_main_params(self, state_dict=None):
        """Copy model params to main params with FS sharding awareness.

        Handles FS-sharded parameters correctly when loading from checkpoint.
        """
        from .cpu_offloading import HybridDeviceOptimizer
        copy_ctx = self._build_param_copy_context()
        use_prec_opt = copy_ctx["use_prec_opt"]
        main_shard_groups = copy_ctx["main_shard_groups"]
        if isinstance(self.optimizer, HybridDeviceOptimizer):
            self.optimizer.update_fp32_param_by_new_param()
            return

        if self.ddp_config.use_megatron_fsdp:
            return

        # Precision-aware optimizer early return
        if use_prec_opt:
            return

        if state_dict is not None:
            # Build mapping for state dict params
            model_param_to_state_dict_param_map = self._build_model_param_to_state_dict_param_map(
                state_dict
            )

        if state_dict is not None:
            get_source_param = lambda model_param: model_param_to_state_dict_param_map[model_param]
        else:
            get_source_param = lambda model_param: model_param
        get_param_range = lambda model_param: self._get_model_param_range_map(model_param)["param"]

        copy_group_params_to_main_shards_(
            model_groups=self.model_float16_groups,
            shard_main_groups=main_shard_groups,
            get_source_param_fn=get_source_param,
            get_param_range_fn=get_param_range,
            get_dion_info_fn=self._param_dion_info,
        )
        copy_group_params_to_main_shards_(
            model_groups=self.model_fp32_groups,
            shard_main_groups=self.shard_fp32_groups,
            get_source_param_fn=get_source_param,
            get_param_range_fn=get_param_range,
            get_dion_info_fn=self._param_dion_info,
        )

    def _copy_main_params_to_model_params(self):
        """Copy parameters with efficient batch flattening for 2D params.

        Copy updated optimizer shards into local model-param shards.

        The parent DistributedOptimizer step path immediately launches param all-gather
        synchronously when `overlap_param_gather=False`, and DDP launches it before the
        next forward when `overlap_param_gather=True`. In the training path we can
        therefore defer full-param materialization and keep local shards canonical here.
        """
        pp_world_size = parallel_state.get_pipeline_model_parallel_world_size()
        rank = dist.get_rank() if dist.is_initialized() else 0
        if pp_world_size > 1:
            logger.info("[DION_PP_DEBUG] rank=%s before copy_main_to_model", rank)

        if self.is_stub_optimizer:
            return
        copy_ctx = self._build_param_copy_context()
        use_prec_opt = copy_ctx["use_prec_opt"]
        main_shard_groups = copy_ctx["main_shard_groups"]

        # FSDP early return
        if self.ddp_config.use_megatron_fsdp:
            for model_chunk in self.model_chunks:
                model_chunk.param_and_grad_buffer.copy_main_weights_to_model_weights()
            return

        # Precision-aware optimizer early return
        if use_prec_opt:
            return

        self._mark_buckets_full_param_ready(False)

        # Copy updated optimizer shards (FP32) to model params (BF16)
        # Use model_float16_groups/shard_fp32_from_float16_groups directly
        # to avoid object identity mismatch in shard lookups

        zero_range_warned = getattr(self, '_zero_range_warned', 0)

        self._check_main_shards(main_shard_groups)

        apply_stock_non_dion_shards_(
            model_groups=self.model_float16_groups,
            shard_groups=main_shard_groups,
            get_bucket_param_data_fn=self._bucket_param_data,
        )
        _, zero_range_warned = apply_group_shards_to_model_params_(
            model_groups=self.model_float16_groups,
            shard_groups=main_shard_groups,
            shard16_groups=self.shard_float16_groups,
            get_data_shard_fn=self._get_data_shard,
            get_param_range_map_fn=self._get_model_param_range_map,
            get_dion_info_fn=self._param_dion_info,
            get_bucket_param_data_fn=self._bucket_param_data,
            zero_range_warned=zero_range_warned,
        )

        self._zero_range_warned = zero_range_warned

        # Keep model_fp32_groups current as local shards as well when deferring
        # the expensive full-param restore path.
        apply_stock_non_dion_shards_(
            model_groups=self.model_fp32_groups,
            shard_groups=self.shard_fp32_groups,
            get_bucket_param_data_fn=self._bucket_param_data,
        )
        _, zero_range_warned = apply_group_shards_to_model_params_(
            model_groups=self.model_fp32_groups,
            shard_groups=self.shard_fp32_groups,
            shard16_groups=self.shard_fp32_groups,
            get_data_shard_fn=self._get_data_shard,
            get_param_range_map_fn=self._get_model_param_range_map,
            get_dion_info_fn=self._param_dion_info,
            get_bucket_param_data_fn=self._bucket_param_data,
            zero_range_warned=zero_range_warned,
        )
        self._zero_range_warned = zero_range_warned

        if os.getenv("DION_DEBUG_COMPARE_FS_PACK", "0") == "1":
            self._debug_fs_pack_compare_armed = True

        if pp_world_size > 1:
            logger.info("[DION_PP_DEBUG] rank=%s after copy_main_to_model norestore", rank)
        return
        if pp_world_size > 1:
            logger.info("[DION_PP_DEBUG] rank=%s after copy_main_to_model restore", rank)



    def _refresh_param_groups(self):
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

        finish_bucket_group_grad_sync_(self.per_model_bucket_groups)

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

        release_rs_buffers_(self.buffers)

    @torch.no_grad()
    def step_with_ready_grads(self):
        """Step optimizer with extra PP debug logging."""
        pp_world_size = parallel_state.get_pipeline_model_parallel_world_size()
        rank = dist.get_rank() if dist.is_initialized() else 0
        if pp_world_size > 1:
            logger.info("[DION_PP_DEBUG] rank=%s before optimizer_step_with_ready_grads", rank)

        timers = self.config.timers
        if timers is not None:
            timers('optimizer-inner-step', log_level=1).start(
                barrier=self.config.barrier_with_L1_time
            )
        if not self.is_stub_optimizer:
            if pp_world_size > 1:
                logger.info("[DION_PP_DEBUG] rank=%s before inner_optimizer_step", rank)
            self.optimizer.step()
            if pp_world_size > 1:
                logger.info("[DION_PP_DEBUG] rank=%s after inner_optimizer_step", rank)
        if timers is not None:
            timers('optimizer-inner-step').stop()

        if timers is not None:
            timers('optimizer-copy-main-to-model-params', log_level=1).start(
                barrier=self.config.barrier_with_L1_time
            )
        if not self.is_stub_optimizer:
            if self.config.reuse_grad_buf_for_mxfp8_param_ag:
                if not self.config.overlap_param_gather:
                    self._copy_main_params_to_param_buffer()
            else:
                self._copy_main_params_to_model_params()
        if timers is not None:
            timers('optimizer-copy-main-to-model-params').stop()

        if pp_world_size > 1:
            logger.info("[DION_PP_DEBUG] rank=%s after optimizer_step_with_ready_grads", rank)
        return True

    def sharded_state_dict(
        self,
        model_sharded_state_dict=None,
        is_loading: bool = False,
        sharding_type=None,
        metadata=None,
    ):
        """Build optimizer checkpoint state with stock common-state structure.

        Legacy checkpoint IO remains on the inherited stock path.
        For distributed checkpoint, keep the stock common optimizer-state layout
        and store only Dion-specific per-param state as an extra payload keyed by
        deterministic param name.
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

        # Part 2: Dion-only persistent parameter state. Keep it separate from the
        # stock distributed-optimizer param_state protocol because Dion carries
        # arbitrary-shape tensors such as Q that the stock sharding formats do not
        # represent.
        dion_param_state = build_named_dion_param_state(
            self.optimizer.param_groups,
            self.optimizer.state,
            self._shard_name,
        )
        state_dict['dion_param_state'] = ShardedObject(
            f'{base_key}.dion_param_state', dion_param_state, (1,), (0,),
            replica_id=replica_id,
        )

        return state_dict

    def load_state_dict(self, state_dict):
        """Load optimizer checkpoint state with stock common-state outer protocol."""
        dion_param_state = state_dict.get('dion_param_state', None)
        if dion_param_state is None and state_dict.get('param_state_sharding_type') == 'dion_non_reshardable':
            dion_param_state = state_dict.get('param_state', None)

        if dion_param_state is None:
            super().load_state_dict(state_dict)
            return

        self._ensure_dion_state_initialized_for_load()

        common_state_dict = {
            k: v
            for k, v in state_dict.items()
            if k not in ('dion_param_state', 'param_state', 'param_state_sharding_type')
        }
        super().load_state_dict(common_state_dict)

        if not isinstance(dion_param_state, dict) or len(dion_param_state) == 0:
            raise RuntimeError("[Dion] distributed checkpoint missing Dion param state payload")

        restore_summary = restore_named_dion_param_state_(
            param_groups=self.optimizer.param_groups,
            optimizer_state=self.optimizer.state,
            get_param_name_fn=self._shard_name,
            name_to_state=dion_param_state,
        )
        if restore_summary["unnamed"] > 0:
            logger.warning(
                '[Dion] Restored %s Dion param states from distributed checkpoint '
                '(no_payload_entry=%s, unnamed=%s)',
                restore_summary["restored"],
                restore_summary["no_payload_entry"],
                restore_summary["unnamed"],
            )
        else:
            logger.info(
                '[Dion] Restored %s Dion param states from distributed checkpoint '
                '(no_payload_entry=%s, unnamed=%s)',
                restore_summary["restored"],
                restore_summary["no_payload_entry"],
                restore_summary["unnamed"],
            )

        # Sync fp32 main params from bf16 model params.
        # The fp32 copies still hold stale values from __init__ (cloned before checkpoint
        # loading). Without this, the first optimizer step would corrupt model params.
        # bf16->fp32 cast has ~1e-3 precision loss, acceptable for training resumption.
        self._copy_model_params_to_main_params()

    def _ensure_dion_state_initialized_for_load(self):
        """Initialize Dion optimizer state tensors before calling stock common-state load."""
        init_state_fn = getattr(self.optimizer, "_init_state", None)
        if init_state_fn is None:
            return
        for group in self.optimizer.param_groups:
            for param in group["params"]:
                state = self.optimizer.state.get(param)
                if state is None:
                    self.optimizer.state[param] = {}
                    state = self.optimizer.state[param]
                if len(state) == 0:
                    init_state_fn(param, state, group)

    def _shard_name(self, shard_param):
        """Get deterministic model param name for a shard param.

        Traces shard_param → model_param → param_to_name mapping.
        Returns None if name cannot be found.
        """
        param_name = self._shard_name_cache().get(id(shard_param))
        if param_name is not None:
            return param_name

        model_param = getattr(shard_param, '_model_param', None)
        if model_param is None:
            _, _, _, shard_to_model = self._grad_norm_maps()
            model_param = shard_to_model.get(id(shard_param))
        if model_param is None:
            model_param = self._model_by_shard().get(id(shard_param))
        if model_param is None:
            return None
        return self._model_name_cache().get(id(model_param))

    def offload_to_cpu(self):
        """Clean up Dion-specific buffers during offload.

        Standard DO pattern handles most cleanup via:
        1. Buffer level: param_and_grad_buffer.offload_to_cpu()
        2. Optimizer state: move_optimizer("cpu")
        3. Q buffers: Auto-cleared when device changes
        """
        if hasattr(self, 'optimizer') and hasattr(self.optimizer, 'offload_to_cpu'):
            self.optimizer.offload_to_cpu()
