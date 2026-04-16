import hashlib
import os
from contextlib import contextmanager
from typing import Callable

import torch

from megatron.core import parallel_state
from megatron.core.tensor_parallel.layers import _initialize_affine_weight_cpu


def _pp_invariant_base_seed() -> int:
    """Return a topology-invariant base seed for model parameter initialization."""

    pp_rank = 0
    if torch.distributed.is_initialized():
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    return (int(torch.initial_seed()) - 100 * int(pp_rank)) % (2**63 - 1)


def topology_invariant_expert_init_seed(
    *, layer_number: int, linear_tag: str, global_expert_idx: int, kind: str
) -> int:
    """Return a stable per-parameter seed for one logical expert tensor."""

    key = (
        f"moe_expert_init|layer={int(layer_number)}|linear={linear_tag}"
        f"|expert={int(global_expert_idx)}|kind={kind}"
    )
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    offset = int.from_bytes(digest, byteorder="little", signed=False) % (2**63 - 1)
    seed = (_pp_invariant_base_seed() + offset) % (2**63 - 1)
    return seed if seed != 0 else 1


@contextmanager
def _fork_cpu_rng(seed: int):
    cpu_rng_state = torch.get_rng_state()
    torch.manual_seed(int(seed))
    try:
        yield
    finally:
        torch.set_rng_state(cpu_rng_state)


@torch.no_grad()
def reinitialize_partitioned_expert_weight_(
    *,
    weight: torch.Tensor,
    full_rows: int,
    full_cols: int,
    partition_dim: int,
    init_method: Callable,
    params_dtype: torch.dtype,
    tp_rank: int,
    tp_world_size: int,
    layer_number: int,
    linear_tag: str,
    global_expert_idx: int,
) -> None:
    """Reinitialize one expert weight shard from a topology-invariant logical seed."""

    if partition_dim not in (0, 1):
        raise RuntimeError(
            f"[MoE] invalid partition_dim={partition_dim} for expert init "
            f"layer={layer_number} linear={linear_tag} expert={global_expert_idx}"
        )

    per_partition_size = int(weight.shape[partition_dim])
    seed = topology_invariant_expert_init_seed(
        layer_number=layer_number,
        linear_tag=linear_tag,
        global_expert_idx=global_expert_idx,
        kind="weight",
    )
    with _fork_cpu_rng(seed):
        _initialize_affine_weight_cpu(
            weight=weight,
            output_size=full_rows,
            input_size=full_cols,
            per_partition_size=per_partition_size,
            partition_dim=partition_dim,
            init_method=init_method,
            params_dtype=params_dtype,
            rank=int(tp_rank),
            world_size=int(tp_world_size),
            skip_set_tensor_parallel_attributes=True,
        )

    if (
        os.getenv("VLM_DEBUG_EXPERT_INIT", "0") == "1"
        and int(layer_number) == 1
        and int(global_expert_idx) == 0
        and linear_tag in ("fc1", "fc2")
    ):
        rank = -1
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        flat = weight.detach().float().contiguous().flatten()
        preview_limit = min(int(os.getenv("VLM_DEBUG_EXPERT_INIT_PREVIEW", "8")), flat.numel())
        preview = ",".join(f"{float(x):.8f}" for x in flat[:preview_limit].tolist())
        weighted = 0.0
        if flat.numel() > 0:
            weights = torch.arange(1, flat.numel() + 1, dtype=flat.dtype, device=flat.device)
            weighted = float((flat * weights).sum().item())
        print(
            "VLM_EXPERT_INIT "
            f"rank={rank} "
            f"layer={int(layer_number)} "
            f"linear={linear_tag} "
            f"expert={int(global_expert_idx)} "
            f"shape={tuple(weight.shape)} "
            f"partition_dim={int(partition_dim)} "
            f"full_rows={int(full_rows)} "
            f"full_cols={int(full_cols)} "
            f"tp_rank={int(tp_rank)} "
            f"tp_world={int(tp_world_size)} "
            f"seed={int(seed)} "
            f"sum={float(flat.sum().item()):.8f} "
            f"abs={float(flat.abs().sum().item()):.8f} "
            f"norm={float(torch.linalg.vector_norm(flat).item()):.8f} "
            f"weighted_sum={weighted:.8f} "
            f"preview=[{preview}]",
            flush=True,
        )
