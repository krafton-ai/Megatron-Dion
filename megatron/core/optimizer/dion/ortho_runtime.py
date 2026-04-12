"""Package-private orthogonalization runtime helpers for MegatronDion."""

from typing import List, Optional

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Shard

from .ortho import (
    distributed_orthogonalize_dtensor_exact,
    logical_sketch_keys,
    make_local_sketch,
    make_seeded_sketch,
    orthogonalize,
)
from .utils import format_meta_id


def distributed_orthogonalize(
    optimizer,
    P_batch: torch.Tensor,
    *,
    ortho_group: Optional[torch.distributed.ProcessGroup],
    orthogonalize_mesh=None,
    oversample: float = 1.25,
    dist_metas: Optional[List] = None,
    real_batch_size: Optional[int] = None,
) -> torch.Tensor:
    """Distributed orthogonalization matching `dion_reference.py::distributed_orthogonalize()`."""
    del real_batch_size
    batch_size = P_batch.size(0)
    m_shard_local = P_batch.size(1)
    r = P_batch.size(2)
    original_dtype = P_batch.dtype

    if ortho_group is not None:
        ortho_world_size = dist.get_world_size(ortho_group)
    else:
        ortho_world_size = 1

    if ortho_group is None or ortho_world_size <= 1:
        result = torch.empty_like(P_batch)
        for index in range(batch_size):
            result[index] = optimizer._orthogonalize(P_batch[index], rcqr_oversample=oversample)
        return result

    if orthogonalize_mesh is None:
        raise RuntimeError(
            "[DION_MISSING_ORTHO_DEVICE_MESH] "
            f"step={optimizer._step_count} rank={optimizer._global_rank}"
        )
    p_dtensor = DTensor.from_local(
        P_batch,
        device_mesh=orthogonalize_mesh,
        placements=(Shard(P_batch.ndim - 2),),
    )
    batch_dist_meta_ids = (
        [format_meta_id(dist_meta) for dist_meta in dist_metas[:batch_size]]
        if dist_metas is not None
        else None
    )
    p_out = distributed_orthogonalize_dtensor_exact(
        p_dtensor,
        oversample=oversample,
        shard_mesh_dim=0,
        logical_seed_keys=sketch_keys_for_update(
            optimizer,
            dist_metas=dist_metas[:batch_size] if dist_metas is not None else None,
            tag="logical_local",
        ),
        batch_meta_ids=batch_dist_meta_ids,
    ).to_local().contiguous()
    return p_out.to(original_dtype)


def make_seeded_sketch_for_update(
    optimizer,
    *,
    dist_metas: Optional[List],
    tag: str,
):
    """Build a topology-independent sketch generator for one optimizer step."""
    return make_seeded_sketch(
        dist_metas=dist_metas,
        tag=tag,
        step_count=optimizer._step_count,
        format_meta_id=format_meta_id,
    )


def sketch_keys_for_update(
    optimizer,
    *,
    dist_metas: Optional[List],
    tag: str,
) -> Optional[List[object]]:
    """Return topology-invariant logical sketch ids for one optimizer step."""
    return logical_sketch_keys(
        dist_metas=dist_metas,
        tag=tag,
        step_count=optimizer._step_count,
    )


def make_local_sketch_for_update(
    optimizer,
    *,
    dist_metas: Optional[List],
):
    """Return the topology-invariant local sketch contract for one optimizer step."""
    return make_local_sketch(
        dist_metas=dist_metas,
        step_count=optimizer._step_count,
        format_meta_id=format_meta_id,
    )


def orthogonalize_local_slice(
    optimizer,
    P_slice: torch.Tensor,
    dist_metas: Optional[List] = None,
    *,
    tag: str = "local",
    sketch_tag: Optional[str] = None,
    sketch_fn=None,
    use_seeded_sketch: bool = True,
) -> torch.Tensor:
    """Local orthogonalization path for a batched slice."""
    if P_slice.size(0) == 0:
        return P_slice

    if use_seeded_sketch and sketch_fn is None and dist_metas is not None:
        sketch_fn = make_seeded_sketch_for_update(
            optimizer,
            dist_metas=dist_metas,
            tag=sketch_tag if sketch_tag is not None else tag,
        )

    P_ortho_slice = orthogonalize(
        P_slice,
        rcqr_oversample=optimizer.defaults['rcqr_oversample'],
        sketch_fn=sketch_fn,
    ).to(torch.float32)
    return P_ortho_slice.to(P_slice.dtype)


def orthogonalize_local_matrix_batch(
    optimizer,
    P_slice: torch.Tensor,
    dist_metas: Optional[List] = None,
    *,
    tag: str = "local_matrix",
) -> torch.Tensor:
    """Regular-Tensor local orthogonalization for one logical full matrix."""
    return orthogonalize_local_slice(
        optimizer,
        P_slice,
        dist_metas=dist_metas,
        tag=tag,
        sketch_tag="logical_local",
    )
