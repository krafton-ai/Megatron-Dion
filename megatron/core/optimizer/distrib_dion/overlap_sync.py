"""Overlap-sync helpers for the Dion distributed optimizer.

These helpers preserve the existing wait/release order for async grad reduce
and param sync integration. They only factor repeated control flow out of the
main optimizer wrapper.
"""

from __future__ import annotations


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


def finish_bucket_group_grad_sync_(per_model_bucket_groups) -> None:
    """Finish grad sync on all registered bucket groups."""
    for bucket_groups in per_model_bucket_groups.values():
        for bucket_group in bucket_groups:
            finish_grad_sync = getattr(bucket_group, "finish_grad_sync", None)
            if finish_grad_sync is not None:
                finish_grad_sync()


def release_rs_buffers_(buffers) -> None:
    """Release cached RS buffers after async grad sync completes."""
    for buffer in buffers or []:
        for bucket in getattr(buffer, "buckets", []):
            if hasattr(bucket, "dion_grad_buffer") and bucket.dion_grad_buffer is not None:
                bucket.dion_grad_buffer = None
            if hasattr(bucket, "_cached_rs_pack_buffer") and bucket._cached_rs_pack_buffer is not None:
                bucket._cached_rs_pack_buffer = None
