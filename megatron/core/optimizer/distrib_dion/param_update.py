"""Parameter update helpers for the Dion distributed optimizer.

These helpers are mechanical refactors only. They preserve the same optimizer
state, copy order, and diagnostics as the original inlined code.
"""

from __future__ import annotations

import logging


logger = logging.getLogger(__name__)


def check_shard_identity_(
    *,
    optimizer,
    model_float16_groups,
    main_shard_groups,
) -> None:
    """Verify shard params still match optimizer param group objects by data_ptr."""
    mismatch_count = 0
    match_count = 0

    opt_param_by_ptr = {}
    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            if param is None or param.numel() == 0:
                continue
            opt_param_by_ptr[param.data_ptr()] = param

    for group_index, (model_group, shard_group) in enumerate(
        zip(model_float16_groups, main_shard_groups)
    ):
        for param_index, (_, shard_param) in enumerate(zip(model_group, shard_group)):
            if shard_param is None:
                continue
            if shard_param.numel() == 0:
                continue

            shard_ptr = shard_param.data_ptr()
            opt_param = opt_param_by_ptr.get(shard_ptr)

            if opt_param is None:
                if mismatch_count < 5:
                    logger.error(
                        "[IDENTITY MISMATCH] main_shard_groups[%s][%s] (shape=%s, ptr=%s) NOT FOUND in optimizer.param_groups! Dion updates will be LOST!",
                        group_index,
                        param_index,
                        shard_param.shape,
                        shard_ptr,
                    )
                mismatch_count += 1
            elif opt_param is not shard_param:
                if mismatch_count < 5:
                    logger.error(
                        "[Dion] main_shard_groups[%s][%s] object mismatch: same data_ptr=%s but different object. id(shard)=%s, id(opt)=%s",
                        group_index,
                        param_index,
                        shard_ptr,
                        id(shard_param),
                        id(opt_param),
                    )
                mismatch_count += 1
            else:
                match_count += 1

    if mismatch_count > 0:
        raise RuntimeError(
            "[Dion] main_shard_groups identity mismatch with optimizer.param_groups: "
            f"mismatches={mismatch_count} matched={match_count}"
        )
