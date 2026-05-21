"""Dion backend for the matrix distributed optimizer."""

from ..matrix.backend import MatrixBackend, MatrixStateSpec


class DionBackend(MatrixBackend):
    """Dion policy behind the matrix backend boundary."""

    name = "dion"
    supports_fs = True
    supports_rp = True
    supports_tp = True
    supports_expert_parallel = True
    supports_split_qkv = True
    supports_split_qkvg = True
    supports_split_linear = True

    def refresh_state(self, adapter, *, param, state, optim_group, dist_meta) -> None:
        adapter._refresh_dion_step_metadata(
            param=param,
            optimizer_state=state,
            optim_group=optim_group,
            dist_meta=dist_meta,
        )

    def use_matrix(self, adapter, *, param, state, optim_group, dist_meta) -> bool:
        return adapter._should_use_distributed_dion_update(
            param,
            state,
            optim_group,
            dist_meta,
        )

    def split_children(self, adapter, *, param, grad, state, optim_group, config, dist_meta):
        return adapter._expand_split_dion_params(
            param=param,
            grad=grad,
            optimizer_state=state,
            optim_group=optim_group,
            config=config,
            dist_meta=dist_meta,
        )

    def sync_state(self, adapter, matrix_params) -> None:
        adapter._sync_dion_state(matrix_params)

    def build_batches(self, adapter, matrix_params):
        return adapter._build_dion_batches(matrix_params)

    def state_spec(self) -> MatrixStateSpec:
        return MatrixStateSpec(
            backend=self.name,
            state_keys=(
                "momentum",
                "Q",
                "r",
                "local_shape",
                "global_shape",
                "per_expert_global_shape",
                "qkv_split_shapes",
                "qkvg_split_shapes",
                "linear_split_rows",
            ),
        )
