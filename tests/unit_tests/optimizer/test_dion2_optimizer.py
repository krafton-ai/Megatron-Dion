import torch

from megatron.core.optimizer.dion2.algorithm import TensorParallelDion2
from megatron.core.optimizer.dion2.kernels import dion2_compute_update, resolve_select_dim
from megatron.core.optimizer.dion2.types import Dion2ParamConfig


def test_dion2_row_selection_decays_only_selected_rows():
    grad = torch.arange(24, dtype=torch.float32).view(4, 6) + 1
    momentum = torch.zeros_like(grad)
    config = Dion2ParamConfig(
        fraction=0.5,
        ef_decay=0.5,
        adjust_lr=None,
        select_dim="row",
        ns_backend="standard",
        coefficient_type="simple",
        num_ns_steps=1,
    )

    update = dion2_compute_update(
        grad=grad,
        momentum=momentum,
        config=config,
        global_shape=(4, 6),
    )

    nonzero_rows = torch.nonzero(update.abs().sum(dim=1) > 0).flatten().tolist()
    unselected_rows = [index for index in range(4) if index not in nonzero_rows]
    assert nonzero_rows == [2, 3]
    assert torch.allclose(momentum[nonzero_rows], grad[nonzero_rows] * 0.5)
    assert torch.allclose(momentum[unselected_rows], grad[unselected_rows])
    assert torch.count_nonzero(update[unselected_rows]) == 0


def test_dion2_column_selection_decays_only_selected_columns():
    grad = torch.arange(24, dtype=torch.float32).view(6, 4) + 1
    momentum = torch.zeros_like(grad)
    config = Dion2ParamConfig(
        fraction=0.5,
        ef_decay=0.5,
        adjust_lr=None,
        select_dim="col",
        ns_backend="standard",
        coefficient_type="simple",
        num_ns_steps=1,
    )

    update = dion2_compute_update(
        grad=grad,
        momentum=momentum,
        config=config,
        global_shape=(6, 4),
    )

    nonzero_cols = torch.nonzero(update.abs().sum(dim=0) > 0).flatten().tolist()
    unselected_cols = [index for index in range(4) if index not in nonzero_cols]
    assert nonzero_cols == [2, 3]
    assert torch.allclose(momentum[:, nonzero_cols], grad[:, nonzero_cols] * 0.5)
    assert torch.allclose(momentum[:, unselected_cols], grad[:, unselected_cols])
    assert torch.count_nonzero(update[:, unselected_cols]) == 0


def test_dion2_auto_selects_only_sharded_axis_when_unique():
    assert (
        resolve_select_dim(
            local_shape=(2, 8),
            global_shape=(4, 8),
            fs_shard_dim=0,
            fs_world_size=2,
            tp_shard_dim=-1,
            tp_world_size=1,
            requested="auto",
        )
        == 0
    )
    assert (
        resolve_select_dim(
            local_shape=(8, 2),
            global_shape=(8, 4),
            fs_shard_dim=-1,
            fs_world_size=1,
            tp_shard_dim=1,
            tp_world_size=2,
            requested="auto",
        )
        == 1
    )


def test_dion2_optimizer_step_updates_matrix_state():
    param = torch.nn.Parameter(torch.randn(5, 7, dtype=torch.float32))
    param.matrix_optimizer_ready = True
    param.grad = torch.randn_like(param)
    before = param.detach().clone()

    optimizer = TensorParallelDion2(
        [param],
        lr=0.01,
        fraction=0.4,
        ef_decay=0.95,
        weight_decay=0.1,
        adjust_lr=None,
        ns_backend="standard",
        coefficient_type="simple",
        num_ns_steps=1,
    )
    optimizer.step()

    state = optimizer.state[param]
    assert "momentum" in state
    assert tuple(state["momentum"].shape) == tuple(param.shape)
    assert torch.isfinite(param).all()
    assert torch.isfinite(state["momentum"]).all()
    assert not torch.equal(param, before)
