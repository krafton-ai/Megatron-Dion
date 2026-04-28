from megatron.core.optimizer.distrib_dion import row_child


def test_row_child_layout_reuses_parent_group(monkeypatch):
    parent_group = object()
    calls = []

    monkeypatch.setattr(
        row_child.dist,
        "get_process_group_ranks",
        lambda group: (3, 4),
    )

    layout = row_child.resolve_row_child_layout(
        parent_group=parent_group,
        parent_world_size=2,
        parent_rank=1,
        child_rows=4,
        child_ranges=((0, 2), (2, 4)),
        label="TP",
        detail="param_uid=x",
        error_prefix="TEST_CHILD",
        create_group=False,
        make_group=lambda ranks, create_group: calls.append((ranks, create_group)),
    )

    assert layout.group is parent_group
    assert layout.world_size == 2
    assert layout.rank == 1
    assert layout.start_idx == 2
    assert layout.end_idx == 4
    assert layout.row_shard_sizes == (2, 2)
    assert calls == []


def test_row_child_layout_compacts_child_group_for_nonmember(monkeypatch):
    parent_group = object()
    child_group = object()
    calls = []

    monkeypatch.setattr(
        row_child.dist,
        "get_process_group_ranks",
        lambda group: (10, 11, 12),
    )

    def make_group(ranks, create_group):
        calls.append((ranks, create_group))
        return child_group

    layout = row_child.resolve_row_child_layout(
        parent_group=parent_group,
        parent_world_size=3,
        parent_rank=0,
        child_rows=5,
        child_ranges=(None, (0, 3), (3, 5)),
        label="FS",
        detail="param_uid=x",
        error_prefix="TEST_CHILD",
        create_group=True,
        make_group=make_group,
    )

    assert layout.group is child_group
    assert layout.world_size == 2
    assert layout.rank == -1
    assert layout.start_idx == -1
    assert layout.end_idx == -1
    assert layout.row_shard_sizes == (3, 2)
    assert calls == [((11, 12), True)]
