from types import SimpleNamespace

import torch

from megatron.core.optimizer.distrib_dion import parameter as dion_parameter


def test_mixed_param_gather_cache_keeps_async_tensors_until_wait(monkeypatch):
    class FakeGroup:
        ranks = (0, 1)

        def size(self):
            return len(self.ranks)

        def rank(self):
            return 1

    group = FakeGroup()
    dion_param = torch.nn.Parameter(torch.empty(2, 1))
    standard_param = torch.nn.Parameter(torch.empty(6))
    dion_entry = dion_parameter.DionShardEntry(
        param=dion_param,
        shard_layout=dion_parameter.DionShardLayout(
            local_shape=(2, 1),
            global_shape=(2, 1),
            fs_shard_dim=0,
            start_idx=0,
            end_idx=2,
        ),
        size_per_rank=1,
        shard_capacity=2,
        shard_offset=0,
        canonical_bucket_start=0,
        canonical_bucket_end=2,
        canonical_rank_flat_segments=(((0, 1),), ((1, 2),)),
        grad_rank_flat_segments=(((0, 1),), ((1, 2),)),
        rank_split_ranges=((0, 1), (1, 2)),
    )
    bucket = SimpleNamespace(
        bucket_id=12,
        param_data=torch.zeros(8),
        params=(dion_param, standard_param),
        params_list=(dion_param, standard_param),
        param_to_index={dion_param: (0, 2), standard_param: (2, 8)},
        dion_param_ids={id(dion_param)},
        has_standard_params=True,
        dion_layout=dion_parameter.DionBucketLayout(
            entries=(dion_entry,),
            shard_size=2,
            gathered_numel=4,
            grad_gathered_numel=0,
            param_ids=frozenset({id(dion_param)}),
        ),
        dion_shard_group=group,
        intra_distributed_optimizer_instance_group=group,
        intra_distributed_optimizer_instance_size=2,
        intra_distributed_optimizer_instance_rank=1,
    )
    optimizer = SimpleNamespace(
        fs_group=group,
        _group_size=lambda group_arg: group_arg.size(),
        _get_data_shard=lambda param: None,
        _param_name=lambda param: None,
    )
    calls = []
    dion_restores = []
    standard_restores = []

    class Work:
        def wait(self):
            return None

    def fake_all_gather_into_tensor(output_tensor, input_tensor, group, async_op):
        del group, async_op
        call_index = len(calls)
        calls.append((input_tensor.data_ptr(), output_tensor.data_ptr()))
        output_view = output_tensor.view(2, -1)
        output_view[0].copy_(input_tensor)
        output_view[1].copy_(input_tensor + float(call_index + 1) * 100.0)
        return Work()

    def fake_restore_dion(
        optimizer_arg, *, bucket, prepared_entries, gathered_buffer, shard_group_size
    ):
        del optimizer_arg, bucket, prepared_entries, shard_group_size
        dion_restores.append(gathered_buffer.clone())

    def fake_restore_standard(bucket_arg, gathered_buffer, route_arg):
        del bucket_arg, route_arg
        standard_restores.append(gathered_buffer.clone())

    monkeypatch.setattr(dion_parameter.dist, "all_gather_into_tensor", fake_all_gather_into_tensor)
    monkeypatch.setattr(
        dion_parameter.dist,
        "get_process_group_ranks",
        lambda group_arg: group_arg.ranks,
    )
    monkeypatch.setattr(dion_parameter, "restore_bucket_param_data_", fake_restore_dion)
    monkeypatch.setattr(dion_parameter, "restore_standard_param_gather_", fake_restore_standard)

    def launch(dion_values, standard_values, *, async_op):
        bucket.param_data.copy_(
            torch.tensor(
                [*dion_values, -1.0, -2.0, *standard_values],
                dtype=torch.float32,
            )
        )
        return dion_parameter.all_gather_bucket_params_(optimizer, bucket, async_op=async_op)

    first_handle = launch([1.0, 2.0], [3.0, 4.0, 5.0, 6.0], async_op=True)
    second_handle = launch([10.0, 20.0], [30.0, 40.0, 50.0, 60.0], async_op=True)

    assert calls[0][0] != calls[1][0]
    assert calls[0][1] != calls[1][1]

    first_handle.wait()
    assert torch.equal(dion_restores[0], torch.tensor([[1.0, 2.0], [101.0, 102.0]]))
    assert torch.equal(
        standard_restores[0],
        torch.tensor(
            [[3.0, 4.0, 5.0, 6.0], [103.0, 104.0, 105.0, 106.0]]
        ),
    )

    second_handle.wait()
    prior_input_ptrs = {calls[0][0], calls[1][0]}
    prior_output_ptrs = {calls[0][1], calls[1][1]}

    assert launch([7.0, 8.0], [9.0, 10.0, 11.0, 12.0], async_op=False) is None
    assert calls[2][0] in prior_input_ptrs
    assert calls[2][1] in prior_output_ptrs
