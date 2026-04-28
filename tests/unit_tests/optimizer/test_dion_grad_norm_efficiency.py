import pytest
import torch

from megatron.core.optimizer.distrib_dion import grad_norm as dion_grad_norm


def test_grad_sum_sq_fp64_casts_in_bounded_chunks(monkeypatch):
    monkeypatch.setattr(dion_grad_norm, "_GRAD_NORM_FP64_CHUNK_BYTES", 16)
    original_to = torch.Tensor.to
    cast_numels = []

    def spy_to(self, *args, **kwargs):
        dtype = kwargs.get("dtype")
        if dtype is None and args and isinstance(args[0], torch.dtype):
            dtype = args[0]
        if dtype is torch.float64:
            cast_numels.append(int(self.numel()))
        return original_to(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", spy_to, raising=False)

    tensor = torch.arange(1, 6, dtype=torch.float32)
    total_sq = dion_grad_norm._grad_sum_sq_fp64(tensor)

    expected = tensor.to(dtype=torch.float64).mul(tensor.to(dtype=torch.float64)).sum()
    assert total_sq.item() == pytest.approx(expected.item())
    assert cast_numels[:3] == [2, 2, 1]
