import pytest
import torch

from megatron.core.optimizer.dion import linear as linear_helpers
from megatron.core.optimizer.dion import qkv as qkv_helpers


def _fail_cat(*args, **kwargs):
    pytest.fail("multi-segment child extraction should not call torch.cat")


def _fail_copy(*args, **kwargs):
    pytest.fail("single-segment alias commit should not call copy_")


def _shares_storage(lhs, rhs):
    return lhs.untyped_storage().data_ptr() == rhs.untyped_storage().data_ptr()


def test_extract_qkv_child_direct_copies_multi_segment_child(monkeypatch):
    parent = torch.arange(24, dtype=torch.float32).view(6, 4)[:, ::2]
    assert not parent.is_contiguous()

    monkeypatch.setattr(torch, "cat", _fail_cat)

    child = qkv_helpers.extract_qkv_child(parent, (1, 1, 1), "q")

    assert torch.equal(child, torch.tensor([[0.0, 2.0], [12.0, 14.0]]))
    assert child.is_contiguous()


def test_scatter_qkv_child_skips_single_segment_alias(monkeypatch):
    parent = torch.arange(12, dtype=torch.float32).view(6, 2)
    original = parent.clone()
    child = qkv_helpers.extract_qkv_child(parent, (2, 2, 2), "k")
    assert _shares_storage(parent, child)

    child.add_(100.0)
    monkeypatch.setattr(torch.Tensor, "copy_", _fail_copy)

    qkv_helpers.scatter_qkv_child_(parent, child, (2, 2, 2), "k")

    expected = original.clone()
    expected[2:4].add_(100.0)
    assert torch.equal(parent, expected)


def test_scatter_qkv_child_writes_single_segment_copy():
    parent = torch.arange(12, dtype=torch.float32).view(6, 2)
    original = parent.clone()
    child = qkv_helpers.extract_qkv_child(parent, (2, 2, 2), "k").clone()
    assert not _shares_storage(parent, child)

    child.add_(100.0)
    assert torch.equal(parent, original)

    qkv_helpers.scatter_qkv_child_(parent, child, (2, 2, 2), "k")

    expected = original.clone()
    expected[2:4].add_(100.0)
    assert torch.equal(parent, expected)


def test_scatter_qkv_child_writes_multi_segment_child():
    parent = torch.arange(12, dtype=torch.float32).view(6, 2)
    original = parent.clone()
    child = qkv_helpers.extract_qkv_child(parent, (1, 1, 1), "q")
    assert not _shares_storage(parent, child)

    child.add_(100.0)
    assert torch.equal(parent, original)

    qkv_helpers.scatter_qkv_child_(parent, child, (1, 1, 1), "q")

    expected = original.clone()
    expected[0] = child[0]
    expected[3] = child[1]
    assert torch.equal(parent, expected)


def test_read_linear_child_direct_copies_multi_segment_child(monkeypatch):
    parent = torch.arange(24, dtype=torch.float32).view(6, 4)[:, ::2]
    assert not parent.is_contiguous()

    monkeypatch.setattr(
        linear_helpers,
        "_linear_child_segments",
        lambda **kwargs: [(1, 3, 0, 2), (4, 5, 2, 3)],
    )
    monkeypatch.setattr(torch, "cat", _fail_cat)

    child = linear_helpers.read_linear_child(parent, (3, 3), None, "gate")

    assert torch.equal(child, torch.tensor([[4.0, 6.0], [8.0, 10.0], [16.0, 18.0]]))
    assert child.is_contiguous()


def test_write_linear_child_skips_single_segment_alias(monkeypatch):
    parent = torch.arange(12, dtype=torch.float32).view(6, 2)
    original = parent.clone()
    child = linear_helpers.read_linear_child(parent, (2, 4), None, "up")
    assert _shares_storage(parent, child)

    child.add_(100.0)
    monkeypatch.setattr(torch.Tensor, "copy_", _fail_copy)

    linear_helpers.write_linear_child_(parent, child, (2, 4), None, "up")

    expected = original.clone()
    expected[2:6].add_(100.0)
    assert torch.equal(parent, expected)


def test_write_linear_child_writes_single_segment_copy():
    parent = torch.arange(12, dtype=torch.float32).view(6, 2)
    original = parent.clone()
    child = linear_helpers.read_linear_child(parent, (2, 4), None, "up").clone()
    assert not _shares_storage(parent, child)

    child.add_(100.0)
    assert torch.equal(parent, original)

    linear_helpers.write_linear_child_(parent, child, (2, 4), None, "up")

    expected = original.clone()
    expected[2:6].add_(100.0)
    assert torch.equal(parent, expected)


def test_write_linear_child_writes_multi_segment_child(monkeypatch):
    parent = torch.arange(12, dtype=torch.float32).view(6, 2)
    original = parent.clone()
    monkeypatch.setattr(
        linear_helpers,
        "_linear_child_segments",
        lambda **kwargs: [(1, 3, 0, 2), (4, 5, 2, 3)],
    )
    child = linear_helpers.read_linear_child(parent, (3, 3), None, "gate")
    assert not _shares_storage(parent, child)

    child.add_(100.0)
    assert torch.equal(parent, original)

    linear_helpers.write_linear_child_(parent, child, (3, 3), None, "gate")

    expected = original.clone()
    expected[1:3] = child[0:2]
    expected[4:5] = child[2:3]
    assert torch.equal(parent, expected)
