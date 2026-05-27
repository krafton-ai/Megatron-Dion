"""Muon distributed checkpoint metadata helpers."""

from __future__ import annotations

import hashlib
from typing import Optional

import torch

from ..backend import MuonBackend

_MUON_PARAM_STATE_FORMAT = "muon_rank_local_state_v1"
_MUON_TENSOR_PAYLOAD_FORMAT = "muon_rank_local_tensor_shards_v1"


def build_muon_checkpoint_metadata(
    *,
    dp_size: int = 1,
    fs_size: int = 1,
    tp_size: int = 1,
    rp_size: int = 1,
    state_replica_size: int = 1,
    requested_type: str = "muon_rank_local_state",
    topology_signature: Optional[dict] = None,
    backend_state_spec=None,
) -> dict:
    """Return backend-owned checkpoint metadata for Muon optimizer state."""
    spec = backend_state_spec if backend_state_spec is not None else MuonBackend().state_spec()
    metadata = {
        "param_state_format": _MUON_PARAM_STATE_FORMAT,
        "requested_type": str(requested_type),
        "dp_size": int(dp_size),
        "fs_size": int(fs_size),
        "tp_size": int(tp_size),
        "rp_size": int(rp_size),
        "state_replica_size": int(state_replica_size),
        "matrix_optimizer": {
            "backend": spec.backend,
            "substrate_version": 1,
            "backend_state_version": int(spec.version),
            "state_keys": tuple(spec.state_keys),
        },
    }
    if topology_signature is not None:
        metadata["topology_signature"] = topology_signature
    return metadata


def validate_muon_checkpoint_metadata(
    checkpoint_metadata=None,
    *,
    metadata=None,
    dp_size: int = 1,
    fs_size: int = 1,
    tp_size: int = 1,
    rp_size: int = 1,
    state_replica_size: int = 1,
    topology_signature: Optional[dict] = None,
    backend_state_spec=None,
) -> None:
    """Validate saved Muon checkpoint metadata against the current topology."""
    if checkpoint_metadata is None:
        checkpoint_metadata = metadata
    checkpoint_metadata = _unwrap_checkpoint_leaf(checkpoint_metadata)
    if checkpoint_metadata is None:
        return
    expected = build_muon_checkpoint_metadata(
        dp_size=dp_size,
        fs_size=fs_size,
        tp_size=tp_size,
        rp_size=rp_size,
        state_replica_size=state_replica_size,
        requested_type=checkpoint_metadata.get("requested_type", "muon_rank_local_state"),
        topology_signature=topology_signature,
        backend_state_spec=backend_state_spec,
    )
    for key in ("dp_size", "fs_size", "tp_size", "rp_size", "state_replica_size"):
        saved_value = checkpoint_metadata.get(key, 1 if key == "rp_size" else -1)
        if int(saved_value) != int(expected[key]):
            raise RuntimeError(
                f"[Muon] unsupported checkpoint topology change for {key}: "
                f"saved={saved_value} current={expected[key]}"
            )
    saved_topology = checkpoint_metadata.get("topology_signature", None)
    if saved_topology is not None and topology_signature is not None:
        normalize = lambda sig: {
            str(key): tuple(int(rank) for rank in ranks)
            for key, ranks in sig.items()
        }
        if normalize(saved_topology) != normalize(topology_signature):
            raise RuntimeError(
                "[Muon] unsupported checkpoint topology identity change: "
                f"saved={saved_topology} current={topology_signature}"
            )
    saved_matrix = checkpoint_metadata.get("matrix_optimizer", {})
    expected_matrix = expected["matrix_optimizer"]
    for key in ("backend", "backend_state_version", "state_keys"):
        if key == "state_keys":
            mismatch = tuple(saved_matrix.get(key, ())) != tuple(expected_matrix[key])
        else:
            mismatch = saved_matrix.get(key) != expected_matrix[key]
        if mismatch:
            raise RuntimeError(
                f"[Muon] matrix optimizer checkpoint metadata mismatch for {key}: "
                f"saved={saved_matrix.get(key)} current={expected_matrix[key]}"
            )


def build_muon_param_state(param_groups, optimizer_state, get_param_key) -> dict:
    """Build a rank-local tensor-state payload."""
    payload = {"format": _MUON_PARAM_STATE_FORMAT, "states": {}}
    ordinal = 0
    for group in param_groups:
        for param in group.get("params", ()):
            state = optimizer_state.get(param, None)
            if state:
                key = get_param_key(param)
                if key is None:
                    key = f"ordinal_{ordinal}"
                state_payload = {"param": param.detach().clone()}
                for state_key, value in state.items():
                    if str(state_key).startswith("_"):
                        continue
                    if torch.is_tensor(value):
                        state_payload[state_key] = value.detach().clone()
                    elif isinstance(value, (int, float, bool, str, tuple, list, dict)):
                        state_payload[state_key] = value
                if state_payload:
                    payload["states"][repr(key)] = state_payload
            ordinal += 1
    return payload


def restore_muon_param_state_(param_groups, optimizer_state, param_state, get_param_key) -> None:
    """Restore a rank-local tensor-state payload."""
    if param_state is None:
        return
    param_state = _materialize_param_state_payload(param_state)
    if not isinstance(param_state, dict) or param_state.get("format") != _MUON_PARAM_STATE_FORMAT:
        raise RuntimeError("[Muon] invalid Muon param-state payload")
    saved = param_state.get("states", {})
    ordinal = 0
    for group in param_groups:
        for param in group.get("params", ()):
            key = get_param_key(param)
            if key is None:
                key = f"ordinal_{ordinal}"
            saved_state = saved.get(repr(key), None)
            if saved_state:
                state = optimizer_state.setdefault(param, {})
                for state_key, value in saved_state.items():
                    value = _unwrap_checkpoint_leaf(value)
                    if torch.is_tensor(value):
                        if state_key == "param":
                            if tuple(param.shape) != tuple(value.shape):
                                raise RuntimeError(
                                    "[Muon] checkpoint tensor shape mismatch for param: "
                                    f"saved={tuple(value.shape)} current={tuple(param.shape)}"
                                )
                            param.data.copy_(value.to(device=param.device, dtype=param.dtype))
                            continue
                        current = state.get(state_key, None)
                        if torch.is_tensor(current) and tuple(current.shape) == tuple(value.shape):
                            current.copy_(value.to(device=current.device, dtype=current.dtype))
                        else:
                            state[state_key] = value.detach().clone().to(device=param.device)
                    else:
                        state[state_key] = value
            ordinal += 1


def _stable_param_key_id(value) -> str:
    return hashlib.sha1(repr(value).encode("utf-8")).hexdigest()


def _unwrap_checkpoint_leaf(value):
    if hasattr(value, "data") and value.__class__.__name__ in {"ShardedObject", "ShardedTensor"}:
        return value.data
    return value


def _wrap_rank_local_tensor(*, tensor, key: str, replica_id, sharded_tensor_cls):
    data = tensor.detach()
    shape = tuple(int(dim) for dim in data.shape)
    return sharded_tensor_cls(
        key=key,
        data=data,
        dtype=data.dtype,
        local_shape=shape,
        global_shape=shape,
        global_offset=tuple(0 for _ in shape),
        axis_fragmentations=tuple(1 for _ in shape),
        replica_id=replica_id,
    )


def build_muon_tensor_param_state(
    *,
    param_groups,
    optimizer_state,
    get_param_key,
    base_key: str,
    rank_key: str,
    state_replica_id,
    sharded_object_cls,
    sharded_tensor_cls,
) -> dict:
    """Build Muon rank-local optimizer state as sharded tensor leaves."""
    metadata_payload = {"format": _MUON_TENSOR_PAYLOAD_FORMAT, "params": {}}
    tensor_payload = {}
    ordinal = 0
    for group in param_groups:
        for param in group.get("params", ()):
            state = optimizer_state.get(param, None)
            if not state:
                ordinal += 1
                continue
            param_key = get_param_key(param)
            if param_key is None:
                param_key = f"ordinal_{ordinal}"
            param_id = _stable_param_key_id(param_key)
            values = {}
            tensor_keys = []
            tensors = {"param": param.detach()}
            for state_key, value in state.items():
                if str(state_key).startswith("_"):
                    continue
                if torch.is_tensor(value):
                    tensors[state_key] = value
                elif isinstance(value, (int, float, bool, str, tuple, list, dict)):
                    values[state_key] = value
            for state_key, value in tensors.items():
                tensor_key = f"{base_key}.muon_param_state.{rank_key}.{param_id}.{state_key}"
                tensor_payload.setdefault(param_id, {})[state_key] = _wrap_rank_local_tensor(
                    tensor=value,
                    key=tensor_key,
                    replica_id=state_replica_id,
                    sharded_tensor_cls=sharded_tensor_cls,
                )
                tensor_keys.append(state_key)
            metadata_payload["params"][param_key] = {
                "id": param_id,
                "values": values,
                "tensor_keys": tuple(tensor_keys),
            }
            ordinal += 1
    return {
        "metadata": sharded_object_cls(
            f"{base_key}.muon_param_state.{rank_key}.metadata",
            metadata_payload,
            (1,),
            (0,),
            replica_id=state_replica_id,
        ),
        "tensors": tensor_payload,
    }


def _materialize_param_state_payload(param_state):
    if not isinstance(param_state, dict):
        return param_state
    if param_state.get("format") == _MUON_PARAM_STATE_FORMAT:
        return param_state
    if "metadata" not in param_state or "tensors" not in param_state:
        return param_state

    metadata = _unwrap_checkpoint_leaf(param_state["metadata"])
    if not isinstance(metadata, dict) or metadata.get("format") != _MUON_TENSOR_PAYLOAD_FORMAT:
        raise RuntimeError("[Muon] invalid sharded Muon param-state metadata")
    tensors_by_param = param_state.get("tensors", {})
    payload = {"format": _MUON_PARAM_STATE_FORMAT, "states": {}}
    for ordinal, (param_key, entry) in enumerate(metadata.get("params", {}).items()):
        param_id = entry.get("id", None)
        values = dict(entry.get("values", {}))
        tensor_state = tensors_by_param.get(param_id, {})
        for state_key in entry.get("tensor_keys", ()):
            if state_key not in tensor_state:
                raise RuntimeError(
                    f"[Muon] sharded checkpoint missing tensor state {state_key!r}"
                )
            values[state_key] = _unwrap_checkpoint_leaf(tensor_state[state_key])
        payload["states"][repr(param_key)] = values
        payload["states"][repr(f"ordinal_{ordinal}")] = values
    return payload


def build_distributed_checkpoint_state(
    *,
    common_state: dict,
    param_groups,
    optimizer_state,
    get_param_key,
    base_key: str,
    common_replica_id,
    state_global_shape,
    state_global_offset,
    state_replica_id,
    checkpoint_metadata: dict,
    sharded_object_cls,
    sharded_tensor_cls=None,
    state_rank_key: Optional[str] = None,
) -> dict:
    """Build standard distributed checkpoint state with Muon rank-local payload."""
    state_dict = {
        key: sharded_object_cls(
            f"{base_key}.{key}",
            value,
            (1,),
            (0,),
            replica_id=common_replica_id,
        )
        for key, value in common_state.items()
    }
    state_dict["muon_checkpoint_metadata"] = sharded_object_cls(
        f"{base_key}.muon_checkpoint_metadata",
        checkpoint_metadata,
        (1,),
        (0,),
        replica_id=common_replica_id,
    )
    if sharded_tensor_cls is None:
        state_dict["muon_param_state"] = sharded_object_cls(
            f"{base_key}.muon_param_state",
            build_muon_param_state(param_groups, optimizer_state, get_param_key),
            tuple(int(dim) for dim in state_global_shape),
            tuple(int(offset) for offset in state_global_offset),
            replica_id=state_replica_id,
        )
    else:
        state_dict["muon_param_state"] = build_muon_tensor_param_state(
            param_groups=param_groups,
            optimizer_state=optimizer_state,
            get_param_key=get_param_key,
            base_key=base_key,
            rank_key=(
                state_rank_key
                if state_rank_key is not None
                else ".".join(str(int(offset)) for offset in state_global_offset)
            ),
            state_replica_id=state_replica_id,
            sharded_object_cls=sharded_object_cls,
            sharded_tensor_cls=sharded_tensor_cls,
        )
    return state_dict


def split_distributed_checkpoint_state(state_dict: dict):
    """Split Muon checkpoint fields from common optimizer state."""
    metadata = state_dict.get("muon_checkpoint_metadata", None)
    param_state = state_dict.get("muon_param_state", None)
    common = {
        key: value
        for key, value in state_dict.items()
        if key not in {"muon_checkpoint_metadata", "muon_param_state"}
    }
    return metadata, param_state, common


__all__ = [
    "build_muon_checkpoint_metadata",
    "build_distributed_checkpoint_state",
    "build_muon_param_state",
    "build_muon_tensor_param_state",
    "restore_muon_param_state_",
    "split_distributed_checkpoint_state",
    "validate_muon_checkpoint_metadata",
]
