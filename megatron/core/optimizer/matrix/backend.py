"""Backend contract for matrix-aware optimizers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class MatrixStateSpec:
    """Backend-owned checkpoint state contract."""

    backend: str
    state_keys: tuple[str, ...]
    version: int = 1


class MatrixBackend(ABC):
    """Minimal base consumed by the distributed matrix wrapper."""

    name: str = ""
    supports_fs: bool = False
    supports_rp: bool = False
    supports_tp: bool = False
    supports_expert_parallel: bool = False
    supports_split_qkv: bool = False
    supports_split_qkvg: bool = False
    supports_split_linear: bool = False

    def validate_topology(
        self,
        *,
        fs_size: int,
        rp_size: int,
        tp_size: int,
        is_expert: bool,
        split_qkv: bool = False,
        split_qkvg: bool = False,
        split_linear: bool = False,
    ) -> None:
        checks = (
            (int(fs_size) > 1 and not self.supports_fs, "FS"),
            (int(rp_size) > 1 and not self.supports_rp, "RP"),
            (int(tp_size) > 1 and not self.supports_tp, "TP"),
            (bool(is_expert) and not self.supports_expert_parallel, "expert parallel"),
            (bool(split_qkv) and not self.supports_split_qkv, "split QKV"),
            (bool(split_qkvg) and not self.supports_split_qkvg, "split QKVG"),
            (bool(split_linear) and not self.supports_split_linear, "split linear"),
        )
        unsupported = [name for failed, name in checks if failed]
        if unsupported:
            backend_name = self.name or type(self).__name__
            raise RuntimeError(
                f"{backend_name} matrix backend does not support: {', '.join(unsupported)}"
            )

    @abstractmethod
    def refresh_state(self, adapter, *, param, state, optim_group, dist_meta) -> None:
        ...

    @abstractmethod
    def use_matrix(self, adapter, *, param, state, optim_group, dist_meta) -> bool:
        ...

    @abstractmethod
    def split_children(self, adapter, *, param, grad, state, optim_group, config, dist_meta):
        ...

    @abstractmethod
    def sync_state(self, adapter, matrix_params) -> None:
        ...

    @abstractmethod
    def build_batches(self, adapter, matrix_params):
        ...

    @abstractmethod
    def state_spec(self) -> MatrixStateSpec:
        ...
