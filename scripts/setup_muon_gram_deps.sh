#!/usr/bin/env bash
# Source this script before launching Muon Gram NS jobs that use Dao/Quack kernels.
#
# It installs optional Gram Newton-Schulz runtime dependencies into a repo-local,
# ignored cache and exports the import/library paths needed by quack-kernels.

set -euo pipefail

if [ -n "${BASH_SOURCE[0]:-}" ]; then
    _MUON_GRAM_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
else
    _MUON_GRAM_SCRIPT_DIR="$(pwd)"
fi

if [ -z "${REPO:-}" ]; then
    REPO="$(cd "${_MUON_GRAM_SCRIPT_DIR}/.." && pwd)"
fi

MUON_GRAM_DEPS_DIR="${MUON_GRAM_DEPS_DIR:-${REPO}/runs/deps/muon_gram}"
MUON_GRAM_PACKAGES="${MUON_GRAM_PACKAGES:-quack-kernels==0.4.1 nvidia-cutlass-dsl==4.4.2 nvidia-cutlass-dsl-libs-base==4.4.2 apache-tvm-ffi==0.1.11 torch-c-dlpack-ext==0.1.5}"
QUACK_CACHE_DIR="${QUACK_CACHE_DIR:-${REPO}/runs/quack_cache}"

mkdir -p "${MUON_GRAM_DEPS_DIR}" "${QUACK_CACHE_DIR}"

_muon_gram_pythonpath="${MUON_GRAM_DEPS_DIR}:${MUON_GRAM_DEPS_DIR}/nvidia_cutlass_dsl/python_packages"
_muon_gram_ldpath="${MUON_GRAM_DEPS_DIR}/nvidia_cutlass_dsl/lib"

_muon_gram_check() {
    PYTHONPATH="${_muon_gram_pythonpath}:${PYTHONPATH:-}" \
    LD_LIBRARY_PATH="${_muon_gram_ldpath}:${LD_LIBRARY_PATH:-}" \
        python - <<'PY' >/dev/null 2>&1
import quack.gemm_interface
import cutlass
PY
}

if ! _muon_gram_check; then
    (
        flock 9
        if ! _muon_gram_check; then
            python -m pip install \
                --target "${MUON_GRAM_DEPS_DIR}" \
                --upgrade \
                --no-deps \
                --no-build-isolation \
                ${MUON_GRAM_PACKAGES}
        fi
    ) 9>"${MUON_GRAM_DEPS_DIR}.lock"
fi

_muon_gram_check

export REPO
export MUON_GRAM_DEPS_DIR
export MUON_GRAM_PACKAGES
export QUACK_CACHE_DIR
export PYTHONPATH="${_muon_gram_pythonpath}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${_muon_gram_ldpath}:${LD_LIBRARY_PATH:-}"

unset _muon_gram_pythonpath
unset _muon_gram_ldpath
