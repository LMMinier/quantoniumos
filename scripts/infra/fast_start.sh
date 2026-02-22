echo "======================================="
echo "  QuantoniumOS Fast Start"
echo "======================================="
echo "Tip: For fully-offline AI chat, see docs/manuals/QUICK_START.md (Run AI Chat Fully Offline)"
#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"
CPU_CORES=$( (command -v nproc >/dev/null 2>&1 && nproc) || (command -v sysctl >/dev/null 2>&1 && sysctl -n hw.ncpu) || echo 4 )

log() { echo "[fast-start] $*"; }

log "Project: ${PROJECT_ROOT}"

if [ ! -d "${VENV_DIR}" ]; then
  log "Creating virtual environment at ${VENV_DIR}"
  python3 -m venv "${VENV_DIR}"
fi
source "${VENV_DIR}/bin/activate"

if [ -f "${PROJECT_ROOT}/pyproject.toml" ]; then
  log "Installing Python dependencies (pyproject.toml)"
  pip install -e "${PROJECT_ROOT}"[dev]
fi

log "Building ASM/C kernels (algorithms/rft/kernels)"
pushd "${PROJECT_ROOT}/algorithms/rft/kernels" >/dev/null
make -j"${CPU_CORES}"
popd >/dev/null

log "Building C++ native module (src/rftmw_native)"
pushd "${PROJECT_ROOT}/src/rftmw_native" >/dev/null
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DRFTMW_ENABLE_ASM=ON
make -j"${CPU_CORES}"
popd >/dev/null

# Copy compiled module into the venv for immediate import
so_path=$(find "${PROJECT_ROOT}/src/rftmw_native/build" -name "rftmw_native.cpython-*.so" -print -quit)
if [ -n "${so_path}" ]; then
  pyver=$(python - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)
  site_target="${VENV_DIR}/lib/python${pyver}/site-packages"
  mkdir -p "${site_target}"
  cp "${so_path}" "${site_target}/"
  log "Installed native module to ${site_target}"
else
  log "WARNING: No rftmw_native .so found after build"
fi

log "Running wiring check (non-fatal)"
python "${PROJECT_ROOT}/scripts/validation/verify_test_wiring.py" || log "Wiring check completed with issues"

log "Fast start complete. Activate venv: source ${VENV_DIR}/bin/activate"
