#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
# setup_optipim.sh — Bootstrap OptiPIM (LLVM, pim-opt, simulator)
#
# Prerequisites:
#   - CMake ≥ 3.20, Ninja, C++17 compiler
#   - Gurobi 11+ installed (GUROBI_HOME set or /Library/gurobi*)
#   - Python 3.11 (for torch-mlir compatibility)
#
# Usage:
#   cd third_party/OptiPIM && bash setup_optipim.sh
# ═══════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
JOBS="$(sysctl -n hw.ncpu 2>/dev/null || nproc)"
cd "$SCRIPT_DIR"

configure_cmake_build() {
  local build_dir="$1"
  local source_dir="$2"
  shift 2

  if [[ -f "$build_dir/CMakeCache.txt" ]]; then
    cmake -S "$source_dir" -B "$build_dir" "$@"
  else
    cmake -G Ninja -S "$source_dir" -B "$build_dir" "$@"
  fi
}

# ── Gurobi environment ───────────────────────────────────────────
export GUROBI_HOME="${GUROBI_HOME:-/Library/gurobi1301/macos_universal2}"
export GRB_LICENSE_FILE="${GRB_LICENSE_FILE:-$HOME/gurobi.lic}"

if [[ ! -d "$GUROBI_HOME" ]]; then
  echo "[ERROR] GUROBI_HOME not found: $GUROBI_HOME"
  echo "        Set GUROBI_HOME to your Gurobi installation."
  exit 1
fi

# ── Step 1: Build LLVM/MLIR ──────────────────────────────────────
echo "═══ Step 1: Building LLVM/MLIR ═══"
if [[ -f llvm-project/build/bin/mlir-opt ]]; then
  echo "  [skip] mlir-opt already exists."
else
  configure_cmake_build "$SCRIPT_DIR/llvm-project/build" "$SCRIPT_DIR/llvm-project/llvm" \
    -DLLVM_ENABLE_PROJECTS="mlir;clang" \
    -DLLVM_TARGETS_TO_BUILD="host" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=ON
  cmake --build "$SCRIPT_DIR/llvm-project/build" --parallel "$JOBS"
fi

# ── Step 2: Build pim-opt ────────────────────────────────────────
echo "═══ Step 2: Building pim-opt ═══"
configure_cmake_build "$SCRIPT_DIR/build" "$SCRIPT_DIR" \
  -DLLVM_DIR="$SCRIPT_DIR/llvm-project/build/lib/cmake/llvm" \
  -DMLIR_DIR="$SCRIPT_DIR/llvm-project/build/lib/cmake/mlir" \
  -DGUROBI_DIR="$GUROBI_HOME"
cmake --build "$SCRIPT_DIR/build" --target pim-opt --parallel "$JOBS"

if [[ -f build/bin/pim-opt ]]; then
  echo "  [ok] pim-opt built at build/bin/pim-opt"
else
  echo "  [ERROR] pim-opt not found after build."
  exit 1
fi

# ── Step 3: Build simulator ──────────────────────────────────────
echo "═══ Step 3: Building simulator (ramulator2) ═══"
configure_cmake_build "$SCRIPT_DIR/simulator/build" "$SCRIPT_DIR/simulator" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "$SCRIPT_DIR/simulator/build" --target ramulator2 --parallel "$JOBS"

if [[ -f simulator/build/ramulator2 ]]; then
  echo "  [ok] ramulator2 built at simulator/build/ramulator2"
else
  echo "  [ERROR] simulator not found after build."
  exit 1
fi

# ── Step 4: Quick smoke test ─────────────────────────────────────
echo "═══ Step 4: Smoke test ═══"
echo "[test] Running pim-opt --help..."
./build/bin/pim-opt --help > /dev/null 2>&1 && echo "  [ok] pim-opt runs" || echo "  [FAIL] pim-opt"

echo "[test] Running simulator --help..."
./simulator/build/ramulator2 2>&1 | head -1 || true
echo "  [ok] simulator runs"

echo ""
echo "═══ OptiPIM setup complete ═══"
echo "  pim-opt:   $SCRIPT_DIR/build/bin/pim-opt"
echo "  simulator: $SCRIPT_DIR/simulator/build/ramulator2"
echo "  arch cfg:  $SCRIPT_DIR/data/hbm_pim_1ch.json"
