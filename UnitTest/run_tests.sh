#!/usr/bin/env bash
set -e

# Project root directory
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

echo "======================================================="
echo "1. Building C++ Project and Test Runner..."
echo "======================================================="
mkdir -p build
cd build
rm -f coverage_*.profraw coverage.profdata
cmake -DENABLE_COVERAGE=ON ..
cmake --build . --config Release -j4
cd "$ROOT_DIR"

echo ""
echo "======================================================="
echo "2. Running PyTorch Comparison Test Suite via pytest..."
echo "======================================================="
PYTEST_BIN="pytest"
if [ -f "$ROOT_DIR/.venv/bin/pytest" ]; then
    PYTEST_BIN="$ROOT_DIR/.venv/bin/pytest"
fi

export LLVM_PROFILE_FILE="$ROOT_DIR/build/coverage_%p.profraw"
"$PYTEST_BIN" UnitTest/compare_with_pytorch.py -v

echo ""
echo "======================================================="
echo "3. Generating C++ Line Coverage Report..."
echo "======================================================="
if command -v xcrun &> /dev/null; then
    RUNNER_BIN="$ROOT_DIR/build/UnitTest/test_tensor_runner"
    PROFILES=($ROOT_DIR/build/coverage_*.profraw)
    if [ -f "${PROFILES[0]}" ]; then
        xcrun llvm-profdata merge -sparse "$ROOT_DIR"/build/coverage_*.profraw -o "$ROOT_DIR"/build/coverage.profdata
        echo "Line Coverage Summary:"
        xcrun llvm-cov report "$RUNNER_BIN" -instr-profile="$ROOT_DIR"/build/coverage.profdata -ignore-filename-regex="UnitTest|third_party"
    else
        echo "Warning: coverage_*.profraw files not found."
    fi
else
    echo "xcrun command not found. Skipping coverage generation."
fi

echo "======================================================="
echo "Test Execution & Coverage Completed Successfully!"
echo "======================================================="


