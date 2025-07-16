#!/bin/bash
set -e

# This script builds and repairs the wheel for the torch-point-ops project.
# It handles the installation of uv, patchelf, builds the wheel
# using uv, and then repairs it to include the necessary PyTorch libraries.

# 1. Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "uv not found, installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "$HOME/.cargo/env"
fi

# 2. Install patchelf from source if not present or version is < 0.14
export PATH="$HOME/.local/bin:$PATH"
if ! command -v patchelf &> /dev/null || ! [[ $(patchelf --version | cut -d' ' -f2) > "0.13" ]]; then
    echo "patchelf not found or version is less than 0.14. Installing patchelf 0.18.0..."
    wget -q https://github.com/NixOS/patchelf/archive/refs/tags/0.18.0.tar.gz
    tar xf 0.18.0.tar.gz
    (
        cd patchelf-0.18.0
        ./bootstrap.sh
        ./configure --prefix="$HOME/.local"
        make -j
        make install
    )
    rm -rf patchelf-0.18.0 0.18.0.tar.gz
fi

# 3. Set up virtual environment and install dependencies
uv venv
source .venv/bin/activate
uv sync

# 4. Build the wheel
uv build

# 5. Repair the wheel
# Find the built wheel file
WHEEL_FILE=$(find dist -name "torch_point_ops-*.whl" | head -n 1)
if [ -z "$WHEEL_FILE" ]; then
    echo "Error: No wheel file found in dist/ directory."
    exit 1
fi

# Set the library path for auditwheel to find torch libraries
PY_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
export LD_LIBRARY_PATH="$(pwd)/.venv/lib/python${PY_VERSION}/site-packages/torch/lib:$LD_LIBRARY_PATH"

# Repair the wheel
auditwheel repair "$WHEEL_FILE" --wheel-dir dist/

# Clean up original wheel
REPAIRED_WHEEL_FILE=$(find dist -name "*manylinux*.whl" | head -n 1)
if [ -f "$REPAIRED_WHEEL_FILE" ] && [ "$WHEEL_FILE" != "$REPAIRED_WHEEL_FILE" ]; then
    rm "$WHEEL_FILE"
fi

echo "Successfully built and repaired wheel: $REPAIRED_WHEEL_FILE" 