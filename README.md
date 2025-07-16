<div align="center">
  <h1>Torch Point Ops</h1>
  <p>
    <b>A high-performance PyTorch library for 3D point cloud operations, including Chamfer Distance and Earth Mover's Distance (EMD) with CUDA support and built-in performance benchmarking.</b>
  </p>
  
  [![PyPI version](https://badge.fury.io/py/torch-point-ops.svg)](https://badge.fury.io/py/torch-point-ops)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  [![Python versions](https://img.shields.io/pypi/pyversions/torch-point-ops.svg)](https://pypi.org/project/torch-point-ops)
  [![Downloads](https://static.pepy.tech/badge/torch-point-ops)](https://pepy.tech/project/torch-point-ops)
  
  [![CUDA Accelerated](https://img.shields.io/badge/CUDA-Accelerated-76B900?logo=nvidia)](https://github.com/satyajitghana/torch-point-ops)
  [![Made with ❤️](https://img.shields.io/badge/Made%20with-❤️-red.svg)](https://github.com/satyajitghana/torch-point-ops)
  [![Powered by Coffee](https://img.shields.io/badge/Powered%20by-Coffee-brown.svg?logo=coffee)](https://github.com/satyajitghana/torch-point-ops)
  [![Speed](https://img.shields.io/badge/Speed-🚀%20Blazing%20Fast-blue.svg)](https://github.com/satyajitghana/torch-point-ops)
  [![Quality](https://img.shields.io/badge/Code%20Quality-✨%20Pristine-brightgreen.svg)](https://github.com/satyajitghana/torch-point-ops)
  [![GPU Power](https://img.shields.io/badge/GPU-💪%20Powered-orange.svg)](https://github.com/satyajitghana/torch-point-ops)

</div>

`torch-point-ops` provides efficient and well-tested implementations of common point cloud operations, designed to be easily integrated into any deep learning pipeline.

## ✨ Features

- **Chamfer Distance**: A fast and efficient implementation of the Chamfer Distance between two point clouds.
- **Earth Mover's Distance (EMD)**: An implementation of the Earth Mover's Distance for comparing point cloud distributions.
- **CUDA Support**: GPU-accelerated operations for high-performance computation.
- **Performance Benchmarking**: Built-in FLOPs benchmarking to measure computational efficiency.
- **Fully Tested**: Includes a comprehensive test suite to ensure correctness and reliability.
- **Production Ready**: Optimized for both research and deployment environments.

## 🚀 Getting Started

Follow these instructions to set up `torch-point-ops` in a local development environment.

### Prerequisites

- Python >= 3.13
- PyTorch >= 2.7.1
- A C++17 compatible compiler
- A CUDA-enabled GPU
- `uv` for package management
- (Optional) `patchelf` >= 0.14 for building the wheel

### Installation from Source

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/satyajitghana/torch-point-ops.git
    cd torch-point-ops
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install dependencies using `uv`:**

    ```bash
    pip install uv
    uv pip install -e .[dev]
    ```
This command installs the library in editable mode, allowing you to modify the source code and see the changes immediately.

### 🪝 Setting Up Code Quality Hooks (Recommended for Contributors)

For the best development experience and to ensure code quality, set up automated formatting and linting:

```bash
bash scripts/setup_hooks.sh
```

This gives you two options:

**Option 1: Pre-commit Framework (Recommended for Teams)**
- 🚀 Industry-standard tool used by major projects
- ✨ Runs Black, Ruff, and other quality checks
- 🔄 Auto-updates hook versions
- 🛡️ More robust than simple git hooks

**Option 2: Simple Git Hook (Basic)**
- 🔧 Simple Black formatter hook
- 📝 Good for solo development
- ⚡ Lightweight setup

> **Production Note**: This repo uses GitHub Actions to enforce formatting on all PRs, so your code will be checked regardless! The hooks just help catch issues early. 🎯

## 🛠️ Building the Wheel

To create a distributable wheel, run the provided build script. This is useful for installing the package in other environments without needing to build from source every time.

```bash
bash scripts/build_wheel.sh
```

This script will:
1.  Install `patchelf` (if not already present).
2.  Build the wheel using `uv`.
3.  Repair the wheel with `auditwheel` to bundle the required shared libraries.

The final wheel will be located in the `dist/` directory.

## 💡 Usage

Here's how you can use the Chamfer Distance and EMD functions in your project:

```python
import torch
from torch_point_ops.chamfer import chamfer_distance
from torch_point_ops.emd import earth_movers_distance

# Create two random point clouds on the GPU
p1 = torch.rand(1, 128, 3).cuda()
p2 = torch.rand(1, 128, 3).cuda()

# --- Chamfer Distance ---
dist1, dist2 = chamfer_distance(p1, p2)
loss = dist1.mean() + dist2.mean()
print(f"Chamfer Distance Loss: {loss.item()}")

# --- Earth Mover's Distance ---
emd_loss = earth_movers_distance(p1, p2)
print(f"Earth Mover's Distance Loss: {emd_loss.mean().item()}")
```

## 📊 Performance Benchmarking

Want to see how fast these operations really are? We've included a comprehensive FLOPs benchmarking script!

### 🚀 Running the Benchmark

```bash
# Activate your environment first
source .venv/bin/activate

# Run the FLOPs benchmark
python benchmark_flops.py
```

### 📈 Performance Highlights

Based on benchmarking with an RTX 3090:

| Operation | Point Cloud Size | Performance | Runtime |
|-----------|------------------|-------------|---------|
| **EMD** | 1024×512 points | **20.06 GFLOPS** | ~6.8ms |
| **Chamfer** | 512×256 points | **10.45 GFLOPS** | ~0.18ms |

**Key Insights:**
- 🎯 **EMD**: Consistent performance across input sizes, excellent for large point clouds
- ⚡ **Chamfer**: Lightning-fast on asymmetric configurations, ideal for real-time applications  
- 🔥 **GPU Scaling**: Both operations show significant performance gains on larger inputs
- 📊 **Efficiency**: Optimized CUDA kernels deliver maximum hardware utilization

*The benchmark script tests various configurations and provides detailed timing statistics, theoretical FLOP counts, and performance analysis.*

## ✅ Running Tests

To ensure everything is working correctly, it is recommended to run the local test suite.

**Note**: These tests require a CUDA-enabled GPU to run.

```bash
pytest
```

This command will automatically discover and run all tests in the `tests/` directory.

## 🤝 Contributing

Contributions are welcome! If you have a feature request, bug report, or want to contribute to the code, please open an issue or submit a pull request on the [GitHub repository](https://github.com/satyajitghana/torch-point-ops).

### Code Formatting

This project uses [Black](https://black.readthedocs.io/) for code formatting. Please ensure your code is formatted before submitting:

```bash
# Format all Python files
black .

# Check formatting without making changes
black --check .
```

**Pro tip**: Set up the git hooks (see above) to automatically format your code! 🚀

### 🤖 Automated Checks

This repository uses **GitHub Actions** to ensure code quality on every PR:

- ✅ **Black formatting** - Code must be properly formatted
- 🔍 **Ruff linting** - Code must pass all lint checks  
- 🚫 **PR blocking** - Improperly formatted code cannot be merged

The workflow runs on Python 3.11, 3.12, and 3.13 to ensure compatibility.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
