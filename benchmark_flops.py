#!/usr/bin/env python3
"""
FLOPs Benchmarking Script for Torch Point Operations

This script benchmarks the computational cost (FLOPs) and runtime of:
1. Earth Mover's Distance (EMD)
2. Chamfer Distance
3. K-Nearest Neighbors (KNN)

It supports:
- Different floating point precisions (float32, float16)
- Eager mode vs. torch.compile() with different modes
- Various point cloud sizes

Usage: python benchmark_flops.py
"""

import torch
import time
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Callable, Any
import sys

# Add a try-except block for imports
try:
    from torch_point_ops.emd import earth_movers_distance
    from torch_point_ops.chamfer import chamfer_distance
    from torch_point_ops.knn import knn_points
except ImportError as e:
    print(f"Error importing torch_point_ops: {e}")
    print(
        "Make sure the package is installed in your environment. Run `pip install -e .`"
    )
    sys.exit(1)


def generate_point_clouds(
    batch_size: int,
    n_points: int,
    m_points: int,
    dtype: torch.dtype,
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate random point clouds for benchmarking."""
    p1 = torch.randn(batch_size, n_points, 3, device=device, dtype=dtype)
    p2 = torch.randn(batch_size, m_points, 3, device=device, dtype=dtype)
    return p1, p2


# --- Theoretical FLOPs Calculations ---


def theoretical_flops_emd(n_points: int, m_points: int) -> int:
    """Approximation for EMD FLOPs."""
    # Distance matrix: n*m*d*2 (mul+sub) -> n*m*3*2 = 6*n*m
    # Optimal matching (auction algorithm): roughly O(n*m*log(n))
    # This is a very rough approximation.
    distance_matrix_flops = n_points * m_points * 6
    matching_flops = n_points * m_points * int(np.log(n_points) + 1)
    return distance_matrix_flops + matching_flops


def theoretical_flops_chamfer(n_points: int, m_points: int) -> int:
    """Estimate theoretical FLOPs for Chamfer Distance."""
    # Distance matrix: (x1-x2)^2 + ... -> 8 FLOPs per pair
    distance_matrix_flops = n_points * m_points * 8
    # Two nearest neighbor searches (min operation) -> 2 * n*m comparisons
    nearest_neighbor_flops = 2 * n_points * m_points
    return distance_matrix_flops + nearest_neighbor_flops


def theoretical_flops_knn(n_points: int, m_points: int, k: int) -> int:
    """Estimate theoretical FLOPs for KNN."""
    # Distance matrix: (x1-x2)^2 + ... -> 8 FLOPs per pair
    distance_matrix_flops = n_points * m_points * 8
    # Top-K selection: O(m*log(k)) for each of n points
    # Approximating as m*k comparisons for simplicity.
    selection_flops = n_points * m_points * k
    return distance_matrix_flops + selection_flops


# --- Benchmarking Core ---


def benchmark_operation(
    op_func: Callable,
    p1: torch.Tensor,
    p2: torch.Tensor,
    op_name: str,
    config_details: Dict,
    num_warmup: int = 5,
    num_runs: int = 20,
) -> Dict[str, float]:
    """Generic benchmark for a point cloud operation."""
    device = p1.device
    batch_size, n_points, m_points = p1.shape[0], p1.shape[1], p2.shape[1]
    k = config_details.get("K", 1)

    # Theoretical FLOPs
    if op_name == "EMD":
        theoretical_flops = theoretical_flops_emd(n_points, m_points)
    elif op_name == "Chamfer":
        theoretical_flops = theoretical_flops_chamfer(n_points, m_points)
    elif op_name == "KNN":
        theoretical_flops = theoretical_flops_knn(n_points, m_points, k)
    else:
        theoretical_flops = 0

    theoretical_flops_per_batch = theoretical_flops * batch_size

    # Warmup
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = op_func(p1, p2)
    torch.cuda.synchronize()

    # Benchmark runtime using CUDA events for precision
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = op_func(p1, p2)
    end_event.record()
    torch.cuda.synchronize()

    mean_time_ms = start_event.elapsed_time(end_event) / num_runs

    gflops = (
        (theoretical_flops_per_batch / (mean_time_ms / 1000)) / 1e9
        if mean_time_ms > 0
        else 0
    )

    return {
        "Mean Time (ms)": mean_time_ms,
        "GFLOPS": gflops,
        "Theoretical FLOPs": theoretical_flops_per_batch,
    }


def main():
    """Main benchmarking function."""
    print("=" * 80)
    print("Benchmarking Torch Point Operations (EMD, Chamfer, KNN)")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("CUDA not available, benchmarks will not run.")
        return

    device = "cuda"
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    # --- Configurations ---
    # Larger sizes for more meaningful benchmarks
    size_configs = [
        (64, 512, 512),
        (32, 1024, 1024),
        (16, 2048, 2048),
    ]
    # Precision configs
    dtypes = [torch.float32, torch.float16]
    # Mode configs
    modes = [
        ("Eager", {}),
        ("Compile (default)", {"mode": "default"}),
        ("Compile (reduce-overhead)", {"mode": "reduce-overhead"}),
        ("Compile (max-autotune)", {"mode": "max-autotune"}),
    ]
    # KNN K values
    k_values = [1, 8, 16]

    all_results = []

    # Define operations to benchmark
    base_ops = {
        "EMD": earth_movers_distance,
        "Chamfer": chamfer_distance,
    }
    for k in k_values:
        base_ops[f"KNN_K{k}"] = lambda p1, p2, k=k: knn_points(p1, p2, K=k)

    # Pre-compile all function variants
    compiled_ops = {}
    for op_key, op_func_base in base_ops.items():
        for mode_name, mode_config in modes:
            if "Compile" not in mode_name:
                continue

            compile_key = (op_key, mode_name)
            print(f"Compiling {op_key} with mode: {mode_name}...")
            try:
                compiled_ops[compile_key] = torch.compile(op_func_base, **mode_config)
            except Exception as e:
                print(f"  Failed to compile: {e}")
                compiled_ops[compile_key] = None

    # --- Main Loop ---
    for batch_size, n_points, m_points in size_configs:
        size_str = f"B{batch_size}_N{n_points}_M{m_points}"
        print(f"\n--- Running Configuration: {size_str} ---")

        for dtype in dtypes:
            dtype_str = "FP32" if dtype == torch.float32 else "FP16"
            p1, p2 = generate_point_clouds(
                batch_size, n_points, m_points, dtype, device
            )

            for op_key, op_func_base in base_ops.items():
                if op_key.startswith("KNN"):
                    op_name = "KNN"
                    k = int(op_key.split("_K")[1])
                else:
                    op_name = op_key
                    k = "N/A"

                if op_name == "EMD" and dtype == torch.float16:
                    continue

                for mode_name, mode_config in modes:
                    config_details = {
                        "Operation": op_name,
                        "Config": size_str,
                        "Precision": dtype_str,
                        "Mode": mode_name,
                        "K": k,
                    }
                    print(
                        f"  Benchmarking: {op_name} (K={k}, {dtype_str}, {mode_name})"
                    )

                    if "Compile" in mode_name:
                        op_func = compiled_ops.get((op_key, mode_name))
                        if op_func is None:
                            continue
                    else:
                        op_func = op_func_base

                    result = benchmark_operation(
                        op_func, p1, p2, op_name, config_details
                    )
                    if result:
                        all_results.append({**config_details, **result})

    # --- Display Results ---
    if not all_results:
        print("\nNo benchmark results to display.")
        return

    df = pd.DataFrame(all_results)
    df["Mean Time (ms)"] = df["Mean Time (ms)"].map("{:.3f}".format)
    df["GFLOPS"] = df["GFLOPS"].map("{:.2f}".format)

    # Reorder columns for display
    display_cols = [
        "Operation",
        "Config",
        "K",
        "Precision",
        "Mode",
        "Mean Time (ms)",
        "GFLOPS",
    ]
    df = df[display_cols]

    print("\n" + "=" * 80)
    print("Benchmark Summary")
    print("=" * 80)
    print(df.to_string())

    # --- Generate Markdown for README ---
    print("\n" + "=" * 80)
    print("README Performance Highlights (Markdown)")
    print("=" * 80)

    # Filter for a specific, representative configuration
    highlight_config = "B16_N2048_M2048"
    df_highlight = df[df["Config"] == highlight_config].copy()

    if df_highlight.empty:
        print(f"No results found for highlight configuration: {highlight_config}")
        return

    # Calculate speedup vs. FP32 Eager
    df_highlight["Mean Time (ms)"] = pd.to_numeric(df_highlight["Mean Time (ms)"])
    baseline_times = {}
    for op_k_prec in (
        df_highlight[["Operation", "K", "Precision"]]
        .drop_duplicates()
        .to_records(index=False)
    ):
        op, k, prec = op_k_prec
        baseline = df_highlight[
            (df_highlight["Operation"] == op)
            & (df_highlight["K"] == k)
            & (df_highlight["Precision"] == prec)
            & (df_highlight["Mode"] == "Eager")
        ]
        if not baseline.empty:
            key = (op, k, prec)
            baseline_times[key] = baseline["Mean Time (ms)"].iloc[0]

    def get_speedup(row):
        baseline_key = (row["Operation"], row["K"], row["Precision"])
        baseline_time = baseline_times.get(baseline_key)
        if baseline_time and baseline_time > 0 and row["Mean Time (ms)"] > 0:
            return f"{baseline_time / row['Mean Time (ms)']:.2f}x"
        return "1.00x"

    df_highlight["Speedup vs Eager"] = df_highlight.apply(get_speedup, axis=1)

    # Build Markdown table
    md_table_header = """
| Operation      | Precision | Mode                      | Runtime (ms) | Speedup vs Eager |
|----------------|-----------|---------------------------|--------------|------------------|"""
    md_table_rows = [md_table_header]

    key_ops = [
        ("KNN", 16),
        ("Chamfer", "N/A"),
        ("EMD", "N/A"),
    ]

    for op, k in key_ops:
        df_op = df_highlight[df_highlight["Operation"] == op]
        if k != "N/A":
            df_op = df_op[df_op["K"] == k]

        if df_op.empty:
            continue

        for _, row in df_op.sort_values(by=["Precision", "Mode"]).iterrows():
            op_str = f"**{op} (K={k})**" if k != "N/A" else f"**{op}**"
            md_table_rows.append(
                f"| {op_str:<14} | {row['Precision']:<9} | {row['Mode']:<25} | {row['Mean Time (ms)']:<12.3f} | {row['Speedup vs Eager']:<16} |"
            )

    md_table = "\n".join(md_table_rows)

    print(f"\nPerformance for configuration: {highlight_config}")
    print(md_table)
    print(
        "\n*Runtimes are for a single forward pass on an NVIDIA GPU. Speedup is relative to the Eager mode of the same precision.*"
    )


if __name__ == "__main__":
    main()
