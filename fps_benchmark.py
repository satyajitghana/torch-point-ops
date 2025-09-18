#!/usr/bin/env python3
"""
FPS vs Quick FPS Performance Benchmark

This script specifically benchmarks FPS (Furthest Point Sampling) implementations:
1. Regular FPS
2. Quick FPS with different KD-tree depths

It compares:
- FP16 vs FP32 precision
- Small vs large point clouds
- Small vs large batch sizes
- Different KD-tree depths for Quick FPS
"""

import torch
import time
import pandas as pd
from typing import Dict, List, Tuple
import numpy as np

try:
    from torch_point_ops.fps import furthest_point_sampling, quick_furthest_point_sampling
except ImportError as e:
    print(f"Error importing torch_point_ops: {e}")
    print("Make sure the package is installed: pip install -e .")
    exit(1)


def benchmark_fps_operation(
    op_func, points: torch.Tensor, nsamples: int,
    num_warmup: int = 5, num_runs: int = 20, **kwargs
) -> Dict[str, float]:
    """Benchmark a single FPS operation."""
    device = points.device
    batch_size, n_points = points.shape[0], points.shape[1]
    
    # Adjust number of runs for very large point clouds to avoid timeouts
    if n_points >= 100000:
        num_warmup = 2
        num_runs = 5
    elif n_points >= 50000:
        num_warmup = 3
        num_runs = 10
    elif n_points >= 10000:
        num_warmup = 4
        num_runs = 15
    
    try:
        # Warmup
        for i in range(num_warmup):
            with torch.no_grad():
                _ = op_func(points, nsamples, **kwargs)
        torch.cuda.synchronize()
        
        # Benchmark with CUDA events
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(num_runs):
            with torch.no_grad():
                _ = op_func(points, nsamples, **kwargs)
        end_event.record()
        torch.cuda.synchronize()
        
        mean_time_ms = start_event.elapsed_time(end_event) / num_runs
        
        # Calculate theoretical FLOPs for FPS
        # Each iteration: distance computation (8 FLOPs/point) + min update + max finding
        flops_per_iteration = n_points * 10  # Approximate
        theoretical_flops = nsamples * flops_per_iteration * batch_size
        gflops = (theoretical_flops / (mean_time_ms / 1000)) / 1e9 if mean_time_ms > 0 else 0
        
        return {
            "mean_time_ms": mean_time_ms,
            "gflops": gflops,
            "theoretical_flops": theoretical_flops
        }
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"        CUDA OOM for {n_points} points, skipping...")
            torch.cuda.empty_cache()
            return None
        else:
            raise e
    except Exception as e:
        print(f"        Error with {n_points} points: {e}")
        return None


def main():
    print("=" * 80)
    print("FPS vs Quick FPS Performance Benchmark")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("CUDA not available, benchmark cannot run.")
        return
    
    device = "cuda"
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    
    # Test configurations
    configs = [
        # (batch_size, n_points, nsamples, description)
        (64, 256, 64, "Small batch, small clouds"),
        (32, 512, 128, "Medium batch, medium clouds"),
        (16, 1024, 256, "Small batch, large clouds"),
        (8, 2048, 512, "Very small batch, very large clouds"),
        (1, 4096, 1024, "Single batch, huge clouds"),
        # Extreme large point cloud configurations
        (1, 10000, 2048, "Single batch, 10K points"),
        (1, 50000, 5000, "Single batch, 50K points"),
        (1, 100000, 10000, "Single batch, 100K points"),
        (1, 500000, 20000, "Single batch, 500K points - EXTREME"),
    ]
    
    dtypes = [
        (torch.float32, "FP32"),
        (torch.float16, "FP16"),
    ]
    
    kd_depths = [3, 4, 5, 6]
    
    results = []
    
    for batch_size, n_points, nsamples, desc in configs:
        print(f"\n--- {desc}: B{batch_size}_N{n_points}_S{nsamples} ---")
        
        for dtype, dtype_str in dtypes:
            print(f"  Testing {dtype_str} precision...")
            
            # Generate test data
            torch.manual_seed(42)  # For reproducible results
            points = torch.randn(batch_size, n_points, 3, device=device, dtype=dtype)
            
            # Benchmark Regular FPS
            print(f"    Regular FPS...")
            try:
                fps_results = benchmark_fps_operation(furthest_point_sampling, points, nsamples)
                if fps_results is not None:
                    results.append({
                        "Algorithm": "FPS",
                        "Config": f"B{batch_size}_N{n_points}",
                        "Precision": dtype_str,
                        "KD_Depth": "N/A",
                        "NSamples": nsamples,
                        "Time_ms": fps_results["mean_time_ms"],
                        "GFLOPS": fps_results["gflops"],
                        "Description": desc
                    })
                else:
                    print(f"        Skipping remaining tests for this configuration due to memory constraints")
                    continue
            except Exception as e:
                print(f"        Error: {e}")
                continue
            
            # Benchmark Quick FPS with different KD depths
            for kd_depth in kd_depths:
                print(f"    Quick FPS (KD={kd_depth})...")
                try:
                    quick_results = benchmark_fps_operation(
                        quick_furthest_point_sampling, points, nsamples, kd_depth=kd_depth
                    )
                    if quick_results is not None:
                        results.append({
                            "Algorithm": "Quick_FPS",
                            "Config": f"B{batch_size}_N{n_points}",
                            "Precision": dtype_str,
                            "KD_Depth": kd_depth,
                            "NSamples": nsamples,
                            "Time_ms": quick_results["mean_time_ms"],
                            "GFLOPS": quick_results["gflops"],
                            "Description": desc
                        })
                    # Note: Don't break on None for Quick FPS, other KD depths might work
                except Exception as e:
                    print(f"        Error: {e}")
                    continue
    
    if not results:
        print("No benchmark results collected.")
        return
    
    # Convert to DataFrame and analyze
    df = pd.DataFrame(results)
    
    print("\n" + "=" * 120)
    print("DETAILED RESULTS")
    print("=" * 120)
    
    # Format numbers for display
    df_display = df.copy()
    df_display["Time_ms"] = df_display["Time_ms"].apply(lambda x: f"{x:.3f}")
    df_display["GFLOPS"] = df_display["GFLOPS"].apply(lambda x: f"{x:.2f}")
    
    print(df_display.to_string(index=False))
    
    print("\n" + "=" * 120)
    print("PERFORMANCE ANALYSIS")
    print("=" * 120)
    
    # Calculate speedups
    analysis_results = []
    
    for config in df["Config"].unique():
        for precision in df["Precision"].unique():
            config_data = df[(df["Config"] == config) & (df["Precision"] == precision)]
            
            if config_data.empty:
                continue
            
            # Get baseline FPS time
            fps_data = config_data[config_data["Algorithm"] == "FPS"]
            if fps_data.empty:
                continue
            
            fps_time = fps_data["Time_ms"].iloc[0]
            fps_gflops = fps_data["GFLOPS"].iloc[0]
            
            # Compare with Quick FPS variants
            quick_data = config_data[config_data["Algorithm"] == "Quick_FPS"]
            
            for _, row in quick_data.iterrows():
                speedup = fps_time / row["Time_ms"] if row["Time_ms"] > 0 else 1.0
                gflops_ratio = row["GFLOPS"] / fps_gflops if fps_gflops > 0 else 1.0
                
                analysis_results.append({
                    "Config": config,
                    "Precision": precision,
                    "KD_Depth": row["KD_Depth"],
                    "FPS_Time_ms": fps_time,
                    "Quick_FPS_Time_ms": row["Time_ms"],
                    "Speedup": speedup,
                    "FPS_GFLOPS": fps_gflops,
                    "Quick_FPS_GFLOPS": row["GFLOPS"],
                    "GFLOPS_Ratio": gflops_ratio
                })
    
    if analysis_results:
        analysis_df = pd.DataFrame(analysis_results)
        
        print("\nSpeedup Analysis (Quick FPS vs Regular FPS):")
        print("-" * 100)
        
        for _, row in analysis_df.iterrows():
            speedup_str = f"{row['Speedup']:.2f}x" if row['Speedup'] != 1.0 else "1.00x (same)"
            if row['Speedup'] > 1.0:
                speedup_str += " (faster)"
            elif row['Speedup'] < 1.0:
                speedup_str += " (slower)"
                
            print(f"{row['Config']:<20} {row['Precision']:<6} KD={row['KD_Depth']:<2} | "
                  f"FPS: {row['FPS_Time_ms']:6.2f}ms | Quick: {row['Quick_FPS_Time_ms']:6.2f}ms | "
                  f"Speedup: {speedup_str}")
    
    # Precision comparison
    print(f"\n{'-'*100}")
    print("FP16 vs FP32 Analysis:")
    print("-" * 50)
    
    precision_analysis = []
    for config in df["Config"].unique():
        for algo in df["Algorithm"].unique():
            for kd_depth in df["KD_Depth"].unique():
                fp32_data = df[(df["Config"] == config) & (df["Algorithm"] == algo) & 
                              (df["Precision"] == "FP32") & (df["KD_Depth"] == kd_depth)]
                fp16_data = df[(df["Config"] == config) & (df["Algorithm"] == algo) & 
                              (df["Precision"] == "FP16") & (df["KD_Depth"] == kd_depth)]
                
                if not fp32_data.empty and not fp16_data.empty:
                    fp32_time = fp32_data["Time_ms"].iloc[0]
                    fp16_time = fp16_data["Time_ms"].iloc[0]
                    speedup = fp32_time / fp16_time if fp16_time > 0 else 1.0
                    
                    kd_str = f"KD={kd_depth}" if kd_depth != "N/A" else "N/A"
                    print(f"{config:<20} {algo:<10} {kd_str:<6} | "
                          f"FP32: {fp32_time:6.2f}ms | FP16: {fp16_time:6.2f}ms | "
                          f"FP16 Speedup: {speedup:.2f}x")
    
    # Summary statistics
    print(f"\n{'-'*100}")
    print("SUMMARY STATISTICS:")
    print("-" * 30)
    
    if analysis_results:
        speedups = [r["Speedup"] for r in analysis_results]
        print(f"Quick FPS vs Regular FPS:")
        print(f"  Average speedup: {np.mean(speedups):.2f}x")
        print(f"  Median speedup: {np.median(speedups):.2f}x")
        print(f"  Best speedup: {np.max(speedups):.2f}x")
        print(f"  Worst speedup: {np.min(speedups):.2f}x")
        
        faster_count = sum(1 for s in speedups if s > 1.0)
        same_count = sum(1 for s in speedups if s == 1.0)
        slower_count = sum(1 for s in speedups if s < 1.0)
        
        print(f"  Quick FPS is faster: {faster_count}/{len(speedups)} cases")
        print(f"  Quick FPS is same: {same_count}/{len(speedups)} cases")
        print(f"  Quick FPS is slower: {slower_count}/{len(speedups)} cases")
    
    print(f"\n{'-'*100}")
    print("Best configurations:")
    print("-" * 30)
    
    # Find best performing configurations
    if not df.empty:
        # Best overall performance (lowest time)
        best_overall = df.loc[df["Time_ms"].idxmin()]
        print(f"Fastest overall: {best_overall['Algorithm']} "
              f"(KD={best_overall['KD_Depth']}) "
              f"{best_overall['Precision']} "
              f"on {best_overall['Config']}: {best_overall['Time_ms']:.2f}ms")
        
        # Best GFLOPS
        best_gflops = df.loc[df["GFLOPS"].idxmax()]
        print(f"Highest GFLOPS: {best_gflops['Algorithm']} "
              f"(KD={best_gflops['KD_Depth']}) "
              f"{best_gflops['Precision']} "
              f"on {best_gflops['Config']}: {best_gflops['GFLOPS']:.2f} GFLOPS")


if __name__ == "__main__":
    main()