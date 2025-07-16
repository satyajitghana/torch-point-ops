#!/usr/bin/env python3
"""
FLOPs Benchmarking Script for Torch Point Operations

This script benchmarks the computational cost (FLOPs) of:
1. Earth Mover's Distance (EMD)
2. Chamfer Distance

Usage: python benchmark_flops.py
"""

import torch
import time
import numpy as np
from typing import List, Tuple, Dict
import sys

try:
    from torch_point_ops.emd import earth_movers_distance
    from torch_point_ops.chamfer import chamfer_distance
except ImportError as e:
    print(f"Error importing torch_point_ops: {e}")
    print("Make sure the package is installed in your environment")
    sys.exit(1)

try:
    from fvcore.nn import FlopCountMode, flop_count
    FVCORE_AVAILABLE = True
except ImportError:
    print("Warning: fvcore not available. Install with: pip install fvcore")
    FVCORE_AVAILABLE = False

def generate_point_clouds(batch_size: int, n_points: int, m_points: int, device: str = "cuda") -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate random point clouds for benchmarking."""
    p1 = torch.randn(batch_size, n_points, 3, device=device, dtype=torch.float32)
    p2 = torch.randn(batch_size, m_points, 3, device=device, dtype=torch.float32)
    return p1, p2

def theoretical_flops_emd(n_points: int, m_points: int) -> int:
    """
    Estimate theoretical FLOPs for EMD operation.
    EMD involves solving an optimal transport problem, typically O(n^3) or O(n^2 m) complexity.
    This is an approximation based on the Hungarian algorithm or auction algorithm.
    """
    # EMD typically involves:
    # 1. Distance matrix computation: O(n * m * 3) for 3D points
    # 2. Optimal matching solution: O(min(n,m)^3) in worst case
    distance_matrix_flops = n_points * m_points * 3 * 2  # multiply + subtract for each dimension
    matching_flops = min(n_points, m_points) ** 3  # Approximation for Hungarian algorithm
    return distance_matrix_flops + matching_flops

def theoretical_flops_chamfer(n_points: int, m_points: int) -> int:
    """
    Estimate theoretical FLOPs for Chamfer Distance.
    Chamfer distance requires computing all pairwise distances and finding nearest neighbors.
    """
    # 1. Distance matrix computation: O(n * m * 3)
    # 2. Two nearest neighbor searches: O(n * m) + O(m * n)
    distance_matrix_flops = n_points * m_points * 3 * 4  # (x1-x2)^2 + (y1-y2)^2 + (z1-z2)^2 for each pair
    nearest_neighbor_flops = 2 * n_points * m_points  # Two NN searches with comparisons
    return distance_matrix_flops + nearest_neighbor_flops

def benchmark_operation(op_func, p1: torch.Tensor, p2: torch.Tensor, op_name: str, num_warmup: int = 10, num_runs: int = 100) -> Dict[str, float]:
    """Benchmark a point cloud operation."""
    device = p1.device
    batch_size, n_points = p1.shape[0], p1.shape[1]
    m_points = p2.shape[1]
    
    print(f"\n{'='*60}")
    print(f"Benchmarking {op_name}")
    print(f"{'='*60}")
    print(f"Input shapes: P1={p1.shape}, P2={p2.shape}")
    print(f"Device: {device}")
    
    # Theoretical FLOPs estimation
    if op_name == "EMD":
        theoretical_flops = theoretical_flops_emd(n_points, m_points)
    else:  # Chamfer
        theoretical_flops = theoretical_flops_chamfer(n_points, m_points)
    
    theoretical_flops_per_batch = theoretical_flops * batch_size
    print(f"Theoretical FLOPs (per batch): {theoretical_flops_per_batch:,}")
    
    # Warmup
    print("Warming up...")
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = op_func(p1, p2)
        if device.type == "cuda":
            torch.cuda.synchronize()
    
    # Benchmark runtime
    print("Benchmarking runtime...")
    times = []
    
    for i in range(num_runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        with torch.no_grad():
            result = op_func(p1, p2)
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        times.append(end_time - start_time)
        
        if (i + 1) % 20 == 0:
            print(f"  Completed {i + 1}/{num_runs} runs")
    
    # Calculate statistics
    mean_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    
    # FLOPs per second calculation
    flops_per_second = theoretical_flops_per_batch / mean_time
    
    print(f"\nRuntime Statistics:")
    print(f"  Mean time: {mean_time*1000:.3f} ± {std_time*1000:.3f} ms")
    print(f"  Min time:  {min_time*1000:.3f} ms")
    print(f"  Max time:  {max_time*1000:.3f} ms")
    print(f"  Theoretical FLOPS: {flops_per_second/1e9:.2f} GFLOPS")
    
    # Try fvcore profiling if available
    if FVCORE_AVAILABLE:
        try:
            print("\nProfiling with fvcore...")
            def wrapper():
                return op_func(p1, p2)
            
            flop_dict, _ = flop_count(wrapper, inputs=())
            total_flops = sum(flop_dict.values()) if flop_dict else 0
            print(f"  fvcore measured FLOPs: {total_flops:,}")
            
            if total_flops > 0:
                actual_flops_per_second = total_flops / mean_time
                print(f"  Actual FLOPS: {actual_flops_per_second/1e9:.2f} GFLOPS")
        except Exception as e:
            print(f"  fvcore profiling failed: {e}")
    
    return {
        'mean_time': mean_time,
        'std_time': std_time,
        'min_time': min_time,
        'max_time': max_time,
        'theoretical_flops': theoretical_flops_per_batch,
        'theoretical_flops_per_second': flops_per_second
    }

def main():
    """Main benchmarking function."""
    print("FLOPs Benchmarking for Torch Point Operations")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"
    else:
        device = "cuda"
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    
    # Test configurations: (batch_size, n_points, m_points)
    configs = [
        (1, 64, 64),      # Small
        (1, 128, 128),    # Medium
        (1, 256, 256),    # Large
        (4, 64, 64),      # Small batch
        (1, 512, 256),    # Asymmetric
        (1, 1024, 512),   # Large asymmetric
    ]
    
    results = {}
    
    for batch_size, n_points, m_points in configs:
        config_name = f"B{batch_size}_N{n_points}_M{m_points}"
        print(f"\n{'#'*80}")
        print(f"CONFIGURATION: {config_name}")
        print(f"{'#'*80}")
        
        # Generate point clouds
        p1, p2 = generate_point_clouds(batch_size, n_points, m_points, device)
        
        # Benchmark EMD
        try:
            emd_results = benchmark_operation(earth_movers_distance, p1, p2, "EMD")
            results[f"{config_name}_EMD"] = emd_results
        except Exception as e:
            print(f"EMD benchmark failed: {e}")
        
        # Benchmark Chamfer Distance
        try:
            chamfer_results = benchmark_operation(chamfer_distance, p1, p2, "Chamfer Distance")
            results[f"{config_name}_Chamfer"] = chamfer_results
        except Exception as e:
            print(f"Chamfer benchmark failed: {e}")
    
    # Summary
    print(f"\n{'='*80}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*80}")
    print(f"{'Configuration':<20} {'Operation':<15} {'Mean Time (ms)':<15} {'GFLOPS':<10}")
    print("-" * 80)
    
    for key, result in results.items():
        config, op = key.rsplit('_', 1)
        mean_time_ms = result['mean_time'] * 1000
        gflops = result['theoretical_flops_per_second'] / 1e9
        print(f"{config:<20} {op:<15} {mean_time_ms:<15.3f} {gflops:<10.2f}")

if __name__ == "__main__":
    main() 