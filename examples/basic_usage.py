#!/usr/bin/env python3
"""
Basic Usage Example for torch-point-ops

This example demonstrates how to use the Chamfer Distance, Earth Mover's Distance,
K-Nearest Neighbors, and Furthest Point Sampling functions for point cloud operations.
"""

import torch
import time

# Clean import style
from torch_point_ops.chamfer import chamfer_distance
from torch_point_ops.emd import earth_movers_distance
from torch_point_ops.knn import knn_points
from torch_point_ops.fps import (
    furthest_point_sampling,
    quick_furthest_point_sampling,
    gather_points,
    farthest_point_sample_and_gather,
    quick_farthest_point_sample_and_gather
)

# Alternative: import from top-level package
# from torch_point_ops import chamfer_distance, earth_movers_distance


def main():
    print("🎯 torch-point-ops Basic Usage Example")
    print("=" * 50)

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create sample point clouds
    batch_size = 2
    n_points_1 = 100
    n_points_2 = 150

    print("\n📊 Creating sample point clouds:")
    print(f"  - Point cloud 1: {batch_size} batches × {n_points_1} points × 3D")
    print(f"  - Point cloud 2: {batch_size} batches × {n_points_2} points × 3D")

    # Generate random point clouds
    torch.manual_seed(42)  # For reproducible results
    points1 = torch.randn(batch_size, n_points_1, 3, device=device, dtype=torch.float32)
    points2 = torch.randn(batch_size, n_points_2, 3, device=device, dtype=torch.float32)

    print("\n" + "=" * 50)
    print("📏 CHAMFER DISTANCE")
    print("=" * 50)

    # Compute Chamfer Distance
    print("Computing Chamfer Distance...")
    dist1, dist2 = chamfer_distance(points1, points2)

    print("✅ Results:")
    print(
        f"  - Distance from points1 to points2: {dist1.shape} -> mean = {dist1.mean().item():.6f}"
    )
    print(
        f"  - Distance from points2 to points1: {dist2.shape} -> mean = {dist2.mean().item():.6f}"
    )
    print(f"  - Total Chamfer Distance: {(dist1.mean() + dist2.mean()).item():.6f}")

    # Using the Chamfer Distance module
    print("\nUsing ChamferDistance module:")
    from torch_point_ops import ChamferDistance

    chamfer_loss = ChamferDistance(reduction="mean")
    total_distance = chamfer_loss(points1, points2)
    print(f"  - Mean Chamfer Distance: {total_distance.item():.6f}")

    print("\n" + "=" * 50)
    print("🌍 EARTH MOVER'S DISTANCE (EMD)")
    print("=" * 50)

    # For EMD, we typically use smaller point clouds due to computational cost
    small_points1 = points1[:, :50, :]  # Use first 50 points
    small_points2 = points2[:, :50, :]  # Use first 50 points

    print(
        f"Computing EMD with smaller point clouds ({small_points1.shape[1]} points each)..."
    )
    emd_distances = earth_movers_distance(small_points1, small_points2)

    print("✅ Results:")
    print(f"  - EMD distances: {emd_distances.shape}")
    print(f"  - Per-batch EMD: {emd_distances.cpu().numpy()}")
    print(f"  - Mean EMD: {emd_distances.mean().item():.6f}")

    # Using the EarthMoverDistance module
    print("\nUsing EarthMoverDistance module:")
    from torch_point_ops import EarthMoverDistance

    emd_loss = EarthMoverDistance()
    total_emd = emd_loss(small_points1, small_points2)
    print(f"  - Mean EMD Distance: {total_emd.item():.6f}")

    print("\n" + "=" * 50)
    print("🎓 COMPARISON & INSIGHTS")
    print("=" * 50)

    print("Key differences:")
    print("  📏 Chamfer Distance:")
    print("     - Fast to compute")
    print("     - Works well for different sized point clouds")
    print("     - Good for general shape similarity")
    print("     - Returns bidirectional distances")

    print("\n  🌍 Earth Mover's Distance:")
    print("     - More computationally expensive")
    print("     - Better for precise geometric matching")
    print("     - Considers optimal transport between points")
    print("     - Single distance value")

    print("\n" + "=" * 50)
    print("🔍 K-NEAREST NEIGHBORS (KNN)")
    print("=" * 50)

    print("Finding K=5 nearest neighbors between point clouds...")
    knn_result = knn_points(points1, points2, K=5, return_nn=True)
    
    print("✅ Results:")
    print(f"  - KNN distances shape: {knn_result.dists.shape}")  # [batch_size, n_points_1, K]
    print(f"  - KNN indices shape: {knn_result.idx.shape}")      # [batch_size, n_points_1, K]
    print(f"  - KNN coordinates shape: {knn_result.knn.shape}")  # [batch_size, n_points_1, K, 3]
    print(f"  - Average distance to nearest neighbor: {knn_result.dists[:, :, 0].mean().item():.6f}")

    # KNN with different precision for performance comparison
    if device == "cuda":
        print("\nTesting KNN with half precision for faster computation:")
        points1_half = points1.half()
        points2_half = points2.half()
        
        start_time = time.time()
        knn_result_half = knn_points(points1_half, points2_half, K=5)
        half_time = time.time() - start_time
        
        start_time = time.time()
        knn_result_float = knn_points(points1, points2, K=5)
        float_time = time.time() - start_time
        
        print(f"  - FP16 time: {half_time*1000:.2f}ms")
        print(f"  - FP32 time: {float_time*1000:.2f}ms")
        print(f"  - Speedup: {float_time/half_time:.2f}x")

    print("\n" + "=" * 50)
    print("📍 FURTHEST POINT SAMPLING (FPS)")
    print("=" * 50)

    # Create a larger point cloud for FPS demonstration
    large_points = torch.randn(batch_size, 500, 3, device=device, dtype=torch.float32)
    n_samples = 64
    
    print(f"Downsampling point cloud from {large_points.shape[1]} to {n_samples} points using FPS...")
    
    # Regular FPS
    start_time = time.time()
    fps_indices = furthest_point_sampling(large_points, n_samples)
    fps_time = time.time() - start_time
    
    print("✅ Regular FPS Results:")
    print(f"  - FPS indices shape: {fps_indices.shape}")  # [batch_size, n_samples]
    print(f"  - First point index (always 0): {fps_indices[0, 0].item()}")
    print(f"  - All indices unique: {len(torch.unique(fps_indices[0])) == n_samples}")
    print(f"  - Computation time: {fps_time*1000:.2f}ms")

    # Gather features using FPS indices
    print("\nGathering point features using FPS indices:")
    large_features = torch.randn(batch_size, 6, 500, device=device)  # 6 features per point
    sampled_features = gather_points(large_features, fps_indices)
    print(f"  - Original features shape: {large_features.shape}")
    print(f"  - Sampled features shape: {sampled_features.shape}")

    # Convenience function
    indices, sampled_points = farthest_point_sample_and_gather(large_points, n_samples)
    print(f"  - Convenience function output shapes: {indices.shape}, {sampled_points.shape}")

    print("\n" + "=" * 50)
    print("⚡ QUICK FURTHEST POINT SAMPLING (Accelerated)")
    print("=" * 50)

    # Create an even larger point cloud where Quick FPS shows its benefits
    very_large_points = torch.randn(batch_size, 2000, 3, device=device, dtype=torch.float32)
    n_samples_large = 128
    
    print(f"Downsampling large point cloud from {very_large_points.shape[1]} to {n_samples_large} points...")
    
    # Compare regular FPS vs Quick FPS
    start_time = time.time()
    regular_indices = furthest_point_sampling(very_large_points, n_samples_large)
    regular_time = time.time() - start_time
    
    start_time = time.time()
    quick_indices = quick_furthest_point_sampling(very_large_points, n_samples_large, kd_depth=6)
    quick_time = time.time() - start_time
    
    print("✅ Performance Comparison:")
    print(f"  - Regular FPS time: {regular_time*1000:.2f}ms")
    print(f"  - Quick FPS time: {quick_time*1000:.2f}ms")
    print(f"  - Speedup: {regular_time/quick_time:.2f}x")
    print(f"  - Quick FPS indices shape: {quick_indices.shape}")
    print(f"  - All indices unique: {len(torch.unique(quick_indices[0])) == n_samples_large}")

    # Test different KD-tree depths
    print("\nTesting different KD-tree depths (higher = more spatial partitioning):")
    for kd_depth in [3, 4, 6, 8]:
        start_time = time.time()
        indices = quick_furthest_point_sampling(very_large_points, n_samples_large, kd_depth=kd_depth)
        elapsed = time.time() - start_time
        print(f"  - KD-depth {kd_depth}: {elapsed*1000:.2f}ms")

    # Quick FPS convenience function
    print("\nUsing Quick FPS convenience function:")
    quick_indices, quick_sampled = quick_farthest_point_sample_and_gather(
        very_large_points, n_samples_large, kd_depth=6
    )
    print(f"  - Quick FPS convenience output shapes: {quick_indices.shape}, {quick_sampled.shape}")

    # Multi-precision Quick FPS
    if device == "cuda":
        print("\nQuick FPS with different precisions:")
        
        # FP16
        very_large_half = very_large_points.half()
        start_time = time.time()
        quick_indices_half = quick_furthest_point_sampling(very_large_half, n_samples_large, kd_depth=6)
        half_time = time.time() - start_time
        
        # FP32
        start_time = time.time()
        quick_indices_float = quick_furthest_point_sampling(very_large_points, n_samples_large, kd_depth=6)
        float_time = time.time() - start_time
        
        print(f"  - Quick FPS FP16: {half_time*1000:.2f}ms")
        print(f"  - Quick FPS FP32: {float_time*1000:.2f}ms")
        print(f"  - Memory savings with FP16: ~50%")

    print("\n" + "=" * 50)
    print("🎓 PERFORMANCE INSIGHTS & BEST PRACTICES")
    print("=" * 50)

    print("📊 When to use each method:")
    print("  🔍 KNN:")
    print("     - Use FP16 for up to 6x speedup on modern GPUs")
    print("     - Automatic kernel selection based on problem size")
    print("     - Best for finding local neighborhoods")
    
    print("\n  📍 Regular FPS:")
    print("     - Best for small to medium point clouds (<1000 points)")
    print("     - Guarantees optimal furthest point selection")
    print("     - Use when exact FPS quality is critical")
    
    print("\n  ⚡ Quick FPS:")
    print("     - Best for large point clouds (>1000 points)")
    print("     - Up to 2-3x faster than regular FPS")
    print("     - Maintains >90% quality of regular FPS")
    print("     - KD-depth 4-6 provides good balance of speed vs quality")
    print("     - Higher KD-depths (7-8) for very large/dense point clouds")

    print("\n🚀 GPU Optimization Tips:")
    print("  - Quick FPS automatically adapts to your GPU (A100, H100, etc.)")
    print("  - Use FP16 for 50% memory reduction and faster computation")
    print("  - Batch multiple point clouds together for better GPU utilization")
    print("  - torch.compile can provide additional speedups")

    print("\n" + "=" * 50)
    print("✨ Example completed successfully!")
    print("✨ Check out the tests/ folder for more advanced usage examples.")
    print("🚀 Try running with different point cloud sizes to see the performance benefits!")


if __name__ == "__main__":
    main()
