#!/usr/bin/env python3
"""
Basic Usage Example for torch-point-ops

This example demonstrates how to use the Chamfer Distance and Earth Mover's Distance
functions for comparing point clouds.
"""

import torch
import numpy as np

# Clean import style
from torch_point_ops.chamfer import chamfer_distance
from torch_point_ops.emd import earth_movers_distance

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
    
    print(f"\n📊 Creating sample point clouds:")
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
    
    print(f"✅ Results:")
    print(f"  - Distance from points1 to points2: {dist1.shape} -> mean = {dist1.mean().item():.6f}")
    print(f"  - Distance from points2 to points1: {dist2.shape} -> mean = {dist2.mean().item():.6f}")
    print(f"  - Total Chamfer Distance: {(dist1.mean() + dist2.mean()).item():.6f}")
    
    # Using the Chamfer Distance module
    print("\nUsing ChamferDistance module:")
    from torch_point_ops import ChamferDistance
    
    chamfer_loss = ChamferDistance(reduction='mean')
    total_distance = chamfer_loss(points1, points2)
    print(f"  - Mean Chamfer Distance: {total_distance.item():.6f}")
    
    print("\n" + "=" * 50)
    print("🌍 EARTH MOVER'S DISTANCE (EMD)")
    print("=" * 50)
    
    # For EMD, we typically use smaller point clouds due to computational cost
    small_points1 = points1[:, :50, :]  # Use first 50 points
    small_points2 = points2[:, :50, :]  # Use first 50 points
    
    print(f"Computing EMD with smaller point clouds ({small_points1.shape[1]} points each)...")
    emd_distances = earth_movers_distance(small_points1, small_points2)
    
    print(f"✅ Results:")
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
    print("✨ Example completed successfully!")
    print("✨ Check out the tests/ folder for more advanced usage examples.")

if __name__ == "__main__":
    main() 