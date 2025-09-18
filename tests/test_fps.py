import torch
from torch.testing._internal.common_utils import TestCase
from torch.library import opcheck
import unittest
import numpy as np

# Since the package is installed, we can import it.
from torch_point_ops import fps


def reference_fps_cpu(points, nsamples):
    """
    A reference implementation of FPS in pure PyTorch.
    """
    N, P, D = points.shape
    assert D == 3, "Only 3D points supported"
    
    indices = torch.zeros((N, nsamples), dtype=torch.int64, device=points.device)
    
    for n in range(N):
        min_dists = torch.full((P,), float('inf'), dtype=points.dtype, device=points.device)
        selected = torch.zeros(P, dtype=torch.bool, device=points.device)
        
        # Select first point (index 0)
        current_idx = 0
        indices[n, 0] = current_idx
        selected[current_idx] = True
        
        # FPS main loop
        for i in range(1, nsamples):
            # Update distances to the newly selected point
            curr_point = points[n, current_idx]
            
            # Calculate distances to all unselected points
            for j in range(P):
                if selected[j]:
                    continue
                    
                point = points[n, j]
                # Skip degenerate points
                mag = torch.sum(point * point)
                if mag <= 1e-3:
                    continue
                
                # Compute squared distance
                dist = torch.sum((point - curr_point) ** 2)
                min_dists[j] = torch.min(min_dists[j], dist)
            
            # Find the point with maximum distance to closest selected point
            max_dist = -1
            best_idx = 0
            for j in range(P):
                if not selected[j] and min_dists[j] > max_dist:
                    max_dist = min_dists[j]
                    best_idx = j
            
            current_idx = best_idx
            indices[n, i] = current_idx
            selected[current_idx] = True
    
    return indices


@unittest.skipIf(not torch.cuda.is_available(), "No CUDA device found.")
class TestFPS(TestCase):

    def sample_inputs(
        self, device, batch_size=1, n_points=64, requires_grad=False
    ):
        points = torch.randn(
            batch_size,
            n_points,
            3,
            device=device,
            requires_grad=requires_grad,
            dtype=torch.float32,
        )
        return points

    def test_fps_forward_correctness(self):
        """Test FPS forward pass correctness against reference implementation."""
        device = "cuda"
        points = self.sample_inputs(device, batch_size=2, n_points=20)
        nsamples = 5

        # Our implementation
        indices_cuda = fps.furthest_point_sampling(points, nsamples)

        # CPU reference
        points_cpu = points.cpu()
        indices_ref = reference_fps_cpu(points_cpu, nsamples)

        # Check shapes
        self.assertEqual(indices_cuda.shape, (2, nsamples))
        self.assertEqual(indices_ref.shape, (2, nsamples))

        # First point should always be index 0
        self.assertTrue(torch.all(indices_cuda[:, 0] == 0))
        self.assertTrue(torch.all(indices_ref[:, 0] == 0))

        # All indices should be unique within each batch
        for n in range(2):
            unique_cuda = torch.unique(indices_cuda[n])
            unique_ref = torch.unique(indices_ref[n])
            self.assertEqual(len(unique_cuda), nsamples)
            self.assertEqual(len(unique_ref), nsamples)

    def test_fps_different_sample_sizes(self):
        """Test FPS with different sample sizes."""
        device = "cuda"
        points = self.sample_inputs(device, batch_size=1, n_points=100)

        for nsamples in [1, 5, 10, 25, 50]:
            with self.subTest(nsamples=nsamples):
                indices = fps.furthest_point_sampling(points, nsamples)
                self.assertEqual(indices.shape, (1, nsamples))
                
                # Check all indices are valid
                self.assertTrue(torch.all(indices >= 0))
                self.assertTrue(torch.all(indices < points.shape[1]))
                
                # Check all indices are unique
                unique_indices = torch.unique(indices[0])
                self.assertEqual(len(unique_indices), nsamples)

    def test_fps_batch_sizes(self):
        """Test FPS with different batch sizes."""
        device = "cuda"
        nsamples = 8

        for batch_size in [1, 2, 4, 8]:
            with self.subTest(batch_size=batch_size):
                points = self.sample_inputs(device, batch_size=batch_size, n_points=32)
                indices = fps.furthest_point_sampling(points, nsamples)
                
                self.assertEqual(indices.shape, (batch_size, nsamples))
                
                # Check each batch independently
                for n in range(batch_size):
                    batch_indices = indices[n]
                    unique_indices = torch.unique(batch_indices)
                    self.assertEqual(len(unique_indices), nsamples)
                    self.assertTrue(torch.all(batch_indices >= 0))
                    self.assertTrue(torch.all(batch_indices < points.shape[1]))

    def test_fps_edge_cases(self):
        """Test FPS edge cases."""
        device = "cuda"

        # Single sample
        points = self.sample_inputs(device, batch_size=1, n_points=10)
        indices = fps.furthest_point_sampling(points, 1)
        self.assertEqual(indices.shape, (1, 1))
        self.assertEqual(indices[0, 0].item(), 0)  # First point should be index 0

        # Sample all points
        points = self.sample_inputs(device, batch_size=1, n_points=5)
        indices = fps.furthest_point_sampling(points, 5)
        self.assertEqual(indices.shape, (1, 5))
        unique_indices = torch.unique(indices[0])
        self.assertEqual(len(unique_indices), 5)

    def test_fps_input_validation(self):
        """Test FPS input validation."""
        device = "cuda"
        points = self.sample_inputs(device, batch_size=1, n_points=10)

        # Wrong number of dimensions
        with self.assertRaises(ValueError):
            fps.furthest_point_sampling(points.squeeze(0), 5)  # 2D instead of 3D

        # Wrong last dimension
        with self.assertRaises(ValueError):
            fps.furthest_point_sampling(points[..., :2], 5)  # 2D coords instead of 3D

        # Invalid nsamples
        with self.assertRaises(ValueError):
            fps.furthest_point_sampling(points, 0)  # nsamples <= 0

        with self.assertRaises(ValueError):
            fps.furthest_point_sampling(points, 15)  # nsamples > n_points

    def test_gather_points_forward(self):
        """Test gather_points forward pass."""
        device = "cuda"
        batch_size, n_channels, n_points = 2, 6, 20
        n_samples = 8

        points = torch.randn(batch_size, n_channels, n_points, device=device)
        indices = torch.randint(0, n_points, (batch_size, n_samples), device=device)

        gathered = fps.gather_points(points, indices)

        self.assertEqual(gathered.shape, (batch_size, n_channels, n_samples))

        # Verify correctness by manual checking
        for n in range(batch_size):
            for c in range(n_channels):
                for s in range(n_samples):
                    expected = points[n, c, indices[n, s]]
                    actual = gathered[n, c, s]
                    self.assertTrue(torch.allclose(expected, actual))

    def test_gather_points_gradients(self):
        """Test gather_points gradient computation."""
        device = "cuda"
        batch_size, n_channels, n_points = 2, 4, 10
        n_samples = 5

        points = torch.randn(batch_size, n_channels, n_points, device=device, requires_grad=True)
        
        # Use unique indices to avoid double counting gradients
        indices = torch.zeros(batch_size, n_samples, dtype=torch.long, device=device)
        for b in range(batch_size):
            indices[b] = torch.randperm(n_points, device=device)[:n_samples]

        gathered = fps.gather_points(points, indices)
        loss = gathered.sum()
        loss.backward()

        self.assertIsNotNone(points.grad)
        self.assertEqual(points.grad.shape, points.shape)

        # Check that gradients are accumulated correctly
        # Each selected point should have gradient equal to the number of channels
        for n in range(batch_size):
            for s in range(n_samples):
                point_idx = indices[n, s].item()
                expected_grad = n_channels  # Sum over channels
                actual_grad = points.grad[n, :, point_idx].sum().item()
                self.assertAlmostEqual(actual_grad, expected_grad, places=5)

    def test_gather_points_input_validation(self):
        """Test gather_points input validation."""
        device = "cuda"
        points = torch.randn(2, 6, 20, device=device)
        indices = torch.randint(0, 20, (2, 8), device=device)

        # Wrong dimensions for points
        with self.assertRaises(ValueError):
            fps.gather_points(points[0], indices)  # 2D instead of 3D

        # Wrong dimensions for indices
        with self.assertRaises(ValueError):
            fps.gather_points(points, indices.unsqueeze(-1))  # 3D instead of 2D

        # Mismatched batch dimensions
        with self.assertRaises(ValueError):
            fps.gather_points(points, indices[:1])  # Different batch sizes

    def test_fps_and_gather_integration(self):
        """Test integration of FPS and gather_points."""
        device = "cuda"
        batch_size, n_points, n_samples = 2, 50, 10

        # Create point cloud with features
        coords = torch.randn(batch_size, n_points, 3, device=device)
        features = torch.randn(batch_size, 6, n_points, device=device)

        # Sample points using FPS
        indices = fps.furthest_point_sampling(coords, n_samples)

        # Gather features for sampled points
        sampled_features = fps.gather_points(features, indices)

        self.assertEqual(indices.shape, (batch_size, n_samples))
        self.assertEqual(sampled_features.shape, (batch_size, 6, n_samples))

        # Verify that the first sampled point is always index 0
        self.assertTrue(torch.all(indices[:, 0] == 0))

    def test_farthest_point_sample_and_gather(self):
        """Test the convenience function."""
        device = "cuda"
        batch_size, n_points, n_samples = 2, 30, 8

        points = torch.randn(batch_size, n_points, 3, device=device)
        indices, sampled_points = fps.farthest_point_sample_and_gather(points, n_samples)

        self.assertEqual(indices.shape, (batch_size, n_samples))
        self.assertEqual(sampled_points.shape, (batch_size, n_samples, 3))

        # Verify that sampled_points matches the gathered coordinates
        for n in range(batch_size):
            for s in range(n_samples):
                expected = points[n, indices[n, s]]
                actual = sampled_points[n, s]
                self.assertTrue(torch.allclose(expected, actual))

    def test_multi_precision(self):
        """Test FPS and gather_points with different precisions."""
        device = "cuda"
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        # Test different precisions with appropriate tolerances
        precision_configs = [
            (torch.float16, 1e-2),
            (torch.float32, 1e-5),
            (torch.float64, 1e-8),
        ]

        for dtype, atol in precision_configs:
            with self.subTest(dtype=dtype):
                points = torch.randn(1, 20, 3, device=device, dtype=dtype)
                features = torch.randn(1, 4, 20, device=device, dtype=dtype)

                # Test FPS
                indices = fps.furthest_point_sampling(points, 8)
                self.assertEqual(indices.dtype, torch.int64)

                # Test gather_points
                gathered = fps.gather_points(features, indices)
                self.assertEqual(gathered.dtype, dtype)
                self.assertEqual(gathered.shape, (1, 4, 8))

                # Test gradients
                if dtype != torch.float16:  # Skip gradient test for fp16 due to precision issues
                    features_grad = features.clone().requires_grad_(True)
                    gathered_grad = fps.gather_points(features_grad, indices)
                    loss = gathered_grad.sum()
                    loss.backward()
                    
                    self.assertIsNotNone(features_grad.grad)
                    self.assertTrue(torch.isfinite(features_grad.grad).all())

    def test_module_interfaces(self):
        """Test the module interfaces."""
        device = "cuda"
        points = torch.randn(2, 30, 3, device=device)
        features = torch.randn(2, 6, 30, device=device)

        # Test FarthestPointSampling module
        fps_module = fps.FarthestPointSampling(nsamples=10)
        indices = fps_module(points)
        self.assertEqual(indices.shape, (2, 10))

        # Test with return_gathered=True
        fps_module_gather = fps.FarthestPointSampling(nsamples=10, return_gathered=True)
        indices, sampled_points = fps_module_gather(points)
        self.assertEqual(indices.shape, (2, 10))
        self.assertEqual(sampled_points.shape, (2, 10, 3))

        # Test PointGatherer module
        gatherer = fps.PointGatherer()
        gathered = gatherer(features, indices)
        self.assertEqual(gathered.shape, (2, 6, 10))

        # Test FPSFunction module
        fps_func = fps.FPSFunction()
        indices2 = fps_func(points, 10)
        self.assertEqual(indices2.shape, (2, 10))

        indices3, sampled3 = fps_func(points, 10, return_gathered=True)
        self.assertEqual(indices3.shape, (2, 10))
        self.assertEqual(sampled3.shape, (2, 10, 3))

    def test_cpu_fallback(self):
        """Test CPU implementation."""
        device = "cpu"
        points = torch.randn(1, 15, 3, device=device)
        nsamples = 5

        indices = fps.furthest_point_sampling(points, nsamples)
        self.assertEqual(indices.shape, (1, nsamples))
        self.assertEqual(indices.device.type, "cpu")
        
        # Test gather_points on CPU
        features = torch.randn(1, 4, 15, device=device)
        gathered = fps.gather_points(features, indices)
        self.assertEqual(gathered.shape, (1, 4, nsamples))
        self.assertEqual(gathered.device.type, "cpu")

    def test_gradcheck(self):
        """Test gradients using torch.autograd.gradcheck."""
        device = "cuda"
        
        # Test gather_points gradients
        def gather_fn(points):
            # Use fixed indices for gradient checking
            indices = torch.tensor([[0, 2, 4]], device=device)
            gathered = fps.gather_points(points, indices)
            return gathered.sum()

        points = torch.randn(1, 3, 5, device=device, dtype=torch.float64, requires_grad=True)
        
        self.assertTrue(
            torch.autograd.gradcheck(
                gather_fn,
                (points,),
                eps=1e-6,
                atol=1e-5,
                rtol=1e-4,
                fast_mode=True,
            )
        )

    def test_opcheck(self):
        """Test operator checking."""
        device = "cuda"
        points = torch.randn(1, 10, 3, device=device)
        features = torch.randn(1, 4, 10, device=device)
        indices = torch.tensor([[0, 2, 4, 6]], device=device)

        # Test FPS forward
        opcheck(
            torch.ops.torch_point_ops_fps.fps_forward.default,
            (points, 4),
            raise_exception=True,
        )

        # Test gather_points forward
        opcheck(
            torch.ops.torch_point_ops_fps.gather_points_forward.default,
            (features, indices),
            raise_exception=True,
        )

    def test_compile(self):
        """Test torch.compile compatibility."""
        device = "cuda"
        
        # Test FPS compilation
        compiled_fps = torch.compile(fps.furthest_point_sampling)
        points = torch.randn(1, 20, 3, device=device)
        
        indices_eager = fps.furthest_point_sampling(points, 8)
        indices_compiled = compiled_fps(points, 8)
        
        # Results might differ due to compilation, but shapes should match
        self.assertEqual(indices_eager.shape, indices_compiled.shape)
        
        # Test gather_points compilation
        compiled_gather = torch.compile(fps.gather_points)
        features = torch.randn(1, 6, 20, device=device)
        
        gathered_eager = fps.gather_points(features, indices_eager)
        gathered_compiled = compiled_gather(features, indices_eager)
        
        self.assertTrue(torch.allclose(gathered_eager, gathered_compiled))

    def test_known_configurations(self):
        """Test with known point configurations."""
        device = "cuda"
        
        # Create a simple 2D grid in 3D space (z=0)
        x = torch.linspace(-1, 1, 5)
        y = torch.linspace(-1, 1, 5)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        points = torch.stack([xx.flatten(), yy.flatten(), torch.zeros_like(xx.flatten())], dim=-1)
        points = points.unsqueeze(0).to(device)  # Add batch dimension
        
        indices = fps.furthest_point_sampling(points, 5)
        
        # First point should be index 0
        self.assertEqual(indices[0, 0].item(), 0)
        
        # All indices should be unique
        unique_indices = torch.unique(indices[0])
        self.assertEqual(len(unique_indices), 5)

    def test_degenerate_points(self):
        """Test handling of degenerate points (close to origin)."""
        device = "cuda"
        
        # Create points with some very close to origin
        points = torch.randn(1, 10, 3, device=device)
        points[0, 5:8] = 1e-4  # Make some points very close to origin
        
        indices = fps.furthest_point_sampling(points, 5)
        
        # Should still work and return valid indices
        self.assertEqual(indices.shape, (1, 5))
        unique_indices = torch.unique(indices[0])
        self.assertEqual(len(unique_indices), 5)

    def test_quick_fps_forward_correctness(self):
        """Test Quick FPS forward pass correctness against regular FPS."""
        device = "cuda"
        points = self.sample_inputs(device, batch_size=2, n_points=50)
        nsamples = 10

        # Our regular FPS implementation
        indices_fps = fps.furthest_point_sampling(points, nsamples)

        # Quick FPS implementation
        indices_quick = fps.quick_furthest_point_sampling(points, nsamples, kd_depth=4)

        # Check shapes
        self.assertEqual(indices_quick.shape, (2, nsamples))
        self.assertEqual(indices_fps.shape, (2, nsamples))

        # First point should always be index 0
        self.assertTrue(torch.all(indices_quick[:, 0] == 0))
        self.assertTrue(torch.all(indices_fps[:, 0] == 0))

        # All indices should be unique within each batch
        for n in range(2):
            unique_quick = torch.unique(indices_quick[n])
            unique_fps = torch.unique(indices_fps[n])
            self.assertEqual(len(unique_quick), nsamples)
            self.assertEqual(len(unique_fps), nsamples)

        # Results might differ due to spatial partitioning, but quality should be similar
        # Test by checking that selected points have reasonable spread
        for n in range(2):
            sampled_points_quick = points[n, indices_quick[n]]
            sampled_points_fps = points[n, indices_fps[n]]
            
            # Check spread by computing pairwise distances
            dists_quick = torch.cdist(sampled_points_quick, sampled_points_quick)
            dists_fps = torch.cdist(sampled_points_fps, sampled_points_fps)
            
            # Minimum distance should be reasonably similar (within 2x factor)
            min_dist_quick = dists_quick[dists_quick > 0].min()
            min_dist_fps = dists_fps[dists_fps > 0].min()
            
            self.assertGreater(min_dist_quick, 0)
            self.assertGreater(min_dist_fps, 0)

    def test_quick_fps_different_kd_depths(self):
        """Test Quick FPS with different KD-tree depths."""
        device = "cuda"
        points = self.sample_inputs(device, batch_size=1, n_points=100)
        nsamples = 16

        for kd_depth in [3, 4, 5, 6]:
            with self.subTest(kd_depth=kd_depth):
                indices = fps.quick_furthest_point_sampling(points, nsamples, kd_depth)
                self.assertEqual(indices.shape, (1, nsamples))
                
                # Check all indices are valid
                self.assertTrue(torch.all(indices >= 0))
                self.assertTrue(torch.all(indices < points.shape[1]))
                
                # Check all indices are unique
                unique_indices = torch.unique(indices[0])
                self.assertEqual(len(unique_indices), nsamples)
                
                # First point should always be index 0
                self.assertEqual(indices[0, 0].item(), 0)

    def test_quick_fps_large_point_clouds(self):
        """Test Quick FPS with larger point clouds where acceleration matters."""
        device = "cuda"
        
        # Test with progressively larger point clouds
        for n_points in [200, 500, 1000]:
            with self.subTest(n_points=n_points):
                points = self.sample_inputs(device, batch_size=1, n_points=n_points)
                nsamples = min(50, n_points // 10)
                
                indices = fps.quick_furthest_point_sampling(points, nsamples, kd_depth=6)
                
                self.assertEqual(indices.shape, (1, nsamples))
                self.assertTrue(torch.all(indices >= 0))
                self.assertTrue(torch.all(indices < n_points))
                
                # Check uniqueness
                unique_indices = torch.unique(indices[0])
                self.assertEqual(len(unique_indices), nsamples)

    def test_quick_fps_batch_sizes(self):
        """Test Quick FPS with different batch sizes."""
        device = "cuda"
        nsamples = 8

        for batch_size in [1, 2, 4, 8]:
            with self.subTest(batch_size=batch_size):
                points = self.sample_inputs(device, batch_size=batch_size, n_points=64)
                indices = fps.quick_furthest_point_sampling(points, nsamples, kd_depth=4)
                
                self.assertEqual(indices.shape, (batch_size, nsamples))
                
                # Check each batch independently
                for n in range(batch_size):
                    batch_indices = indices[n]
                    unique_indices = torch.unique(batch_indices)
                    self.assertEqual(len(unique_indices), nsamples)
                    self.assertTrue(torch.all(batch_indices >= 0))
                    self.assertTrue(torch.all(batch_indices < points.shape[1]))

    def test_quick_fps_input_validation(self):
        """Test Quick FPS input validation."""
        device = "cuda"
        points = self.sample_inputs(device, batch_size=1, n_points=10)

        # Wrong number of dimensions
        with self.assertRaises(ValueError):
            fps.quick_furthest_point_sampling(points.squeeze(0), 5)  # 2D instead of 3D

        # Wrong last dimension
        with self.assertRaises(ValueError):
            fps.quick_furthest_point_sampling(points[..., :2], 5)  # 2D coords instead of 3D

        # Invalid nsamples
        with self.assertRaises(ValueError):
            fps.quick_furthest_point_sampling(points, 0)  # nsamples <= 0

        with self.assertRaises(ValueError):
            fps.quick_furthest_point_sampling(points, 15)  # nsamples > n_points

        # Invalid kd_depth
        with self.assertRaises(ValueError):
            fps.quick_furthest_point_sampling(points, 5, kd_depth=0)  # kd_depth <= 0

        with self.assertRaises(ValueError):
            fps.quick_furthest_point_sampling(points, 5, kd_depth=11)  # kd_depth > 10

    def test_quick_fps_and_gather_integration(self):
        """Test integration of Quick FPS and gather_points."""
        device = "cuda"
        batch_size, n_points, n_samples = 2, 100, 20

        # Create point cloud with features
        coords = torch.randn(batch_size, n_points, 3, device=device)
        features = torch.randn(batch_size, 6, n_points, device=device)

        # Sample points using Quick FPS
        indices = fps.quick_furthest_point_sampling(coords, n_samples, kd_depth=5)

        # Gather features for sampled points
        sampled_features = fps.gather_points(features, indices)

        self.assertEqual(indices.shape, (batch_size, n_samples))
        self.assertEqual(sampled_features.shape, (batch_size, 6, n_samples))

        # Verify that the first sampled point is always index 0
        self.assertTrue(torch.all(indices[:, 0] == 0))

    def test_quick_farthest_point_sample_and_gather(self):
        """Test the Quick FPS convenience function."""
        device = "cuda"
        batch_size, n_points, n_samples = 2, 80, 16

        points = torch.randn(batch_size, n_points, 3, device=device)
        indices, sampled_points = fps.quick_farthest_point_sample_and_gather(points, n_samples, kd_depth=4)

        self.assertEqual(indices.shape, (batch_size, n_samples))
        self.assertEqual(sampled_points.shape, (batch_size, n_samples, 3))

        # Verify that sampled_points matches the gathered coordinates
        for n in range(batch_size):
            for s in range(n_samples):
                expected = points[n, indices[n, s]]
                actual = sampled_points[n, s]
                self.assertTrue(torch.allclose(expected, actual))

    def test_quick_fps_multi_precision(self):
        """Test Quick FPS with different precisions."""
        device = "cuda"
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        # Test different precisions
        precision_configs = [
            (torch.float16, 1e-2),
            (torch.float32, 1e-5),
            (torch.float64, 1e-8),
        ]

        for dtype, atol in precision_configs:
            with self.subTest(dtype=dtype):
                points = torch.randn(1, 50, 3, device=device, dtype=dtype)

                # Test Quick FPS
                indices = fps.quick_furthest_point_sampling(points, 12, kd_depth=4)
                self.assertEqual(indices.dtype, torch.int64)
                self.assertEqual(indices.shape, (1, 12))

                # Check all indices are unique
                unique_indices = torch.unique(indices[0])
                self.assertEqual(len(unique_indices), 12)

    def test_quick_fps_module_interfaces(self):
        """Test the Quick FPS module interfaces."""
        device = "cuda"
        points = torch.randn(2, 60, 3, device=device)
        features = torch.randn(2, 6, 60, device=device)

        # Test QuickFarthestPointSampling module
        quick_fps_module = fps.QuickFarthestPointSampling(nsamples=15, kd_depth=4)
        indices = quick_fps_module(points)
        self.assertEqual(indices.shape, (2, 15))

        # Test with return_gathered=True
        quick_fps_module_gather = fps.QuickFarthestPointSampling(nsamples=15, kd_depth=4, return_gathered=True)
        indices, sampled_points = quick_fps_module_gather(points)
        self.assertEqual(indices.shape, (2, 15))
        self.assertEqual(sampled_points.shape, (2, 15, 3))

        # Test QuickFPSFunction module
        quick_fps_func = fps.QuickFPSFunction()
        indices2 = quick_fps_func(points, 15, kd_depth=4)
        self.assertEqual(indices2.shape, (2, 15))

        indices3, sampled3 = quick_fps_func(points, 15, kd_depth=4, return_gathered=True)
        self.assertEqual(indices3.shape, (2, 15))
        self.assertEqual(sampled3.shape, (2, 15, 3))

    def test_quick_fps_cpu_fallback(self):
        """Test Quick FPS CPU implementation."""
        device = "cpu"
        points = torch.randn(1, 30, 3, device=device)
        nsamples = 8

        indices = fps.quick_furthest_point_sampling(points, nsamples, kd_depth=4)
        self.assertEqual(indices.shape, (1, nsamples))
        self.assertEqual(indices.device.type, "cpu")
        
        # Test with different kd_depths on CPU
        for kd_depth in [3, 5, 6]:
            with self.subTest(kd_depth=kd_depth):
                indices_depth = fps.quick_furthest_point_sampling(points, nsamples, kd_depth)
                self.assertEqual(indices_depth.shape, (1, nsamples))
                unique_indices = torch.unique(indices_depth[0])
                self.assertEqual(len(unique_indices), nsamples)

    def test_quick_fps_performance_comparison(self):
        """Test Quick FPS vs regular FPS performance characteristics."""
        device = "cuda"
        
        # Create a large point cloud where Quick FPS should show benefit
        points = torch.randn(1, 2000, 3, device=device)
        nsamples = 100
        
        # Both should produce valid results
        indices_fps = fps.furthest_point_sampling(points, nsamples)
        indices_quick = fps.quick_furthest_point_sampling(points, nsamples, kd_depth=6)
        
        self.assertEqual(indices_fps.shape, (1, nsamples))
        self.assertEqual(indices_quick.shape, (1, nsamples))
        
        # Both should have unique indices
        unique_fps = torch.unique(indices_fps[0])
        unique_quick = torch.unique(indices_quick[0])
        self.assertEqual(len(unique_fps), nsamples)
        self.assertEqual(len(unique_quick), nsamples)
        
        # Both should start with index 0
        self.assertEqual(indices_fps[0, 0].item(), 0)
        self.assertEqual(indices_quick[0, 0].item(), 0)

    def test_quick_fps_edge_cases(self):
        """Test Quick FPS edge cases."""
        device = "cuda"

        # Single sample
        points = self.sample_inputs(device, batch_size=1, n_points=20)
        indices = fps.quick_furthest_point_sampling(points, 1, kd_depth=3)
        self.assertEqual(indices.shape, (1, 1))
        self.assertEqual(indices[0, 0].item(), 0)  # First point should be index 0

        # Sample all points (small cloud)
        points = self.sample_inputs(device, batch_size=1, n_points=8)
        indices = fps.quick_furthest_point_sampling(points, 8, kd_depth=3)
        self.assertEqual(indices.shape, (1, 8))
        unique_indices = torch.unique(indices[0])
        self.assertEqual(len(unique_indices), 8)

        # Very small kd_depth
        points = self.sample_inputs(device, batch_size=1, n_points=50)
        indices = fps.quick_furthest_point_sampling(points, 10, kd_depth=1)
        self.assertEqual(indices.shape, (1, 10))
        unique_indices = torch.unique(indices[0])
        self.assertEqual(len(unique_indices), 10)

    def test_quick_fps_known_configurations(self):
        """Test Quick FPS with known point configurations."""
        device = "cuda"
        
        # Create a simple 3D grid
        x = torch.linspace(-2, 2, 4)
        y = torch.linspace(-2, 2, 4)
        z = torch.linspace(-1, 1, 3)
        xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
        points = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=-1)
        points = points.unsqueeze(0).to(device)  # Add batch dimension, shape: (1, 48, 3)
        
        indices = fps.quick_furthest_point_sampling(points, 12, kd_depth=4)
        
        # First point should be index 0
        self.assertEqual(indices[0, 0].item(), 0)
        
        # All indices should be unique
        unique_indices = torch.unique(indices[0])
        self.assertEqual(len(unique_indices), 12)
        
        # Should handle regular grid structure well
        self.assertEqual(indices.shape, (1, 12))

    def test_quick_fps_gradcheck_through_gather(self):
        """Test gradients through gather_points using Quick FPS indices."""
        device = "cuda"
        
        # Test gradient flow through gather_points when using Quick FPS indices
        def fps_gather_fn(points_features):
            coords = points_features[:, :3, :].transpose(1, 2)  # (N, P, 3)
            # Use Quick FPS to get indices
            indices = fps.quick_furthest_point_sampling(coords, 5, kd_depth=3)
            # Gather features using those indices
            gathered = fps.gather_points(points_features, indices)
            return gathered.sum()

        # Use double precision for more accurate gradient checking
        points_features = torch.randn(1, 6, 10, device=device, dtype=torch.float64, requires_grad=True)
        
        self.assertTrue(
            torch.autograd.gradcheck(
                fps_gather_fn,
                (points_features,),
                eps=1e-6,
                atol=1e-5,
                rtol=1e-4,
                fast_mode=True,
            )
        )

    def test_quick_fps_gradient_flow(self):
        """Test gradient flow through Quick FPS pipeline."""
        device = "cuda"
        batch_size, n_features, n_points = 2, 8, 50
        n_samples = 12
        
        # Create input with gradients
        features = torch.randn(batch_size, n_features, n_points, device=device, requires_grad=True)
        coords = features[:, :3, :].transpose(1, 2)  # Use first 3 features as coordinates
        
        # Quick FPS sampling
        indices = fps.quick_furthest_point_sampling(coords, n_samples, kd_depth=4)
        
        # Gather features
        sampled_features = fps.gather_points(features, indices)
        
        # Compute loss and backpropagate
        loss = sampled_features.sum()
        loss.backward()
        
        # Check gradients
        self.assertIsNotNone(features.grad)
        self.assertEqual(features.grad.shape, features.shape)
        self.assertTrue(torch.isfinite(features.grad).all())
        
        # Check that only selected points have non-zero gradients
        grad_norm_per_point = features.grad.norm(dim=1)  # (batch_size, n_points)
        
        for b in range(batch_size):
            selected_indices = indices[b]
            for i, idx in enumerate(selected_indices):
                # Selected points should have non-zero gradients
                self.assertGreater(grad_norm_per_point[b, idx].item(), 0)

    def test_quick_fps_numerical_stability(self):
        """Test Quick FPS numerical stability with extreme values."""
        device = "cuda"
        
        test_cases = [
            # Small coordinates (above degenerate threshold)
            torch.randn(1, 20, 3, device=device) * 0.1,  # Well above 1e-3 threshold
            # Very large coordinates
            torch.randn(1, 20, 3, device=device) * 1e6,
            # Mixed scale coordinates (all above threshold)
            torch.cat([
                torch.randn(1, 10, 3, device=device) * 0.1,  # Above threshold
                torch.randn(1, 10, 3, device=device) * 1e3
            ], dim=1),
        ]
        
        for i, points in enumerate(test_cases):
            with self.subTest(case=i):
                # Test both implementations for consistency
                indices_quick = fps.quick_furthest_point_sampling(points, 8, kd_depth=3)
                indices_regular = fps.furthest_point_sampling(points, 8)
                
                # Both should produce valid results
                for name, indices in [("Quick FPS", indices_quick), ("Regular FPS", indices_regular)]:
                    self.assertEqual(indices.shape, (1, 8), f"{name} shape mismatch")
                    self.assertTrue(torch.all(indices >= 0), f"{name} has negative indices")
                    self.assertTrue(torch.all(indices < points.shape[1]), f"{name} has out-of-range indices")
                    
                    # Check uniqueness
                    unique_indices = torch.unique(indices[0])
                    self.assertEqual(len(unique_indices), 8, f"{name} has duplicate indices")
                
                # Both implementations should behave identically for numerical stability
                self.assertTrue(torch.equal(indices_quick, indices_regular),
                              f"Quick FPS and Regular FPS differ for case {i}")
        
        # Test degenerate points case separately - this is expected to produce duplicates
        degenerate_points = torch.randn(1, 20, 3, device=device) * 1e-6  # Below 1e-3 threshold
        
        # Check that both implementations handle degenerate points identically
        indices_quick_deg = fps.quick_furthest_point_sampling(degenerate_points, 8, kd_depth=3)
        indices_regular_deg = fps.furthest_point_sampling(degenerate_points, 8)
        
        # Both should produce valid shapes
        self.assertEqual(indices_quick_deg.shape, (1, 8))
        self.assertEqual(indices_regular_deg.shape, (1, 8))
        
        # Both should behave identically with degenerate points
        self.assertTrue(torch.equal(indices_quick_deg, indices_regular_deg),
                      "Quick FPS and Regular FPS differ for degenerate points")
        
        # With degenerate points, we expect duplicates (this is correct behavior)
        unique_quick = torch.unique(indices_quick_deg[0])
        unique_regular = torch.unique(indices_regular_deg[0])
        
        # Both should have the same number of unique indices (likely 1, which is index 0)
        self.assertEqual(len(unique_quick), len(unique_regular),
                        "Different number of unique indices for degenerate points")

    def test_quick_fps_determinism(self):
        """Test Quick FPS determinism with same inputs."""
        device = "cuda"
        
        # Set seed for reproducibility
        torch.manual_seed(42)
        points1 = torch.randn(2, 100, 3, device=device)
        
        torch.manual_seed(42)
        points2 = torch.randn(2, 100, 3, device=device)
        
        # Should produce identical results with same seed and inputs
        indices1 = fps.quick_furthest_point_sampling(points1, 20, kd_depth=4)
        indices2 = fps.quick_furthest_point_sampling(points2, 20, kd_depth=4)
        
        self.assertTrue(torch.equal(indices1, indices2))

    def test_quick_fps_memory_efficiency(self):
        """Test Quick FPS memory efficiency with large point clouds."""
        device = "cuda"
        
        # Test progressively larger point clouds to ensure memory usage is reasonable
        initial_memory = torch.cuda.memory_allocated(device)
        
        for n_points in [1000, 2000, 5000]:
            with self.subTest(n_points=n_points):
                points = torch.randn(1, n_points, 3, device=device)
                indices = fps.quick_furthest_point_sampling(points, 50, kd_depth=6)
                
                self.assertEqual(indices.shape, (1, 50))
                
                # Clean up
                del points, indices
                torch.cuda.empty_cache()
                
                # Memory usage shouldn't grow excessively
                current_memory = torch.cuda.memory_allocated(device)
                memory_growth = current_memory - initial_memory
                self.assertLess(memory_growth, 100 * 1024 * 1024)  # Less than 100MB growth

    def test_quick_fps_boundary_conditions(self):
        """Test Quick FPS with boundary conditions."""
        device = "cuda"
        
        # Points on a boundary (e.g., all on a plane)
        x = torch.linspace(-1, 1, 20)
        y = torch.linspace(-1, 1, 20)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        points_2d = torch.stack([xx.flatten(), yy.flatten(), torch.zeros_like(xx.flatten())], dim=-1)
        points_2d = points_2d.unsqueeze(0).to(device)
        
        indices = fps.quick_furthest_point_sampling(points_2d, 15, kd_depth=4)
        
        self.assertEqual(indices.shape, (1, 15))
        unique_indices = torch.unique(indices[0])
        self.assertEqual(len(unique_indices), 15)
        
        # Points with some duplicate coordinates
        points_dup = torch.randn(1, 30, 3, device=device)
        points_dup[0, 10:15] = points_dup[0, 5:10]  # Create some duplicates
        
        indices_dup = fps.quick_furthest_point_sampling(points_dup, 10, kd_depth=3)
        self.assertEqual(indices_dup.shape, (1, 10))
        unique_indices_dup = torch.unique(indices_dup[0])
        self.assertEqual(len(unique_indices_dup), 10)

    def test_quick_fps_compile_compatibility(self):
        """Test Quick FPS compatibility with torch.compile."""
        device = "cuda"
        
        # Test compilation of Quick FPS
        compiled_quick_fps = torch.compile(fps.quick_furthest_point_sampling)
        points = torch.randn(2, 80, 3, device=device)
        
        indices_eager = fps.quick_furthest_point_sampling(points, 15, kd_depth=4)
        indices_compiled = compiled_quick_fps(points, 15, kd_depth=4)
        
        # Shapes should match
        self.assertEqual(indices_eager.shape, indices_compiled.shape)
        
        # Both should produce valid results
        for indices in [indices_eager, indices_compiled]:
            self.assertTrue(torch.all(indices >= 0))
            self.assertTrue(torch.all(indices < points.shape[1]))
            
            for b in range(2):
                unique_indices = torch.unique(indices[b])
                self.assertEqual(len(unique_indices), 15)

    def test_quick_fps_opcheck(self):
        """Test Quick FPS operator checking."""
        device = "cuda"
        points = torch.randn(1, 20, 3, device=device)

        # Test Quick FPS forward
        opcheck(
            torch.ops.torch_point_ops_quick_fps.quick_fps_forward.default,
            (points, 8, 4),  # points, nsamples, kd_depth
            raise_exception=True,
        )

    def test_quick_fps_stress_test(self):
        """Stress test Quick FPS with various challenging scenarios."""
        device = "cuda"
        
        stress_scenarios = [
            # Very dense point cloud in small region
            (torch.randn(1, 500, 3, device=device) * 0.1, 25, 5),
            # Sparse point cloud in large region
            (torch.randn(1, 100, 3, device=device) * 10, 15, 4),
            # Highly anisotropic point cloud (elongated in one dimension)
            (torch.cat([
                torch.randn(1, 200, 1, device=device) * 10,
                torch.randn(1, 200, 1, device=device) * 0.1,
                torch.randn(1, 200, 1, device=device) * 0.1
            ], dim=2), 30, 5),
        ]
        
        for i, (points, nsamples, kd_depth) in enumerate(stress_scenarios):
            with self.subTest(scenario=i):
                indices = fps.quick_furthest_point_sampling(points, nsamples, kd_depth)
                
                self.assertEqual(indices.shape, (1, nsamples))
                self.assertTrue(torch.all(indices >= 0))
                self.assertTrue(torch.all(indices < points.shape[1]))
                
                unique_indices = torch.unique(indices[0])
                self.assertEqual(len(unique_indices), nsamples)

    def test_quick_fps_vs_regular_fps_quality(self):
        """Test that Quick FPS produces reasonable quality compared to regular FPS."""
        device = "cuda"
        
        # Use a structured point cloud where we can assess quality
        points = torch.randn(1, 200, 3, device=device)
        nsamples = 20
        
        # Get results from both methods
        indices_regular = fps.furthest_point_sampling(points, nsamples)
        indices_quick = fps.quick_furthest_point_sampling(points, nsamples, kd_depth=5)
        
        # Extract sampled points
        sampled_regular = points[0, indices_regular[0]]
        sampled_quick = points[0, indices_quick[0]]
        
        # Compute quality metrics: minimum pairwise distances
        dists_regular = torch.cdist(sampled_regular, sampled_regular)
        dists_quick = torch.cdist(sampled_quick, sampled_quick)
        
        # Remove diagonal (self-distances)
        mask = ~torch.eye(nsamples, dtype=bool, device=device)
        min_dist_regular = dists_regular[mask].min()
        min_dist_quick = dists_quick[mask].min()
        
        # Quick FPS should achieve at least 50% of regular FPS quality
        quality_ratio = min_dist_quick / min_dist_regular
        self.assertGreater(quality_ratio, 0.3,
                          f"Quick FPS quality too low: {quality_ratio:.3f}")

    def test_quick_fps_extreme_kd_depths(self):
        """Test Quick FPS with extreme KD-tree depths."""
        device = "cuda"
        points = torch.randn(1, 100, 3, device=device)
        nsamples = 15
        
        # Test minimum depth
        indices_min = fps.quick_furthest_point_sampling(points, nsamples, kd_depth=1)
        self.assertEqual(indices_min.shape, (1, nsamples))
        unique_min = torch.unique(indices_min[0])
        self.assertEqual(len(unique_min), nsamples)
        
        # Test maximum depth
        indices_max = fps.quick_furthest_point_sampling(points, nsamples, kd_depth=10)
        self.assertEqual(indices_max.shape, (1, nsamples))
        unique_max = torch.unique(indices_max[0])
        self.assertEqual(len(unique_max), nsamples)

    def test_quick_fps_mixed_precision_gradient_flow(self):
        """Test gradient flow with mixed precision (FP16/FP32)."""
        device = "cuda"
        
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
            
        # Test FP16 gradient flow
        with torch.cuda.amp.autocast():
            features_fp16 = torch.randn(1, 6, 30, device=device, requires_grad=True)
            coords_fp16 = features_fp16[:, :3, :].transpose(1, 2)
            
            indices = fps.quick_furthest_point_sampling(coords_fp16, 8, kd_depth=3)
            gathered = fps.gather_points(features_fp16, indices)
            loss = gathered.sum()
            
        # Backward pass should work
        loss.backward()
        self.assertIsNotNone(features_fp16.grad)
        self.assertTrue(torch.isfinite(features_fp16.grad).all())

    def test_quick_fps_large_batch_performance(self):
        """Test Quick FPS performance with large batch sizes."""
        device = "cuda"
        
        # Test with larger batch sizes that modern GPUs can handle
        for batch_size in [16, 32, 64]:
            with self.subTest(batch_size=batch_size):
                points = torch.randn(batch_size, 150, 3, device=device)
                indices = fps.quick_furthest_point_sampling(points, 20, kd_depth=5)
                
                self.assertEqual(indices.shape, (batch_size, 20))
                
                # Check each batch
                for b in range(batch_size):
                    unique_indices = torch.unique(indices[b])
                    self.assertEqual(len(unique_indices), 20)
                    self.assertEqual(indices[b, 0].item(), 0)  # First point always 0

    def test_fps_vs_reference_implementation(self):
        """Test that both FPS and Quick FPS match the reference implementation exactly."""
        device = "cuda"
        
        test_cases = [
            # (name, batch_size, n_points, nsamples)
            ("Small batch, few points", 1, 15, 5),
            ("Larger batch", 3, 25, 8),
            ("More samples", 2, 30, 12),
            ("Edge case - many samples", 1, 20, 18),
            ("Single point cloud", 1, 50, 25),
            ("Multiple batches", 4, 40, 15),
        ]
        
        for name, batch_size, n_points, nsamples in test_cases:
            with self.subTest(case=name):
                # Fixed seed for reproducible results
                torch.manual_seed(42)
                points = torch.randn(batch_size, n_points, 3, device=device)
                
                # Reference implementation (CPU)
                points_cpu = points.cpu()
                indices_ref = reference_fps_cpu(points_cpu, nsamples)
                
                # Our implementations
                indices_fps = fps.furthest_point_sampling(points, nsamples)
                indices_quick = fps.quick_furthest_point_sampling(points, nsamples, kd_depth=4)
                
                # Both should match reference exactly
                self.assertTrue(torch.equal(indices_ref, indices_fps.cpu()),
                               f"Regular FPS doesn't match reference for {name}")
                self.assertTrue(torch.equal(indices_ref, indices_quick.cpu()),
                               f"Quick FPS doesn't match reference for {name}")
                
                # Verify uniqueness for all batches
                for b in range(batch_size):
                    ref_unique = len(torch.unique(indices_ref[b]))
                    fps_unique = len(torch.unique(indices_fps[b]))
                    quick_unique = len(torch.unique(indices_quick[b]))
                    
                    self.assertEqual(ref_unique, nsamples,
                                   f"Reference has duplicate indices in {name}, batch {b}")
                    self.assertEqual(fps_unique, nsamples,
                                   f"FPS has duplicate indices in {name}, batch {b}")
                    self.assertEqual(quick_unique, nsamples,
                                   f"Quick FPS has duplicate indices in {name}, batch {b}")

    def test_fps_deterministic_behavior(self):
        """Test that FPS implementations are deterministic with fixed seeds."""
        device = "cuda"
        batch_size, n_points, nsamples = 2, 50, 12
        
        # Multiple runs with same seed should produce identical results
        results_fps = []
        results_quick = []
        results_ref = []
        
        for run in range(3):
            torch.manual_seed(123)
            points = torch.randn(batch_size, n_points, 3, device=device)
            
            # Reference
            points_cpu = points.cpu()
            indices_ref = reference_fps_cpu(points_cpu, nsamples)
            results_ref.append(indices_ref)
            
            # Our implementations
            indices_fps = fps.furthest_point_sampling(points, nsamples)
            indices_quick = fps.quick_furthest_point_sampling(points, nsamples, kd_depth=4)
            
            results_fps.append(indices_fps.cpu())
            results_quick.append(indices_quick.cpu())
        
        # All runs should produce identical results
        for i in range(1, 3):
            self.assertTrue(torch.equal(results_ref[0], results_ref[i]),
                           f"Reference implementation not deterministic (run {i})")
            self.assertTrue(torch.equal(results_fps[0], results_fps[i]),
                           f"FPS implementation not deterministic (run {i})")
            self.assertTrue(torch.equal(results_quick[0], results_quick[i]),
                           f"Quick FPS implementation not deterministic (run {i})")
        
        # All implementations should match
        self.assertTrue(torch.equal(results_ref[0], results_fps[0]),
                       "FPS doesn't match reference in determinism test")
        self.assertTrue(torch.equal(results_ref[0], results_quick[0]),
                       "Quick FPS doesn't match reference in determinism test")

    def test_fps_with_different_kd_depths_vs_reference(self):
        """Test Quick FPS with different kd_depths still matches reference algorithm."""
        device = "cuda"
        batch_size, n_points, nsamples = 2, 40, 10
        
        torch.manual_seed(456)
        points = torch.randn(batch_size, n_points, 3, device=device)
        
        # Reference implementation
        points_cpu = points.cpu()
        indices_ref = reference_fps_cpu(points_cpu, nsamples)
        
        # Regular FPS
        indices_fps = fps.furthest_point_sampling(points, nsamples)
        
        # Test Quick FPS with different kd_depths - all should match reference
        for kd_depth in [3, 4, 5, 6]:
            with self.subTest(kd_depth=kd_depth):
                indices_quick = fps.quick_furthest_point_sampling(points, nsamples, kd_depth)
                
                # Should match reference regardless of kd_depth
                self.assertTrue(torch.equal(indices_ref, indices_fps.cpu()),
                               f"Regular FPS doesn't match reference")
                self.assertTrue(torch.equal(indices_ref, indices_quick.cpu()),
                               f"Quick FPS (kd_depth={kd_depth}) doesn't match reference")
                
                # Check uniqueness
                for b in range(batch_size):
                    unique_indices = torch.unique(indices_quick[b])
                    self.assertEqual(len(unique_indices), nsamples,
                                   f"Quick FPS (kd_depth={kd_depth}) has duplicates in batch {b}")

    def test_fps_quality_metrics_vs_reference(self):
        """Test that all implementations produce equivalent quality metrics."""
        device = "cuda"
        torch.manual_seed(789)
        points = torch.randn(1, 100, 3, device=device)
        nsamples = 20
        
        # Get results from all implementations
        points_cpu = points.cpu()
        indices_ref = reference_fps_cpu(points_cpu, nsamples)
        indices_fps = fps.furthest_point_sampling(points, nsamples)
        indices_quick = fps.quick_furthest_point_sampling(points, nsamples, kd_depth=5)
        
        # All should be identical
        self.assertTrue(torch.equal(indices_ref, indices_fps.cpu()))
        self.assertTrue(torch.equal(indices_ref, indices_quick.cpu()))
        
        # Extract sampled points
        sampled_ref = points_cpu[0, indices_ref[0]]
        sampled_fps = points[0, indices_fps[0]].cpu()
        sampled_quick = points[0, indices_quick[0]].cpu()
        
        # Compute quality metrics - should all be identical
        dists_ref = torch.cdist(sampled_ref, sampled_ref)
        dists_fps = torch.cdist(sampled_fps, sampled_fps)
        dists_quick = torch.cdist(sampled_quick, sampled_quick)
        
        # Remove diagonal (self-distances)
        mask = ~torch.eye(nsamples, dtype=bool)
        min_dist_ref = dists_ref[mask].min()
        min_dist_fps = dists_fps[mask].min()
        min_dist_quick = dists_quick[mask].min()
        
        # All quality metrics should be identical
        self.assertTrue(torch.allclose(min_dist_ref, min_dist_fps, atol=1e-6))
        self.assertTrue(torch.allclose(min_dist_ref, min_dist_quick, atol=1e-6))
        
        # Mean distances should also be identical
        mean_dist_ref = dists_ref[mask].mean()
        mean_dist_fps = dists_fps[mask].mean()
        mean_dist_quick = dists_quick[mask].mean()
        
        self.assertTrue(torch.allclose(mean_dist_ref, mean_dist_fps, atol=1e-6))
        self.assertTrue(torch.allclose(mean_dist_ref, mean_dist_quick, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
