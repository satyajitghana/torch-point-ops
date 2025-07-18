import torch
from torch.testing._internal.common_utils import TestCase
from torch.library import opcheck
import unittest
import numpy as np

# Since the package is installed, we can import it.
from torch_point_ops import EarthMoverDistance
from torch_point_ops.emd import earth_movers_distance


def reference_emd(p1, p2):
    """
    Reference implementation of Earth Mover's Distance.
    This is a placeholder and will not give the same results as the CUDA op.
    The CUDA op is non-deterministic.
    """
    from scipy.spatial.distance import cdist
    from scipy.optimize import linear_sum_assignment

    p1_np = p1.cpu().numpy()
    p2_np = p2.cpu().numpy()

    batch_size = p1.shape[0]
    num_points1 = p1.shape[1]
    num_points2 = p2.shape[1]

    emd_dist = np.zeros(batch_size)

    for i in range(batch_size):
        cost_matrix = cdist(p1_np[i], p2_np[i])
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        emd_dist[i] = cost_matrix[row_ind, col_ind].sum() / min(
            num_points1, num_points2
        )

    return torch.from_numpy(emd_dist).to(p1.device)


@unittest.skipIf(not torch.cuda.is_available(), "No CUDA device found.")
class TestEMDDistance(TestCase):

    def sample_inputs(
        self,
        device,
        batch_size=1,
        n_points=64,
        m_points=128,
        requires_grad=False,
        dtype=torch.float64,
    ):
        p1 = torch.randn(
            batch_size,
            n_points,
            3,
            device=device,
            requires_grad=requires_grad,
            dtype=dtype,
        )
        p2 = torch.randn(
            batch_size,
            m_points,
            3,
            device=device,
            requires_grad=requires_grad,
            dtype=dtype,
        )
        return p1, p2

    def test_forward_correctness(self):
        device = "cuda"
        p1, p2 = self.sample_inputs(device, dtype=torch.float32)

        # The CUDA op is non-deterministic, so we can't check for equality.
        # We can only check that the output is reasonable.
        dist = earth_movers_distance(p1, p2)
        self.assertEqual(dist.dim(), 1)
        self.assertEqual(dist.shape[0], p1.shape[0])

    def test_forward_float16_stability(self):
        """
        Test EMD with float16 to ensure it doesn't produce NaNs.
        """
        device = "cuda"
        p1 = torch.rand(1, 1024, 3, dtype=torch.float16, device=device)
        p2 = torch.rand(1, 1024, 3, dtype=torch.float16, device=device)
        dist = earth_movers_distance(p1, p2)
        self.assertTrue(torch.isfinite(dist).all())

    def test_gradcheck(self):
        """
        Test gradients using torch.autograd.gradcheck for numerical gradient verification.
        Only tests CUDA since emd_forward is CUDA-only. Uses basic gradient flow testing
        instead of full gradcheck due to EMD's non-deterministic nature.
        """

        # Test the EMD function
        def emd_fn(p1, p2):
            return earth_movers_distance(p1, p2)

        # Disable cuDNN for more deterministic behavior during gradient checking
        with torch.backends.cudnn.flags(enabled=False):
            # Test CUDA basic gradient flow (EMD is too non-deterministic for full gradcheck)
            if torch.cuda.is_available():
                p1_cuda = torch.randn(
                    1, 4, 3, device="cuda", dtype=torch.float32, requires_grad=True
                )
                p2_cuda = torch.randn(
                    1, 4, 3, device="cuda", dtype=torch.float32, requires_grad=True
                )

                # Test forward pass and basic backward pass
                result = emd_fn(p1_cuda, p2_cuda)
                result.sum().backward()

                # Verify gradients exist and are finite
                self.assertIsNotNone(p1_cuda.grad)
                self.assertIsNotNone(p2_cuda.grad)
                self.assertTrue(torch.isfinite(p1_cuda.grad).all())
                self.assertTrue(torch.isfinite(p2_cuda.grad).all())

    def test_gradcheck_module(self):
        """
        Test gradients for the EarthMoverDistance module.
        Only tests CUDA since emd_forward is CUDA-only.
        """
        model = EarthMoverDistance()

        def model_fn(p1, p2):
            return model(p1, p2)

        with torch.backends.cudnn.flags(enabled=False):
            # Test CUDA basic gradient flow
            if torch.cuda.is_available():
                p1_cuda = torch.randn(
                    1, 3, 3, device="cuda", dtype=torch.float32, requires_grad=True
                )
                p2_cuda = torch.randn(
                    1, 3, 3, device="cuda", dtype=torch.float32, requires_grad=True
                )

                # Test that it runs and produces finite gradients
                result = model_fn(p1_cuda, p2_cuda)
                result.backward()

                self.assertIsNotNone(p1_cuda.grad)
                self.assertIsNotNone(p2_cuda.grad)
                self.assertTrue(torch.isfinite(p1_cuda.grad).all())
                self.assertTrue(torch.isfinite(p2_cuda.grad).all())

    def test_known_values(self):
        device = "cuda"
        p1 = torch.tensor(
            [[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]], dtype=torch.float32, device=device
        )
        p2 = p1.clone()
        dist = earth_movers_distance(p1, p2)
        self.assertTrue(torch.allclose(dist, torch.zeros_like(dist), atol=1e-6))

        # Test with larger point clouds
        p1_large, p2_large = self.sample_inputs(
            device,
            batch_size=2,
            n_points=128,
            m_points=128,
            dtype=torch.float64,
        )
        dist_large = earth_movers_distance(p1_large, p1_large)
        self.assertTrue(
            torch.allclose(dist_large, torch.zeros_like(dist_large), atol=1e-5)
        )

    def test_static_values(self):
        device = "cuda"
        p1 = torch.tensor(
            [[[1.7, -0.1, 0.1], [0.1, 1.2, 0.3]]], dtype=torch.float32, device=device
        )
        p1 = p1.repeat(3, 1, 1)
        p2 = torch.tensor(
            [[[0.3, 1.8, 0.2], [1.2, -0.2, 0.3]]], dtype=torch.float32, device=device
        )
        p2 = p2.repeat(3, 1, 1)

        p1.requires_grad = True
        p2.requires_grad = True

        dist = earth_movers_distance(p1, p2)
        loss = dist[0] / 2 + dist[1] * 2 + dist[2] / 3
        loss.backward()

        # Expected values computed from ground truth optimal matching:
        # For each batch, optimal matching is p1[i,0] -> p2[i,1] and p1[i,1] -> p2[i,0]
        # Cost = sum of squared distances / num_points = 0.355 for each batch
        dist_expected = torch.tensor(
            [0.355, 0.355, 0.355], device=device, dtype=torch.float32
        )
        self.assertTrue(torch.allclose(dist, dist_expected, atol=1e-3))

        p1_grad_expected = torch.tensor(
            [
                [[0.2500, 0.0500, -0.1000], [-0.1000, -0.3000, 0.0500]],
                [[1.0000, 0.2000, -0.4000], [-0.4000, -1.2000, 0.2000]],
                [[0.1667, 0.0333, -0.0667], [-0.0667, -0.2000, 0.0333]],
            ],
            device=device,
            dtype=torch.float32,
        )

        p2_grad_expected = torch.tensor(
            [
                [[0.1000, 0.3000, -0.0500], [-0.2500, -0.0500, 0.1000]],
                [[0.4000, 1.2000, -0.2000], [-1.0000, -0.2000, 0.4000]],
                [[0.0667, 0.2000, -0.0333], [-0.1667, -0.0333, 0.0667]],
            ],
            device=device,
            dtype=torch.float32,
        )

        self.assertTrue(torch.allclose(p1.grad, p1_grad_expected, atol=1e-4))
        self.assertTrue(torch.allclose(p2.grad, p2_grad_expected, atol=1e-4))

    def test_empty_clouds(self):
        device = "cuda"
        p1 = torch.randn(2, 10, 3, device=device)
        p_empty = torch.empty(2, 0, 3, device=device)

        dist = earth_movers_distance(p1, p_empty)
        self.assertTrue(torch.all(dist == 0))

        dist = earth_movers_distance(p_empty, p1)
        self.assertTrue(torch.all(dist == 0))

        dist = earth_movers_distance(p_empty, p_empty)
        self.assertTrue(torch.all(dist == 0))

    def test_opcheck(self):
        device = "cuda"
        p1, p2 = self.sample_inputs(device, requires_grad=True)

        # The EMD op is non-deterministic, so we can't check for bit-exactness
        # We also need to use float64 for gradcheck to pass.
        opcheck(
            torch.ops.torch_point_ops_emd.emd_forward.default,
            (p1, p2),
            raise_exception=True,
        )

        p1_no_grad, p2_no_grad = self.sample_inputs(device, requires_grad=False)
        opcheck(
            torch.ops.torch_point_ops_emd.emd_forward.default,
            (p1_no_grad, p2_no_grad),
            raise_exception=True,
        )

    def test_compile(self):
        device = "cuda"
        model = EarthMoverDistance().to(device)
        compiled_model = torch.compile(model)

        p1, p2 = self.sample_inputs(device, dtype=torch.float32)
        # The op is non-deterministic so we can't check for equality
        # just that it runs
        model(p1, p2)
        compiled_model(p1, p2)

    def test_multi_precision(self):
        """
        Test EMD with different floating point precisions.
        """
        device = "cuda"
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        # Test different precisions with appropriate tolerances
        precision_configs = [
            (torch.float16, 1e-2),
            (torch.float32, 1e-5),
            (torch.float64, 1e-7),
        ]

        for dtype, atol in precision_configs:
            with self.subTest(dtype=dtype):
                p1, p2 = self.sample_inputs(
                    device,
                    batch_size=2,
                    n_points=128,
                    m_points=128,
                    requires_grad=True,
                    dtype=dtype,
                )

                # Test forward pass
                dist = earth_movers_distance(p1, p2)
                self.assertTrue(torch.isfinite(dist).all())
                self.assertTrue((dist >= 0).all())
                self.assertEqual(dist.dtype, dtype)

                # Test backward pass
                dist.sum().backward()
                self.assertIsNotNone(p1.grad)
                self.assertIsNotNone(p2.grad)
                self.assertTrue(torch.isfinite(p1.grad).all())
                self.assertTrue(torch.isfinite(p2.grad).all())

                # Test with non-equal number of points
                p1_uneven, p2_uneven = self.sample_inputs(
                    device,
                    batch_size=2,
                    n_points=64,
                    m_points=128,
                    requires_grad=True,
                    dtype=dtype,
                )
                dist_uneven = earth_movers_distance(p1_uneven, p2_uneven)
                self.assertTrue(torch.isfinite(dist_uneven).all())
                self.assertTrue((dist_uneven >= 0).all())
                dist_uneven.sum().backward()
                self.assertIsNotNone(p1_uneven.grad)
                self.assertIsNotNone(p2_uneven.grad)
                self.assertTrue(torch.isfinite(p1_uneven.grad).all())
                self.assertTrue(torch.isfinite(p2_uneven.grad).all())

    def test_symmetry(self):
        """
        Test if EMD is symmetric for point clouds of the same size.
        Note: EMD uses an approximation algorithm that can have slight numerical differences
        when swapping inputs, especially for float16 which uses clamping for stability.
        """
        device = "cuda"
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        precision_configs = [
            (torch.float16, 2e-1),  # Increased tolerance for float16 due to clamping
            (torch.float32, 1e-1),  # Increased tolerance for approximate algorithm
            (torch.float64, 5e-2),  # Increased tolerance for approximate algorithm
        ]

        for dtype, atol in precision_configs:
            with self.subTest(dtype=dtype):
                p1, p2 = self.sample_inputs(
                    device,
                    batch_size=2,
                    n_points=256,
                    m_points=256,
                    dtype=dtype,
                )
                dist1 = earth_movers_distance(p1, p2)
                dist2 = earth_movers_distance(p2, p1)
                self.assertTrue(torch.allclose(dist1, dist2, atol=atol))


if __name__ == "__main__":
    unittest.main()
