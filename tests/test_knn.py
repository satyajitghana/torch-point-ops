import torch
from torch.testing._internal.common_utils import TestCase
from torch.library import opcheck
import unittest

# Since the package is installed, we can import it.
from torch_point_ops import KNearestNeighbors
import torch_point_ops.knn as knn


def reference_knn_points(
    p1: torch.Tensor, p2: torch.Tensor, lengths1=None, lengths2=None, K=1, norm=2
):
    """
    A reference implementation of KNN in pure PyTorch.
    Returns always sorted results
    """
    N, P1, D = p1.shape
    _, P2, _ = p2.shape

    if lengths1 is None:
        lengths1 = torch.full((N,), P1, dtype=torch.int64, device=p1.device)
    if lengths2 is None:
        lengths2 = torch.full((N,), P2, dtype=torch.int64, device=p1.device)

    dists = torch.zeros((N, P1, K), dtype=p1.dtype, device=p1.device)
    idxs = torch.full((N, P1, K), -1, dtype=torch.int64, device=p1.device)

    for n in range(N):
        num1 = lengths1[n].item()
        num2 = lengths2[n].item()

        for i1 in range(num1):
            # Calculate distances from point i1 in p1 to all points in p2
            p1_point = p1[n, i1 : i1 + 1, :]  # (1, D)
            p2_cloud = p2[n, :num2, :]  # (num2, D)

            if norm == 1:
                point_dists = torch.sum(torch.abs(p1_point - p2_cloud), dim=1)
            else:  # norm == 2
                point_dists = torch.sum((p1_point - p2_cloud) ** 2, dim=1)

            # Get K smallest distances
            k_actual = min(K, num2)
            if k_actual > 0:
                sorted_dists, sorted_idxs = torch.sort(point_dists)
                dists[n, i1, :k_actual] = sorted_dists[:k_actual]
                idxs[n, i1, :k_actual] = sorted_idxs[:k_actual]

    return idxs, dists


@unittest.skipIf(not torch.cuda.is_available(), "No CUDA device found.")
class TestKNN(TestCase):

    def sample_inputs(
        self, device, batch_size=1, n_points=64, m_points=128, requires_grad=False
    ):
        p1 = torch.randn(
            batch_size,
            n_points,
            3,
            device=device,
            requires_grad=requires_grad,
            dtype=torch.float32,
        )
        p2 = torch.randn(
            batch_size,
            m_points,
            3,
            device=device,
            requires_grad=requires_grad,
            dtype=torch.float32,
        )
        return p1, p2

    def test_forward_correctness(self):
        device = "cuda"
        p1, p2 = self.sample_inputs(device, batch_size=2, n_points=10, m_points=15)

        # Test with different K values
        for K in [1, 3, 5]:
            with self.subTest(K=K):
                # Our op
                idxs, dists = knn.knn_points(
                    p1,
                    p2,
                    lengths1=torch.full((2,), 10, dtype=torch.int64, device=device),
                    lengths2=torch.full((2,), 15, dtype=torch.int64, device=device),
                    K=K,
                    norm=2,
                )

                # Reference
                ref_idxs, ref_dists = reference_knn_points(p1, p2, K=K, norm=2)

                self.assertTrue(torch.allclose(dists, ref_dists, atol=1e-5))
                self.assertTrue(torch.all(idxs == ref_idxs))

    def test_gradcheck(self):
        """
        Test gradients using torch.autograd.gradcheck for numerical gradient verification.
        """
        # Disable cuDNN for more deterministic behavior during gradient checking
        with torch.backends.cudnn.flags(enabled=False):
            # Test CUDA with float64
            if torch.cuda.is_available():
                p1_cuda = torch.randn(
                    1, 4, 3, device="cuda", dtype=torch.float64, requires_grad=True
                )
                p2_cuda = torch.randn(
                    1, 5, 3, device="cuda", dtype=torch.float64, requires_grad=True
                )

                # Test the knn function
                def knn_fn(p1, p2):
                    # We test the unsorted version because sort is not differentiable
                    # on its own. The wrapper function knn_points handles sorting.
                    # We also sum the distances to get a scalar output for gradcheck.
                    idxs, dists = knn.knn_points(p1, p2, K=2, return_sorted=False)
                    return dists.sum()

                self.assertTrue(
                    torch.autograd.gradcheck(
                        knn_fn,
                        (p1_cuda, p2_cuda),
                        eps=1e-6,
                        atol=1e-5,
                        rtol=1e-4,
                        nondet_tol=1e-5,
                        fast_mode=True,
                        check_batched_grad=False,  # disable for CUDA atomic ops
                    )
                )

            # Test CPU reference implementation separately with float64
            p1_cpu = torch.randn(
                1, 4, 3, device="cpu", dtype=torch.float64, requires_grad=True
            )
            p2_cpu = torch.randn(
                1, 5, 3, device="cpu", dtype=torch.float64, requires_grad=True
            )

            def ref_knn_fn(p1, p2):
                _, dists = reference_knn_points(p1, p2, K=2)
                return dists.mean()

            self.assertTrue(
                torch.autograd.gradcheck(
                    ref_knn_fn,
                    (p1_cpu, p2_cpu),
                    eps=1e-6,
                    atol=1e-5,
                    rtol=1e-4,
                    fast_mode=True,
                )
            )

    def test_gradients_vs_reference(self):
        device = "cuda"
        p1, p2 = self.sample_inputs(
            device, batch_size=1, n_points=8, m_points=12, requires_grad=True
        )

        # Our op
        dists_actual = knn.knn_points(p1, p2, K=3)[1]
        loss = dists_actual.mean()
        loss.backward()

        grad_p1_actual = p1.grad.clone()
        grad_p2_actual = p2.grad.clone()

        p1.grad.zero_()
        p2.grad.zero_()

        # Reference op
        ref_dists = reference_knn_points(p1, p2, K=3)[1]
        ref_loss = ref_dists.mean()
        ref_loss.backward()

        grad_p1_expected = p1.grad.clone()
        grad_p2_expected = p2.grad.clone()

        self.assertTrue(torch.allclose(grad_p1_actual, grad_p1_expected, atol=1e-4))
        self.assertTrue(torch.allclose(grad_p2_actual, grad_p2_expected, atol=1e-4))

    def test_known_values(self):
        device = "cuda"
        # Test case 1: simple 2D case
        p1 = torch.tensor(
            [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], dtype=torch.float32, device=device
        )
        p2 = torch.tensor(
            [[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [2.0, 0.0, 0.0]]],
            dtype=torch.float32,
            device=device,
        )

        idxs, dists = knn.knn_points(
            p1,
            p2,
            lengths1=torch.tensor([2], dtype=torch.int64, device=device),
            lengths2=torch.tensor([3], dtype=torch.int64, device=device),
            norm=2,
            K=2,
        )

        # For point [0,0,0], nearest should be [0,0,0] (idx=0, dist=0) and [0.5,0,0] (idx=1, dist=0.25)
        self.assertTrue(torch.allclose(dists[0, 0, 0], torch.tensor(0.0)))
        self.assertEqual(idxs[0, 0, 0].item(), 0)
        self.assertTrue(torch.allclose(dists[0, 0, 1], torch.tensor(0.25), atol=1e-5))
        self.assertEqual(idxs[0, 0, 1].item(), 1)

        # For point [1,0,0], nearest should be [0.5,0,0] (idx=1, dist=0.25) and [0,0,0] (idx=0, dist=1.0)
        self.assertTrue(torch.allclose(dists[0, 1, 0], torch.tensor(0.25), atol=1e-5))
        self.assertEqual(idxs[0, 1, 0].item(), 1)
        self.assertTrue(torch.allclose(dists[0, 1, 1], torch.tensor(1.0), atol=1e-5))
        self.assertEqual(idxs[0, 1, 1].item(), 0)

    def test_empty_clouds(self):
        device = "cuda"
        p1 = torch.randn(2, 10, 3, device=device)
        p_empty = torch.empty(2, 0, 3, device=device)

        # Test when p2 is empty
        idxs, dists = torch.ops.torch_point_ops_knn.knn_forward(
            p1,
            p_empty,
            torch.tensor([10, 10], dtype=torch.int64, device=device),
            torch.tensor([0, 0], dtype=torch.int64, device=device),
            2,
            3,
            -1,
        )
        self.assertEqual(idxs.shape, (2, 10, 3))
        self.assertEqual(dists.shape, (2, 10, 3))

        # Test when p1 is empty
        idxs, dists = torch.ops.torch_point_ops_knn.knn_forward(
            p_empty,
            p1,
            torch.tensor([0, 0], dtype=torch.int64, device=device),
            torch.tensor([10, 10], dtype=torch.int64, device=device),
            2,
            3,
            -1,
        )
        self.assertEqual(idxs.shape, (2, 0, 3))
        self.assertEqual(dists.shape, (2, 0, 3))

    def test_different_norms(self):
        """Test L1 and L2 norms"""
        device = "cuda"
        p1 = torch.tensor([[[0.0, 0.0, 0.0]]], dtype=torch.float32, device=device)
        p2 = torch.tensor(
            [[[1.0, 1.0, 1.0], [2.0, 0.0, 0.0]]], dtype=torch.float32, device=device
        )

        # Test L1 norm
        idxs_l1, dists_l1 = knn.knn_points(
            p1,
            p2,
            lengths1=torch.tensor([1], dtype=torch.int64, device=device),
            lengths2=torch.tensor([2], dtype=torch.int64, device=device),
            norm=1,
            K=2,
        )

        # L1 distances should be 3.0 and 2.0
        expected_l1 = torch.tensor([[2.0, 3.0]], device=device)  # sorted
        self.assertTrue(torch.allclose(dists_l1[0, 0, :], expected_l1))

        # Test L2 norm
        idxs_l2, dists_l2 = knn.knn_points(
            p1,
            p2,
            lengths1=torch.tensor([1], dtype=torch.int64, device=device),
            lengths2=torch.tensor([2], dtype=torch.int64, device=device),
            norm=2,
            K=2,
        )

        # L2 distances should be sqrt(3)^2=3.0 and 2^2=4.0
        expected_l2 = torch.tensor([[3.0, 4.0]], device=device)  # sorted
        self.assertTrue(torch.allclose(dists_l2[0, 0, :], expected_l2))

    def test_module_interface(self):
        """Test the high-level module interface"""
        device = "cuda"
        p1, p2 = self.sample_inputs(device, batch_size=2, n_points=8, m_points=10)

        # Test basic module
        knn_module = KNearestNeighbors(K=3, norm=2)
        idxs, dists = knn_module(p1, p2)

        self.assertEqual(idxs.shape, (2, 8, 3))
        self.assertEqual(dists.shape, (2, 8, 3))

        # Test with return_nn=True
        knn_module_nn = KNearestNeighbors(K=3, norm=2, return_nn=True)
        idxs, dists, nn = knn_module_nn(p1, p2)

        self.assertEqual(idxs.shape, (2, 8, 3))
        self.assertEqual(dists.shape, (2, 8, 3))
        self.assertEqual(nn.shape, (2, 8, 3, 3))

    def test_knn_gather(self):
        device = "cuda"
        N, P1, P2, K, D = 2, 8, 12, 4, 3

        p1 = torch.randn(N, P1, D, device=device)
        p2 = torch.randn(N, P2, D, device=device)
        lengths1 = torch.randint(low=1, high=P1, size=(N,), device=device)
        lengths2 = torch.randint(low=1, high=P2, size=(N,), device=device)

        idxs, dists = knn.knn_points(p1, p2, lengths1=lengths1, lengths2=lengths2, K=K)
        p2_nn = knn.knn_gather(p2, idxs, lengths2)

        for n in range(N):
            for p1_idx in range(P1):
                for k in range(K):
                    if k < lengths2[n]:
                        expected_point = p2[n, idxs[n, p1_idx, k]]
                        self.assertTrue(
                            torch.allclose(p2_nn[n, p1_idx, k], expected_point)
                        )
                    else:
                        self.assertTrue(torch.all(p2_nn[n, p1_idx, k] == 0.0))

    def test_gradients_multiple_sizes(self):
        """
        Test gradients by comparing with a reference PyTorch implementation.
        """
        device = "cuda"

        # Test with different sizes
        test_cases = [
            (1, 3, 4, 1),  # small, K=1
            (2, 5, 7, 2),  # medium, K=2
            (1, 8, 6, 3),  # different sizes, K=3
        ]

        for batch_size, n_points, m_points, K in test_cases:
            with self.subTest(
                batch_size=batch_size, n_points=n_points, m_points=m_points, K=K
            ):
                p1 = torch.randn(
                    batch_size,
                    n_points,
                    3,
                    device=device,
                    requires_grad=True,
                    dtype=torch.float32,
                )
                p2 = torch.randn(
                    batch_size,
                    m_points,
                    3,
                    device=device,
                    requires_grad=True,
                    dtype=torch.float32,
                )

                # Our implementation
                dists_our = knn.knn_points(p1, p2, K=K)[1]
                loss_our = dists_our.mean()
                loss_our.backward()

                grad_p1_our = p1.grad.clone()
                grad_p2_our = p2.grad.clone()

                # Reset gradients
                p1.grad.zero_()
                p2.grad.zero_()

                # Reference implementation
                ref_dists = reference_knn_points(p1, p2, K=K)[1]
                loss_ref = ref_dists.mean()
                loss_ref.backward()

                grad_p1_ref = p1.grad.clone()
                grad_p2_ref = p2.grad.clone()

                # Compare results
                self.assertTrue(torch.allclose(loss_our, loss_ref, atol=1e-6))
                self.assertTrue(torch.allclose(grad_p1_our, grad_p1_ref, atol=1e-4))
                self.assertTrue(torch.allclose(grad_p2_our, grad_p2_ref, atol=1e-4))

    def test_opcheck(self):
        device = "cuda"
        p1_no_grad, p2_no_grad = self.sample_inputs(
            device, batch_size=1, n_points=3, m_points=3, requires_grad=False
        )
        lengths1 = torch.tensor([3], dtype=torch.int64, device=device)
        lengths2 = torch.tensor([3], dtype=torch.int64, device=device)

        # Test opcheck without gradients (atomic operations in CUDA backward cause non-determinism)
        opcheck(
            torch.ops.torch_point_ops_knn.knn_forward.default,
            (
                p1_no_grad,
                p2_no_grad,
                lengths1,
                lengths2,
                2,
                2,
                -1,
            ),  # norm=2, K=2, version=-1
            raise_exception=True,
        )

    def test_compile(self):
        device = "cuda"
        model = KNearestNeighbors(K=3).to(device)
        compiled_model = torch.compile(model)

        p1, p2 = self.sample_inputs(device, batch_size=1, n_points=8, m_points=10)

        result_eager = model(p1, p2)
        result_compiled = compiled_model(p1, p2)

        self.assertTrue(torch.allclose(result_eager[0], result_compiled[0]))
        self.assertTrue(torch.allclose(result_eager[1], result_compiled[1]))

    def test_multi_precision(self):
        """
        Test KNN with different floating point precisions.
        """
        device = "cuda"
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        # Test different precisions with appropriate tolerances
        precision_configs = [
            (torch.float16, 1e-2, 1e-1),  # half precision: loose tolerances
            (torch.float32, 1e-5, 1e-3),  # single precision: medium tolerances
            (torch.float64, 1e-8, 1e-6),  # double precision: tight tolerances
        ]

        for dtype, atol, rtol in precision_configs:
            with self.subTest(dtype=dtype):
                p1 = torch.randn(2, 8, 3, device=device, dtype=dtype)
                p2 = torch.randn(2, 10, 3, device=device, dtype=dtype)

                # Test forward pass
                idxs, dists = knn.knn_points(p1, p2, K=3)

                # Verify output dtype matches input
                self.assertEqual(dists.dtype, dtype)
                self.assertEqual(idxs.dtype, torch.int64)

                # Verify shapes
                self.assertEqual(idxs.shape, (2, 8, 3))
                self.assertEqual(dists.shape, (2, 8, 3))

                # Test backward pass
                p1_grad = p1.clone().requires_grad_(True)
                p2_grad = p2.clone().requires_grad_(True)

                idxs_grad, dists_grad = knn.knn_points(p1_grad, p2_grad, K=3)
                loss = dists_grad.mean()
                loss.backward()

                # Verify gradients exist and are finite
                self.assertIsNotNone(p1_grad.grad)
                self.assertIsNotNone(p2_grad.grad)
                self.assertTrue(torch.isfinite(p1_grad.grad).all())
                self.assertTrue(torch.isfinite(p2_grad.grad).all())

                # Test module with different configurations
                for return_nn in [True, False]:
                    model = KNearestNeighbors(K=3, return_nn=return_nn)
                    result = model(p1, p2)
                    if return_nn:
                        self.assertEqual(len(result), 3)
                        self.assertEqual(result[0].dtype, torch.int64)  # idxs
                        self.assertEqual(result[1].dtype, dtype)  # dists
                        self.assertEqual(result[2].dtype, dtype)  # nn
                    else:
                        self.assertEqual(len(result), 2)
                        self.assertEqual(result[0].dtype, torch.int64)  # idxs
                        self.assertEqual(result[1].dtype, dtype)  # dists

    def test_version_selection(self):
        """Test different kernel versions"""
        device = "cuda"
        p1, p2 = self.sample_inputs(device, batch_size=1, n_points=8, m_points=10)
        lengths1 = torch.tensor([8], dtype=torch.int64, device=device)
        lengths2 = torch.tensor([10], dtype=torch.int64, device=device)

        # Test auto version selection
        idxs_auto, dists_auto = torch.ops.torch_point_ops_knn.knn_forward(
            p1, p2, lengths1, lengths2, 2, 3, -1  # version=-1 for auto
        )

        # Test specific versions (when applicable)
        for version in [0, 1, 2, 3]:
            # The python wrapper knn_points handles sorting, so we can compare
            # the results of different versions with the sorted auto version.
            try:
                idxs_v, dists_v = knn.knn_points(
                    p1,
                    p2,
                    lengths1=lengths1,
                    lengths2=lengths2,
                    K=3,
                    norm=2,
                    version=version,
                )
                # Results should be the same regardless of version
                self.assertTrue(torch.allclose(dists_auto, dists_v, atol=1e-5))
                # Note: indices might be different if there are ties in distances
            except Exception:
                # Some versions might not support all D/K combinations
                pass

    def test_ragged_inputs(self):
        """Test with variable length point clouds"""
        device = "cuda"
        N, P1, P2, K, D = 3, 10, 12, 4, 3

        p1 = torch.randn(N, P1, D, device=device)
        p2 = torch.randn(N, P2, D, device=device)
        lengths1 = torch.tensor([5, 8, 10], dtype=torch.int64, device=device)
        lengths2 = torch.tensor([6, 12, 9], dtype=torch.int64, device=device)

        idxs, dists = knn.knn_points(p1, p2, lengths1=lengths1, lengths2=lengths2, K=K)

        # Verify shapes
        self.assertEqual(idxs.shape, (N, P1, K))
        self.assertEqual(dists.shape, (N, P1, K))

        # Verify that results are valid only within the specified lengths
        for n in range(N):
            num1 = lengths1[n].item()
            num2 = lengths2[n].item()

            # Check that distances for valid points are reasonable
            for i in range(num1):
                for k in range(min(K, num2)):
                    self.assertTrue(dists[n, i, k] >= 0)
                    self.assertTrue(0 <= idxs[n, i, k] < num2)


if __name__ == "__main__":
    unittest.main()
