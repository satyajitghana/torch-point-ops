import torch
from torch.testing._internal.common_utils import TestCase
from torch.library import opcheck
import unittest

# Since the package is installed, we can import it.
from torch_point_ops import ChamferDistance
import torch_point_ops.chamfer as chamfer

def reference_chamfer_distance(p1: torch.Tensor, p2: torch.Tensor):
    """
    A reference implementation of Chamfer distance in pure PyTorch.
    """
    p1_expanded = p1.unsqueeze(2)
    p2_expanded = p2.unsqueeze(1)
    
    # Calculate squared distances
    dist_matrix = torch.sum((p1_expanded - p2_expanded) ** 2, dim=3)
    
    # Find nearest neighbors
    dist1, _ = torch.min(dist_matrix, dim=2)
    dist2, _ = torch.min(dist_matrix, dim=1)
    
    return dist1, dist2

@unittest.skipIf(not torch.cuda.is_available(), "No CUDA device found.")
class TestChamferDistance(TestCase):

    def sample_inputs(self, device, batch_size=1, n_points=64, m_points=128, requires_grad=False):
        p1 = torch.randn(batch_size, n_points, 3, device=device, requires_grad=requires_grad, dtype=torch.float32)
        p2 = torch.randn(batch_size, m_points, 3, device=device, requires_grad=requires_grad, dtype=torch.float32)
        return p1, p2

    def test_correctness(self):
        device = "cuda"
        p1, p2 = self.sample_inputs(device)
        
        # Our op
        dist1, dist2, _, _ = torch.ops.torch_point_ops_chamfer.chamfer_forward(p1, p2)
        
        # Reference
        ref_dist1, ref_dist2 = reference_chamfer_distance(p1, p2)
        
        self.assertTrue(torch.allclose(dist1, ref_dist1))
        self.assertTrue(torch.allclose(dist2, ref_dist2))

    def test_gradients(self):
        device = "cuda"
        p1, p2 = self.sample_inputs(device, requires_grad=True)
        
        # Our op
        dist1_actual, dist2_actual = chamfer.chamfer_distance(p1, p2)
        loss = dist1_actual.mean() + dist2_actual.mean()
        loss.backward()
        
        grad_p1_actual = p1.grad.clone()
        grad_p2_actual = p2.grad.clone()
        
        p1.grad.zero_()
        p2.grad.zero_()
        
        # Reference op
        ref_dist1, ref_dist2 = reference_chamfer_distance(p1, p2)
        ref_loss = ref_dist1.mean() + ref_dist2.mean()
        ref_loss.backward()
        
        grad_p1_expected = p1.grad.clone()
        grad_p2_expected = p2.grad.clone()

        self.assertTrue(torch.allclose(grad_p1_actual, grad_p1_expected))
        self.assertTrue(torch.allclose(grad_p2_actual, grad_p2_expected))

    def test_known_values(self):
        device = "cuda"
        # Test case 1: identical point clouds
        p1 = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]], dtype=torch.float32, device=device)
        p2 = p1.clone()
        dist1, dist2, _, _ = torch.ops.torch_point_ops_chamfer.chamfer_forward(p1, p2)
        self.assertTrue(torch.allclose(dist1, torch.zeros_like(dist1)))
        self.assertTrue(torch.allclose(dist2, torch.zeros_like(dist2)))

        # Test case 2: one point each
        p1 = torch.tensor([[[0.0, 0.0, 0.0]]], dtype=torch.float32, device=device)
        p2 = torch.tensor([[[1.0, 2.0, 3.0]]], dtype=torch.float32, device=device)
        dist1, dist2, _, _ = torch.ops.torch_point_ops_chamfer.chamfer_forward(p1, p2)
        expected_dist = torch.tensor([1.0**2 + 2.0**2 + 3.0**2], device=device)
        self.assertTrue(torch.allclose(dist1.squeeze(), expected_dist))
        self.assertTrue(torch.allclose(dist2.squeeze(), expected_dist))

    def test_empty_clouds(self):
        device = "cuda"
        p1 = torch.randn(2, 10, 3, device=device)
        p_empty = torch.empty(2, 0, 3, device=device)

        dist1, dist2, _, _ = torch.ops.torch_point_ops_chamfer.chamfer_forward(p1, p_empty)
        self.assertEqual(dist1.shape, (2, 10))
        self.assertEqual(dist2.shape, (2, 0))

        dist1, dist2, _, _ = torch.ops.torch_point_ops_chamfer.chamfer_forward(p_empty, p1)
        self.assertEqual(dist1.shape, (2, 0))
        self.assertEqual(dist2.shape, (2, 10))

        dist1, dist2, _, _ = torch.ops.torch_point_ops_chamfer.chamfer_forward(p_empty, p_empty)
        self.assertEqual(dist1.shape, (2, 0))
        self.assertEqual(dist2.shape, (2, 0))

    def test_gradients_vs_reference(self):
        """
        Test gradients by comparing with a reference PyTorch implementation.
        This is more reliable than opcheck for CUDA ops using atomic operations.
        """
        device = "cuda"
        
        # Test with different sizes
        test_cases = [
            (1, 3, 4),   # small
            (2, 5, 7),   # medium  
            (1, 8, 6),   # different sizes
        ]
        
        for batch_size, n_points, m_points in test_cases:
            with self.subTest(batch_size=batch_size, n_points=n_points, m_points=m_points):
                p1 = torch.randn(batch_size, n_points, 3, device=device, requires_grad=True, dtype=torch.float32)
                p2 = torch.randn(batch_size, m_points, 3, device=device, requires_grad=True, dtype=torch.float32)
                
                # Our implementation
                dist1_our, dist2_our = chamfer.chamfer_distance(p1, p2)
                loss_our = dist1_our.mean() + dist2_our.mean()
                loss_our.backward()
                
                grad_p1_our = p1.grad.clone()
                grad_p2_our = p2.grad.clone()
                
                # Reset gradients
                p1.grad.zero_()
                p2.grad.zero_()
                
                # Reference implementation
                ref_dist1, ref_dist2 = reference_chamfer_distance(p1, p2)
                loss_ref = ref_dist1.mean() + ref_dist2.mean()
                loss_ref.backward()
                
                grad_p1_ref = p1.grad.clone()
                grad_p2_ref = p2.grad.clone()
                
                # Compare results
                self.assertTrue(torch.allclose(loss_our, loss_ref, atol=1e-6))
                self.assertTrue(torch.allclose(grad_p1_our, grad_p1_ref, atol=1e-4))
                self.assertTrue(torch.allclose(grad_p2_our, grad_p2_ref, atol=1e-4))

    def test_chamfer_reduction_gradients(self):
        """
        Test gradients for mean reduction following PyTorch3D testing patterns.
        """
        device = "cuda"
        N, P1, P2 = 2, 6, 8
        
        p1 = torch.randn(N, P1, 3, device=device, dtype=torch.float32)
        p2 = torch.randn(N, P2, 3, device=device, dtype=torch.float32)
        
        # Test mean reduction (most commonly used)
        p1_our = p1.clone().detach().requires_grad_(True)
        p2_our = p2.clone().detach().requires_grad_(True)
        
        model = ChamferDistance(reduction='mean')
        loss_our = model(p1_our, p2_our)
        loss_our.backward()
        
        grad_p1_our = p1_our.grad.clone()
        grad_p2_our = p2_our.grad.clone()
        
        # Reference implementation
        p1_ref = p1.clone().detach().requires_grad_(True)
        p2_ref = p2.clone().detach().requires_grad_(True)
        
        ref_dist1, ref_dist2 = reference_chamfer_distance(p1_ref, p2_ref)
        loss_ref = ref_dist1.mean() + ref_dist2.mean()
            
        loss_ref.backward()
        
        grad_p1_ref = p1_ref.grad.clone()
        grad_p2_ref = p2_ref.grad.clone()
        
        # Compare
        self.assertTrue(torch.allclose(loss_our, loss_ref, atol=1e-6))
        self.assertTrue(torch.allclose(grad_p1_our, grad_p1_ref, atol=1e-4))
        self.assertTrue(torch.allclose(grad_p2_our, grad_p2_ref, atol=1e-4))

    def _check_gradients(self, loss_our, loss_ref, p1_our, p1_ref, p2_our, p2_ref):
        """
        Helper function to check gradients following PyTorch3D patterns.
        """
        # Generate random upstream gradients
        grad_output = torch.rand_like(loss_our)
        
        # Compute gradients
        (loss_our * grad_output).sum().backward()
        (loss_ref * grad_output).sum().backward()
        
        # Compare gradients
        self.assertTrue(torch.allclose(p1_our.grad, p1_ref.grad, atol=1e-4))
        self.assertTrue(torch.allclose(p2_our.grad, p2_ref.grad, atol=1e-4))

    def test_opcheck(self):
        device = "cuda"
        p1_no_grad, p2_no_grad = self.sample_inputs(device, batch_size=1, n_points=3, m_points=3, requires_grad=False)
        
        # Test opcheck without gradients (atomic operations in CUDA backward cause non-determinism)
        opcheck(
            torch.ops.torch_point_ops_chamfer.chamfer_forward.default, 
            (p1_no_grad, p2_no_grad), 
            raise_exception=True
        )

    def test_compile(self):
        device = "cuda"
        model = ChamferDistance().to(device)
        compiled_model = torch.compile(model)
        
        p1, p2 = self.sample_inputs(device)
        
        result_eager = model(p1, p2)
        result_compiled = compiled_model(p1, p2)
        
        self.assertEqual(result_eager, result_compiled)

if __name__ == '__main__':
    unittest.main() 