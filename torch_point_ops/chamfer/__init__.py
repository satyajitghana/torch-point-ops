import torch
from . import _C


# Define the backward pass function
def _chamfer_backward(ctx, grad_dist1, grad_dist2, grad_idx1, grad_idx2):
    xyz1, xyz2, idx1, idx2 = ctx.saved_tensors
    grad_xyz1, grad_xyz2 = torch.ops.torch_point_ops_chamfer.chamfer_backward(
        xyz1, xyz2, idx1, idx2, grad_dist1, grad_dist2
    )
    return grad_xyz1, grad_xyz2


# Define the setup context function to save tensors for the backward pass
def _chamfer_setup_context(ctx, inputs, output):
    xyz1, xyz2 = inputs
    _, _, idx1, idx2 = output
    ctx.save_for_backward(xyz1, xyz2, idx1, idx2)


# Register the autograd formula for the 'chamfer_forward' operator
torch.library.register_autograd(
    "torch_point_ops_chamfer::chamfer_forward",
    _chamfer_backward,
    setup_context=_chamfer_setup_context,
)


@torch.library.register_fake("torch_point_ops_chamfer::chamfer_forward")
def _(xyz1, xyz2):
    batch_size, n, _ = xyz1.shape
    _, m, _ = xyz2.shape
    dist1 = torch.empty(batch_size, n, device=xyz1.device, dtype=torch.float32)
    dist2 = torch.empty(batch_size, m, device=xyz1.device, dtype=torch.float32)
    idx1 = torch.empty(batch_size, n, device=xyz1.device, dtype=torch.int32)
    idx2 = torch.empty(batch_size, m, device=xyz1.device, dtype=torch.int32)
    return dist1, dist2, idx1, idx2


@torch.library.register_fake("torch_point_ops_chamfer::chamfer_backward")
def _(xyz1, xyz2, idx1, idx2, grad_dist1, grad_dist2):
    return torch.empty_like(xyz1), torch.empty_like(xyz2)


# This is a wrapper to expose a single chamfer_distance op to the user
# with autograd support.
def chamfer_distance(
    xyz1: torch.Tensor, xyz2: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Chamfer Distance between two point clouds

    Args:
        xyz1 (torch.Tensor): (B, N, 3) - Point cloud 1
        xyz2 (torch.Tensor): (B, M, 3) - Point cloud 2

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - dist1: (B, N) distances from xyz1 to xyz2
            - dist2: (B, M) distances from xyz2 to xyz1

    Note:
        Tensors are automatically made contiguous internally - no need for users to call .contiguous()
    """
    # Ensure tensors are contiguous (required by CUDA implementation)
    xyz1 = xyz1.contiguous()
    xyz2 = xyz2.contiguous()

    dist1, dist2, _, _ = torch.ops.torch_point_ops_chamfer.chamfer_forward(xyz1, xyz2)
    return dist1, dist2


# Expose the chamfer_distance function from the chamfer module
from .chamfer import ChamferDistance
