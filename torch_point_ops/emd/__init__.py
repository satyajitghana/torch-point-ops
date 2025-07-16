import torch
from . import _C


# Define the backward pass function
def _emd_backward(ctx, grad_cost, grad_match):
    xyz1, xyz2, match = ctx.saved_tensors
    grad_xyz1, grad_xyz2 = torch.ops.torch_point_ops_emd.emd_backward(
        grad_cost, xyz1, xyz2, match
    )
    return grad_xyz1, grad_xyz2


# Define the setup context function to save tensors for the backward pass
def _emd_setup_context(ctx, inputs, output):
    xyz1, xyz2 = inputs
    _, match = output
    ctx.save_for_backward(xyz1, xyz2, match)


# Register the autograd formula for the 'emd_forward' operator
torch.library.register_autograd(
    "torch_point_ops_emd::emd_forward", _emd_backward, setup_context=_emd_setup_context
)


@torch.library.register_fake("torch_point_ops_emd::emd_forward")
def _(xyz1, xyz2):
    batch_size, n, _ = xyz1.shape
    _, m, _ = xyz2.shape
    cost = torch.empty(batch_size, device=xyz1.device, dtype=xyz1.dtype)
    match = torch.empty(batch_size, m, n, device=xyz1.device, dtype=xyz1.dtype)
    return cost, match


@torch.library.register_fake("torch_point_ops_emd::emd_backward")
def _(grad_cost, xyz1, xyz2, match):
    return torch.empty_like(xyz1), torch.empty_like(xyz2)


# This is a wrapper to expose a single emd op to the user
# with autograd support.
def emd(xyz1: torch.Tensor, xyz2: torch.Tensor) -> torch.Tensor:
    """
    Earth Mover's Distance

    Args:
        xyz1 (torch.Tensor): (B, N, 3) - Point cloud 1
        xyz2 (torch.Tensor): (B, M, 3) - Point cloud 2

    Returns:
        torch.Tensor: (B) EMD distances

    Note:
        Tensors are automatically made contiguous internally - no need for users to call .contiguous()
    """
    # Ensure tensors are contiguous (required by CUDA implementation)
    xyz1 = xyz1.contiguous()
    xyz2 = xyz2.contiguous()

    cost, _ = torch.ops.torch_point_ops_emd.emd_forward(xyz1, xyz2)
    # Handle empty point clouds
    if xyz1.size(1) == 0:
        return torch.zeros_like(cost)
    return cost / xyz1.size(1)


# Alias for more descriptive naming
earth_movers_distance = emd

# Expose the emd function from the emd module
from .emd import EarthMoverDistance
