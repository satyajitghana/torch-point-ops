import torch
from . import emd

class EarthMoverDistance(torch.nn.Module):
    """
    Computes the Earth Mover's Distance between two point clouds.
    """
    def forward(self, xyz1, xyz2):
        """
        Args:
            xyz1 (torch.Tensor): (B, N, 3)
            xyz2 (torch.Tensor): (B, M, 3)
        Returns:
            torch.Tensor: (1) mean EMD distance
        """
        cost = emd(xyz1, xyz2)
        return cost.mean() 