import torch
from . import chamfer_distance


class ChamferDistance(torch.nn.Module):
    """
    Computes the Chamfer distance between two point clouds.
    """

    def __init__(self, reduction="mean"):
        super().__init__()
        if reduction not in ["mean", "sum", None, "none"]:
            raise ValueError(
                f"reduction must be one of ['mean', 'sum', None, 'none'], but got {reduction}"
            )
        self.reduction = reduction

    def forward(self, xyz1, xyz2):
        dist1, dist2 = chamfer_distance(xyz1, xyz2)

        if self.reduction == "mean":
            return torch.mean(dist1) + torch.mean(dist2)
        elif self.reduction == "sum":
            return torch.sum(dist1) + torch.sum(dist2)
        else:  # 'none' or None
            return dist1, dist2
