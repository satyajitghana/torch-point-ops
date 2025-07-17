import torch
from . import knn_points


class KNearestNeighbors(torch.nn.Module):
    """
    K-Nearest Neighbors for point clouds.

    This module finds the K nearest neighbors between two point clouds
    and optionally returns the gathered neighbor points.
    """

    def __init__(self, K=1, norm=2, return_nn=False, return_sorted=True, version=-1):
        """
        Initialize the KNN module.

        Args:
            K (int): Number of nearest neighbors to find. Default: 1
            norm (int): Distance norm. 1 for L1, 2 for L2. Default: 2
            return_nn (bool): Whether to return the actual neighbor points. Default: False
            return_sorted (bool): Whether to return results sorted by distance. Default: True
            version (int): Which kernel version to use. -1 for auto-select. Default: -1
        """
        super().__init__()
        if K < 1:
            raise ValueError(f"K must be >= 1, but got {K}")
        if norm not in [1, 2]:
            raise ValueError(f"norm must be 1 or 2, but got {norm}")

        self.K = K
        self.norm = norm
        self.return_nn = return_nn
        self.return_sorted = return_sorted
        self.version = version

    def forward(self, p1, p2, lengths1=None, lengths2=None):
        """
        Find K nearest neighbors between point clouds.

        Args:
            p1 (torch.Tensor): (N, P1, D) - Source point cloud
            p2 (torch.Tensor): (N, P2, D) - Target point cloud
            lengths1 (torch.Tensor, optional): (N,) - Number of points in each cloud in p1
            lengths2 (torch.Tensor, optional): (N,) - Number of points in each cloud in p2

        Returns:
            If return_nn is False:
                tuple[torch.Tensor, torch.Tensor]:
                    - idxs: (N, P1, K) indices of K nearest neighbors
                    - dists: (N, P1, K) distances to K nearest neighbors
            If return_nn is True:
                tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                    - idxs: (N, P1, K) indices of K nearest neighbors
                    - dists: (N, P1, K) distances to K nearest neighbors
                    - nn: (N, P1, K, D) the actual K nearest neighbor points
        """
        idxs, dists = knn_points(
            p1,
            p2,
            lengths1=lengths1,
            lengths2=lengths2,
            norm=self.norm,
            K=self.K,
            version=self.version,
            return_sorted=self.return_sorted,
        )

        if self.return_nn:
            from . import knn_gather

            nn = knn_gather(p2, idxs, lengths2)
            return idxs, dists, nn
        else:
            return idxs, dists


class KNNFunction(torch.nn.Module):
    """
    Functional interface for KNN that matches PyTorch3D style.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        p1,
        p2,
        lengths1=None,
        lengths2=None,
        norm=2,
        K=1,
        version=-1,
        return_nn=False,
        return_sorted=True,
    ):
        """
        Functional KNN interface.

        Args:
            p1 (torch.Tensor): (N, P1, D) - Source point cloud
            p2 (torch.Tensor): (N, P2, D) - Target point cloud
            lengths1 (torch.Tensor, optional): (N,) - Number of points in each cloud in p1
            lengths2 (torch.Tensor, optional): (N,) - Number of points in each cloud in p2
            norm (int): Distance norm. 1 for L1, 2 for L2. Default: 2
            K (int): Number of nearest neighbors to find. Default: 1
            version (int): Which kernel version to use. -1 for auto-select. Default: -1
            return_nn (bool): Whether to return the actual neighbor points. Default: False
            return_sorted (bool): Whether to return results sorted by distance. Default: True

        Returns:
            Same as KNearestNeighbors.forward()
        """
        idxs, dists = knn_points(
            p1,
            p2,
            lengths1=lengths1,
            lengths2=lengths2,
            norm=norm,
            K=K,
            version=version,
            return_sorted=return_sorted,
        )

        if return_nn:
            from . import knn_gather

            nn = knn_gather(p2, idxs, lengths2)
            return idxs, dists, nn
        else:
            return idxs, dists


# Create a default functional instance for convenience
knn_function = KNNFunction()
