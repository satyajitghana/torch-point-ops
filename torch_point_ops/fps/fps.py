import torch
from . import furthest_point_sampling, gather_points, quick_furthest_point_sampling


class FarthestPointSampling(torch.nn.Module):
    """
    Furthest Point Sampling module for point clouds.
    
    This module performs furthest point sampling (FPS) to downsample point clouds
    while maintaining good spatial coverage. FPS is commonly used in 3D deep learning
    pipelines for point cloud processing.
    """
    
    def __init__(self, nsamples, return_gathered=False):
        """
        Initialize the FPS module.
        
        Args:
            nsamples (int): Number of points to sample
            return_gathered (bool): If True, return both indices and gathered points.
                                  If False, return only indices. Default: False
        """
        super().__init__()
        if nsamples <= 0:
            raise ValueError(f"nsamples must be positive, but got {nsamples}")
        
        self.nsamples = nsamples
        self.return_gathered = return_gathered
    
    def forward(self, points):
        """
        Apply furthest point sampling to input points.
        
        Args:
            points (torch.Tensor): Input point cloud of shape (N, P, 3)
            
        Returns:
            If return_gathered is False:
                torch.Tensor: Indices of sampled points of shape (N, nsamples)
            If return_gathered is True:
                tuple[torch.Tensor, torch.Tensor]:
                    - indices: (N, nsamples) sampled indices
                    - sampled_points: (N, nsamples, 3) sampled points
        """
        indices = furthest_point_sampling(points, self.nsamples)
        
        if self.return_gathered:
            # Transpose for gather_points and back
            points_t = points.transpose(1, 2)  # (N, 3, P)
            sampled_points_t = gather_points(points_t, indices)  # (N, 3, nsamples)
            sampled_points = sampled_points_t.transpose(1, 2)  # (N, nsamples, 3)
            return indices, sampled_points
        else:
            return indices


class PointGatherer(torch.nn.Module):
    """
    Point gathering module with gradient support.
    
    This module gathers points based on provided indices, maintaining gradient
    flow for the point coordinates. Commonly used after sampling operations.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, points, indices):
        """
        Gather points based on indices.
        
        Args:
            points (torch.Tensor): Input points of shape (N, C, P)
            indices (torch.Tensor): Indices to gather of shape (N, M)
            
        Returns:
            torch.Tensor: Gathered points of shape (N, C, M)
        """
        return gather_points(points, indices)


class FPSFunction(torch.nn.Module):
    """
    Functional interface for FPS that matches other torch-point-ops modules.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        points,
        nsamples,
        return_gathered=False,
    ):
        """
        Functional FPS interface.
        
        Args:
            points (torch.Tensor): Input point cloud of shape (N, P, 3)
            nsamples (int): Number of points to sample
            return_gathered (bool): Whether to return gathered points. Default: False
            
        Returns:
            Same as FarthestPointSampling.forward()
        """
        indices = furthest_point_sampling(points, nsamples)
        
        if return_gathered:
            points_t = points.transpose(1, 2)
            sampled_points_t = gather_points(points_t, indices)
            sampled_points = sampled_points_t.transpose(1, 2)
            return indices, sampled_points
        else:
            return indices


class QuickFarthestPointSampling(torch.nn.Module):
    """
    Quick Furthest Point Sampling module using KD-tree spatial partitioning.
    
    This module performs accelerated furthest point sampling (FPS) that uses spatial
    partitioning to reduce computational complexity, especially beneficial for large 
    point clouds.
    """
    
    def __init__(self, nsamples, kd_depth=4, return_gathered=False):
        """
        Initialize the Quick FPS module.
        
        Args:
            nsamples (int): Number of points to sample
            kd_depth (int): Depth of KD-tree for spatial partitioning. Default: 4
            return_gathered (bool): If True, return both indices and gathered points.
                                  If False, return only indices. Default: False
        """
        super().__init__()
        if nsamples <= 0:
            raise ValueError(f"nsamples must be positive, but got {nsamples}")
        if kd_depth <= 0 or kd_depth > 10:
            raise ValueError(f"kd_depth must be between 1 and 10, but got {kd_depth}")
        
        self.nsamples = nsamples
        self.kd_depth = kd_depth
        self.return_gathered = return_gathered
    
    def forward(self, points):
        """
        Apply quick furthest point sampling to input points.
        
        Args:
            points (torch.Tensor): Input point cloud of shape (N, P, 3)
            
        Returns:
            If return_gathered is False:
                torch.Tensor: Indices of sampled points of shape (N, nsamples)
            If return_gathered is True:
                tuple[torch.Tensor, torch.Tensor]:
                    - indices: (N, nsamples) sampled indices
                    - sampled_points: (N, nsamples, 3) sampled points
        """
        indices = quick_furthest_point_sampling(points, self.nsamples, self.kd_depth)
        
        if self.return_gathered:
            # Transpose for gather_points and back
            points_t = points.transpose(1, 2)  # (N, 3, P)
            sampled_points_t = gather_points(points_t, indices)  # (N, 3, nsamples)
            sampled_points = sampled_points_t.transpose(1, 2)  # (N, nsamples, 3)
            return indices, sampled_points
        else:
            return indices


class QuickFPSFunction(torch.nn.Module):
    """
    Functional interface for Quick FPS that matches other torch-point-ops modules.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        points,
        nsamples,
        kd_depth=4,
        return_gathered=False,
    ):
        """
        Functional Quick FPS interface.
        
        Args:
            points (torch.Tensor): Input point cloud of shape (N, P, 3)
            nsamples (int): Number of points to sample
            kd_depth (int): Depth of KD-tree for spatial partitioning. Default: 4
            return_gathered (bool): Whether to return gathered points. Default: False
            
        Returns:
            Same as QuickFarthestPointSampling.forward()
        """
        indices = quick_furthest_point_sampling(points, nsamples, kd_depth)
        
        if return_gathered:
            points_t = points.transpose(1, 2)
            sampled_points_t = gather_points(points_t, indices)
            sampled_points = sampled_points_t.transpose(1, 2)
            return indices, sampled_points
        else:
            return indices


# Create default functional instances for convenience
fps_function = FPSFunction()
quick_fps_function = QuickFPSFunction()
