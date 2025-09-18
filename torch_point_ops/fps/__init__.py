import torch
from . import _C


# Define the backward pass function for gather_points
def _gather_points_backward(ctx, grad_output):
    idx, n_points = ctx.saved_tensors_and_info
    grad_points = torch.ops.torch_point_ops_fps.gather_points_backward(
        grad_output, idx, n_points
    )
    return grad_points, None


# Define the setup context function for gather_points
def _gather_points_setup_context(ctx, inputs, output):
    points, idx = inputs
    ctx.save_for_backward(idx)
    ctx.saved_tensors_and_info = (idx, points.shape[2])


# Register the autograd formula for the 'gather_points_forward' operator
torch.library.register_autograd(
    "torch_point_ops_fps::gather_points_forward",
    _gather_points_backward,
    setup_context=_gather_points_setup_context,
)


@torch.library.register_fake("torch_point_ops_fps::fps_forward")
def _(points, nsamples):
    batch_size, n_points, _ = points.shape
    # FPS returns int64 indices
    idxs = torch.empty(batch_size, nsamples, device=points.device, dtype=torch.int64)
    return idxs


@torch.library.register_fake("torch_point_ops_fps::gather_points_forward")
def _(points, idx):
    batch_size, n_channels, _ = points.shape
    _, n_samples = idx.shape
    # Gather returns same dtype as input points
    output = torch.empty(batch_size, n_channels, n_samples, device=points.device, dtype=points.dtype)
    return output


@torch.library.register_fake("torch_point_ops_fps::gather_points_backward")
def _(grad_output, idx, n_points):
    batch_size, n_channels, _ = grad_output.shape
    return torch.empty(batch_size, n_channels, n_points, device=grad_output.device, dtype=grad_output.dtype)


@torch.library.register_fake("torch_point_ops_quick_fps::quick_fps_forward")
def _(points, nsamples, kd_depth=4):
    batch_size, n_points, _ = points.shape
    # Quick FPS returns int64 indices
    idxs = torch.empty(batch_size, nsamples, device=points.device, dtype=torch.int64)
    return idxs


def furthest_point_sampling(points, nsamples):
    """
    Furthest Point Sampling (FPS) algorithm.
    
    Iteratively selects points that are furthest from already selected points,
    providing a good coverage of the point cloud geometry.
    
    Args:
        points (torch.Tensor): Input point cloud of shape (N, P, 3) where:
            - N is the batch size
            - P is the number of points
            - 3 is the spatial dimension (x, y, z coordinates)
        nsamples (int): Number of points to sample (must be <= P)
        
    Returns:
        torch.Tensor: Indices of sampled points of shape (N, nsamples)
        
    Example:
        >>> import torch
        >>> from torch_point_ops.fps import furthest_point_sampling
        >>> 
        >>> # Create a random point cloud
        >>> points = torch.randn(2, 1000, 3).cuda()
        >>> 
        >>> # Sample 64 points using FPS
        >>> indices = furthest_point_sampling(points, 64)
        >>> print(indices.shape)  # (2, 64)
        >>> 
        >>> # Get the sampled points
        >>> sampled_points = torch.gather(points, 1, 
        ...     indices.unsqueeze(-1).expand(-1, -1, 3))
        >>> print(sampled_points.shape)  # (2, 64, 3)
        
    Note:
        - FPS is not differentiable with respect to point coordinates
        - The first selected point is always at index 0
        - Points too close to origin (magnitude < 1e-3) are skipped
        - Supports all floating point precisions (float16, float32, float64)
    """
    # Input validation
    if points.dim() != 3 or points.shape[2] != 3:
        raise ValueError(f"Expected points to have shape (N, P, 3), but got {points.shape}")
    if nsamples <= 0:
        raise ValueError(f"nsamples must be positive, but got {nsamples}")
    if nsamples > points.shape[1]:
        raise ValueError(f"nsamples ({nsamples}) cannot exceed number of points ({points.shape[1]})")
    
    # Ensure tensor is contiguous
    if not points.is_contiguous():
        points = points.contiguous()
    
    return torch.ops.torch_point_ops_fps.fps_forward(points, nsamples)


def gather_points(points, idx):
    """
    Gather points based on indices with gradient support.
    
    This function selects points from the input tensor based on the provided indices.
    It's commonly used after FPS to extract the sampled points while maintaining
    gradient flow for the point coordinates.
    
    Args:
        points (torch.Tensor): Input points of shape (N, C, P) where:
            - N is the batch size
            - C is the number of channels/features per point
            - P is the number of points
        idx (torch.Tensor): Indices to gather of shape (N, M) where:
            - N is the batch size
            - M is the number of points to gather
            
    Returns:
        torch.Tensor: Gathered points of shape (N, C, M)
        
    Example:
        >>> import torch
        >>> from torch_point_ops.fps import furthest_point_sampling, gather_points
        >>>
        >>> # Create point cloud with features
        >>> points = torch.randn(2, 6, 1000).cuda()  # 6 features per point
        >>> coords = points[:, :3, :]  # First 3 features are coordinates
        >>>
        >>> # Sample using FPS
        >>> indices = furthest_point_sampling(coords.transpose(1, 2), 64)
        >>>
        >>> # Gather all features for sampled points
        >>> sampled_features = gather_points(points, indices)
        >>> print(sampled_features.shape)  # (2, 6, 64)
        
    Note:
        - This function is differentiable with respect to the points tensor
        - Gradients are scattered back to the original point locations
        - Uses optimized atomic operations for gradient accumulation
    """
    # Input validation
    if points.dim() != 3:
        raise ValueError(f"Expected points to have 3 dimensions (N, C, P), but got {points.dim()}")
    if idx.dim() != 2:
        raise ValueError(f"Expected idx to have 2 dimensions (N, M), but got {idx.dim()}")
    if points.shape[0] != idx.shape[0]:
        raise ValueError(f"Batch dimensions must match: points {points.shape[0]} vs idx {idx.shape[0]}")
    
    # Ensure tensors are contiguous
    if not points.is_contiguous():
        points = points.contiguous()
    if not idx.is_contiguous():
        idx = idx.contiguous()
    
    return torch.ops.torch_point_ops_fps.gather_points_forward(points, idx)


# Convenience function that combines FPS and gathering
def farthest_point_sample_and_gather(points, nsamples):
    """
    Combine FPS and gathering in a single function.
    
    This is a convenience function that performs furthest point sampling
    and then gathers the sampled points, returning both indices and points.
    
    Args:
        points (torch.Tensor): Input point cloud of shape (N, P, 3)
        nsamples (int): Number of points to sample
        
    Returns:
        tuple[torch.Tensor, torch.Tensor]: 
            - indices: (N, nsamples) sampled indices
            - sampled_points: (N, nsamples, 3) sampled points
            
    Example:
        >>> import torch
        >>> from torch_point_ops.fps import farthest_point_sample_and_gather
        >>> 
        >>> points = torch.randn(2, 1000, 3).cuda()
        >>> indices, sampled_points = farthest_point_sample_and_gather(points, 64)
        >>> print(indices.shape)        # (2, 64)
        >>> print(sampled_points.shape) # (2, 64, 3)
    """
    indices = furthest_point_sampling(points, nsamples)
    # Transpose points to (N, 3, P) for gather_points, then transpose back
    points_t = points.transpose(1, 2)  # (N, 3, P)
    sampled_points_t = gather_points(points_t, indices)  # (N, 3, nsamples)
    sampled_points = sampled_points_t.transpose(1, 2)  # (N, nsamples, 3)
    return indices, sampled_points


def quick_furthest_point_sampling(points, nsamples, kd_depth=4):
    """
    Quick Furthest Point Sampling (FPS) algorithm using KD-tree spatial partitioning.
    
    This is an accelerated version of FPS that uses spatial partitioning to reduce
    computational complexity, especially beneficial for large point clouds.
    
    Args:
        points (torch.Tensor): Input point cloud of shape (N, P, 3) where:
            - N is the batch size
            - P is the number of points
            - 3 is the spatial dimension (x, y, z coordinates)
        nsamples (int): Number of points to sample (must be <= P)
        kd_depth (int, optional): Depth of KD-tree for spatial partitioning.
            Higher values provide better acceleration but use more memory.
            Default: 4 (16 spatial buckets)
        
    Returns:
        torch.Tensor: Indices of sampled points of shape (N, nsamples)
        
    Example:
        >>> import torch
        >>> from torch_point_ops.fps import quick_furthest_point_sampling
        >>> 
        >>> # Create a large random point cloud
        >>> points = torch.randn(2, 10000, 3).cuda()
        >>> 
        >>> # Sample 64 points using Quick FPS with deeper tree
        >>> indices = quick_furthest_point_sampling(points, 64, kd_depth=6)
        >>> print(indices.shape)  # (2, 64)
        >>> 
        >>> # Get the sampled points
        >>> sampled_points = torch.gather(points, 1, 
        ...     indices.unsqueeze(-1).expand(-1, -1, 3))
        >>> print(sampled_points.shape)  # (2, 64, 3)
        
    Note:
        - Quick FPS provides significant speedup over regular FPS for large point clouds
        - The first selected point is always at index 0
        - Points too close to origin (magnitude < 1e-3) are skipped  
        - Supports all floating point precisions (float16, float32, float64)
        - KD-tree depth controls speed vs. memory tradeoff:
          * depth=3: 8 buckets, fast but less spatial precision
          * depth=4: 16 buckets, good balance (default)
          * depth=6: 64 buckets, better for very large point clouds
          * depth=8: 256 buckets, maximum spatial precision
    """
    # Input validation
    if points.dim() != 3 or points.shape[2] != 3:
        raise ValueError(f"Expected points to have shape (N, P, 3), but got {points.shape}")
    if nsamples <= 0:
        raise ValueError(f"nsamples must be positive, but got {nsamples}")
    if nsamples > points.shape[1]:
        raise ValueError(f"nsamples ({nsamples}) cannot exceed number of points ({points.shape[1]})")
    if kd_depth <= 0 or kd_depth > 10:
        raise ValueError(f"kd_depth must be between 1 and 10, but got {kd_depth}")
    
    # Ensure tensor is contiguous
    if not points.is_contiguous():
        points = points.contiguous()
    
    return torch.ops.torch_point_ops_quick_fps.quick_fps_forward(points, nsamples, kd_depth)


# Convenience function that combines Quick FPS and gathering
def quick_farthest_point_sample_and_gather(points, nsamples, kd_depth=4):
    """
    Combine Quick FPS and gathering in a single function.
    
    This is a convenience function that performs quick furthest point sampling
    and then gathers the sampled points, returning both indices and points.
    
    Args:
        points (torch.Tensor): Input point cloud of shape (N, P, 3)
        nsamples (int): Number of points to sample
        kd_depth (int, optional): Depth of KD-tree for spatial partitioning. Default: 4
        
    Returns:
        tuple[torch.Tensor, torch.Tensor]: 
            - indices: (N, nsamples) sampled indices
            - sampled_points: (N, nsamples, 3) sampled points
            
    Example:
        >>> import torch
        >>> from torch_point_ops.fps import quick_farthest_point_sample_and_gather
        >>> 
        >>> points = torch.randn(2, 10000, 3).cuda()
        >>> indices, sampled_points = quick_farthest_point_sample_and_gather(
        ...     points, 64, kd_depth=6)
        >>> print(indices.shape)        # (2, 64)
        >>> print(sampled_points.shape) # (2, 64, 3)
    """
    indices = quick_furthest_point_sampling(points, nsamples, kd_depth)
    # Transpose points to (N, 3, P) for gather_points, then transpose back
    points_t = points.transpose(1, 2)  # (N, 3, P)
    sampled_points_t = gather_points(points_t, indices)  # (N, 3, nsamples)
    sampled_points = sampled_points_t.transpose(1, 2)  # (N, nsamples, 3)
    return indices, sampled_points


# Expose the fps functionality from the fps module
from .fps import (
    FarthestPointSampling, 
    PointGatherer, 
    FPSFunction,
    QuickFarthestPointSampling,
    QuickFPSFunction,
    fps_function,
    quick_fps_function
)

__all__ = [
    "furthest_point_sampling",
    "gather_points", 
    "farthest_point_sample_and_gather",
    "quick_furthest_point_sampling",
    "quick_farthest_point_sample_and_gather",
    "FarthestPointSampling",
    "PointGatherer", 
    "FPSFunction",
    "QuickFarthestPointSampling",
    "QuickFPSFunction",
    "fps_function",
    "quick_fps_function",
]
