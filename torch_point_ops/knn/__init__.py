import torch
from . import _C


# Define the backward pass function
def _knn_backward(ctx, grad_idxs, grad_dists):
    p1, p2, lengths1, lengths2, idxs, norm = ctx.saved_tensors_and_info
    grad_p1, grad_p2 = torch.ops.torch_point_ops_knn.knn_backward(
        p1, p2, lengths1, lengths2, idxs, norm, grad_dists
    )
    return grad_p1, grad_p2, None, None, None, None, None


# Define the setup context function to save tensors for the backward pass
def _knn_setup_context(ctx, inputs, output):
    p1, p2, lengths1, lengths2, norm, K, version = inputs
    idxs, dists = output
    ctx.save_for_backward(p1, p2, lengths1, lengths2, idxs)
    ctx.saved_tensors_and_info = (p1, p2, lengths1, lengths2, idxs, norm)


# Register the autograd formula for the 'knn_forward' operator
torch.library.register_autograd(
    "torch_point_ops_knn::knn_forward",
    _knn_backward,
    setup_context=_knn_setup_context,
)


@torch.library.register_fake("torch_point_ops_knn::knn_forward")
def _(p1, p2, lengths1, lengths2, norm, K, version):
    batch_size, n, _ = p1.shape
    _, m, _ = p2.shape
    # Use input dtype for distances and int64 for indices
    idxs = torch.empty(batch_size, n, K, device=p1.device, dtype=torch.int64)
    dists = torch.empty(batch_size, n, K, device=p1.device, dtype=p1.dtype)
    return idxs, dists


@torch.library.register_fake("torch_point_ops_knn::knn_backward")
def _(p1, p2, lengths1, lengths2, idxs, norm, grad_dists):
    return torch.empty_like(p1), torch.empty_like(p2)


# This is a wrapper to expose a single knn_points op to the user
# with autograd support.
def knn_points(
    p1: torch.Tensor,
    p2: torch.Tensor,
    lengths1: torch.Tensor = None,
    lengths2: torch.Tensor = None,
    norm: int = 2,
    K: int = 1,
    version: int = -1,
    return_sorted: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    K-Nearest neighbors on point clouds.

    Args:
        p1 (torch.Tensor): (N, P1, D) - Point cloud 1
        p2 (torch.Tensor): (N, P2, D) - Point cloud 2
        lengths1 (torch.Tensor, optional): (N,) - Number of points in each cloud in p1.
                                          If None, assumes all clouds have P1 points.
        lengths2 (torch.Tensor, optional): (N,) - Number of points in each cloud in p2.
                                          If None, assumes all clouds have P2 points.
        norm (int): Distance norm. 1 for L1, 2 for L2. Default: 2
        K (int): Number of nearest neighbors to find. Default: 1
        version (int): Which kernel version to use. -1 for auto-select. Default: -1
        return_sorted (bool): Whether to return results sorted by distance. Default: True

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - idxs: (N, P1, K) indices of K nearest neighbors in p2 for each point in p1
            - dists: (N, P1, K) distances to K nearest neighbors

    Note:
        Tensors are automatically made contiguous internally - no need for users to call .contiguous()
    """
    # Input validation
    if p1.shape[0] != p2.shape[0]:
        raise ValueError("p1 and p2 must have the same batch dimension.")
    if p1.shape[2] != p2.shape[2]:
        raise ValueError("p1 and p2 must have the same point dimension.")

    N, P1, D = p1.shape
    _, P2, _ = p2.shape

    # Create default lengths if not provided (more efficient)
    if lengths1 is None:
        lengths1 = torch.full((N,), P1, dtype=torch.int64, device=p1.device)
    if lengths2 is None:
        lengths2 = torch.full((N,), P2, dtype=torch.int64, device=p1.device)

    # Ensure tensors are contiguous (only if necessary to avoid copies)
    if not p1.is_contiguous():
        p1 = p1.contiguous()
    if not p2.is_contiguous():
        p2 = p2.contiguous()
    if not lengths1.is_contiguous():
        lengths1 = lengths1.contiguous()
    if not lengths2.is_contiguous():
        lengths2 = lengths2.contiguous()

    # Call the operator
    idxs, dists = torch.ops.torch_point_ops_knn.knn_forward(
        p1, p2, lengths1, lengths2, norm, K, version
    )

    # Sort results if requested and K > 1
    if K > 1 and return_sorted:
        if lengths2.min() < K:
            # Handle case where some clouds have fewer than K points
            mask = lengths2[:, None] <= torch.arange(K, device=dists.device)[None]
            # mask has shape [N, K], true where dists irrelevant
            mask = mask[:, None].expand(-1, P1, -1)
            # mask has shape [N, P1, K], true where dists irrelevant
            dists_copy = dists.clone()
            dists_copy[mask] = float("inf")
            dists_sorted, sort_idx = dists_copy.sort(dim=2)
            dists_sorted[mask] = 0
            dists = dists_sorted
        else:
            dists, sort_idx = dists.sort(dim=2)
        idxs = idxs.gather(2, sort_idx)

    return idxs, dists


def knn_gather(
    x: torch.Tensor, idx: torch.Tensor, lengths: torch.Tensor = None
) -> torch.Tensor:
    """
    A helper function for knn that allows indexing a tensor x with the indices `idx`
    returned by `knn_points`.

    Args:
        x (torch.Tensor): (N, M, U) containing U-dimensional features to be gathered.
        idx (torch.Tensor): (N, L, K) giving the indices returned by `knn_points`.
        lengths (torch.Tensor, optional): (N,) of values in the range [0, M], giving the
                                         length of each example in the batch in x.
                                         Or None to indicate that every example has length M.

    Returns:
        torch.Tensor: (N, L, K, U) resulting from gathering the elements of x
                     with idx, s.t. `x_out[n, l, k] = x[n, idx[n, l, k]]`.
                     If `k > lengths[n]` then `x_out[n, l, k]` is filled with 0.0.
    """
    N, M, U = x.shape
    _N, L, K = idx.shape

    if N != _N:
        raise ValueError("x and idx must have same batch dimension.")

    if lengths is None:
        lengths = torch.full((x.shape[0],), M, dtype=torch.int64, device=x.device)

    idx_expanded = idx[:, :, :, None].expand(-1, -1, -1, U)
    # idx_expanded has shape [N, L, K, U]

    x_out = x[:, :, None].expand(-1, -1, K, -1).gather(1, idx_expanded)
    # x_out has shape [N, L, K, U]

    needs_mask = lengths.min() < K
    if needs_mask:
        # mask has shape [N, K], true where idx is irrelevant because
        # there is less number of points in p2 than K
        mask = lengths[:, None] <= torch.arange(K, device=x.device)[None]

        # expand mask to shape [N, L, K, U]
        mask = mask[:, None].expand(-1, L, -1)
        mask = mask[:, :, :, None].expand(-1, -1, -1, U)
        x_out[mask] = 0.0

    return x_out


# Expose the knn functionality from the knn module
from .knn import KNearestNeighbors
