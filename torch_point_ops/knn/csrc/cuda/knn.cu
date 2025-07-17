#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Atomic.cuh>
#include <ATen/native/cuda/KernelUtils.cuh>
#include <torch/all.h>
#include <torch/library.h>
#include <c10/cuda/CUDAGuard.h>
#include <vector>
#include <tuple>

#include "utils/dispatch.cuh"
#include "utils/mink.cuh"

namespace torch_point_ops {

// Version 0: Basic kernel with no template optimizations
template<typename scalar_t>
__global__ void KNearestNeighborKernelV0(
    const scalar_t* __restrict__ points1,
    const scalar_t* __restrict__ points2,
    const int64_t* __restrict__ lengths1,
    const int64_t* __restrict__ lengths2,
    scalar_t* __restrict__ dists,
    int64_t* __restrict__ idxs,
    const size_t N,
    const size_t P1,
    const size_t P2,
    const size_t D,
    const size_t K,
    const size_t norm) {
  
  const int64_t chunks_per_cloud = (1 + (P1 - 1) / blockDim.x);
  const int64_t chunks_to_do = N * chunks_per_cloud;
  
  for (int64_t chunk = blockIdx.x; chunk < chunks_to_do; chunk += gridDim.x) {
    const int64_t n = chunk / chunks_per_cloud;
    const int64_t start_point = blockDim.x * (chunk % chunks_per_cloud);
    int64_t p1 = start_point + threadIdx.x;
    
    if (p1 >= lengths1[n]) continue;
    
    int offset = n * P1 * K + p1 * K;
    int64_t length2 = lengths2[n];
    MinK<scalar_t, int64_t> mink(dists + offset, idxs + offset, K);
    
    for (int p2 = 0; p2 < length2; ++p2) {
      scalar_t dist = 0;
      for (int d = 0; d < D; ++d) {
        scalar_t coord1 = points1[n * P1 * D + p1 * D + d];
        scalar_t coord2 = points2[n * P2 * D + p2 * D + d];
        scalar_t diff = coord1 - coord2;
        if (norm == 2) {
          dist += diff * diff;
        } else {
          dist += (diff > 0) ? diff : -diff;
        }
      }
      mink.add(dist, p2);
    }
  }
}

// Version 1: Template specialization on D for better register usage
template<typename scalar_t, int64_t D>
__global__ void KNearestNeighborKernelV1(
    const scalar_t* __restrict__ points1,
    const scalar_t* __restrict__ points2,
    const int64_t* __restrict__ lengths1,
    const int64_t* __restrict__ lengths2,
    scalar_t* __restrict__ dists,
    int64_t* __restrict__ idxs,
    const size_t N,
    const size_t P1,
    const size_t P2,
    const size_t K,
    const size_t norm) {
  
  scalar_t cur_point[D];
  const int64_t chunks_per_cloud = (1 + (P1 - 1) / blockDim.x);
  const int64_t chunks_to_do = N * chunks_per_cloud;
  
  for (int64_t chunk = blockIdx.x; chunk < chunks_to_do; chunk += gridDim.x) {
    const int64_t n = chunk / chunks_per_cloud;
    const int64_t start_point = blockDim.x * (chunk % chunks_per_cloud);
    int64_t p1 = start_point + threadIdx.x;
    
    if (p1 >= lengths1[n]) continue;
    
    // Cache current point in registers
    for (int d = 0; d < D; ++d) {
      cur_point[d] = points1[n * P1 * D + p1 * D + d];
    }
    
    int offset = n * P1 * K + p1 * K;
    int64_t length2 = lengths2[n];
    MinK<scalar_t, int64_t> mink(dists + offset, idxs + offset, K);
    
    for (int p2 = 0; p2 < length2; ++p2) {
      scalar_t dist = 0;
      for (int d = 0; d < D; ++d) {
        scalar_t diff = cur_point[d] - points2[n * P2 * D + p2 * D + d];
        if (norm == 2) {
          dist += diff * diff;
        } else {
          dist += (diff > 0) ? diff : -diff;
        }
      }
      mink.add(dist, p2);
    }
  }
}

// Version 2: Template specialization on both D and K for better optimization
template<typename scalar_t, int64_t D, int64_t K>
__global__ void KNearestNeighborKernelV2(
    const scalar_t* __restrict__ points1,
    const scalar_t* __restrict__ points2,
    const int64_t* __restrict__ lengths1,
    const int64_t* __restrict__ lengths2,
    scalar_t* __restrict__ dists,
    int64_t* __restrict__ idxs,
    const int64_t N,
    const int64_t P1,
    const int64_t P2,
    const size_t norm) {
  
  scalar_t cur_point[D];
  scalar_t min_dists[K];
  int min_idxs[K];
  
  const int64_t chunks_per_cloud = (1 + (P1 - 1) / blockDim.x);
  const int64_t chunks_to_do = N * chunks_per_cloud;
  
  for (int64_t chunk = blockIdx.x; chunk < chunks_to_do; chunk += gridDim.x) {
    const int64_t n = chunk / chunks_per_cloud;
    const int64_t start_point = blockDim.x * (chunk % chunks_per_cloud);
    int64_t p1 = start_point + threadIdx.x;
    
    if (p1 >= lengths1[n]) continue;
    
    // Cache current point in registers
    for (int d = 0; d < D; ++d) {
      cur_point[d] = points1[n * P1 * D + p1 * D + d];
    }
    
    int64_t length2 = lengths2[n];
    MinK<scalar_t, int> mink(min_dists, min_idxs, K);
    
    for (int p2 = 0; p2 < length2; ++p2) {
      scalar_t dist = 0;
      for (int d = 0; d < D; ++d) {
        int offset = n * P2 * D + p2 * D + d;
        scalar_t diff = cur_point[d] - points2[offset];
        if (norm == 2) {
          dist += diff * diff;
        } else {
          dist += (diff > 0) ? diff : -diff;
        }
      }
      mink.add(dist, p2);
    }
    
    // Copy results back to global memory
    for (int k = 0; k < mink.size(); ++k) {
      idxs[n * P1 * K + p1 * K + k] = min_idxs[k];
      dists[n * P1 * K + p1 * K + k] = min_dists[k];
    }
  }
}

// Version 3: Optimized register version for very small K
template<typename scalar_t, int D, int K>
__global__ void KNearestNeighborKernelV3(
    const scalar_t* __restrict__ points1,
    const scalar_t* __restrict__ points2,
    const int64_t* __restrict__ lengths1,
    const int64_t* __restrict__ lengths2,
    scalar_t* __restrict__ dists,
    int64_t* __restrict__ idxs,
    const size_t N,
    const size_t P1,
    const size_t P2,
    const size_t norm) {
  
  scalar_t cur_point[D];
  scalar_t min_dists[K];
  int min_idxs[K];
  
  const int64_t chunks_per_cloud = (1 + (P1 - 1) / blockDim.x);
  const int64_t chunks_to_do = N * chunks_per_cloud;
  
  for (int64_t chunk = blockIdx.x; chunk < chunks_to_do; chunk += gridDim.x) {
    const int64_t n = chunk / chunks_per_cloud;
    const int64_t start_point = blockDim.x * (chunk % chunks_per_cloud);
    int64_t p1 = start_point + threadIdx.x;
    
    if (p1 >= lengths1[n]) continue;
    
    // Cache current point in registers
    for (int d = 0; d < D; ++d) {
      cur_point[d] = points1[n * P1 * D + p1 * D + d];
    }
    
    int64_t length2 = lengths2[n];
    RegisterMinK<scalar_t, int, K> mink(min_dists, min_idxs);
    
    for (int p2 = 0; p2 < length2; ++p2) {
      scalar_t dist = 0;
      for (int d = 0; d < D; ++d) {
        int offset = n * P2 * D + p2 * D + d;
        scalar_t diff = cur_point[d] - points2[offset];
        if (norm == 2) {
          dist += diff * diff;
        } else {
          dist += (diff > 0) ? diff : -diff;
        }
      }
      mink.add(dist, p2);
    }
    
    mink.finalize();
    // Copy results back to global memory
    for (int k = 0; k < mink.size(); ++k) {
      idxs[n * P1 * K + p1 * K + k] = min_idxs[k];
      dists[n * P1 * K + p1 * K + k] = min_dists[k];
    }
  }
}

// Functor wrappers for dispatch
template<typename scalar_t, int64_t D>
struct KNearestNeighborV1Functor {
  static void run(
      size_t blocks,
      size_t threads,
      const scalar_t* points1,
      const scalar_t* points2,
      const int64_t* lengths1,
      const int64_t* lengths2,
      scalar_t* dists,
      int64_t* idxs,
      const size_t N,
      const size_t P1,
      const size_t P2,
      const size_t K,
      const size_t norm) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    KNearestNeighborKernelV1<scalar_t, D><<<blocks, threads, 0, stream>>>(
        points1, points2, lengths1, lengths2, dists, idxs, N, P1, P2, K, norm);
  }
};

template<typename scalar_t, int64_t D, int64_t K>
struct KNearestNeighborV2Functor {
  static void run(
      size_t blocks,
      size_t threads,
      const scalar_t* points1,
      const scalar_t* points2,
      const int64_t* lengths1,
      const int64_t* lengths2,
      scalar_t* dists,
      int64_t* idxs,
      const int64_t N,
      const int64_t P1,
      const int64_t P2,
      const size_t norm) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    KNearestNeighborKernelV2<scalar_t, D, K><<<blocks, threads, 0, stream>>>(
        points1, points2, lengths1, lengths2, dists, idxs, N, P1, P2, norm);
  }
};

template<typename scalar_t, int64_t D, int64_t K>
struct KNearestNeighborV3Functor {
  static void run(
      size_t blocks,
      size_t threads,
      const scalar_t* points1,
      const scalar_t* points2,
      const int64_t* lengths1,
      const int64_t* lengths2,
      scalar_t* dists,
      int64_t* idxs,
      const size_t N,
      const size_t P1,
      const size_t P2,
      const size_t norm) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    KNearestNeighborKernelV3<scalar_t, D, K><<<blocks, threads, 0, stream>>>(
        points1, points2, lengths1, lengths2, dists, idxs, N, P1, P2, norm);
  }
};

// Main forward function
std::tuple<torch::Tensor, torch::Tensor> knn_cuda_forward(
    torch::Tensor p1,
    torch::Tensor p2,
    torch::Tensor lengths1,
    torch::Tensor lengths2,
    int64_t norm,
    int64_t K,
    int64_t version) {
  
  // Check inputs are on the same device
  const c10::Device device = p1.device();
  TORCH_CHECK(p1.device() == p2.device(), "p1 and p2 must be on the same device");
  TORCH_CHECK(p1.device() == lengths1.device(), "p1 and lengths1 must be on the same device");
  TORCH_CHECK(p1.device() == lengths2.device(), "p1 and lengths2 must be on the same device");
  
  // Set device guard
  at::cuda::CUDAGuard device_guard(device);
  
  const auto N = p1.size(0);
  const auto P1 = p1.size(1);
  const auto P2 = p2.size(1);
  const auto D = p1.size(2);
  const int64_t K_64 = K;
  
  TORCH_CHECK((norm == 1) || (norm == 2), "Norm must be 1 or 2.");
  TORCH_CHECK(p2.size(2) == D, "Point sets must have the same last dimension");
  
  // Create output tensors with same dtype as input
  auto long_dtype = lengths1.options().dtype(at::kLong);
  auto idxs = torch::zeros({N, P1, K}, long_dtype);
  auto dists = torch::zeros({N, P1, K}, p1.options());
  
  if (idxs.numel() == 0) {
    return std::make_tuple(idxs, dists);
  }
  
  // Auto-select version if requested
  if (version < 0) {
    version = ChooseVersion(D, K);
  } else if (!KnnCheckVersion(version, D, K)) {
    int new_version = ChooseVersion(D, K);
    // Warning: could add logging here
    version = new_version;
  }
  
  const size_t threads = 256;
  const size_t blocks = 256;
  
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(p1.scalar_type(), "knn_forward", ([&] {
    // Ensure tensors are contiguous
    auto p1_cont = p1.contiguous();
    auto p2_cont = p2.contiguous();
    auto lengths1_cont = lengths1.contiguous();
    auto lengths2_cont = lengths2.contiguous();
    
    if (version == 0) {
      KNearestNeighborKernelV0<scalar_t><<<blocks, threads>>>(
          p1_cont.data_ptr<scalar_t>(),
          p2_cont.data_ptr<scalar_t>(),
          lengths1_cont.data_ptr<int64_t>(),
          lengths2_cont.data_ptr<int64_t>(),
          dists.data_ptr<scalar_t>(),
          idxs.data_ptr<int64_t>(),
          N, P1, P2, D, K, norm);
    } else if (version == 1) {
      DispatchKernel1D<KNearestNeighborV1Functor, scalar_t, V1_MIN_D, V1_MAX_D>(
          D, blocks, threads,
          p1_cont.data_ptr<scalar_t>(),
          p2_cont.data_ptr<scalar_t>(),
          lengths1_cont.data_ptr<int64_t>(),
          lengths2_cont.data_ptr<int64_t>(),
          dists.data_ptr<scalar_t>(),
          idxs.data_ptr<int64_t>(),
          N, P1, P2, K, norm);
    } else if (version == 2) {
      DispatchKernel2D<KNearestNeighborV2Functor, scalar_t, V2_MIN_D, V2_MAX_D, V2_MIN_K, V2_MAX_K>(
          D, K_64, blocks, threads,
          p1_cont.data_ptr<scalar_t>(),
          p2_cont.data_ptr<scalar_t>(),
          lengths1_cont.data_ptr<int64_t>(),
          lengths2_cont.data_ptr<int64_t>(),
          dists.data_ptr<scalar_t>(),
          idxs.data_ptr<int64_t>(),
          N, P1, P2, norm);
    } else if (version == 3) {
      DispatchKernel2D<KNearestNeighborV3Functor, scalar_t, V3_MIN_D, V3_MAX_D, V3_MIN_K, V3_MAX_K>(
          D, K_64, blocks, threads,
          p1_cont.data_ptr<scalar_t>(),
          p2_cont.data_ptr<scalar_t>(),
          lengths1_cont.data_ptr<int64_t>(),
          lengths2_cont.data_ptr<int64_t>(),
          dists.data_ptr<scalar_t>(),
          idxs.data_ptr<int64_t>(),
          N, P1, P2, norm);
    }
  }));
  
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    TORCH_CHECK(false, "CUDA error in knn_forward: ", cudaGetErrorString(err));
  }
  
  return std::make_tuple(idxs, dists);
}

// Backward kernel with optimized atomic operations
template<typename scalar_t>
__global__ void KNearestNeighborBackwardKernel(
    const scalar_t* __restrict__ p1,
    const scalar_t* __restrict__ p2,
    const int64_t* __restrict__ lengths1,
    const int64_t* __restrict__ lengths2,
    const int64_t* __restrict__ idxs,
    const scalar_t* __restrict__ grad_dists,
    scalar_t* __restrict__ grad_p1,
    scalar_t* __restrict__ grad_p2,
    const size_t N,
    const size_t P1,
    const size_t P2,
    const size_t K,
    const size_t D,
    const size_t norm,
    int64_t grad_p1_numel,
    int64_t grad_p2_numel) {
  
  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = gridDim.x * blockDim.x;
  
  for (size_t i = tid; i < N * P1 * K * D; i += stride) {
    const size_t n = i / (P1 * K * D);
    size_t rem = i % (P1 * K * D);
    const size_t p1_idx = rem / (K * D);
    rem = rem % (K * D);
    const size_t k = rem / D;
    const size_t d = rem % D;
    
    const size_t num1 = lengths1[n];
    const size_t num2 = lengths2[n];
    
    if (p1_idx < num1) {
      const scalar_t grad_dist = grad_dists[n * P1 * K + p1_idx * K + k];
      const int64_t p2_idx = idxs[n * P1 * K + p1_idx * K + k];
      
      // Skip invalid indices
      if (p2_idx == -1) continue;
      
      const scalar_t p1_val = p1[n * P1 * D + p1_idx * D + d];
      const scalar_t p2_val = p2[n * P2 * D + p2_idx * D + d];

      scalar_t diff = 0;
      if (norm == 1) {
        scalar_t sign = (p1_val > p2_val) ? scalar_t(1) : scalar_t(-1);
        if (p1_val == p2_val) sign = scalar_t(0);
        diff = grad_dist * sign;
      } else { // norm == 2
        diff = scalar_t(2) * grad_dist * (p1_val - p2_val);
      }
      
      // Use fastSpecializedAtomicAdd for optimized atomic operations
      at::native::fastSpecializedAtomicAdd(
          grad_p1, 
          static_cast<int64_t>(n * P1 * D + p1_idx * D + d), 
          grad_p1_numel, 
          diff);
      at::native::fastSpecializedAtomicAdd(
          grad_p2, 
          static_cast<int64_t>(n * P2 * D + p2_idx * D + d), 
          grad_p2_numel, 
          -diff);
    }
  }
}

// Backward function
std::tuple<torch::Tensor, torch::Tensor> knn_cuda_backward(
    torch::Tensor p1,
    torch::Tensor p2,
    torch::Tensor lengths1,
    torch::Tensor lengths2,
    torch::Tensor idxs,
    int64_t norm,
    torch::Tensor grad_dists) {
  
  // Set device guard
  at::cuda::CUDAGuard device_guard(p1.device());
  
  const auto N = p1.size(0);
  const auto P1 = p1.size(1);
  const auto P2 = p2.size(1);
  const auto D = p1.size(2);
  const auto K = idxs.size(2);
  
  TORCH_CHECK(p1.size(2) == D, "Point sets must have the same last dimension");
  TORCH_CHECK(idxs.size(0) == N, "KNN idxs must have the same batch dimension");
  TORCH_CHECK(idxs.size(1) == P1, "KNN idxs must have the same point dimension as p1");
  TORCH_CHECK(grad_dists.size(0) == N);
  TORCH_CHECK(grad_dists.size(1) == P1);
  TORCH_CHECK(grad_dists.size(2) == K);
  
  auto grad_p1 = torch::zeros_like(p1);
  auto grad_p2 = torch::zeros_like(p2);
  
  if (grad_p1.numel() == 0 || grad_p2.numel() == 0) {
    return std::make_tuple(grad_p1, grad_p2);
  }
  
  int64_t grad_p1_numel = grad_p1.numel();
  int64_t grad_p2_numel = grad_p2.numel();
  
  const int blocks = 64;
  const int threads = 512;
  
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(p1.scalar_type(), "knn_backward", ([&] {
    // Ensure tensors are contiguous
    auto p1_cont = p1.contiguous();
    auto p2_cont = p2.contiguous();
    auto lengths1_cont = lengths1.contiguous();
    auto lengths2_cont = lengths2.contiguous();
    auto idxs_cont = idxs.contiguous();
    auto grad_dists_cont = grad_dists.contiguous();
    
    KNearestNeighborBackwardKernel<scalar_t><<<blocks, threads>>>(
        p1_cont.data_ptr<scalar_t>(),
        p2_cont.data_ptr<scalar_t>(),
        lengths1_cont.data_ptr<int64_t>(),
        lengths2_cont.data_ptr<int64_t>(),
        idxs_cont.data_ptr<int64_t>(),
        grad_dists_cont.data_ptr<scalar_t>(),
        grad_p1.data_ptr<scalar_t>(),
        grad_p2.data_ptr<scalar_t>(),
        N, P1, P2, K, D, norm, grad_p1_numel, grad_p2_numel);
  }));
  
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    TORCH_CHECK(false, "CUDA error in knn_backward: ", cudaGetErrorString(err));
  }
  
  return std::make_tuple(grad_p1, grad_p2);
}

TORCH_LIBRARY_IMPL(torch_point_ops_knn, CUDA, m) {
  m.impl("knn_forward", &knn_cuda_forward);
  m.impl("knn_backward", &knn_cuda_backward);
}

} // namespace torch_point_ops 