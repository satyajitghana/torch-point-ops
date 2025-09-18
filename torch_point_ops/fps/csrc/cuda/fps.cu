#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Atomic.cuh>
#include <ATen/native/cuda/KernelUtils.cuh>
#include <torch/all.h>
#include <torch/library.h>
#include <c10/cuda/CUDAGuard.h>
#include <vector>
#include <tuple>

namespace torch_point_ops {

// Helper functions for type-safe comparisons (matching chamfer pattern)
template<typename scalar_t>
__device__ __forceinline__ bool is_less(const scalar_t& a, const scalar_t& b) {
  return a < b;
}

template<typename scalar_t>
__device__ __forceinline__ bool is_greater(const scalar_t& a, const scalar_t& b) {
  return a > b;
}

template<typename scalar_t>
__device__ __forceinline__ scalar_t min_val(const scalar_t& a, const scalar_t& b) {
  return a < b ? a : b;
}

template<typename scalar_t>
__device__ __forceinline__ scalar_t max_val(const scalar_t& a, const scalar_t& b) {
  return a > b ? a : b;
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
// Specializations for c10::Half to avoid operator ambiguity
template<>
__device__ __forceinline__ bool is_less<c10::Half>(const c10::Half& a, const c10::Half& b) {
  return static_cast<float>(a) < static_cast<float>(b);
}

template<>
__device__ __forceinline__ bool is_greater<c10::Half>(const c10::Half& a, const c10::Half& b) {
  return static_cast<float>(a) > static_cast<float>(b);
}

template<>
__device__ __forceinline__ c10::Half min_val<c10::Half>(const c10::Half& a, const c10::Half& b) {
  return static_cast<float>(a) < static_cast<float>(b) ? a : b;
}

template<>
__device__ __forceinline__ c10::Half max_val<c10::Half>(const c10::Half& a, const c10::Half& b) {
  return static_cast<float>(a) > static_cast<float>(b) ? a : b;
}
#endif

// Optimized thread calculation
inline unsigned int opt_n_threads(int work_size) {
  const int pow2 = std::min(512, 1 << (int)std::floor(std::log2(work_size)));
  return (unsigned int)pow2;
}

// Template-based gather points kernel
template<typename scalar_t>
__global__ void gather_points_kernel(int b, int c, int n, int m,
                                     const scalar_t *__restrict__ points,
                                     const int64_t *__restrict__ idx,
                                     scalar_t *__restrict__ out) {
  for (int i = blockIdx.x; i < b; i += gridDim.x) {
    for (int l = blockIdx.y; l < c; l += gridDim.y) {
      for (int j = threadIdx.x; j < m; j += blockDim.x) {
        int64_t a = idx[i * m + j];
        out[(i * c + l) * m + j] = points[(i * c + l) * n + a];
      }
    }
  }
}

// Template-based gather points gradient kernel
template<typename scalar_t>
__global__ void gather_points_grad_kernel(int b, int c, int n, int m,
                                          const scalar_t *__restrict__ grad_out,
                                          const int64_t *__restrict__ idx,
                                          scalar_t *__restrict__ grad_points,
                                          int64_t grad_points_numel) {
  for (int i = blockIdx.x; i < b; i += gridDim.x) {
    for (int l = blockIdx.y; l < c; l += gridDim.y) {
      for (int j = threadIdx.x; j < m; j += blockDim.x) {
        int64_t a = idx[i * m + j];
        int64_t idx_offset = (i * c + l) * n + a;
        scalar_t grad_val = grad_out[(i * c + l) * m + j];
        
        // Use optimized atomic operations like in chamfer
        at::native::fastSpecializedAtomicAdd(
            grad_points, idx_offset, grad_points_numel, grad_val);
      }
    }
  }
}

// Template-based update function for reduction
template<typename scalar_t>
__device__ void __update(scalar_t *__restrict__ dists, int *__restrict__ dists_i,
                         int idx1, int idx2) {
  const scalar_t v1 = dists[idx1], v2 = dists[idx2];
  const int i1 = dists_i[idx1], i2 = dists_i[idx2];
  dists[idx1] = max_val(v1, v2);
  dists_i[idx1] = is_greater(v2, v1) ? i2 : i1;
}

// Template-based FPS kernel
template<typename scalar_t, unsigned int block_size>
__global__ void furthest_point_sampling_kernel(
    int b, int n, int m, const scalar_t *__restrict__ dataset,
    scalar_t *__restrict__ temp, int64_t *__restrict__ idxs) {
  if (m <= 0) return;
  __shared__ scalar_t dists[block_size];
  __shared__ int dists_i[block_size];

  int batch_index = blockIdx.x;
  dataset += batch_index * n * 3;
  temp += batch_index * n;
  idxs += batch_index * m;

  int tid = threadIdx.x;
  const int stride = block_size;

  int old = 0;
  if (threadIdx.x == 0) idxs[0] = old;

  __syncthreads();
  for (int j = 1; j < m; j++) {
    int besti = 0;
    scalar_t best = scalar_t(-1);
    scalar_t x1 = dataset[old * 3 + 0];
    scalar_t y1 = dataset[old * 3 + 1];
    scalar_t z1 = dataset[old * 3 + 2];
    for (int k = tid; k < n; k += stride) {
      scalar_t x2, y2, z2;
      x2 = dataset[k * 3 + 0];
      y2 = dataset[k * 3 + 1];
      z2 = dataset[k * 3 + 2];
      scalar_t mag = (x2 * x2) + (y2 * y2) + (z2 * z2);
      if (!is_greater(mag, scalar_t(1e-3))) continue;

      scalar_t d = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1);
      scalar_t d2 = min_val(d, temp[k]);
      temp[k] = d2;
      besti = is_greater(d2, best) ? k : besti;
      best = is_greater(d2, best) ? d2 : best;
    }
    dists[tid] = best;
    dists_i[tid] = besti;
    __syncthreads();

    // Manual unrolling for better compatibility
    if (block_size >= 512) {
      if (tid < 256) {
        __update<scalar_t>(dists, dists_i, tid, tid + 256);
      }
      __syncthreads();
    }
    if (block_size >= 256) {
      if (tid < 128) {
        __update<scalar_t>(dists, dists_i, tid, tid + 128);
      }
      __syncthreads();
    }
    if (block_size >= 128) {
      if (tid < 64) {
        __update<scalar_t>(dists, dists_i, tid, tid + 64);
      }
      __syncthreads();
    }
    if (block_size >= 64) {
      if (tid < 32) {
        __update<scalar_t>(dists, dists_i, tid, tid + 32);
      }
      __syncthreads();
    }
    if (block_size >= 32) {
      if (tid < 16) {
        __update<scalar_t>(dists, dists_i, tid, tid + 16);
      }
      __syncthreads();
    }
    if (block_size >= 16) {
      if (tid < 8) {
        __update<scalar_t>(dists, dists_i, tid, tid + 8);
      }
      __syncthreads();
    }
    if (block_size >= 8) {
      if (tid < 4) {
        __update<scalar_t>(dists, dists_i, tid, tid + 4);
      }
      __syncthreads();
    }
    if (block_size >= 4) {
      if (tid < 2) {
        __update<scalar_t>(dists, dists_i, tid, tid + 2);
      }
      __syncthreads();
    }
    if (block_size >= 2) {
      if (tid < 1) {
        __update<scalar_t>(dists, dists_i, tid, tid + 1);
      }
      __syncthreads();
    }

    old = dists_i[0];
    if (tid == 0) idxs[j] = old;
  }
}

// Template dispatch function for different block sizes
template<typename scalar_t>
void dispatch_fps_kernel(int b, int n, int m,
                         const scalar_t* dataset, 
                         scalar_t* temp,
                         int64_t* idxs) {
  unsigned int n_threads = opt_n_threads(n);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  switch (n_threads) {
    case 512:
      furthest_point_sampling_kernel<scalar_t, 512>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
      break;
    case 256:
      furthest_point_sampling_kernel<scalar_t, 256>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
      break;
    case 128:
      furthest_point_sampling_kernel<scalar_t, 128>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
      break;
    case 64:
      furthest_point_sampling_kernel<scalar_t, 64>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
      break;
    case 32:
      furthest_point_sampling_kernel<scalar_t, 32>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
      break;
    case 16:
      furthest_point_sampling_kernel<scalar_t, 16>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
      break;
    case 8:
      furthest_point_sampling_kernel<scalar_t, 8>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
      break;
    case 4:
      furthest_point_sampling_kernel<scalar_t, 4>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
      break;
    case 2:
      furthest_point_sampling_kernel<scalar_t, 2>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
      break;
    case 1:
      furthest_point_sampling_kernel<scalar_t, 1>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
      break;
    default:
      furthest_point_sampling_kernel<scalar_t, 512>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
  }
}

// Main FPS forward function
torch::Tensor fps_cuda_forward(torch::Tensor points, int64_t nsamples) {
  // Check inputs
  const c10::Device device = points.device();
  at::cuda::CUDAGuard device_guard(device);
  
  const auto N = points.size(0);
  const auto P = points.size(1);
  const auto D = points.size(2);
  
  TORCH_CHECK(D == 3, "FPS only supports 3D points, but got dimension ", D);
  TORCH_CHECK(nsamples > 0, "nsamples must be positive, but got ", nsamples);
  TORCH_CHECK(nsamples <= P, "nsamples (", nsamples, ") cannot exceed number of points (", P, ")");
  
  // Create output tensor
  auto idxs = torch::zeros({N, nsamples}, points.options().dtype(torch::kInt64));
  
  // Create temporary distance tensor with same precision as input
  // Use smaller value for FP16 to avoid overflow (FP16 max ~65504)
  float large_val = (points.scalar_type() == torch::kFloat16) ? 1e4f : 1e10f;
  auto temp = torch::full({N, P}, large_val, points.options());
  
  if (idxs.numel() == 0) {
    return idxs;
  }
  
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(points.scalar_type(), "fps_forward", ([&] {
    // Ensure tensors are contiguous
    auto points_cont = points.contiguous();
    
    dispatch_fps_kernel<scalar_t>(
        N, P, nsamples,
        points_cont.data_ptr<scalar_t>(),
        temp.data_ptr<scalar_t>(),
        idxs.data_ptr<int64_t>());
  }));
  
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Error in fps_cuda_forward: %s\n", cudaGetErrorString(err));
  }
  
  return idxs;
}

// Gather points function
torch::Tensor gather_points_cuda_forward(torch::Tensor points, torch::Tensor idx) {
  // Check inputs
  const c10::Device device = points.device();
  TORCH_CHECK(points.device() == idx.device(), "points and idx must be on the same device");
  
  at::cuda::CUDAGuard device_guard(device);
  
  const auto N = points.size(0);
  const auto C = points.size(1);
  const auto P = points.size(2);
  const auto M = idx.size(1);
  
  // Create output tensor
  auto output = torch::zeros({N, C, M}, points.options());
  
  if (output.numel() == 0) {
    return output;
  }
  
  const int blocks = std::min(32, (int)N);
  const int threads = opt_n_threads(M);
  
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(points.scalar_type(), "gather_points_forward", ([&] {
    auto points_cont = points.contiguous();
    auto idx_cont = idx.contiguous();
    
    gather_points_kernel<scalar_t><<<dim3(blocks, C, 1), threads>>>(
        N, C, P, M,
        points_cont.data_ptr<scalar_t>(),
        idx_cont.data_ptr<int64_t>(),
        output.data_ptr<scalar_t>());
  }));
  
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Error in gather_points_cuda_forward: %s\n", cudaGetErrorString(err));
  }
  
  return output;
}

// Gather points backward function
torch::Tensor gather_points_cuda_backward(
    torch::Tensor grad_out, 
    torch::Tensor idx, 
    int64_t n) {
  
  // Check inputs
  const c10::Device device = grad_out.device();
  TORCH_CHECK(grad_out.device() == idx.device(), "grad_out and idx must be on the same device");
  
  at::cuda::CUDAGuard device_guard(device);
  
  const auto N = grad_out.size(0);
  const auto C = grad_out.size(1);
  const auto M = grad_out.size(2);
  
  // Create output tensor
  auto grad_points = torch::zeros({N, C, n}, grad_out.options());
  
  if (grad_points.numel() == 0) {
    return grad_points;
  }
  
  const int blocks = std::min(32, (int)N);
  const int threads = opt_n_threads(M);
  int64_t grad_points_numel = grad_points.numel();
  
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_out.scalar_type(), "gather_points_backward", ([&] {
    auto grad_out_cont = grad_out.contiguous();
    auto idx_cont = idx.contiguous();
    
    gather_points_grad_kernel<scalar_t><<<dim3(blocks, C, 1), threads>>>(
        N, C, n, M,
        grad_out_cont.data_ptr<scalar_t>(),
        idx_cont.data_ptr<int64_t>(),
        grad_points.data_ptr<scalar_t>(),
        grad_points_numel);
  }));
  
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Error in gather_points_cuda_backward: %s\n", cudaGetErrorString(err));
  }
  
  return grad_points;
}

TORCH_LIBRARY_IMPL(torch_point_ops_fps, CUDA, m) {
  m.impl("fps_forward", &fps_cuda_forward);
  m.impl("gather_points_forward", &gather_points_cuda_forward);
  m.impl("gather_points_backward", &gather_points_cuda_backward);
}

} // namespace torch_point_ops
