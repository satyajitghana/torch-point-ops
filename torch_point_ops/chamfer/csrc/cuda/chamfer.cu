#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Atomic.cuh>
#include <ATen/native/cuda/KernelUtils.cuh>
#include <torch/all.h>
#include <torch/library.h>
#include <c10/cuda/CUDAGuard.h>
#include <vector>
#include <tuple>

namespace torch_point_ops {

// Helper functions for type-safe comparisons
template<typename scalar_t>
__device__ __forceinline__ bool is_less(const scalar_t& a, const scalar_t& b) {
  return a < b;
}

template<typename scalar_t>
__device__ __forceinline__ bool is_greater(const scalar_t& a, const scalar_t& b) {
  return a > b;
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
#endif

template<typename scalar_t>
__global__ void chamfer_dist_kernel(int batch_size,
                                    int n,
                                    const scalar_t* xyz1,
                                    int m,
                                    const scalar_t* xyz2,
                                    scalar_t* dist,
                                    int* indexes) {
  const int batch = 512;
  __shared__ scalar_t buf[batch * 3];
  for (int i = blockIdx.x; i < batch_size; i += gridDim.x) {
    for (int k2 = 0; k2 < m; k2 += batch) {
      int end_k = min(m, k2 + batch) - k2;
      for (int j = threadIdx.x; j < end_k * 3; j += blockDim.x) {
        buf[j] = xyz2[(i * m + k2) * 3 + j];
      }
      __syncthreads();
      for (int j = threadIdx.x + blockIdx.y * blockDim.x; j < n;
           j += blockDim.x * gridDim.y) {
        scalar_t x1            = xyz1[(i * n + j) * 3 + 0];
        scalar_t y1            = xyz1[(i * n + j) * 3 + 1];
        scalar_t z1            = xyz1[(i * n + j) * 3 + 2];
        scalar_t best_dist     = 0;
        int best_dist_index = 0;
        int end_ka          = end_k - (end_k & 3);
        if (end_ka == batch) {
          for (int k = 0; k < batch; k += 4) {
            {
              scalar_t x2   = buf[k * 3 + 0] - x1;
              scalar_t y2   = buf[k * 3 + 1] - y1;
              scalar_t z2   = buf[k * 3 + 2] - z1;
              scalar_t dist = x2 * x2 + y2 * y2 + z2 * z2;

              if (k == 0 || is_less(dist, best_dist)) {
                best_dist       = dist;
                best_dist_index = k + k2;
              }
            }
            {
              scalar_t x2   = buf[k * 3 + 3] - x1;
              scalar_t y2   = buf[k * 3 + 4] - y1;
              scalar_t z2   = buf[k * 3 + 5] - z1;
              scalar_t dist = x2 * x2 + y2 * y2 + z2 * z2;
              if (is_less(dist, best_dist)) {
                best_dist       = dist;
                best_dist_index = k + k2 + 1;
              }
            }
            {
              scalar_t x2   = buf[k * 3 + 6] - x1;
              scalar_t y2   = buf[k * 3 + 7] - y1;
              scalar_t z2   = buf[k * 3 + 8] - z1;
              scalar_t dist = x2 * x2 + y2 * y2 + z2 * z2;
              if (is_less(dist, best_dist)) {
                best_dist       = dist;
                best_dist_index = k + k2 + 2;
              }
            }
            {
              scalar_t x2   = buf[k * 3 + 9] - x1;
              scalar_t y2   = buf[k * 3 + 10] - y1;
              scalar_t z2   = buf[k * 3 + 11] - z1;
              scalar_t dist = x2 * x2 + y2 * y2 + z2 * z2;
              if (is_less(dist, best_dist)) {
                best_dist       = dist;
                best_dist_index = k + k2 + 3;
              }
            }
          }
        } else {
          for (int k = 0; k < end_ka; k += 4) {
            {
              scalar_t x2   = buf[k * 3 + 0] - x1;
              scalar_t y2   = buf[k * 3 + 1] - y1;
              scalar_t z2   = buf[k * 3 + 2] - z1;
              scalar_t dist = x2 * x2 + y2 * y2 + z2 * z2;
              if (k == 0 || is_less(dist, best_dist)) {
                best_dist       = dist;
                best_dist_index = k + k2;
              }
            }
            {
              scalar_t x2   = buf[k * 3 + 3] - x1;
              scalar_t y2   = buf[k * 3 + 4] - y1;
              scalar_t z2   = buf[k * 3 + 5] - z1;
              scalar_t dist = x2 * x2 + y2 * y2 + z2 * z2;
              if (is_less(dist, best_dist)) {
                best_dist       = dist;
                best_dist_index = k + k2 + 1;
              }
            }
            {
              scalar_t x2   = buf[k * 3 + 6] - x1;
              scalar_t y2   = buf[k * 3 + 7] - y1;
              scalar_t z2   = buf[k * 3 + 8] - z1;
              scalar_t dist = x2 * x2 + y2 * y2 + z2 * z2;
              if (is_less(dist, best_dist)) {
                best_dist       = dist;
                best_dist_index = k + k2 + 2;
              }
            }
            {
              scalar_t x2   = buf[k * 3 + 9] - x1;
              scalar_t y2   = buf[k * 3 + 10] - y1;
              scalar_t z2   = buf[k * 3 + 11] - z1;
              scalar_t dist = x2 * x2 + y2 * y2 + z2 * z2;
              if (is_less(dist, best_dist)) {
                best_dist       = dist;
                best_dist_index = k + k2 + 3;
              }
            }
          }
        }
        for (int k = end_ka; k < end_k; k++) {
          scalar_t x2   = buf[k * 3 + 0] - x1;
          scalar_t y2   = buf[k * 3 + 1] - y1;
          scalar_t z2   = buf[k * 3 + 2] - z1;
          scalar_t dist = x2 * x2 + y2 * y2 + z2 * z2;
          if (k == 0 || is_less(dist, best_dist)) {
            best_dist       = dist;
            best_dist_index = k + k2;
          }
        }
        if (k2 == 0 || is_greater(dist[(i * n + j)], best_dist)) {
          dist[(i * n + j)]    = best_dist;
          indexes[(i * n + j)] = best_dist_index;
        }
      }
      __syncthreads();
    }
  }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
chamfer_cuda_forward(torch::Tensor xyz1, torch::Tensor xyz2) {
  const int batch_size = xyz1.size(0);
  const int n          = xyz1.size(1);  // num_points point cloud A
  const int m          = xyz2.size(1);  // num_points point cloud B
  
  // Use the same dtype as input tensors instead of hardcoding float32
  torch::Tensor dist1 =
    torch::zeros({batch_size, n}, xyz1.options().dtype(xyz1.dtype()));
  torch::Tensor dist2 =
    torch::zeros({batch_size, m}, xyz1.options().dtype(xyz1.dtype()));
  torch::Tensor idx1 = torch::zeros({batch_size, n}, xyz1.options().dtype(torch::kInt));
  torch::Tensor idx2 = torch::zeros({batch_size, m}, xyz1.options().dtype(torch::kInt));

  // Use AT_DISPATCH_FLOATING_TYPES_AND_HALF for half precision support
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(xyz1.scalar_type(), "chamfer_forward", ([&] {
    chamfer_dist_kernel<scalar_t><<<dim3(32, 16, 1), 512>>>(
      batch_size, n, xyz1.data_ptr<scalar_t>(), m, xyz2.data_ptr<scalar_t>(),
      dist1.data_ptr<scalar_t>(), idx1.data_ptr<int>());
    chamfer_dist_kernel<scalar_t><<<dim3(32, 16, 1), 512>>>(
      batch_size, m, xyz2.data_ptr<scalar_t>(), n, xyz1.data_ptr<scalar_t>(),
      dist2.data_ptr<scalar_t>(), idx2.data_ptr<int>());
  }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Error in chamfer_cuda_forward: %s\n", cudaGetErrorString(err));
  }
  return {dist1, dist2, idx1, idx2};
}

template<typename scalar_t>
__global__ void chamfer_dist_grad_kernel(int b,
                                         int n,
                                         const scalar_t* xyz1,
                                         int m,
                                         const scalar_t* xyz2,
                                         const scalar_t* grad_dist1,
                                         const int* idx1,
                                         scalar_t* grad_xyz1,
                                         scalar_t* grad_xyz2,
                                         int64_t grad_xyz1_numel,
                                         int64_t grad_xyz2_numel) {
  for (int i = blockIdx.x; i < b; i += gridDim.x) {
    for (int j = threadIdx.x + blockIdx.y * blockDim.x; j < n;
         j += blockDim.x * gridDim.y) {
      scalar_t x1 = xyz1[(i * n + j) * 3 + 0];
      scalar_t y1 = xyz1[(i * n + j) * 3 + 1];
      scalar_t z1 = xyz1[(i * n + j) * 3 + 2];
      int j2   = idx1[i * n + j];
      scalar_t x2 = xyz2[(i * m + j2) * 3 + 0];
      scalar_t y2 = xyz2[(i * m + j2) * 3 + 1];
      scalar_t z2 = xyz2[(i * m + j2) * 3 + 2];
      scalar_t g  = grad_dist1[i * n + j] * scalar_t(2);
      
      // Use fastSpecializedAtomicAdd for optimized atomic operations (up to 6x speedup for half precision)
      // Cast indices to int64_t to match numel parameter type for template deduction
      at::native::fastSpecializedAtomicAdd(grad_xyz1, static_cast<int64_t>((i * n + j) * 3 + 0), grad_xyz1_numel, g * (x1 - x2));
      at::native::fastSpecializedAtomicAdd(grad_xyz1, static_cast<int64_t>((i * n + j) * 3 + 1), grad_xyz1_numel, g * (y1 - y2));
      at::native::fastSpecializedAtomicAdd(grad_xyz1, static_cast<int64_t>((i * n + j) * 3 + 2), grad_xyz1_numel, g * (z1 - z2));
      at::native::fastSpecializedAtomicAdd(grad_xyz2, static_cast<int64_t>((i * m + j2) * 3 + 0), grad_xyz2_numel, -(g * (x1 - x2)));
      at::native::fastSpecializedAtomicAdd(grad_xyz2, static_cast<int64_t>((i * m + j2) * 3 + 1), grad_xyz2_numel, -(g * (y1 - y2)));
      at::native::fastSpecializedAtomicAdd(grad_xyz2, static_cast<int64_t>((i * m + j2) * 3 + 2), grad_xyz2_numel, -(g * (z1 - z2)));
    }
  }
}

std::tuple<torch::Tensor, torch::Tensor>
chamfer_cuda_backward(torch::Tensor xyz1,
                      torch::Tensor xyz2,
                      torch::Tensor idx1,
                      torch::Tensor idx2,
                      torch::Tensor grad_dist1,
                      torch::Tensor grad_dist2) {
  const int batch_size    = xyz1.size(0);
  const int n             = xyz1.size(1);  // num_points point cloud A
  const int m             = xyz2.size(1);  // num_points point cloud B
  torch::Tensor grad_xyz1 = torch::zeros_like(xyz1);
  torch::Tensor grad_xyz2 = torch::zeros_like(xyz2);

  // Calculate numel for boundary checking in fastSpecializedAtomicAdd
  int64_t grad_xyz1_numel = grad_xyz1.numel();
  int64_t grad_xyz2_numel = grad_xyz2.numel();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(xyz1.scalar_type(), "chamfer_backward", ([&] {
    chamfer_dist_grad_kernel<scalar_t><<<dim3(1, 16, 1), 256>>>(
      batch_size, n, xyz1.data_ptr<scalar_t>(), m, xyz2.data_ptr<scalar_t>(),
      grad_dist1.data_ptr<scalar_t>(), idx1.data_ptr<int>(),
      grad_xyz1.data_ptr<scalar_t>(), grad_xyz2.data_ptr<scalar_t>(),
      grad_xyz1_numel, grad_xyz2_numel);
    chamfer_dist_grad_kernel<scalar_t><<<dim3(1, 16, 1), 256>>>(
      batch_size, m, xyz2.data_ptr<scalar_t>(), n, xyz1.data_ptr<scalar_t>(),
      grad_dist2.data_ptr<scalar_t>(), idx2.data_ptr<int>(),
      grad_xyz2.data_ptr<scalar_t>(), grad_xyz1.data_ptr<scalar_t>(),
      grad_xyz2_numel, grad_xyz1_numel);
  }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Error in chamfer_cuda_backward: %s\n", cudaGetErrorString(err));
  }
  return {grad_xyz1, grad_xyz2};
}

TORCH_LIBRARY_IMPL(torch_point_ops_chamfer, CUDA, m) {
  m.impl("chamfer_forward", &chamfer_cuda_forward);
  m.impl("chamfer_backward", &chamfer_cuda_backward);
}

} // namespace torch_point_ops 