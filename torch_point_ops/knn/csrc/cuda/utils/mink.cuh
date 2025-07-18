#pragma once

#include <cuda_runtime.h>
#include <algorithm>
#include <limits>

// Helper to get infinity for different types
template<typename T>
struct NumericLimits {
    __device__ __forceinline__ static T lower_bound() {
        return std::numeric_limits<T>::lowest();
    }
    __device__ __forceinline__ static T upper_bound() {
        return std::numeric_limits<T>::max();
    }
    __device__ __forceinline__ static T inf() {
        return std::numeric_limits<T>::infinity();
    }
};

// Specializations for half and bfloat16 if necessary
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
#include <cuda_fp16.h>
template<>
struct NumericLimits<__half> {
    __device__ __forceinline__ static __half lower_bound() {
        return __float2half_rn(-65504.0f);
    }
    __device__ __forceinline__ static __half upper_bound() {
        return __float2half_rn(65504.0f);
    }
    __device__ __forceinline__ static __half inf() {
        return __float2half_rn(std::numeric_limits<float>::infinity());
    }
};
#endif


namespace torch_point_ops {

// MinK data structure for efficiently maintaining K smallest elements
// in global memory (for larger K values)
template<typename scalar_t, typename index_t>
class MinK {
public:
  __device__ MinK(scalar_t* dists, index_t* idxs, int K) : 
    dists_(dists), idxs_(idxs), K_(K), size_(0) {
    // Initialize with infinity
    for (int i = 0; i < K_; ++i) {
      dists_[i] = NumericLimits<scalar_t>::inf();
      idxs_[i] = -1;
    }
  }

  __device__ void add(scalar_t dist, index_t idx) {
    if (size_ < K_) {
      // Array not full, just add
      dists_[size_] = dist;
      idxs_[size_] = idx;
      size_++;
    } else {
      // Find max element
      int max_idx = 0;
      scalar_t max_dist = dists_[0];
      for (int i = 1; i < K_; ++i) {
        if (dists_[i] > max_dist) {
          max_dist = dists_[i];
          max_idx = i;
        }
      }
      
      // Replace if new distance is smaller
      if (dist < max_dist) {
        dists_[max_idx] = dist;
        idxs_[max_idx] = idx;
      }
    }
  }

  __device__ int size() const { return size_; }

private:
  scalar_t* dists_;
  index_t* idxs_;
  int K_;
  int size_;
};

// RegisterMinK: optimized version using registers for very small K
// Template specializations for common small K values
template<typename scalar_t, typename index_t, int K>
class RegisterMinK {
public:
  __device__ RegisterMinK(scalar_t* dists, index_t* idxs) : 
    dists_(dists), idxs_(idxs), size_(0) {
    // Initialize arrays
    for (int i = 0; i < K; ++i) {
      local_dists_[i] = NumericLimits<scalar_t>::inf();
      local_idxs_[i] = -1;
    }
  }

  __device__ void add(scalar_t dist, index_t idx) {
    if (size_ < K) {
      local_dists_[size_] = dist;
      local_idxs_[size_] = idx;
      size_++;
    } else {
      // Find max element
      int max_idx = 0;
      scalar_t max_dist = local_dists_[0];
      for (int i = 1; i < K; ++i) {
        if (local_dists_[i] > max_dist) {
          max_dist = local_dists_[i];
          max_idx = i;
        }
      }
      
      // Replace if new distance is smaller
      if (dist < max_dist) {
        local_dists_[max_idx] = dist;
        local_idxs_[max_idx] = idx;
      }
    }
  }

  __device__ int size() const { return size_; }

  __device__ void finalize() {
    // Copy back to output arrays
    for (int i = 0; i < size_; ++i) {
      dists_[i] = local_dists_[i];
      idxs_[i] = local_idxs_[i];
    }
  }

private:
  scalar_t local_dists_[K];
  index_t local_idxs_[K];
  scalar_t* dists_;
  index_t* idxs_;
  int size_;
};

// Specialization for K=1 (most common case)
template<typename scalar_t, typename index_t>
class RegisterMinK<scalar_t, index_t, 1> {
public:
  __device__ RegisterMinK(scalar_t* dists, index_t* idxs) : 
    dists_(dists), idxs_(idxs), min_dist_(NumericLimits<scalar_t>::inf()), min_idx_(-1), size_(0) {}

  __device__ void add(scalar_t dist, index_t idx) {
    if (size_ == 0 || dist < min_dist_) {
      min_dist_ = dist;
      min_idx_ = idx;
      if (size_ == 0) size_ = 1;
    }
  }

  __device__ int size() const { return size_; }

  __device__ void finalize() {
    if (size_ > 0) {
      dists_[0] = min_dist_;
      idxs_[0] = min_idx_;
    }
  }

private:
  scalar_t min_dist_;
  index_t min_idx_;
  scalar_t* dists_;
  index_t* idxs_;
  int size_;
};

// Specialization for K=2
template<typename scalar_t, typename index_t>
class RegisterMinK<scalar_t, index_t, 2> {
public:
  __device__ RegisterMinK(scalar_t* dists, index_t* idxs) : 
    dists_(dists), idxs_(idxs), size_(0) {
    min_dists_[0] = min_dists_[1] = NumericLimits<scalar_t>::inf();
    min_idxs_[0] = min_idxs_[1] = -1;
  }

  __device__ void add(scalar_t dist, index_t idx) {
    if (size_ == 0) {
      min_dists_[0] = dist;
      min_idxs_[0] = idx;
      size_ = 1;
    } else if (size_ == 1) {
      if (dist < min_dists_[0]) {
        min_dists_[1] = min_dists_[0];
        min_idxs_[1] = min_idxs_[0];
        min_dists_[0] = dist;
        min_idxs_[0] = idx;
      } else {
        min_dists_[1] = dist;
        min_idxs_[1] = idx;
      }
      size_ = 2;
    } else {
      if (dist < min_dists_[0]) {
        min_dists_[1] = min_dists_[0];
        min_idxs_[1] = min_idxs_[0];
        min_dists_[0] = dist;
        min_idxs_[0] = idx;
      } else if (dist < min_dists_[1]) {
        min_dists_[1] = dist;
        min_idxs_[1] = idx;
      }
    }
  }

  __device__ int size() const { return size_; }

  __device__ void finalize() {
    for (int i = 0; i < size_; ++i) {
      dists_[i] = min_dists_[i];
      idxs_[i] = min_idxs_[i];
    }
  }

private:
  scalar_t min_dists_[2];
  index_t min_idxs_[2];
  scalar_t* dists_;
  index_t* idxs_;
  int size_;
};

} // namespace torch_point_ops 