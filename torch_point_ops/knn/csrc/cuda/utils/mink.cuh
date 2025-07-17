#pragma once

#include <cuda_runtime.h>
#include <algorithm>

#ifndef CUDA_INF
#define CUDA_INF __int_as_float(0x7f800000)
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
      dists_[i] = INFINITY;
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
      local_dists_[i] = INFINITY;
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
    dists_(dists), idxs_(idxs), min_dist_(INFINITY), min_idx_(-1), size_(0) {}

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
    min_dists_[0] = min_dists_[1] = INFINITY;
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