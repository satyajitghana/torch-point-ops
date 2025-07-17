#pragma once

#include <cuda_runtime.h>
#include <algorithm>

namespace torch_point_ops {

// Helper for dispatching on K given fixed D
template<template<typename, int64_t, int64_t> class Functor, typename scalar_t, 
         int64_t D, int64_t MIN_K, int64_t MAX_K>
void DispatchKernel2D_K(
    int64_t K,
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
  
  if constexpr (MIN_K <= MAX_K) {
    if (K == MIN_K) {
      Functor<scalar_t, D, MIN_K>::run(blocks, threads, points1, points2, lengths1, lengths2,
                                       dists, idxs, N, P1, P2, norm);
      return;
    }
    if constexpr (MIN_K < MAX_K) {
      DispatchKernel2D_K<Functor, scalar_t, D, MIN_K + 1, MAX_K>(
          K, blocks, threads, points1, points2, lengths1, lengths2,
          dists, idxs, N, P1, P2, norm);
    }
  }
}

// 1D dispatch for kernels that need template specialization on dimension D
template<template<typename, int64_t> class Functor, typename scalar_t, int64_t MIN_D, int64_t MAX_D>
void DispatchKernel1D(
    int64_t D,
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
  
  // Use compile-time recursion to dispatch to the correct template
  if constexpr (MIN_D <= MAX_D) {
    if (D == MIN_D) {
      Functor<scalar_t, MIN_D>::run(blocks, threads, points1, points2, lengths1, lengths2, 
                                    dists, idxs, N, P1, P2, K, norm);
      return;
    }
    if constexpr (MIN_D < MAX_D) {
      DispatchKernel1D<Functor, scalar_t, MIN_D + 1, MAX_D>(
          D, blocks, threads, points1, points2, lengths1, lengths2, 
          dists, idxs, N, P1, P2, K, norm);
    }
  }
}

// 2D dispatch for kernels that need template specialization on both D and K
template<template<typename, int64_t, int64_t> class Functor, typename scalar_t, 
         int64_t MIN_D, int64_t MAX_D, int64_t MIN_K, int64_t MAX_K>
void DispatchKernel2D(
    int64_t D,
    int64_t K,
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
  
  // First dispatch on D
  if constexpr (MIN_D <= MAX_D) {
    if (D == MIN_D) {
      DispatchKernel2D_K<Functor, scalar_t, MIN_D, MIN_K, MAX_K>(
          K, blocks, threads, points1, points2, lengths1, lengths2,
          dists, idxs, N, P1, P2, norm);
      return;
    }
    if constexpr (MIN_D < MAX_D) {
      DispatchKernel2D<Functor, scalar_t, MIN_D + 1, MAX_D, MIN_K, MAX_K>(
          D, K, blocks, threads, points1, points2, lengths1, lengths2,
          dists, idxs, N, P1, P2, norm);
    }
  }
}

// Version checker for determining which kernel version to use
inline bool InBounds(const int64_t min, const int64_t x, const int64_t max) {
  return min <= x && x <= max;
}

// Kernel version bounds
constexpr int V1_MIN_D = 1;
constexpr int V1_MAX_D = 32;

constexpr int V2_MIN_D = 1;
constexpr int V2_MAX_D = 8;
constexpr int V2_MIN_K = 1;
constexpr int V2_MAX_K = 32;

constexpr int V3_MIN_D = 1;
constexpr int V3_MAX_D = 8;
constexpr int V3_MIN_K = 1;
constexpr int V3_MAX_K = 4;

inline bool KnnCheckVersion(int version, const int64_t D, const int64_t K) {
  if (version == 0) {
    return true;
  } else if (version == 1) {
    return InBounds(V1_MIN_D, D, V1_MAX_D);
  } else if (version == 2) {
    return InBounds(V2_MIN_D, D, V2_MAX_D) && InBounds(V2_MIN_K, K, V2_MAX_K);
  } else if (version == 3) {
    return InBounds(V3_MIN_D, D, V3_MAX_D) && InBounds(V3_MIN_K, K, V3_MAX_K);
  }
  return false;
}

inline int ChooseVersion(const int64_t D, const int64_t K) {
  for (int version = 3; version >= 1; version--) {
    if (KnnCheckVersion(version, D, K)) {
      return version;
    }
  }
  return 0;
}

} // namespace torch_point_ops 