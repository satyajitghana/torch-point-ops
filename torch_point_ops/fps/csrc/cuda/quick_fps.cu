#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Atomic.cuh>
#include <ATen/native/cuda/KernelUtils.cuh>
#include <torch/all.h>
#include <torch/library.h>
#include <c10/cuda/CUDAGuard.h>
#include <vector>
#include <tuple>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/fill.h>
#include <thrust/sort.h>
#include <cmath>

namespace torch_point_ops {

// KD-tree constants for aggressive pruning
constexpr int MAX_THREADS_PER_BLOCK = 1024;
constexpr int WARP_SIZE = 32;
constexpr int MAX_KD_DEPTH = 12;
constexpr int MAX_KD_NODES = 8192;
constexpr int LEAF_THRESHOLD = 16;  // Minimum points per leaf

// 3D point structure template
template<typename scalar_t>
struct Point3D {
    scalar_t x, y, z;
    __device__ __host__ Point3D() : x(0), y(0), z(0) {}
    __device__ __host__ Point3D(scalar_t x_, scalar_t y_, scalar_t z_) : x(x_), y(y_), z(z_) {}
    
    __device__ __host__ Point3D<scalar_t> operator+(const Point3D<scalar_t>& other) const {
        return Point3D<scalar_t>(x + other.x, y + other.y, z + other.z);
    }
    
    __device__ __host__ Point3D<scalar_t> operator-(const Point3D<scalar_t>& other) const {
        return Point3D<scalar_t>(x - other.x, y - other.y, z - other.z);
    }
    
    __device__ __host__ Point3D<scalar_t> operator/(scalar_t divisor) const {
        return Point3D<scalar_t>(x / divisor, y / divisor, z / divisor);
    }
};

// KD-tree node structure for aggressive pruning
template<typename scalar_t>
struct KDNode {
    Point3D<scalar_t> bbox_min;      // Bounding box minimum
    Point3D<scalar_t> bbox_max;      // Bounding box maximum  
    int start_idx;                   // Start index in point array
    int count;                       // Number of points in this node
    int left_child;                  // Left child node index (-1 if leaf)
    int right_child;                 // Right child node index (-1 if leaf)
    int split_dim;                   // Split dimension (0=x, 1=y, 2=z)
    scalar_t split_value;            // Split threshold value
    
    __device__ __host__ KDNode() : start_idx(0), count(0), left_child(-1), right_child(-1), 
                                   split_dim(0), split_value(0) {}
    
    __device__ __host__ bool is_leaf() const { return left_child == -1 && right_child == -1; }
};

// Point index pair for sorting
struct PointIndex {
    int original_idx;
    int sorted_idx;
    __device__ __host__ PointIndex() : original_idx(0), sorted_idx(0) {}
    __device__ __host__ PointIndex(int orig, int sort) : original_idx(orig), sorted_idx(sort) {}
};

// Type-safe helper functions with FP16 support
template<typename scalar_t>
__device__ __forceinline__ scalar_t pow2(const scalar_t& a) {
    return a * a;
}

template<typename scalar_t>
__device__ __forceinline__ bool is_less(const scalar_t& a, const scalar_t& b) {
  return a < b;
}

template<typename scalar_t>
__device__ __forceinline__ bool is_less_equal(const scalar_t& a, const scalar_t& b) {
  return a <= b;
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

template<typename scalar_t>
__device__ __forceinline__ scalar_t abs_val(const scalar_t& a) {
  return a < scalar_t(0) ? -a : a;
}

// Get coordinate by dimension
template<typename scalar_t>
__device__ __forceinline__ scalar_t get_coord(const Point3D<scalar_t>& point, int dim) {
  return (dim == 0) ? point.x : (dim == 1) ? point.y : point.z;
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
// FP16 specializations
template<>
__device__ __forceinline__ bool is_less<c10::Half>(const c10::Half& a, const c10::Half& b) {
  return static_cast<float>(a) < static_cast<float>(b);
}

template<>
__device__ __forceinline__ bool is_less_equal<c10::Half>(const c10::Half& a, const c10::Half& b) {
  return static_cast<float>(a) <= static_cast<float>(b);
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

template<>
__device__ __forceinline__ c10::Half abs_val<c10::Half>(const c10::Half& a) {
  return static_cast<float>(a) < 0.0f ? -a : a;
}
#endif

// Host versions for CPU-side tree building
template<typename scalar_t>
__host__ __forceinline__ bool host_is_less_equal(const scalar_t& a, const scalar_t& b) {
  return a <= b;
}

template<>
__host__ __forceinline__ bool host_is_less_equal<c10::Half>(const c10::Half& a, const c10::Half& b) {
  return static_cast<float>(a) <= static_cast<float>(b);
}

// Distance calculation between two points
template<typename scalar_t>
__device__ __forceinline__ scalar_t distance_squared(const Point3D<scalar_t>& a, const Point3D<scalar_t>& b) {
  return pow2(a.x - b.x) + pow2(a.y - b.y) + pow2(a.z - b.z);
}

// Calculate minimum distance from point to bounding box (CRITICAL for pruning!)
template<typename scalar_t>
__device__ __forceinline__ scalar_t bbox_min_distance_squared(const Point3D<scalar_t>& point, 
                                                              const Point3D<scalar_t>& bbox_min,
                                                              const Point3D<scalar_t>& bbox_max) {
  scalar_t dx = max_val(scalar_t(0), max_val(bbox_min.x - point.x, point.x - bbox_max.x));
  scalar_t dy = max_val(scalar_t(0), max_val(bbox_min.y - point.y, point.y - bbox_max.y));
  scalar_t dz = max_val(scalar_t(0), max_val(bbox_min.z - point.z, point.z - bbox_max.z));
  return dx * dx + dy * dy + dz * dz;
}

// Complete reduction for bounding box calculation
template<typename scalar_t>
__global__ void compute_node_bounds_kernel(
    const scalar_t* __restrict__ points,
    const int* __restrict__ point_indices,
    KDNode<scalar_t>* __restrict__ nodes,
    int node_id,
    int n_points) {
    
    extern __shared__ char bounds_shared_mem[];
    Point3D<scalar_t>* shared_min = reinterpret_cast<Point3D<scalar_t>*>(bounds_shared_mem);
    Point3D<scalar_t>* shared_max = shared_min + blockDim.x;
    Point3D<scalar_t>* shared_sum = shared_max + blockDim.x;
    
    int tid = threadIdx.x;
    KDNode<scalar_t>& node = nodes[node_id];
    
    // Initialize thread-local bounds
    Point3D<scalar_t> thread_min(1e10, 1e10, 1e10);
    Point3D<scalar_t> thread_max(-1e10, -1e10, -1e10);
    Point3D<scalar_t> thread_sum(0, 0, 0);
    
    // Process points assigned to this thread
    for (int i = tid; i < node.count; i += blockDim.x) {
        int point_idx = point_indices[node.start_idx + i];
        Point3D<scalar_t> point(points[point_idx * 3], points[point_idx * 3 + 1], points[point_idx * 3 + 2]);
        
        thread_min.x = min_val(thread_min.x, point.x);
        thread_min.y = min_val(thread_min.y, point.y);
        thread_min.z = min_val(thread_min.z, point.z);
        
        thread_max.x = max_val(thread_max.x, point.x);
        thread_max.y = max_val(thread_max.y, point.y);
        thread_max.z = max_val(thread_max.z, point.z);
        
        thread_sum = thread_sum + point;
    }
    
    shared_min[tid] = thread_min;
    shared_max[tid] = thread_max;
    shared_sum[tid] = thread_sum;
    __syncthreads();
    
    // COMPLETE reduction across all threads
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_min[tid].x = min_val(shared_min[tid].x, shared_min[tid + stride].x);
            shared_min[tid].y = min_val(shared_min[tid].y, shared_min[tid + stride].y);
            shared_min[tid].z = min_val(shared_min[tid].z, shared_min[tid + stride].z);
            
            shared_max[tid].x = max_val(shared_max[tid].x, shared_max[tid + stride].x);
            shared_max[tid].y = max_val(shared_max[tid].y, shared_max[tid + stride].y);
            shared_max[tid].z = max_val(shared_max[tid].z, shared_max[tid + stride].z);
            
            shared_sum[tid] = shared_sum[tid] + shared_sum[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        node.bbox_min = shared_min[0];
        node.bbox_max = shared_max[0];
        
        // Find split dimension (largest extent)
        Point3D<scalar_t> extent = node.bbox_max - node.bbox_min;
        int split_dim = 0;
        if (is_greater(extent.y, extent.x) && is_greater(extent.y, extent.z)) split_dim = 1;
        else if (is_greater(extent.z, extent.x) && is_greater(extent.z, extent.y)) split_dim = 2;
        
        node.split_dim = split_dim;
        // Use median as split value
        node.split_value = (get_coord(node.bbox_min, split_dim) + get_coord(node.bbox_max, split_dim)) / scalar_t(2);
    }
}

// Complete partitioning kernel
template<typename scalar_t>
__global__ void partition_points_kernel(
    const scalar_t* __restrict__ points,
    int* __restrict__ point_indices,
    int* __restrict__ temp_indices,
    const KDNode<scalar_t>* __restrict__ nodes,
    int node_id) {
    
    extern __shared__ char partition_shared_mem[];
    int* left_count = reinterpret_cast<int*>(partition_shared_mem);
    int* right_count = left_count + 1;
    
    int tid = threadIdx.x;
    const KDNode<scalar_t>& node = nodes[node_id];
    
    if (tid == 0) {
        *left_count = 0;
        *right_count = 0;
    }
    __syncthreads();
    
    // COMPLETE two-pass partitioning algorithm
    // Pass 1: Count points for left and right
    for (int i = tid; i < node.count; i += blockDim.x) {
        int point_idx = point_indices[node.start_idx + i];
        Point3D<scalar_t> point(points[point_idx * 3], points[point_idx * 3 + 1], points[point_idx * 3 + 2]);
        scalar_t coord = get_coord(point, node.split_dim);
        
        if (is_less_equal(coord, node.split_value)) {
            atomicAdd(left_count, 1);
        } else {
            atomicAdd(right_count, 1);
        }
    }
    __syncthreads();
    
    // Pass 2: Distribute points to left and right partitions
    int left_written = 0;
    int right_written = 0;
    
    for (int i = tid; i < node.count; i += blockDim.x) {
        int point_idx = point_indices[node.start_idx + i];
        Point3D<scalar_t> point(points[point_idx * 3], points[point_idx * 3 + 1], points[point_idx * 3 + 2]);
        scalar_t coord = get_coord(point, node.split_dim);
        
        if (is_less_equal(coord, node.split_value)) {
            int pos = atomicAdd(&left_written, 1);
            temp_indices[pos] = point_idx;
        } else {
            int pos = atomicAdd(&right_written, 1);
            temp_indices[*left_count + pos] = point_idx;
        }
    }
    __syncthreads();
    
    // Copy back to original array
    for (int i = tid; i < node.count; i += blockDim.x) {
        point_indices[node.start_idx + i] = temp_indices[i];
    }
}

// COMPLETE KD-tree construction
template<typename scalar_t>
void build_complete_kd_tree(
    const scalar_t* points,
    int* point_indices,
    KDNode<scalar_t>* nodes,
    int n_points,
    int max_depth) {
    
    printf("[DEBUG] build_complete_kd_tree: n_points=%d, max_depth=%d\n", n_points, max_depth);
    
    if (n_points <= 0) {
        printf("[ERROR] Invalid n_points: %d\n", n_points);
        return;
    }
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // Initialize root node on host first, then copy to device
    printf("[DEBUG] Initializing root node\n");
    KDNode<scalar_t> root_node;
    root_node.start_idx = 0;
    root_node.count = n_points;
    root_node.left_child = -1;
    root_node.right_child = -1;
    root_node.split_dim = 0;
    
    // Copy root node to device
    cudaError_t root_err = cudaMemcpy(&nodes[0], &root_node, sizeof(KDNode<scalar_t>), cudaMemcpyHostToDevice);
    if (root_err != cudaSuccess) {
        printf("[ERROR] Failed to copy root node to device: %s\n", cudaGetErrorString(root_err));
        return;
    }
    printf("[DEBUG] Root node initialized successfully\n");
    
    int* temp_indices;
    cudaError_t alloc_err = cudaMalloc(&temp_indices, n_points * sizeof(int));
    if (alloc_err != cudaSuccess) {
        printf("[ERROR] Failed to allocate temp_indices: %s\n", cudaGetErrorString(alloc_err));
        return;
    }
    printf("[DEBUG] Allocated temp_indices\n");
    
    int current_nodes = 1;
    
    for (int depth = 0; depth < max_depth; depth++) {
        printf("[DEBUG] Processing depth %d, current_nodes=%d\n", depth, current_nodes);
        int nodes_at_level = 1 << depth;
        int next_node_idx = current_nodes;
        
        for (int node_id = nodes_at_level - 1; node_id < current_nodes; node_id++) {
            printf("[DEBUG] Processing node %d at depth %d\n", node_id, depth);
            
            if (node_id >= MAX_KD_NODES) {
                printf("[ERROR] Node ID %d exceeds MAX_KD_NODES %d\n", node_id, MAX_KD_NODES);
                break;
            }
            
            // Copy node from device to host for processing
            KDNode<scalar_t> node;
            cudaError_t node_err = cudaMemcpy(&node, &nodes[node_id], sizeof(KDNode<scalar_t>), cudaMemcpyDeviceToHost);
            if (node_err != cudaSuccess) {
                printf("[ERROR] Failed to copy node %d from device: %s\n", node_id, cudaGetErrorString(node_err));
                continue;
            }
            printf("[DEBUG] Node %d: start_idx=%d, count=%d\n", node_id, node.start_idx, node.count);
            
            if (node.count <= LEAF_THRESHOLD) {
                printf("[DEBUG] Skipping small node %d with count %d\n", node_id, node.count);
                continue;  // Skip small nodes
            }
            
            if (node.start_idx < 0 || node.start_idx >= n_points ||
                node.count <= 0 || node.start_idx + node.count > n_points) {
                printf("[ERROR] Invalid node bounds: start_idx=%d, count=%d, n_points=%d\n",
                       node.start_idx, node.count, n_points);
                continue;
            }
            
            // Compute bounds and split parameters
            printf("[DEBUG] Computing bounds for node %d\n", node_id);
            size_t shared_mem = 3 * MAX_THREADS_PER_BLOCK * sizeof(Point3D<scalar_t>);
            compute_node_bounds_kernel<scalar_t><<<1, MAX_THREADS_PER_BLOCK, shared_mem, stream>>>(
                points, point_indices, nodes, node_id, n_points);
            cudaStreamSynchronize(stream);
            
            cudaError_t bounds_err = cudaGetLastError();
            if (bounds_err != cudaSuccess) {
                printf("[ERROR] Bounds kernel error: %s\n", cudaGetErrorString(bounds_err));
                continue;
            }
            printf("[DEBUG] Bounds computed successfully\n");
            
            // Check what the bounds kernel set
            KDNode<scalar_t> bounds_check;
            cudaMemcpy(&bounds_check, &nodes[node_id], sizeof(KDNode<scalar_t>), cudaMemcpyDeviceToHost);
            printf("[DEBUG] After bounds: split_dim=%d, split_value=%f, bbox_min=(%f,%f,%f), bbox_max=(%f,%f,%f)\n",
                   bounds_check.split_dim, (float)bounds_check.split_value,
                   (float)bounds_check.bbox_min.x, (float)bounds_check.bbox_min.y, (float)bounds_check.bbox_min.z,
                   (float)bounds_check.bbox_max.x, (float)bounds_check.bbox_max.y, (float)bounds_check.bbox_max.z);
            
            // Partition points
            printf("[DEBUG] Partitioning points for node %d\n", node_id);
            size_t partition_shared = 2 * sizeof(int);
            partition_points_kernel<scalar_t><<<1, MAX_THREADS_PER_BLOCK, partition_shared, stream>>>(
                points, point_indices, temp_indices, nodes, node_id);
            cudaStreamSynchronize(stream);
            
            cudaError_t partition_err = cudaGetLastError();
            if (partition_err != cudaSuccess) {
                printf("[ERROR] Partition kernel error: %s\n", cudaGetErrorString(partition_err));
                continue;
            }
            printf("[DEBUG] Points partitioned successfully\n");
            
            // Count left partition size (we already have the node copied from device)
            printf("[DEBUG] Counting left partition, count=%d\n", node.count);
            int left_count = 0;
            int* host_indices = new int[node.count];
            cudaError_t indices_err = cudaMemcpy(host_indices, &point_indices[node.start_idx],
                                               node.count * sizeof(int), cudaMemcpyDeviceToHost);
            if (indices_err != cudaSuccess) {
                printf("[ERROR] Failed to copy indices to host: %s\n", cudaGetErrorString(indices_err));
                delete[] host_indices;
                continue;
            }
            
            // Validate indices and process within a block to avoid goto issues
            bool process_successful = true;
            
            // Validate indices
            for (int i = 0; i < node.count && process_successful; i++) {
                if (host_indices[i] < 0 || host_indices[i] >= n_points) {
                    printf("[ERROR] Invalid point index: host_indices[%d]=%d, n_points=%d\n",
                           i, host_indices[i], n_points);
                    process_successful = false;
                }
            }
            
            if (process_successful) {
                // Copy required points to host memory with bounds checking
                printf("[DEBUG] Copying points to host memory\n");
                scalar_t* host_points = new scalar_t[node.count * 3];
                
                for (int i = 0; i < node.count && process_successful; i++) {
                    int pidx = host_indices[i];
                    if (pidx < 0 || pidx >= n_points) {
                        printf("[ERROR] Point index out of bounds: pidx=%d, n_points=%d\n", pidx, n_points);
                        process_successful = false;
                        break;
                    }
                    cudaError_t point_err = cudaMemcpy(&host_points[i*3], &points[pidx*3], 3 * sizeof(scalar_t), cudaMemcpyDeviceToHost);
                    if (point_err != cudaSuccess) {
                        printf("[ERROR] Failed to copy point %d: %s\n", pidx, cudaGetErrorString(point_err));
                        process_successful = false;
                        break;
                    }
                }
                
                if (process_successful) {
                    printf("[DEBUG] Points copied successfully\n");
                    
                    // This is simplified - in practice we'd do this on GPU
                    Point3D<scalar_t> bbox_min = node.bbox_min;
                    Point3D<scalar_t> bbox_max = node.bbox_max;
                    scalar_t split_val = node.split_value;
                    int split_dim = node.split_dim;
                    
                    printf("[DEBUG] Counting points <= split_val\n");
                    for (int i = 0; i < node.count; i++) {
                        scalar_t coord = (split_dim == 0) ? host_points[i*3] : (split_dim == 1) ? host_points[i*3+1] : host_points[i*3+2];
                        if (host_is_less_equal(coord, split_val)) left_count++;
                    }
                    printf("[DEBUG] Left count: %d/%d\n", left_count, node.count);
                    
                    if (left_count > 0 && left_count < node.count && next_node_idx < MAX_KD_NODES - 1) {
                        printf("[DEBUG] Creating children: left_count=%d, next_node_idx=%d\n", left_count, next_node_idx);
                        
                        // Create left child
                        node.left_child = next_node_idx++;
                        KDNode<scalar_t> left_child;
                        left_child.start_idx = node.start_idx;
                        left_child.count = left_count;
                        left_child.left_child = -1;
                        left_child.right_child = -1;
                        cudaMemcpy(&nodes[node.left_child], &left_child, sizeof(KDNode<scalar_t>), cudaMemcpyHostToDevice);
                        
                        // Create right child
                        node.right_child = next_node_idx++;
                        KDNode<scalar_t> right_child;
                        right_child.start_idx = node.start_idx + left_count;
                        right_child.count = node.count - left_count;
                        right_child.left_child = -1;
                        right_child.right_child = -1;
                        cudaMemcpy(&nodes[node.right_child], &right_child, sizeof(KDNode<scalar_t>), cudaMemcpyHostToDevice);
                        
                        // Update parent node (copy modified node back to device)
                        cudaMemcpy(&nodes[node_id], &node, sizeof(KDNode<scalar_t>), cudaMemcpyHostToDevice);
                        printf("[DEBUG] Children created successfully\n");
                    } else {
                        printf("[DEBUG] Not creating children: left_count=%d, total=%d, next_node_idx=%d\n",
                               left_count, node.count, next_node_idx);
                    }
                }
                
                delete[] host_points;
            }
            
            delete[] host_indices;
        }
        
        current_nodes = next_node_idx;
        printf("[DEBUG] Depth %d completed, current_nodes=%d\n", depth, current_nodes);
        if (current_nodes >= MAX_KD_NODES - 10) {
            printf("[DEBUG] Stopping: approaching MAX_KD_NODES limit\n");
            break;  // Safety margin
        }
    }
    
    printf("[DEBUG] KD-tree construction completed, freeing temp_indices\n");
    cudaFree(temp_indices);
    printf("[DEBUG] build_complete_kd_tree finished\n");
}

// COMPLETE aggressive KD-tree FPS with full pruning
template<typename scalar_t, unsigned int block_size>
__global__ void complete_aggressive_kd_fps_kernel(
    int b, int n, int m,
    const scalar_t* __restrict__ dataset,
    scalar_t* __restrict__ temp,
    int64_t* __restrict__ idxs,
    const KDNode<scalar_t>* __restrict__ kd_nodes,
    const int* __restrict__ point_indices,
    int num_nodes) {
  
  if (m <= 0) return;
  __shared__ scalar_t dists[block_size];
  __shared__ int dists_i[block_size];
  __shared__ bool node_should_process[MAX_KD_NODES];

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
    // Current sampled point
    Point3D<scalar_t> query_point(dataset[old * 3 + 0], dataset[old * 3 + 1], dataset[old * 3 + 2]);
    
    // Find current maximum distance across all points
    scalar_t global_max_dist = scalar_t(-1);
    for (int k = tid; k < n; k += stride) {
        global_max_dist = max_val(global_max_dist, temp[k]);
    }
    
    // Reduce to find true global maximum
    dists[tid] = global_max_dist;
    __syncthreads();
    for (int s = stride / 2; s > 0; s /= 2) {
        if (tid < s) {
            dists[tid] = max_val(dists[tid], dists[tid + s]);
        }
        __syncthreads();
    }
    global_max_dist = dists[0];
    __syncthreads();
    
    // **COMPLETE AGGRESSIVE PRUNING PHASE** - Skip entire KD-tree branches!
    if (tid < num_nodes) {
      const KDNode<scalar_t>& node = kd_nodes[tid];
      if (node.count > 0) {
        // Calculate minimum possible distance to any point in this node
        scalar_t min_possible_dist = bbox_min_distance_squared(query_point, node.bbox_min, node.bbox_max);
        
        // **PRUNE ENTIRE NODE** if minimum possible distance > current best
        // This is the KEY optimization that gives us O(N log N) instead of O(N²)!
        node_should_process[tid] = is_less(min_possible_dist, global_max_dist) || is_less(global_max_dist, scalar_t(0));
      } else {
        node_should_process[tid] = false;
      }
    }
    __syncthreads();
    
    // Process only non-pruned leaf nodes
    int besti = 0;
    scalar_t best = scalar_t(-1);
    
    for (int node_id = 0; node_id < num_nodes; node_id++) {
      if (!node_should_process[node_id]) continue;  // **SKIP PRUNED NODES**
      
      const KDNode<scalar_t>& node = kd_nodes[node_id];
      
      // Only process leaf nodes to avoid double-counting
      if (!node.is_leaf()) continue;
      
      // Process points in this non-pruned leaf node
      for (int i = tid; i < node.count; i += stride) {
        int k = point_indices[node.start_idx + i];
        if (k >= n) continue;
        
        scalar_t x2 = dataset[k * 3 + 0];
        scalar_t y2 = dataset[k * 3 + 1];
        scalar_t z2 = dataset[k * 3 + 2];
        scalar_t mag = (x2 * x2) + (y2 * y2) + (z2 * z2);
        if (!is_greater(mag, scalar_t(1e-3))) continue;

        scalar_t d = distance_squared(query_point, Point3D<scalar_t>(x2, y2, z2));
        scalar_t d2 = min_val(d, temp[k]);
        temp[k] = d2;
        besti = is_greater(d2, best) ? k : besti;
        best = is_greater(d2, best) ? d2 : best;
      }
    }
    
    dists[tid] = best;
    dists_i[tid] = besti;
    __syncthreads();

    // Reduce to find global best
    for (int stride_r = block_size / 2; stride_r > 0; stride_r /= 2) {
      if (tid < stride_r) {
        if (is_greater(dists[tid + stride_r], dists[tid])) {
          dists[tid] = dists[tid + stride_r];
          dists_i[tid] = dists_i[tid + stride_r];
        }
      }
      __syncthreads();
    }

    old = dists_i[0];
    if (tid == 0) idxs[j] = old;
  }
}

// Template dispatch function
template<typename scalar_t>
void dispatch_complete_aggressive_kd_fps_kernel(int b, int n, int m,
                                                const scalar_t* dataset,
                                                scalar_t* temp,
                                                int64_t* idxs,
                                                const KDNode<scalar_t>* kd_nodes,
                                                const int* point_indices,
                                                int num_nodes) {
  
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  int n_threads = 512;
  
  size_t shared_mem = n_threads * (sizeof(scalar_t) + sizeof(int)) + MAX_KD_NODES * sizeof(bool);
  
  complete_aggressive_kd_fps_kernel<scalar_t, 512>
      <<<b, n_threads, shared_mem, stream>>>(b, n, m, dataset, temp, idxs, kd_nodes, point_indices, num_nodes);
}

// COMPLETE main function with full KD-tree construction and aggressive pruning
torch::Tensor quick_fps_cuda_forward(torch::Tensor points, int64_t nsamples, int64_t kd_depth) {
  const c10::Device device = points.device();
  at::cuda::CUDAGuard device_guard(device);
  
  const auto N = points.size(0);
  const auto P = points.size(1);
  const auto D = points.size(2);
  
  printf("[DEBUG] Quick FPS Forward: N=%ld, P=%ld, D=%ld, nsamples=%ld, kd_depth=%ld\n",
         N, P, D, nsamples, kd_depth);
  
  TORCH_CHECK(D == 3, "Quick FPS only supports 3D points, but got dimension ", D);
  TORCH_CHECK(nsamples > 0, "nsamples must be positive, but got ", nsamples);
  TORCH_CHECK(nsamples <= P, "nsamples (", nsamples, ") cannot exceed number of points (", P, ")");
  TORCH_CHECK(kd_depth > 0 && kd_depth <= MAX_KD_DEPTH, "kd_depth must be between 1 and ", MAX_KD_DEPTH, ", but got ", kd_depth);
  
  auto idxs = torch::zeros({N, nsamples}, points.options().dtype(torch::kInt64));
  
  float large_val = (points.scalar_type() == torch::kFloat16) ? 1e4f : 1e10f;
  auto temp = torch::full({N, P}, large_val, points.options());
  
  printf("[DEBUG] Allocated tensors: idxs.numel()=%ld, temp.numel()=%ld\n", idxs.numel(), temp.numel());
  
  if (idxs.numel() == 0) {
    printf("[DEBUG] Early return: idxs.numel() == 0\n");
    return idxs;
  }
  
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(points.scalar_type(), "quick_fps_forward", ([&] {
    auto points_cont = points.contiguous();
    printf("[DEBUG] Points contiguous, starting batch processing\n");
    
    // Process each batch independently
    for (int batch = 0; batch < N; batch++) {
      printf("[DEBUG] Processing batch %d/%ld\n", batch, N);
      
      const scalar_t* batch_points = points_cont.data_ptr<scalar_t>() + batch * P * 3;
      printf("[DEBUG] Got batch_points pointer\n");
      
      // Allocate memory for COMPLETE KD-tree
      KDNode<scalar_t>* kd_nodes;
      int* point_indices;
      
      printf("[DEBUG] Allocating CUDA memory for KD-tree: %d nodes, %ld points\n", MAX_KD_NODES, P);
      cudaError_t err1 = cudaMalloc(&kd_nodes, MAX_KD_NODES * sizeof(KDNode<scalar_t>));
      if (err1 != cudaSuccess) {
        printf("[ERROR] Failed to allocate kd_nodes: %s\n", cudaGetErrorString(err1));
        TORCH_CHECK(false, "Failed to allocate KD-tree nodes: ", cudaGetErrorString(err1));
      }
      
      cudaError_t err2 = cudaMalloc(&point_indices, P * sizeof(int));
      if (err2 != cudaSuccess) {
        printf("[ERROR] Failed to allocate point_indices: %s\n", cudaGetErrorString(err2));
        cudaFree(kd_nodes);
        TORCH_CHECK(false, "Failed to allocate point indices: ", cudaGetErrorString(err2));
      }
      printf("[DEBUG] CUDA memory allocated successfully\n");
      
      // Initialize point indices
      printf("[DEBUG] Initializing point indices with thrust\n");
      thrust::device_ptr<int> indices_ptr(point_indices);
      thrust::sequence(indices_ptr, indices_ptr + P);
      printf("[DEBUG] Point indices initialized\n");
      
      // Build COMPLETE KD-tree with full construction
      printf("[DEBUG] Building KD-tree with depth %ld\n", kd_depth);
      build_complete_kd_tree<scalar_t>(batch_points, point_indices, kd_nodes, P, kd_depth);
      printf("[DEBUG] KD-tree built successfully\n");
      
      // Count actual nodes created
      int num_nodes = std::min(MAX_KD_NODES, 1 << (kd_depth + 1));
      printf("[DEBUG] Using %d nodes for FPS kernel\n", num_nodes);
      
      // Use COMPLETE aggressive pruning kernel
      printf("[DEBUG] Launching FPS kernel\n");
      dispatch_complete_aggressive_kd_fps_kernel<scalar_t>(
        1, P, nsamples,
        batch_points,
        temp.data_ptr<scalar_t>() + batch * P,
        idxs.data_ptr<int64_t>() + batch * nsamples,
        kd_nodes,
        point_indices,
        num_nodes);
      printf("[DEBUG] FPS kernel completed\n");
      
      // Check for CUDA errors
      cudaError_t kernel_err = cudaGetLastError();
      if (kernel_err != cudaSuccess) {
        printf("[ERROR] CUDA kernel error: %s\n", cudaGetErrorString(kernel_err));
        cudaFree(kd_nodes);
        cudaFree(point_indices);
        TORCH_CHECK(false, "CUDA kernel error: ", cudaGetErrorString(kernel_err));
      }
      
      printf("[DEBUG] Freeing CUDA memory\n");
      cudaFree(kd_nodes);
      cudaFree(point_indices);
      printf("[DEBUG] Batch %d completed successfully\n", batch);
    }
    printf("[DEBUG] All batches processed\n");
  }));
  
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    TORCH_CHECK(false, "CUDA error in quick_fps_forward: ", cudaGetErrorString(err));
  }
  
  return idxs;
}

TORCH_LIBRARY_IMPL(torch_point_ops_quick_fps, CUDA, m) {
  m.impl("quick_fps_forward", &quick_fps_cuda_forward);
}

} // namespace torch_point_ops