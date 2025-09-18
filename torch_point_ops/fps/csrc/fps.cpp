#include <Python.h>
#include <torch/all.h>
#include <torch/library.h>
#include <vector>
#include <tuple>
#include <queue>
#include <algorithm>

namespace torch_point_ops {

// Forward declarations of CUDA functions
torch::Tensor fps_cuda_forward(torch::Tensor points, int64_t nsamples);
torch::Tensor gather_points_cuda_forward(torch::Tensor points, torch::Tensor idx);
torch::Tensor gather_points_cuda_backward(torch::Tensor grad_out, torch::Tensor idx, int64_t n);
torch::Tensor quick_fps_cuda_forward(torch::Tensor points, int64_t nsamples, int64_t kd_depth);

// CPU implementation of furthest point sampling
torch::Tensor fps_cpu_forward(torch::Tensor points, int64_t nsamples) {
  const auto N = points.size(0);
  const auto P = points.size(1);
  const auto D = points.size(2);
  
  TORCH_CHECK(D == 3, "FPS only supports 3D points, but got dimension ", D);
  TORCH_CHECK(nsamples > 0, "nsamples must be positive, but got ", nsamples);
  TORCH_CHECK(nsamples <= P, "nsamples (", nsamples, ") cannot exceed number of points (", P, ")");
  
  auto idxs = torch::zeros({N, nsamples}, torch::kInt64);
  
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(points.scalar_type(), "fps_cpu_forward", [&] {
    auto points_a = points.accessor<scalar_t, 3>();
    auto idxs_a = idxs.accessor<int64_t, 2>();
    
    for (int n = 0; n < N; ++n) {
      std::vector<scalar_t> min_dists(P, std::numeric_limits<scalar_t>::max());
      std::vector<bool> selected(P, false);
      
      // Select first point (index 0)
      int64_t current_idx = 0;
      idxs_a[n][0] = current_idx;
      selected[current_idx] = true;
      
      // FPS main loop
      for (int64_t i = 1; i < nsamples; ++i) {
        // Update distances to the newly selected point
        scalar_t curr_x = points_a[n][current_idx][0];
        scalar_t curr_y = points_a[n][current_idx][1];
        scalar_t curr_z = points_a[n][current_idx][2];
        
        for (int64_t j = 0; j < P; ++j) {
          if (selected[j]) continue;
          
          scalar_t x = points_a[n][j][0];
          scalar_t y = points_a[n][j][1];
          scalar_t z = points_a[n][j][2];
          
          // Skip degenerate points
          scalar_t mag = x * x + y * y + z * z;
          if (mag <= scalar_t(1e-3)) continue;
          
          // Compute squared distance
          scalar_t dx = x - curr_x;
          scalar_t dy = y - curr_y;
          scalar_t dz = z - curr_z;
          scalar_t dist = dx * dx + dy * dy + dz * dz;
          
          // Update minimum distance to any selected point
          min_dists[j] = std::min(min_dists[j], dist);
        }
        
        // Find the point with maximum distance to closest selected point
        scalar_t max_dist = -1;
        int64_t best_idx = 0;
        for (int64_t j = 0; j < P; ++j) {
          if (!selected[j] && min_dists[j] > max_dist) {
            max_dist = min_dists[j];
            best_idx = j;
          }
        }
        
        current_idx = best_idx;
        idxs_a[n][i] = current_idx;
        selected[current_idx] = true;
      }
    }
  });
  
  return idxs;
}

// CPU implementation of gather points
torch::Tensor gather_points_cpu_forward(torch::Tensor points, torch::Tensor idx) {
  const auto N = points.size(0);
  const auto C = points.size(1);
  // const auto P = points.size(2);  // Unused variable
  const auto M = idx.size(1);
  
  auto output = torch::zeros({N, C, M}, points.options());
  
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(points.scalar_type(), "gather_points_cpu_forward", [&] {
    auto points_a = points.accessor<scalar_t, 3>();
    auto idx_a = idx.accessor<int64_t, 2>();
    auto output_a = output.accessor<scalar_t, 3>();
    
    for (int n = 0; n < N; ++n) {
      for (int c = 0; c < C; ++c) {
        for (int m = 0; m < M; ++m) {
          int64_t point_idx = idx_a[n][m];
          output_a[n][c][m] = points_a[n][c][point_idx];
        }
      }
    }
  });
  
  return output;
}

// CPU implementation of gather points backward
torch::Tensor gather_points_cpu_backward(torch::Tensor grad_out, torch::Tensor idx, int64_t n) {
  const auto N = grad_out.size(0);
  const auto C = grad_out.size(1);
  const auto M = grad_out.size(2);
  
  auto grad_points = torch::zeros({N, C, n}, grad_out.options());
  
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_out.scalar_type(), "gather_points_cpu_backward", [&] {
    auto grad_out_a = grad_out.accessor<scalar_t, 3>();
    auto idx_a = idx.accessor<int64_t, 2>();
    auto grad_points_a = grad_points.accessor<scalar_t, 3>();
    
    for (int batch = 0; batch < N; ++batch) {
      for (int c = 0; c < C; ++c) {
        for (int m = 0; m < M; ++m) {
          int64_t point_idx = idx_a[batch][m];
          grad_points_a[batch][c][point_idx] += grad_out_a[batch][c][m];
        }
      }
    }
  });
  
  return grad_points;
}

// CPU implementation of quick FPS - simplified version that just calls regular FPS
torch::Tensor quick_fps_cpu_forward(torch::Tensor points, int64_t nsamples, int64_t kd_depth) {
  const auto P = points.size(1);
  const auto D = points.size(2);
  
  TORCH_CHECK(D == 3, "Quick FPS only supports 3D points, but got dimension ", D);
  TORCH_CHECK(nsamples > 0, "nsamples must be positive, but got ", nsamples);
  TORCH_CHECK(nsamples <= P, "nsamples (", nsamples, ") cannot exceed number of points (", P, ")");
  TORCH_CHECK(kd_depth > 0 && kd_depth <= 10, "kd_depth must be between 1 and 10, but got ", kd_depth);
  
  // For CPU, just use regular FPS (kd_depth parameter is ignored on CPU)
  return fps_cpu_forward(points, nsamples);
}

// Define the operators
TORCH_LIBRARY(torch_point_ops_fps, m) {
  m.def("fps_forward(Tensor points, int nsamples) -> Tensor");
  m.def("gather_points_forward(Tensor points, Tensor idx) -> Tensor");
  m.def("gather_points_backward(Tensor grad_out, Tensor idx, int n) -> Tensor");
}

// Define the quick FPS operators
TORCH_LIBRARY(torch_point_ops_quick_fps, m) {
  m.def("quick_fps_forward(Tensor points, int nsamples, int kd_depth=4) -> Tensor");
}

// CPU implementations
TORCH_LIBRARY_IMPL(torch_point_ops_fps, CPU, m) {
  m.impl("fps_forward", &fps_cpu_forward);
  m.impl("gather_points_forward", &gather_points_cpu_forward);
  m.impl("gather_points_backward", &gather_points_cpu_backward);
}

// Quick FPS CPU implementation
TORCH_LIBRARY_IMPL(torch_point_ops_quick_fps, CPU, m) {
  m.impl("quick_fps_forward", &quick_fps_cpu_forward);
}

} // namespace torch_point_ops

extern "C" {
/* Creates a dummy empty _C module that can be imported from Python.
   The import from Python will load the .so consisting of this file
   in this extension, so that the TORCH_LIBRARY static initializers
   above are run. */
PyObject *PyInit__C(void) {
  static struct PyModuleDef module_def = {
      PyModuleDef_HEAD_INIT,
      "_C",   /* name of module */
      NULL,   /* module documentation, may be NULL */
      -1,     /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
      NULL,   /* methods */
  };
  return PyModule_Create(&module_def);
}
}
