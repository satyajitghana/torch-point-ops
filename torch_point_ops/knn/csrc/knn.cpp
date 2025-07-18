#include <Python.h>
#include <torch/all.h>
#include <torch/library.h>
#include <vector>
#include <tuple>
#include <queue>

namespace torch_point_ops {

// Forward declarations of CUDA functions
std::tuple<torch::Tensor, torch::Tensor> knn_cuda_forward(
    torch::Tensor p1,
    torch::Tensor p2,
    torch::Tensor lengths1,
    torch::Tensor lengths2,
    int64_t norm,
    int64_t K,
    int64_t version);

std::tuple<torch::Tensor, torch::Tensor> knn_cuda_backward(
    torch::Tensor p1,
    torch::Tensor p2,
    torch::Tensor lengths1,
    torch::Tensor lengths2,
    torch::Tensor idxs,
    int64_t norm,
    torch::Tensor grad_dists);

// CPU forward implementation
std::tuple<torch::Tensor, torch::Tensor> knn_cpu_forward(
    torch::Tensor p1,
    torch::Tensor p2,
    torch::Tensor lengths1,
    torch::Tensor lengths2,
    int64_t norm,
    int64_t K,
    int64_t version) {
  
  const int N = p1.size(0);
  const int P1 = p1.size(1);
  const int D = p1.size(2);

  auto long_opts = lengths1.options().dtype(torch::kInt64);
  torch::Tensor idxs = torch::full({N, P1, K}, -1, long_opts);
  torch::Tensor dists = torch::zeros({N, P1, K}, p1.options());

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(p1.scalar_type(), "knn_cpu_forward", [&] {
    auto p1_a = p1.accessor<scalar_t, 3>();
    auto p2_a = p2.accessor<scalar_t, 3>();
    auto lengths1_a = lengths1.accessor<int64_t, 1>();
    auto lengths2_a = lengths2.accessor<int64_t, 1>();
    auto idxs_a = idxs.accessor<int64_t, 3>();
    auto dists_a = dists.accessor<scalar_t, 3>();

    for (int n = 0; n < N; ++n) {
      const int64_t length1 = lengths1_a[n];
      const int64_t length2 = lengths2_a[n];
      
      for (int64_t i1 = 0; i1 < length1; ++i1) {
        // Use a priority queue to store (distance, index) tuples
        std::priority_queue<std::tuple<scalar_t, int>> q;
        
        for (int64_t i2 = 0; i2 < length2; ++i2) {
          scalar_t dist = 0;
          for (int d = 0; d < D; ++d) {
            scalar_t diff = p1_a[n][i1][d] - p2_a[n][i2][d];
            if (norm == 1) {
              dist += std::abs(diff);
            } else { // norm is 2 (default)
              dist += diff * diff;
            }
          }
          
          int size = static_cast<int>(q.size());
          if (size < K || dist < std::get<0>(q.top())) {
            q.emplace(dist, i2);
            if (size >= K) {
              q.pop();
            }
          }
        }
        
        // Extract results from priority queue
        std::vector<std::tuple<scalar_t, int>> results;
        while (!q.empty()) {
          results.push_back(q.top());
          q.pop();
        }
        
        // Store results in reverse order (smallest distance first)
        for (size_t k = 0; k < results.size(); ++k) {
          size_t idx = results.size() - 1 - k;
          dists_a[n][i1][k] = std::get<0>(results[idx]);
          idxs_a[n][i1][k] = std::get<1>(results[idx]);
        }
      }
    }
  });

  return std::make_tuple(idxs, dists);
}

// CPU backward implementation
std::tuple<torch::Tensor, torch::Tensor> knn_cpu_backward(
    torch::Tensor p1,
    torch::Tensor p2,
    torch::Tensor lengths1,
    torch::Tensor lengths2,
    torch::Tensor idxs,
    int64_t norm,
    torch::Tensor grad_dists) {
  
  const int N = p1.size(0);
  const int D = p1.size(2);
  const int K = idxs.size(2);

  torch::Tensor grad_p1 = torch::zeros_like(p1);
  torch::Tensor grad_p2 = torch::zeros_like(p2);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(p1.scalar_type(), "knn_cpu_backward", [&] {
    auto p1_a = p1.accessor<scalar_t, 3>();
    auto p2_a = p2.accessor<scalar_t, 3>();
    auto lengths1_a = lengths1.accessor<int64_t, 1>();
    auto lengths2_a = lengths2.accessor<int64_t, 1>();
    auto idxs_a = idxs.accessor<int64_t, 3>();
    auto grad_dists_a = grad_dists.accessor<scalar_t, 3>();
    auto grad_p1_a = grad_p1.accessor<scalar_t, 3>();
    auto grad_p2_a = grad_p2.accessor<scalar_t, 3>();

    for (int n = 0; n < N; ++n) {
      const int64_t length1 = lengths1_a[n];
      int64_t length2 = lengths2_a[n];
      length2 = (length2 < K) ? length2 : K;
      
      for (int64_t i1 = 0; i1 < length1; ++i1) {
        for (int64_t k = 0; k < length2; ++k) {
          const int64_t i2 = idxs_a[n][i1][k];
          // If the index is the pad value of -1 then ignore it
          if (i2 == -1) {
            continue;
          }
          
          for (int64_t d = 0; d < D; ++d) {
            scalar_t p1_val = p1_a[n][i1][d];
            scalar_t p2_val = p2_a[n][i2][d];
            scalar_t grad_dist_val = grad_dists_a[n][i1][k];
            scalar_t diff = 0;

            if (norm == 1) {
              scalar_t sign = (p1_val > p2_val) ? 1.0 : -1.0;
              diff = grad_dist_val * sign;
            } else { // norm is 2 (default)
              diff = 2.0 * grad_dist_val * (p1_val - p2_val);
            }
            grad_p1_a[n][i1][d] += diff;
            grad_p2_a[n][i2][d] -= diff;
          }
        }
      }
    }
  });

  return std::make_tuple(grad_p1, grad_p2);
}

// Define the operators
TORCH_LIBRARY(torch_point_ops_knn, m) {
  m.def("knn_forward(Tensor p1, Tensor p2, Tensor lengths1, Tensor lengths2, "
        "int norm, int K, int version) -> (Tensor, Tensor)");
  m.def("knn_backward(Tensor p1, Tensor p2, Tensor lengths1, Tensor lengths2, "
        "Tensor idxs, int norm, Tensor grad_dists) -> (Tensor, Tensor)");
}

// CPU implementations
TORCH_LIBRARY_IMPL(torch_point_ops_knn, CPU, m) {
  m.impl("knn_forward", &knn_cpu_forward);
  m.impl("knn_backward", &knn_cpu_backward);
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