#include <Python.h>
#include <torch/all.h>
#include <torch/library.h>
#include <vector>
#include <tuple>

namespace torch_point_ops {

// Forward declarations of the functions implemented in chamfer.cu
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
chamfer_cuda_forward(torch::Tensor xyz1, torch::Tensor xyz2);

std::tuple<torch::Tensor, torch::Tensor>
chamfer_cuda_backward(torch::Tensor xyz1,
                      torch::Tensor xyz2,
                      torch::Tensor idx1,
                      torch::Tensor idx2,
                      torch::Tensor grad_dist1,
                      torch::Tensor grad_dist2);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
chamfer_cpu_forward(torch::Tensor xyz1, torch::Tensor xyz2) {
  TORCH_CHECK(false, "chamfer_forward is not implemented for CPU");
}

std::tuple<torch::Tensor, torch::Tensor>
chamfer_cpu_backward(torch::Tensor xyz1, torch::Tensor xyz2, torch::Tensor idx1,
                     torch::Tensor idx2, torch::Tensor grad_dist1,
                     torch::Tensor grad_dist2) {
  TORCH_CHECK(false, "chamfer_backward is not implemented for CPU");
}

// Defines the operators
TORCH_LIBRARY(torch_point_ops_chamfer, m) {
  m.def("chamfer_forward(Tensor xyz1, Tensor xyz2) -> (Tensor, Tensor, Tensor, "
        "Tensor)");
  m.def("chamfer_backward(Tensor xyz1, Tensor xyz2, Tensor idx1, Tensor idx2, "
        "Tensor grad_dist1, Tensor grad_dist2) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(torch_point_ops_chamfer, CPU, m) {
  m.impl("chamfer_forward", &chamfer_cpu_forward);
  m.impl("chamfer_backward", &chamfer_cpu_backward);
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