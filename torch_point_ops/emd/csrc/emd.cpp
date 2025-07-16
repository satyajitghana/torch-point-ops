#include <Python.h>
#include <torch/all.h>
#include <torch/library.h>
#include <vector>
#include <tuple>

namespace torch_point_ops {

// Forward declarations of the functions implemented in emd.cu
std::tuple<torch::Tensor, torch::Tensor>
emd_cuda_forward(torch::Tensor xyz1, torch::Tensor xyz2);

std::tuple<torch::Tensor, torch::Tensor>
emd_cuda_backward(torch::Tensor grad_cost,
                  torch::Tensor xyz1,
                  torch::Tensor xyz2,
                  torch::Tensor match);

std::tuple<torch::Tensor, torch::Tensor>
emd_cpu_forward(torch::Tensor xyz1, torch::Tensor xyz2) {
  TORCH_CHECK(false, "emd_forward is not implemented for CPU");
}

std::tuple<torch::Tensor, torch::Tensor>
emd_cpu_backward(torch::Tensor grad_cost, torch::Tensor xyz1, 
                 torch::Tensor xyz2, torch::Tensor match) {
  TORCH_CHECK(false, "emd_backward is not implemented for CPU");
}

// Defines the operators
TORCH_LIBRARY(torch_point_ops_emd, m) {
    m.def("emd_forward(Tensor xyz1, Tensor xyz2) -> (Tensor, Tensor)");
    m.def("emd_backward(Tensor grad_cost, Tensor xyz1, Tensor xyz2, Tensor match) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(torch_point_ops_emd, CPU, m) {
    m.impl("emd_forward", &emd_cpu_forward);
    m.impl("emd_backward", &emd_cpu_backward);
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