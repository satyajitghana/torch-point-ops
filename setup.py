import glob
import os

import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# The name of the project's root package.
library_name = "torch_point_ops"


def create_extension(module_name):
    """
    Creates a CUDA extension for the given module.

    Args:
        module_name (str): Name of the module (e.g., 'chamfer', 'emd', 'knn')

    Returns:
        CUDAExtension: Configured extension for the module
    """
    use_cuda = torch.cuda.is_available() or os.getenv("FORCE_CUDA", "0") == "1"

    # Get C++ sources
    cpp_sources = glob.glob(f"{library_name}/{module_name}/csrc/*.cpp")

    # Get CUDA sources
    cuda_sources = glob.glob(f"{library_name}/{module_name}/csrc/cuda/*.cu")

    # Combine sources
    sources = cpp_sources
    if use_cuda:
        sources.extend(cuda_sources)

    # Compilation flags
    extra_compile_args = {"cxx": ["-O2"], "nvcc": ["-O2"]}

    return CUDAExtension(
        name=f"{library_name}.{module_name}._C",
        sources=sources,
        extra_compile_args=extra_compile_args,
    )


def get_extensions():
    """
    Builds the C++/CUDA extensions for all modules.
    """
    # List of modules to build extensions for
    modules = ["chamfer", "emd", "knn"]

    # Create extensions for all modules
    extensions = [create_extension(module) for module in modules]

    return extensions


setup(
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtension},
)
