import glob
import os

import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# The name of the project's root package.
library_name = "torch_point_ops"


def get_extensions():
    """
    Builds the C++/CUDA extensions.
    Paths are relative to the project root and are specified manually
    to avoid issues with build environments.
    """
    use_cuda = torch.cuda.is_available() or os.getenv("FORCE_CUDA", "0") == "1"

    extensions = []
    
    # Chamfer extension
    chamfer_sources = glob.glob(f"{library_name}/chamfer/csrc/*.cpp")
    chamfer_cuda_sources = glob.glob(f"{library_name}/chamfer/csrc/cuda/*.cu")

    if use_cuda:
        chamfer_sources.extend(chamfer_cuda_sources)

    extra_compile_args = {"cxx": ["-O2"], "nvcc": ["-O2"]}

    extensions.append(
        CUDAExtension(
            name=f"{library_name}.chamfer._C",
            sources=chamfer_sources,
            extra_compile_args=extra_compile_args,
        )
    )

    # EMD extension
    emd_sources = glob.glob(f"{library_name}/emd/csrc/*.cpp")
    emd_cuda_sources = glob.glob(f"{library_name}/emd/csrc/cuda/*.cu")

    if use_cuda:
        emd_sources.extend(emd_cuda_sources)
    
    extensions.append(
        CUDAExtension(
            name=f"{library_name}.emd._C",
            sources=emd_sources,
            extra_compile_args=extra_compile_args,
        )
    )

    return extensions


setup(
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtension},
) 