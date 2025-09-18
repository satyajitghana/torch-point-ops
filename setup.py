import glob
import os
import sys
from pathlib import Path

import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# The name of the project's root package.
library_name = "torch_point_ops"

# Get absolute directory for include paths (ninja compatibility)
this_dir = os.path.dirname(os.path.abspath(__file__))


class NinjaBuildExtension(BuildExtension):
    """
    Ninja build extension for faster compilation.
    Automatically determines the optimal number of jobs based on system resources.
    """
    def __init__(self, *args, **kwargs) -> None:
        # do not override env MAX_JOBS if already exists
        if not os.environ.get("MAX_JOBS"):
            try:
                import psutil
                
                # calculate the maximum allowed NUM_JOBS based on cores
                max_num_jobs_cores = max(1, os.cpu_count() // 2)
                
                # calculate the maximum allowed NUM_JOBS based on free memory
                free_memory_gb = psutil.virtual_memory().available / (1024 ** 3)  # free memory in GB
                max_num_jobs_memory = int(free_memory_gb / 4)  # each JOB peak memory cost is ~3-4GB for point ops
                
                # pick lower value of jobs based on cores vs memory metric to minimize oom and swap usage during compilation
                max_jobs = max(1, min(max_num_jobs_cores, max_num_jobs_memory))
                os.environ["MAX_JOBS"] = str(max_jobs)
                print(f"Setting MAX_JOBS={max_jobs} (cores: {max_num_jobs_cores}, memory: {max_num_jobs_memory})")
            except ImportError:
                # Fallback if psutil is not available
                max_jobs = max(1, os.cpu_count() // 2)
                os.environ["MAX_JOBS"] = str(max_jobs)
                print(f"Setting MAX_JOBS={max_jobs} (psutil not available, using CPU count)")
        
        super().__init__(*args, **kwargs)


def append_nvcc_threads(nvcc_extra_args):
    """Add threading support to nvcc args"""
    nvcc_threads = os.getenv("NVCC_THREADS") or "4"
    return nvcc_extra_args + ["--threads", nvcc_threads]


def create_extension(module_name):
    """
    Creates a CUDA extension for the given module with optimized compilation flags.

    Args:
        module_name (str): Name of the module (e.g., 'chamfer', 'emd', 'knn', 'fps')

    Returns:
        CUDAExtension: Configured extension for the module
    """
    use_cuda = torch.cuda.is_available() or os.getenv("FORCE_CUDA", "0") == "1"

    # Get C++ sources (use relative paths for setup.py)
    cpp_sources = glob.glob(os.path.join(library_name, module_name, "csrc", "*.cpp"))

    # Get CUDA sources (use relative paths for setup.py)
    cuda_sources = glob.glob(os.path.join(library_name, module_name, "csrc", "cuda", "*.cu"))

    # Combine sources
    sources = cpp_sources
    if use_cuda:
        sources.extend(cuda_sources)

    # Compilation flags optimized for half precision and AMP support
    cxx_flags = [
        "-O3",
        "-std=c++17",
    ]
    
    nvcc_flags = [
        "-O3",
        "-std=c++17",
        # Enable half precision support
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        # Enable bfloat16 support for AMP
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    ]

    # Build include directories (use absolute paths for ninja compatibility)
    include_dirs = []
    
    # Add module-specific include directories if they exist
    module_include_dir = os.path.join(this_dir, library_name, module_name, "csrc")
    if os.path.exists(module_include_dir):
        include_dirs.append(module_include_dir)
    
    # Add CUDA include directories if they exist
    cuda_include_dir = os.path.join(this_dir, library_name, module_name, "csrc", "cuda")
    if os.path.exists(cuda_include_dir):
        include_dirs.append(cuda_include_dir)

    extra_compile_args = {
        "cxx": cxx_flags,
        "nvcc": append_nvcc_threads(nvcc_flags) if use_cuda else [],
    }

    return CUDAExtension(
        name=f"{library_name}.{module_name}._C",
        sources=sources,
        extra_compile_args=extra_compile_args,
        include_dirs=include_dirs,
    )


def get_extensions():
    """
    Builds the C++/CUDA extensions for all modules.
    """
    # List of modules to build extensions for
    modules = ["chamfer", "emd", "knn", "fps"]

    print(f"\n\nBuilding extensions for torch-point-ops...")
    print(f"torch.__version__ = {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Building modules: {modules}\n")

    # Create extensions for all modules
    extensions = []
    for module in modules:
        try:
            ext = create_extension(module)
            extensions.append(ext)
            print(f"✓ Created extension for {module}")
        except Exception as e:
            print(f"✗ Failed to create extension for {module}: {e}")
            # Continue with other modules instead of failing completely
            continue

    print(f"\nSuccessfully created {len(extensions)} extensions\n")
    return extensions


setup(
    ext_modules=get_extensions(),
    cmdclass={"build_ext": NinjaBuildExtension},
    setup_requires=[
        "ninja",
        "psutil",  # For memory-based job calculation (optional)
        "numpy"
    ],
)
