#!/usr/bin/env python3
"""
CUDA Memory Hog - Gradually consumes GPU memory to cause OOM for other processes
"""

import torch
import time
import gc
import argparse
from typing import List


def get_gpu_memory_info(device_id: int = 0):
    """Get current GPU memory usage information"""
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{device_id}")
        total_memory = torch.cuda.get_device_properties(device).total_memory
        allocated_memory = torch.cuda.memory_allocated(device)
        cached_memory = torch.cuda.memory_reserved(device)

        return {
            "total": total_memory / (1024**3),  # Convert to GB
            "allocated": allocated_memory / (1024**3),
            "cached": cached_memory / (1024**3),
            "pytorch_allocated": allocated_memory / (1024**3),
            "pytorch_cached": cached_memory / (1024**3),
        }
    return None


def allocate_memory_chunk(size_mb: float, device_id: int = 0) -> torch.Tensor:
    """Allocate a chunk of GPU memory"""
    device = torch.device(f"cuda:{device_id}")
    # Calculate number of float32 elements for the given size in MB
    elements = int((size_mb * 1024 * 1024) // 4)  # 4 bytes per float32
    if elements < 1:
        elements = 1  # Minimum allocation
    tensor = torch.ones(elements, dtype=torch.float32, device=device)
    return tensor


def memory_hog(
    chunk_size_mb: float = 0.1,
    delay_seconds: float = 0.5,
    max_attempts: int = 1000,
    device_id: int = 0,
    verbose: bool = True,
    aggressive: bool = True,
):
    """
    Gradually consume GPU memory using adaptive allocation strategy

    Args:
        chunk_size_mb: Starting size of each memory chunk to allocate (MB)
        delay_seconds: Delay between allocations (seconds)
        max_attempts: Maximum number of allocation attempts before giving up
        device_id: CUDA device ID to target
        verbose: Print memory status updates
        aggressive: Try ultra-small allocations and cache clearing (default: True)
    """

    if not torch.cuda.is_available():
        print("CUDA is not available!")
        return

    if device_id >= torch.cuda.device_count():
        print(
            f"Device {device_id} not available. Available devices: {torch.cuda.device_count()}"
        )
        return

    print(f"Starting CUDA memory allocation on device {device_id}")
    print(f"Starting chunk size: {chunk_size_mb}MB, Delay: {delay_seconds}s")
    print("Strategy: Adaptive allocation until GPU memory is exhausted")
    print("Press Ctrl+C to stop\n")

    allocated_tensors: List[torch.Tensor] = []
    current_chunk_size = chunk_size_mb
    consecutive_failures = 0
    successful_allocations = 0

    try:
        for attempt in range(max_attempts):
            # Get current memory status
            mem_info = get_gpu_memory_info(device_id)
            if mem_info is None:
                break

            if verbose and (attempt % 10 == 0 or consecutive_failures > 0):
                print(
                    f"Attempt {attempt+1}: PyTorch allocated: {mem_info['allocated']:.3f}GB, "
                    f"cached: {mem_info['cached']:.3f}GB"
                )

            # Try to allocate memory with current chunk size
            allocation_successful = False

            # Try progressively smaller chunks
            test_sizes = [current_chunk_size]
            if current_chunk_size > 0.1:
                test_sizes.extend(
                    [current_chunk_size / 2, current_chunk_size / 4, 0.1, 0.05, 0.01]
                )
            elif current_chunk_size > 0.01:
                test_sizes.extend([current_chunk_size / 2, 0.01, 0.005])
            else:
                test_sizes.extend([0.005, 0.001])

            for test_size in test_sizes:
                if test_size < 0.001:
                    break

                try:
                    if verbose and consecutive_failures > 0:
                        print(f"  Trying {test_size:.3f}MB...")

                    tensor = allocate_memory_chunk(test_size, device_id)
                    allocated_tensors.append(tensor)

                    # Force GPU to actually allocate the memory
                    torch.cuda.synchronize()

                    # Update chunk size for next iteration (slightly increase if successful)
                    current_chunk_size = min(test_size * 1.1, chunk_size_mb)
                    consecutive_failures = 0
                    successful_allocations += 1
                    allocation_successful = True

                    if verbose:
                        updated_mem_info = get_gpu_memory_info(device_id)
                        print(
                            f"✓ Allocated {test_size:.3f}MB (total chunks: {len(allocated_tensors)}, "
                            f"PyTorch allocated: {updated_mem_info['allocated']:.3f}GB)"
                        )
                    break

                except torch.cuda.OutOfMemoryError:
                    if verbose and consecutive_failures == 0:
                        print(f"✗ Failed to allocate {test_size:.3f}MB")
                    continue
                except Exception as e:
                    print(f"Unexpected error with {test_size:.3f}MB: {e}")
                    continue

            if not allocation_successful:
                consecutive_failures += 1
                current_chunk_size *= 0.5  # Reduce chunk size for next attempt

                # Try aggressive mode if regular allocation fails
                if consecutive_failures >= 3 and aggressive:
                    if verbose:
                        print(
                            "Entering aggressive mode - clearing cache and trying ultra-small allocations..."
                        )

                    # Clear any cached memory
                    torch.cuda.empty_cache()
                    gc.collect()

                    # Try ultra-small allocations
                    ultra_small_sizes = [
                        0.0001,
                        0.00005,
                        0.00001,
                    ]  # 0.1KB, 0.05KB, 0.01KB

                    for ultra_size in ultra_small_sizes:
                        try:
                            if verbose:
                                print(
                                    f"  Ultra-aggressive: trying {ultra_size*1000:.2f}KB..."
                                )

                            # Allocate minimum possible - just a few bytes
                            elements = max(1, int((ultra_size * 1024 * 1024) // 4))
                            device = torch.device(f"cuda:{device_id}")
                            tensor = torch.ones(
                                elements, dtype=torch.float32, device=device
                            )
                            allocated_tensors.append(tensor)
                            torch.cuda.synchronize()

                            consecutive_failures = 0
                            allocation_successful = True

                            if verbose:
                                updated_mem_info = get_gpu_memory_info(device_id)
                                print(
                                    f"✓ Ultra-small allocation successful: {ultra_size*1000:.2f}KB "
                                    f"(chunks: {len(allocated_tensors)})"
                                )
                            break

                        except torch.cuda.OutOfMemoryError:
                            continue
                        except Exception as e:
                            if verbose:
                                print(f"Error with ultra-small allocation: {e}")
                            continue

                if consecutive_failures >= 5:
                    print(f"\nGPU memory appears to be completely exhausted!")
                    print(
                        f"Could not allocate even the smallest possible amount after {consecutive_failures} attempts"
                    )
                    if len(allocated_tensors) > 0:
                        print(
                            f"✓ However, successfully allocated {len(allocated_tensors)} chunks before exhaustion"
                        )
                    break

                if verbose and not allocation_successful:
                    print(
                        f"Reducing chunk size to {current_chunk_size:.4f}MB for next attempt"
                    )

            # Wait before next allocation
            if delay_seconds > 0:
                time.sleep(delay_seconds)

    except KeyboardInterrupt:
        print("\nStopped by user")

    # Final memory status
    final_mem_info = get_gpu_memory_info(device_id)
    if final_mem_info:
        print(f"\nFinal Status:")
        print(f"✓ Successfully allocated {len(allocated_tensors)} chunks")
        print(f"✓ Total PyTorch allocated: {final_mem_info['allocated']:.3f}GB")
        print(f"✓ Total PyTorch cached: {final_mem_info['cached']:.3f}GB")

        if successful_allocations > 0:
            print(
                f"✓ GPU memory pressure created - other processes should now get OOM errors"
            )
        else:
            print("⚠ No memory was allocated - GPU may already be at capacity")

    # Keep memory allocated until user decides to release
    input("\nPress Enter to release all allocated memory and exit...")

    # Clean up
    print("Releasing memory...")
    del allocated_tensors
    gc.collect()
    torch.cuda.empty_cache()
    print("Memory released!")


def main():
    parser = argparse.ArgumentParser(
        description="CUDA Memory Hog - Gradually consume GPU memory"
    )
    parser.add_argument(
        "--chunk-size",
        type=float,
        default=0.1,
        help="Starting size of each memory chunk in MB (default: 0.1)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay between allocations in seconds (default: 0.5)",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=1000,
        help="Maximum number of allocation attempts (default: 1000)",
    )
    parser.add_argument(
        "--device", type=int, default=0, help="CUDA device ID to target (default: 0)"
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    parser.add_argument(
        "--no-aggressive",
        action="store_true",
        help="Disable aggressive ultra-small allocation mode",
    )
    parser.add_argument(
        "--ultra-conservative",
        action="store_true",
        help="Start with ultra-small allocations (0.001MB)",
    )

    args = parser.parse_args()

    # Adjust chunk size for ultra-conservative mode
    if args.ultra_conservative:
        args.chunk_size = 0.001

    memory_hog(
        chunk_size_mb=args.chunk_size,
        delay_seconds=args.delay,
        max_attempts=args.max_attempts,
        device_id=args.device,
        verbose=not args.quiet,
        aggressive=not args.no_aggressive,
    )


if __name__ == "__main__":
    main()
