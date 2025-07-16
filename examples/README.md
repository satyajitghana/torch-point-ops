# torch-point-ops Examples

This folder contains example scripts demonstrating how to use torch-point-ops for point cloud distance computations.

## Available Examples

### `basic_usage.py`
A comprehensive example showing how to:
- Import and use both Chamfer Distance and Earth Mover's Distance
- Work with different sized point clouds  
- Use both functional and module-based APIs
- Compare the characteristics of different distance metrics

**Run the example:**
```bash
python examples/basic_usage.py
```

## Import Styles

torch-point-ops supports clean, intuitive imports:

```python
# Method 1: Import from submodules (recommended)
from torch_point_ops.chamfer import chamfer_distance
from torch_point_ops.emd import earth_movers_distance

# Method 2: Import from top-level package
from torch_point_ops import chamfer_distance, earth_movers_distance

# Method 3: Import modules for loss functions
from torch_point_ops import ChamferDistance, EarthMoverDistance
```

## Quick Reference

### Chamfer Distance
```python
# Returns bidirectional distances
dist1, dist2 = chamfer_distance(points1, points2)

# Using as a loss module  
chamfer_loss = ChamferDistance(reduction='mean')
loss = chamfer_loss(points1, points2)
```

### Earth Mover's Distance  
```python
# Returns EMD distances per batch
emd_dist = earth_movers_distance(points1, points2)

# Using as a loss module
emd_loss = EarthMoverDistance()
loss = emd_loss(points1, points2)
```

## Requirements

- PyTorch with CUDA support (for GPU acceleration)
- Point clouds as tensors with shape `(batch_size, num_points, 3)`

## Notes

- **No need to worry about tensor contiguity**: The library automatically handles `contiguous()` calls internally
- **Flexible tensor inputs**: Works with both contiguous and non-contiguous tensors
- **GPU acceleration**: Automatically uses CUDA when available

For more advanced usage, check out the test files in `tests/`. 