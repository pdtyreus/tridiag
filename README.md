# tridiag: High-Performance, Differentiable Tridiagonal Solvers

`tridiag` is a Python library for efficiently solving tridiagonal linear systems ($Ax=d$). It provides a variety of implementations optimized for different architectures and problem scales—from pure Python to Numba-accelerated parallel solvers, multi-dimensional batch solvers, and natively differentiable, GPU-accelerated MLX solvers with custom Metal kernels for Apple Silicon.

## Benchmarks & Performance Analysis

To understand the performance of the various implementations across different system sizes and batch configurations:
- **Single Systems:** [Tridiagonal Solvers Benchmark](https://pdtyreus.github.io/numerical_methods/tridiagonal_solvers.html)
- **Batch Systems (Apple Silicon GPU & CPU):** [Tridiagonal Solvers: MLX Batch](https://pdtyreus.github.io/numerical_methods/tridiagonal_mlx_batch.html)

<p align="center">
  <img src="https://pdtyreus.github.io/numerical_methods/tridiagonal/all_impl.png" alt="Single System Benchmarks" width="48%" />
  <img src="https://pdtyreus.github.io/numerical_methods/tridiagonal/batch_mlx.png" alt="Batch Benchmarks" width="48%" />
</p>

## Key Features

- **Multiple Algorithms:** 
    - **Thomas Algorithm ([TDMA](https://pdtyreus.github.io/numerical_methods/tridiagonal_thomas.html)):** Efficient sequential solver for small to medium single systems, and parallelized across batches.
    - **Cyclic Reduction ([CR](https://pdtyreus.github.io/numerical_methods/tridiagonal_cyclic_reduction.html)):** Parallelizable solver for large-scale single systems.
    - **Parallel Cyclic Reduction ([PCR](https://pdtyreus.github.io/numerical_methods/tridiagonal_parallel_cyclic_reduction.html)):** Massively parallel solver optimized for GPU launch efficiency.
- **Multidimensional & Batched Solving:**
    - Solve thousands of systems simultaneously along any dimension (`axis`) for 2D/3D Alternating Direction Implicit (ADI) PDE simulations, splines, and image processing.
    - Supports both shared (1D) and independent (multidimensional) diagonal coefficients.
- **Optimized Backends:**
    - **Metal Shading Language (MSL):** Custom fused GPU compute kernel (`mx.fast.metal_kernel`) keeping intermediate variables in GPU registers for maximum batch throughput.
    - **MLX:** GPU-accelerated solvers optimized for Apple Silicon with native reverse-mode automatic differentiation (`mx.grad`).
    - **Numba:** JIT-compiled CPU acceleration with multi-core parallel execution.
- **Smart Dispatch:** Automatically selects the best solver based on hardware, array dimensions, dtype, and problem size via `method='auto'`.

## Performance Guide

Based on benchmarks on Apple Silicon (M5), here is a guide for choosing the right solver:

### 1. Single Systems ($Ax = d$, 1D arrays)

| System Size ($N$) | Recommended Method | Why? |
| :--- | :--- | :--- |
| $N < 10,000$ | `thomas_numba` | Lowest overhead, highly efficient sequential JIT code. |
| $10,000 < N < 1,000,000$ | `cr_numba` | CPU Parallelism (8+ cores) beats sequential Thomas. |
| $N > 1,000,000$ (float32) | `cr_numba` or `cr_mlx` | CPU/GPU memory bandwidth becomes the bottleneck. |

### 2. Batched Systems ($M$ systems along an `axis`, e.g., 2D/3D grids)

| Workload Type | Recommended Method | Why? |
| :--- | :--- | :--- |
| Batched Systems (float32, Apple Silicon) | `thomas_metal` | Fused Metal kernel eliminates intermediate global memory round-trips and keeps variables in GPU registers. |
| Batched Systems (float64 or CPU) | `thomas_numba` | Multi-core parallel execution across CPU threads via Numba `prange`. |
| Differentiable Batches (Machine Learning) | `thomas_mlx` or `thomas_metal` | Seamless integration with `mx.grad` for neural network layers and inverse PDE problems. |

## Numba vs. MLX / Metal: Which to Choose?

While **Numba (Parallel CPU)** is often faster for a single massive 1D system, the **MLX and Metal (GPU)** backends are the superior choice in several key scenarios:

1. **Large Batches (ADI & Multidimensional Grids):** For 2D/3D simulations requiring thousands of simultaneous tridiagonal solves, the fused Metal GPU kernel (`thomas_metal`) significantly outperforms multi-core CPU solvers by taking advantage of high unified memory bandwidth.
2. **Machine Learning Integration & Autograd:** MLX solvers are written using native MLX operations and are **fully differentiable**. If you use tridiagonal solvers within neural networks (e.g. implicit layers, spline layers, or physics-informed neural networks), you can directly backpropagate gradients using `mx.grad`.
3. **Data Locality:** If tensors already reside on the GPU from upstream MLX operations, using MLX/Metal avoids expensive GPU-CPU memory copies.
4. **Graph Fusion:** MLX's lazy evaluation allows the compiler to fuse the tridiagonal solve with surrounding operations.

### Differentiable Solvers Example

```python
import mlx.core as mx
import tridiag

# Create arrays
n = 127
a = mx.random.uniform(shape=(n-1,))
b = mx.random.uniform(shape=(n,)) + 2.0  # diagonally dominant
c = mx.random.uniform(shape=(n-1,))
d = mx.random.uniform(shape=(n,))
target = mx.random.uniform(shape=(n,))

def loss_fn(b_val):
    # Solve Ax = d on Apple Silicon GPU/CPU
    x = tridiag.solve(a, b_val, c, d, method='cr_mlx')
    return mx.mean((x - target) ** 2)

# Compute gradient of loss with respect to diagonal coefficients b
grad_b = mx.grad(loss_fn)(b)
print("Gradient computed successfully:", grad_b.shape)
```

## Important Note on Precision (float32 vs float64)

Apple Silicon GPUs **do not natively support float64** (Double Precision). 
* Use `float32` for maximum performance on MLX and Metal.
* For high-precision `float64` needs, use `cr_numba` (single system) or `thomas_numba` (batch), which are heavily optimized for multi-core CPUs.

## Installation

```bash
# Basic installation
pip install tridiag

# Installation with GPU support (Apple Silicon)
pip install "tridiag[gpu]"
```

## Quick Start

### Single System

```python
import tridiag
from tridiag.utils import generate_tridiagonal_system

# Create a single tridiagonal system (N = 1023)
n = 1023
a, b, c, d, _ = generate_tridiagonal_system(n)

# Solve using smart auto-dispatch
x = tridiag.solve(a, b, c, d, method='auto')
print(f"Solution size: {len(x)}")
```

### Batched & Multidimensional Systems

```python
import numpy as np
import tridiag

# Solve a batch of 1,000 systems of size N = 256
batch_size, n = 1000, 256
a = np.random.uniform(-1, 0, (n - 1,)).astype(np.float32)
b = np.random.uniform(2, 3, (n,)).astype(np.float32)
c = np.random.uniform(-1, 0, (n - 1,)).astype(np.float32)
d = np.random.uniform(0, 1, (batch_size, n)).astype(np.float32)

# Solve along axis=-1 using auto-dispatch (routes to MLX on Apple Silicon)
x = tridiag.solve(a, b, c, d, axis=-1, method='auto')

# Or explicitly use the high-performance Metal GPU kernel:
x_metal = tridiag.solve(a, b, c, d, axis=-1, method='thomas_metal')
```

## Advanced Usage

Explicitly choose a solver:

```python
# Single system solvers:
x_numba = tridiag.solve(a, b, c, d, method='cr_numba')
x_pcr   = tridiag.solve(a, b, c, d, method='pcr_mlx')
x_seq   = tridiag.solve(a, b, c, d, method='thomas_numba')

# Batch solvers:
x_metal = tridiag.solve(a, b, c, d_batch, axis=-1, method='thomas_metal')  # Metal GPU kernel
x_mlx   = tridiag.solve(a, b, c, d_batch, axis=-1, method='thomas_mlx')    # Pure MLX GPU
x_cpu   = tridiag.solve(a, b, c, d_batch, axis=-1, method='thomas_numba')  # Multi-core CPU
```

## Application Examples

The repository includes complete scientific simulation examples:

- **[2D Thermal Plume Simulation (`examples/thermal_plume_2d_adi.py`)](examples/thermal_plume_2d_adi.py):**
  Simulates advection-diffusion of heat from an off-center continuous heat source in laminar channel flow using the Peaceman-Rachford Alternating Direction Implicit (ADI) scheme. Demonstrates solving non-symmetric tridiagonal systems ($a \ne c$) in alternating horizontal (`axis=1`) and vertical (`axis=0`) batched sweeps.

- **[2D Reaction-Diffusion & Turing Patterns (`examples/reaction_diffusion_2d_adi.py`)](examples/reaction_diffusion_2d_adi.py):**
  Simulates the coupled non-linear Gray-Scott equations to model morphogenetic Turing pattern formation (solitons, spots, branching rings, and labyrinths) across a spatial feed gradient, solved via batched ADI sweeps with zero-flux Neumann boundary conditions.

- **[1D Heat Conduction (`examples/heat_conduction_1d.py` / `examples/heat_conduction_mlx.py`)](examples/heat_conduction_1d.py):**
  Classic 1D transient heat diffusion using Crank-Nicolson implicit time stepping, comparing CPU and MLX differentiable solves.

## Development

If you want to run benchmarks locally, you can set up the environment using `poetry`:

1. **Install dependencies:**
   ```bash
   poetry install --all-extras
   ```

2. **Run tests:**
   ```bash
   poetry run pytest
   ```

3. **Run the linter:**
   ```bash
   poetry run ruff check src/
   ```

4. **Run benchmarks:**
   ```bash
   # Standard single-system benchmark (compares against SciPy baseline)
   poetry run python benchmarks/run_benchmarks.py

   # Massive single systems (up to N=134M)
   poetry run python benchmarks/run_benchmarks.py --massive

   # Include MLX solver
   poetry run python benchmarks/run_benchmarks.py --massive --mlx

   # Batch solver benchmarks
   poetry run python benchmarks/run_batch_benchmarks.py

   # Batch solvers with MLX & Metal (Apple Silicon)
   poetry run python benchmarks/run_batch_benchmarks.py \
      --mlx --dtype float32 --massive --shared
   ```

5. **Run Examples:**
   ```bash
   poetry run python examples/thermal_plume_2d_adi.py
   poetry run python examples/reaction_diffusion_2d_adi.py
   poetry run python examples/heat_conduction_1d.py
   ```
