# tridiag: High-Performance, Differentiable Tridiagonal Solvers

`tridiag` is a Python library for efficiently solving tridiagonal linear systems ($Ax=d$). It provides a variety of implementations optimized for different architectures and system sizes, from pure Python to Numba-accelerated parallel solvers and natively differentiable, GPU-accelerated MLX solvers for Apple Silicon.

To understand the performance of the various implementations across different system sizes and hardware capabilities, see these [tridiagonal solver benchmark results](https://pdtyreus.github.io/numerical_methods/tridiagonal_solvers.html).

![](https://pdtyreus.github.io/numerical_methods/tridiagonal/all_impl.png)

## Key Features

- **Multiple Algorithms:** 
    - **Thomas Algorithm ([TDMA](https://pdtyreus.github.io/numerical_methods/tridiagonal_thomas.html)):** Efficient sequential solver for small to medium systems.
    - **Cyclic Reduction ([CR](https://pdtyreus.github.io/numerical_methods/tridiagonal_cyclic_reduction.html)):** Parallelizable solver for large-scale systems.
    - **Parallel Cyclic Reduction ([PCR](https://pdtyreus.github.io/numerical_methods/tridiagonal_parallel_cyclic_reduction.html)):** Massively parallel solver optimized for GPU launch efficiency.
- **Optimized Backends:**
    - **Numba:** JIT-compiled CPU acceleration with multi-core parallelization.
    - **MLX:** GPU-accelerated solvers optimized for Apple Silicon.
- **Smart Dispatch:** Use `method='auto'` to automatically select the best solver for your hardware and problem size.

## Performance Guide

Based on benchmarks on Apple Silicon (M5), here is a guide for choosing the right solver:

| System Size (N) | Recommended Method | Why? |
| :--- | :--- | :--- |
| $N < 10,000$ | `thomas_numba` | Lowest overhead, highly efficient sequential JIT code. |
| $10,000 < N < 1,000,000$ | `cr_numba` | CPU Parallelism (8+ cores) beats sequential Thomas. |
| $N > 1,000,000$ (float32) | `cr_numba` or `cr_mlx` | CPU/GPU memory bandwidth becomes the bottleneck. |

## Numba vs. MLX: Which to Choose?

While our benchmarks show that **Numba (Parallel CPU)** can be faster for a single massive system due to low-latency cache pre-fetching, there are several scenarios where the **MLX (GPU)** backend is the superior choice:

1.  **Machine Learning Integration:** The MLX implementation is built using native MLX primitives, making it **fully differentiable**. If you are using this solver inside a neural network, you can use `mx.grad` to backpropagate through the solver.

### Differentiable Solvers Example

Because the MLX solvers are written using native MLX operations, they are fully compatible with autograd. You can easily backpropagate through the solver:

```python
import mlx.core as mx
import numpy as np
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
2.  **Data Locality:** If your tensors are already on the GPU from a previous MLX operation, using the MLX solver avoids the expensive synchronization and conversion costs required to move data back to the CPU for Numba.
3.  **Graph Fusion:** MLX's lazy evaluation allows the compiler to "fuse" the tridiagonal solve with surrounding operations (like activations or subsequent matrix math), potentially optimizing the entire execution pipeline.

## Important Note on Precision (float32 vs float64)

Apple Silicon GPUs **do not natively support float64** (Double Precision). 
*   **Recommendation:** Use `float32` for maximum performance on MLX.
*   **Fallback:** If you use `float64` with MLX backends, the library will automatically switch to the CPU stream, which is significantly slower.
*   **Better Alternative:** For high-precision `float64` needs, the `cr_numba` or `thomas_numba` methods are highly optimized for the CPU and will outperform the MLX fallback.

## Installation

```bash
# Basic installation
pip install tridiag

# Installation with GPU support (Apple Silicon)
pip install "tridiag[gpu]"
```
*(Requires `numpy` and `numba`. For GPU acceleration, `mlx` is required.)*

## Quick Start

```python
import tridiag
import numpy as np
from tridiag.utils import generate_tridiagonal_system

# Create a tridiagonal system
n = 1023
a, b, c, d, _ = generate_tridiagonal_system(n)

# Solve using the smart auto-dispatcher
x = tridiag.solve(a, b, c, d, method='auto')

print(f"Solution size: {len(x)}")
```

## Advanced Usage

Explicitly choose a solver:

```python
x_numba = tridiag.solve(a, b, c, d, method='cr_numba')
x_pcr   = tridiag.solve(a, b, c, d, method='pcr_mlx')
x_seq   = tridiag.solve(a, b, c, d, method='thomas_numba')
```

## Development

If you want to contribute or run the benchmarks locally, you can set up the environment using `poetry`:

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
   # Standard benchmark (compares against SciPy baseline)
   poetry run python benchmarks/run_benchmarks.py

   # Massive systems (up to N=134M)
   poetry run python benchmarks/run_benchmarks.py --massive

   # Include MLX solver
   poetry run python benchmarks/run_benchmarks.py --massive --mlx
   ```

5. **Run Examples:**
   ```bash
   poetry run python examples/heat_conduction_1d.py
   ```