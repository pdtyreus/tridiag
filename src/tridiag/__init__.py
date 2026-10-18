"""Solvers for tridiagonal systems."""

import importlib.util

import numpy as np

from .solvers import cyclic_reduction, pcr, thomas
from .utils import HAS_MLX, ArrayLike, is_float32, moveaxis, pad_system


def _solve_batch(
    a: ArrayLike,
    b: ArrayLike,
    c: ArrayLike,
    d: ArrayLike,
    axis: int = -1,
    method: str = "auto",
) -> ArrayLike:
    """Solve multidimensional / batched tridiagonal systems."""
    ndim = d.ndim
    if axis < -ndim or axis >= ndim:
        raise ValueError(f"axis {axis} is out of bounds for array of dimension {ndim}")
    # 1. Normalize target axis to the last dimension (-1)
    axis_norm = axis if axis >= 0 else axis + ndim
    need_moveaxis = axis_norm != (ndim - 1)
    if need_moveaxis:
        d = moveaxis(d, axis_norm, -1)
        if getattr(a, "ndim", 1) == ndim:
            a = moveaxis(a, axis_norm, -1)
        if getattr(b, "ndim", 1) == ndim:
            b = moveaxis(b, axis_norm, -1)
        if getattr(c, "ndim", 1) == ndim:
            c = moveaxis(c, axis_norm, -1)
    orig_shape = d.shape
    n = orig_shape[-1]
    if n < 2:
        raise ValueError(f"Tridiagonal systems must have size N >= 2 (received N={n}).")
    # 2. Flatten leading dimensions to (M, N)
    batch_shape = orig_shape[:-1]
    m = int(np.prod(batch_shape))
    d_2d = d.reshape((m, n))
    a_2d = a.reshape((m, n - 1)) if getattr(a, "ndim", 1) == ndim else a
    b_2d = b.reshape((m, n)) if getattr(b, "ndim", 1) == ndim else b
    c_2d = c.reshape((m, n - 1)) if getattr(c, "ndim", 1) == ndim else c
    # 3. Method dispatch
    batch_solvers = {
        "thomas_numba": thomas.solve_numba_batch,
        "thomas_numba_batch": thomas.solve_numba_batch,
        "thomas_mlx": thomas.solve_mlx_batch,
        "thomas_mlx_batch": thomas.solve_mlx_batch,
        "thomas_mlx_metal": thomas.solve_mlx_metal,
        "thomas_metal": thomas.solve_mlx_metal,
    }
    if method == "auto":
        if (HAS_MLX and is_float32(d)):
            solver_func = thomas.solve_mlx_batch  
        else :
            solver_func = thomas.solve_numba_batch
    elif method in batch_solvers:
        solver_func = batch_solvers[method]
    else:
        raise ValueError(
            f"Method '{method}' is not supported for batched/multidimensional inputs. "
            f"Available: {list(batch_solvers.keys()) + ['auto']}"
        )
    x_2d = solver_func(a_2d, b_2d, c_2d, d_2d)
    # 4. Reshape back and restore axis
    x = x_2d.reshape(orig_shape)
    if need_moveaxis:
        x = moveaxis(x, -1, axis_norm)
    return x

def _solve(
    a: ArrayLike, b: ArrayLike, c: ArrayLike, d: ArrayLike, method: str = "auto"
) -> ArrayLike:
    """Solve a single tridiagonal system."""
    n = len(d)
    if n < 2:
        raise ValueError(f"Tridiagonal systems must have size N >= 2 (received N={n}).")

    # The input dtype will be a NumPy dtype (e.g. np.float32 or np.float64)
    input_dtype = a.dtype

    # Cyclic reduction methods require padding to 2^k - 1
    is_cr = "cr" in method or method == "auto" or "pcr" in method

    if is_cr and (n & (n + 1)) != 0:
        a, b, c, d = pad_system(a, b, c, d)

    has_mlx = importlib.util.find_spec("mlx") is not None

    if method == "auto":
        # Smart Dispatch Strategy:
        # 1. If float64, always use Numba (MLX would fallback to slower CPU anyway)
        if input_dtype == np.float64:
            if n > 10_000_000:
                x = cyclic_reduction.solve_numba_parallel(a, b, c, d)
            else:
                x = thomas.solve_numba(a, b, c, d)
        # 2. If float32 and massive, use MLX GPU (if available)
        elif n > 1_000_000 and has_mlx:
            x = cyclic_reduction.solve_mlx(a, b, c, d)
        # 3. Otherwise, use Numba (Sequential wins until ~10^7)
        elif n > 10_000_000:
            x = cyclic_reduction.solve_numba_parallel(a, b, c, d)
        else:
            x = cyclic_reduction.solve_numba(a, b, c, d)
    else:
        # Explicit methods

        solvers = {
            "thomas_vanilla": thomas.solve_vanilla,
            "thomas_numba": thomas.solve_numba,
            "thomas_metal": thomas.solve_mlx_metal,
            "cr_vanilla": cyclic_reduction.solve_vanilla,
            "cr_numpy": cyclic_reduction.solve_numpy,
            "cr_numba": cyclic_reduction.solve_numba,
            "cr_numba_parallel": cyclic_reduction.solve_numba_parallel,
            "cr_mlx": cyclic_reduction.solve_mlx,
            "pcr_numpy": pcr.solve_numpy,
            "pcr_numba": pcr.solve_numba,
            "pcr_mlx": pcr.solve_mlx,
        }

        if method not in solvers:
            raise ValueError(f"Unknown method: {method}. Available: {list(solvers.keys())}")

        x = solvers[method](a, b, c, d)

    return x[:n]


def solve(
    a: ArrayLike, b: ArrayLike, c: ArrayLike, d: ArrayLike, axis:int = -1, method: str = "auto"
) -> ArrayLike:
    """
    Solve a tridiagonal system Ax = d.

    Parameters
    ----------
    a, b, c : Diagonals of the matrix.
    d       : Right-hand side.
    axis    : Dimension of input arrays to use
    method  : 'auto', 'thomas_numba', 'cr_numba', 'cr_numba_parallel',
              'cr_mlx', 'pcr_numba', 'pcr_mlx'
    """
    # Route to batch solver if input is multidimensional
    if getattr(d, "ndim", 1) > 1:
        return _solve_batch(a, b, c, d, axis=axis, method=method)
    else:
        return _solve(a,b,c,d,method=method)

__all__ = ["solve", "thomas", "cyclic_reduction", "pcr", "utils"]
