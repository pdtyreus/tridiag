"""Solvers for tridiagonal systems."""

import importlib.util
import warnings
from typing import Any

import numpy as np
import numpy.typing as npt

from .solvers import cyclic_reduction, pcr, thomas
from .utils import pad_system


def solve(
    a: npt.NDArray, b: npt.NDArray, c: npt.NDArray, d: npt.NDArray, method: str = "auto"
) -> npt.NDArray | Any:
    """
    Solve a tridiagonal system Ax = d.

    Parameters
    ----------
    a, b, c : Diagonals of the matrix.
    d       : Right-hand side.
    method  : 'auto', 'thomas_numba', 'cr_numba', 'cr_numba_parallel',
              'cr_mlx', 'pcr_numba', 'pcr_mlx'
    """
    n = len(d)
    if n == 0:
        raise ValueError("System must have size N >= 1.")
    if n == 1:
        return d / b

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
        if "mlx" in method and input_dtype == np.float64:
            warnings.warn(
                f"Method '{method}' with float64 will run on the CPU (MLX fallback). "
                "For better float64 performance, use 'cr_numba' or 'thomas_numba'.",
                UserWarning,
                stacklevel=2,
            )

        solvers = {
            "thomas_vanilla": thomas.solve_vanilla,
            "thomas_numba": thomas.solve_numba,
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

    # Final Result Handling (including padding removal)
    if has_mlx and hasattr(x, "tolist"):  # Result is an MLX array
        import mlx.core as mx

        # Inside MLX block, use MLX dtypes
        if x.dtype == mx.float64:
            with mx.stream(mx.cpu):
                return x[:n]
        else:
            with mx.stream(mx.gpu):
                return x[:n]

    return x[:n]


__all__ = ["solve", "thomas", "cyclic_reduction", "pcr", "utils"]
