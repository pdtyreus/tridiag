"""Implementations of Parallel Cyclic Reduction solvers."""

from typing import Any

import numpy as np
import numpy.typing as npt
from numba import njit, prange

from ..utils import ArrayLike

# --- Numba Implementation ---


@njit(parallel=True)
def _pcr_step_numba(
    a: npt.NDArray, b: npt.NDArray, c: npt.NDArray, d: npt.NDArray, distance: int
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    n = b.shape[0]
    dtype = b.dtype

    new_a = np.zeros(n, dtype=dtype)
    new_b = np.zeros(n, dtype=dtype)
    new_c = np.zeros(n, dtype=dtype)
    new_d = np.zeros(n, dtype=dtype)

    for i in prange(n):
        im = i - distance
        ip = i + distance

        if im >= 0:
            alpha = a[i] / b[im]
            new_a[i] = -alpha * a[im]
            new_b[i] = b[i] - alpha * c[im]
            new_d[i] = d[i] - alpha * d[im]
        else:
            new_b[i] = b[i]
            new_d[i] = d[i]

        if ip < n:
            beta = c[i] / b[ip]
            new_b[i] -= beta * a[ip]
            new_c[i] = -beta * c[ip]
            new_d[i] -= beta * d[ip]

    return new_a, new_b, new_c, new_d


def solve_numba(a: npt.NDArray, b: npt.NDArray, c: npt.NDArray, d: npt.NDArray) -> npt.NDArray:
    """Solve a tridiagonal system Ax = d using Parallel Cyclic Reduction (PCR) with Numba.

    Parameters
    ----------
    a : ndarray
        The lower diagonal of the matrix of shape (N - 1,).
    b : ndarray
        The main diagonal of the matrix of shape (N,).
    c : ndarray
        The upper diagonal of the matrix of shape (N - 1,).
    d : ndarray
        The right-hand side vector of shape (N,).

    Returns
    -------
    ndarray
        The solution vector x of shape (N,).
    """
    n = b.shape[0]
    curr_b = b.copy()
    curr_d = d.copy()
    curr_a = np.zeros(n, dtype=b.dtype)
    curr_a[1:] = a
    curr_c = np.zeros(n, dtype=b.dtype)
    curr_c[:-1] = c
    num_steps = int(np.ceil(np.log2(n)))
    for k in range(num_steps):
        distance = 2**k
        curr_a, curr_b, curr_c, curr_d = _pcr_step_numba(curr_a, curr_b, curr_c, curr_d, distance)
    return curr_d / curr_b


# --- NumPy Implementation (Vectorized) ---


def solve_numpy(a: npt.NDArray, b: npt.NDArray, c: npt.NDArray, d: npt.NDArray) -> npt.NDArray:
    """Solve a tridiagonal system Ax = d using Parallel Cyclic Reduction (PCR) with NumPy.

    Uses vectorized slicing for O(N log N) total work.

    Parameters
    ----------
    a : ndarray
        The lower diagonal of the matrix of shape (N - 1,).
    b : ndarray
        The main diagonal of the matrix of shape (N,).
    c : ndarray
        The upper diagonal of the matrix of shape (N - 1,).
    d : ndarray
        The right-hand side vector of shape (N,).

    Returns
    -------
    ndarray
        The solution vector x of shape (N,).
    """
    n = b.shape[0]
    dtype = b.dtype

    curr_b = b.copy()
    curr_d = d.copy()
    curr_a = np.zeros(n, dtype=dtype)
    curr_a[1:] = a
    curr_c = np.zeros(n, dtype=dtype)
    curr_c[:-1] = c

    num_steps = int(np.ceil(np.log2(n)))

    for k in range(num_steps):
        distance = 2**k

        # alpha_i = a_i / b_{i-distance}
        # beta_i = c_i / b_{i+distance}
        alpha = np.zeros(n, dtype=dtype)
        beta = np.zeros(n, dtype=dtype)

        # i >= distance
        alpha[distance:] = curr_a[distance:] / curr_b[:-distance]
        # i < n - distance
        beta[:-distance] = curr_c[:-distance] / curr_b[distance:]

        # New coefficients
        new_a = np.zeros(n, dtype=dtype)
        new_c = np.zeros(n, dtype=dtype)

        # new_a_i = -alpha_i * a_{i-distance}
        new_a[distance:] = -alpha[distance:] * curr_a[:-distance]
        # new_c_i = -beta_i * c_{i+distance}
        new_c[:-distance] = -beta[:-distance] * curr_c[distance:]

        # new_b_i = b_i - alpha_i * c_{i-distance} - beta_i * a_{i+distance}
        new_b = curr_b.copy()
        new_b[distance:] -= alpha[distance:] * curr_c[:-distance]
        new_b[:-distance] -= beta[:-distance] * curr_a[distance:]

        # new_d_i = d_i - alpha_i * d_{i-distance} - beta_i * d_{i+distance}
        new_d = curr_d.copy()
        new_d[distance:] -= alpha[distance:] * curr_d[:-distance]
        new_d[:-distance] -= beta[:-distance] * curr_d[distance:]

        curr_a, curr_b, curr_c, curr_d = new_a, new_b, new_c, new_d

    return curr_d / curr_b


# --- MLX Implementation ---

try:
    import mlx.core as mx

    @mx.compile
    def _pcr_step_mlx(a, b, c, d, distance):
        n = b.shape[0]
        idx_minus = mx.maximum(mx.arange(n) - distance, 0)
        idx_plus = mx.minimum(mx.arange(n) + distance, n - 1)
        b_minus = b[idx_minus]
        b_plus = b[idx_plus]
        mask_minus = mx.arange(n) >= distance
        mask_plus = mx.arange(n) < n - distance
        alpha = mx.where(mask_minus, a / b_minus, 0.0)
        beta = mx.where(mask_plus, c / b_plus, 0.0)
        d_minus = d[idx_minus]
        d_plus = d[idx_plus]
        a_minus = a[idx_minus]
        c_minus = c[idx_minus]
        a_plus = a[idx_plus]
        c_plus = c[idx_plus]
        new_a = -alpha * a_minus
        new_c = -beta * c_plus
        new_b = b - alpha * c_minus - beta * a_plus
        new_d = d - alpha * d_minus - beta * d_plus
        return new_a, new_b, new_c, new_d

    def solve_mlx(a: ArrayLike, b: ArrayLike, c: ArrayLike, d: ArrayLike) -> Any:
        """Solve a tridiagonal system Ax = d using Parallel Cyclic Reduction (PCR) in MLX.

        Parameters
        ----------
        a : ndarray
            The lower diagonal of the matrix of shape (N - 1,).
        b : ndarray or mlx.core.array
            The main diagonal of the matrix of shape (N,).
        c : ndarray
            The upper diagonal of the matrix of shape (N - 1,).
        d : ndarray or mlx.core.array
            The right-hand side vector of shape (N,).

        Returns
        -------
        mlx.core.array
            The solution vector x of shape (N,).
        """
        n = b.shape[0]
        dtype = mx.float32 if a.dtype in (np.float32, mx.float32) else mx.float64
        device = mx.cpu if dtype == mx.float64 else mx.gpu
        with mx.stream(device):
            curr_b = mx.array(b, dtype=dtype)
            curr_d = mx.array(d, dtype=dtype)
            curr_a = mx.zeros((n,), dtype=dtype)
            curr_a[1:] = mx.array(a, dtype=dtype)
            curr_c = mx.zeros((n,), dtype=dtype)
            curr_c[:-1] = mx.array(c, dtype=dtype)
            num_steps = int(np.ceil(np.log2(n)))
            for k in range(num_steps):
                distance = 2**k
                curr_a, curr_b, curr_c, curr_d = _pcr_step_mlx(
                    curr_a, curr_b, curr_c, curr_d, distance
                )
            x = curr_d / curr_b
            mx.eval(x)
            return x

except ImportError:

    def solve_mlx(a: ArrayLike, b: ArrayLike, c: ArrayLike, d: ArrayLike) -> Any:
        """Handle calls to mlx solver when mlx is not installed."""
        raise ImportError("MLX not found. Please install mlx to use pcr_mlx.")
