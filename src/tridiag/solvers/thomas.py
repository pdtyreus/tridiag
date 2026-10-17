"""Implementations of Thomas Algorithm solvers."""

from typing import Any

import numpy as np
import numpy.typing as npt
from numba import njit, prange

from ..utils import ArrayLike


def solve_vanilla(a: npt.NDArray, b: npt.NDArray, c: npt.NDArray, d: npt.NDArray) -> npt.NDArray:
    """Solve a tridiagonal system Ax = d using the Thomas algorithm (pure Python).

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
    n = len(d)
    cp = [0.0] * (n - 1)
    dp = [0.0] * n
    x = [0.0] * n

    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]

    for i in range(1, n - 1):
        den = b[i] - a[i - 1] * cp[i - 1]
        cp[i] = c[i] / den
        dp[i] = (d[i] - a[i - 1] * dp[i - 1]) / den

    dp[n - 1] = (d[n - 1] - a[n - 2] * dp[n - 2]) / (b[n - 1] - a[n - 2] * cp[n - 2])

    x[n - 1] = dp[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return np.array(x)


@njit
def solve_numba(a: npt.NDArray, b: npt.NDArray, c: npt.NDArray, d: npt.NDArray) -> npt.NDArray:
    """Solve a tridiagonal system Ax = d using the Numba-compiled Thomas algorithm.

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
    n = len(d)
    cp = np.empty(n - 1, dtype=d.dtype)
    dp = np.empty(n, dtype=d.dtype)
    x = np.empty(n, dtype=d.dtype)

    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]

    for i in range(1, n - 1):
        den = b[i] - a[i - 1] * cp[i - 1]
        cp[i] = c[i] / den
        dp[i] = (d[i] - a[i - 1] * dp[i - 1]) / den

    dp[n - 1] = (d[n - 1] - a[n - 2] * dp[n - 2]) / (b[n - 1] - a[n - 2] * cp[n - 2])

    x[n - 1] = dp[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x

@njit(parallel=True)
def solve_numba_batch(
    a: npt.NDArray, b: npt.NDArray, c: npt.NDArray, d: npt.NDArray
) -> npt.NDArray:
    """Solve a batch of M independent tridiagonal systems using the Numba-compiled Thomas algorithm.

    Parameters
    ----------
    a : ndarray
        The lower diagonal(s) of shape (M, N - 1) or broadcastable (N - 1,).
    b : ndarray
        The main diagonal(s) of shape (M, N) or broadcastable (N,).
    c : ndarray
        The upper diagonal(s) of shape (M, N - 1) or broadcastable (N - 1,).
    d : ndarray
        The right-hand side matrix of shape (M, N).

    Returns
    -------
    ndarray
        The solution matrix x of shape (M, N).
    """
    m, n = d.shape
    x = np.empty((m, n), dtype=d.dtype)

    is_a_1d = a.ndim == 1
    is_b_1d = b.ndim == 1
    is_c_1d = c.ndim == 1

    for k in prange(m):
        a_k = a if is_a_1d else a[k]
        b_k = b if is_b_1d else b[k]
        c_k = c if is_c_1d else c[k]
        d_k = d[k]

        # we could probably have just reused `solve_numba` here:
        #   x[k, :] = solve_numba(a_k, b_k, c_k, d_k)
        # but it does use an extra allocation of an x array at
        # each step, so it's worth duplicating a little code
        # for performance

        cp = np.empty(n - 1, dtype=d.dtype)
        dp = np.empty(n, dtype=d.dtype)

        cp[0] = c_k[0] / b_k[0]
        dp[0] = d_k[0] / b_k[0]

        for i in range(1, n - 1):
            den = b_k[i] - a_k[i - 1] * cp[i - 1]
            cp[i] = c_k[i] / den
            dp[i] = (d_k[i] - a_k[i - 1] * dp[i - 1]) / den

        dp[n - 1] = (d_k[n - 1] - a_k[n - 2] * dp[n - 2]) / (b_k[n - 1] - a_k[n - 2] * cp[n - 2])

        x[k, n - 1] = dp[n - 1]
        for i in range(n - 2, -1, -1):
            x[k, i] = dp[i] - cp[i] * x[k, i + 1]

    return x

try:
    import mlx.core as mx

    @mx.compile
    def _thomas_batch_core(a, b, c, d):
        """Inner vectorized Thomas loop in pure MLX."""
        m, n = d.shape
        is_a_1d = a.ndim == 1
        is_b_1d = b.ndim == 1
        is_c_1d = c.ndim == 1

        # Transpose to (N, M) so that row i is a contiguous 1D array of length M
        # This avoids striding operations
        # .T is overloaded to call the native MLX transpose operation
        d_T = d.T
        b_T = b if is_b_1d else b.T
        a_T = a if is_a_1d else a.T
        c_T = c if is_c_1d else c.T

        # Forward Elimination
        cp = [c_T[0] / b_T[0]]
        dp = [d_T[0] / b_T[0]]
        for i in range(1, n - 1):
            den = b_T[i] - a_T[i - 1] * cp[-1]
            cp.append(c_T[i] / den)
            dp.append((d_T[i] - a_T[i - 1] * dp[-1]) / den)
        den_last = b_T[n - 1] - a_T[n - 2] * cp[-1]
        dp.append((d_T[n - 1] - a_T[n - 2] * dp[-1]) / den_last)

        # Back Substitution
        x_rev = [dp[-1]]
        for i in range(n - 2, -1, -1):
            x_prev = dp[i] - cp[i] * x_rev[-1]
            x_rev.append(x_prev)
        return mx.stack(x_rev[::-1], axis=-1)

    def solve_mlx_batch(
        a: ArrayLike, b: ArrayLike, c: ArrayLike, d: ArrayLike
    ) -> Any:
        """Solve a batch of M independent tridiagonal on Apple silicon.
        
        The method uses the Thomas algorithm but parallelizes the work
        across GPU cores using MLX.
        
        Parameters
        ----------
        a : ndarray
            The lower diagonal(s) of shape (M, N - 1) or broadcastable (N - 1,).
        b : ndarray
            The main diagonal(s) of shape (M, N) or broadcastable (N,).
        c : ndarray
            The upper diagonal(s) of shape (M, N - 1) or broadcastable (N - 1,).
        d : ndarray
            The right-hand side matrix of shape (M, N).
        
        Returns
        -------
        ndarray
            The solution matrix x of shape (M, N).
        """
        if d.dtype in (np.float64, mx.float64):
            raise TypeError(
                "MLX solvers require 'float32' for Apple Silicon GPU acceleration. "
                "Metal GPUs do not support 64-bit floating point (float64).\n"
                "  - To use the GPU: cast your inputs to float32 (e.g. `d.astype(np.float32)`).\n"
                "  - To use float64: use the optimized CPU solver (`method='thomas_numba_batch'`)."
            )

        mx_a = mx.array(a, dtype=mx.float32) if not isinstance(a, mx.array) else a
        mx_b = mx.array(b, dtype=mx.float32) if not isinstance(b, mx.array) else b
        mx_c = mx.array(c, dtype=mx.float32) if not isinstance(c, mx.array) else c
        mx_d = mx.array(d, dtype=mx.float32) if not isinstance(d, mx.array) else d

        return _thomas_batch_core(mx_a, mx_b, mx_c, mx_d)

except ImportError:

    def solve_mlx_batch(
        a: ArrayLike, b: ArrayLike, c: ArrayLike, d: ArrayLike
    ) -> Any:
        """Handle calls to mlx solver when mlx is not installed."""
        raise ImportError("MLX not found. Please install mlx to use solve_mlx_batch.")