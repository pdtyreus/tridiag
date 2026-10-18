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
        d_trans = d.T
        b_trans = b if is_b_1d else b.T
        a_trans = a if is_a_1d else a.T
        c_trans = c if is_c_1d else c.T

        # Forward Elimination
        cp = [c_trans[0] / b_trans[0]]
        dp = [d_trans[0] / b_trans[0]]
        for i in range(1, n - 1):
            den = b_trans[i] - a_trans[i - 1] * cp[-1]
            cp.append(c_trans[i] / den)
            dp.append((d_trans[i] - a_trans[i - 1] * dp[-1]) / den)
        den_last = b_trans[n - 1] - a_trans[n - 2] * cp[-1]
        dp.append((d_trans[n - 1] - a_trans[n - 2] * dp[-1]) / den_last)

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

    _batch_thomas_metal_source = """
        uint tid = thread_position_in_grid.x;
        if (tid >= M) return;

        float cp[N];

        float b0 = IS_SHARED ? b[0] : b[0 * M + tid];
        float c0 = IS_SHARED ? c[0] : c[0 * M + tid];

        cp[0] = c0 / b0;
        x[0 * M + tid] = d[0 * M + tid] / b0;

        for (int i = 1; i < N - 1; i++) {
            float a_prev = IS_SHARED ? a[i - 1] : a[(i - 1) * M + tid];
            float b_curr = IS_SHARED ? b[i] : b[i * M + tid];
            float c_curr = IS_SHARED ? c[i] : c[i * M + tid];

            float den = b_curr - a_prev * cp[i - 1];
            cp[i] = c_curr / den;
            float d_val = d[i * M + tid];
            float prev_x = x[(i - 1) * M + tid];
            x[i * M + tid] = (d_val - a_prev * prev_x) / den;
        }

        float a_last = IS_SHARED ? a[N - 2] : a[(N - 2) * M + tid];
        float b_last = IS_SHARED ? b[N - 1] : b[(N - 1) * M + tid];
        float den_last = b_last - a_last * cp[N - 2];
        float d_last = d[(N - 1) * M + tid];
        float prev_x = x[(N - 2) * M + tid];
        x[(N - 1) * M + tid] = (d_last - a_last * prev_x) / den_last;

        for (int i = N - 2; i >= 0; i--) {
            x[i * M + tid] -= cp[i] * x[(i + 1) * M + tid];
        }
    """

    _batch_thomas_metal_kernel = mx.fast.metal_kernel(
        name="batch_thomas_universal",
        input_names=["a", "b", "c", "d"],
        output_names=["x"],
        source=_batch_thomas_metal_source,
    )

    def solve_mlx_metal(
        a: ArrayLike, b: ArrayLike, c: ArrayLike, d: ArrayLike
    ) -> Any:
        """Solve a batch of tridiagonal systems using a custom Metal kernel.
        
        Sovles M independent tridiagonal systems Ax = d using a 
        custom Metal kernel on Apple Silicon.

        Parameters
        ----------
        a : ndarray or mlx.core.array
            The lower diagonal(s) of shape (M, N - 1) or broadcastable (N - 1,).
        b : ndarray or mlx.core.array
            The main diagonal(s) of shape (M, N) or broadcastable (N,).
        c : ndarray or mlx.core.array
            The upper diagonal(s) of shape (M, N - 1) or broadcastable (N - 1,).
        d : ndarray or mlx.core.array
            The right-hand side matrix of shape (M, N).

        Returns
        -------
        mlx.core.array
            The solution matrix x of shape (M, N).
        """
        if d.dtype in (np.float64, mx.float64):
            raise TypeError(
                "MLX solvers require 'float32' for Apple Silicon GPU acceleration. "
                "Metal GPUs do not support 64-bit floating point (float64).\n"
                "  - To use the GPU: cast your inputs to float32 (e.g. `d.astype(np.float32)`).\n"
                "  - To use float64: use the optimized CPU solver (`method='thomas_numba_batch'`)."
            )

        m, n = d.shape
        is_shared = (a.ndim == 1)

        mx_a = mx.array(a, dtype=mx.float32) if not isinstance(a, mx.array) else a
        mx_b = mx.array(b, dtype=mx.float32) if not isinstance(b, mx.array) else b
        mx_c = mx.array(c, dtype=mx.float32) if not isinstance(c, mx.array) else c
        mx_d = mx.array(d, dtype=mx.float32) if not isinstance(d, mx.array) else d

        a_in = mx_a if is_shared else mx_a.T
        b_in = mx_b if is_shared else mx_b.T
        c_in = mx_c if is_shared else mx_c.T
        d_in = mx_d.T

        res_trans = _batch_thomas_metal_kernel(
            inputs=[a_in, b_in, c_in, d_in],
            template=[("IS_SHARED", is_shared), ("N", n), ("M", m)],
            grid=(m, 1, 1),
            threadgroup=(min(m, 256), 1, 1),
            output_shapes=[(n, m)],
            output_dtypes=[mx.float32],
        )[0]

        return res_trans.T

except ImportError:

    def solve_mlx_batch(
        a: ArrayLike, b: ArrayLike, c: ArrayLike, d: ArrayLike
    ) -> Any:
        """Handle calls to mlx solver when mlx is not installed."""
        raise ImportError("MLX not found. Please install mlx to use solve_mlx_batch.")

    def solve_mlx_metal(
        a: ArrayLike, b: ArrayLike, c: ArrayLike, d: ArrayLike
    ) -> Any:
        """Handle calls to mlx metal solver when mlx is not installed."""
        raise ImportError("MLX not found. Please install mlx to use solve_mlx_metal.")