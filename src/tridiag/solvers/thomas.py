"""Implementations of Thomas Algorithm solvers."""

import numpy as np
import numpy.typing as npt
from numba import njit, prange


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
