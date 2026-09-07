"""Implementations of Cyclic Reduction solvers."""

from typing import Any

import numpy as np
import numpy.typing as npt
from numba import njit, prange

from ..utils import ArrayLike

# --- Numba Step Functions ---


@njit(parallel=True)
def _reduce_step_parallel(
    curr_a: npt.NDArray, curr_b: npt.NDArray, curr_c: npt.NDArray, curr_d: npt.NDArray
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    n = len(curr_d)
    m = (n - 1) // 2
    dtype = curr_d.dtype
    next_a = np.empty(m - 1, dtype=dtype)
    next_b = np.empty(m, dtype=dtype)
    next_c = np.empty(m - 1, dtype=dtype)
    next_d = np.empty(m, dtype=dtype)
    for j in prange(m):
        i = 2 * j + 1
        alpha = curr_a[i - 1] / curr_b[i - 1]
        beta = curr_c[i] / curr_b[i + 1]
        next_b[j] = curr_b[i] - alpha * curr_c[i - 1] - beta * curr_a[i]
        next_d[j] = curr_d[i] - alpha * curr_d[i - 1] - beta * curr_d[i + 1]
        if j > 0:
            next_a[j - 1] = -alpha * curr_a[i - 2]
        if j < m - 1:
            next_c[j] = -beta * curr_c[i + 1]
    return next_a, next_b, next_c, next_d


@njit
def _reduce_step_sequential(
    curr_a: npt.NDArray, curr_b: npt.NDArray, curr_c: npt.NDArray, curr_d: npt.NDArray
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    n = len(curr_d)
    m = (n - 1) // 2
    dtype = curr_d.dtype
    next_a = np.empty(m - 1, dtype=dtype)
    next_b = np.empty(m, dtype=dtype)
    next_c = np.empty(m - 1, dtype=dtype)
    next_d = np.empty(m, dtype=dtype)
    for j in range(m):
        i = 2 * j + 1
        alpha = curr_a[i - 1] / curr_b[i - 1]
        beta = curr_c[i] / curr_b[i + 1]
        next_b[j] = curr_b[i] - alpha * curr_c[i - 1] - beta * curr_a[i]
        next_d[j] = curr_d[i] - alpha * curr_d[i - 1] - beta * curr_d[i + 1]
        if j > 0:
            next_a[j - 1] = -alpha * curr_a[i - 2]
        if j < m - 1:
            next_c[j] = -beta * curr_c[i + 1]
    return next_a, next_b, next_c, next_d


@njit(parallel=True)
def _substitute_step_parallel(
    prev_a: npt.NDArray,
    prev_b: npt.NDArray,
    prev_c: npt.NDArray,
    prev_d: npt.NDArray,
    x: npt.NDArray,
) -> npt.NDArray:
    n = len(prev_d)
    dtype = prev_d.dtype
    next_x = np.empty(n, dtype=dtype)
    m_x = len(x)
    for j in prange(m_x):
        next_x[2 * j + 1] = x[j]
    num_even = (n + 1) // 2
    for j in prange(num_even):
        i = 2 * j
        rhs = prev_d[i]
        if i > 0:
            rhs -= prev_a[i - 1] * next_x[i - 1]
        if i < n - 1:
            rhs -= prev_c[i] * next_x[i + 1]
        next_x[i] = rhs / prev_b[i]
    return next_x


@njit
def _substitute_step_sequential(
    prev_a: npt.NDArray,
    prev_b: npt.NDArray,
    prev_c: npt.NDArray,
    prev_d: npt.NDArray,
    x: npt.NDArray,
) -> npt.NDArray:
    n = len(prev_d)
    dtype = prev_d.dtype
    next_x = np.empty(n, dtype=dtype)
    m_x = len(x)
    for j in range(m_x):
        next_x[2 * j + 1] = x[j]
    num_even = (n + 1) // 2
    for j in range(num_even):
        i = 2 * j
        rhs = prev_d[i]
        if i > 0:
            rhs -= prev_a[i - 1] * next_x[i - 1]
        if i < n - 1:
            rhs -= prev_c[i] * next_x[i + 1]
        next_x[i] = rhs / prev_b[i]
    return next_x


# --- Public Solvers ---


def solve_numba(a: npt.NDArray, b: npt.NDArray, c: npt.NDArray, d: npt.NDArray) -> npt.NDArray:
    """Solve a tridiagonal system Ax = d using sequential Numba-accelerated Cyclic Reduction.

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
    history: list[tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]] = []
    curr_a, curr_b, curr_c, curr_d = a, b, c, d
    while len(curr_d) > 1:
        history.append((curr_a, curr_b, curr_c, curr_d))
        curr_a, curr_b, curr_c, curr_d = _reduce_step_sequential(curr_a, curr_b, curr_c, curr_d)
    x = np.array([curr_d[0] / curr_b[0]], dtype=curr_d.dtype)
    for prev_a, prev_b, prev_c, prev_d in reversed(history):
        x = _substitute_step_sequential(prev_a, prev_b, prev_c, prev_d, x)
    return x


def solve_numba_parallel(
    a: npt.NDArray, b: npt.NDArray, c: npt.NDArray, d: npt.NDArray
) -> npt.NDArray:
    """Solve a tridiagonal system Ax = d using parallel Numba-accelerated Cyclic Reduction.

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
    history: list[tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]] = []
    curr_a, curr_b, curr_c, curr_d = a, b, c, d
    while len(curr_d) > 1:
        history.append((curr_a, curr_b, curr_c, curr_d))
        curr_a, curr_b, curr_c, curr_d = _reduce_step_parallel(curr_a, curr_b, curr_c, curr_d)
    x = np.array([curr_d[0] / curr_b[0]], dtype=curr_d.dtype)
    for prev_a, prev_b, prev_c, prev_d in reversed(history):
        x = _substitute_step_parallel(prev_a, prev_b, prev_c, prev_d, x)
    return x


def solve_numpy(a: npt.NDArray, b: npt.NDArray, c: npt.NDArray, d: npt.NDArray) -> npt.NDArray:
    """Solve a tridiagonal system Ax = d using vectorized NumPy Cyclic Reduction.

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
    history = []
    curr_a, curr_b, curr_c, curr_d = a, b, c, d
    while len(curr_d) > 1:
        history.append((curr_a, curr_b, curr_c, curr_d))
        current_n = len(curr_d)
        odd_indices = np.arange(1, current_n, 2)
        alpha_factors = curr_a[odd_indices - 1] / curr_b[odd_indices - 1]
        beta_factors = curr_c[odd_indices] / curr_b[odd_indices + 1]
        new_b = (
            curr_b[odd_indices]
            - alpha_factors * curr_c[odd_indices - 1]
            - beta_factors * curr_a[odd_indices]
        )
        new_d = (
            curr_d[odd_indices]
            - alpha_factors * curr_d[odd_indices - 1]
            - beta_factors * curr_d[odd_indices + 1]
        )
        new_a = -alpha_factors[1:] * curr_a[odd_indices[1:] - 2]
        new_c = -beta_factors[:-1] * curr_c[odd_indices[:-1] + 1]
        curr_a, curr_b, curr_c, curr_d = new_a, new_b, new_c, new_d
    x = curr_d / curr_b
    for prev_a, prev_b, prev_c, prev_d in reversed(history):
        next_x = np.zeros(len(prev_d), dtype=prev_d.dtype)
        next_x[1::2] = x
        rhs = prev_d[0::2].copy()
        rhs[: len(prev_c[0::2])] -= prev_c[0::2] * next_x[1::2]
        rhs[1:] -= prev_a[1::2] * next_x[1:-1:2]
        next_x[0::2] = rhs / prev_b[0::2]
        x = next_x
    return x


def solve_vanilla(a: npt.NDArray, b: npt.NDArray, c: npt.NDArray, d: npt.NDArray) -> npt.NDArray:
    """Solve a tridiagonal system Ax = d using sequential pure Python Cyclic Reduction.

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
    # Note: Using lists for true vanilla performance
    history = []
    curr_a, curr_b, curr_c, curr_d = a.tolist(), b.tolist(), c.tolist(), d.tolist()
    while len(curr_d) > 1:
        history.append((curr_a, curr_b, curr_c, curr_d))
        next_a, next_b, next_c, next_d = [], [], [], []
        for i in range(1, len(curr_d), 2):
            alpha = curr_a[i - 1] / curr_b[i - 1]
            beta = curr_c[i] / curr_b[i + 1]
            next_b.append(curr_b[i] - alpha * curr_c[i - 1] - beta * curr_a[i])
            next_d.append(curr_d[i] - alpha * curr_d[i - 1] - beta * curr_d[i + 1])
            if i > 1:
                next_a.append(-alpha * curr_a[i - 2])
            if i < len(curr_d) - 2:
                next_c.append(-beta * curr_c[i + 1])
        curr_a, curr_b, curr_c, curr_d = next_a, next_b, next_c, next_d
    x = [curr_d[0] / curr_b[0]]
    for prev_a, prev_b, prev_c, prev_d in reversed(history):
        next_x = [0.0] * len(prev_d)
        for j in range(len(x)):
            next_x[2 * j + 1] = x[j]
        for i in range(0, len(prev_d), 2):
            rhs = prev_d[i]
            if i > 0:
                rhs -= prev_a[i - 1] * next_x[i - 1]
            if i < len(prev_d) - 1:
                rhs -= prev_c[i] * next_x[i + 1]
            next_x[i] = rhs / prev_b[i]
        x = next_x
    return np.array(x)


# --- MLX Implementation ---
try:
    import mlx.core as mx

    def solve_mlx(a: ArrayLike, b: ArrayLike, c: ArrayLike, d: ArrayLike) -> Any:
        """Solve Ax = d using differentiable GPU-accelerated MLX Cyclic Reduction.

        Automatically routes float64 inputs to the CPU stream, as Apple Silicon GPUs
        only natively support float32/float16.

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
        # Detect dtype and device
        dtype = mx.float32 if a.dtype in (np.float32, mx.float32) else mx.float64
        device = mx.cpu if dtype == mx.float64 else mx.gpu

        with mx.stream(device):
            curr_a, curr_b, curr_c, curr_d = (
                mx.array(a, dtype=dtype),
                mx.array(b, dtype=dtype),
                mx.array(c, dtype=dtype),
                mx.array(d, dtype=dtype),
            )
            history = []
            while curr_d.size > 1:
                history.append((curr_a, curr_b, curr_c, curr_d))
                # Length m
                alpha = curr_a[::2] / curr_b[:-1:2]
                beta  = curr_c[1::2] / curr_b[2::2]
                # Length m
                new_b = curr_b[1::2] - alpha * curr_c[::2] - beta * curr_a[1::2]
                new_d = curr_d[1::2] - alpha * curr_d[:-1:2] - beta * curr_d[2::2]
                # Length m - 1
                new_a = -alpha[1:] * curr_a[1::2][:-1]
                new_c = -beta[:-1] * curr_c[2::2]
                curr_a, curr_b, curr_c, curr_d = new_a, new_b, new_c, new_d
            x = curr_d / curr_b
            for prev_a, prev_b, prev_c, prev_d in reversed(history):
                rhs = prev_d[0::2]

                # Subtract contribution from right neighbor
                c_even = prev_c[0::2]
                num_c_even = c_even.size
                rhs = mx.concatenate(
                    [
                        rhs[:num_c_even] - c_even * x[:num_c_even],
                        rhs[num_c_even:],
                    ]
                )

                # Subtract contribution from left neighbor
                a_even = prev_a[1::2]
                if a_even.size > 0:
                    rhs = mx.concatenate([rhs[:1], rhs[1:] - a_even * x[:a_even.size]])

                # Interleave even indices (rhs / prev_b[0::2]) and odd indices (x)
                even_part = rhs / prev_b[0::2]
                pad_size = even_part.size - x.size
                if pad_size > 0:
                    odd_padded = mx.concatenate([x, mx.zeros((pad_size,), dtype=x.dtype)])
                else:
                    odd_padded = x
                x = mx.reshape(mx.stack([even_part, odd_padded], axis=-1), (-1,))[:prev_d.size]
            return x
except ImportError:

    def solve_mlx(a: ArrayLike, b: ArrayLike, c: ArrayLike, d: ArrayLike) -> Any:
        """Handle calls to mlx solver when mlx is not installed."""
        raise ImportError("MLX not found. Please install mlx to use solve_mlx.")
