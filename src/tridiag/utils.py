"""Utilities for tridiagonal systems."""

import numpy as np
import numpy.typing as npt


def generate_tridiagonal_system(
    n: int, seed: int = 42
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """
    Generate a random diagonally dominant tridiagonal system Ax = d.

    Returns: (sub_diag, main_diag, super_diag, rhs, true_x)
    """
    np.random.seed(seed)

    # Generate random vectors for the diagonals
    sub_diag = np.random.rand(n - 1)
    main_diag = np.random.rand(n)
    super_diag = np.random.rand(n - 1)

    # Force diagonal dominance for numerical stability
    main_diag += (np.append(sub_diag, 0) + np.insert(super_diag, 0, 0)) + 0.1

    true_x = np.random.rand(n)

    # Calculate b = T @ true_x
    rhs = main_diag * true_x
    rhs[:-1] += super_diag * true_x[1:]
    rhs[1:] += sub_diag * true_x[:-1]

    return sub_diag, main_diag, super_diag, rhs, true_x


def pad_system(
    a: npt.NDArray, b: npt.NDArray, c: npt.NDArray, d: npt.NDArray
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """Pad a tridiagonal system to size 2^k - 1 for Cyclic Reduction."""
    n = len(d)
    k = int(np.ceil(np.log2(n + 1)))
    new_n = 2**k - 1

    if new_n == n:
        return a, b, c, d

    new_a = np.zeros(new_n - 1, dtype=a.dtype)
    new_b = np.ones(new_n, dtype=b.dtype)
    new_c = np.zeros(new_n - 1, dtype=c.dtype)
    new_d = np.zeros(new_n, dtype=d.dtype)

    new_a[: len(a)] = a
    new_b[:n] = b
    new_c[: len(c)] = c
    new_d[:n] = d

    return new_a, new_b, new_c, new_d
