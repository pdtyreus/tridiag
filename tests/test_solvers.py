import numpy as np
import pytest

import tridiag
from tridiag.utils import generate_tridiagonal_system


def to_numpy(x):
    """Helper to convert potentially MLX array to numpy."""
    if hasattr(x, "tolist"):
        return np.array(x.tolist())
    return x

# All available methods to test
METHODS = [
    'thomas_vanilla',
    'thomas_numba',
    'cr_vanilla',
    'cr_numpy',
    'cr_numba',
    'cr_numba_parallel',
    'cr_mlx',
    'pcr_numpy',
    'pcr_numba',
    'pcr_mlx',
    'auto'
]

# Sizes to test: small, medium, and non-power-of-2 for CR padding
SIZES = [1, 2, 3, 15, 127, 100, 1024]


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_all_solvers(method, n, dtype):
    """Consistent test for all solvers across different sizes and dtypes."""
    # Skip MLX if not available
    if 'mlx' in method:
        try:
            import mlx.core as mx  # noqa: F401
        except ImportError:
            pytest.skip("MLX not installed")

    sub, main, sup, rhs, true_x = generate_tridiagonal_system(n)
    
    # Cast to requested dtype
    sub = sub.astype(dtype)
    main = main.astype(dtype)
    sup = sup.astype(dtype)
    rhs = rhs.astype(dtype)
    true_x = true_x.astype(dtype)
    
    try:
        # Capture and verify the expected fallback warning for MLX float64
        if 'mlx' in method and dtype == np.float64:
            with pytest.warns(UserWarning, match="will run on the CPU"):
                calculated_x = tridiag.solve(sub, main, sup, rhs, method=method)
        else:
            calculated_x = tridiag.solve(sub, main, sup, rhs, method=method)
    except ImportError as e:
        if "MLX" in str(e):
            pytest.skip("MLX not installed")
        raise e

    calculated_x = to_numpy(calculated_x)
    
    # Check shape
    assert calculated_x.shape == (n,)
    
    # Check correctness
    # Use looser tolerance for float32 and for MLX float64
    if dtype == np.float32:
        tol = 1e-4
    elif 'mlx' in method or method == 'auto' and n > 1_000_000:
        tol = 1e-6 # MLX float64 can be slightly less precise due to accumulation
    else:
        tol = 1e-10
    
    np.testing.assert_allclose(calculated_x, true_x, rtol=tol, atol=tol)

def test_diagonal_dominance_failure():
    """Test with a non-diagonally dominant matrix to see if it still works or fails gracefully."""
    # This is more of a numerical stability test
    n = 5
    a = np.ones(n-1) * 2.0
    b = np.ones(n) * 1.0 # Not dominant
    c = np.ones(n-1) * 2.0
    d = np.ones(n)
    
    # Most of our solvers (Thomas, CR) don't explicitly check for stability, 
    # but they might produce NaNs or incorrect results.
    _ = tridiag.solve(a, b, c, d, method="thomas_numba")
