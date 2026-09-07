"""Benchmark suite for tridiagonal solver implementations."""

import argparse
import time

import numpy as np
from scipy.linalg import lapack

import tridiag
from tridiag.utils import generate_tridiagonal_system


def solve_scipy(a, b, c, d):
    """SciPy/LAPACK (dgtsv/sgtsv) implementation for comparison."""
    func = lapack.get_lapack_funcs("gtsv", (a, b, c, d))
    _, _, _, x, info = func(a, b, c, d)
    if info != 0:
        raise ValueError(f"LAPACK gtsv failed with info {info}")
    return x


def benchmark_solver(method, sub, main, sup, rhs, iterations=5):
    """Solves using the supplied method with warmup."""
    # Warm-up
    if method == "thomas_scipy":
        _ = solve_scipy(sub, main, sup, rhs)
    else:
        res = tridiag.solve(sub, main, sup, rhs, method=method)
        if type(res).__module__.startswith("mlx"):
            import mlx.core as mx
            mx.eval(res)
        else:
            pass

    # Time the runs
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        if method == "thomas_scipy":
            res = solve_scipy(sub, main, sup, rhs)
        else:
            res = tridiag.solve(sub, main, sup, rhs, method=method)
        # Ensure result is computed (important for lazy frameworks like MLX)
        if hasattr(res, "tolist"):
            # If MLX, force graph evaluation on the device without Python heap object conversion
            if type(res).__module__.startswith("mlx"):
                import mlx.core as mx
                mx.eval(res)
            else:
                pass  # NumPy is already evaluated
        times.append(time.perf_counter() - start)

    return np.mean(times)


def run_suite(sizes, methods, dtype=np.float32):
    """Run the benchmark suite."""
    print(f"\nRunning Benchmarks (dtype={dtype.__name__})")
    print("Values are mean execution time (s)")
    header = f"{'Size (N)':>12} | " + " | ".join([f"{m:>17}" for m in methods])
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    for n in sizes:
        sub, main, sup, rhs, _ = generate_tridiagonal_system(n)
        sub = sub.astype(dtype)
        main = main.astype(dtype)
        sup = sup.astype(dtype)
        rhs = rhs.astype(dtype)

        row = f"{n:>12} | "
        results = []
        for method in methods:
            # Skip vanilla for large N
            if "vanilla" in method and n > 10000:
                results.append("n/a")
                continue

            try:
                t = benchmark_solver(method, sub, main, sup, rhs)
                results.append(f"{t:>17.6f}")
            except Exception:
                results.append(f"{'error':>17}")

        print(row + " | ".join(results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark tridiag solvers.")
    parser.add_argument("--quick", action="store_true", help="Run only small sizes.")
    parser.add_argument("--massive", action="store_true", help="Run up to N=100M.")
    parser.add_argument("--mlx", action="store_true", help="Include MLX benchmark.")
    args = parser.parse_args()

    if args.quick:
        sizes = [2**k - 1 for k in [7, 10, 13]]
    elif args.massive:
        sizes = [2**k - 1 for k in [10, 15, 20, 25, 27]]
    else:
        sizes = [2**k - 1 for k in [10, 15, 18, 21]]

    methods = ["thomas_scipy", "thomas_numba", "cr_numpy", "cr_numba", "cr_numba_parallel"]

    if args.mlx:
        methods += ['cr_mlx']

    run_suite(sizes, methods, dtype=np.float32)
