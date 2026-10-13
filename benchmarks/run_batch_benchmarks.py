"""Benchmark suite for batch tridiagonal solvers."""

import time

import numpy as np

import tridiag
from tridiag.solvers import thomas
from tridiag.utils import generate_tridiagonal_batch_system


def benchmark_fn(fn, args, iterations=5):
    """Warm up and time a function."""
    _ = fn(*args)  # Warm-up / JIT compile
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = fn(*args)
        times.append(time.perf_counter() - t0)
    return np.mean(times)


def solve_sequential(a, b, c, d):
    """Solve a system using a reference sequential algorithm."""
    m = d.shape[0]
    x = np.empty_like(d)
    is_1d = a.ndim == 1
    for i in range(m):
        a_i = a if is_1d else a[i]
        b_i = b if is_1d else b[i]
        c_i = c if is_1d else c[i]
        x[i] = thomas.solve_numba(a_i, b_i, c_i, d[i])
    return x


def run_batch_suite(n=128, batch_sizes=(10, 100, 1000, 10000, 50000), dtype=np.float64):
    """Run the batch benchmark suite."""
    print(f"\nBatch Solver Benchmark (System Size N={n}, dtype={dtype.__name__})")
    print(
        f"{'M (Batch)':>10} | {'Sequential (s)':>15} | {'Batch Numba (s)':>16} | "
        f"{'Speedup':>9} | {'Throughput (sys/s)':>20}"
        )
    print("-" * 80)

    for m in batch_sizes:
        sub, main, sup, rhs, _ = generate_tridiagonal_batch_system(m, n, shared_diagonals=True)
        sub = sub.astype(dtype)
        main = main.astype(dtype)
        sup = sup.astype(dtype)
        rhs = rhs.astype(dtype)

        # 1. Sequential reference
        t_seq = benchmark_fn(solve_sequential, (sub, main, sup, rhs))

        # 2. Batched solver (via top-level solve)
        t_batch = benchmark_fn(
            lambda a, b, c, d: tridiag.solve(a, b, c, d, method="thomas_numba_batch"),
            (sub, main, sup, rhs)
        )

        speedup = t_seq / t_batch
        throughput = m / t_batch  # Systems solved per second

        print(
            f"{m:>10} | {t_seq:>15.5f} | {t_batch:>16.5f} | {speedup:>8.1f}x | {throughput:>18,.0f}"
        )


if __name__ == "__main__":
    run_batch_suite(n=128)