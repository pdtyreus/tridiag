"""Benchmark suite for batch tridiagonal solvers."""

import time
import argparse

import numpy as np

import tridiag
from tridiag.solvers import thomas
from tridiag.utils import generate_tridiagonal_batch_system


def benchmark_fn(fn, args, iterations=5):
    """Warm up and time a function."""
    res = fn(*args)  # Warm-up / JIT compile
    if hasattr(res, "tolist") and type(res).__module__.startswith("mlx"):
        import mlx.core as mx
        mx.eval(res)

    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        res = fn(*args)
        if hasattr(res, "tolist") and type(res).__module__.startswith("mlx"):
            import mlx.core as mx
            mx.eval(res)
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


def run_batch_suite(method, n=128, batch_sizes=(10, 100, 1000, 10000, 50000), dtype=np.float64):
    """Run the batch benchmark suite."""
    dtype_name = np.dtype(dtype).name
    print(f"\nBatch Solver Benchmark (Method={method}, Size N={n}, dtype={dtype_name})")
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

        if "mlx" in method:
            # don't penalize MLX for converting arrays from numpy to mlx on each step
            # in a normal simulation, the arrays would already be mx.array
            import mlx.core as mx
            sub = mx.array(sub, dtype=mx.float32)
            main = mx.array(main, dtype=mx.float32)
            sup = mx.array(sup, dtype=mx.float32)
            rhs = mx.array(rhs, dtype=mx.float32)
            mx.eval(sub, main, sup, rhs)

        # 2. Batched solver (via top-level solve)
        t_batch = benchmark_fn(
            lambda a, b, c, d: tridiag.solve(a, b, c, d, method=method),
            (sub, main, sup, rhs)
        )

        speedup = t_seq / t_batch
        throughput = m / t_batch  # Systems solved per second

        print(
            f"{m:>10} | {t_seq:>15.5f} | {t_batch:>16.5f} | {speedup:>8.1f}x | {throughput:>18,.0f}"
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Benchmark batch tridiag solvers.")
    parser.add_argument("--size", default=128, help="System size N")
    parser.add_argument("--dtype", default="float64", help="dtype (e.g. float32, float64)")
    parser.add_argument("--mlx", action="store_true", help="Include MLX benchmark.")
    parser.add_argument("--massive", action="store_true", help="Run up to N=100M.")
    args = parser.parse_args()

    methods = ["thomas_numba_batch"]

    if args.mlx:
        methods += ['thomas_mlx_batch']

    dt = np.dtype(args.dtype)

    if args.massive:
        batch_sizes=(1000, 10000, 50000, 100000, 200000)
    else:
        batch_sizes=(10, 100, 1000, 10000, 50000)

    for method in methods:
        run_batch_suite(method, n=128, dtype=dt, batch_sizes=batch_sizes)