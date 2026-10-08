import argparse
import time
import numpy as np
import tsrocket

DEFAULT_FEATURES = [
    "total_sum",
    "mean",
    "variance",
    "std_dev",
    "min_value",
    "max_value",
    "median",
    "skewness",
    "kurtosis",
    "mad",
    "iqr",
    "entropy",
    "energy",
    "rms",
    "zero_crossing_rate",
    "peak_count",
    "mean_abs_change",
    "mean_change",
    "cid_ce",
    "auc",
    "abs_sum_change",
    "count_above_mean",
    "count_below_mean",
    "longest_strike_above_mean",
    "longest_strike_below_mean",
    "abs_max",
    "first_loc_max",
    "last_loc_max",
    "first_loc_min",
    "last_loc_min",
]


def run_synthetic_benchmark(
    n_cols: int = 10,
    total_len: int = 10000,
    initial_size: int = 500,
    increment_size: int = 100,
    warmup: bool = True,
):
    print("=" * 70)
    print(f"TSRocket ExpandingExtractor Streaming Benchmark")
    print(f"Columns (Series): {n_cols} | Total Points: {total_len}")
    print(f"Initial Window: {initial_size} | Increment Size: {increment_size}")
    print(f"Features Computed: {len(DEFAULT_FEATURES)}")
    print("=" * 70)

    # Generate synthetic series
    np.random.seed(42)
    data = np.random.randn(n_cols, total_len).astype(np.float32)

    exp_ext = tsrocket.ExpandingExtractor(DEFAULT_FEATURES, n_cols)

    # Warmup
    if warmup:
        warmup_batch = np.ascontiguousarray(data[:, :50])
        _ = exp_ext.update(warmup_batch)
        # Re-initialize after warmup
        exp_ext = tsrocket.ExpandingExtractor(DEFAULT_FEATURES, n_cols)

    # Initial batch
    batch_initial = np.ascontiguousarray(data[:, :initial_size])
    t0 = time.perf_counter()
    _ = exp_ext.update(batch_initial)
    initial_ms = (time.perf_counter() - t0) * 1000
    print(f"Initial batch ({initial_size} pts/col): {initial_ms:.4f} ms")

    # Incremental updates
    timings = []
    current_size = initial_size
    while current_size + increment_size <= total_len:
        batch_inc = np.ascontiguousarray(
            data[:, current_size : current_size + increment_size]
        )
        t0 = time.perf_counter()
        _ = exp_ext.update(batch_inc)
        timings.append((time.perf_counter() - t0) * 1000)
        current_size += increment_size

    timings = np.array(timings)
    n_updates = len(timings)
    total_inc_ms = np.sum(timings)
    total_data_points = n_cols * (total_len - initial_size)

    print("\n--- Incremental Update Summary ---")
    print(f"Number of Incremental Updates: {n_updates}")
    print(
        f"Mean Time / Update ({increment_size} pts): {np.mean(timings):.4f} ms (±{np.std(timings):.4f} ms)"
    )
    print(f"Median Time / Update:           {np.median(timings):.4f} ms")
    print(
        f"Min / Max Update Time:          {np.min(timings):.4f} ms / {np.max(timings):.4f} ms"
    )
    print(
        f"Total Time for Updates:         {total_inc_ms:.2f} ms ({total_inc_ms / 1000:.4f} s)"
    )
    throughput_pts_sec = total_data_points / (total_inc_ms / 1000)
    print(f"Throughput (Points Processed):  {throughput_pts_sec:,.0f} pts/sec")
    print(
        f"Feature Evals / Second:         {throughput_pts_sec * len(DEFAULT_FEATURES):,.0f} evals/sec"
    )
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TSRocket ExpandingExtractor Benchmark")
    parser.add_argument(
        "--n_cols", type=int, default=10, help="Number of concurrent series"
    )
    parser.add_argument(
        "--total_len", type=int, default=10000, help="Total length of series"
    )
    parser.add_argument(
        "--initial_size", type=int, default=500, help="Initial points for window"
    )
    parser.add_argument(
        "--increment_size", type=int, default=100, help="Points per incremental update"
    )
    args = parser.parse_args()

    run_synthetic_benchmark(
        n_cols=args.n_cols,
        total_len=args.total_len,
        initial_size=args.initial_size,
        increment_size=args.increment_size,
    )
