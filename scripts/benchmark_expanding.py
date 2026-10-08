import os
import argparse
import time
import numpy as np
import pandas as pd
import tsrocket
import warnings

warnings.filterwarnings("ignore")

DEFAULT_CATEGORIES = ["N", "NS", "OT", "UT"]
SIGNAL_COL = [2, 3]
COLNAMES = [
    "Time (ms)",
    "Nset (1/min)",
    "Torque (Nm)",
    "Current (V)",
    "Angle (deg)",
    "Depth (mm)",
]

FEATURES = [
    "total_sum",
    "mean",
    "variance",
    "std_dev",
    "min_value",
    "max_value",
    "energy",
    "rms",
    "zero_crossing_rate",
    "peak_count",
    "mean_abs_change",
    "mean_change",
    "cid_ce",
    "auc",
]


def load_data(data_dir: str):
    dfs = []
    if os.path.exists(data_dir):
        print(f"Loading dataset from {data_dir}...")
        for cat in DEFAULT_CATEGORIES:
            cat_dir = os.path.join(data_dir, cat)
            if not os.path.exists(cat_dir):
                continue
            files = [f for f in os.listdir(cat_dir) if f.endswith(".csv")]
            print(f"  {cat}: loading {len(files)} files")
            for f in files:
                try:
                    df = pd.read_csv(os.path.join(cat_dir, f))
                    df = df.iloc[:, SIGNAL_COL].values.astype(np.float32).T
                    dfs.append(df)
                except Exception:
                    continue
    if not dfs:
        print(
            "Real dataset path not found or empty. Generating synthetic tightening/industrial data..."
        )
        np.random.seed(42)
        n_synthetic = 100
        length = 1500
        for _ in range(n_synthetic):
            torque = np.cumsum(np.random.normal(0.01, 0.1, length)).astype(np.float32)
            current = np.abs(np.random.normal(2.0, 0.5, length)).astype(np.float32)
            dfs.append(np.vstack([torque, current]))
    return dfs


def run_batched_benchmark(X, initial_size=500, expansion_size=100):
    total_samples = len(X)
    max_len = max(x.shape[1] for x in X)
    n_total_cols = total_samples * len(SIGNAL_COL)
    print(
        f"\n--- Running Batched Benchmark (Total series: {total_samples}, Total signals: {n_total_cols}) ---"
    )

    static_times = []
    expanding_times = []

    static_ext = tsrocket.Extractor(FEATURES)
    exp_ext = tsrocket.ExpandingExtractor(FEATURES, n_total_cols)

    # Initial 500 points
    # Static
    t0 = time.perf_counter()
    for x in X:
        if x.shape[1] < initial_size:
            continue
        batch = np.ascontiguousarray(x[:, :initial_size], dtype=np.float32)
        static_ext.process_2d_floats(batch)
    static_times.append(time.perf_counter() - t0)

    # Expanding
    arrays = []
    for s_idx, x in enumerate(X):
        if x.shape[1] < initial_size:
            continue
        for i in range(x.shape[0]):
            arrays.append(x[i, :initial_size])
    batch = np.stack(arrays).astype(np.float32)
    t0 = time.perf_counter()
    exp_ext.update(batch)
    expanding_times.append(time.perf_counter() - t0)

    print(
        f"Step 0 (size {initial_size}): Static={static_times[-1]:.4f}s, Expanding={expanding_times[-1]:.4f}s"
    )

    # Subsequent increments
    step = 1
    current_size = initial_size
    while current_size + expansion_size <= max_len:
        current_size += expansion_size

        # Static: re-extract full prefix
        t0 = time.perf_counter()
        count = 0
        for x in X:
            if x.shape[1] < current_size:
                continue
            batch = np.ascontiguousarray(x[:, :current_size], dtype=np.float32)
            static_ext.process_2d_floats(batch)
            count += 1
        if count == 0:
            break
        static_times.append(time.perf_counter() - t0)

        # Expanding: process only new points
        arrays = []
        for s_idx, x in enumerate(X):
            if x.shape[1] < current_size:
                continue
            for i in range(x.shape[0]):
                arrays.append(x[i, current_size - expansion_size : current_size])
        batch = np.stack(arrays).astype(np.float32)
        t0 = time.perf_counter()
        exp_ext.update(batch)
        expanding_times.append(time.perf_counter() - t0)

        print(
            f"Step {step} (size {current_size}): Static={static_times[-1]:.4f}s, Expanding={expanding_times[-1]:.4f}s (Speedup: {static_times[-1] / expanding_times[-1]:.2f}x)"
        )
        step += 1

    print("\nSummary Results:")
    print(
        f"{'Step':>4} | {'Size':>5} | {'Static (s)':>12} | {'Expanding (s)':>15} | {'Speedup':>8}"
    )
    print("-" * 55)
    for i in range(len(static_times)):
        sz = initial_size + i * expansion_size
        sp = static_times[i] / expanding_times[i] if expanding_times[i] > 0 else 0
        print(
            f"{i:4d} | {sz:5d} | {static_times[i]:12.4f} | {expanding_times[i]:15.4f} | {sp:7.2f}x"
        )


def run_per_series_benchmark(X, initial_size=500, expansion_size=100):
    total_samples = len(X)
    print(f"\n--- Running Per-Series Individual Benchmark ({total_samples} series) ---")
    static_step_timings = {}
    expanding_step_timings = {}
    static_ext = tsrocket.Extractor(FEATURES)

    for s_idx, x in enumerate(X):
        if x.shape[1] < initial_size:
            continue
        exp_ext = tsrocket.ExpandingExtractor(FEATURES, len(SIGNAL_COL))

        # Initial
        batch_static = np.ascontiguousarray(x[:, :initial_size], dtype=np.float32)
        t0 = time.perf_counter()
        static_ext.process_2d_floats(batch_static)
        static_step_timings.setdefault(0, []).append(time.perf_counter() - t0)

        batch_exp = np.ascontiguousarray(x[:, :initial_size], dtype=np.float32)
        t0 = time.perf_counter()
        exp_ext.update(batch_exp)
        expanding_step_timings.setdefault(0, []).append(time.perf_counter() - t0)

        step = 1
        current_size = initial_size
        while current_size + expansion_size <= x.shape[1]:
            current_size += expansion_size
            # Static
            b_s = np.ascontiguousarray(x[:, :current_size], dtype=np.float32)
            t0 = time.perf_counter()
            static_ext.process_2d_floats(b_s)
            static_step_timings.setdefault(step, []).append(time.perf_counter() - t0)

            # Expanding
            b_e = np.ascontiguousarray(
                x[:, current_size - expansion_size : current_size], dtype=np.float32
            )
            t0 = time.perf_counter()
            exp_ext.update(b_e)
            expanding_step_timings.setdefault(step, []).append(time.perf_counter() - t0)
            step += 1

    print("\nAveraged Results Per-Series:")
    print(
        f"{'Step':>4} | {'Size':>5} | {'Static (ms)':>12} | {'Expanding (ms)':>15} | {'Speedup':>8} | {'Samples':>8}"
    )
    print("-" * 65)
    for step in sorted(static_step_timings.keys()):
        sz = initial_size + step * expansion_size
        avg_s = np.mean(static_step_timings[step]) * 1000
        avg_e = np.mean(expanding_step_timings[step]) * 1000
        sp = avg_s / avg_e if avg_e > 0 else 0
        print(
            f"{step:4d} | {sz:5d} | {avg_s:12.4f} | {avg_e:15.4f} | {sp:7.2f}x | {len(static_step_timings[step]):8d}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TSRocket Expanding Window Real/Synthetic Benchmark"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="prev-data/Dataset/Intrinsic data",
        help="Path to real dataset",
    )
    parser.add_argument(
        "--mode", type=str, choices=["batched", "per-series", "both"], default="batched"
    )
    args = parser.parse_args()

    data = load_data(args.data_dir)
    print(f"Loaded {len(data)} total series.")

    if args.mode in ("batched", "both"):
        run_batched_benchmark(data)
    if args.mode in ("per-series", "both"):
        run_per_series_benchmark(data)
