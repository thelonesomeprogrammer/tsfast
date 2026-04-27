import time
import numpy as np
import pyarrow as pa
import tsfast
import argparse

def run_benchmark(n_cols=10, total_len=10000, initial_size=500, increment_size=100):
    # Generate synthetic data
    data = np.random.randn(n_cols, total_len).astype(np.float32)
    col_names = [f"col_{i}" for i in range(n_cols)]
    
    features = [
        "total_sum", "mean", "variance", "std", "min", "max", "median",
        "skewness", "kurtosis", "mad", "iqr", "entropy",
        "energy", "rms", "zero_crossing_rate", "peak_count",
        "mean_abs_change", "mean_change", "cid_ce", "auc",
        "abs_sum_change", "count_above_mean", "count_below_mean",
        "longest_strike_above_mean", "longest_strike_below_mean",
        "abs_max", "first_loc_max", "last_loc_max", "first_loc_min", "last_loc_min"
    ]
    
    print(f"Benchmarking ExpandingExtractor with {n_cols} columns and {total_len} total points.")
    print(f"Features: {len(features)}")
    
    exp_ext = tsfast.ExpandingExtractor(features, n_cols)
    
    # Initial batch
    batch = pa.RecordBatch.from_arrays([pa.array(data[i, :initial_size]) for i in range(n_cols)], names=col_names)
    start = time.perf_counter()
    exp_ext.update(batch)
    end = time.perf_counter()
    print(f"Initial batch ({initial_size} points) took {(end-start)*1000:.4f} ms")
    
    timings = []
    current_size = initial_size
    while current_size + increment_size <= total_len:
        batch = pa.RecordBatch.from_arrays(
            [pa.array(data[i, current_size:current_size+increment_size]) for i in range(n_cols)], 
            names=col_names
        )
        start = time.perf_counter()
        exp_ext.update(batch)
        end = time.perf_counter()
        timings.append((end - start) * 1000)
        current_size += increment_size
        
    print(f"Average incremental update ({increment_size} points) took {np.mean(timings):.4f} ms")
    print(f"Total time for all incremental updates: {np.sum(timings):.4f} ms")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_cols", type=int, default=10)
    parser.add_argument("--total_len", type=int, default=10000)
    args = parser.parse_args()
    run_benchmark(n_cols=args.n_cols, total_len=args.total_len)
