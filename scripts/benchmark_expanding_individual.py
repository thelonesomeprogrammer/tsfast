import os
import time
import numpy as np
import pandas as pd
import tsfast
import pyarrow as pa
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

DATA_DIR = "prev-data/Dataset/Intrinsic data"
CATEGORIES = ["N", "NS", "OT", "UT"]
SIGNAL_COL = [2, 3]
COLNAMES = ["Time (ms)", "Nset (1/min)", "Torque (Nm)", "Current (V)", "Angle (deg)", "Depth (mm)"]

def load_data():
    dfs = []
    print(f"Loading FULL dataset from {DATA_DIR}...")
    for cat in CATEGORIES:
        cat_dir = os.path.join(DATA_DIR, cat)
        if not os.path.exists(cat_dir):
            continue
        files = [f for f in os.listdir(cat_dir) if f.endswith(".csv")]
        print(f"  {cat}: loading {len(files)} files")
        for f in files:
            df = pd.read_csv(os.path.join(cat_dir, f))
            # (length, 2) -> (2, length)
            df = df.iloc[:, SIGNAL_COL].values.astype(np.float32).T
            dfs.append(df)
    return dfs

def benchmark_per_series():
    X = load_data()
    total_samples = len(X)
    print(f"Total samples: {total_samples}")

    features = [
        "total_sum", "mean", "variance", "std", "min", "max",
        "energy", "rms", "zero_crossing_rate", "peak_count",
        "mean_abs_change", "mean_change", "cid_ce", "auc"
    ]
    
    initial_size = 500
    expansion_size = 100
    max_len = max(x.shape[1] for x in X)
    
    # Store timings per step: step_index -> list of times
    static_step_timings = {}
    expanding_step_timings = {}
    
    static_ext = tsfast.Extractor(features)
    names = [COLNAMES[i] for i in SIGNAL_COL]

    print(f"\nBenchmarking {total_samples} series individually...")
    
    for s_idx, x in enumerate(X):
        if x.shape[1] < initial_size:
            continue
            
        # --- Expanding Extractor for this series ---
        exp_ext = tsfast.ExpandingExtractor(features, len(SIGNAL_COL))
        
        # Step 0: Initial 500 points
        # Static
        batch_static = pa.RecordBatch.from_arrays([pa.array(x[i, :initial_size]) for i in range(x.shape[0])], names=names)
        start = time.perf_counter()
        static_ext.process_2d_floats(batch_static)
        t_static = time.perf_counter() - start
        static_step_timings.setdefault(0, []).append(t_static)
        
        # Expanding
        batch_exp = pa.RecordBatch.from_arrays([pa.array(x[i, :initial_size]) for i in range(x.shape[0])], names=names)
        start = time.perf_counter()
        exp_ext.update(batch_exp)
        t_exp = time.perf_counter() - start
        expanding_step_timings.setdefault(0, []).append(t_exp)
        
        # Subsequent steps
        current_size = initial_size
        step = 1
        while current_size + expansion_size <= x.shape[1]:
            current_size += expansion_size
            
            # Static: process full prefix
            batch_static = pa.RecordBatch.from_arrays([pa.array(x[i, :current_size]) for i in range(x.shape[0])], names=names)
            start = time.perf_counter()
            static_ext.process_2d_floats(batch_static)
            t_static = time.perf_counter() - start
            static_step_timings.setdefault(step, []).append(t_static)
            
            # Expanding: process only increment
            batch_exp = pa.RecordBatch.from_arrays([pa.array(x[i, current_size-expansion_size:current_size]) for i in range(x.shape[0])], names=names)
            start = time.perf_counter()
            exp_ext.update(batch_exp)
            t_exp = time.perf_counter() - start
            expanding_step_timings.setdefault(step, []).append(t_exp)
            
            step += 1
            
        if (s_idx + 1) % 100 == 0:
            print(f"  Processed {s_idx + 1}/{total_samples} series...")

    print("\nFinal Averaged Results (per series per step):")
    print(f"{'Step':>4} | {'Size':>5} | {'Static (ms)':>12} | {'Expanding (ms)':>15} | {'Speedup':>8} | {'Samples':>8}")
    print("-" * 65)
    
    sorted_steps = sorted(static_step_timings.keys())
    for step in sorted_steps:
        size = initial_size + step * expansion_size
        avg_static = np.mean(static_step_timings[step]) * 1000 # to ms
        avg_expanding = np.mean(expanding_step_timings[step]) * 1000 # to ms
        samples = len(static_step_timings[step])
        speedup = avg_static / avg_expanding if avg_expanding > 0 else 0
        print(f"{step:4d} | {size:5d} | {avg_static:12.4f} | {avg_expanding:15.4f} | {speedup:7.2f}x | {samples:8d}")

if __name__ == "__main__":
    benchmark_per_series()
