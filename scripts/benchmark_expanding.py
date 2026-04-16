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

def benchmark():
    X = load_data()
    total_samples = len(X)
    print(f"Total samples: {total_samples}")

    features = [
        "total_sum", "mean", "variance", "std", "min", "max",
        "energy", "rms", "zero_crossing_rate", "peak_count",
        "mean_abs_change", "mean_change", "cid_ce", "auc"
    ]
    
    # We'll benchmark a subset of samples to avoid taking too long if needed,
    # but the user said "full dataset".
    
    initial_size = 500
    expansion_size = 100
    
    # Expand to see the maximum length
    max_len = max(x.shape[1] for x in X)
    print(f"Max series length: {max_len}")
    
    # Static benchmarking
    static_times = []
    static_ext = tsfast.Extractor(features)
    
    # Expanding benchmarking
    expanding_times = []
    # We need one ExpandingExtractor per series if we process them in parallel,
    # or we can reset/re-create. But ExpandingExtractor is stateful.
    # To benchmark "update" performance, we'll process all series together in batches if possible,
    # but the current ExpandingExtractor state is per-column.
    
    # Actually, ExpandingExtractor.update(batch) processes 'batch' and updates its internal state.
    # If we have 1000 series, we can think of them as 2000 columns (if 2 signals per series).
    # But wait, update() returns features for the WHOLE series seen so far.
    
    print("\nStarting Benchmark...")
    
    # Step 0: Initial 500 points
    # Static
    start = time.time()
    for x in X:
        if x.shape[1] < initial_size: continue
        batch = pa.RecordBatch.from_arrays([pa.array(x[i, :initial_size]) for i in range(x.shape[0])], names=[COLNAMES[i] for i in SIGNAL_COL])
        static_ext.process_2d_floats(batch)
    static_times.append(time.time() - start)
    
    # Expanding
    start = time.time()
    # We'll use one extractor that handles all columns of all series? 
    # That would be a lot of columns. 
    # Let's say we have N series, each with 2 signals. That's 2*N columns.
    n_total_cols = total_samples * len(SIGNAL_COL)
    exp_ext = tsfast.ExpandingExtractor(features, n_total_cols)
    
    # Flatten all series into one big batch for the first 500 points
    arrays = []
    names = []
    for s_idx, x in enumerate(X):
        if x.shape[1] < initial_size: continue
        for i in range(x.shape[0]):
            arrays.append(pa.array(x[i, :initial_size]))
            names.append(f"s{s_idx}_c{i}")
    
    batch = pa.RecordBatch.from_arrays(arrays, names=names)
    exp_ext.update(batch)
    expanding_times.append(time.time() - start)
    
    print(f"Step 0 (size {initial_size}): Static={static_times[-1]:.4f}s, Expanding={expanding_times[-1]:.4f}s")

    # Subsequent steps: +100 points
    step = 1
    current_size = initial_size
    while True:
        current_size += expansion_size
        if current_size > max_len:
            break
            
        # Static: re-process full prefix
        start = time.time()
        count = 0
        for x in X:
            if x.shape[1] < current_size: continue
            batch = pa.RecordBatch.from_arrays([pa.array(x[i, :current_size]) for i in range(x.shape[0])], names=[COLNAMES[i] for i in SIGNAL_COL])
            static_ext.process_2d_floats(batch)
            count += 1
        if count == 0: break
        static_times.append(time.time() - start)
        
        # Expanding: process only the new 100 points
        start = time.time()
        arrays = []
        names = []
        for s_idx, x in enumerate(X):
            if x.shape[1] < current_size: continue
            for i in range(x.shape[0]):
                arrays.append(pa.array(x[i, current_size-expansion_size:current_size]))
                names.append(f"s{s_idx}_c{i}")
        batch = pa.RecordBatch.from_arrays(arrays, names=names)
        exp_ext.update(batch)
        expanding_times.append(time.time() - start)
        
        print(f"Step {step} (size {current_size}): Static={static_times[-1]:.4f}s, Expanding={expanding_times[-1]:.4f}s, Samples={count}")
        step += 1
        
        # Stop early if it takes too long or we reached a reasonable limit
        # The user said "0, 1..10..rest"
        # If I do first 10 steps, then maybe every 10? 
        # Let's just do all of them for now if it's fast.

    print("\nFinal Results:")
    print("Step | Size | Static (s) | Expanding (s) | Speedup")
    print("-" * 50)
    for i in range(len(static_times)):
        size = initial_size + i * expansion_size
        speedup = static_times[i] / expanding_times[i]
        print(f"{i:4d} | {size:4d} | {static_times[i]:10.4f} | {expanding_times[i]:13.4f} | {speedup:7.2f}x")

if __name__ == "__main__":
    benchmark()
