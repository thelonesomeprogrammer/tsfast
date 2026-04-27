import tsfast
import numpy as np
import pyarrow as pa
import time
import pandas as pd
from tsfresh.feature_extraction import extract_features, EfficientFCParameters
import tsfel

def benchmark():
    # 1. Prepare data: 100 samples, each 1000 points
    n_samples = 100
    n_points = 1000
    np.random.seed(42)
    data = np.random.randn(n_samples, n_points).astype(np.float32)
    
    # --- TSFRESH FEATURES ---
    tsfresh_features = ["maximum", "autocorrelation__lag_1", "last_location_of_maximum", "first_location_of_maximum", "absolute_maximum"]
    tsfast_fresh_equiv = ["max_value", "autocorr-1", "last_loc_max", "first_loc_max", "abs_max"]
    
    # --- TSFEL FEATURES ---
    tsfel_cfg = tsfel.get_features_by_domain("statistical")
    # We'll pick a few from statistical domain that match
    tsfel_match = ["Max", "Min", "Mean", "Variance", "Skewness", "Kurtosis"]
    tsfast_tsfel_equiv = ["max_value", "min_value", "mean", "variance", "skewness", "kurtosis"]

    print(f"Benchmarking {n_samples} samples of {n_points} points each...")

    # --- TSFAST (Fresh Subset) ---
    start = time.time()
    extractor_fresh = tsfast.Extractor(tsfast_fresh_equiv)
    tsfast_fresh_results = []
    for i in range(n_samples):
        batch = pa.RecordBatch.from_arrays([pa.array(data[i])], names=["v"])
        res = extractor_fresh.process_2d_floats(batch)
        tsfast_fresh_results.append([res.column(f)[0].as_py() for f in tsfast_fresh_equiv])
    tsfast_fresh_time = time.time() - start
    tsfast_fresh_results = np.array(tsfast_fresh_results)

    # --- TSFRESH ---
    df_list = []
    for i in range(n_samples):
        df_list.append(pd.DataFrame({"id": i, "time": np.arange(n_points), "v": data[i]}))
    full_df = pd.concat(df_list)
    fc_params = {"maximum": None, "autocorrelation": [{"lag": 1}], "last_location_of_maximum": None, "first_location_of_maximum": None, "absolute_maximum": None}
    start = time.time()
    tsfresh_res_df = extract_features(full_df, column_id="id", column_sort="time", default_fc_parameters=fc_params, n_jobs=1, disable_progressbar=True)
    tsfresh_time = time.time() - start
    tsfresh_cols = ["v__maximum", "v__autocorrelation__lag_1", "v__last_location_of_maximum", "v__first_location_of_maximum", "v__absolute_maximum"]
    tsfresh_results = tsfresh_res_df[tsfresh_cols].values

    # --- TSFEL ---
    start = time.time()
    tsfel_results_all = []
    for i in range(n_samples):
        res = tsfel.time_series_features_extractor(tsfel_cfg, data[i], fs=100, verbose=0)
        tsfel_results_all.append(res)
    tsfel_time = time.time() - start
    tsfel_res_df = pd.concat(tsfel_results_all)
    
    # Extract matched columns (TSEL names them like '0_Max')
    tsfel_results = tsfel_res_df[["0_Max", "0_Min", "0_Mean", "0_Variance", "0_Skewness", "0_Kurtosis"]].values

    # --- TSFAST (TSFEL Subset) ---
    start = time.time()
    extractor_tsfel = tsfast.Extractor(tsfast_tsfel_equiv)
    tsfast_tsfel_results = []
    for i in range(n_samples):
        batch = pa.RecordBatch.from_arrays([pa.array(data[i])], names=["v"])
        res = extractor_tsfel.process_2d_floats(batch)
        tsfast_tsfel_results.append([res.column(f)[0].as_py() for f in tsfast_tsfel_equiv])
    tsfast_tsfel_time = time.time() - start
    tsfast_tsfel_results = np.array(tsfast_tsfel_results)

    print("\n--- PERFORMANCE ---")
    print(f"TSFAST  (tsfresh-subset): {tsfast_fresh_time:.4f}s")
    print(f"TSFRESH (tsfresh-subset): {tsfresh_time:.4f}s (x{tsfresh_time/tsfast_fresh_time:.1f} slower)")
    print(f"TSFAST  (tsfel-subset):   {tsfast_tsfel_time:.4f}s")
    print(f"TSFEL   (tsfel-subset):   {tsfel_time:.4f}s (x{tsfel_time/tsfast_tsfel_time:.1f} slower)")

    print("\n--- BEHAVIORAL PARITY (TSFAST vs TSFRESH) ---")
    for idx, (f_fast, f_fresh) in enumerate(zip(tsfast_fresh_equiv, tsfresh_cols)):
        diff = np.abs(tsfast_fresh_results[:, idx] - tsfresh_results[:, idx])
        print(f"{f_fast} vs {f_fresh}: Max Diff = {np.max(diff):.6f}, Avg Diff = {np.mean(diff):.6f}")

    print("\n--- BEHAVIORAL PARITY (TSFAST vs TSFEL) ---")
    tsfel_cols = ["0_Max", "0_Min", "0_Mean", "0_Variance", "0_Skewness", "0_Kurtosis"]
    for idx, (f_fast, f_tsfel) in enumerate(zip(tsfast_tsfel_equiv, tsfel_cols)):
        diff = np.abs(tsfast_tsfel_results[:, idx] - tsfel_results[:, idx])
        print(f"{f_fast} vs {f_tsfel}: Max Diff = {np.max(diff):.6f}, Avg Diff = {np.mean(diff):.6f}")

if __name__ == "__main__":
    benchmark()
