import os
import time
import warnings
import numpy as np
import pandas as pd
import tsfast
import tsfel
import pyarrow as pa
from tsfresh.feature_extraction import extract_features
from pathlib import Path
from typing import List

# Suppress warnings
warnings.filterwarnings("ignore")

def load_column_as_list(folder_path: str, column_name: str, file_pattern: str = "*.csv") -> List[np.ndarray]:
    folder = Path(folder_path)
    csv_files = list(folder.glob(file_pattern))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found matching pattern '{file_pattern}' in {folder_path}")

    result = []
    for csv_file in sorted(csv_files):
        try:
            df = pd.read_csv(csv_file)
            if column_name not in df.columns:
                continue
            column_data = df[column_name].values.astype(np.float32)
            result.append(column_data)
        except Exception as e:
            print(f"Error reading {csv_file.name}: {e}")
            continue
    return result

# Mapping tsfresh features to tsfast
TSFRESH_TO_TSFAST = {
    "value__maximum": "max_value",
    "value__partial_autocorrelation__lag_2": "partial_autocorr-2",
    "value__fft_coefficient__attr_\"real\"__coeff_8": "fft_coeff-8-real",
    "value__agg_linear_trend__attr_\"intercept\"__chunk_len_50__f_agg_\"var\"": "agg_linear_trend-intercept-50-var",
    "value__last_location_of_maximum": "last_loc_max",
    "value__fft_coefficient__attr_\"real\"__coeff_1": "fft_coeff-1-real",
    "value__agg_linear_trend__attr_\"stderr\"__chunk_len_10__f_agg_\"var\"": "agg_linear_trend-stderr-10-var",
    "value__autocorrelation__lag_1": "autocorr-1",
    "value__agg_linear_trend__attr_\"slope\"__chunk_len_50__f_agg_\"var\"": "agg_linear_trend-slope-50-var",
    "value__time_reversal_asymmetry_statistic__lag_1": "time_reversal_asymmetry-1",
    "value__first_location_of_maximum": "first_loc_max",
    "value__agg_linear_trend__attr_\"slope\"__chunk_len_10__f_agg_\"var\"": "agg_linear_trend-slope-10-var",
    "value__partial_autocorrelation__lag_1": "partial_autocorr-1",
    "value__time_reversal_asymmetry_statistic__lag_2": "time_reversal_asymmetry-2",
    "value__agg_linear_trend__attr_\"stderr\"__chunk_len_50__f_agg_\"var\"": "agg_linear_trend-stderr-50-var",
    "value__fft_coefficient__attr_\"imag\"__coeff_6": "fft_coeff-6-imag",
    "value__approximate_entropy__m_2__r_0.7": "approx_entropy-2-0.7",
    "value__agg_linear_trend__attr_\"intercept\"__chunk_len_10__f_agg_\"var\"": "agg_linear_trend-intercept-10-var",
    "value__absolute_maximum": "abs_max",
    "value__time_reversal_asymmetry_statistic__lag_3": "time_reversal_asymmetry-3",
}

# TSFRESH parameters for these features
TSFRESH_PARAMS = {
    "maximum": None,
    "partial_autocorrelation": [{"lag": 1}, {"lag": 2}],
    "fft_coefficient": [{"coeff": 1, "attr": "real"}, {"coeff": 8, "attr": "real"}, {"coeff": 6, "attr": "imag"}],
    "agg_linear_trend": [
        {"attr": "intercept", "chunk_len": 50, "f_agg": "var"},
        {"attr": "stderr", "chunk_len": 10, "f_agg": "var"},
        {"attr": "slope", "chunk_len": 50, "f_agg": "var"},
        {"attr": "slope", "chunk_len": 10, "f_agg": "var"},
        {"attr": "stderr", "chunk_len": 50, "f_agg": "var"},
        {"attr": "intercept", "chunk_len": 10, "f_agg": "var"},
    ],
    "last_location_of_maximum": None,
    "autocorrelation": [{"lag": 1}],
    "time_reversal_asymmetry_statistic": [{"lag": 1}, {"lag": 2}, {"lag": 3}],
    "first_location_of_maximum": None,
    "approximate_entropy": [{"m": 2, "r": 0.7}],
    "absolute_maximum": None,
}

# TSFEL top features (only mapping those supported by tsfast)
TSFEL_TO_TSFAST = {
    "torque_Max": "max_value",
    "torque_Skewness": "skewness",
    "torque_Min": "min_value",
    "torque_Kurtosis": "kurtosis",
    "torque_Variance": "variance",
}

def benchmark_tsfast(X, features, batch=True):
    start = time.time()
    extractor = tsfast.Extractor(features)
    
    if batch:
        # Pad all series to the maximum length
        max_len = max(len(x) for x in X)
        padded_X = [np.pad(x, (0, max_len - len(x)), mode='constant') for x in X]
        
        # Create a single RecordBatch with one column per series
        arrays = [pa.array(x) for x in padded_X]
        names = [f"v_{i}" for i in range(len(X))]
        rb = pa.RecordBatch.from_arrays(arrays, names=names)
        
        res_batch = extractor.process_2d_floats(rb)
        
        # Extract results
        extracted = []
        for i in range(len(X)):
            extracted.append([res_batch.column(f)[i].as_py() for f in features])
    else:
        extracted = []
        for x in X:
            rb = pa.RecordBatch.from_arrays([pa.array(x)], names=["v"])
            res_batch = extractor.process_2d_floats(rb)
            extracted.append([res_batch.column(f)[0].as_py() for f in features])
            
    end = time.time()
    return np.array(extracted), end - start

def benchmark_tsfresh(X, params):
    data_list = []
    for i, x in enumerate(X):
        temp_df = pd.DataFrame({"id": i, "time": np.arange(len(x)), "v": x})
        data_list.append(temp_df)
    full_df = pd.concat(data_list)
    start = time.time()
    extracted = extract_features(full_df, column_id="id", column_sort="time", 
                                 default_fc_parameters=params, disable_progressbar=True, n_jobs=4)
    end = time.time()
    return extracted.values, end - start

def benchmark_tsfel(X):
    # For TSFEL, we use the statistical domain and then subset to match our mapped features
    cfg = tsfel.get_features_by_domain("statistical")
    start = time.time()
    extracted = []
    for x in X:
        feat_vals = tsfel.time_series_features_extractor(cfg, x, fs=100, verbose=0)
        # Match TSFEL_TO_TSFAST keys
        subset = feat_vals[["0_Max", "0_Skewness", "0_Min", "0_Kurtosis", "0_Variance"]]
        extracted.append(subset.values[0])
    end = time.time()
    return np.array(extracted), end - start

if __name__ == "__main__":
    folder_path = "prev-data/Dataset/Intrinsic data/N"
    column_name = "Torque (Nm)"
    
    print(f"Loading data from {folder_path}...")
    X = load_column_as_list(folder_path, column_name)
    print(f"Loaded {len(X)} series.")

    # 1. TSFRESH Top Features (WITH Approx Entropy)
    tsfresh_features = list(TSFRESH_TO_TSFAST.keys())
    tsfast_features_for_tsfresh = [TSFRESH_TO_TSFAST[f] for f in tsfresh_features]
    
    print(f"\n--- TSFRESH Top Features (20 features, WITH Approx Entropy) ---")
    print("Running tsfast (batched)...")
    feats_tsfast_1, time_tsfast_1 = benchmark_tsfast(X, tsfast_features_for_tsfresh, batch=True)
    print("Running tsfresh...")
    feats_tsfresh, time_tsfresh = benchmark_tsfresh(X, TSFRESH_PARAMS)
    
    print(f"tsfast time:  {time_tsfast_1:.4f}s")
    print(f"tsfresh time: {time_tsfresh:.4f}s")
    print(f"Speedup: {time_tsfresh/time_tsfast_1:.1f}x")

    # 2. TSFRESH Top Features (WITHOUT Approx Entropy)
    tsfresh_features_no_ae = [f for f in tsfresh_features if "approximate_entropy" not in f]
    tsfast_features_no_ae = [TSFRESH_TO_TSFAST[f] for f in tsfresh_features_no_ae]
    tsfresh_params_no_ae = TSFRESH_PARAMS.copy()
    del tsfresh_params_no_ae["approximate_entropy"]

    print(f"\n--- TSFRESH Top Features (19 features, WITHOUT Approx Entropy) ---")
    print("Running tsfast (batched)...")
    feats_tsfast_no_ae, time_tsfast_no_ae = benchmark_tsfast(X, tsfast_features_no_ae, batch=True)
    print("Running tsfresh...")
    feats_tsfresh_no_ae, time_tsfresh_no_ae = benchmark_tsfresh(X, tsfresh_params_no_ae)
    
    print(f"tsfast time:  {time_tsfast_no_ae:.4f}s")
    print(f"tsfresh time: {time_tsfresh_no_ae:.4f}s")
    print(f"Speedup: {time_tsfresh_no_ae/time_tsfast_no_ae:.1f}x")

    # 3. TSFEL Top Features
    tsfel_mapped_features = list(TSFEL_TO_TSFAST.values())
    
    print(f"\n--- TSFEL Top Features (5 features) ---")
    print("Running tsfast (batched)...")
    feats_tsfast_2, time_tsfast_2 = benchmark_tsfast(X, tsfel_mapped_features, batch=True)
    print("Running TSFEL...")
    feats_tsfel, time_tsfel = benchmark_tsfel(X)
    
    print(f"tsfast time: {time_tsfast_2:.4f}s")
    print(f"TSFEL time:  {time_tsfel:.4f}s")
    print(f"Speedup: {time_tsfel/time_tsfast_2:.1f}x")
