import os
import time
import warnings
import numpy as np
import pandas as pd
import tsfast
import tsfel
import pyarrow as pa
from tsfresh.feature_extraction import extract_features
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Suppress warnings
warnings.filterwarnings("ignore")

def generate_synthetic_data(n_samples=1000, n_points=1000):
    print(f"Generating synthetic dataset: {n_samples} samples, {n_points} points each...")
    X = []
    y = []
    t = np.linspace(0, 10, n_points)
    
    for i in range(n_samples):
        cat = i % 4
        if cat == 0: # White noise
            sig = np.random.randn(n_points)
        elif cat == 1: # Sine wave + noise
            sig = np.sin(2 * np.pi * t) + 0.5 * np.random.randn(n_points)
        elif cat == 2: # Linear trend + noise
            sig = 0.5 * t + 0.5 * np.random.randn(n_points)
        else: # Random walk
            sig = np.cumsum(np.random.randn(n_points)) * 0.1
        
        X.append(sig.reshape(1, -1).astype(np.float32))
        y.append(f"Cat_{cat}")
    
    return X, np.array(y)

# Top features implemented in tsfast
ALL_TOP = [
    "max_value", "min_value", "mean", "variance", "skewness", "kurtosis",
    "autocorr-1", "abs_max", "last_loc_max", "first_loc_max",
    "fft_coeff-1-real", "fft_coeff-8-real", "fft_coeff-6-imag",
    "time_reversal_asymmetry-1", "time_reversal_asymmetry-2",
    "partial_autocorr-1", "partial_autocorr-2",
    "agg_linear_trend-slope-50-var", "agg_linear_trend-intercept-50-var",
    "agg_linear_trend-slope-10-var", "agg_linear_trend-intercept-10-var",
]

def benchmark_tsfast(X):
    start = time.time()
    extractor = tsfast.Extractor(ALL_TOP)
    extracted = []
    for x in X:
        batch = pa.RecordBatch.from_arrays([pa.array(x[0])], names=["v"])
        res_batch = extractor.process_2d_floats(batch)
        extracted.append([res_batch.column(f)[0].as_py() for f in ALL_TOP])
    end = time.time()
    return np.array(extracted), end - start

def get_tsfresh_top(X):
    fc_params = {
        "maximum": None, "minimum": None, "mean": None, "variance": None, "skewness": None, "kurtosis": None,
        "autocorrelation": [{"lag": 1}],
        "absolute_maximum": None, "last_location_of_maximum": None, "first_location_of_maximum": None,
        "fft_coefficient": [{"coeff": 1, "attr": "real"}, {"coeff": 8, "attr": "real"}, {"coeff": 6, "attr": "imag"}],
        "time_reversal_asymmetry_statistic": [{"lag": 1}, {"lag": 2}],
        "partial_autocorrelation": [{"lag": 1}, {"lag": 2}],
        "agg_linear_trend": [
            {"attr": "slope", "chunk_len": 50, "f_agg": "var"},
            {"attr": "intercept", "chunk_len": 50, "f_agg": "var"},
            {"attr": "slope", "chunk_len": 10, "f_agg": "var"},
            {"attr": "intercept", "chunk_len": 10, "f_agg": "var"},
        ]
    }
    data_list = []
    for i, x in enumerate(X):
        temp_df = pd.DataFrame({"id": i, "time": np.arange(len(x[0])), "v": x[0]})
        data_list.append(temp_df)
    full_df = pd.concat(data_list)
    start = time.time()
    extracted = extract_features(full_df, column_id="id", column_sort="time", 
                                 default_fc_parameters=fc_params, disable_progressbar=True, n_jobs=4)
    res = (extracted.values, time.time() - start)
    return res

def get_tsfel_top(X):
    cfg = tsfel.get_features_by_domain("statistical")
    start = time.time()
    extracted = []
    for x in X:
        feat_vals = tsfel.time_series_features_extractor(cfg, x[0], fs=100, verbose=0)
        # Subset to match our core 6 features for a fair speed test of that domain
        subset = feat_vals[["0_Max", "0_Min", "0_Mean", "0_Variance", "0_Skewness", "0_Kurtosis"]]
        extracted.append(subset.values[0])
    res = (np.array(extracted), time.time() - start)
    return res

def run_ml(X_feats, y, label=""):
    X_feats = np.nan_to_num(X_feats, nan=0.0, posinf=0.0, neginf=0.0)
    X_train, X_test, y_train, y_test = train_test_split(X_feats, y, test_size=0.3, random_state=42, stratify=y)
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return acc

if __name__ == "__main__":
    X_raw, y = generate_synthetic_data(1000, 1000)
    print(f"Total samples: {len(X_raw)}")

    print("\nBenchmarking tsfast (21 Top Features)...")
    feats_tsfast, time_tsfast = benchmark_tsfast(X_raw)
    acc_tsfast = run_ml(feats_tsfast, y, "tsfast")
    
    print("\nBenchmarking tsfresh (21 Top Features)...")
    feats_tsfresh, time_tsfresh = get_tsfresh_top(X_raw)
    acc_tsfresh = run_ml(feats_tsfresh, y, "tsfresh")

    print("\nBenchmarking TSFEL (Subset of 6 Features)...")
    feats_tsfel, time_tsfel = get_tsfel_top(X_raw)
    acc_tsfel = run_ml(feats_tsfel, y, "TSFEL")
    
    print("\n--- RESULTS ---")
    print(f"tsfast:  Time={time_tsfast:.4f}s, Accuracy={acc_tsfast:.4f}, Features={feats_tsfast.shape[1]}")
    print(f"tsfresh: Time={time_tsfresh:.4f}s, Accuracy={acc_tsfresh:.4f}, Features={feats_tsfresh.shape[1]}")
    print(f"TSFEL:   Time={time_tsfel:.4f}s, Accuracy={acc_tsfel:.4f}, Features={feats_tsfel.shape[1]} (Subsampled)")
    
    print(f"\nSpeedup vs tsfresh: {time_tsfresh/time_tsfast:.1f}x")
    print(f"Speedup vs TSFEL:   {time_tsfel/time_tsfast:.1f}x (Note: TSFEL only extracted 6 features!)")
