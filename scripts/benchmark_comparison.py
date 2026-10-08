import time
import warnings
import numpy as np
import pandas as pd
import tsrocket
import tsfel
from tsfresh.feature_extraction import extract_features
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

# 21 Top features matching tsfresh
ALL_TOP = [
    "max_value",
    "min_value",
    "mean",
    "variance",
    "skewness",
    "kurtosis",
    "autocorr-1",
    "abs_max",
    "last_loc_max",
    "first_loc_max",
    "fft_coeff-1-real",
    "fft_coeff-8-real",
    "fft_coeff-6-imag",
    "time_reversal_asymmetry-1",
    "time_reversal_asymmetry-2",
    "partial_autocorr-1",
    "partial_autocorr-2",
    "agg_linear_trend-slope-50-var",
    "agg_linear_trend-intercept-50-var",
    "agg_linear_trend-slope-10-var",
    "agg_linear_trend-intercept-10-var",
]


def generate_synthetic_data(n_samples=1000, n_points=1000):
    print(
        f"Generating synthetic dataset: {n_samples} samples, {n_points} points each..."
    )
    X = []
    y = []
    t = np.linspace(0, 10, n_points)

    for i in range(n_samples):
        cat = i % 4
        if cat == 0:  # White noise
            sig = np.random.randn(n_points)
        elif cat == 1:  # Sine wave + noise
            sig = np.sin(2 * np.pi * t) + 0.5 * np.random.randn(n_points)
        elif cat == 2:  # Linear trend + noise
            sig = 0.5 * t + 0.5 * np.random.randn(n_points)
        else:  # Random walk
            sig = np.cumsum(np.random.randn(n_points)) * 0.1

        X.append(sig.astype(np.float32))
        y.append(f"Cat_{cat}")

    return np.array(X), np.array(y)


def benchmark_tsrocket_batched(X):
    """Idiomatic production mode: batch of N series passed to Rust with Rayon parallelism."""
    n_samples, n_points = X.shape
    extractor = tsrocket.Extractor(ALL_TOP)

    # One series per row
    batch = np.ascontiguousarray(X, dtype=np.float32)

    # Warmup
    _ = extractor.process_2d_floats(batch)

    # Timed extraction
    t0 = time.perf_counter()
    extracted = extractor.process_2d_floats(batch)  # columns in ALL_TOP order
    extraction_time = time.perf_counter() - t0
    return extracted, extraction_time


def benchmark_tsrocket_iterative(X):
    """Iterative mode: processing each series individually."""
    n_samples = len(X)
    extractor = tsrocket.Extractor(ALL_TOP)

    # Pre-build single-series batches to measure extractor throughput
    batches = [
        np.ascontiguousarray(X[i : i + 1], dtype=np.float32) for i in range(n_samples)
    ]

    t0 = time.perf_counter()
    results = [extractor.process_2d_floats(b) for b in batches]
    extraction_time = time.perf_counter() - t0

    extracted = np.concatenate(results)
    return extracted, extraction_time


def benchmark_tsfresh(X):
    fc_params = {
        "maximum": None,
        "minimum": None,
        "mean": None,
        "variance": None,
        "skewness": None,
        "kurtosis": None,
        "autocorrelation": [{"lag": 1}],
        "absolute_maximum": None,
        "last_location_of_maximum": None,
        "first_location_of_maximum": None,
        "fft_coefficient": [
            {"coeff": 1, "attr": "real"},
            {"coeff": 8, "attr": "real"},
            {"coeff": 6, "attr": "imag"},
        ],
        "time_reversal_asymmetry_statistic": [{"lag": 1}, {"lag": 2}],
        "partial_autocorrelation": [{"lag": 1}, {"lag": 2}],
        "agg_linear_trend": [
            {"attr": "slope", "chunk_len": 50, "f_agg": "var"},
            {"attr": "intercept", "chunk_len": 50, "f_agg": "var"},
            {"attr": "slope", "chunk_len": 10, "f_agg": "var"},
            {"attr": "intercept", "chunk_len": 10, "f_agg": "var"},
        ],
    }
    data_list = []
    n_samples, n_points = X.shape
    for i in range(n_samples):
        temp_df = pd.DataFrame({"id": i, "time": np.arange(n_points), "v": X[i]})
        data_list.append(temp_df)
    full_df = pd.concat(data_list, ignore_index=True)

    t0 = time.perf_counter()
    extracted = extract_features(
        full_df,
        column_id="id",
        column_sort="time",
        default_fc_parameters=fc_params,
        disable_progressbar=True,
        n_jobs=4,
    )
    extraction_time = time.perf_counter() - t0
    return extracted.values, extraction_time


def benchmark_tsfel(X):
    cfg = tsfel.get_features_by_domain("statistical")
    matched_keys = [
        "0_Max",
        "0_Min",
        "0_Mean",
        "0_Variance",
        "0_Skewness",
        "0_Kurtosis",
    ]

    t0 = time.perf_counter()
    extracted = []
    for i in range(len(X)):
        feat_vals = tsfel.time_series_features_extractor(cfg, X[i], fs=100, verbose=0)
        subset = feat_vals[matched_keys]
        extracted.append(subset.values[0])
    extraction_time = time.perf_counter() - t0
    return np.array(extracted), extraction_time


def evaluate_ml(X_feats, y, label=""):
    X_feats = np.nan_to_num(X_feats, nan=0.0, posinf=0.0, neginf=0.0)
    X_train, X_test, y_train, y_test = train_test_split(
        X_feats, y, test_size=0.3, random_state=42, stratify=y
    )
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    return accuracy_score(y_test, y_pred)


if __name__ == "__main__":
    X_raw, y = generate_synthetic_data(1000, 1000)
    print(
        f"Dataset Shape: {X_raw.shape} ({len(X_raw)} series of {X_raw.shape[1]} points)"
    )

    print("\n1. Benchmarking tsrocket (Batched, Rayon parallel, 21 Top Features)...")
    feats_tsrocket_batch, time_tsrocket_batch = benchmark_tsrocket_batched(X_raw)
    acc_tsrocket_batch = evaluate_ml(feats_tsrocket_batch, y, "tsrocket-batched")

    print("\n2. Benchmarking tsrocket (Iterative single-batch, 21 Top Features)...")
    feats_tsrocket_iter, time_tsrocket_iter = benchmark_tsrocket_iterative(X_raw)

    print("\n3. Benchmarking tsfresh (n_jobs=4, 21 Matching Features)...")
    feats_tsfresh, time_tsfresh = benchmark_tsfresh(X_raw)
    acc_tsfresh = evaluate_ml(feats_tsfresh, y, "tsfresh")

    print("\n4. Benchmarking TSFEL (Statistical subset, 6 Features)...")
    feats_tsfel, time_tsfel = benchmark_tsfel(X_raw)
    acc_tsfel = evaluate_ml(feats_tsfel, y, "TSFEL")

    print("\n" + "=" * 78)
    print(
        f"{'Library / Mode':<30} | {'Features':<8} | {'Time (s)':<10} | {'Accuracy':<8} | {'Speedup':<8}"
    )
    print("-" * 78)
    print(
        f"{'tsrocket (Batched, Rayon)':<30} | {feats_tsrocket_batch.shape[1]:<8} | {time_tsrocket_batch:<10.4f} | {acc_tsrocket_batch:<8.4f} | {'baseline'}"
    )
    print(
        f"{'tsrocket (Iterative)':<30} | {feats_tsrocket_iter.shape[1]:<8} | {time_tsrocket_iter:<10.4f} | {acc_tsrocket_batch:<8.4f} | {time_tsrocket_iter / time_tsrocket_batch:.1f}x slower"
    )
    print(
        f"{'tsfresh (n_jobs=4)':<30} | {feats_tsfresh.shape[1]:<8} | {time_tsfresh:<10.4f} | {acc_tsfresh:<8.4f} | {time_tsfresh / time_tsrocket_batch:.1f}x slower"
    )
    print(
        f"{'TSFEL (subsampled 6 feats)':<30} | {feats_tsfel.shape[1]:<8} | {time_tsfel:<10.4f} | {acc_tsfel:<8.4f} | {time_tsfel / time_tsrocket_batch:.1f}x slower"
    )
    print("=" * 78)
