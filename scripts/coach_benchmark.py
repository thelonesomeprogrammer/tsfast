import time
import datetime
import subprocess
import os
import numpy as np
import pandas as pd
import pyarrow as pa
import tsfast
import tsfel
from tsfresh.feature_extraction import extract_features
from tabulate import tabulate
import warnings

warnings.filterwarnings("ignore")

FEATURE_MAPPING = {
    "mean": {"tsfresh": {"mean": None}, "tsfel": ("statistical", "Mean")},
    "variance": {"tsfresh": {"variance": None}, "tsfel": ("statistical", "Variance")},
    "std_dev": {"tsfresh": {"standard_deviation": None}, "tsfel": ("statistical", "Standard deviation")},
    "min_value": {"tsfresh": {"minimum": None}, "tsfel": ("statistical", "Min")},
    "max_value": {"tsfresh": {"maximum": None}, "tsfel": ("statistical", "Max")},
    "median": {"tsfresh": {"median": None}, "tsfel": ("statistical", "Median")},
    "skewness": {"tsfresh": {"skewness": None}, "tsfel": ("statistical", "Skewness")},
    "kurtosis": {"tsfresh": {"kurtosis": None}, "tsfel": None},
    "biased_fisher_kurtosis": {"tsfresh": None, "tsfel": ("statistical", "Kurtosis")},
    "abs_max": {"tsfresh": {"absolute_maximum": None}, "tsfel": None},
    "first_loc_max": {"tsfresh": {"first_location_of_maximum": None}, "tsfel": None},
    "last_loc_max": {"tsfresh": {"last_location_of_maximum": None}, "tsfel": None},
    "first_loc_min": {"tsfresh": {"first_location_of_minimum": None}, "tsfel": None},
    "last_loc_min": {"tsfresh": {"last_location_of_minimum": None}, "tsfel": None},
    "autocorr-1": {"tsfresh": {"autocorrelation": [{"lag": 1}]}, "tsfel": None},
    "autocorrelation": {"tsfresh": None, "tsfel": ("temporal", "Autocorrelation")},
    "mean_abs_change": {"tsfresh": {"mean_abs_change": None}, "tsfel": None},
    "mean_change": {"tsfresh": {"mean_change": None}, "tsfel": ("statistical", "Mean diff")},
    "zero_crossing_rate": {"tsfresh": None, "tsfel": ("temporal", "Zero crossing rate")},
    "energy": {"tsfresh": {"abs_energy": None}, "tsfel": ("statistical", "Absolute energy")},
    "rms": {"tsfresh": {"root_mean_square": None}, "tsfel": ("statistical", "Root mean square")},
    "total_sum": {"tsfresh": {"sum_values": None}, "tsfel": ("statistical", "Sum")},
    "iqr": {"tsfresh": None, "tsfel": ("statistical", "Interquartile range")},
    "mad": {"tsfresh": None, "tsfel": ("statistical", "Mean absolute deviation")},
    "auc": {"tsfresh": None, "tsfel": ("statistical", "Area under the curve")},
    "count_above_mean": {"tsfresh": {"count_above_mean": None}, "tsfel": None},
    "count_below_mean": {"tsfresh": {"count_below_mean": None}, "tsfel": None},
    "longest_strike_above_mean": {"tsfresh": {"longest_strike_above_mean": None}, "tsfel": None},
    "longest_strike_below_mean": {"tsfresh": {"longest_strike_below_mean": None}, "tsfel": None},
    "variation_coefficient": {"tsfresh": {"variation_coefficient": None}, "tsfel": None},
    "quantile-0.5": {"tsfresh": {"quantile": [{"q": 0.5}]}, "tsfel": None},
    "quantile-0.1": {"tsfresh": {"quantile": [{"q": 0.1}]}, "tsfel": None},
    "quantile-0.9": {"tsfresh": {"quantile": [{"q": 0.9}]}, "tsfel": None},
    "fft_coeff-1-abs": {"tsfresh": {"fft_coefficient": [{"attr": "abs", "coeff": 1}]}, "tsfel": None},
    "fft_coeff-1-real": {"tsfresh": {"fft_coefficient": [{"attr": "real", "coeff": 1}]}, "tsfel": None},
    "fft_coeff-1-imag": {"tsfresh": {"fft_coefficient": [{"attr": "imag", "coeff": 1}]}, "tsfel": None},
    "fft_coeff-1-angle": {"tsfresh": {"fft_coefficient": [{"attr": "angle", "coeff": 1}]}, "tsfel": None},
    "cid_ce": {"tsfresh": {"cid_ce": [{"normalize": False}]}, "tsfel": None},
    "c3-5": {"tsfresh": {"c3": [{"lag": 5}]}, "tsfel": None},
    "benford_correlation": {"tsfresh": {"benford_correlation": None}, "tsfel": None},
    "abs_sum_change": {"tsfresh": {"absolute_sum_of_changes": None}, "tsfel": None},
    "mean_n_absolute_max-5": {"tsfresh": {"mean_n_absolute_max": [{"number_of_maxima": 5}]}, "tsfel": None},
    "peak_count": {"tsfresh": {"number_peaks": [{"n": 1}]}, "tsfel": None},
    "mean_second_derivative_central": {"tsfresh": {"mean_second_derivative_central": None}, "tsfel": None},
    "large_standard_deviation-0.05": {"tsfresh": {"large_standard_deviation": [{"r": 0.05}]}, "tsfel": None},
    "symmetry_looking-0.05": {"tsfresh": {"symmetry_looking": [{"r": 0.05}]}, "tsfel": None}
}

def get_tsfresh_params(features):
    params = {}
    for f in features:
        p = FEATURE_MAPPING[f]["tsfresh"]
        if p:
            for k, v in p.items():
                if k not in params:
                    params[k] = v
                else:
                    if v is not None:
                        params[k].extend(v)
    return params

def get_tsfel_cfg(features):
    cfg = {}
    for f in features:
        item = FEATURE_MAPPING[f]["tsfel"]
        if item:
            domain, name = item
            if domain not in cfg:
                cfg[domain] = {}
            full_cfg = tsfel.get_features_by_domain(domain)
            if name in full_cfg[domain]:
                cfg[domain][name] = full_cfg[domain][name]
    return cfg

def get_git_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()[:7]
    except:
        return 'unknown'

def append_to_csv(filepath, new_rows):
    header = "Date,CommitHash,Benchmark_Name,Metric_Value,Unit,Delta_From_Last"
    if not os.path.exists(filepath):
        with open(filepath, 'w') as f:
            f.write(header + '\n')

    df = pd.read_csv(filepath)
    date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    commit_hash = get_git_hash()

    for row in new_rows:
        bench_name, metric_val, unit = row
        # Calculate Delta_From_Last
        past_runs = df[df['Benchmark_Name'] == bench_name]
        delta = 0.0
        if not past_runs.empty:
            last_val = past_runs.iloc[-1]['Metric_Value']
            if last_val != 0:
                delta = ((metric_val - last_val) / last_val) * 100.0

        new_df = pd.DataFrame([{
            'Date': date_str,
            'CommitHash': commit_hash,
            'Benchmark_Name': bench_name,
            'Metric_Value': metric_val,
            'Unit': unit,
            'Delta_From_Last': round(delta, 2)
        }])
        df = pd.concat([df, new_df], ignore_index=True)

    df.to_csv(filepath, index=False)

def main():
    n_samples = 40
    n_points = 1000
    window_size = 200
    chunk_size = 50
    np.random.seed(42)
    data = np.random.randn(n_samples, n_points).astype(np.float32)

    features = list(FEATURE_MAPPING.keys())

    # tsfast
    start = time.time()
    extractor = tsfast.SlidingExtractor(features, n_samples, window_size, stride=chunk_size)
    num_windows_tsfast = 0
    for j in range(0, n_points, chunk_size):
        chunk = data[:, j:j+chunk_size]
        batch = pa.RecordBatch.from_arrays([pa.array(chunk[i]) for i in range(n_samples)], names=[f"c{i}" for i in range(n_samples)])
        res = extractor.update(batch)
        if res is not None and res.num_rows > 0:
            num_windows_tsfast += 1
    tsfast_time = time.time() - start
    tsfast_ms_per_window = (tsfast_time * 1000) / max(1, num_windows_tsfast)

    # tsfresh
    tsfresh_features = [f for f in features if FEATURE_MAPPING[f]["tsfresh"]]
    tsfresh_params = get_tsfresh_params(tsfresh_features)
    start = time.time()
    num_windows_tsfresh = 0
    for j in range(window_size, n_points + 1, chunk_size):
        window_data = data[:, j-window_size:j]
        df_list = [pd.DataFrame({"id": i, "v": window_data[i]}) for i in range(n_samples)]
        full_df = pd.concat(df_list)
        tsfresh_df = extract_features(full_df, column_id="id", default_fc_parameters=tsfresh_params, n_jobs=1, disable_progressbar=True)
        num_windows_tsfresh += 1
    tsfresh_time = time.time() - start
    tsfresh_ms_per_window = (tsfresh_time * 1000) / max(1, num_windows_tsfresh)

    # tsfel
    tsfel_features = [f for f in features if FEATURE_MAPPING[f]["tsfel"]]
    tsfel_cfg = get_tsfel_cfg(tsfel_features)
    start = time.time()
    num_windows_tsfel = 0
    for j in range(window_size, n_points + 1, chunk_size):
        window_data = data[:, j-window_size:j]
        tsfel_res_list = []
        for i in range(n_samples):
            res = tsfel.time_series_features_extractor(tsfel_cfg, window_data[i], fs=100, verbose=0)
            tsfel_res_list.append(res)
        tsfel_res_df = pd.concat(tsfel_res_list, ignore_index=True)
        num_windows_tsfel += 1
    tsfel_time = time.time() - start
    tsfel_ms_per_window = (tsfel_time * 1000) / max(1, num_windows_tsfel)

    rows = [
        ("tsfast_ms_per_window", tsfast_ms_per_window, "ms"),
        ("tsfresh_ms_per_window", tsfresh_ms_per_window, "ms"),
        ("tsfel_ms_per_window", tsfel_ms_per_window, "ms"),
        ("tsfast_compatible_features", len(features), "count"),
        ("tsfresh_compatible_features", len(tsfresh_features), "count"),
        ("tsfel_compatible_features", len(tsfel_features), "count")
    ]

    append_to_csv('.jules/benchmarks.csv', rows)
    print("Benchmark complete and saved to .jules/benchmarks.csv")

if __name__ == "__main__":
    main()
