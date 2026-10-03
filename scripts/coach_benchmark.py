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
import matplotlib.pyplot as plt
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
    import csv
    header = ["Date", "CommitHash", "Benchmark_Name", "Metric_Value", "Unit", "Delta_From_Last"]
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    file_exists = os.path.exists(filepath)

    if file_exists:
        df = pd.read_csv(filepath)
    else:
        df = pd.DataFrame(columns=header)

    date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    commit_hash = get_git_hash()

    with open(filepath, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(header)

        for row in new_rows:
            bench_name, metric_val, unit = row
            # Calculate Delta_From_Last
            past_runs = df[df['Benchmark_Name'] == bench_name]
            delta = 0.0
            if not past_runs.empty:
                last_val = past_runs.iloc[-1]['Metric_Value']
                if last_val != 0:
                    delta = ((metric_val - last_val) / last_val) * 100.0

            writer.writerow([date_str, commit_hash, bench_name, metric_val, unit, round(delta, 2)])

def make_batch(data_slice):
    n_samples = data_slice.shape[0]
    return pa.RecordBatch.from_arrays(
        [pa.array(data_slice[i]) for i in range(n_samples)], 
        names=[f"c{i}" for i in range(n_samples)]
    )

def generate_chart():
    csv_path = '.jules/benchmarks.csv'
    chart_path = '.jules/benchmark_trends.png'

    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        print("Error: DataFrame is empty.")
        return

    df['Date'] = pd.to_datetime(df['Date'])

    tsfast_sliding = df[df['Benchmark_Name'] == 'tsfast_sliding_avg_next_50']['Metric_Value'].values
    tsfresh_sliding = df[df['Benchmark_Name'] == 'tsfresh_sliding_avg_next_50']['Metric_Value'].values
    tsfel_sliding = df[df['Benchmark_Name'] == 'tsfel_sliding_avg_next_50']['Metric_Value'].values

    tsfast_exp = df[df['Benchmark_Name'] == 'tsfast_expanding_avg_next_50']['Metric_Value'].values
    tsfresh_exp = df[df['Benchmark_Name'] == 'tsfresh_expanding_avg_next_50']['Metric_Value'].values
    tsfel_exp = df[df['Benchmark_Name'] == 'tsfel_expanding_avg_next_50']['Metric_Value'].values

    tsfast_feats = df[df['Benchmark_Name'] == 'tsfast_compatible_features']['Metric_Value'].values
    tsfresh_feats = df[df['Benchmark_Name'] == 'tsfresh_compatible_features']['Metric_Value'].values
    tsfel_feats = df[df['Benchmark_Name'] == 'tsfel_compatible_features']['Metric_Value'].values

    runs = range(1, len(tsfast_sliding) + 1)

    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        pass  # Fallback if style not available
        
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    color_fast = 'tab:blue'
    color_fresh = 'tab:green'
    color_fel = 'tab:red'

    # Top Plot: Execution Time
    ax1.set_title("Benchmark Execution Time Trends", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Time per window (ms) [Log Scale]", fontsize=12)
    
    if len(tsfast_sliding) > 0:
        ax1.plot(runs, tsfast_sliding, color=color_fast, linestyle='-', label='tsfast Sliding', marker='o')
    if len(tsfresh_sliding) > 0:
        ax1.plot(runs, tsfresh_sliding, color=color_fresh, linestyle='-', label='tsfresh Sliding', marker='s')
    if len(tsfel_sliding) > 0:
        ax1.plot(runs, tsfel_sliding, color=color_fel, linestyle='-', label='tsfel Sliding', marker='^')
        
    if len(tsfast_exp) > 0:
        ax1.plot(runs, tsfast_exp, color=color_fast, linestyle='--', label='tsfast Expanding', marker='o', alpha=0.7)
    if len(tsfresh_exp) > 0:
        ax1.plot(runs, tsfresh_exp, color=color_fresh, linestyle='--', label='tsfresh Expanding', marker='s', alpha=0.7)
    if len(tsfel_exp) > 0:
        ax1.plot(runs, tsfel_exp, color=color_fel, linestyle='--', label='tsfel Expanding', marker='^', alpha=0.7)

    ax1.set_yscale('log')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax1.grid(True, which="both", ls="--", alpha=0.5)

    # Bottom Plot: Features
    ax2.set_title("Compatible Feature Count Trends", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Benchmark Run", fontsize=12)
    ax2.set_ylabel("Number of Features", fontsize=12)
    
    if len(tsfast_feats) > 0:
        ax2.plot(runs, tsfast_feats, color=color_fast, linestyle='-', label='tsfast', marker='o')
    if len(tsfresh_feats) > 0:
        ax2.plot(runs, tsfresh_feats, color=color_fresh, linestyle='-', label='tsfresh', marker='s')
    if len(tsfel_feats) > 0:
        ax2.plot(runs, tsfel_feats, color=color_fel, linestyle='-', label='tsfel', marker='^')

    max_feats = max(max(tsfast_feats, default=1), max(tsfresh_feats, default=1), max(tsfel_feats, default=1))
    ax2.set_ylim(0, max_feats + 10)
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax2.grid(True, ls="--", alpha=0.5)

    fig.tight_layout()
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    print(f"Chart saved to {chart_path}")


def update_readme():
    df = pd.read_csv('.jules/benchmarks.csv')
    latest_date = df['Date'].max()
    latest_df = df[df['Date'] == latest_date].copy()

    # Add emojis
    def get_emoji(val, unit):
        if unit == "count":
            return "⚪"

        # Check if lower is better (time, memory size)
        lower_is_better = unit in ["ms", "ns/iter", "MB"]

        if val > 5.0:
            return "🔴" if lower_is_better else "🟢"
        elif val < -5.0:
            return "🟢" if lower_is_better else "🔴"
        return "⚪"

    latest_df['Direction'] = latest_df.apply(lambda row: get_emoji(row['Delta_From_Last'], row['Unit']), axis=1)

    md_table = latest_df[['Date', 'CommitHash', 'Benchmark_Name', 'Metric_Value', 'Unit', 'Delta_From_Last', 'Direction']].to_markdown(index=False)

    try:
        with open('README.md', 'r') as f:
            readme_content = f.read()
    except FileNotFoundError:
        readme_content = "## Latest Results\n"

    start_index = readme_content.find('## Latest Results')
    if start_index != -1:
        new_content = readme_content[:start_index] + "## Latest Results\n" + md_table + "\n"
        with open('README.md', 'w') as f:
            f.write(new_content)
        print("README.md updated.")
    else:
        print("Could not find '## Latest Results' in README.md")

def main():
    n_samples = 40
    first_window = 100
    chunk_size = 50
    num_windows = 50
    n_points = first_window + chunk_size * num_windows
    np.random.seed(42)
    data = np.random.randn(n_samples, n_points).astype(np.float32)

    features = list(FEATURE_MAPPING.keys())

    rows = []

    def record_metrics(prefix, times):
        if len(times) > 0:
            rows.append((f"{prefix}_first_window", times[0] * 1000, "ms"))
        if len(times) > 1:
            avg_time = sum(times[1:]) / len(times[1:])
            rows.append((f"{prefix}_avg_next_{num_windows}", avg_time * 1000, "ms"))

    # 1. tsfast sliding (stateful)
    extractor = tsfast.SlidingExtractor(features, n_samples, first_window, stride=chunk_size)
    times_tsfast_sliding = []
    accumulated_time = 0.0
    for j in range(0, n_points, chunk_size):
        chunk = data[:, j:j+chunk_size]
        batch = make_batch(chunk)
        start = time.time()
        res = extractor.update(batch)
        elapsed = time.time() - start
        if res is not None and res.num_rows > 0:
            times_tsfast_sliding.append(accumulated_time + elapsed)
            accumulated_time = 0.0
        else:
            accumulated_time += elapsed
    record_metrics("tsfast_sliding", times_tsfast_sliding)

    # 2. tsfast expanding (stateful)
    extractor = tsfast.ExpandingExtractor(features, n_samples)
    times_tsfast_expanding = []
    # first window
    chunk = data[:, :first_window]
    batch = make_batch(chunk)
    start = time.time()
    res = extractor.update(batch)
    times_tsfast_expanding.append(time.time() - start)
    # next windows
    for j in range(first_window, n_points, chunk_size):
        chunk = data[:, j:j+chunk_size]
        batch = make_batch(chunk)
        start = time.time()
        res = extractor.update(batch)
        times_tsfast_expanding.append(time.time() - start)
    record_metrics("tsfast_expanding", times_tsfast_expanding)

    # 3. tsfast static sliding
    extractor = tsfast.Extractor(features)
    times_tsfast_static_sliding = []
    for j in range(first_window, n_points + 1, chunk_size):
        chunk = data[:, j-first_window:j]
        batch = make_batch(chunk)
        start = time.time()
        res = extractor.process_2d_floats(batch)
        times_tsfast_static_sliding.append(time.time() - start)
    record_metrics("tsfast_static_sliding", times_tsfast_static_sliding)

    # 4. tsfast static expanding
    extractor = tsfast.Extractor(features)
    times_tsfast_static_expanding = []
    for j in range(first_window, n_points + 1, chunk_size):
        chunk = data[:, :j]
        batch = make_batch(chunk)
        start = time.time()
        res = extractor.process_2d_floats(batch)
        times_tsfast_static_expanding.append(time.time() - start)
    record_metrics("tsfast_static_expanding", times_tsfast_static_expanding)

    # tsfresh
    tsfresh_features = [f for f in features if FEATURE_MAPPING[f]["tsfresh"]]
    tsfresh_params = get_tsfresh_params(tsfresh_features)
    
    # 5. tsfresh sliding
    times_tsfresh_sliding = []
    for j in range(first_window, n_points + 1, chunk_size):
        window_data = data[:, j-first_window:j]
        df_list = [pd.DataFrame({"id": i, "v": window_data[i]}) for i in range(n_samples)]
        full_df = pd.concat(df_list)
        start = time.time()
        _ = extract_features(full_df, column_id="id", default_fc_parameters=tsfresh_params, n_jobs=1, disable_progressbar=True)
        times_tsfresh_sliding.append(time.time() - start)
    record_metrics("tsfresh_sliding", times_tsfresh_sliding)

    # 6. tsfresh expanding
    times_tsfresh_expanding = []
    for j in range(first_window, n_points + 1, chunk_size):
        window_data = data[:, :j]
        df_list = [pd.DataFrame({"id": i, "v": window_data[i]}) for i in range(n_samples)]
        full_df = pd.concat(df_list)
        start = time.time()
        _ = extract_features(full_df, column_id="id", default_fc_parameters=tsfresh_params, n_jobs=1, disable_progressbar=True)
        times_tsfresh_expanding.append(time.time() - start)
    record_metrics("tsfresh_expanding", times_tsfresh_expanding)

    # tsfel
    tsfel_features = [f for f in features if FEATURE_MAPPING[f]["tsfel"]]
    tsfel_cfg = get_tsfel_cfg(tsfel_features)
    
    # 7. tsfel sliding
    times_tsfel_sliding = []
    for j in range(first_window, n_points + 1, chunk_size):
        window_data = data[:, j-first_window:j]
        start = time.time()
        for i in range(n_samples):
            _ = tsfel.time_series_features_extractor(tsfel_cfg, window_data[i], fs=100, verbose=0)
        times_tsfel_sliding.append(time.time() - start)
    record_metrics("tsfel_sliding", times_tsfel_sliding)

    # 8. tsfel expanding
    times_tsfel_expanding = []
    for j in range(first_window, n_points + 1, chunk_size):
        window_data = data[:, :j]
        start = time.time()
        for i in range(n_samples):
            _ = tsfel.time_series_features_extractor(tsfel_cfg, window_data[i], fs=100, verbose=0)
        times_tsfel_expanding.append(time.time() - start)
    record_metrics("tsfel_expanding", times_tsfel_expanding)

    rows.extend([
        ("tsfast_compatible_features", len(features), "count"),
        ("tsfresh_compatible_features", len(tsfresh_features), "count"),
        ("tsfel_compatible_features", len(tsfel_features), "count")
    ])

    append_to_csv('.jules/benchmarks.csv', rows)
    print("Benchmark complete and saved to .jules/benchmarks.csv")

    generate_chart()
    update_readme()

if __name__ == "__main__":
    main()
