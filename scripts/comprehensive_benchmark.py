import tsfast
import numpy as np
import pandas as pd
import pyarrow as pa
import time
from tsfresh.feature_extraction import extract_features
import tsfel
from tabulate import tabulate
import warnings

warnings.filterwarnings("ignore")

# Feature mappings
# format: (tsfast_name, tsfresh_params, tsfel_cfg_item)
# tsfel_cfg_item: (domain, feature_name)
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
    "peak_count": {"tsfresh": {"number_peaks": [{"n": 1}]}, "tsfel": ("statistical", "Peak count")},
}

def get_tsfresh_params(features):
    params = {}
    for f in features:
        p = FEATURE_MAPPING[f]["tsfresh"]
        if p:
            for k, v in p.items():
                if k in params:
                    if v:
                        if isinstance(params[k], list):
                            params[k].extend(v)
                        else:
                            params[k] = [params[k], *v]
                else:
                    params[k] = v
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

def benchmark_static(data, features):
    n_samples, n_points = data.shape
    start = time.time()
    extractor = tsfast.Extractor(features)
    tsfast_res = []
    for i in range(n_samples):
        batch = pa.RecordBatch.from_arrays([pa.array(data[i])], names=["v"])
        res = extractor.process_2d_floats(batch)
        tsfast_res.append([res.column(f)[0].as_py() for f in features])
    tsfast_time = time.time() - start
    tsfast_res = np.array(tsfast_res)

    tsfresh_features = [f for f in features if FEATURE_MAPPING[f]["tsfresh"]]
    tsfresh_params = get_tsfresh_params(tsfresh_features)
    df_list = [pd.DataFrame({"id": i, "v": data[i]}) for i in range(n_samples)]
    full_df = pd.concat(df_list)
    start = time.time()
    tsfresh_df = extract_features(full_df, column_id="id", default_fc_parameters=tsfresh_params, n_jobs=1, disable_progressbar=True)
    tsfresh_time = time.time() - start
    
    tsfel_features = [f for f in features if FEATURE_MAPPING[f]["tsfel"]]
    tsfel_cfg = get_tsfel_cfg(tsfel_features)
    start = time.time()
    tsfel_res_list = []
    for i in range(n_samples):
        res = tsfel.time_series_features_extractor(tsfel_cfg, data[i], fs=100, verbose=0)
        tsfel_res_list.append(res)
    tsfel_time = time.time() - start
    tsfel_res_df = pd.concat(tsfel_res_list, ignore_index=True)

    return {
        "tsfast": (tsfast_res, tsfast_time),
        "tsfresh": (tsfresh_df, tsfresh_time, tsfresh_features),
        "tsfel": (tsfel_res_df, tsfel_time, tsfel_features)
    }

def benchmark_expanding(data, features):
    n_samples, n_points = data.shape
    chunk_size = 50
    start = time.time()
    extractor = tsfast.ExpandingExtractor(features, n_samples)
    for j in range(0, n_points, chunk_size):
        chunk = data[:, j:j+chunk_size]
        batch = pa.RecordBatch.from_arrays([pa.array(chunk[i]) for i in range(n_samples)], names=[f"c{i}" for i in range(n_samples)])
        res = extractor.update(batch)
    tsfast_time = time.time() - start
    tsfast_final = np.array([[res.column(f)[i].as_py() for f in features] for i in range(n_samples)])

    tsfresh_features = [f for f in features if FEATURE_MAPPING[f]["tsfresh"]]
    tsfresh_params = get_tsfresh_params(tsfresh_features)
    start = time.time()
    tsfresh_df = None
    for j in range(chunk_size, n_points + 1, chunk_size):
        current_data = data[:, :j]
        df_list = [pd.DataFrame({"id": i, "v": current_data[i]}) for i in range(n_samples)]
        full_df = pd.concat(df_list)
        tsfresh_df = extract_features(full_df, column_id="id", default_fc_parameters=tsfresh_params, n_jobs=1, disable_progressbar=True)
    tsfresh_time = time.time() - start

    tsfel_features = [f for f in features if FEATURE_MAPPING[f]["tsfel"]]
    tsfel_cfg = get_tsfel_cfg(tsfel_features)
    start = time.time()
    tsfel_res_df = None
    for j in range(chunk_size, n_points + 1, chunk_size):
        current_data = data[:, :j]
        tsfel_res_list = []
        for i in range(n_samples):
            res = tsfel.time_series_features_extractor(tsfel_cfg, current_data[i], fs=100, verbose=0)
            tsfel_res_list.append(res)
        tsfel_res_df = pd.concat(tsfel_res_list, ignore_index=True)
    tsfel_time = time.time() - start

    return {
        "tsfast": (tsfast_final, tsfast_time),
        "tsfresh": (tsfresh_df, tsfresh_time, tsfresh_features),
        "tsfel": (tsfel_res_df, tsfel_time, tsfel_features)
    }

def benchmark_sliding(data, features, window_size):
    n_samples, n_points = data.shape
    chunk_size = 50
    start = time.time()
    extractor = tsfast.SlidingExtractor(features, n_samples, window_size, stride=chunk_size)
    res = None
    for j in range(0, n_points, chunk_size):
        chunk = data[:, j:j+chunk_size]
        batch = pa.RecordBatch.from_arrays([pa.array(chunk[i]) for i in range(n_samples)], names=[f"c{i}" for i in range(n_samples)])
        res = extractor.update(batch)
    tsfast_time = time.time() - start
    
    # Each update now returns exactly 1 result per column because stride == chunk_size
    # and we started from empty. Wait, the first window comes at window_size.
    # The subsequent updates come every stride.
    
    # If res is None or empty (e.g. n_points < window_size), handle it
    if res is None or res.num_rows == 0:
        tsfast_final = np.zeros((n_samples, len(features)))
    else:
        # res has n_samples rows, one per column, representing the LAST window.
        tsfast_final = np.array([[res.column(f)[i].as_py() for f in features] for i in range(n_samples)])

    tsfresh_features = [f for f in features if FEATURE_MAPPING[f]["tsfresh"]]
    tsfresh_params = get_tsfresh_params(tsfresh_features)
    start = time.time()
    tsfresh_df = None
    for j in range(window_size, n_points + 1, chunk_size):
        window_data = data[:, j-window_size:j]
        df_list = [pd.DataFrame({"id": i, "v": window_data[i]}) for i in range(n_samples)]
        full_df = pd.concat(df_list)
        tsfresh_df = extract_features(full_df, column_id="id", default_fc_parameters=tsfresh_params, n_jobs=1, disable_progressbar=True)
    tsfresh_time = time.time() - start

    tsfel_features = [f for f in features if FEATURE_MAPPING[f]["tsfel"]]
    tsfel_cfg = get_tsfel_cfg(tsfel_features)
    start = time.time()
    tsfel_res_df = None
    for j in range(window_size, n_points + 1, chunk_size):
        window_data = data[:, j-window_size:j]
        tsfel_res_list = []
        for i in range(n_samples):
            res = tsfel.time_series_features_extractor(tsfel_cfg, window_data[i], fs=100, verbose=0)
            tsfel_res_list.append(res)
        tsfel_res_df = pd.concat(tsfel_res_list, ignore_index=True)
    tsfel_time = time.time() - start

    return {
        "tsfast": (tsfast_final, tsfast_time),
        "tsfresh": (tsfresh_df, tsfresh_time, tsfresh_features),
        "tsfel": (tsfel_res_df, tsfel_time, tsfel_features)
    }

def compare_results(features, tsfast_res, other_res, other_features, other_name, n_points):
    diffs = {}
    mapping = {
        "mean": "v__mean", "variance": "v__variance", "std_dev": "v__standard_deviation",
        "min_value": "v__minimum", "max_value": "v__maximum", "median": "v__median",
        "skewness": "v__skewness", "kurtosis": "v__kurtosis", "abs_max": "v__absolute_maximum",
        "first_loc_max": "v__first_location_of_maximum", "last_loc_max": "v__last_location_of_maximum",
        "first_loc_min": "v__first_location_of_minimum", "last_loc_min": "v__last_location_of_minimum",
        "autocorr-1": "v__autocorrelation__lag_1", "mean_abs_change": "v__mean_abs_change",
        "mean_change": "v__mean_change", "energy": "v__abs_energy", "rms": "v__root_mean_square",
        "total_sum": "v__sum_values", "count_above_mean": "v__count_above_mean",
        "count_below_mean": "v__count_below_mean", "longest_strike_above_mean": "v__longest_strike_above_mean",
        "longest_strike_below_mean": "v__longest_strike_below_mean", "variation_coefficient": "v__variation_coefficient",
        "quantile-0.5": 'v__quantile__q_0.5', "quantile-0.1": 'v__quantile__q_0.1', "quantile-0.9": 'v__quantile__q_0.9',
        "fft_coeff-1-abs": 'v__fft_coefficient__attr_"abs"__coeff_1',
        "fft_coeff-1-real": 'v__fft_coefficient__attr_"real"__coeff_1',
        "fft_coeff-1-imag": 'v__fft_coefficient__attr_"imag"__coeff_1',
        "fft_coeff-1-angle": 'v__fft_coefficient__attr_"angle"__coeff_1',
        "cid_ce": "v__cid_ce__normalize_False", "c3-5": "v__c3__lag_5",
        "benford_correlation": "v__benford_correlation", "abs_sum_change": "v__absolute_sum_of_changes",
        "mean_n_absolute_max-5": "v__mean_n_absolute_max__number_of_maxima_5",
        "peak_count": 'v__number_peaks__n_1'
    }
    
    for i, f in enumerate(features):
        if f in other_features:
            vals = None
            if other_name == "tsfresh":
                target_col = mapping.get(f)
                if target_col and target_col in other_res.columns:
                    vals = other_res.sort_index()[target_col].values
                    if f == "variance": vals = vals * (n_points / (n_points - 1))
                    elif f == "std_dev": vals = vals * np.sqrt(n_points / (n_points - 1))
            else: # tsfel
                item = FEATURE_MAPPING[f]["tsfel"]
                if item:
                    _, tsfel_name = item
                    for col in other_res.columns:
                        if col.split("_", 1)[1] == tsfel_name:
                            vals = other_res[col].values
                            if f == "variance": vals = vals * (n_points / (n_points - 1))
                            elif f == "zero_crossing_rate" and np.any(vals > 1.0): vals = vals / n_points
                            break
            
            if vals is not None:
                mask = ~np.isnan(tsfast_res[:, i]) & ~np.isnan(vals)
                if np.any(mask):
                    abs_diff = np.abs(tsfast_res[mask, i] - vals[mask])
                    denom = np.abs(vals[mask])
                    perc_diff = np.where(denom > 1e-6, (abs_diff / denom) * 100, abs_diff * 100)
                    diffs[f] = np.max(perc_diff)
                else: diffs[f] = np.nan
    return diffs

def main():
    n_samples = 40
    n_points = 1000
    window_size = 200
    np.random.seed(42)
    data = np.random.randn(n_samples, n_points).astype(np.float32)
    features = list(FEATURE_MAPPING.keys())

    print(f"Comprehensive Benchmark: {n_samples} samples, {n_points} points, window={window_size}")
    
    modes = ["Static", "Expanding", "Sliding"]
    summary_data = []
    perf_data = []

    for mode in modes:
        print(f"\n--- {mode} Mode ---")
        if mode == "Static": results = benchmark_static(data, features); curr_n = n_points
        elif mode == "Expanding": results = benchmark_expanding(data, features); curr_n = n_points
        else: results = benchmark_sliding(data, features, window_size); curr_n = window_size
        
        tf_res, tf_time = results["tsfast"]
        fresh_df, fresh_time, fresh_feats = results["tsfresh"]
        fel_df, fel_time, fel_feats = results["tsfel"]
        
        fresh_diffs = compare_results(features, tf_res, fresh_df, fresh_feats, "tsfresh", curr_n)
        fel_diffs = compare_results(features, tf_res, fel_df, fel_feats, "tsfel", curr_n)
        
        for f in features:
            f_diff = fresh_diffs.get(f, "-")
            l_diff = fel_diffs.get(f, "-")
            f_diff_str = f"{int(f_diff)}%" if isinstance(f_diff, (float, np.float64, np.float32)) else str(f_diff)
            l_diff_str = f"{int(l_diff)}%" if isinstance(l_diff, (float, np.float64, np.float32)) else str(l_diff)
            summary_data.append([mode, f, f_diff_str, l_diff_str])
        
        perf_data.append([mode, f"{tf_time:.4f}s", f"{fresh_time:.4f}s", f"{fel_time:.4f}s", f"{fresh_time/tf_time:.1f}x", f"{fel_time/tf_time:.1f}x"])

    print("\n--- BEHAVIORAL COMPARISON (Max % Difference) ---")
    print(tabulate(summary_data, headers=["Mode", "Feature", "vs tsfresh", "vs tsfel"], tablefmt="github"))
    print("\n--- PERFORMANCE COMPARISON ---")
    print(tabulate(perf_data, headers=["Mode", "tsfast", "tsfresh", "tsfel", "Speedup fresh", "Speedup fel"], tablefmt="github"))

if __name__ == "__main__":
    main()
