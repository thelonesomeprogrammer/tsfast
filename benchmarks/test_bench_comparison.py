import pytest
import numpy as np
import pandas as pd
import tsfast
import tsfresh
from tsfresh.feature_extraction import extract_features
import tsfel

COMMON_FEATURES_TSFAST = [
    "mean",
    "variance",
    "std_dev",
    "min_value",
    "max_value",
    "skewness",
    "kurtosis",
    "energy",
    "rms",
    "total_sum",
]

TSFRESH_PARAMS = {
    "mean": None,
    "variance": None,
    "standard_deviation": None,
    "minimum": None,
    "maximum": None,
    "skewness": None,
    "kurtosis": None,
    "abs_energy": None,
    "root_mean_square": None,
    "sum_values": None,
}


@pytest.fixture(scope="module")
def benchmark_data_50():
    np.random.seed(42)
    n_samples = 50
    n_points = 500
    return np.random.randn(n_samples, n_points).astype(np.float32)


@pytest.fixture(scope="module")
def tsfast_input(benchmark_data_50):
    return benchmark_data_50  # (n_samples, n_points): one series per row


@pytest.fixture(scope="module")
def tsfresh_dataframe(benchmark_data_50):
    n_samples, n_points = benchmark_data_50.shape
    df_list = []
    for i in range(n_samples):
        df_list.append(
            pd.DataFrame(
                {"id": i, "time": np.arange(n_points), "v": benchmark_data_50[i]}
            )
        )
    return pd.concat(df_list, ignore_index=True)


@pytest.fixture(scope="module")
def tsfel_cfg():
    cfg = tsfel.get_features_by_domain("statistical")
    # Keep only the matched features
    matched = [
        "Mean",
        "Variance",
        "Standard deviation",
        "Min",
        "Max",
        "Skewness",
        "Kurtosis",
        "Absolute energy",
        "Root mean square",
        "Sum",
    ]
    filtered_cfg = {
        "statistical": {
            k: cfg["statistical"][k] for k in matched if k in cfg["statistical"]
        }
    }
    return filtered_cfg


@pytest.mark.benchmark(group="cross_library_comparison")
def test_bench_compare_tsfast(benchmark, tsfast_input):
    extractor = tsfast.Extractor(COMMON_FEATURES_TSFAST)
    # Warmup
    _ = extractor.process_2d_floats(tsfast_input)
    benchmark(extractor.process_2d_floats, tsfast_input)


@pytest.mark.benchmark(group="cross_library_comparison")
def test_bench_compare_tsfresh(benchmark, tsfresh_dataframe):
    def run_tsfresh():
        return extract_features(
            tsfresh_dataframe,
            column_id="id",
            column_sort="time",
            default_fc_parameters=TSFRESH_PARAMS,
            disable_progressbar=True,
            n_jobs=1,
        )

    benchmark(run_tsfresh)


@pytest.mark.benchmark(group="cross_library_comparison")
def test_bench_compare_tsfel(benchmark, benchmark_data_50, tsfel_cfg):
    def run_tsfel():
        results = []
        for i in range(len(benchmark_data_50)):
            res = tsfel.time_series_features_extractor(
                tsfel_cfg, benchmark_data_50[i], fs=100, verbose=0
            )
            results.append(res)
        return results

    benchmark(run_tsfel)
