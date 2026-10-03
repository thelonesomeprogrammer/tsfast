import pytest
import numpy as np
import pyarrow as pa
import tsfast

@pytest.fixture(scope="session")
def basic_features():
    return ["mean", "variance", "std_dev", "min_value", "max_value", "total_sum", "energy", "rms"]

@pytest.fixture(scope="session")
def advanced_features():
    return [
        "mean", "variance", "std_dev", "min_value", "max_value",
        "skewness", "kurtosis", "autocorr-1", "abs_max", "last_loc_max", "first_loc_max",
        "fft_coeff-1-real", "fft_coeff-1-imag", "c3-5", "cid_ce", "mean_abs_change"
    ]

@pytest.fixture(scope="session")
def full_30_features():
    return [
        "total_sum", "mean", "variance", "std_dev", "min_value", "max_value", "median",
        "skewness", "kurtosis", "mad", "iqr", "entropy",
        "energy", "rms", "zero_crossing_rate", "peak_count",
        "mean_abs_change", "mean_change", "cid_ce", "auc",
        "pk_pk_distance", "zero_cross", "max_power_spectrum",
        "abs_sum_change", "count_above_mean", "count_below_mean",
        "longest_strike_above_mean", "longest_strike_below_mean",
        "abs_max", "first_loc_max", "last_loc_max", "first_loc_min", "last_loc_min",
        "length", "variance_larger_than_standard_deviation", "percentage_of_reoccurring_datapoints_to_all_datapoints", "percentage_of_reoccurring_values_to_all_values", "ratio_value_number_to_time_series_length",
        "spectral_centroid", "spectral_spread", "spectral_entropy"
    ]

@pytest.fixture(scope="session")
def single_series():
    np.random.seed(42)
    return np.random.randn(1000).astype(np.float32)

@pytest.fixture(scope="session")
def single_batch(single_series):
    return pa.RecordBatch.from_arrays([pa.array(single_series)], names=["c0"])

@pytest.fixture(scope="session")
def batch_100_series():
    np.random.seed(42)
    data = np.random.randn(1000, 100).astype(np.float32)
    arrays = [pa.array(data[:, i]) for i in range(100)]
    names = [f"c_{i}" for i in range(100)]
    return pa.RecordBatch.from_arrays(arrays, names=names)

@pytest.fixture(scope="session")
def batch_1000_series():
    np.random.seed(42)
    data = np.random.randn(1000, 1000).astype(np.float32)
    arrays = [pa.array(data[:, i]) for i in range(1000)]
    names = [f"c_{i}" for i in range(1000)]
    return pa.RecordBatch.from_arrays(arrays, names=names)

@pytest.fixture(scope="session")
def streaming_chunks():
    np.random.seed(42)
    n_cols = 10
    total_len = 2000
    chunk_size = 100
    data = np.random.randn(n_cols, total_len).astype(np.float32)
    col_names = [f"c_{i}" for i in range(n_cols)]
    
    chunks = []
    for j in range(0, total_len, chunk_size):
        chunk = data[:, j:j + chunk_size]
        batch = pa.RecordBatch.from_arrays(
            [pa.array(chunk[i]) for i in range(n_cols)],
            names=col_names
        )
        chunks.append(batch)
    return chunks
