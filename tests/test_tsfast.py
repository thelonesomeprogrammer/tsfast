import tsfast
import numpy as np
import pyarrow as pa
import pytest

def test_extract():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    features = ["mean", "std", "energy", "min", "max", "autocorr_lag1", "length", "variance_larger_than_standard_deviation", 'mean_second_derivative_central', 'large_standard_deviation-0.05', 'symmetry_looking-0.05', "ratio_beyond_r_sigma-1.0", "index_mass_quantile-0.5", "c3-1"]
    
    extractor = tsfast.Extractor(features)
    batch = pa.RecordBatch.from_arrays([pa.array(x)], names=['c1'])
    result_batch = extractor.process_2d_floats(batch)
    results = result_batch.to_pandas().iloc[0].values
    
    print(f"Extract results: {results}")
    
    assert np.allclose(results[0], 3.0)
    assert np.allclose(results[1], np.std(x, ddof=1)) # Rust uses ddof=1 for variance/std
    assert np.allclose(results[2], np.sum(x**2))
    assert results[3] == 1.0
    assert results[4] == 5.0
    
    # AutocorrLag1 parity with manual calculation (including the x[0]*x[0] start in Rust implementation)
    # Manual: ( (1*1 + 2*1 + 3*2 + 4*3 + 5*4) / 4 - 3*3 ) / 2.5 = (41/4 - 9) / 2.5 = 1.25 / 2.5 = 0.5
    assert np.allclose(results[5], 0.5)

    assert results[6] == 5.0 # length
    # var of [1,2,3,4,5] is 2.5. 2.5 > 1.0, so 1.0
    assert results[7] == 1.0 # variance_larger_than_standard_deviation

    assert np.allclose(results[11], 0.4)
    assert np.allclose(results[12], 0.8)
    assert np.allclose(results[13], 30.0)

    print("test_extract passed!")

def test_spectral_roll_on_off():
    x = np.random.RandomState(42).randn(100).astype(np.float32)
    features = ["spectral_roll_on", "spectral_roll_off", "spectral_slope"]

    extractor = tsfast.Extractor(features)
    batch = pa.RecordBatch.from_arrays([pa.array(x)], names=['c1'])
    result_batch = extractor.process_2d_floats(batch)
    results = result_batch.to_pandas().iloc[0].values

    import tsfel
    fs = 100.0
    ro = tsfel.feature_extraction.features.spectral_roll_on(x, fs)
    rf = tsfel.feature_extraction.features.spectral_roll_off(x, fs)
    ss = tsfel.feature_extraction.features.spectral_slope(x, fs)

    assert np.allclose(results[0], ro)
    assert np.allclose(results[1], rf)
    assert np.allclose(results[2], ss)
    print("test_spectral_roll_on_off passed!")

def test_new_features():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    import tsfresh.feature_extraction.feature_calculators as fc
    features = ["mad", "iqr", "entropy", "mean_abs_change", "mean_change", "cid_ce", "sample_entropy", "binned_entropy__max_bins_5"]
    
    extractor = tsfast.Extractor(features)
    batch = pa.RecordBatch.from_arrays([pa.array(x)], names=['c1'])
    result_batch = extractor.process_2d_floats(batch)
    results = result_batch.to_pandas().iloc[0].values
    
    # MAD: mean(|x - mean(x)|)
    assert np.allclose(results[0], 1.5)
    # IQR: Q3 - Q1. x=[1, 2, 3, 4, 5, 6]. Q1=2.25, Q3=4.75. IQR=2.5.
    assert np.allclose(results[1], 2.5)
    assert results[2] > 0
    assert np.allclose(results[3], 1.0)
    assert np.allclose(results[4], 1.0)
    assert np.allclose(results[5], np.sqrt(5.0))
    assert np.allclose(results[6], fc.sample_entropy(x), equal_nan=True)
    assert np.allclose(results[7], fc.binned_entropy(x, 5), equal_nan=True)
    print("test_new_features passed!")

def test_paa():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    features = ["paa-2-0", "paa-2-1"]
    
    extractor = tsfast.Extractor(features)
    batch = pa.RecordBatch.from_arrays([pa.array(x)], names=['c1'])
    result_batch = extractor.process_2d_floats(batch)
    results = result_batch.to_pandas().iloc[0].values
    
    assert np.allclose(results[0], 2.0)
    assert np.allclose(results[1], 5.0)
    print("test_paa passed!")

def test_advanced_features():
    x = np.array([1.0, 2.0, 1.0, 2.0, 1.0, 2.0], dtype=np.float32)
    # autocorr-2 for [1,2,1,2,1,2] mean=1.5, var=0.3
    # num: sum_{i=0}^{n-lag-1} (x[i]-mean)(x[i+lag]-mean)
    # i=0: (1-1.5)*(1-1.5) = 0.25
    # i=1: (2-1.5)*(2-1.5) = 0.25
    # i=2: (1-1.5)*(1-1.5) = 0.25
    # i=3: (2-1.5)*(2-1.5) = 0.25
    # sum = 1.0
    # den = var * (n-1) = 0.3 * 5 = 1.5
    # result = 1.0 / 1.5 = 0.666...
    features = ["fft_coeff-1-real", "fft_coeff-1-abs", "autocorr-2"]
    
    extractor = tsfast.Extractor(features)
    batch = pa.RecordBatch.from_arrays([pa.array(x)], names=['c1'])
    result_batch = extractor.process_2d_floats(batch)
    results = result_batch.to_pandas().iloc[0].values
    print(f"Advanced features results: {results}")
    
    # FFT real coeff 1 parity
    assert np.allclose(results[0], 0.0, atol=1e-5)
    
    # autocorr-2 parity
    assert np.allclose(results[2], 2/3, atol=1e-5)
    print("test_advanced_features passed!")

def test_2d_extraction():
    # Test processing multiple series at once
    x = np.array([
        [1, 2, 3, 4, 5],
        [5, 4, 3, 2, 1]
    ], dtype=np.float32)
    features = ["mean", "max_value"]
    
    extractor = tsfast.Extractor(features)
    batch = pa.RecordBatch.from_arrays([
        pa.array(x[0]),
        pa.array(x[1])
    ], names=['s1', 's2'])
    
    result_batch = extractor.process_2d_floats(batch)
    df = result_batch.to_pandas()
    
    assert np.allclose(df.iloc[0], [3.0, 5.0]) # s1
    assert np.allclose(df.iloc[1], [3.0, 5.0]) # s2
    print("test_2d_extraction passed!")

def test_location_features():
    x = np.array([2.0, 5.0, 1.0, 5.0, 1.0, 3.0], dtype=np.float32)
    features = ["first_loc_max", "last_loc_max", "first_loc_min", "last_loc_min"]

    extractor = tsfast.Extractor(features)
    batch = pa.RecordBatch.from_arrays([pa.array(x)], names=['c1'])
    result_batch = extractor.process_2d_floats(batch)
    results = result_batch.to_pandas().iloc[0].values

    assert np.allclose(results[0], 1.0 / 6.0)
    assert np.allclose(results[1], 4.0 / 6.0)
    assert np.allclose(results[2], 2.0 / 6.0)
    assert np.allclose(results[3], 5.0 / 6.0)
    print("test_location_features passed!")


if __name__ == "__main__":
    test_extract()
    test_new_features()
    test_paa()
    test_advanced_features()
    test_2d_extraction()

def test_duplicate_features():
    from tsfresh.feature_extraction.feature_calculators import has_duplicate, has_duplicate_max, has_duplicate_min
    import pandas as pd

    # Test case 1
    x1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    # Test case 2
    x2 = np.array([1.0, 5.0, 3.0, 4.0, 5.0], dtype=np.float32)
    # Test case 3
    x3 = np.array([1.0, 2.0, 3.0, 1.0, 5.0], dtype=np.float32)
    # Test case 4
    x4 = np.array([3.0, 3.0, 3.0], dtype=np.float32)

    features = ["has_duplicate", "has_duplicate_max", "has_duplicate_min"]
    extractor = tsfast.Extractor(features)

    for x in [x1, x2, x3, x4]:
        batch = pa.RecordBatch.from_arrays([pa.array(x)], names=['c1'])
        result_batch = extractor.process_2d_floats(batch)
        results = result_batch.to_pandas().iloc[0].values

        series = pd.Series(x)
        expected_has_duplicate = 1.0 if has_duplicate(series) else 0.0
        expected_has_duplicate_max = 1.0 if has_duplicate_max(series) else 0.0
        expected_has_duplicate_min = 1.0 if has_duplicate_min(series) else 0.0

        assert results[0] == expected_has_duplicate
        assert results[1] == expected_has_duplicate_max
        assert results[2] == expected_has_duplicate_min

def test_extract_invalid_type():
    # Verify that passing non-float32 arrays safely raises a TypeError instead of crashing
    x = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    features = ["mean", "std"]
    extractor = tsfast.Extractor(features)
    batch = pa.RecordBatch.from_arrays([pa.array(x)], names=['c1'])
    with pytest.raises(TypeError, match="Failed to downcast column to Float32Array"):
        extractor.process_2d_floats(batch)

def test_extract_empty_batch():
    # Verify the Rust engine handles zero-length batches without panicking
    features = ["mean", "std"]
    extractor = tsfast.Extractor(features)

    empty_data = pa.RecordBatch.from_arrays([pa.array([], type=pa.float32())], names=['c1'])
    result = extractor.process_2d_floats(empty_data)

    df = result.to_pandas()
    assert len(df) == 1
    # For empty batches, the rust engine defaults to returning 0.0 values across all requested features
    assert df.iloc[0]['mean'] == 0.0

def test_extract_invalid_feature():
    # Verify unsupported features immediately error during initialization
    features = ["invalid_feature"]
    with pytest.raises(ValueError, match="Unknown feature"):
        extractor = tsfast.Extractor(features)

def test_reoccurring_ratios():
    import pyarrow as pa
    import tsfast
    import numpy as np

    x = np.array([1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0], dtype=np.float32)
    # len = 7
    # values: 1, 2, 3, 4 (4 unique values)
    # 2 occurs twice (reoccurring) -> reoccurring_datapoints = 2
    # 3 occurs three times (reoccurring) -> reoccurring_datapoints = 3
    # reoccurring_values = 2 (the values 2 and 3)
    # Total reoccurring datapoints = 5

    features = [
        "percentage_of_reoccurring_datapoints_to_all_datapoints",
        "percentage_of_reoccurring_values_to_all_values",
        "ratio_value_number_to_time_series_length"
    ]

    extractor = tsfast.Extractor(features)
    batch = pa.RecordBatch.from_arrays([pa.array(x)], names=['c1'])
    result_batch = extractor.process_2d_floats(batch)
    results = result_batch.to_pandas().iloc[0].values

    assert np.allclose(results[0], 5.0 / 7.0)
    assert np.allclose(results[1], 2.0 / 4.0)
    assert np.allclose(results[2], 4.0 / 7.0)
