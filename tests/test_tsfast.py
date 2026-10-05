import tsfast
import numpy as np
import pyarrow as pa
import pytest

def test_extract():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    features = ["mean", "std", "energy", "min", "max", "autocorr_lag1", "length", "variance_larger_than_standard_deviation", 'mean_second_derivative_central', 'large_standard_deviation-0.05', 'symmetry_looking-0.05', "ratio_beyond_r_sigma-1.0", "index_mass_quantile-0.5", "c3-1", "agg_autocorrelation-mean-2", "agg_autocorrelation-var-2", "agg_autocorrelation-max-2", "agg_autocorrelation-min-2"]
    
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

    # In Rust engine, FULL_AUTOCORR uses FFT which yields slightly different values than manual standard calculation for small N.
    # We will test agg_autocorrelation with random data directly against tsfresh below.
    pass

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
    features = ["mad", "iqr", "entropy", "mean_abs_change", "mean_change", "cid_ce", "sample_entropy", "binned_entropy__max_bins_5", "energy_ratio_by_chunks_num_segments_3__segment_focus_0", "energy_ratio_by_chunks_num_segments_3__segment_focus_1", "energy_ratio_by_chunks_num_segments_3__segment_focus_2"]
    
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
    assert len(df) == 0

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

def test_ecdf_pk_centroid():
    import tsfel
    import tsfast
    import pyarrow as pa
    x = np.array([1.0, -2.0, 3.0, 4.0, 5.0, 1.0, 0.0], dtype=np.float32)
    features = ["ecdf-10", "ecdf-3", "pk_pk_distance", "calc_centroid-100", "calc_centroid-50", "negative_turning", "positive_turning"]
    extractor = tsfast.Extractor(features)

    batch = pa.RecordBatch.from_arrays([pa.array(x)], names=['c1'])
    results = extractor.process_2d_floats(batch).to_pandas().iloc[0].values

    tsfel_neg = tsfel.feature_extraction.features.negative_turning(x)
    tsfel_pos = tsfel.feature_extraction.features.positive_turning(x)
    tsfel_ecdf_10 = tsfel.feature_extraction.features.ecdf(x, d=10)
    tsfel_ecdf_3 = tsfel.feature_extraction.features.ecdf(x, d=3)
    tsfel_pk = tsfel.feature_extraction.features.pk_pk_distance(x)
    tsfel_centroid = tsfel.feature_extraction.features.calc_centroid(x, fs=100)
    tsfel_centroid_50 = tsfel.feature_extraction.features.calc_centroid(x, fs=50)

    assert np.allclose(results[0], min(10.0 / len(x), 1.0))
    assert np.allclose(results[1], min(3.0 / len(x), 1.0))
    assert np.allclose(results[2], tsfel_pk)
    assert np.allclose(results[3], tsfel_centroid)
    assert np.allclose(results[4], tsfel_centroid_50)
    assert np.allclose(results[5], tsfel_neg)
    assert np.allclose(results[6], tsfel_pos)

def test_mfcc_wavelet():
    np.random.seed(0)
    x = np.random.randn(100).astype(np.float32)

    features = ["mfcc-0", "mfcc-11", "wavelet_energy-0", "wavelet_energy-8", "wavelet_entropy"]
    
    extractor = tsfast.Extractor(features)
    batch = pa.RecordBatch.from_arrays([pa.array(x)], names=['c1'])
    result_batch = extractor.process_2d_floats(batch)
    results = result_batch.to_pandas().iloc[0].values
    
    # TSFEL values
    # The first mfcc-0 with liftering but NO mean subtraction is roughly 424.3
    # The last mfcc-11 is roughly 329.9
    # The wavelet_energy-0 is 0.824
    # The wavelet_energy-8 is 0.999
    # The wavelet_entropy is 2.193
    assert np.all(results != 0.0)
    assert np.abs(results[2] - 0.824) < 0.1
    assert np.abs(results[3] - 0.999) < 0.1
    assert np.abs(results[4] - 2.193) < 0.1

def test_spectral_shape():
    np.random.seed(42)
    data = np.random.randn(100).astype(np.float32)
    batch = pa.RecordBatch.from_arrays([pa.array(data)], names=["x"])

    features = ["spectral_centroid", "spectral_spread", "spectral_entropy"]
    extractor = tsfast.Extractor(features)
    result_rust = extractor.process_2d_floats(batch)

    from tsfel.feature_extraction.features import spectral_centroid, spectral_spread, spectral_entropy

    fs = 100
    # tsfel needs sampling freq for these, default 100
    tsfel_centroid = spectral_centroid(data, fs)
    tsfel_spread = spectral_spread(data, fs)
    tsfel_entropy = spectral_entropy(data, fs)

    # tsfast calculates spectral centroid and spread as bin indices (e.g., 0, 1, 2, ... N/2)
    # to match tsfel we must multiply by (fs / len(data))
    freq_resolution = fs / len(data)
    assert result_rust.column("spectral_centroid")[0].as_py() * freq_resolution == pytest.approx(tsfel_centroid, rel=1e-5)
    assert result_rust.column("spectral_spread")[0].as_py() * freq_resolution == pytest.approx(tsfel_spread, rel=1e-5)
    assert result_rust.column("spectral_entropy")[0].as_py() == pytest.approx(tsfel_entropy, rel=1e-5)

def test_dynamic_features():
    from tsfresh.feature_extraction import feature_calculators as fc
    np.random.seed(42)
    # We need a large array so tsfresh's pd.qcut doesn't fail with duplicate bin edges
    x = np.cumsum(np.random.randn(5000)).astype(np.float32)

    features = [
        "ar_coefficient-2-0",
        "ar_coefficient-2-1",
        "ar_coefficient-2-2",
        "friedrich_coefficients-3-30-0",
        "friedrich_coefficients-3-30-1",
        "friedrich_coefficients-3-30-2",
        "friedrich_coefficients-3-30-3",
        "max_langevin_fixed_point-3-30"
    ]

    extractor = tsfast.Extractor(features)
    batch = pa.RecordBatch.from_arrays([pa.array(x)], names=['c1'])
    result_batch = extractor.process_2d_floats(batch)
    results = result_batch.to_pandas().iloc[0].values

    assert not np.isnan(results).any()

    ar_ref = dict(fc.ar_coefficient(x, [{"k": 2, "coeff": 0}, {"k": 2, "coeff": 1}, {"k": 2, "coeff": 2}]))

    friedrich_ref = dict(fc.friedrich_coefficients(x, [
        {"m": 3, "r": 30, "coeff": 0},
        {"m": 3, "r": 30, "coeff": 1},
        {"m": 3, "r": 30, "coeff": 2},
        {"m": 3, "r": 30, "coeff": 3}
    ]))

    mlfp_ref = fc.max_langevin_fixed_point(x, m=3, r=30)

    # Check AR
    assert np.allclose(results[0], ar_ref["coeff_0__k_2"], equal_nan=True, rtol=1e-1, atol=0.02)
    assert np.allclose(results[1], ar_ref["coeff_1__k_2"], equal_nan=True, rtol=1e-1, atol=0.02)
    assert np.allclose(results[2], ar_ref["coeff_2__k_2"], equal_nan=True, rtol=1e-1, atol=0.02)

    # Check Friedrich (tsfresh polyfit outputs descending order [x^m, x^m-1, ...], we should match)
    assert np.allclose(results[3], friedrich_ref["coeff_0__m_3__r_30"], equal_nan=True, rtol=1e-1, atol=0.02)
    assert np.allclose(results[4], friedrich_ref["coeff_1__m_3__r_30"], equal_nan=True, rtol=1e-1, atol=0.02)
    assert np.allclose(results[5], friedrich_ref["coeff_2__m_3__r_30"], equal_nan=True, rtol=1e-1, atol=0.02)
    assert np.allclose(results[6], friedrich_ref["coeff_3__m_3__r_30"], equal_nan=True, rtol=1e-1, atol=0.02)

    # Check Max Langevin
    assert np.allclose(results[7], mlfp_ref, equal_nan=True, rtol=1e-1, atol=0.02)


def test_lpcc():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    features = [f"lpcc-{i}" for i in range(12)]

    import tsfel
    extractor = tsfast.Extractor(features)
    batch = pa.RecordBatch.from_arrays([pa.array(x)], names=['c1'])
    results = extractor.process_2d_floats(batch).to_pandas().iloc[0].values
    expected = np.array(tsfel.feature_extraction.features.lpcc(x))
    assert np.allclose(results, expected, atol=1e-5)

def test_augmented_dickey_fuller():
    import numpy as np
    import tsfast
    import tsfresh.feature_extraction.feature_calculators as fc

    np.random.seed(42)
    # Test vector 1: Random noise
    x1 = np.random.randn(100).astype(np.float32)

    # Test vector 2: Random walk
    x2 = np.cumsum(np.random.randn(100)).astype(np.float32)

    # Test vector 3: Linear trend
    x3 = (np.arange(100) * 0.1 + np.random.randn(100)).astype(np.float32)

    features = [
        "augmented_dickey_fuller-teststat",
        "augmented_dickey_fuller-pvalue",
        "augmented_dickey_fuller-usedlag"
    ]

    for x in [x1, x2, x3]:
        tsfresh_teststat = fc.augmented_dickey_fuller(x, [{"attr": "teststat"}])[0][1]
        tsfresh_pvalue = fc.augmented_dickey_fuller(x, [{"attr": "pvalue"}])[0][1]
        tsfresh_usedlag = fc.augmented_dickey_fuller(x, [{"attr": "usedlag"}])[0][1]

        batch = pa.RecordBatch.from_arrays([pa.array(x)], names=['x'])
        res = tsfast.Extractor(features).process_2d_floats(batch)

        # Teststat
        assert np.isclose(res[0][0].as_py(), tsfresh_teststat, rtol=1e-2, atol=0.02)
        # P-value (MacKinnon interpolation might differ slightly)
        assert np.isclose(res[1][0].as_py(), tsfresh_pvalue, rtol=1e-2, atol=0.02)
        # Used lag
        assert res[2][0].as_py() == tsfresh_usedlag

def test_median_diff_features():
    x = np.random.RandomState(42).randn(100).astype(np.float32)
    features = ["median_diff", "median_abs_diff"]

    import tsfel
    extractor = tsfast.Extractor(features)
    batch = pa.RecordBatch.from_arrays([pa.array(x)], names=['c1'])
    results = extractor.process_2d_floats(batch).to_pandas().iloc[0].values
    assert np.allclose(results[0], tsfel.feature_extraction.features.median_diff(x))
    assert np.allclose(results[1], tsfel.feature_extraction.features.median_abs_diff(x))

def test_agg_autocorrelation():
    np.random.seed(42)
    x = np.random.randn(100).astype(np.float32)
    features = [
        "agg_autocorrelation-mean-10",
        "agg_autocorrelation-var-10",
        "agg_autocorrelation-max-10",
        "agg_autocorrelation-min-10"
    ]
    import tsfast
    import pyarrow as pa
    extractor = tsfast.Extractor(features)
    batch = pa.RecordBatch.from_arrays([pa.array(x)], names=['c1'])
    results = extractor.process_2d_floats(batch).to_pandas().iloc[0].values

    import tsfresh
    expected = [
        np.mean([tsfresh.feature_extraction.feature_calculators.autocorrelation(x, lag) for lag in range(1, 10)]),
        np.var([tsfresh.feature_extraction.feature_calculators.autocorrelation(x, lag) for lag in range(1, 10)]),
        np.max([tsfresh.feature_extraction.feature_calculators.autocorrelation(x, lag) for lag in range(1, 10)]),
        np.min([tsfresh.feature_extraction.feature_calculators.autocorrelation(x, lag) for lag in range(1, 10)])
    ]
    assert np.allclose(results, expected, atol=0.02)

    import tsfel
    tsfel_ecdf_10 = tsfel.feature_extraction.features.ecdf(x, d=10)
    tsfel_ecdf_3 = tsfel.feature_extraction.features.ecdf(x, d=3)
    tsfel_pk = tsfel.feature_extraction.features.pk_pk_distance(x)
    tsfel_centroid = tsfel.feature_extraction.features.calc_centroid(x, fs=100)
    tsfel_centroid_50 = tsfel.feature_extraction.features.calc_centroid(x, fs=50)

    from tsfresh.feature_extraction.feature_calculators import agg_autocorrelation
    # tsfresh returns a dictionary: { "f_agg_"mean"_maxlag_10": val, ... }

    t_mean = agg_autocorrelation(x, [{"f_agg": "mean", "maxlag": 10}])[0][1]
    t_var = agg_autocorrelation(x, [{"f_agg": "var", "maxlag": 10}])[0][1]

    # Check bounds or logic
    assert len(results) == 4

    from tsfresh.feature_extraction.feature_calculators import agg_autocorrelation

    t_mean = agg_autocorrelation(x, [{"f_agg": "mean", "maxlag": 10}])[0][1]
    t_var = agg_autocorrelation(x, [{"f_agg": "var", "maxlag": 10}])[0][1]
    t_max = agg_autocorrelation(x, [{"f_agg": "max", "maxlag": 10}])[0][1]
    t_min = agg_autocorrelation(x, [{"f_agg": "min", "maxlag": 10}])[0][1]

    assert np.allclose(results[0], t_mean, atol=0.02)
    assert np.allclose(results[1], t_var, atol=0.02)
    assert np.allclose(results[2], t_max, atol=1.5e-2)
    assert np.allclose(results[3], t_min, atol=1.5e-2)

def test_change_quantiles():
    from tsfresh.feature_extraction.feature_calculators import change_quantiles
    import tsfast
    import numpy as np
    import pyarrow as pa

    x = np.array([3.0, 1.0, 4.0, 1.5, 9.0, 2.0, 6.0, 5.0, 3.5, 8.0, 9.0], dtype=np.float32)

    features = [
        "change_quantiles-0.2-0.8-True-mean",
        "change_quantiles-0.2-0.8-False-var",
        "change_quantiles-0.0-1.0-True-max",
        "change_quantiles-0.1-0.9-False-min",
    ]

    extractor = tsfast.Extractor(features)
    batch = pa.RecordBatch.from_arrays([pa.array(x)], names=['c1'])
    result_batch = extractor.process_2d_floats(batch)
    results = result_batch.to_pandas().iloc[0].values


    assert len(results) == 4

    expected = [
        change_quantiles(x, 0.2, 0.8, True, "mean"),
        change_quantiles(x, 0.2, 0.8, False, "var"),
        change_quantiles(x, 0.0, 1.0, True, "max"),
        change_quantiles(x, 0.1, 0.9, False, "min"),
    ]
    assert np.allclose(results, expected, atol=0.02)


    expected_1 = change_quantiles(x, 0.2, 0.8, True, "mean")
    expected_2 = change_quantiles(x, 0.2, 0.8, False, "var")
    expected_3 = change_quantiles(x, 0.0, 1.0, True, "max")
    expected_4 = change_quantiles(x, 0.1, 0.9, False, "min")

    assert np.allclose(results[0], expected_1)
    assert np.allclose(results[1], expected_2)
    assert np.allclose(results[2], expected_3)
    assert np.allclose(results[3], expected_4)

from tsfresh.feature_extraction.feature_calculators import permutation_entropy, value_count

def test_permutation_entropy_and_value_count():
    import numpy as np
    data = np.array([4.0, 7.0, 9.0, 10.0, 6.0, 11.0, 3.0, 3.0, np.nan, 3.0, np.nan], dtype=np.float32)
    batch = pa.RecordBatch.from_arrays([pa.array(data)], names=["col0"])

    features = [
        "permutation_entropy-1-3",
        "permutation_entropy-2-3",
        "value_count-3.0",
        "value_count-6.0",
        "value_count-nan"
    ]

    extractor = tsfast.Extractor(features)
    res = extractor.process_2d_floats(batch)

    assert res is not None
    assert res.num_columns == len(features)

    res_pe1 = res[0][0].as_py()
    res_pe2 = res[1][0].as_py()
    res_vc3 = res[2][0].as_py()
    res_vc6 = res[3][0].as_py()
    res_vcn = res[4][0].as_py()

    ts_pe1 = permutation_entropy(data, tau=1, dimension=3)
    ts_pe2 = permutation_entropy(data, tau=2, dimension=3)
    ts_vc3 = value_count(data, 3.0)
    ts_vc6 = value_count(data, 6.0)
    ts_vcn = value_count(data, np.nan)

    np.testing.assert_allclose(res_pe1, ts_pe1, rtol=1e-5)
    np.testing.assert_allclose(res_pe2, ts_pe2, rtol=1e-5)
    assert res_vc3 == ts_vc3
    assert res_vc6 == ts_vc6
    assert res_vcn == ts_vcn

def test_fractal_dimensions():
    import numpy as np
    import pyarrow as pa
    import tsfast
    from tsfel.feature_extraction.features import higuchi_fractal_dimension
    import warnings
    warnings.filterwarnings('ignore')

    x = np.random.randn(200).astype(np.float32)
    features = ["higuchi_fd"]

    extractor = tsfast.Extractor(features)
    batch = pa.RecordBatch.from_arrays([pa.array(x)], names=['c1'])
    result_batch = extractor.process_2d_floats(batch)
    results = result_batch.to_pandas().iloc[0].values

    hfd = higuchi_fractal_dimension(x)

    assert np.allclose(results[0], hfd, atol=5e-2)
    print("test_fractal_dimensions passed!")

def test_invalid_type():
    import pytest
    import pyarrow as pa
    import numpy as np
    import tsfast

    x = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    features = ["mean"]
    extractor = tsfast.Extractor(features)
    batch = pa.RecordBatch.from_arrays([pa.array(x)], names=['c1'])
    with pytest.raises(TypeError, match="Failed to downcast column to Float32Array"):
        extractor.process_2d_floats(batch)

def test_empty_batch():
    import pyarrow as pa
    import tsfast

    empty_data = pa.RecordBatch.from_arrays([pa.array([], type=pa.float32())], names=['c'])
    features = ["mean"]
    extractor = tsfast.Extractor(features)
    result = extractor.process_2d_floats(empty_data)
    assert result.num_rows == 0


# Shared reference for mean_second_derivative_central, large_standard_deviation,
# symmetry_looking, turning_points and slope_sign_change (also used by the
# sliding/expanding tests).
SHAPE_FEATURES = [
    "mean_second_derivative_central",
    "large_standard_deviation-0.25",
    "symmetry_looking-0.05",
    "turning_points",
    "slope_sign_change",
]


def shape_features_reference(x):
    import tsfresh.feature_extraction.feature_calculators as fc

    x = np.asarray(x, dtype=np.float64)
    mid = x[1:-1]
    left, right = x[:-2], x[2:]
    turning = np.sum(((mid > left) & (mid > right)) | ((mid < left) & (mid < right)))
    ssc = np.sum((mid - left) * (mid - right) >= 0)
    return [
        fc.mean_second_derivative_central(x),
        float(fc.large_standard_deviation(x, 0.25)),
        float(dict(fc.symmetry_looking(x, [{"r": 0.05}]))["r_0.05"]),
        float(turning),
        float(ssc),
    ]


SHAPE_CASES = {
    "randn": np.random.RandomState(0).randn(200),
    "exponential": np.random.RandomState(1).exponential(size=120),
    "constant": np.full(50, 3.0),
    "plateaus": np.array([0, 1, 1, 0, -1, -1, 0, 2, 2, 2, 1], dtype=float),
}


@pytest.mark.parametrize("case", SHAPE_CASES)
def test_shape_features_static(case):
    x = SHAPE_CASES[case].astype(np.float32)
    batch = pa.RecordBatch.from_arrays([pa.array(x)], names=["c1"])
    results = tsfast.Extractor(SHAPE_FEATURES).process_2d_floats(batch).to_pandas().iloc[0].values
    assert np.allclose(results, shape_features_reference(x), atol=1e-5, equal_nan=True)


def test_undefined_moments_of_constant_series_are_zero():
    # Guarded features (zero variance) report 0.0, not NaN.
    x = np.full(20, 2.0, dtype=np.float32)
    batch = pa.RecordBatch.from_arrays([pa.array(x)], names=["c1"])
    feats = ["skewness", "kurtosis", "biased_fisher_kurtosis", "autocorr_lag1"]
    results = tsfast.Extractor(feats).process_2d_floats(batch).to_pandas().iloc[0].values
    assert np.array_equal(results, np.zeros(len(feats)))
