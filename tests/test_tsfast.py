import tsfast
import numpy as np
import pytest
from helpers import frame


def test_extract():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    features = [
        "mean",
        "std",
        "energy",
        "min",
        "max",
        "autocorr_lag1",
        "length",
        "variance_larger_than_standard_deviation",
        "mean_second_derivative_central",
        "large_standard_deviation-0.05",
        "symmetry_looking-0.05",
        "ratio_beyond_r_sigma-1.0",
        "index_mass_quantile-0.5",
        "c3-1",
        "agg_autocorrelation-mean-2",
        "agg_autocorrelation-var-2",
        "agg_autocorrelation-max-2",
        "agg_autocorrelation-min-2",
    ]

    extractor = tsfast.Extractor(features)
    batch = np.stack([x])
    out = extractor.process_2d_floats(batch)
    results = out[0]

    print(f"Extract results: {results}")

    assert np.allclose(results[0], 3.0)
    assert np.allclose(results[1], np.std(x))  # ddof=0, as tsfresh/TSFEL
    assert np.allclose(results[2], np.sum(x**2))
    assert results[3] == 1.0
    assert results[4] == 5.0

    # AutocorrLag1 parity with manual calculation (including the x[0]*x[0] start in Rust implementation)
    # Manual: ( (1*1 + 2*1 + 3*2 + 4*3 + 5*4) / 4 - 3*3 ) / 2.5 = (41/4 - 9) / 2.5 = 1.25 / 2.5 = 0.5
    assert np.allclose(results[5], 0.5)

    assert results[6] == 5.0  # length
    # var of [1,2,3,4,5] is 2.5. 2.5 > 1.0, so 1.0
    assert results[7] == 1.0  # variance_larger_than_standard_deviation

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
    batch = np.stack([x])
    out = extractor.process_2d_floats(batch)
    results = out[0]

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

    features = [
        "mad",
        "iqr",
        "entropy",
        "mean_abs_change",
        "mean_change",
        "cid_ce",
        "sample_entropy",
        "binned_entropy__max_bins_5",
        "energy_ratio_by_chunks_num_segments_3__segment_focus_0",
        "energy_ratio_by_chunks_num_segments_3__segment_focus_1",
        "energy_ratio_by_chunks_num_segments_3__segment_focus_2",
    ]

    extractor = tsfast.Extractor(features)
    batch = np.stack([x])
    out = extractor.process_2d_floats(batch)
    results = out[0]

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
    batch = np.stack([x])
    out = extractor.process_2d_floats(batch)
    results = out[0]

    assert np.allclose(results[0], 2.0)
    assert np.allclose(results[1], 5.0)
    print("test_paa passed!")


def test_advanced_features():
    x = np.array([1.0, 2.0, 1.0, 2.0, 1.0, 2.0], dtype=np.float32)
    # autocorr-2 for [1,2,1,2,1,2] mean=1.5, population var=0.25
    # num: sum_{i=0}^{n-lag-1} (x[i]-mean)(x[i+lag]-mean)
    # i=0: (1-1.5)*(1-1.5) = 0.25
    # i=1: (2-1.5)*(2-1.5) = 0.25
    # i=2: (1-1.5)*(1-1.5) = 0.25
    # i=3: (2-1.5)*(2-1.5) = 0.25
    # sum = 1.0
    # tsfresh normalises lag l by (n - l) * population variance:
    # den = (6 - 2) * 0.25 = 1.0
    # result = 1.0 / 1.0 = 1.0
    features = ["fft_coeff-1-real", "fft_coeff-1-abs", "autocorr-2"]

    extractor = tsfast.Extractor(features)
    batch = np.stack([x])
    out = extractor.process_2d_floats(batch)
    results = out[0]
    print(f"Advanced features results: {results}")

    # FFT real coeff 1 parity
    assert np.allclose(results[0], 0.0, atol=1e-5)

    # autocorr-2 parity
    assert np.allclose(results[2], 1.0, atol=1e-5)
    print("test_advanced_features passed!")


def test_2d_extraction():
    # Test processing multiple series at once
    x = np.array([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]], dtype=np.float32)
    features = ["mean", "max_value"]

    extractor = tsfast.Extractor(features)
    batch = x

    out = extractor.process_2d_floats(batch)
    df = frame(extractor, out)

    assert np.allclose(df.iloc[0], [3.0, 5.0])  # s1
    assert np.allclose(df.iloc[1], [3.0, 5.0])  # s2
    print("test_2d_extraction passed!")


def test_location_features():
    x = np.array([2.0, 5.0, 1.0, 5.0, 1.0, 3.0], dtype=np.float32)
    features = ["first_loc_max", "last_loc_max", "first_loc_min", "last_loc_min"]

    extractor = tsfast.Extractor(features)
    batch = np.stack([x])
    out = extractor.process_2d_floats(batch)
    results = out[0]

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
    from tsfresh.feature_extraction.feature_calculators import (
        has_duplicate,
        has_duplicate_max,
        has_duplicate_min,
    )
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
        batch = np.stack([x])
        out = extractor.process_2d_floats(batch)
        results = out[0]

        series = pd.Series(x)
        expected_has_duplicate = 1.0 if has_duplicate(series) else 0.0
        expected_has_duplicate_max = 1.0 if has_duplicate_max(series) else 0.0
        expected_has_duplicate_min = 1.0 if has_duplicate_min(series) else 0.0

        assert results[0] == expected_has_duplicate
        assert results[1] == expected_has_duplicate_max
        assert results[2] == expected_has_duplicate_min


def test_extract_invalid_type():
    # Verify that passing non-float arrays safely raises a TypeError instead of crashing
    x = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    features = ["mean", "std"]
    extractor = tsfast.Extractor(features)
    batch = np.stack([x])
    with pytest.raises(TypeError, match="expected a float32 or float64 numpy array"):
        extractor.process_2d_floats(batch)


def test_extract_empty_batch():
    # Verify the Rust engine handles zero-length batches without panicking
    features = ["mean", "std"]
    extractor = tsfast.Extractor(features)

    empty_data = np.stack([[]])
    result = extractor.process_2d_floats(empty_data)

    df = frame(extractor, result)
    assert len(df) == 0


def test_extract_invalid_feature():
    # Verify unsupported features immediately error during initialization
    features = ["invalid_feature"]
    with pytest.raises(ValueError, match="Unknown feature"):
        extractor = tsfast.Extractor(features)


def test_reoccurring_ratios():
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
        "ratio_value_number_to_time_series_length",
    ]

    extractor = tsfast.Extractor(features)
    batch = np.stack([x])
    out = extractor.process_2d_floats(batch)
    results = out[0]

    assert np.allclose(results[0], 5.0 / 7.0)
    assert np.allclose(results[1], 2.0 / 4.0)
    assert np.allclose(results[2], 4.0 / 7.0)


def test_ecdf_pk_centroid():
    import tsfel
    import tsfast

    x = np.array([1.0, -2.0, 3.0, 4.0, 5.0, 1.0, 0.0], dtype=np.float32)
    features = [
        "ecdf-10",
        "ecdf-3",
        "pk_pk_distance",
        "calc_centroid-100",
        "calc_centroid-50",
        "negative_turning",
        "positive_turning",
    ]
    extractor = tsfast.Extractor(features)

    batch = np.stack([x])
    results = extractor.process_2d_floats(batch)[0]

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

    features = [
        "mfcc-0",
        "mfcc-11",
        "wavelet_energy-0",
        "wavelet_energy-8",
        "wavelet_entropy",
    ]

    extractor = tsfast.Extractor(features)
    batch = np.stack([x])
    out = extractor.process_2d_floats(batch)
    results = out[0]

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
    batch = np.stack([data])

    features = ["spectral_centroid", "spectral_spread", "spectral_entropy"]
    extractor = tsfast.Extractor(features)
    result_rust = extractor.process_2d_floats(batch)

    from tsfel.feature_extraction.features import (
        spectral_centroid,
        spectral_spread,
        spectral_entropy,
    )

    fs = 100
    # tsfel needs sampling freq for these, default 100
    tsfel_centroid = spectral_centroid(data, fs)
    tsfel_spread = spectral_spread(data, fs)
    tsfel_entropy = spectral_entropy(data, fs)

    # Centroid and spread are in Hz, as TSFEL (fs = 100).
    assert result_rust[0, features.index("spectral_centroid")] == pytest.approx(
        tsfel_centroid, rel=1e-5
    )
    assert result_rust[0, features.index("spectral_spread")] == pytest.approx(
        tsfel_spread, rel=1e-5
    )
    assert result_rust[0, features.index("spectral_entropy")] == pytest.approx(
        tsfel_entropy, rel=1e-5
    )


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
        "max_langevin_fixed_point-3-30",
    ]

    extractor = tsfast.Extractor(features)
    batch = np.stack([x])
    out = extractor.process_2d_floats(batch)
    results = out[0]

    assert not np.isnan(results).any()

    ar_ref = dict(
        fc.ar_coefficient(
            x, [{"k": 2, "coeff": 0}, {"k": 2, "coeff": 1}, {"k": 2, "coeff": 2}]
        )
    )

    friedrich_ref = dict(
        fc.friedrich_coefficients(
            x,
            [
                {"m": 3, "r": 30, "coeff": 0},
                {"m": 3, "r": 30, "coeff": 1},
                {"m": 3, "r": 30, "coeff": 2},
                {"m": 3, "r": 30, "coeff": 3},
            ],
        )
    )

    mlfp_ref = fc.max_langevin_fixed_point(x, m=3, r=30)

    # Check AR
    assert np.allclose(
        results[0], ar_ref["coeff_0__k_2"], equal_nan=True, rtol=1e-1, atol=0.02
    )
    assert np.allclose(
        results[1], ar_ref["coeff_1__k_2"], equal_nan=True, rtol=1e-1, atol=0.02
    )
    assert np.allclose(
        results[2], ar_ref["coeff_2__k_2"], equal_nan=True, rtol=1e-1, atol=0.02
    )

    # Check Friedrich (tsfresh polyfit outputs descending order [x^m, x^m-1, ...], we should match)
    assert np.allclose(
        results[3],
        friedrich_ref["coeff_0__m_3__r_30"],
        equal_nan=True,
        rtol=1e-1,
        atol=0.02,
    )
    assert np.allclose(
        results[4],
        friedrich_ref["coeff_1__m_3__r_30"],
        equal_nan=True,
        rtol=1e-1,
        atol=0.02,
    )
    assert np.allclose(
        results[5],
        friedrich_ref["coeff_2__m_3__r_30"],
        equal_nan=True,
        rtol=1e-1,
        atol=0.02,
    )
    assert np.allclose(
        results[6],
        friedrich_ref["coeff_3__m_3__r_30"],
        equal_nan=True,
        rtol=1e-1,
        atol=0.02,
    )

    # Check Max Langevin
    assert np.allclose(results[7], mlfp_ref, equal_nan=True, rtol=1e-1, atol=0.02)


def test_lpcc():
    x = np.array(
        [1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        dtype=np.float32,
    )
    features = [f"lpcc-{i}" for i in range(12)]

    import tsfel

    extractor = tsfast.Extractor(features)
    batch = np.stack([x])
    results = extractor.process_2d_floats(batch)[0]
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
        "augmented_dickey_fuller-usedlag",
    ]

    for x in [x1, x2, x3]:
        tsfresh_teststat = fc.augmented_dickey_fuller(x, [{"attr": "teststat"}])[0][1]
        tsfresh_pvalue = fc.augmented_dickey_fuller(x, [{"attr": "pvalue"}])[0][1]
        tsfresh_usedlag = fc.augmented_dickey_fuller(x, [{"attr": "usedlag"}])[0][1]

        batch = np.stack([x])
        res = tsfast.Extractor(features).process_2d_floats(batch)

        # Teststat
        assert np.isclose(res[0, 0], tsfresh_teststat, rtol=1e-2, atol=0.02)
        # P-value (MacKinnon interpolation might differ slightly)
        assert np.isclose(res[0, 1], tsfresh_pvalue, rtol=1e-2, atol=0.02)
        # Used lag
        assert res[0, 2] == tsfresh_usedlag


def test_median_diff_features():
    x = np.random.RandomState(42).randn(100).astype(np.float32)
    features = ["median_diff", "median_abs_diff"]

    import tsfel

    extractor = tsfast.Extractor(features)
    batch = np.stack([x])
    results = extractor.process_2d_floats(batch)[0]
    assert np.allclose(results[0], tsfel.feature_extraction.features.median_diff(x))
    assert np.allclose(results[1], tsfel.feature_extraction.features.median_abs_diff(x))


def test_agg_autocorrelation():
    np.random.seed(42)
    x = np.random.randn(100).astype(np.float32)
    features = [
        "agg_autocorrelation-mean-10",
        "agg_autocorrelation-var-10",
        "agg_autocorrelation-max-10",
        "agg_autocorrelation-min-10",
    ]
    import tsfast

    extractor = tsfast.Extractor(features)
    batch = np.stack([x])
    results = extractor.process_2d_floats(batch)[0]

    import tsfresh

    expected = [
        np.mean(
            [
                tsfresh.feature_extraction.feature_calculators.autocorrelation(x, lag)
                for lag in range(1, 10)
            ]
        ),
        np.var(
            [
                tsfresh.feature_extraction.feature_calculators.autocorrelation(x, lag)
                for lag in range(1, 10)
            ]
        ),
        np.max(
            [
                tsfresh.feature_extraction.feature_calculators.autocorrelation(x, lag)
                for lag in range(1, 10)
            ]
        ),
        np.min(
            [
                tsfresh.feature_extraction.feature_calculators.autocorrelation(x, lag)
                for lag in range(1, 10)
            ]
        ),
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

    x = np.array(
        [3.0, 1.0, 4.0, 1.5, 9.0, 2.0, 6.0, 5.0, 3.5, 8.0, 9.0], dtype=np.float32
    )

    features = [
        "change_quantiles-0.2-0.8-True-mean",
        "change_quantiles-0.2-0.8-False-var",
        "change_quantiles-0.0-1.0-True-max",
        "change_quantiles-0.1-0.9-False-min",
    ]

    extractor = tsfast.Extractor(features)
    batch = np.stack([x])
    out = extractor.process_2d_floats(batch)
    results = out[0]

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


from tsfresh.feature_extraction.feature_calculators import (
    permutation_entropy,
    value_count,
)


def test_permutation_entropy_and_value_count():
    import numpy as np

    data = np.array(
        [4.0, 7.0, 9.0, 10.0, 6.0, 11.0, 3.0, 3.0, np.nan, 3.0, np.nan],
        dtype=np.float32,
    )
    batch = np.stack([data])

    features = [
        "permutation_entropy-1-3",
        "permutation_entropy-2-3",
        "value_count-3.0",
        "value_count-6.0",
        "value_count-nan",
    ]

    extractor = tsfast.Extractor(features)
    res = extractor.process_2d_floats(batch)

    assert res is not None
    assert res.shape[1] == len(features)

    res_pe1 = res[0, 0]
    res_pe2 = res[0, 1]
    res_vc3 = res[0, 2]
    res_vc6 = res[0, 3]
    res_vcn = res[0, 4]

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
    import tsfast
    from tsfel.feature_extraction.features import higuchi_fractal_dimension
    import warnings

    warnings.filterwarnings("ignore")

    x = np.random.randn(200).astype(np.float32)
    features = ["higuchi_fd"]

    extractor = tsfast.Extractor(features)
    batch = np.stack([x])
    out = extractor.process_2d_floats(batch)
    results = out[0]

    hfd = higuchi_fractal_dimension(x)

    assert np.allclose(results[0], hfd, atol=5e-2)
    print("test_fractal_dimensions passed!")


def test_invalid_type():
    import pytest
    import numpy as np
    import tsfast

    x = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    features = ["mean"]
    extractor = tsfast.Extractor(features)
    batch = np.stack([x])
    with pytest.raises(TypeError, match="expected a float32 or float64 numpy array"):
        extractor.process_2d_floats(batch)


def test_empty_batch():
    import tsfast

    empty_data = np.stack([[]])
    features = ["mean"]
    extractor = tsfast.Extractor(features)
    result = extractor.process_2d_floats(empty_data)
    assert result.shape[0] == 0


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
    batch = np.stack([x])
    results = tsfast.Extractor(SHAPE_FEATURES).process_2d_floats(batch)[0]
    assert np.allclose(results, shape_features_reference(x), atol=1e-5, equal_nan=True)


def test_undefined_moments_of_constant_series_are_zero():
    # Guarded features (zero variance) report 0.0, not NaN.
    x = np.full(20, 2.0, dtype=np.float32)
    batch = np.stack([x])
    feats = ["skewness", "kurtosis", "biased_fisher_kurtosis", "autocorr_lag1"]
    results = tsfast.Extractor(feats).process_2d_floats(batch)[0]
    assert np.array_equal(results, np.zeros(len(feats)))


def _sample_features():
    from pathlib import Path

    lines = (Path(__file__).parent / "feature_samples.txt").read_text().splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


def test_output_is_float32_array_with_feature_names():
    data = np.random.default_rng(0).normal(size=(5, 100)).astype(np.float32)
    ext = tsfast.Extractor(["mean", "max_value", "c3-1"])
    out = ext.process_2d_floats(data)
    assert ext.feature_names == ["mean", "max_value", "c3-1"]
    assert out.dtype == np.float32 and out.shape == (5, 3)
    np.testing.assert_allclose(out[:, 0], data.mean(axis=1), rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(out[:, 1], data.max(axis=1))


def test_process_2d_floats_copies_other_layouts_and_dtypes():
    data = np.random.default_rng(1).normal(size=(4, 64)).astype(np.float32)
    ext = tsfast.Extractor(
        ["mean", "energy", "median", "autocorr-2", "fft_coeff-1-abs"]
    )
    expected = ext.process_2d_floats(data)
    wide = np.zeros((4, 128), dtype=np.float32)
    wide[:, ::2] = data
    for variant in [np.asfortranarray(data), wide[:, ::2], data.astype(np.float64)]:
        np.testing.assert_allclose(ext.process_2d_floats(variant), expected, rtol=1e-6)


def test_many_series_match_one_at_a_time():
    # Enough work to take the multi-threaded path; each series alone takes the serial one.
    data = np.random.default_rng(2).normal(size=(300, 128)).astype(np.float32)
    ext = tsfast.Extractor(_sample_features())
    together = ext.process_2d_floats(data)
    for i in [0, 1, 150, 299]:
        np.testing.assert_array_equal(
            together[i], ext.process_2d_floats(data[i : i + 1])[0]
        )


def test_process_2d_floats_edge_shapes():
    ext = tsfast.Extractor(["mean", "energy"])
    assert ext.process_2d_floats(np.zeros((3, 0), dtype=np.float32)).shape == (0, 2)
    assert ext.process_2d_floats(np.zeros((0, 10), dtype=np.float32)).shape == (0, 2)
    np.testing.assert_array_equal(
        ext.process_2d_floats(np.ones((1, 1), dtype=np.float32)), [[1.0, 1.0]]
    )


def test_process_2d_floats_rejects_bad_input():
    ext = tsfast.Extractor(["mean"])
    with pytest.raises(ValueError, match="2-D"):
        ext.process_2d_floats(np.zeros(10, dtype=np.float32))
    with pytest.raises(ValueError, match="2-D"):
        ext.process_2d_floats(np.zeros((2, 3, 4), dtype=np.float32))
    with pytest.raises(TypeError, match="int64"):
        ext.process_2d_floats(np.zeros((2, 3), dtype=np.int64))
    with pytest.raises(TypeError, match="list"):
        ext.process_2d_floats([[1.0, 2.0]])

def test_mse():
    import numpy as np
    import tsfast
    from tsfel.feature_extraction.features import mse

    np.random.seed(42)
    signal = np.random.rand(500).astype(np.float32)

    e = tsfast.Extractor(["mse-3", "mse-2-10"])
    res = e.process_2d_floats(np.array([signal]))

    tol = 0.2 * np.std(signal)
    ref_3_0 = mse(signal, m=3, maxscale=None, tolerance=tol)
    ref_2_10 = mse(signal, m=2, maxscale=10, tolerance=tol)

    assert np.isclose(res[0][0], ref_3_0, atol=1e-2)
    assert np.isclose(res[0][1], ref_2_10, atol=1e-2)

    # Test short signal (returns NaN)
    short_signal = np.random.rand(100).astype(np.float32)
    res_short = e.process_2d_floats(np.array([short_signal]))
    assert np.isnan(res_short[0][0])
    assert np.isnan(res_short[0][1])

    # Test constant signal (returns NaN)
    const_signal = np.ones(500, dtype=np.float32)
    res_const = e.process_2d_floats(np.array([const_signal]))
    assert np.isnan(res_const[0][0])
    assert np.isnan(res_const[0][1])

def test_ecdf_features():
    # Normal case with ties
    x = np.array([1, 2, 2, 2, 5, 5, 7, 8, 9, 10], dtype=np.float32)
    # Constant case
    x_const = np.array([3, 3, 3, 3], dtype=np.float32)
    # Short window
    x_short = np.array([1, 2], dtype=np.float32)

    features = ["ecdf_percentile-0.5", "ecdf_percentile_count-0.5", "ecdf_slope-0.2-0.5"]

    import tsfel
    for arr in [x, x_const, x_short]:
        ext = tsfast.Extractor(features)
        batch = np.stack([arr])
        res = ext.process_2d_floats(batch)[0]

        # ecdf_percentile-0.5
        ref_perc = tsfel.feature_extraction.features.ecdf_percentile(arr, [0.5])
        if np.isscalar(ref_perc):
            ref_perc = float(ref_perc)
        else:
            ref_perc = float(ref_perc[0])

        # ecdf_percentile_count-0.5
        if np.max(arr) == np.min(arr):
            ref_count = float(len(arr))
        else:
            ref_count = float(np.sum(arr <= ref_perc))

        # ecdf_slope-0.2-0.5
        try:
            ref_slope = tsfel.feature_extraction.features.ecdf_slope(arr, 0.2, 0.5)
        except Exception:
            ref_slope = np.nan

        if np.isnan(ref_perc):
            assert np.isnan(res[0])
        else:
            assert np.allclose(res[0], ref_perc)

        if np.isnan(ref_count):
            assert np.isnan(res[1])
        else:
            assert np.allclose(res[1], ref_count)

        if np.isnan(ref_slope):
            assert np.isnan(res[2])
        elif np.isinf(ref_slope):
            assert np.isinf(res[2])
        else:
            assert np.allclose(res[2], ref_slope)

def test_maximum_fractal_length():
    import numpy as np
    from tsfel.feature_extraction.features import maximum_fractal_length

    np.random.seed(42)
    # Sine, random walk, flat
    x_sine = np.sin(np.linspace(0, 10 * np.pi, 200))
    x_rw = np.cumsum(np.random.randn(200))
    x_flat = np.ones(200) * 5.0
    x_short = np.random.randn(9)

    features = ["maximum_fractal_length"]
    ext = tsfast.Extractor(features)

    batch = np.vstack([x_sine, x_rw, x_flat])
    res = ext.process_2d_floats(batch)

    mfl_sine = maximum_fractal_length(x_sine)
    mfl_rw = maximum_fractal_length(x_rw)
    mfl_flat = maximum_fractal_length(x_flat)

    np.testing.assert_allclose([res[i][0] for i in range(len(res))], [mfl_sine, mfl_rw, mfl_flat], atol=1e-5)

    # Below minimum size
    res_short = ext.process_2d_floats(np.atleast_2d(x_short))
    assert np.isnan(res_short[0][0])


def test_count_above_below_and_range_count():
    import tsfresh.feature_extraction.feature_calculators as fc

    features = ["count_above-3.0", "count_below-3.0", "range_count-2-4"]
    ext = tsfast.Extractor(features)

    def check(x):
        x = np.asarray(x, dtype=np.float32)
        res = ext.process_2d_floats(np.atleast_2d(x))[0]
        assert np.allclose(res[0], fc.count_above(x, 3.0))
        assert np.allclose(res[1], fc.count_below(x, 3.0))
        assert np.allclose(res[2], fc.range_count(x, 2, 4))

    # Values exactly equal to the threshold/boundaries.
    check([1.0, 2.0, 3.0, 3.0, 4.0, 5.0])
    # Constant series.
    check([2.0, 2.0, 2.0, 2.0])
    # Series of length 1.
    check([3.0])


def test_power_bandwidth_positive_turning_variation():
    import tsfel.feature_extraction.features as F

    fs = 100.0
    rng = np.random.RandomState(7)
    features = ["power_bandwidth", "spectral_positive_turning", "spectral_variation"]
    ext = tsfast.Extractor(features)

    def check(x):
        x = np.asarray(x, dtype=np.float32)
        res = ext.process_2d_floats(np.atleast_2d(x))[0]
        assert np.allclose(res[0], F.power_bandwidth(x, fs), atol=1e-3)
        assert np.allclose(res[1], F.spectral_positive_turning(x, fs))
        assert np.allclose(res[2], F.spectral_variation(x, fs), atol=1e-5)

    t = np.arange(300)
    check(np.sin(2 * np.pi * 5.37 * t / fs))  # sine
    check(np.sin(2 * np.pi * (1 + 0.05 * t) * t / fs))  # chirp
    check(rng.randn(300))  # noise

    # Short window.
    t_short = np.arange(8)
    check(np.sin(2 * np.pi * 5.37 * t_short / fs))
    check(rng.randn(8))

    # Constant series: AC spectral content is zero mathematically, but
    # TSFEL's own reference value for it is float-noise-dependent (its
    # non-DC FFT bins don't reliably cancel to exactly zero at double
    # precision either, and how well they do varies with the exact value),
    # so comparing against it here would assert on TSFEL's rounding error
    # rather than on tsfast's correctness. Assert the mathematically correct
    # values directly instead.
    for x in (np.full(300, 3.0), np.full(8, -1.5)):
        x = np.asarray(x, dtype=np.float32)
        res = ext.process_2d_floats(np.atleast_2d(x))[0]
        assert np.allclose(res[0], 0.0)
        assert np.allclose(res[1], 0.0)
        assert np.allclose(res[2], 1.0)


def test_wavelet_abs_mean_std_var():
    import tsfel.feature_extraction.features as F

    fs = 100.0
    features = ["wavelet_abs_mean-3", "wavelet_std-3", "wavelet_var-3"]
    extractor = tsfast.Extractor(features)

    np.random.seed(0)
    x = np.random.randn(100).astype(np.float32)
    results = extractor.process_2d_floats(np.stack([x]))[0]

    assert np.allclose(results[0], F.wavelet_abs_mean(x, fs)["values"][3], atol=1e-3)
    assert np.allclose(results[1], F.wavelet_std(x, fs)["values"][3], atol=1e-3)
    assert np.allclose(results[2], F.wavelet_var(x, fs)["values"][3], atol=1e-3)

    # Constant series: the mexh wavelet's coefficients on a flat signal are
    # nonzero near the boundary (finite-support truncation), so TSFEL's own
    # values are nonzero too; compare against those rather than zero.
    const = np.full(50, 2.0, dtype=np.float32)
    results = extractor.process_2d_floats(np.stack([const]))[0]
    assert np.allclose(results[0], F.wavelet_abs_mean(const, fs)["values"][3], atol=1e-3)
    assert np.allclose(results[1], F.wavelet_std(const, fs)["values"][3], atol=1e-3)
    assert np.allclose(results[2], F.wavelet_var(const, fs)["values"][3], atol=1e-3)


def test_fourier_entropy_and_fft_aggregated():
    import tsfresh.feature_extraction.feature_calculators as fc

    def one(result):
        return list(result)[0][1]

    features = [
        "fourier_entropy-5",
        "fft_aggregated-centroid",
        "fft_aggregated-variance",
        "fft_aggregated-skew",
        "fft_aggregated-kurtosis",
    ]
    ext = tsfast.Extractor(features)

    def check(x):
        x = np.asarray(x, dtype=np.float32)
        res = ext.process_2d_floats(np.atleast_2d(x))[0]
        x64 = x.astype(np.float64)
        assert np.allclose(res[0], fc.fourier_entropy(x64, 5), rtol=1e-2, equal_nan=True)
        for i, aggtype in enumerate(["centroid", "variance", "skew", "kurtosis"], start=1):
            want = one(fc.fft_aggregated(x64, [{"aggtype": aggtype}]))
            assert np.allclose(res[i], want, atol=1e-3, equal_nan=True)

    rng = np.random.RandomState(3)
    check(rng.randn(300))
    check(np.sin(2 * np.pi * 5.37 * np.arange(200) / 100.0))

    # Constant series: Welch PSD is all zero (mean-subtracted per segment), so
    # tsfresh divides 0/0 into NaN; fft_aggregated's variance collapses below
    # its 0.5 cutoff too, so skew/kurtosis are NaN regardless of the DC level.
    check(np.full(300, 3.0))
    # Short window.
    check(np.full(8, -1.5))


def test_petrosian_fractal_dimension_and_average_power():
    from tsfel.feature_extraction.features import (
        average_power,
        petrosian_fractal_dimension,
    )

    features = ["petrosian_fractal_dimension", "average_power-100"]
    ext = tsfast.Extractor(features)

    def check(x):
        x = np.asarray(x, dtype=np.float32)
        res = ext.process_2d_floats(np.atleast_2d(x))[0]
        x64 = x.astype(np.float64)
        assert np.allclose(
            res[0], petrosian_fractal_dimension(x64), rtol=1e-2, equal_nan=True
        )
        assert np.allclose(
            res[1], average_power(x64, 100.0), rtol=1e-2, equal_nan=True
        )

    rng = np.random.RandomState(7)
    check(rng.randn(150))
    check(np.sin(2 * np.pi * 5.37 * np.arange(150) / 100.0))

    # Constant series: no sign changes, so petrosian is 1.0; average_power is
    # just energy scaled by fs/(n-1).
    check(np.full(64, 3.0))
    # Short window with a flat run then a step, to exercise the
    # flat-to-nonflat sign transition.
    check(np.array([1.0, 1.0, 2.0, 2.0, -1.0]))


def test_hist_mode_and_neighbourhood_peaks():
    from tsfel.feature_extraction.features import hist_mode, neighbourhood_peaks

    features = ["hist_mode", "hist_mode-3", "neighbourhood_peaks", "neighbourhood_peaks-2"]
    ext = tsfast.Extractor(features)
    # Defaults resolve to explicit parameters; neighbourhood_peaks is
    # tsfresh's number_peaks.
    assert ext.feature_names == [
        "hist_mode-10",
        "hist_mode-3",
        "number_peaks__n_10",
        "number_peaks__n_2",
    ]

    def check(x):
        x = np.asarray(x, dtype=np.float32)
        res = ext.process_2d_floats(np.atleast_2d(x))[0]
        x64 = x.astype(np.float64)
        assert np.isclose(res[0], hist_mode(x64, 10), rtol=1e-5, atol=1e-6)
        assert np.isclose(res[1], hist_mode(x64, 3), rtol=1e-5, atol=1e-6)
        assert res[2] == neighbourhood_peaks(x64, 10)
        assert res[3] == neighbourhood_peaks(x64, 2)

    rng = np.random.RandomState(5)
    check(rng.randn(300))
    check(np.sin(np.arange(200) / 3.0))
    # Few distinct levels: many values sit exactly on bin edges, and equal
    # neighbours (plateaus) must not count as peaks.
    for _ in range(50):
        check(rng.randint(0, 5, size=rng.randint(5, 80)))
    # Monotone runs exercise the peak scan's skip logic.
    check(np.arange(100.0))
    check(np.arange(100.0)[::-1])
    check(np.concatenate([np.arange(30.0), np.arange(30.0)[::-1], np.arange(30.0)]))
    # Constant series: histogram range widens to [c - 0.5, c + 0.5].
    check(np.full(64, 3.0))
    # Shorter than 2n + 1: no peaks.
    check(np.array([1.0, 3.0, 2.0, 5.0, 1.0]))


def test_spectrogram_mean_coeff():
    from tsfel.feature_extraction.features import spectrogram_mean_coeff

    def check(x, bins):
        x = np.asarray(x, dtype=np.float32)
        names = [f"spectrogram_mean_coeff-{k}-{bins}" for k in range(bins)]
        res = tsfast.Extractor(names).process_2d_floats(np.atleast_2d(x))[0]
        expected = np.zeros(bins)
        if len(x) >= 2:
            # TSFEL clamps bins to len // 2 + 1; the extra coefficients report 0.
            ref = spectrogram_mean_coeff(x.astype(np.float64), 100.0, bins)["values"]
            expected[: len(ref)] = ref
        atol = 1e-6 * max(np.abs(expected).max(), 1e-12)
        np.testing.assert_allclose(res, expected, rtol=1e-3, atol=atol)

    rng = np.random.RandomState(41)
    check(rng.randn(500), 32)
    check(rng.randn(300).cumsum(), 32)
    check(np.sin(2 * np.pi * 7.3 * np.arange(400) / 100.0), 16)
    for bins in (2, 3, 5, 64):
        check(rng.randn(200), bins)
    # Short windows: bins clamped to len // 2 + 1, single segment.
    for n in (1, 2, 3, 10, 61, 62, 63):
        check(rng.randn(n), 32)
    # Constant series: every segment is zero after detrending.
    check(np.full(100, 2.5), 32)
    # Default bins=32 and the name it is reported under.
    ext = tsfast.Extractor(["spectrogram_mean_coeff-4"])
    assert ext.feature_names == ["spectrogram_mean_coeff-4"]


def test_tsfresh_sample_entropy():
    import tsfresh.feature_extraction.feature_calculators as fc
    import tsfel.feature_extraction.features as F

    ext = tsfast.Extractor(["tsfresh_sample_entropy", "sample_entropy"])
    assert ext.feature_names == ["tsfresh_sample_entropy", "sample_entropy"]
    # tsfresh column-name alias.
    assert tsfast.Extractor(["value__sample_entropy"]).feature_names == [
        "tsfresh_sample_entropy"
    ]

    rng = np.random.RandomState(7)
    for x in [
        rng.randn(200),
        np.sin(np.arange(300) / 7) + 0.1 * rng.randn(300),
        rng.randint(0, 4, size=80),
        # Constant: r = 0 but every template still matches.
        np.full(50, 3.0),
        # So short that no length-3 template matches: tsfresh gives inf.
        np.array([0.0, 5.0, -3.0, 9.0]),
    ]:
        x = np.asarray(x, dtype=np.float32)
        got, tsfel_got = ext.process_2d_floats(x[None])[0]
        x64 = x.astype(np.float64)
        assert np.isclose(got, fc.sample_entropy(x64), rtol=1e-4, equal_nan=True)
        if np.std(x) > 0 and len(x) > 10:
            # The two definitions disagree, so each needs its own feature.
            ref = F.sample_entropy(x64, 2, 0.2 * np.std(x64))
            assert np.isclose(tsfel_got, ref, rtol=1e-4)
            assert not np.isclose(got, tsfel_got, rtol=1e-4)


def test_spectral_entropy_short_series_dc_bin():
    # TSFEL normalises by log2 of the number of non-zero power bins, and its DC
    # bin is exactly 0 only when numpy's f64 mean of the series is exact (then
    # x - mean cancels exactly). Counting it wrongly is off by
    # log2(N) / log2(N - 1) - 1: 2.2% at n = 32, so this checks to 0.1%.
    from tsfel.feature_extraction.features import spectral_entropy

    ext = tsfast.Extractor(["spectral_entropy"])

    def check(x, dc_zero):
        x = np.asarray(x, dtype=np.float32)
        x64 = x.astype(np.float64)
        assert (np.fft.rfft(x64 - np.mean(x64))[0] == 0) == dc_zero
        got = ext.process_2d_floats(x[None])[0, 0]
        assert got == pytest.approx(spectral_entropy(x64, 100.0), rel=1e-3)

    rng = np.random.RandomState(29)
    # Power-of-two lengths: the mean is always exact.
    for n in [16, 32, 64, 128]:
        check(rng.randn(n), dc_zero=True)
        check(np.cumsum(rng.randn(n)), dc_zero=True)
    # n = 45 = 9 * 5 (odd, so no Nyquist bin): integer data has an exact mean
    # iff 45 divides the sum.
    ints = rng.randint(-5, 5, size=45).astype(float)
    ints[0] -= ints.sum() % 45
    check(ints, dc_zero=True)
    # Inexact mean: the DC residue counts. (Small integers are a poor example
    # here: numpy's summation often rounds their residue away, see below.)
    check(rng.randn(45), dc_zero=False)
    check(rng.randn(60), dc_zero=False)


def _spectral_entropy_known_misses():
    rng = np.random.RandomState(29)
    for n in [16, 32, 64, 128]:
        rng.randn(n), rng.randn(n)
    nyquist = rng.randint(-5, 5, size=48).astype(float)
    nyquist[0] += 1 - nyquist.sum() % 3
    rng = np.random.RandomState(37)
    rng.randn(120)
    swallowed = rng.randint(-5, 5, size=90)[4:28].astype(float)
    return [
        pytest.param(nyquist, id="nyquist_residue"),
        pytest.param(swallowed, id="dc_residue_rounded_away"),
    ]


@pytest.mark.xfail(
    strict=True,
    reason="TSFEL counts float residue as a non-zero bin; tsfast predicts only "
    "the DC bin, from whether the f64 mean is exact. Missed: (1) an exactly-zero "
    "Nyquist bin (alternating sum 0) that numpy's inexact x - mean leaves "
    "residue in; (2) an inexact mean whose DC residue numpy's summation rounds "
    "to exactly 0. Both depend on pocketfft's rounding order.",
)
@pytest.mark.parametrize("x", _spectral_entropy_known_misses())
def test_spectral_entropy_residue_bins_known_misses(x):
    from tsfel.feature_extraction.features import spectral_entropy

    x = np.asarray(x, dtype=np.float32)
    got = tsfast.Extractor(["spectral_entropy"]).process_2d_floats(x[None])[0, 0]
    assert got == pytest.approx(spectral_entropy(x.astype(np.float64), 100.0), rel=1e-3)


def test_fresh_fft_meta_feature_is_ignored():
    # `fresh-N` only configures the sliding engine; the static engine always
    # runs a fresh FFT. Accepted so one feature list works with every engine.
    features = ["mean", "spectral_entropy"]
    x = np.random.RandomState(43).randn(2, 50).astype(np.float32)
    ext = tsfast.Extractor(features + ["fresh-4"])
    assert ext.feature_names == features
    np.testing.assert_array_equal(
        ext.process_2d_floats(x), tsfast.Extractor(features).process_2d_floats(x)
    )
    with pytest.raises(ValueError, match="fresh-N"):
        tsfast.Extractor(["mean", "fresh-0"])
