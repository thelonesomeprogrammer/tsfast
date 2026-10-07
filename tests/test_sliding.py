import pytest
import numpy as np
from tsfast._tsfast import SlidingExtractor
from helpers import frame
import tsfast


def test_sliding_multiple_columns():
    features = [
        "mean",
        "total_sum",
        "mean_second_derivative_central",
        "large_standard_deviation-0.05",
        "symmetry_looking-0.05",
        "ratio_beyond_r_sigma-1.0",
        "index_mass_quantile-0.5",
        "c3-1",
    ]
    n_cols = 2
    window_size = 2
    stride = 1
    extractor = SlidingExtractor(features, n_cols, window_size, stride)

    # Update 1: 3 rows
    # Col 1: [1.0, 2.0, 3.0]
    # Col 2: [10.0, 20.0, 30.0]
    data1 = np.stack([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])

    result1 = frame(extractor, extractor.update(data1))
    # Output rows expected:
    # 1. Window [1.0, 2.0], [10.0, 20.0] -> col1 mean: 1.5, max: 2.0, sum: 3.0
    #                                      col2 mean: 15.0, max: 20.0, sum: 30.0
    # 2. Window [2.0, 3.0], [20.0, 30.0] -> col1 mean: 2.5, max: 3.0, sum: 5.0
    #                                      col2 mean: 25.0, max: 30.0, sum: 50.0

    # update() returns (n_series, n_windows, n_features); frame() flattens it
    # series-major: every window of col1, then every window of col2.

    # Let's just calculate expected flattened values
    # result1 will have n_cols * 2 rows = 4 rows
    assert len(result1) == 4

    # Col1: 2 windows
    assert np.allclose(result1.iloc[0].iloc[:2], [1.5, 3.0])
    assert np.allclose(result1.iloc[1].iloc[:2], [2.5, 5.0])

    # Col2: 2 windows
    assert np.allclose(result1.iloc[2].iloc[:2], [15.0, 30.0])
    assert np.allclose(result1.iloc[3].iloc[:2], [25.0, 50.0])


def test_basic_features():
    features = [
        "mean",
        "total_sum",
        "min_value",
        "max_value",
        "energy",
        "root_mean_square",
        "length",
        "variance_larger_than_standard_deviation",
    ]
    n_cols = 1
    window_size = 3
    stride = 1
    extractor = SlidingExtractor(features, n_cols, window_size, stride)

    x = [1.0, 2.0, 3.0, 4.0]
    data = np.stack([np.asarray(x, dtype=np.float32)])
    res = frame(extractor, extractor.update(data))

    # Window 1: [1.0, 2.0, 3.0]
    # Window 2: [2.0, 3.0, 4.0]
    assert len(res) == 2

    w1 = np.array([1.0, 2.0, 3.0])
    assert np.allclose(res.iloc[0]["mean"], np.mean(w1))
    assert np.allclose(res.iloc[0]["total_sum"], np.sum(w1))
    assert np.allclose(res.iloc[0]["min_value"], np.min(w1))
    assert np.allclose(res.iloc[0]["max_value"], np.max(w1))
    assert np.allclose(res.iloc[0]["energy"], np.sum(w1**2))
    assert np.allclose(res.iloc[0]["root_mean_square"], np.sqrt(np.mean(w1**2)))

    w2 = np.array([2.0, 3.0, 4.0])
    assert np.allclose(res.iloc[1]["mean"], np.mean(w2))
    assert np.allclose(res.iloc[1]["total_sum"], np.sum(w2))
    assert np.allclose(res.iloc[1]["min_value"], np.min(w2))
    assert np.allclose(res.iloc[1]["max_value"], np.max(w2))
    assert np.allclose(res.iloc[1]["energy"], np.sum(w2**2))
    assert np.allclose(res.iloc[1]["root_mean_square"], np.sqrt(np.mean(w2**2)))


def test_empty_batch():
    features = ["mean"]
    extractor = SlidingExtractor(features, 1, 2, 1)
    empty_data = np.stack([[]])
    result = extractor.update(empty_data)
    assert result.shape[1] == 0


def test_stride_behavior():
    features = ["mean"]
    n_cols = 1
    window_size = 4
    stride = 2
    extractor = SlidingExtractor(features, n_cols, window_size, stride)

    # Input size 6: indices [0, 1, 2, 3, 4, 5]
    # Window 1 (len 4): [0, 1, 2, 3]
    # Window 2 (len 4, strided by 2): [2, 3, 4, 5]
    x = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    data = np.stack([np.asarray(x, dtype=np.float32)])
    res = frame(extractor, extractor.update(data))

    assert len(res) == 2
    assert np.allclose(res.iloc[0]["mean"], np.mean([0.0, 1.0, 2.0, 3.0]))
    assert np.allclose(res.iloc[1]["mean"], np.mean([2.0, 3.0, 4.0, 5.0]))


if __name__ == "__main__":
    pytest.main([__file__])


def test_sliding_invalid_feature():
    with pytest.raises(ValueError):
        SlidingExtractor(["invalid_feature"], 1, 3, 1)


def test_sliding_paa():
    features = ["paa-2-0", "paa-2-1"]
    n_cols = 1
    window_size = 4
    stride = 1
    extractor = SlidingExtractor(features, n_cols, window_size, stride)

    x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    data = np.stack([np.asarray(x, dtype=np.float32)])
    res = frame(extractor, extractor.update(data))

    # window 1: [1, 2, 3, 4] -> paa-2-0 is mean(1,2)=1.5, paa-2-1 is mean(3,4)=3.5
    # window 2: [2, 3, 4, 5] -> paa-2-0 is mean(2,3)=2.5, paa-2-1 is mean(4,5)=4.5
    assert len(res) == 3
    assert np.allclose(res.iloc[0]["paa-2-0"], 1.5)
    assert np.allclose(res.iloc[0]["paa-2-1"], 3.5)
    assert np.allclose(res.iloc[1]["paa-2-0"], 2.5)
    assert np.allclose(res.iloc[1]["paa-2-1"], 4.5)


def test_sliding_higher_moments():
    features = ["mean", "std_dev", "skewness", "kurtosis"]
    extractor = SlidingExtractor(features, 1, 5, 1)

    x = np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.float32)
    data = np.stack([np.asarray(x, dtype=np.float32)])
    res = frame(extractor, extractor.update(data))

    assert len(res) == 3
    # window 1: [1, 2, 3, 4, 5]
    w1 = x[:5]
    assert np.allclose(res.iloc[0]["mean"], np.mean(w1))
    assert np.allclose(res.iloc[0]["std_dev"], np.std(w1))  # ddof=0, as tsfresh/TSFEL

    from scipy.stats import kurtosis, skew

    expected_skew = skew(w1, bias=False)  # tsfresh (pandas) skewness
    expected_kurt = kurtosis(w1, fisher=True, bias=False)

    assert np.allclose(res.iloc[0]["skewness"], expected_skew, atol=1e-5)
    assert np.allclose(res.iloc[0]["kurtosis"], expected_kurt, atol=1e-5)


def test_sliding_duplicate_features():
    from tsfast._tsfast import SlidingExtractor
    import pandas as pd
    from tsfresh.feature_extraction.feature_calculators import (
        has_duplicate,
        has_duplicate_max,
        has_duplicate_min,
    )

    features = ["has_duplicate", "has_duplicate_max", "has_duplicate_min"]
    n_cols = 1
    window_size = 3
    stride = 1
    extractor = SlidingExtractor(features, n_cols, window_size, stride)

    # 5 rows total
    x = np.array([1.0, 5.0, 3.0, 5.0, 1.0], dtype=np.float32)
    data = np.stack([x])

    result = frame(extractor, extractor.update(data))

    # Windows:
    # 1: [1.0, 5.0, 3.0]
    # 2: [5.0, 3.0, 5.0]
    # 3: [3.0, 5.0, 1.0]

    series1 = pd.Series([1.0, 5.0, 3.0])
    series2 = pd.Series([5.0, 3.0, 5.0])
    series3 = pd.Series([3.0, 5.0, 1.0])

    expected_has_duplicate = [
        1.0 if has_duplicate(series1) else 0.0,
        1.0 if has_duplicate(series2) else 0.0,
        1.0 if has_duplicate(series3) else 0.0,
    ]
    expected_has_duplicate_max = [
        1.0 if has_duplicate_max(series1) else 0.0,
        1.0 if has_duplicate_max(series2) else 0.0,
        1.0 if has_duplicate_max(series3) else 0.0,
    ]
    expected_has_duplicate_min = [
        1.0 if has_duplicate_min(series1) else 0.0,
        1.0 if has_duplicate_min(series2) else 0.0,
        1.0 if has_duplicate_min(series3) else 0.0,
    ]

    np.testing.assert_allclose(result["has_duplicate"].values, expected_has_duplicate)
    np.testing.assert_allclose(
        result["has_duplicate_max"].values, expected_has_duplicate_max
    )
    np.testing.assert_allclose(
        result["has_duplicate_min"].values, expected_has_duplicate_min
    )


def test_reoccurring_ratios_sliding():
    import tsfast
    import numpy as np

    x = np.array([1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0], dtype=np.float32)

    features = [
        "percentage_of_reoccurring_datapoints_to_all_datapoints",
        "percentage_of_reoccurring_values_to_all_values",
        "ratio_value_number_to_time_series_length",
    ]

    extractor = tsfast.SlidingExtractor(features, 1, 5, 1)
    batch = np.stack([x])
    out = extractor.update(batch)
    results = frame(extractor, out)

    # window 0: [1, 2, 2, 3, 3]
    # len=5, values={1,2,3} (3 unique), reoccur_dp=4 (two 2s, two 3s), reoccur_val=2 (2 and 3)
    # res: [4/5, 2/3, 3/5]
    assert np.allclose(results.iloc[0].values, [4.0 / 5.0, 2.0 / 3.0, 3.0 / 5.0])

    # window 1: [2, 2, 3, 3, 3]
    # len=5, values={2,3} (2 unique), reoccur_dp=5 (two 2s, three 3s), reoccur_val=2 (2 and 3)
    # res: [5/5, 2/2, 2/5]
    assert np.allclose(results.iloc[1].values, [5.0 / 5.0, 2.0 / 2.0, 2.0 / 5.0])

    # window 2: [2, 3, 3, 3, 4]
    # len=5, values={2,3,4} (3 unique), reoccur_dp=3 (three 3s), reoccur_val=1 (3)
    # res: [3/5, 1/3, 3/5]
    assert np.allclose(results.iloc[2].values, [3.0 / 5.0, 1.0 / 3.0, 3.0 / 5.0])


def test_ecdf_pk_centroid_sliding():
    import tsfel
    import tsfast

    x = np.array([1.0, -2.0, 3.0, 4.0, 5.0, 1.0, 0.0, 5.0, -1.0, 3.0], dtype=np.float32)
    features = [
        "ecdf-10",
        "ecdf-3",
        "pk_pk_distance",
        "calc_centroid-100",
        "calc_centroid-50",
        "negative_turning",
        "positive_turning",
    ]
    window_size = 7
    extractor = tsfast.SlidingExtractor(features, n_cols=1, window_size=window_size)

    batch = np.stack([x])
    df = frame(extractor, extractor.update(batch))
    results = df.iloc[-1].values

    windowed_x = x[-window_size:]
    tsfel_neg = tsfel.feature_extraction.features.negative_turning(windowed_x)
    tsfel_pos = tsfel.feature_extraction.features.positive_turning(windowed_x)
    tsfel_ecdf_10 = tsfel.feature_extraction.features.ecdf(windowed_x, d=10)
    tsfel_ecdf_3 = tsfel.feature_extraction.features.ecdf(windowed_x, d=3)
    tsfel_pk = tsfel.feature_extraction.features.pk_pk_distance(windowed_x)
    tsfel_centroid = tsfel.feature_extraction.features.calc_centroid(windowed_x, fs=100)
    tsfel_centroid_50 = tsfel.feature_extraction.features.calc_centroid(
        windowed_x, fs=50
    )

    assert np.allclose(results[0], min(10.0 / window_size, 1.0))
    assert np.allclose(results[1], min(3.0 / window_size, 1.0))
    assert np.allclose(results[2], tsfel_pk)
    assert np.allclose(results[3], tsfel_centroid)
    assert np.allclose(results[4], tsfel_centroid_50)
    assert np.allclose(results[5], tsfel_neg)
    assert np.allclose(results[6], tsfel_pos)


def test_sliding_invalid_type():
    # Verify that passing non-float arrays safely raises a TypeError instead of crashing
    x = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    features = ["mean", "std_dev"]
    extractor = SlidingExtractor(features, 1, 2, 1)
    batch = np.stack([x])
    with pytest.raises(TypeError, match="expected a float32 or float64 numpy array"):
        extractor.update(batch)


def test_median_diff_sliding():
    import tsfast

    data = np.random.randn(200).astype(np.float32)
    batch = np.stack([data])

    features = ["median_diff", "median_abs_diff"]
    extractor = tsfast.SlidingExtractor(features, 1, 100, 50)

    import tsfel

    res = frame(extractor, extractor.update(batch))

    # First window
    w1 = data[:100]
    assert np.allclose(
        res.iloc[0].values[0], tsfel.feature_extraction.features.median_diff(w1)
    )
    assert np.allclose(
        res.iloc[0].values[1], tsfel.feature_extraction.features.median_abs_diff(w1)
    )

    # Second window
    w2 = data[50:150]
    assert np.allclose(
        res.iloc[1].values[0], tsfel.feature_extraction.features.median_diff(w2)
    )
    assert np.allclose(
        res.iloc[1].values[1], tsfel.feature_extraction.features.median_abs_diff(w2)
    )


def test_sliding_energy_ratio_by_chunks():
    import tsfast
    import numpy as np

    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], dtype=np.float32)
    features = [
        "energy_ratio_by_chunks_num_segments_3__segment_focus_0",
        "energy_ratio_by_chunks_num_segments_3__segment_focus_1",
        "energy_ratio_by_chunks_num_segments_3__segment_focus_2",
    ]
    window_size = 6
    extractor = tsfast.SlidingExtractor(features, n_cols=1, window_size=window_size)
    batch = np.stack([x])
    df = frame(extractor, extractor.update(batch))
    res = df.iloc[-1].values
    assert np.isclose(res[0], (4**2 + 5**2) / sum([i**2 for i in [4, 5, 6, 7, 8, 9]]))


def test_sliding_permutation_entropy_and_value_count():
    import numpy as np
    import math
    import tsfast

    data = np.array(
        [4.0, 7.0, 9.0, 10.0, 6.0, 11.0, 3.0, 3.0, np.nan, 3.0, np.nan],
        dtype=np.float32,
    )
    features = ["permutation_entropy-1-3", "value_count-3.0"]

    batch = np.stack([data])
    extractor = tsfast.SlidingExtractor(features, n_cols=1, window_size=5, stride=1)
    res = extractor.update(batch)

    from tsfresh.feature_extraction.feature_calculators import (
        permutation_entropy,
        value_count,
    )

    for i in range(len(data) - 5 + 1):
        window = data[i : i + 5]
        pe = permutation_entropy(window, tau=1, dimension=3)
        vc = value_count(window, 3.0)

        if math.isnan(pe):
            assert math.isnan(res[0, i, 0])
        else:
            np.testing.assert_allclose(res[0, i, 0], pe, rtol=1e-5)
        assert res[0, i, 1] == vc


def test_sliding_fractal_dimensions():
    import numpy as np
    from tsfast._tsfast import SlidingExtractor
    from tsfel.feature_extraction.features import higuchi_fractal_dimension
    import warnings

    warnings.filterwarnings("ignore")

    features = ["higuchi_fd"]
    n_cols = 1
    window_size = 200
    stride = 100
    extractor = SlidingExtractor(features, n_cols, window_size, stride)

    x = np.random.randn(300).astype(np.float32)
    batch = np.stack([x])

    res = frame(extractor, extractor.update(batch))

    # Window 1: x[0:200]
    # Window 2: x[100:300]
    h1 = higuchi_fractal_dimension(x[0:200])
    h2 = higuchi_fractal_dimension(x[100:300])

    assert np.allclose(res.iloc[0, 0], h1, atol=5e-2)
    assert np.allclose(res.iloc[1, 0], h2, atol=5e-2)
    print("test_sliding_fractal_dimensions passed!")


@pytest.mark.parametrize(
    "window_size,stride", [(40, 1), (37, 1), (10, 3), (64, 16), (5, 5)]
)
def test_sliding_min_max_every_window(window_size, stride):
    # Regression: incremental updates used to wipe the min/max queue.
    x = np.random.RandomState(window_size * stride).randn(300).astype(np.float32)
    extractor = SlidingExtractor(
        ["min", "max", "pk_pk_distance"], 1, window_size, stride
    )
    rows = []
    for chunk in np.array_split(x, 7):
        batch = np.stack([chunk])
        rows.extend(frame(extractor, extractor.update(batch)).values)
    assert len(rows) == (len(x) - window_size) // stride + 1
    for k, row in enumerate(rows):
        w = x[k * stride : k * stride + window_size]
        assert np.allclose(row, [w.min(), w.max(), w.max() - w.min()], atol=1e-6), k


@pytest.mark.parametrize("case", ["randn", "exponential", "constant"])
def test_shape_features_sliding(case):
    from test_tsfast import SHAPE_CASES, SHAPE_FEATURES, shape_features_reference

    x = SHAPE_CASES[case].astype(np.float32)
    window_size, stride = 40, 3
    extractor = SlidingExtractor(SHAPE_FEATURES, 1, window_size, stride)
    batch = np.stack([x])
    rows = frame(extractor, extractor.update(batch)).values
    for k, row in enumerate(rows):
        w = x[k * stride : k * stride + window_size]
        assert np.allclose(
            row, shape_features_reference(w), atol=1e-5, equal_nan=True
        ), k


def test_sliding_output_shape_and_names():
    features = ["mean", "max_value"]
    extractor = SlidingExtractor(features, 2, 3, 1)
    assert extractor.feature_names == features
    data = np.arange(10, dtype=np.float32).reshape(2, 5)
    out = extractor.update(data)
    # 5 samples in a window of 3: windows end at samples 3, 4 and 5.
    assert out.dtype == np.float32 and out.shape == (2, 3, 2)
    np.testing.assert_allclose(out[1, :, 0], [6.0, 7.0, 8.0])
    np.testing.assert_array_equal(out[1, :, 1], [7.0, 8.0, 9.0])
    assert extractor.update(np.zeros((2, 0), dtype=np.float32)).shape == (2, 0, 2)
    # float64 is accepted and converted.
    np.testing.assert_array_equal(extractor.update(np.array([[5.0], [10.0]]))[:, 0, 1], [5.0, 10.0])

def test_sliding_mse():
    import numpy as np
    import tsfast
    from tsfel.feature_extraction.features import mse

    np.random.seed(42)
    signal = np.random.rand(500).astype(np.float32)

    e = tsfast.SlidingExtractor(["mse-3"], 1, 200, 100)
    res = e.update(np.array([signal]))

    # Check one window
    window = signal[100:300]
    tol = 0.2 * np.std(window)
    ref = mse(window, m=3, maxscale=None, tolerance=tol)
    assert np.isclose(res[0][1][0], ref, atol=1e-2)

def test_ecdf_features():
    # Normal case with ties
    x = np.array([1, 2, 2, 2, 5, 5, 7, 8, 9, 10], dtype=np.float32)
    features = ["ecdf_percentile-0.5", "ecdf_percentile_count-0.5", "ecdf_slope-0.2-0.5"]

    import tsfel
    ext = tsfast.SlidingExtractor(features, 1, 5, 1)
    for i in range(len(x)):
        arr = x[:i+1]
        out = ext.update(np.stack([arr[-1:]]).T)
        if i >= 4:
            window = x[i-4:i+1]


            ref_perc = tsfel.feature_extraction.features.ecdf_percentile(window, [0.5])
            if np.isscalar(ref_perc):
                ref_perc = float(ref_perc)
            else:
                ref_perc = float(ref_perc[0])

            if np.max(window) == np.min(window):
                ref_count = float(len(window))
            else:
                ref_count = float(np.sum(window <= ref_perc))

            try:
                ref_slope = tsfel.feature_extraction.features.ecdf_slope(window, 0.2, 0.5)
            except Exception:
                ref_slope = np.nan

            if np.isnan(ref_perc):
                assert np.isnan(out[0][0][0])
            else:
                assert np.allclose(out[0][0][0], ref_perc)

            if np.isnan(ref_count):
                assert np.isnan(out[0][0][1])
            else:
                assert np.allclose(out[0][0][1], ref_count)

            if np.isnan(ref_slope):
                assert np.isnan(out[0][0][2])
            elif np.isinf(ref_slope):
                assert np.isinf(out[0][0][2])
            else:
                assert np.allclose(out[0][0][2], ref_slope)

def test_sliding_maximum_fractal_length():
    import numpy as np
    import tsfast
    from tsfel.feature_extraction.features import maximum_fractal_length

    np.random.seed(42)
    x = np.cumsum(np.random.randn(300)).astype(np.float32)

    features = ["maximum_fractal_length"]
    ext = tsfast.SlidingExtractor(features, n_cols=1, window_size=200, stride=100)

    res1 = ext.update(x[0:200].reshape(1, -1))
    res2 = ext.update(x[200:300].reshape(1, -1))

    mfl1 = maximum_fractal_length(x[0:200])
    mfl2 = maximum_fractal_length(x[100:300])

    np.testing.assert_allclose(res1[0][0], mfl1, atol=1e-5)
    np.testing.assert_allclose(res2[0][0], mfl2, atol=1e-5)


def test_sliding_count_above_below_and_range_count():
    import tsfresh.feature_extraction.feature_calculators as fc

    features = ["count_above-3.0", "count_below-3.0", "range_count-2-4"]

    def check(x, window_size, stride=1):
        x = np.asarray(x, dtype=np.float32)
        ext = SlidingExtractor(features, 1, window_size, stride)
        res = frame(ext, ext.update(x.reshape(1, -1)))
        for w in range(len(res)):
            start = w * stride
            window = x[start : start + window_size]
            assert np.allclose(res.iloc[w]["count_above-3"], fc.count_above(window, 3.0))
            assert np.allclose(res.iloc[w]["count_below-3"], fc.count_below(window, 3.0))
            assert np.allclose(res.iloc[w]["range_count-2-4"], fc.range_count(window, 2, 4))

    # Values exactly equal to the threshold/boundaries, strided across windows.
    check([1.0, 2.0, 3.0, 3.0, 4.0, 5.0], window_size=3)
    # Constant series.
    check([2.0, 2.0, 2.0, 2.0, 2.0], window_size=3)
    # Window of length 1.
    check([1.0, 2.0, 3.0, 4.0, 5.0], window_size=1)


def test_sliding_power_bandwidth_positive_turning_variation():
    import tsfel.feature_extraction.features as F

    fs = 100.0
    features = ["power_bandwidth", "spectral_positive_turning", "spectral_variation"]

    def check(x, window_size, stride=1, chunk=7):
        x = np.asarray(x, dtype=np.float32)
        ext = SlidingExtractor(features, 1, window_size, stride)
        # Feed the series in small chunks to exercise many incremental sliding
        # DFT updates rather than one fresh FFT per window.
        res = frame(ext, np.concatenate(
            [ext.update(x[i : i + chunk].reshape(1, -1)) for i in range(0, len(x), chunk)],
            axis=1,
        ))
        for w in range(len(res)):
            start = w * stride
            window = x[start : start + window_size]
            assert np.allclose(res.iloc[w]["power_bandwidth"], F.power_bandwidth(window, fs), atol=1e-3)
            assert np.allclose(res.iloc[w]["spectral_positive_turning"], F.spectral_positive_turning(window, fs))
            assert np.allclose(res.iloc[w]["spectral_variation"], F.spectral_variation(window, fs), atol=1e-5)

    t = np.arange(400)
    rng = np.random.RandomState(11)
    check(np.sin(2 * np.pi * 5.37 * t / fs), window_size=64)  # sine, long run of updates
    check(np.sin(2 * np.pi * (1 + 0.05 * t) * t / fs), window_size=64)  # chirp
    check(rng.randn(400), window_size=64)  # noise

    # Short window.
    check(rng.randn(40), window_size=8)

    # Constant series: every window is flat, so AC spectral content is zero
    # mathematically, but TSFEL's own reference value for it is
    # float-noise-dependent (see test_tsfast.py's analogous case), so assert
    # the mathematically correct values directly instead of against TSFEL.
    x = np.full(400, -2.0, dtype=np.float32)
    ext = SlidingExtractor(features, 1, 64, 1)
    res = frame(ext, ext.update(x.reshape(1, -1)))
    assert np.allclose(res["power_bandwidth"].values, 0.0)
    assert np.allclose(res["spectral_positive_turning"].values, 0.0)
    assert np.allclose(res["spectral_variation"].values, 1.0)


def test_sliding_wavelet_abs_mean_std_var():
    import tsfel.feature_extraction.features as F

    fs = 100.0
    features = ["wavelet_abs_mean-3", "wavelet_std-3", "wavelet_var-3"]
    window_size = 50

    rng = np.random.RandomState(7)
    x = rng.randn(120).astype(np.float32)
    ext = SlidingExtractor(features, 1, window_size)
    df = frame(ext, ext.update(x.reshape(1, -1)))

    for w in range(len(df)):
        window = x[w : w + window_size]
        tsfel_abs_mean = F.wavelet_abs_mean(window, fs)["values"][3]
        tsfel_std = F.wavelet_std(window, fs)["values"][3]
        tsfel_var = F.wavelet_var(window, fs)["values"][3]
        assert np.allclose(df.iloc[w]["wavelet_abs_mean-3"], tsfel_abs_mean, atol=1e-3)
        assert np.allclose(df.iloc[w]["wavelet_std-3"], tsfel_std, atol=1e-3)
        assert np.allclose(df.iloc[w]["wavelet_var-3"], tsfel_var, atol=1e-3)

    # Constant window: the mexh wavelet's coefficients on a flat signal are
    # nonzero near the boundary (finite-support truncation), so TSFEL's own
    # values are nonzero too; compare against those rather than zero.
    const = np.full(window_size, 3.0, dtype=np.float32)
    ext = SlidingExtractor(features, 1, window_size)
    res = frame(ext, ext.update(const.reshape(1, -1)))
    assert np.allclose(res["wavelet_abs_mean-3"].values, F.wavelet_abs_mean(const, fs)["values"][3], atol=1e-3)
    assert np.allclose(res["wavelet_std-3"].values, F.wavelet_std(const, fs)["values"][3], atol=1e-3)
    assert np.allclose(res["wavelet_var-3"].values, F.wavelet_var(const, fs)["values"][3], atol=1e-3)
