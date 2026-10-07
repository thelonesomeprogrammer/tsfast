import pytest
import numpy as np
from tsfast._tsfast import ExpandingExtractor
from helpers import frame
import tsfast


def test_expanding_multiple_columns():
    features = [
        "mean",
        "max_value",
        "total_sum",
        "mean_second_derivative_central",
        "large_standard_deviation-0.05",
        "symmetry_looking-0.05",
        "ratio_beyond_r_sigma-1.0",
        "index_mass_quantile-0.5",
        "c3-1",
    ]
    n_cols = 2
    extractor = ExpandingExtractor(features, n_cols)

    # Update 1
    data1 = np.stack([[1.0, 2.0], [10.0, 20.0]])

    result1 = frame(extractor, extractor.update(data1))
    # col1 mean: 1.5, max: 2.0, sum: 3.0
    # col2 mean: 15.0, max: 20.0, sum: 30.0

    assert np.allclose(result1.iloc[0].iloc[:3], [1.5, 2.0, 3.0])
    assert np.allclose(result1.iloc[1].iloc[:3], [15.0, 20.0, 30.0])

    # Update 2
    data2 = np.stack([[3.0], [30.0]])

    result2 = frame(extractor, extractor.update(data2))
    # col1 mean: (1+2+3)/3 = 2.0, max: 3.0, sum: 6.0
    # col2 mean: (10+20+30)/3 = 20.0, max: 30.0, sum: 60.0

    assert np.allclose(result2.iloc[0].iloc[:3], [2.0, 3.0, 6.0])
    assert np.allclose(result2.iloc[1].iloc[:3], [20.0, 30.0, 60.0])


def test_expanding_all_basic_features():
    features = [
        "mean",
        "total_sum",
        "min_value",
        "max_value",
        "energy",
        "root_mean_square",
        "mean_abs_change",
        "mean_change",
        "length",
        "variance_larger_than_standard_deviation",
    ]
    n_cols = 1
    extractor = ExpandingExtractor(features, n_cols)

    x = []
    for val in [1.0, 2.0, -1.0, 5.0]:
        x.append(val)
        data = np.stack([[val]])
        res = frame(extractor, extractor.update(data)).iloc[0]

        arr = np.array(x)
        assert np.allclose(res["mean"], np.mean(arr))
        assert np.allclose(res["total_sum"], np.sum(arr))
        assert np.allclose(res["min_value"], np.min(arr))
        assert np.allclose(res["max_value"], np.max(arr))
        assert np.allclose(res["energy"], np.sum(arr**2))
        assert np.allclose(res["root_mean_square"], np.sqrt(np.mean(arr**2)))

        if len(x) > 1:
            diffs = np.diff(arr)
            assert np.allclose(res["mean_abs_change"], np.mean(np.abs(diffs)))
            assert np.allclose(res["mean_change"], np.mean(diffs))


def test_expanding_empty_batch():
    features = ["mean"]
    extractor = ExpandingExtractor(features, 1)
    empty_data = np.stack([[]])
    result = extractor.update(empty_data)
    assert result.shape[0] == 0


def test_expanding_paa():
    # PAA in expanding window is tricky because boundaries change.
    # Current implementation recalculates boundaries based on total N.
    features = ["paa-2-0", "paa-2-1"]
    extractor = ExpandingExtractor(features, 1)

    # Update 1: [1, 2] -> N=2. paa-2-0: [1], paa-2-1: [2]
    data1 = np.stack([[1.0, 2.0]])
    res1 = frame(extractor, extractor.update(data1)).iloc[0]
    assert np.allclose(res1["paa-2-0"], 1.0)
    assert np.allclose(res1["paa-2-1"], 2.0)

    # Update 2: add [3, 4] -> N=4. paa-2-0: [1, 2] -> mean 1.5, paa-2-1: [3, 4] -> mean 3.5
    data2 = np.stack([[3.0, 4.0]])
    res2 = frame(extractor, extractor.update(data2)).iloc[0]
    assert np.allclose(res2["paa-2-0"], 1.5)
    assert np.allclose(res2["paa-2-1"], 3.5)


def test_expanding_higher_moments():
    features = ["mean", "std_dev", "skewness", "kurtosis"]
    extractor = ExpandingExtractor(features, 1)

    # Use a larger dataset for skew/kurtosis to be more stable
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float32)

    # Update with whole array
    data = np.stack([np.asarray(x, dtype=np.float32)])
    res = frame(extractor, extractor.update(data)).iloc[0]

    from scipy.stats import kurtosis, skew

    assert np.allclose(res["mean"], np.mean(x))
    assert np.allclose(res["std_dev"], np.std(x))  # ddof=0, as tsfresh/TSFEL
    expected_skew = skew(x, bias=False)  # tsfresh (pandas) skewness
    expected_kurt = kurtosis(x, fisher=True, bias=False)
    assert np.allclose(res["skewness"], expected_skew, atol=1e-5)
    assert np.allclose(res["kurtosis"], expected_kurt, atol=1e-5)


def test_expanding_c3():
    # c3-lag: mean(x[t] * x[t-lag] * x[t-2*lag])
    features = ["c3-1", "c3-2"]
    extractor = ExpandingExtractor(features, 1)

    x = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)
    data = np.stack([np.asarray(x, dtype=np.float32)])
    res = frame(extractor, extractor.update(data)).iloc[0]

    # c3-1: N=6. lags are at t=2, 3, 4, 5 (since we need t-2*lag >= 0)
    # t=2: x[2]*x[1]*x[0] = 3*2*1 = 6
    # t=3: x[3]*x[2]*x[1] = 4*3*2 = 24
    # t=4: x[4]*x[3]*x[2] = 5*4*3 = 60
    # t=5: x[5]*x[4]*x[3] = 6*5*4 = 120
    # mean: (6+24+60+120)/4 = 210/4 = 52.5
    assert np.allclose(res["c3-1"], 52.5)

    # c3-2: N=6. lag is 2. 2*lag=4. t=4, 5.
    # t=4: x[4]*x[2]*x[0] = 5*3*1 = 15
    # t=5: x[5]*x[3]*x[1] = 6*4*2 = 48
    # mean: (15+48)/2 = 63/2 = 31.5
    assert np.allclose(res["c3-2"], 31.5)


if __name__ == "__main__":
    pytest.main([__file__])


def test_expanding_duplicate_features():
    from tsfast._tsfast import ExpandingExtractor
    import pandas as pd
    from tsfresh.feature_extraction.feature_calculators import (
        has_duplicate,
        has_duplicate_max,
        has_duplicate_min,
    )

    features = ["has_duplicate", "has_duplicate_max", "has_duplicate_min"]
    n_cols = 1
    extractor = ExpandingExtractor(features, n_cols)

    # Update 1
    x1 = np.array([1.0, 5.0, 3.0], dtype=np.float32)
    data1 = np.stack([x1])

    result1 = frame(extractor, extractor.update(data1))

    # Update 2
    x2 = np.array([4.0, 5.0, 1.0, 2.0], dtype=np.float32)
    data2 = np.stack([x2])

    result2 = frame(extractor, extractor.update(data2))

    full_series = np.concatenate([x1, x2])

    # Validate result1 (prefix chunks)
    prefix1 = x1
    series1 = pd.Series(prefix1)

    assert result1.iloc[0].iloc[0] == (1.0 if has_duplicate(series1) else 0.0)
    assert result1.iloc[0].iloc[1] == (1.0 if has_duplicate_max(series1) else 0.0)
    assert result1.iloc[0].iloc[2] == (1.0 if has_duplicate_min(series1) else 0.0)

    # Validate result2 (prefix chunks)
    series2 = pd.Series(full_series)

    assert result2.iloc[0].iloc[0] == (1.0 if has_duplicate(series2) else 0.0)
    assert result2.iloc[0].iloc[1] == (1.0 if has_duplicate_max(series2) else 0.0)
    assert result2.iloc[0].iloc[2] == (1.0 if has_duplicate_min(series2) else 0.0)


def test_reoccurring_ratios_expanding():
    import tsfast
    import numpy as np

    x = np.array([1.0, 2.0, 2.0, 3.0, 3.0], dtype=np.float32)

    features = [
        "percentage_of_reoccurring_datapoints_to_all_datapoints",
        "percentage_of_reoccurring_values_to_all_values",
        "ratio_value_number_to_time_series_length",
    ]

    extractor = tsfast.ExpandingExtractor(features, 1)
    batch = np.stack([x])
    out = extractor.update(batch)
    results = frame(extractor, out)

    # window 0 (idx 2): [1, 2, 2]
    # len=3, values={1,2} (2 unique), reoccur_dp=2 (two 2s), reoccur_val=1 (2)
    # res: [2/3, 1/2, 2/3]
    assert np.allclose(results.iloc[0].values, [4.0 / 5.0, 2.0 / 3.0, 3.0 / 5.0])


def test_ecdf_pk_centroid_expanding():
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
    extractor = tsfast.ExpandingExtractor(features, n_cols=1)

    batch = np.stack([x])
    results = frame(extractor, extractor.update(batch)).iloc[-1].values

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


def test_expanding_invalid_type():
    # Verify that passing non-float arrays safely raises a TypeError instead of crashing
    x = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    features = ["mean", "std_dev"]
    extractor = ExpandingExtractor(features, 1)
    batch = np.stack([x])
    with pytest.raises(TypeError, match="expected a float32 or float64 numpy array"):
        extractor.update(batch)


def test_median_diff_expanding():
    data = np.random.randn(200).astype(np.float32)
    chunk1 = data[:100]
    chunk2 = data[100:]

    batch1 = np.stack([chunk1])
    batch2 = np.stack([chunk2])

    features = ["median_diff", "median_abs_diff"]
    import tsfast

    extractor = tsfast.ExpandingExtractor(features, 1)

    import tsfel

    res1 = frame(extractor, extractor.update(batch1)).iloc[0].values
    assert np.allclose(res1[0], tsfel.feature_extraction.features.median_diff(chunk1))
    assert np.allclose(
        res1[1], tsfel.feature_extraction.features.median_abs_diff(chunk1)
    )

    res2 = frame(extractor, extractor.update(batch2)).iloc[0].values
    assert np.allclose(res2[0], tsfel.feature_extraction.features.median_diff(data))
    assert np.allclose(res2[1], tsfel.feature_extraction.features.median_abs_diff(data))


def test_expanding_energy_ratio_by_chunks():
    import tsfast
    import numpy as np

    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], dtype=np.float32)
    features = [
        "energy_ratio_by_chunks_num_segments_3__segment_focus_0",
        "energy_ratio_by_chunks_num_segments_3__segment_focus_1",
        "energy_ratio_by_chunks_num_segments_3__segment_focus_2",
    ]
    extractor = tsfast.ExpandingExtractor(features, n_cols=1)
    batch = np.stack([x])
    df = frame(extractor, extractor.update(batch))
    res = df.iloc[-1].values
    assert np.isclose(res[0], (1**2 + 2**2 + 3**2) / sum([i**2 for i in range(1, 10)]))


def test_expanding_permutation_entropy_and_value_count():
    import numpy as np
    import math
    import tsfast

    data = np.array(
        [4.0, 7.0, 9.0, 10.0, 6.0, 11.0, 3.0, 3.0, np.nan, 3.0, np.nan],
        dtype=np.float32,
    )
    features = ["permutation_entropy-1-3", "value_count-3.0"]

    extractor = tsfast.ExpandingExtractor(features, n_cols=1)
    from tsfresh.feature_extraction.feature_calculators import (
        permutation_entropy,
        value_count,
    )

    for i in range(len(data)):
        batch = np.stack([[data[i]]])
        res = extractor.update(batch)

        window = data[: i + 1]
        pe = permutation_entropy(window, tau=1, dimension=3)
        vc = value_count(window, 3.0)

        if math.isnan(pe):
            assert math.isnan(res[0, 0])
        else:
            np.testing.assert_allclose(res[0, 0], pe, rtol=1e-5)
        assert res[0, 1] == vc


def test_expanding_fractal_dimensions():
    import numpy as np
    from tsfast._tsfast import ExpandingExtractor
    from tsfel.feature_extraction.features import higuchi_fractal_dimension
    import warnings

    warnings.filterwarnings("ignore")

    features = ["higuchi_fd"]
    n_cols = 1
    extractor = ExpandingExtractor(features, n_cols)

    x1 = np.random.randn(150).astype(np.float32)
    x2 = np.random.randn(100).astype(np.float32)

    b1 = np.stack([x1])
    b2 = np.stack([x2])

    res1 = frame(extractor, extractor.update(b1))
    res2 = frame(extractor, extractor.update(b2))

    # After b1, length is 150
    h1 = higuchi_fractal_dimension(x1)

    # After b2, length is 250
    full_x = np.concatenate([x1, x2])
    h2 = higuchi_fractal_dimension(full_x)

    if np.isnan(h1):
        pass
    else:
        assert np.allclose(res1.iloc[0, 0], h1, atol=5e-2)

    if np.isnan(h2):
        pass
    else:
        assert np.allclose(res2.iloc[0, 0], h2, atol=5e-2)
    print("test_expanding_fractal_dimensions passed!")


@pytest.mark.parametrize("case", ["randn", "exponential", "constant", "plateaus"])
def test_shape_features_expanding(case):
    from test_tsfast import SHAPE_CASES, SHAPE_FEATURES, shape_features_reference

    x = SHAPE_CASES[case].astype(np.float32)
    extractor = ExpandingExtractor(SHAPE_FEATURES, 1)
    seen = np.array([], dtype=np.float32)
    for chunk in np.array_split(x, 3):
        seen = np.concatenate([seen, chunk])
        batch = np.stack([chunk])
        row = frame(extractor, extractor.update(batch)).iloc[-1].values
        assert np.allclose(
            row, shape_features_reference(seen), atol=1e-5, equal_nan=True
        )


def test_expanding_output_shape_and_names():
    features = ["mean", "max_value"]
    extractor = ExpandingExtractor(features, 2)
    assert extractor.feature_names == features
    out = extractor.update(np.array([[1.0, 2.0], [10.0, 20.0]], dtype=np.float32))
    assert out.dtype == np.float32 and out.shape == (2, 2)
    np.testing.assert_allclose(out, [[1.5, 2.0], [15.0, 20.0]])
    # float64 is accepted and converted; features cover everything seen so far.
    np.testing.assert_allclose(extractor.update(np.array([[3.0], [30.0]])), [[2.0, 3.0], [20.0, 30.0]])

def test_expanding_mse():
    import numpy as np
    import tsfast
    from tsfel.feature_extraction.features import mse

    np.random.seed(42)
    signal = np.random.rand(500).astype(np.float32)

    e = tsfast.ExpandingExtractor(["mse-3"], 1)
    res1 = e.update(np.array([signal[:200]]))
    res2 = e.update(np.array([signal[200:]]))

    # Check intermediate
    window1 = signal[:200]
    tol1 = 0.2 * np.std(window1)
    ref1 = mse(window1, m=3, maxscale=None, tolerance=tol1)
    assert np.isclose(res1[0][0], ref1, atol=1e-2)

    # Check end
    tol2 = 0.2 * np.std(signal)
    ref2 = mse(signal, m=3, maxscale=None, tolerance=tol2)
    assert np.isclose(res2[0][0], ref2, atol=1e-2)

def test_ecdf_features():
    # Normal case with ties
    x = np.array([1, 2, 2, 2, 5, 5, 7, 8, 9, 10], dtype=np.float32)
    features = ["ecdf_percentile-0.5", "ecdf_percentile_count-0.5", "ecdf_slope-0.2-0.5"]

    import tsfel
    ext = tsfast.ExpandingExtractor(features, 1)
    for i in range(len(x)):
        arr = x[:i+1]
        out = ext.update(np.stack([arr[-1:]]).T)


        ref_perc = tsfel.feature_extraction.features.ecdf_percentile(arr, [0.5])
        if np.isscalar(ref_perc):
            ref_perc = float(ref_perc)
        else:
            ref_perc = float(ref_perc[0])

        if np.max(arr) == np.min(arr):
            ref_count = float(len(arr))
        else:
            ref_count = float(np.sum(arr <= ref_perc))

        try:
            ref_slope = tsfel.feature_extraction.features.ecdf_slope(arr, 0.2, 0.5)
        except Exception:
            ref_slope = np.nan

        if np.isnan(ref_perc):
            assert np.isnan(out[0][0])
        else:
            assert np.allclose(out[0][0], ref_perc)

        if np.isnan(ref_count):
            assert np.isnan(out[0][1])
        else:
            assert np.allclose(out[0][1], ref_count)

        if np.isnan(ref_slope):
            assert np.isnan(out[0][2])
        elif np.isinf(ref_slope):
            assert np.isinf(out[0][2])
        else:
            assert np.allclose(out[0][2], ref_slope)

def test_expanding_maximum_fractal_length():
    import numpy as np
    import tsfast
    from tsfel.feature_extraction.features import maximum_fractal_length

    np.random.seed(42)
    x1 = np.cumsum(np.random.randn(200)).astype(np.float32)
    x2 = np.cumsum(np.random.randn(50)).astype(np.float32)
    full_x = np.concatenate([x1, x2])

    features = ["maximum_fractal_length"]
    ext = tsfast.ExpandingExtractor(features, n_cols=1)

    res1 = ext.update(x1.reshape(1, -1))
    res2 = ext.update(x2.reshape(1, -1))

    mfl1 = maximum_fractal_length(x1)
    mfl2 = maximum_fractal_length(full_x)

    np.testing.assert_allclose(res1[0][0], mfl1, atol=1e-5)
    np.testing.assert_allclose(res2[0][0], mfl2, atol=1e-5)

def test_expanding_invalid_feature():
    # Verify unsupported features immediately error during initialization
    with pytest.raises(ValueError, match="Unknown feature"):
        ExpandingExtractor(["invalid_feature"], 1)


def test_expanding_count_above_below_and_range_count():
    import tsfresh.feature_extraction.feature_calculators as fc

    features = ["count_above-3.0", "count_below-3.0", "range_count-2-4"]

    def check(chunks):
        ext = ExpandingExtractor(features, 1)
        seen = np.array([], dtype=np.float32)
        for chunk in chunks:
            chunk = np.asarray(chunk, dtype=np.float32)
            res = ext.update(chunk.reshape(1, -1))
            seen = np.concatenate([seen, chunk])
            assert np.allclose(res[0][0], fc.count_above(seen, 3.0))
            assert np.allclose(res[0][1], fc.count_below(seen, 3.0))
            assert np.allclose(res[0][2], fc.range_count(seen, 2, 4))

    # Values exactly equal to the threshold/boundaries, appended in two chunks.
    check([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])
    # Constant series.
    check([[2.0, 2.0, 2.0], [2.0, 2.0]])
    # First window of length 1.
    check([[3.0], [1.0, 4.0]])


def test_expanding_power_bandwidth_positive_turning_variation():
    import tsfel.feature_extraction.features as F

    fs = 100.0
    features = ["power_bandwidth", "spectral_positive_turning", "spectral_variation"]

    def check(chunks):
        ext = ExpandingExtractor(features, 1)
        seen = np.array([], dtype=np.float32)
        for chunk in chunks:
            chunk = np.asarray(chunk, dtype=np.float32)
            res = ext.update(chunk.reshape(1, -1))
            seen = np.concatenate([seen, chunk])
            assert np.allclose(res[0][0], F.power_bandwidth(seen, fs), atol=1e-3)
            assert np.allclose(res[0][1], F.spectral_positive_turning(seen, fs))
            assert np.allclose(res[0][2], F.spectral_variation(seen, fs), atol=1e-5)

    rng = np.random.RandomState(13)
    t = np.arange(200)
    sine = np.sin(2 * np.pi * 5.37 * t / fs)
    chirp = np.sin(2 * np.pi * (1 + 0.05 * t) * t / fs)
    noise = rng.randn(200)
    constant = np.full(200, 4.0)

    check([sine[:100], sine[100:]])
    check([chirp[:100], chirp[100:]])
    check([noise[:100], noise[100:]])
    # Short first window.
    check([rng.randn(8), noise[:50]])

    # Constant series: AC spectral content is zero mathematically, but
    # TSFEL's own reference value for it is float-noise-dependent (see
    # test_tsfast.py's analogous case), so assert the mathematically correct
    # values directly instead of against TSFEL.
    ext = ExpandingExtractor(features, 1)
    for chunk in (constant[:100], constant[100:]):
        res = ext.update(chunk.astype(np.float32).reshape(1, -1))
        assert np.allclose(res[0][0], 0.0)
        assert np.allclose(res[0][1], 0.0)
        assert np.allclose(res[0][2], 1.0)


def test_expanding_wavelet_abs_mean_std_var():
    import tsfel.feature_extraction.features as F

    fs = 100.0
    features = ["wavelet_abs_mean-3", "wavelet_std-3", "wavelet_var-3"]

    def check(chunks):
        ext = ExpandingExtractor(features, 1)
        seen = np.array([], dtype=np.float32)
        for chunk in chunks:
            chunk = np.asarray(chunk, dtype=np.float32)
            res = ext.update(chunk.reshape(1, -1))
            seen = np.concatenate([seen, chunk])
            assert np.allclose(res[0][0], F.wavelet_abs_mean(seen, fs)["values"][3], atol=1e-3)
            assert np.allclose(res[0][1], F.wavelet_std(seen, fs)["values"][3], atol=1e-3)
            assert np.allclose(res[0][2], F.wavelet_var(seen, fs)["values"][3], atol=1e-3)

    rng = np.random.RandomState(17)
    noise = rng.randn(100)
    check([noise[:40], noise[40:]])
    # Short first window.
    check([rng.randn(5), noise[:30]])

    # Constant series: the mexh wavelet's coefficients on a flat signal are
    # nonzero near the boundary (finite-support truncation), so TSFEL's own
    # values are nonzero too; compare against those rather than zero.
    constant = np.full(80, -1.5, dtype=np.float32)
    ext = ExpandingExtractor(features, 1)
    seen = np.array([], dtype=np.float32)
    for chunk in (constant[:40], constant[40:]):
        res = ext.update(chunk.reshape(1, -1))
        seen = np.concatenate([seen, chunk])
        assert np.allclose(res[0][0], F.wavelet_abs_mean(seen, fs)["values"][3], atol=1e-3)
        assert np.allclose(res[0][1], F.wavelet_std(seen, fs)["values"][3], atol=1e-3)
        assert np.allclose(res[0][2], F.wavelet_var(seen, fs)["values"][3], atol=1e-3)


def test_expanding_fourier_entropy_and_fft_aggregated():
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

    def check(chunks):
        ext = ExpandingExtractor(features, 1)
        seen = np.array([], dtype=np.float32)
        for chunk in chunks:
            chunk = np.asarray(chunk, dtype=np.float32)
            res = ext.update(chunk.reshape(1, -1))
            seen = np.concatenate([seen, chunk])
            seen64 = seen.astype(np.float64)
            assert np.allclose(res[0][0], fc.fourier_entropy(seen64, 5), rtol=1e-2, equal_nan=True)
            for i, aggtype in enumerate(["centroid", "variance", "skew", "kurtosis"], start=1):
                want = one(fc.fft_aggregated(seen64, [{"aggtype": aggtype}]))
                assert np.allclose(res[0][i], want, atol=1e-3, equal_nan=True)

    rng = np.random.RandomState(19)
    noise = rng.randn(100)
    check([noise[:40], noise[40:]])
    # Short first window.
    check([rng.randn(5), noise[:30]])

    # Constant series: Welch PSD is all zero, so fourier_entropy's 0/0 divide
    # produces NaN, and fft_aggregated's variance collapses below its 0.5
    # cutoff, so skew/kurtosis are also NaN.
    check([np.full(40, -1.5), np.full(40, -1.5)])


def test_expanding_petrosian_fractal_dimension_and_average_power():
    from tsfel.feature_extraction.features import (
        average_power,
        petrosian_fractal_dimension,
    )

    features = ["petrosian_fractal_dimension", "average_power-100"]

    def check(chunks):
        ext = ExpandingExtractor(features, 1)
        seen = np.array([], dtype=np.float32)
        for chunk in chunks:
            chunk = np.asarray(chunk, dtype=np.float32)
            res = ext.update(chunk.reshape(1, -1))
            seen = np.concatenate([seen, chunk])
            seen64 = seen.astype(np.float64)
            assert np.allclose(
                res[0][0], petrosian_fractal_dimension(seen64), rtol=1e-2, equal_nan=True
            )
            assert np.allclose(
                res[0][1], average_power(seen64, 100.0), rtol=1e-2, equal_nan=True
            )

    rng = np.random.RandomState(23)
    noise = rng.randn(100)
    check([noise[:40], noise[40:]])
    # Short first window.
    check([rng.randn(5), noise[:30]])
    # Constant series.
    check([np.full(40, -1.5), np.full(40, -1.5)])


def test_expanding_hist_mode_and_neighbourhood_peaks():
    from tsfel.feature_extraction.features import hist_mode, neighbourhood_peaks

    features = ["hist_mode-10", "neighbourhood_peaks-3"]

    def check(chunks):
        ext = ExpandingExtractor(features, 1)
        seen = np.array([], dtype=np.float32)
        for chunk in chunks:
            chunk = np.asarray(chunk, dtype=np.float32)
            res = ext.update(chunk.reshape(1, -1))
            seen = np.concatenate([seen, chunk])
            seen64 = seen.astype(np.float64)
            assert np.isclose(res[0][0], hist_mode(seen64, 10), rtol=1e-5, atol=1e-6)
            assert res[0][1] == neighbourhood_peaks(seen64, 3)

    rng = np.random.RandomState(29)
    noise = rng.randn(100)
    check([noise[:40], noise[40:]])
    # Short first window (no peaks possible yet).
    check([rng.randn(5), noise[:30]])
    check([rng.randint(0, 4, size=50), rng.randint(0, 4, size=50)])
    # Constant series.
    check([np.full(40, -1.5), np.full(40, -1.5)])


def test_expanding_spectrogram_mean_coeff():
    from tsfel.feature_extraction.features import spectrogram_mean_coeff

    features = ["spectrogram_mean_coeff-1", "spectrogram_mean_coeff-10"]

    def check(chunks):
        ext = ExpandingExtractor(features, 1)
        seen = np.array([], dtype=np.float32)
        for chunk in chunks:
            chunk = np.asarray(chunk, dtype=np.float32)
            res = ext.update(chunk.reshape(1, -1))
            seen = np.concatenate([seen, chunk])
            ref = spectrogram_mean_coeff(seen.astype(np.float64), 100.0)["values"]
            # Coefficients past the clamped bin count report 0.
            expected = np.array([ref[k] if k < len(ref) else 0.0 for k in (1, 10)])
            atol = 1e-6 * max(np.abs(ref).max(), 1e-12)
            np.testing.assert_allclose(res[0], expected, rtol=1e-3, atol=atol)

    rng = np.random.RandomState(31)
    noise = rng.randn(300)
    # Grows through the clamped regime (n < 62) into multi-segment windows.
    check([noise[:10], noise[10:40], noise[40:100], noise[100:]])
    # Constant series.
    check([np.full(50, 1.5), np.full(50, 1.5)])


def _tsfresh_linear_trend_timewise(x, period_s):
    import pandas as pd
    from tsfresh.feature_extraction.feature_calculators import linear_trend_timewise

    ix = pd.date_range("2000-01-01", periods=len(x), freq=pd.Timedelta(seconds=period_s))
    res = linear_trend_timewise(pd.Series(x, index=ix), [{"attr": a} for a in LTT_ATTRS])
    return np.array([v for _, v in res])


LTT_ATTRS = ["slope", "intercept", "rvalue", "pvalue", "stderr"]


def test_expanding_linear_trend_timewise():
    def check(chunks, period_s, attrs=LTT_ATTRS):
        ext = ExpandingExtractor([f"linear_trend_timewise-{a}-{period_s}" for a in attrs], 1)
        idx = [LTT_ATTRS.index(a) for a in attrs]
        seen = np.array([], dtype=np.float32)
        for chunk in chunks:
            chunk = np.asarray(chunk, dtype=np.float32)
            res = ext.update(chunk.reshape(1, -1))
            seen = np.concatenate([seen, chunk])
            expected = _tsfresh_linear_trend_timewise(seen.astype(np.float64), period_s)[idx]
            np.testing.assert_allclose(res[0], expected, rtol=1e-3, atol=1e-4)

    rng = np.random.RandomState(9)
    noise = rng.randn(200).cumsum()
    check([noise[:5], noise[5:40], noise[40:]], 60)
    check([noise[:50], noise[50:]], 0.25)
    # Constant series.
    check([np.full(30, 4.0), np.full(30, 4.0)], 60, ["slope", "intercept"])
