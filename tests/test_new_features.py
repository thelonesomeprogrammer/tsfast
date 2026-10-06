import tsfast
import numpy as np
import pytest
from tsfresh.feature_extraction import feature_calculators as fc


def test_linear_trend():
    x = np.array([1.0, 3.0, 2.0, 5.0, 4.0, 6.0], dtype=np.float32)
    features = [
        'linear_trend__attr_"rvalue"',
        'linear_trend__attr_"intercept"',
        'linear_trend__attr_"slope"',
        'linear_trend__attr_"stderr"',
    ]

    extractor = tsfast.Extractor(features)
    batch = np.stack([x])
    out = extractor.process_2d_floats(batch)
    results = out[0]

    tsfresh_param = [
        {"attr": "rvalue"},
        {"attr": "intercept"},
        {"attr": "slope"},
        {"attr": "stderr"},
    ]
    tsfresh_results = fc.linear_trend(x, tsfresh_param)

    for i, (_, val) in enumerate(tsfresh_results):
        assert np.allclose(results[i], val, atol=1e-5), (
            f"Mismatch in {features[i]}: {results[i]} != {val}"
        )


def test_agg_linear_trend():
    x = np.array([1.0, 3.0, 2.0, 5.0, 4.0, 6.0, 2.0, 1.0, 3.0], dtype=np.float32)
    features = [
        'agg_linear_trend__attr_"intercept"__chunk_len_3__f_agg_"mean"',
        'agg_linear_trend__attr_"slope"__chunk_len_3__f_agg_"max"',
    ]

    extractor = tsfast.Extractor(features)
    batch = np.stack([x])
    out = extractor.process_2d_floats(batch)
    results = out[0]

    tsfresh_param = [
        {"attr": "intercept", "chunk_len": 3, "f_agg": "mean"},
        {"attr": "slope", "chunk_len": 3, "f_agg": "max"},
    ]
    tsfresh_results = list(fc.agg_linear_trend(x, tsfresh_param))

    for i, (_, val) in enumerate(tsfresh_results):
        assert np.allclose(results[i], val, atol=1e-5), (
            f"Mismatch in {features[i]}: {results[i]} != {val}"
        )


def test_time_reversal_asymmetry_statistic():
    x = np.array([1.0, 3.0, 2.0, 5.0, 4.0, 6.0], dtype=np.float32)
    features = [
        "time_reversal_asymmetry_statistic__lag_1",
        "time_reversal_asymmetry_statistic__lag_2",
    ]

    extractor = tsfast.Extractor(features)
    batch = np.stack([x])
    out = extractor.process_2d_floats(batch)
    results = out[0]

    tsfresh_val_1 = fc.time_reversal_asymmetry_statistic(x, 1)
    tsfresh_val_2 = fc.time_reversal_asymmetry_statistic(x, 2)

    assert np.allclose(results[0], tsfresh_val_1, atol=1e-5)
    assert np.allclose(results[1], tsfresh_val_2, atol=1e-5)


def test_incremental_linear_trend():
    x = np.array([1.0, 3.0, 2.0, 5.0, 4.0, 6.0], dtype=np.float32)
    features = [
        'linear_trend__attr_"rvalue"',
        'linear_trend__attr_"slope"',
    ]

    # Static extractor
    extractor_static = tsfast.Extractor(features)
    batch = np.stack([x])
    results_static = extractor_static.process_2d_floats(batch)[0]

    # Sliding extractor (window size 6)
    extractor_sliding = tsfast.SlidingExtractor(features, 1, 6, 1)
    results_sliding = None
    for i in range(len(x)):
        batch_i = np.stack([[x[i]]])
        res = extractor_sliding.update(batch_i)
        if i == len(x) - 1:
            results_sliding = res[0, 0]

    # Expanding extractor
    extractor_exp = tsfast.ExpandingExtractor(features, 1)
    results_exp = None
    for i in range(len(x)):
        batch_i = np.stack([[x[i]]])
        res = extractor_exp.update(batch_i)
        if i == len(x) - 1:
            results_exp = res[0]

    assert np.allclose(results_static, results_sliding, atol=1e-5)
    assert np.allclose(results_static, results_exp, atol=1e-5)


def test_incremental_time_reversal_asymmetry():
    x = np.array([1.0, 3.0, 2.0, 5.0, 4.0, 6.0], dtype=np.float32)
    features = [
        "time_reversal_asymmetry_statistic__lag_1",
    ]

    extractor_static = tsfast.Extractor(features)
    batch = np.stack([x])
    results_static = extractor_static.process_2d_floats(batch)[0]

    extractor_sliding = tsfast.SlidingExtractor(features, 1, 6, 1)
    results_sliding = None
    for i in range(len(x)):
        batch_i = np.stack([[x[i]]])
        res = extractor_sliding.update(batch_i)
        if i == len(x) - 1:
            results_sliding = res[0, 0]

    extractor_exp = tsfast.ExpandingExtractor(features, 1)
    results_exp = None
    for i in range(len(x)):
        batch_i = np.stack([[x[i]]])
        res = extractor_exp.update(batch_i)
        if i == len(x) - 1:
            results_exp = res[0]

    assert np.allclose(results_static, results_sliding, atol=1e-5)
    assert np.allclose(results_static, results_exp, atol=1e-5)
