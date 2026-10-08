"""The ``fs`` constructor option: TSFEL features evaluated at a non-default
sampling frequency must match TSFEL called with that same ``fs``."""

import numpy as np
import pytest

import references
import tsrocket
from references import MIN_LENGTH, REFERENCES

RTOL = 0.01
ATOL = 1e-5
FS_VALUES = [1.0, 37.5, 250.0]
TSFEL_FEATURES = sorted(f for f, (lib, _) in REFERENCES.items() if lib == "tsfel")


def _series(n):
    # Same generator as test_references.
    rng = np.random.default_rng(n)
    levels = rng.normal(0.5, 1.0, size=40)
    return rng.choice(levels, size=n).astype(np.float32)


def _engines(feature, fs, x):
    n = len(x)
    batch = np.stack([x])
    yield "static", tsrocket.Extractor([feature], fs=fs).process_2d_floats(batch)[0, 0]
    sliding = tsrocket.SlidingExtractor([feature], 1, n, fs=fs)
    yield "sliding", sliding.update(batch)[0, -1, 0]
    expanding = tsrocket.ExpandingExtractor([feature], 1, fs=fs)
    yield "expanding", expanding.update(batch)[0, 0]


@pytest.mark.parametrize("fs", FS_VALUES)
@pytest.mark.parametrize("feature", TSFEL_FEATURES)
def test_tsfel_features_follow_fs(feature, fs, monkeypatch):
    n = 200
    if n < MIN_LENGTH.get(feature, 0):
        pytest.skip(f"reference is NaN below {MIN_LENGTH[feature]} samples")
    monkeypatch.setattr(references, "FS", fs)
    x = _series(n)
    _, ref = REFERENCES[feature]
    want = float(ref(x.astype(np.float64)))
    for engine, got in _engines(feature, fs, x):
        np.testing.assert_allclose(
            got, want, rtol=RTOL, atol=ATOL, equal_nan=True, err_msg=engine
        )


def test_fs_default_is_100():
    x = np.stack([_series(300)])
    names = ["spectral_centroid", "max_frequency", "mfcc-3", "power_bandwidth"]
    default = tsrocket.Extractor(names).process_2d_floats(x)
    explicit = tsrocket.Extractor(names, fs=100.0).process_2d_floats(x)
    np.testing.assert_array_equal(default, explicit)


def test_fs_scales_frequency_features():
    # A 5 Hz sine sampled at 50 Hz: the dominant frequency is reported in Hz.
    fs = 50.0
    t = np.arange(500) / fs
    x = np.stack([np.sin(2 * np.pi * 5.0 * t)]).astype(np.float32)
    got = tsrocket.Extractor(["spectral_centroid"], fs=fs).process_2d_floats(x)[0, 0]
    assert got == pytest.approx(5.0, rel=0.02)


@pytest.mark.parametrize("fs", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_fs_is_rejected(fs):
    with pytest.raises(ValueError, match="fs"):
        tsrocket.Extractor(["mean"], fs=fs)
    with pytest.raises(ValueError, match="fs"):
        tsrocket.SlidingExtractor(["mean"], 1, 10, fs=fs)
    with pytest.raises(ValueError, match="fs"):
        tsrocket.ExpandingExtractor(["mean"], 1, fs=fs)


@pytest.mark.parametrize("base", ["human_range_energy", "average_power", "calc_centroid"])
def test_fs_in_name_overrides_extractor_fs(base):
    x = np.stack([_series(300)])
    bare = tsrocket.Extractor([base], fs=50.0)
    pinned = tsrocket.Extractor([f"{base}-50"], fs=250.0)
    assert bare.feature_names == [base]
    assert pinned.feature_names == [f"{base}-50"]
    np.testing.assert_array_equal(bare.process_2d_floats(x), pinned.process_2d_floats(x))
