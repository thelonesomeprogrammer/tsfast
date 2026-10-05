"""Cross-engine consistency: every feature must give the same value whether it is
computed by the static, sliding or expanding engine.

The static `Extractor` is the reference. A sliding window must match the static
result on that window, and an expanding update must match the static result on
everything seen so far. Data is fed in uneven chunks so both the batch and the
one-value-at-a-time update paths run.

The feature list lives in tests/feature_samples.txt; `cargo test` fails if a
`Feature` variant is missing from it.
"""
import functools
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

import tsfast

FEATURES = [
    line.strip()
    for line in (Path(__file__).parent / "feature_samples.txt").read_text().splitlines()
    if line.strip() and not line.startswith("#")
]

WINDOW = 64
SLIDING_CHUNKS = [70, 1, 1, 5, 40, 83]  # first chunk fills the window, then mixed sizes
EXPANDING_CHUNKS = [32, 1, 30, 37, 100]
RTOL = 1e-3
ATOL = 1e-4

# Known engine disagreements. strict=True: once fixed, the XPASS fails the run
# so the entry gets removed.
_PAA = "engines round PAA segment boundaries differently when len % segments != 0"
_EXP_SPECTRAL = "expanding FFT processor never computes spectral skewness/kurtosis (always 0)"
_EXP_CWT = "expanding CWT (wavelet_energy/wavelet_entropy) differs from static"
_CONST_FFT = "constant series: FFT round-off leaves ~1e-7 bins and spectral ratios blow up"
KNOWN_FAILURES = {
    "sliding": {
        "paa-3-2": _PAA,
    },
    "expanding": {
        "paa-3-2": _PAA,
        "paa-4-1": _PAA,
        "spectral_kurtosis": _EXP_SPECTRAL,
        "spectral_skewness": _EXP_SPECTRAL,
        "wavelet_energy-0": _EXP_CWT,
        "wavelet_energy-3": _EXP_CWT,
        "wavelet_entropy": _EXP_CWT,
    },
    "constant": {
        "spectral_decrease": _CONST_FFT,
        "spectral_kurtosis": _CONST_FFT,
        "spectral_skewness": _CONST_FFT,
        "wavelet_energy-0": _EXP_CWT,
        "wavelet_energy-3": _EXP_CWT,
        "wavelet_entropy": _EXP_CWT,
    },
}


def _params(engine):
    known = KNOWN_FAILURES[engine]
    return [
        pytest.param(f, marks=pytest.mark.xfail(reason=known[f], strict=True)) if f in known else f
        for f in FEATURES
    ]


def _series():
    # Drawn from 40 levels so values repeat (reoccurrence/duplicate features), but
    # the levels are random so a window mean never ties exactly with a value.
    rng = np.random.default_rng(0)
    levels = rng.normal(0.5, 1.0, size=40)
    return rng.choice(levels, size=sum(SLIDING_CHUNKS)).astype(np.float32)


def _batch(*cols):
    return pa.RecordBatch.from_arrays(
        [pa.array(c, type=pa.float32()) for c in cols], names=[f"c{i}" for i in range(len(cols))]
    )


def _static(feature, segments):
    """Static result for each segment (all segments must have equal length)."""
    out = tsfast.Extractor([feature]).process_2d_floats(_batch(*segments))
    return out.column(0).to_numpy()


def _static_each(feature, segments):
    return np.array([_static(feature, [s])[0] for s in segments])


def _chunks(x, sizes):
    start = 0
    for size in sizes:
        yield x[start:start + size]
        start += size
    assert start == len(x)


@pytest.mark.parametrize("feature", _params("sliding"))
def test_sliding_matches_static(feature):
    x = _series()
    extractor = tsfast.SlidingExtractor([feature], 1, WINDOW)
    got = np.concatenate(
        [extractor.update(_batch(c)).column(0).to_numpy() for c in _chunks(x, SLIDING_CHUNKS)]
    )
    windows = [x[i:i + WINDOW] for i in range(len(x) - WINDOW + 1)]
    np.testing.assert_allclose(got, _static(feature, windows), rtol=RTOL, atol=ATOL, equal_nan=True)


@pytest.mark.parametrize("feature", _params("expanding"))
def test_expanding_matches_static(feature):
    x = _series()[: sum(EXPANDING_CHUNKS)]
    extractor = tsfast.ExpandingExtractor([feature], 1)
    got = np.concatenate(
        [extractor.update(_batch(c)).column(0).to_numpy() for c in _chunks(x, EXPANDING_CHUNKS)]
    )
    prefixes = [x[:end] for end in np.cumsum(EXPANDING_CHUNKS)]
    np.testing.assert_allclose(got, _static_each(feature, prefixes), rtol=RTOL, atol=ATOL, equal_nan=True)


@pytest.mark.parametrize("feature", _params("constant"))
def test_constant_series_matches_static(feature):
    x = np.full(WINDOW + 10, 2.5, dtype=np.float32)
    sliding = tsfast.SlidingExtractor([feature], 1, WINDOW).update(_batch(x)).column(0).to_numpy()
    expanding = tsfast.ExpandingExtractor([feature], 1).update(_batch(x)).column(0).to_numpy()
    want_window = _static(feature, [x[:WINDOW]])[0]
    np.testing.assert_allclose(sliding, want_window, rtol=RTOL, atol=ATOL, equal_nan=True)
    np.testing.assert_allclose(expanding, _static(feature, [x]), rtol=RTOL, atol=ATOL, equal_nan=True)


@functools.cache
def _all_at_once(engine):
    """Every feature in one extractor, so shared state/buffers interact."""
    x = _series()
    if engine == "static":
        return tsfast.Extractor(FEATURES).process_2d_floats(_batch(x))
    return tsfast.SlidingExtractor(FEATURES, 1, WINDOW).update(_batch(x))


@pytest.mark.parametrize("engine", ["static", "sliding"])
@pytest.mark.parametrize("feature", FEATURES)
def test_feature_independent_of_other_features(feature, engine):
    x = _series()
    if engine == "static":
        alone = tsfast.Extractor([feature]).process_2d_floats(_batch(x))
    else:
        alone = tsfast.SlidingExtractor([feature], 1, WINDOW).update(_batch(x))
    together = _all_at_once(engine).column(FEATURES.index(feature)).to_numpy()
    np.testing.assert_allclose(together, alone.column(0).to_numpy(), rtol=RTOL, atol=ATOL, equal_nan=True)
