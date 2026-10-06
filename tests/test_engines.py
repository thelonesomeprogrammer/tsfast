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
KNOWN_FAILURES = {
    "sliding": {},
    "expanding": {},
    "constant": {},
}


def _params(engine):
    known = KNOWN_FAILURES[engine]
    return [
        pytest.param(f, marks=pytest.mark.xfail(reason=known[f], strict=True))
        if f in known
        else f
        for f in FEATURES
    ]


def _series():
    # Drawn from 40 levels so values repeat (reoccurrence/duplicate features), but
    # the levels are random so a window mean never ties exactly with a value.
    rng = np.random.default_rng(0)
    levels = rng.normal(0.5, 1.0, size=40)
    return rng.choice(levels, size=sum(SLIDING_CHUNKS)).astype(np.float32)


def _batch(*cols):
    return np.stack(cols).astype(np.float32)


def _column(out, feature_idx=0):
    """One feature's values over every output row (sliding: every window)."""
    return out.reshape(-1, out.shape[-1])[:, feature_idx]


def _static(feature, segments):
    """Static result for each segment (all segments must have equal length)."""
    out = tsfast.Extractor([feature]).process_2d_floats(_batch(*segments))
    return _column(out)


def _static_each(feature, segments):
    return np.array([_static(feature, [s])[0] for s in segments])


def _chunks(x, sizes):
    start = 0
    for size in sizes:
        yield x[start : start + size]
        start += size
    assert start == len(x)


# 37 is prime: exercises the exact-length sliding DFT and its periodic re-sync.
@pytest.mark.parametrize("window", [WINDOW, 37])
@pytest.mark.parametrize("feature", _params("sliding"))
def test_sliding_matches_static(feature, window):
    x = _series()
    extractor = tsfast.SlidingExtractor([feature], 1, window)
    got = np.concatenate(
        [_column(extractor.update(_batch(c))) for c in _chunks(x, SLIDING_CHUNKS)]
    )
    windows = [x[i : i + window] for i in range(len(x) - window + 1)]
    np.testing.assert_allclose(
        got, _static(feature, windows), rtol=RTOL, atol=ATOL, equal_nan=True
    )


@pytest.mark.parametrize("feature", _params("expanding"))
def test_expanding_matches_static(feature):
    x = _series()[: sum(EXPANDING_CHUNKS)]
    extractor = tsfast.ExpandingExtractor([feature], 1)
    got = np.concatenate(
        [_column(extractor.update(_batch(c))) for c in _chunks(x, EXPANDING_CHUNKS)]
    )
    prefixes = [x[:end] for end in np.cumsum(EXPANDING_CHUNKS)]
    np.testing.assert_allclose(
        got, _static_each(feature, prefixes), rtol=RTOL, atol=ATOL, equal_nan=True
    )


@pytest.mark.parametrize("feature", _params("constant"))
def test_constant_series_matches_static(feature):
    x = np.full(WINDOW + 10, 2.5, dtype=np.float32)
    sliding = _column(tsfast.SlidingExtractor([feature], 1, WINDOW).update(_batch(x)))
    expanding = _column(tsfast.ExpandingExtractor([feature], 1).update(_batch(x)))
    want_window = _static(feature, [x[:WINDOW]])[0]
    np.testing.assert_allclose(
        sliding, want_window, rtol=RTOL, atol=ATOL, equal_nan=True
    )
    np.testing.assert_allclose(
        expanding, _static(feature, [x]), rtol=RTOL, atol=ATOL, equal_nan=True
    )


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
    together = _column(_all_at_once(engine), FEATURES.index(feature))
    np.testing.assert_allclose(
        together, _column(alone), rtol=RTOL, atol=ATOL, equal_nan=True
    )
