"""Every feature must be within 1% of its tsfresh/TSFEL reference (references.py)."""

from pathlib import Path

import numpy as np
import pytest

import tsfast
from references import MIN_LENGTH, NO_REFERENCE, REFERENCES

FEATURES = [
    line.strip()
    for line in (Path(__file__).parent / "feature_samples.txt").read_text().splitlines()
    if line.strip() and not line.startswith("#")
]
RTOL = 0.01
ATOL = 1e-5
LENGTHS = [64, 150, 200, 600]  # 600: several Welch segments


def _series(n):
    # Same generator as test_engines: repeated levels, offset from zero.
    rng = np.random.default_rng(n)
    levels = rng.normal(0.5, 1.0, size=40)
    return rng.choice(levels, size=n).astype(np.float32)


def test_every_feature_has_a_reference_entry():
    listed = set(REFERENCES) | set(NO_REFERENCE)
    assert set(REFERENCES).isdisjoint(NO_REFERENCE)
    assert sorted(set(FEATURES) - listed) == [], "add to tests/references.py"
    assert sorted(listed - set(FEATURES)) == [], (
        "remove stale entries from tests/references.py"
    )


@pytest.mark.parametrize("n", LENGTHS)
@pytest.mark.parametrize("feature", sorted(REFERENCES))
def test_matches_reference(feature, n):
    if n < MIN_LENGTH.get(feature, 0):
        pytest.skip(f"reference is NaN below {MIN_LENGTH[feature]} samples")
    x = _series(n)
    batch = np.stack([x])
    got = tsfast.Extractor([feature]).process_2d_floats(batch)[0, 0]
    _, ref = REFERENCES[feature]
    want = float(ref(x.astype(np.float64)))
    np.testing.assert_allclose(got, want, rtol=RTOL, atol=ATOL, equal_nan=True)
