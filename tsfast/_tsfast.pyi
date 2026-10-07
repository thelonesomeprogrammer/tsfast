"""Type stubs for the Rust extension (src/static_ext.rs, src/sliding.rs, src/expanding.rs).

Feature names are tsfresh/TSFEL names, e.g. ``"mean"``,
``"energy_ratio_by_chunks_num_segments_3__segment_focus_1"`` or
``"human_range_energy-100"``. ``tests/feature_samples.txt`` lists one valid
name per feature. An unknown or invalid name raises ``ValueError``.

``fs`` (default 100 Hz, TSFEL's default) is the sampling frequency the TSFEL
frequency-domain features (spectral centroid, roll-off, MFCC, spectrogram,
power bandwidth, ...) are evaluated at. ``human_range_energy``,
``average_power`` and ``calc_centroid`` follow it too, unless the name pins a
frequency: ``human_range_energy-100`` always uses 100 Hz.

A feature list may also hold meta features, which configure the engine and
produce no output column (``feature_names`` leaves them out). Every engine
accepts them and ignores the ones that do not apply:

- ``"fresh-N"`` (N >= 1): ``SlidingExtractor`` rebuilds its sliding DFT from a
  fresh FFT at least every N samples instead of every ``window_size``.
  ``"fresh-1"`` gives every window a fresh FFT, matching ``Extractor`` bit for
  bit, at up to ~1.7x the cost. See ``docs/sliding-dft-drift.md``.

Input arrays are 2-D float32/float64 with one series per row; float32
C-contiguous input is read without a copy. Output is always float32, with the
last axis in ``feature_names`` order. Every extractor grows to accept more
series (rows) than it was created with.
"""

import numpy as np
import numpy.typing as npt

_Values = npt.NDArray[np.float32] | npt.NDArray[np.float64]

class Extractor:
    """Features of whole series, one output row per input row."""

    def __init__(
        self, feature_str: list[str], max_size: int | None = None, fs: float = 100.0
    ) -> None:
        """``max_size``: expected series length, used to pre-plan the FFT.
        ``fs``: sampling frequency in Hz assumed by the frequency-domain features."""
    @property
    def feature_names(self) -> list[str]:
        """Canonical names of the output columns, in order."""
    def process_2d_floats(self, values: _Values) -> npt.NDArray[np.float32]:
        """(n_series, n_samples) -> (n_series, n_features)."""

class SlidingExtractor:
    """Features of fixed-size windows over streaming series."""

    def __init__(
        self,
        feature_str: list[str],
        n_cols: int,
        window_size: int,
        stride: int = 1,
        fs: float = 100.0,
    ) -> None:
        """``n_cols``: initial number of series; a window is emitted every ``stride`` samples once full.
        ``fs``: sampling frequency in Hz assumed by the frequency-domain features.
        Add ``"fresh-N"`` to ``feature_str`` to rebuild the spectrum from a fresh
        FFT every N samples (see the module docstring)."""
    @property
    def feature_names(self) -> list[str]:
        """Canonical names of the last output axis, in order."""
    def update(self, values: _Values) -> npt.NDArray[np.float32]:
        """Append (n_series, n_new_samples); returns (n_series, n_windows, n_features)
        for every window these samples completed, oldest first."""

class ExpandingExtractor:
    """Features of everything seen so far, updated incrementally."""

    def __init__(
        self,
        feature_str: list[str],
        n_cols: int,
        max_size: int | None = None,
        fft_update_period: int = 1,
        fs: float = 100.0,
    ) -> None:
        """``max_size``: expected final length, used to pre-plan the FFT.
        ``fft_update_period``: recompute the FFT only after this many new samples
        (1 = every update, exact).
        ``fs``: sampling frequency in Hz assumed by the frequency-domain features."""
    @property
    def feature_names(self) -> list[str]:
        """Canonical names of the output columns, in order."""
    def update(self, values: _Values) -> npt.NDArray[np.float32]:
        """Append (n_series, n_new_samples); returns (n_series, n_features)."""
