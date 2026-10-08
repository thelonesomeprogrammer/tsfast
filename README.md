# tsrocket

[![PyPI](https://img.shields.io/pypi/v/tsrocket)](https://pypi.org/project/tsrocket/)
[![Python](https://img.shields.io/pypi/pyversions/tsrocket)](https://pypi.org/project/tsrocket/)
[![CI](https://github.com/thelonesomeprogrammer/tsrocket/actions/workflows/ci.yml/badge.svg)](https://github.com/thelonesomeprogrammer/tsrocket/actions/workflows/ci.yml)
[![License: GPL v3](https://img.shields.io/badge/license-GPL--3.0-blue)](LICENSE)

**Every [tsfresh](https://github.com/blue-yonder/tsfresh) and
[TSFEL](https://github.com/fraunhoferportugal/tsfel) time-series feature,
computed in Rust: the same values, ~90× faster on one core, and incremental
for streaming data.**

Explore and select features with tsfresh or TSFEL as usual. To serve the
model, hand tsrocket the training frame's column names and get the same
features, in the same order, without tsfresh or TSFEL installed.

```python
import tsrocket

# X_train came from tsfresh.extract_features (or TSFEL); the model was fit on it.
X = tsrocket.extract_features(live_df, X_train.columns, column_id="id", column_sort="time")
model.predict(X)
```

## Install

```sh
pip install tsrocket
```

Prebuilt wheels for Linux, macOS and Windows (x86-64 and ARM), Python 3.10+.
The only runtime dependency is numpy; pandas is needed only for
`tsrocket.extract_features`.

## Speed

100 series × 1000 samples on an Intel i7-8750H; the same features and data
for every library ([`scripts/readme_benchmark.py`](scripts/readme_benchmark.py)):

| Workload | Reference | tsrocket, 1 thread | tsrocket, 12 threads |
| :--- | ---: | ---: | ---: |
| tsfresh `ComprehensiveFCParameters`, 783 features | tsfresh: 124 s | 1.39 s (**89×**) | 0.23 s (**548×**) |
| TSFEL, all domains, 163 features | TSFEL: 29 s | 0.34 s (**84×**) | 0.061 s (**484×**) |
| Streaming: all 783 features of the latest 256-sample window, per new sample | tsfresh: 220 ms | 0.62 ms (**356×**) | |

tsfresh ran with `n_jobs=0` and TSFEL in a single process, so the 1-thread
column is the like-for-like comparison. tsrocket uses every core by default;
set `RAYON_NUM_THREADS` to limit it.

## Same values

The test suite (~2,400 tests) runs tsfresh's `ComprehensiveFCParameters`
and every TSFEL domain, then checks that **every column they produce** parses
in tsrocket and matches it to within 1%. It also checks that the static,
sliding and expanding engines agree with each other. See
[Known differences](#known-differences) for the edge cases float32 can't
reproduce exactly.

## Usage

### Feature names

Use whichever spelling you have; [docs/features.md](docs/features.md) lists
every feature in all three:

| Spelling | Example |
| :--- | :--- |
| tsfresh column name | `value__fft_coefficient__attr_"abs"__coeff_3` (the `value__` kind prefix is optional) |
| TSFEL column name | `0_Spectral centroid`, `0_LPCC_3`, `0_Wavelet energy_12.5Hz` |
| tsrocket name | `fft_coeff-3-abs`, `spectral_centroid`, `lpcc-3` |

Parameters outside tsfresh's and TSFEL's defaults work too: `fft_coeff-7-angle`,
`quantile-0.33`, `agg_autocorrelation-median-20`. A misspelled name raises
`ValueError` and suggests the closest valid one.

### pandas

`tsrocket.extract_features` takes the same inputs as
`tsfresh.extract_features`: a wide frame with an id column, an optional sort
column and one column per signal, or a long frame with `column_kind` and
`column_value`. Each feature name's tsfresh kind (`temp__mean`) or TSFEL
channel (`temp_Mean`) picks the signal it is computed on.

```python
X = tsrocket.extract_features(df, ["temp__mean", "pressure__maximum"], column_id="id", column_sort="time")
```

The result has one row per id and the columns in the order you gave them.

### NumPy

```python
import numpy as np
import tsrocket

ext = tsrocket.Extractor(["mean", 'value__fft_coefficient__attr_"abs"__coeff_3', "0_Spectral centroid"], fs=100)
ext.feature_names                                  # ['mean', 'fft_coeff-3-abs', 'spectral_centroid']
X = ext.process_2d_floats(np.random.rand(8, 500))  # one series per row -> float32 array of shape (8, 3)
```

`fs` is the sampling frequency in Hz that the frequency-domain features assume
(TSFEL's default is 100).

### Streaming

tsfresh and TSFEL recompute every window from scratch. tsrocket's sliding and
expanding engines update their state as samples arrive, so each new window
costs a fraction of a full extraction.

```python
features = ["mean", "std_dev", "spectral_entropy", "0_MFCC_2"]

# Fixed-size windows over 3 sensors: one output row per completed window.
sliding = tsrocket.SlidingExtractor(features, n_cols=3, window_size=256, stride=16)
out = sliding.update(np.random.rand(3, 512))  # shape (3, completed windows, 4)

# Everything seen so far, updated in place.
expanding = tsrocket.ExpandingExtractor(features, n_cols=3)
out = expanding.update(np.random.rand(3, 100))  # shape (3, 4)
```

The sliding engine updates its spectrum with a sliding DFT and refreshes it
periodically; [docs/sliding-dft-drift.md](docs/sliding-dft-drift.md) explains
the accuracy trade-off and the `fresh-N` option.

### Feature selection

`tsrocket.selection.select_features(X, y)` is tsfresh's FRESH filter: it
drops constant and highly correlated features, then keeps the significant ones
under false-discovery-rate control. It needs scipy: `pip install tsrocket[selection]`.

## Known differences

tsrocket computes in float32. A few cases can't match a float64 reference
exactly:

- **Exact equality.** Two distinct float64 values can round to the same
  float32, which changes `has_duplicate`, the `*_reoccurring_*` features and
  `value_count`. With random float64 data this affected about 1 series in 60
  of length 150.
- **Bin edges.** A value that lies exactly on a bin edge in float64 can
  land in the neighbouring bin after rounding (`lempel_ziv_complexity`,
  `binned_entropy`).
- **Ties in `permutation_entropy`.** tsfresh ranks tied values with numpy's
  unstable sort, so its own result depends on the CPU. tsrocket breaks ties
  by position.
- **Ill-conditioned fits.** `max_langevin_fixed_point`, and an
  `agg_linear_trend` rvalue near zero, can differ by more than 1% when the
  underlying polynomial or regression is badly conditioned.

## Development

```sh
uv sync          # needs nightly Rust; rust-toolchain.toml pins it
uv run pytest    # rebuilds the extension when src/ changes
cargo test
```

[AGENTS.md](AGENTS.md) describes the layout and the checklist for adding a
feature. Benchmark history is in [benchmarks/README.md](benchmarks/README.md).

tsrocket was called tsfast until version 0.1; the name was taken on PyPI.

## License

[GPL-3.0-or-later](LICENSE).
