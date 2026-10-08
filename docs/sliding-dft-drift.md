# Sliding-window spectrum drift

`SlidingExtractor` doesn't run a new FFT for every window. It updates the
previous window's spectrum in O(window) per sample (a *sliding DFT*) and only
rebuilds it from a fresh FFT every `window_size` samples. In f32 each update
rounds a little, so between rebuilds its frequency bins are close to the
static engine's (`Extractor` on the same window), but not bit-identical.

For most features that difference is far below anything that matters. It
matters for features that compare bins **exactly**, on data where bins are
exactly zero or exactly tied. This page measures when that happens and what
the `fresh-N` meta feature costs to avoid it.

**Short version:**

- **Affected:** `spectral_entropy`, `spectral_positive_turning`,
  `fundamental_frequency` and `fft_coeff-*-angle`, in `SlidingExtractor` only.
- **Only on quantized data:** integer-valued signals (ADC counts) and
  especially binary or event signals, with windows of about 32 samples or less.
  On continuous-valued data there were no mismatches in any measurement here.
- **When it hits, the error is large, not slight:** 8–33% in `spectral_entropy`,
  a whole extra turning point, a jump of one frequency bin, or a 360° angle flip.
- **The fix:** add `"fresh-1"` to the feature list. Every window then gets a
  fresh FFT and matches `Extractor` bit for bit. It costs 1.1× at W=16 and up
  to 1.7× at W=1024, for the spectrum part of the work.

## Reading guide

- **Reference:** every number compares `SlidingExtractor` with `Extractor` on
  the same window. Static is the reference because it always runs a fresh FFT.
  This page is about the engines agreeing with each other. How the static engine
  compares with tsfresh/TSFEL is a separate question, tested by
  `tests/test_references.py`.
- **Drift:** |sliding − static| for one window. "Largest" is the maximum over
  all windows of a series. For a frequency bin it's the distance between the two
  complex values.
- **Mismatch:** a window where the two engines disagree by more than
  `rtol=1e-3, atol=1e-4`, the tolerance of `tests/test_engines.py`. A table
  showing "40 → 0" means 40 such windows before and none after.
- **before → after:** tables 1 and 2 compare the sliding DFT before and after
  its twiddle factors were made exact (see [Twiddle fix](#twiddle-fix)). "After"
  is the current default, without `fresh-N`.
- **Largest relative:** the drift divided by the static value, over mismatched
  windows only. Where the static value is 0, a ratio doesn't mean anything, so
  the table says "static 0" instead.
- **W:** `window_size`. **k:** bin index, so k = n/4 is a quarter of the
  sampling rate and n/2 is the Nyquist bin. All runs use stride 1.
- **Data**, all cast to float32:

  | Name | What | Why |
  |---|---|---|
  | ints | integers drawn uniformly from [−5, 5) | quantized, e.g. ADC counts |
  | binary | 0 or 1 | event and on/off signals: the worst case |
  | levels | 40 random normal levels, repeated | the generator `tests/test_engines.py` uses |
  | randn | standard normal | continuous-valued data |

  Tables 1 and 2 use 1200 samples per series (945–1185 windows). Table 3 uses
  3000 (1977–2985 windows).

## Which features are affected, and why

The sliding DFT error is about 1e-6 to 1e-4 per bin (table 1). A feature that
uses bin values smoothly passes that through unchanged: `spectral_centroid`
never drifts more than 2.7e-5. A feature that *compares* bins turns it into a
discrete jump:

| Feature | Exact comparison | What a tiny drift does |
|---|---|---|
| `spectral_entropy` | counts bins with power `> 0` (TSFEL's normaliser) | an exactly-zero bin becomes 1e-12, is counted, and the normaliser log2(N) changes |
| `spectral_positive_turning` | `prev < mag >= next` between neighbours | breaks ties between equal bins into a new turning point, or removes one |
| `fundamental_frequency` | same tie-sensitive peak search | picks a different bin, a jump of fs/W Hz |
| `fft_coeff-*-angle` | `atan2(imag, real)` | the angle of an exactly-zero bin becomes noise; an exactly-real bin flips between +180° and −180° |

Exactly-zero and exactly-tied bins only appear when the data is quantized.
With continuous values they essentially never occur, which is why all the
mismatches below are on `ints` and `binary`.

## Table 1: raw bin drift

Largest |sliding − static| per bin, before → after the twiddle fix. "–" means
that bin doesn't exist for this W.

| W | Data | DC | k=1 | k=3 | n/4 | n/3 | Nyquist |
|---|---|---|---|---|---|---|---|
| 16 | ints | 0 → 0 | 1.0e-5 → 1.1e-5 | 1.1e-5 → 1.1e-5 | 1.1e-5 → **0** | – | 3.9e-5 → **0** |
| 16 | binary | 0 → 0 | 2.1e-6 → 2.0e-6 | 1.9e-6 → 1.9e-6 | 2.2e-6 → **0** | – | 5.4e-6 → **0** |
| 16 | levels | 3.8e-6 → 3.8e-6 | 3.6e-6 → 4.4e-6 | 3.6e-6 → 3.6e-6 | 3.5e-6 → 1.1e-6 | – | 9.0e-6 → 1.8e-6 |
| 16 | randn | 1.9e-6 → 1.9e-6 | 5.1e-6 → 3.6e-6 | 4.5e-6 → 4.5e-6 | 4.4e-6 → 1.1e-6 | – | 1.1e-5 → 1.9e-6 |
| 24 | ints | 0 → 0 | 9.6e-6 → 9.6e-6 | 1.8e-5 → 1.8e-5 | 2.0e-5 → 2.5e-6 | 3.3e-5 → 1.5e-5 | 8.2e-5 → **0** |
| 24 | binary | 0 → 0 | 2.1e-6 → 2.1e-6 | 1.5e-6 → 1.5e-6 | 4.6e-6 → 3.0e-7 | 5.5e-6 → 1.7e-6 | 8.0e-6 → **0** |
| 24 | levels | 5.7e-6 → 5.7e-6 | 3.6e-6 → 3.6e-6 | 3.1e-6 → 3.1e-6 | 9.1e-6 → 1.3e-6 | 1.2e-5 → 5.4e-6 | 1.3e-5 → 2.9e-6 |
| 24 | randn | 2.9e-6 → 2.9e-6 | 5.9e-6 → 5.9e-6 | 4.6e-6 → 4.6e-6 | 9.0e-6 → 2.1e-6 | 1.3e-5 → 4.0e-6 | 2.0e-5 → 2.9e-6 |
| 64 | ints | 0 → 0 | 1.8e-5 → 1.8e-5 | 5.6e-5 → 5.0e-5 | 1.0e-4 → **0** | – | 2.4e-4 → **0** |
| 64 | binary | 0 → 0 | 4.4e-6 → 4.4e-6 | 8.9e-6 → 8.6e-6 | 1.4e-5 → **0** | – | 3.0e-5 → **0** |
| 64 | levels | 2.7e-5 → 2.7e-5 | 6.3e-6 → 6.3e-6 | 2.0e-5 → 1.8e-5 | 3.6e-5 → 6.4e-6 | – | 7.1e-5 → 8.1e-6 |
| 64 | randn | 3.8e-6 → 3.8e-6 | 7.8e-6 → 7.8e-6 | 2.1e-5 → 1.6e-5 | 4.1e-5 → 5.1e-6 | – | 6.2e-5 → 4.8e-6 |
| 256 | ints | 0 → 0 | 5.0e-5 → 5.0e-5 | 1.9e-4 → 1.9e-4 | 3.1e-4 → **0** | – | 1.5e-3 → **0** |
| 256 | binary | 0 → 0 | 8.9e-6 → 8.9e-6 | 3.9e-5 → 3.9e-5 | 4.9e-5 → **0** | – | 1.6e-4 → **0** |
| 256 | levels | 9.2e-5 → 9.2e-5 | 1.3e-5 → 1.3e-5 | 9.2e-5 → 9.2e-5 | 1.4e-4 → 1.1e-5 | – | 3.3e-4 → 1.6e-5 |
| 256 | randn | 1.4e-5 → 1.4e-5 | 2.3e-5 → 2.3e-5 | 1.8e-4 → 1.8e-4 | 2.5e-4 → 1.1e-5 | – | 1.8e-4 → 5.7e-6 |

How to read it:

- **Bins at a quarter turn** (DC, n/4, Nyquist) now have exact twiddles: 1, i
  and −1. On quantized data every update is then exact, so the drift is **0**.
  On continuous data their drift drops 5–30×.
- **Every other bin** (k=1, k=3, n/3) is unchanged. Its twiddle is irrational,
  so each update rounds. The drift grows with W, up to about 2e-4 at W=256,
  because it builds up over the W updates between rebuilds.
- **DC never drifted on quantized data**: its twiddle was always exactly 1.

## Table 2: feature drift

Mismatched windows (out of 945–1185), with the largest drift, before → after
the twiddle fix. Combinations not listed had no mismatches before or after.

| Feature | W | Data | Mismatched windows | Largest drift | Largest relative |
|---|---|---|---|---|---|
| `spectral_entropy` | 16 | ints | 40 → **0** | 6.2e-2 → 3.6e-7 | 6.4% → 0% |
| | 16 | binary | 250 → 19 | 0.30 → 0.30 | 33% → 33% |
| | 24 | ints | 41 → 2 | 3.1e-2 → 2.6e-2 | 3.5% → 3.2% |
| | 24 | binary | 211 → 96 | 0.17 → 7.4e-2 | 25% → 8.4% |
| | 64 | ints | 24 → **0** | 8.5e-3 → 4.8e-7 | 0.9% → 0% |
| | 64 | binary | 67 → **0** | 1.7e-2 → 4.8e-7 | 1.9% → 0% |
| | 256 | ints | 3 → **0** | 1.5e-3 → 8.9e-7 | 0.2% → 0% |
| | 256 | binary | 39 → **0** | 1.5e-3 → 9.5e-7 | 0.2% → 0% |
| `spectral_positive_turning` | 16 | binary | 8 → 10 | 3 → 3 turning points | static 0 |
| | 24 | binary | 34 → 32 | 2 → 3 turning points | 200% → 200% |
| `fundamental_frequency` | 16 | binary | 1 → 1 | 6.2 Hz → 6.2 Hz | 100% (static 0 Hz) |
| | 24 | binary | 7 → 4 | 8.3 Hz → 8.3 Hz | 100% |
| `fft_coeff-2-angle` | 16 | ints | 1 → 1 | 360° → 360° | ±180° flip |
| | 16 | binary | 23 → 23 | 360° → 360° | static 0 |
| | 24 | binary | 20 → 20 | 360° → 360° | static 0 |
| `spectral_distance` | 64 | randn | 2 → **0** | 2.3e-3 → 5.7e-4 | 0.2% → 0% |
| | 256 | ints | 4 → **0** | 0.23 → 0.12 | 0.4% → 0% |
| | 256 | randn | 3 → 2 | 0.15 → 4.1e-2 | 0.1% → 0.2% |
| `spectral_centroid` | all | all | 0 → 0 | ≤ 2.7e-5 | – |

How to read it:

- **Mismatches come in whole steps.** The four exact-comparison features don't
  drift a little: they're either right or off by a whole bin count, turning
  point, frequency bin or half turn. That's why a 1e-6 bin drift becomes a 33%
  error.
- **`spectral_entropy`** gained the most from the twiddle fix. From W=64 up it's
  now exact on quantized data, and at W=16 on integers too. What's left is
  binary data at W ≤ 24, where an exactly-zero bin sits on an irrational
  twiddle.
- **The other three exact-comparison features** depend on ties and zeros in
  those irrational-twiddle bins, so the fix barely changed them. Only `fresh-1`
  fixes them (table 3).
- **`spectral_distance`** isn't an exact comparison. It's a cumulative sum with
  large values, so a 1e-5 bin drift becomes about 1e-1 in absolute terms. That's
  still only 0.1–0.2% relative, right at the tolerance.

## Table 3: the `fresh-N` trade-off

`"fresh-N"` in the feature list makes the sliding engine rebuild from a fresh
FFT at least every N samples (default: every W). The table covers each
setting, measured in the current code:

- **Cost:** ns per window, for 8 series with only `spectral_entropy`. That makes
  the spectrum the bulk of the work, so this is the worst-case relative cost;
  with a full feature list the slowdown is smaller.
- **Mismatches:** binary data, for `spectral_entropy` /
  `spectral_positive_turning` / `fundamental_frequency` / `fft_coeff-2-angle`.
- **Drift:** randn data. "Bin" is bin k=3; "centroid" is `spectral_centroid`.

| W | Setting | ns/window | vs default | Mismatches (binary) | Bin drift | Centroid drift |
|---|---|---|---|---|---|---|
| 16 | default (= fresh-16) | 237 | 1.00× | 42 / 22 / 1 / 58 of 2985 | 4.3e-6 | 9.5e-6 |
| 16 | fresh-8 | 234 | 0.99× | 37 / 19 / 1 / 38 | 3.1e-6 | 9.5e-6 |
| 16 | fresh-4 | 244 | 1.03× | 17 / 20 / 1 / 26 | 2.0e-6 | 9.5e-6 |
| 16 | fresh-2 | 248 | 1.05× | 0 / 17 / 2 / 6 | 1.9e-6 | 9.5e-6 |
| 16 | **fresh-1** | 266 | **1.12×** | **0 / 0 / 0 / 0** | **0** | **0** |
| 64 | default | 388 | 1.00× | 0 / 0 / 0 / 0 of 2937 | 1.6e-5 | 1.3e-5 |
| 64 | fresh-16 | 397 | 1.02× | 0 / 0 / 0 / 0 | 7.3e-6 | 1.5e-5 |
| 64 | fresh-8 | 410 | 1.06× | 0 / 0 / 0 / 0 | 4.8e-6 | 9.5e-6 |
| 64 | fresh-2 | 462 | 1.19× | 0 / 0 / 0 / 0 | 3.8e-6 | 9.5e-6 |
| 64 | **fresh-1** | 533 | **1.37×** | 0 / 0 / 0 / 0 | **0** | **0** |
| 256 | default | 989 | 1.00× | 0 / 0 / 0 / 0 of 2745 | 1.4e-4 | 2.1e-5 |
| 256 | fresh-64 | 951 | 0.96× | 0 / 0 / 0 / 0 | 4.1e-5 | 1.9e-5 |
| 256 | fresh-8 | 1010 | 1.02× | 0 / 0 / 0 / 0 | 1.3e-5 | 1.1e-5 |
| 256 | fresh-2 | 1167 | 1.18× | 0 / 0 / 0 / 0 | 8.0e-6 | 9.5e-6 |
| 256 | **fresh-1** | 1409 | **1.42×** | 0 / 0 / 0 / 0 | **0** | **0** |
| 1024 | default | 3097 | 1.00× | 0 / 1 / 0 / 0 of 1977 | 3.7e-4 | 5.0e-5 |
| 1024 | fresh-256 | 3116 | 1.01× | 0 / 0 / 0 / 0 | 2.0e-4 | 3.6e-5 |
| 1024 | fresh-8 | 3390 | 1.09× | 0 / 0 / 0 / 0 | 2.0e-5 | 1.1e-5 |
| 1024 | fresh-2 | 4194 | 1.35× | 0 / 0 / 0 / 0 | 1.7e-5 | 9.5e-6 |
| 1024 | **fresh-1** | 5300 | **1.71×** | 0 / 0 / 0 / 0 | **0** | **0** |

How to read it:

- **Only `fresh-1` fixes the exact comparisons.** A window goes wrong after its
  first incremental update, so rebuilding every N > 1 samples still leaves most
  windows in between affected. At W=16, `fresh-8` still has 37 mismatched
  `spectral_entropy` windows against 42 by default.
- **Intermediate N reduces the drift size.** It's useful when you want smaller
  drift on continuous data without paying for a fresh FFT every window: at
  W=1024, `fresh-8` cuts the bin drift 19× for 1.09×.
- **The cost grows with W** because a fresh FFT is O(W log W) against the
  sliding DFT's O(W) per sample. Differences of a few percent (e.g. 0.96× and
  0.99×) are timing noise. All numbers come from one machine.
- **`fresh-1` gives exactly the static engine's result**, also with stride > 1
  and with input fed in uneven chunks (`test_sliding_fresh_fft_meta_feature`).

## What to use

| Your situation | Use |
|---|---|
| Continuous-valued data (float sensors, prices, …) | the default |
| None of the four features above | the default: the other features see drift of ≤ ~1e-4 per bin |
| Quantized/integer data and W ≥ 64 | the default (no mismatches measured) |
| Quantized or binary data, W ≤ ~32, any of the four features | `"fresh-1"` (costs ~1.1× there) |
| You need sliding output identical to `Extractor` (e.g. training features computed statically, served with sliding windows) | `"fresh-1"` |
| Long windows, smaller drift, but not bit-exact | `"fresh-8"` or so |

`fresh-N` is a meta feature: it produces no output column and doesn't appear
in `feature_names`. `Extractor` and `ExpandingExtractor` accept it and ignore
it, because they always run a fresh FFT. In `ExpandingExtractor`,
`fft_update_period` controls how often that happens. N must be ≥ 1, values of N
≥ W act as the default, and if `fresh-N` appears more than once the smallest N
wins.

```python
from tsrocket import SlidingExtractor

ext = SlidingExtractor(["spectral_entropy", "fft_coeff-2-angle", "fresh-1"], n_cols=1, window_size=16)
ext.feature_names  # ['spectral_entropy', 'fft_coeff-2-angle']
```

## Twiddle fix

Before this change, the sliding DFT computed its twiddle factors
e^(2πik/W) in f32. For exact angles that's already wrong: `sin(π as f32)`
is −8.7e-8, not 0, and `cos(π/2 as f32)` is −4.4e-8. So the n/4 and Nyquist
bins drifted from the first update, even on exact integer data. Now:

- **Quarter-turn bins** (DC, n/4, Nyquist) use exactly 1, i and −1.
- **All other bins** are computed in f64 and then rounded to f32.

The twiddles are built once per extractor, so this has no runtime cost. It
explains every "→ **0**" in tables 1 and 2.

## Not covered here

`fresh-1` makes sliding equal static, not static equal to TSFEL. The static
engine's `spectral_entropy` has its own known differences from TSFEL on short
series. TSFEL counts a bin of float residue (~1e-30) as non-zero, and whether
numpy leaves such a residue depends on its rounding. tsrocket predicts the DC
bin's residue from whether the f64 mean is exact (`f64_mean_is_exact` in
`src/spectral.rs`). The cases it can't predict are the strict `xfail` tests in
`tests/test_static.py` (`test_spectral_entropy_residue_bins_known_misses`).

## How these numbers were measured

- **Setup:** for each W and data set, a series was fed to `SlidingExtractor` in
  one `update` call. Each window was also cut out and passed to `Extractor`,
  and the two were compared per window.
- **Bin drift:** read through `fft_coeff-k-real` and `fft_coeff-k-imag`.
- **Seeds:** `numpy.random.default_rng(3)` for `ints` and `binary` (in that
  order, so `binary` is the second draw), `default_rng(4)` for `randn` in table
  3, and `default_rng(0)` for the timing input.
- **"Before":** the same measurements on a build with the original f32
  twiddles.
- **Regression tests:** in `tests/test_sliding.py`.
  - `test_sliding_spectral_entropy_exact_zero_bins` locks in the twiddle fix.
  - `test_sliding_exact_bin_comparisons_binary_data` (strict `xfail`) records
    the default's remaining mismatches.
  - `test_sliding_fresh_fft_meta_feature` checks that `fresh-1` is exact.
