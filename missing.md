# Missing Features in tsfast

> **Agents:** this is the backlog. Pick tasks from here, and delete a row in the same PR that implements it.

Audited against the installed `tsfresh` (76 feature calculators) and `tsfel`
(67 feature functions). Every name below was checked against
`src/types/feature.rs`, `src/types/parse.rs`, and `tests/feature_samples.txt`
and does not exist under any alias. Everything *not* listed here is
implemented and verified within 1% of its reference by
`tests/test_references.py`.

---

## TSFresh (3 missing)

| Feature | Notes |
| :--- | :--- |
| **`fourier_entropy`** | Binned entropy of the FFT power spectrum (histogram over `bins` parameter). |
| **`fft_aggregated`** | Centroid/variance/skew/kurtosis of the power spectrum, tsfresh's own normalization (distinct from TSFEL's `spectral_centroid`/`spectral_spread`/etc., which use `FS` and are already implemented). |
| **`linear_trend_timewise`** | OLS regression against an explicit `DatetimeIndex`. tsfast's engines take no timestamp input, so this needs a design decision before implementation, not just a port. |

## TSFEL (5 missing)

| Feature | Notes |
| :--- | :--- |
| **`petrosian_fractal_dimension`** | Derivative sign-change count; cheap to add (adjacent diff + sign mask + popcount). |
| **`neighbourhood_peaks`** | Peaks that dominate a local window neighbourhood. |
| **`hist_mode`** | Mode of the value histogram; needs SIMD min/max for bin edges + vectorized binning. |
| **`average_power`** | `abs_energy / length`, i.e. mean square value (distinct from `rms`, which is its square root). |
| **`spectrogram_mean_coeff`** | tsfast's existing `spectrogram-N-T` feature does **not** implement this — see the note in `tests/references.py` (it ignores the time parameter and returns a single spectrum bin). The real TSFEL spectrogram-mean-coefficient feature is unimplemented. |
