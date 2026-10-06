# Missing Features in tsfast

> **Agents:** this is the backlog. Pick tasks from here, and delete a row in the same PR that implements it.

Based on a comprehensive audit of **TSFresh** (76 features) and **TSFEL** (68 features), here are the features that are **NOT yet implemented** in `tsfast`. The estimated complexity indicates the effort and algorithmic difficulty of implementing these optimally in Rust using SIMD.

---

## 1. High-Priority Algorithms
These features are heavily used in Python but suffer from extreme performance bottlenecks. Implementing them in Rust provides massive (10x-500x) speedups.

| Feature | Source | Complexity | Rust/SIMD Implementation Strategy |
| :--- | :--- | :--- | :--- |
| **`lempel_ziv_complexity`** | TSFresh/TSFEL | **High** | LZ78 complexity of discretized series. In Rust: binarize via SIMD `_mm256_movemask_pd`, then parse using a zero-allocation array-based trie or flat hash set. |

---

## 2. Statistical & Distribution Domain
Features summarizing signal amplitude distribution, central tendency, and dispersion.

| Feature | Source | Complexity | Rust/SIMD Implementation Strategy |
| :--- | :--- | :--- | :--- |
| **`ecdf_percentile`** | TSFEL | **Medium** | Value corresponding to target ECDF percentile. Quickselect / linear interpolation in Rust. |
| **`ecdf_percentile_count`**| TSFEL | **Low** | Vectorized compare `_mm256_cmp_pd` + mask sum/popcount. |
| **`ecdf_slope`** | TSFEL | **Medium** | Slope between two ECDF percentiles. Quickselect + division. |
| **`count_above` / `count_below`** | TSFresh | **Low** | Count values above/below an arbitrary threshold $t$. SIMD compare + popcount. |
| **`range_count`** | TSFresh | **Low** | Count points inside interval $[min, max)$. Double SIMD compare + AND mask + popcount. |

---

## 3. Temporal & Difference Domain
Features sensitive to the temporal order, differences, and zero crossings.

| Feature | Source | Complexity | Rust/SIMD Implementation Strategy |
| :--- | :--- | :--- | :--- |
| **`abs_percentage_sum_of_changes`**| TSFresh | **Low** | Total absolute change divided by mean/sum. |

---

## 4. Spectral & Frequency Domain
Features derived from Fourier transforms, PSD, or Cepstrum. Many can share a single FFT calculation in `tsfast`.

| Feature | Source | Complexity | Rust/SIMD Implementation Strategy |
| :--- | :--- | :--- | :--- |
| **`spectral_variation`** | TSFEL | **Medium** | Normalized spectral flux. Vectorized cross-correlation across FFT frames. |
| **`spectral_positive_turning`**| TSFEL | **Low** | Peak count in FFT magnitude. 3-way SIMD compare on FFT output. |
| **`fundamental_frequency`** | TSFEL | **Medium** | Dominant pitch frequency. Peak search in FFT magnitude or Cepstrum using SIMD argmax. |
| **`max_frequency`** | TSFEL | **Low** | Frequency of max spectral amplitude. Post-FFT SIMD argmax over magnitude. |
| **`median_frequency`** | TSFEL | **Medium** | Frequency dividing power into two equal halves. Prefix sum scan over power spectrum. |
| **`power_bandwidth`** | TSFEL | **Medium** | Bandwidth above threshold. SIMD threshold filter + index range diff on PSD. |

---

## 6. Stationarity, Autoregressive & Correlation
| Feature | Source | Complexity | Rust/SIMD Implementation Strategy |
| :--- | :--- | :--- | :--- |
| **`linear_trend_timewise`** | TSFresh | **Specialized**| OLS regression against explicit DatetimeIndex (requires timestamp injection). |

---

## 7. Fractal / Complexity Domain (Non-linear Dynamics)
Features measuring signal non-linearity, self-similarity, and fractal dimension.

| Feature | Source | Complexity | Rust/SIMD Implementation Strategy |
| :--- | :--- | :--- | :--- |
| **`petrosian_fractal_dimension`**| TSFEL | **Low** | Relies on derivative sign changes. SIMD adjacent diff + sign mask + popcount (extremely fast). |
| **`maximum_fractal_length`** | TSFEL | **High** | Shared computation with HFD; vectorized max reduction. |
| **`dfa`** (Detrended Fluct.) | TSFEL | **High** | Cumulative sum + chunked linear regressions and residual sum of squares. |
| **`hurst_exponent`** | TSFEL | **High** | Parallelized multi-scale chunking, vectorized prefix sum and Rescaled Range (R/S). |
| **`mse`** (Multiscale Entropy)| TSFEL | **Very High**| Coarse-graining via SIMD chunk mean, followed by 2D Chebyshev distance count. |
