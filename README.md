# TSFast: Ultra-Fast Time Series Feature Extraction

TSFast is a high-performance time-series feature extraction library written in Rust with Python bindings. It is designed for extreme speed, efficient memory usage, and interoperability with Apache Arrow.

## Key Features

- **Blazing Fast**: Core engine implemented in Rust with SIMD (Portable SIMD) for maximum performance.
- **O(n) Expanding Windows**: Highly optimized algorithms for expanding window feature extraction (prefix statistics).
- **Arrow Integration**: Uses Apache Arrow for efficient, zero-copy-ready data handling via `pyarrow`.
- **Selective Execution**: Only computes the features you request, using a bitmask-based engine to skip unnecessary calculations.
- **Python-Friendly**: Simple API based on `Extractor` and `ExpandingExtractor` classes.

## Benchmarks & Comparisons

TSFast demonstrates a **>100x speedup** over TSFEL and a **>6000x speedup** over tsfresh for common feature sets, while maintaining identical predictive power.

| Library | Extraction Time (s) | Feature Count | Primary Strength |
|---------|----------------------|---------------|------------------|
| **TSFast** | **0.0081s** | ~50 (Optimized) | Extreme Speed & Streaming |
| TSFEL | 0.9002s | 60+ | Domain-specific (Health/Acoustic) |
| tsfresh | 56.0196s | 777+ | Exhaustive Feature Mining |

### Library Strengths & When to Use

1. **TSFast**: Best for high-throughput production environments, real-time streaming (via `ExpandingExtractor`), and scenarios where speed is critical. It focuses on a highly optimized subset of the most predictive features.
2. **tsfresh**: Best for exploratory data analysis where you want to exhaustively mine hundreds of features to find any potential signal, regardless of computational cost.
3. **TSFEL**: A great middle ground, offering a solid library of features with specific domains (like medical or signal processing) and better performance than tsfresh.

*Note: Benchmarks were performed on 1000 samples of 1000 points each, using comparable feature subsets where possible to ensure a fair representation of core engine performance.*

## Installation

```bash
# Requires Rust (nightly) for SIMD features
pip install .
```

## Quick Start

### 1. Batch Extraction (Static)

```python
import tsfast
import numpy as np
import pyarrow as pa

# Sample signal (1000 samples)
x = np.random.randn(1000).astype(np.float32)

# Initialize Extractor
features = ["mean", "std_dev", "energy", "min_value", "max_value", "autocorr_lag1"]
extractor = tsfast.Extractor(features)

# Wrap data in an Arrow RecordBatch (each column is a time series)
batch = pa.RecordBatch.from_arrays([pa.array(x)], names=['series1'])

# Process
result_batch = extractor.process_2d_floats(batch)

# Get results as a Pandas DataFrame or NumPy array
results_df = result_batch.to_pandas()
print(results_df)
```

### 2. Expanding Window Extraction

```python
import tsfast
import pyarrow as pa
import numpy as np

# Initialize ExpandingExtractor for 1 series
features = ["mean", "max_value", "total_sum"]
extractor = tsfast.ExpandingExtractor(features, n_cols=1)

# Stream data in batches
for i in range(5):
    chunk = np.random.randn(10).astype(np.float32)
    batch = pa.RecordBatch.from_arrays([pa.array(chunk)], names=['c1'])
    
    # Returns expanding features for EACH point in the chunk
    expanding_results = extractor.update(batch).to_pandas()
    print(f"Batch {i} processed, shape: {expanding_results.shape}")
```

## Supported Features

TSFast supports a wide range of features, including:

- **Statistical**: `mean`, `variance`, `std_dev`, `min_value`, `max_value`, `median`, `skewness`, `kurtosis`, `mad`, `iqr`, `entropy`, `variation_coefficient`.
- **Energy/Signal**: `total_sum`, `energy`, `rms`, `root_mean_square`, `zero_crossing_rate`, `peak_count`, `auc`, `abs_max`.
- **Temporal/Change**: `mean_abs_change`, `mean_change`, `abs_sum_change`, `cid_ce`.
- **Location-based**: `first_loc_max`, `last_loc_max`, `first_loc_min`, `last_loc_min`.
- **Advanced Statistics**:
  - `autocorr_lag1`, `autocorr-N` (e.g., `autocorr-5`)
  - `partial_autocorr-N`
  - `c3-N` (higher-order statistics)
  - `time_reversal_asymmetry-N`
- **Transforms**:
  - `paa-N-M` (Piecewise Aggregate Approximation: N segments, return index M)
  - `fft_coeff-N-ATTR` (FFT coefficient N, ATTR is real/imag/abs/angle)
- **Complex Features**:
  - `approx_entropy-M-R` (Approximate Entropy)
  - `agg_linear_trend-ATTR-CHUNK_LEN-FUNC` (Aggregated linear trend)

## License

GPLv3
