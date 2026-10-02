# TSFast: Ultra-Fast Time Series Feature Extraction

TSFast is a high-performance time-series feature extraction library written in Rust with Python bindings. It is designed for extreme speed, efficient memory usage, and interoperability with Apache Arrow.

## Key Features

- **Blazing Fast**: Core engine implemented in Rust with SIMD (Portable SIMD) for maximum performance.
- **O(n) Expanding Windows**: Highly optimized algorithms for expanding window feature extraction (prefix statistics).
- **Arrow Integration**: Uses Apache Arrow for efficient, zero-copy-ready data handling via `pyarrow`.
- **Selective Execution**: Only computes the features you request, using a bitmask-based engine to skip unnecessary calculations.
- **Python-Friendly**: Simple API based on `Extractor` and `ExpandingExtractor` classes.

## Benchmarks

![Benchmark Trends](.jules/benchmark_trends.png)

## Latest Results
| Date                | CommitHash   | Benchmark_Name              |   Metric_Value | Unit   |   Delta_From_Last |
|:--------------------|:-------------|:----------------------------|---------------:|:-------|------------------:|
| 2026-10-02 11:23:43 | 63fbb8e      | tsfast_ms_per_window        |        4.31971 | ms     |                 0 |
| 2026-10-02 11:23:43 | 63fbb8e      | tsfresh_ms_per_window       |      183.944   | ms     |                 0 |
| 2026-10-02 11:23:43 | 63fbb8e      | tsfel_ms_per_window         |      237.853   | ms     |                 0 |
| 2026-10-02 11:23:43 | 63fbb8e      | tsfast_compatible_features  |       46       | count  |                 0 |
| 2026-10-02 11:23:43 | 63fbb8e      | tsfresh_compatible_features |       40       | count  |                 0 |
| 2026-10-02 11:23:43 | 63fbb8e      | tsfel_compatible_features   |       17       | count  |                 0 |
