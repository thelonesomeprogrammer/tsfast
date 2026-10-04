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
| Date                | CommitHash   | Benchmark_Name                       |   Metric_Value | Unit   |   Delta_From_Last | Direction   |
|:--------------------|:-------------|:-------------------------------------|---------------:|:-------|------------------:|:------------|
| 2026-10-04 12:03:57 | c4011a5      | tsfast_sliding_first_window          |       3.05915  | ms     |             89.84 | 🔴          |
| 2026-10-04 12:03:57 | c4011a5      | tsfast_sliding_avg_next_50           |       0.477333 | ms     |              8.48 | 🔴          |
| 2026-10-04 12:03:57 | c4011a5      | tsfast_expanding_first_window        |       0.307083 | ms     |             -1.83 | ⚪          |
| 2026-10-04 12:03:57 | c4011a5      | tsfast_expanding_avg_next_50         |       1.05596  | ms     |              7.23 | 🔴          |
| 2026-10-04 12:03:57 | c4011a5      | tsfast_static_sliding_first_window   |       0.54121  | ms     |             -3.24 | ⚪          |
| 2026-10-04 12:03:57 | c4011a5      | tsfast_static_sliding_avg_next_50    |       0.394392 | ms     |             -2.07 | ⚪          |
| 2026-10-04 12:03:57 | c4011a5      | tsfast_static_expanding_first_window |       0.387192 | ms     |             -0.55 | ⚪          |
| 2026-10-04 12:03:57 | c4011a5      | tsfast_static_expanding_avg_next_50  |       2.11765  | ms     |             17.59 | 🔴          |
| 2026-10-04 12:03:57 | c4011a5      | tsfresh_sliding_first_window         |      67.2317   | ms     |              9.72 | 🔴          |
| 2026-10-04 12:03:57 | c4011a5      | tsfresh_sliding_avg_next_50          |      56.0014   | ms     |              1    | ⚪          |
| 2026-10-04 12:03:57 | c4011a5      | tsfresh_expanding_first_window       |      54.2254   | ms     |             -0.71 | ⚪          |
| 2026-10-04 12:03:57 | c4011a5      | tsfresh_expanding_avg_next_50        |     269.9      | ms     |             -0.48 | ⚪          |
| 2026-10-04 12:03:57 | c4011a5      | tsfel_sliding_first_window           |      83.8804   | ms     |              0.22 | ⚪          |
| 2026-10-04 12:03:57 | c4011a5      | tsfel_sliding_avg_next_50            |      81.3537   | ms     |             -0.32 | ⚪          |
| 2026-10-04 12:03:57 | c4011a5      | tsfel_expanding_first_window         |      81.2352   | ms     |             -0.41 | ⚪          |
| 2026-10-04 12:03:57 | c4011a5      | tsfel_expanding_avg_next_50          |      93.9229   | ms     |             -0.29 | ⚪          |
| 2026-10-04 12:03:57 | c4011a5      | tsfast_compatible_features           |      46        | count  |              0    | ⚪          |
| 2026-10-04 12:03:57 | c4011a5      | tsfresh_compatible_features          |      40        | count  |              0    | ⚪          |
| 2026-10-04 12:03:57 | c4011a5      | tsfel_compatible_features            |      17        | count  |              0    | ⚪          |
