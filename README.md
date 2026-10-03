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
| Date                |   CommitHash | Benchmark_Name                       |   Metric_Value | Unit   |   Delta_From_Last | Direction   |
|:--------------------|-------------:|:-------------------------------------|---------------:|:-------|------------------:|:------------|
| 2026-10-03 23:38:05 |      1891821 | tsfast_sliding_first_window          |       1.2219   | ms     |            -24.18 | 🟢          |
| 2026-10-03 23:38:05 |      1891821 | tsfast_sliding_avg_next_50           |       0.471468 | ms     |              7.15 | 🔴          |
| 2026-10-03 23:38:05 |      1891821 | tsfast_expanding_first_window        |       0.310421 | ms     |             -0.76 | ⚪          |
| 2026-10-03 23:38:05 |      1891821 | tsfast_expanding_avg_next_50         |       1.02478  | ms     |              4.07 | ⚪          |
| 2026-10-03 23:38:05 |      1891821 | tsfast_static_sliding_first_window   |       0.469446 | ms     |            -16.07 | 🟢          |
| 2026-10-03 23:38:05 |      1891821 | tsfast_static_sliding_avg_next_50    |       0.397716 | ms     |             -1.24 | ⚪          |
| 2026-10-03 23:38:05 |      1891821 | tsfast_static_expanding_first_window |       0.385523 | ms     |             -0.98 | ⚪          |
| 2026-10-03 23:38:05 |      1891821 | tsfast_static_expanding_avg_next_50  |       1.79659  | ms     |             -0.24 | ⚪          |
| 2026-10-03 23:38:05 |      1891821 | tsfresh_sliding_first_window         |      61.6548   | ms     |              0.62 | ⚪          |
| 2026-10-03 23:38:05 |      1891821 | tsfresh_sliding_avg_next_50          |      56.2862   | ms     |              1.51 | ⚪          |
| 2026-10-03 23:38:05 |      1891821 | tsfresh_expanding_first_window       |      54.5392   | ms     |             -0.14 | ⚪          |
| 2026-10-03 23:38:05 |      1891821 | tsfresh_expanding_avg_next_50        |     271.827    | ms     |              0.23 | ⚪          |
| 2026-10-03 23:38:05 |      1891821 | tsfel_sliding_first_window           |      84.0225   | ms     |              0.39 | ⚪          |
| 2026-10-03 23:38:05 |      1891821 | tsfel_sliding_avg_next_50            |      82.16     | ms     |              0.67 | ⚪          |
| 2026-10-03 23:38:05 |      1891821 | tsfel_expanding_first_window         |      81.9354   | ms     |              0.44 | ⚪          |
| 2026-10-03 23:38:05 |      1891821 | tsfel_expanding_avg_next_50          |      94.7913   | ms     |              0.63 | ⚪          |
| 2026-10-03 23:38:05 |      1891821 | tsfast_compatible_features           |      46        | count  |              0    | ⚪          |
| 2026-10-03 23:38:05 |      1891821 | tsfresh_compatible_features          |      40        | count  |              0    | ⚪          |
| 2026-10-03 23:38:05 |      1891821 | tsfel_compatible_features            |      17        | count  |              0    | ⚪          |
