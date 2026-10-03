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
| 2026-10-03 16:43:54 |      7848612 | tsfast_sliding_first_window          |        9.92322 | ms     |            515.79 | 🔴          |
| 2026-10-03 16:43:54 |      7848612 | tsfast_sliding_avg_next_50           |        7.86527 | ms     |           1687.48 | 🔴          |
| 2026-10-03 16:43:54 |      7848612 | tsfast_expanding_first_window        |        3.34477 | ms     |            969.28 | 🔴          |
| 2026-10-03 16:43:54 |      7848612 | tsfast_expanding_avg_next_50         |       13.6694  | ms     |           1288.14 | 🔴          |
| 2026-10-03 16:43:54 |      7848612 | tsfast_static_sliding_first_window   |        7.01022 | ms     |           1153.32 | 🔴          |
| 2026-10-03 16:43:54 |      7848612 | tsfast_static_sliding_avg_next_50    |        6.41944 | ms     |           1494.01 | 🔴          |
| 2026-10-03 16:43:54 |      7848612 | tsfast_static_expanding_first_window |        5.69129 | ms     |           1361.79 | 🔴          |
| 2026-10-03 16:43:54 |      7848612 | tsfast_static_expanding_avg_next_50  |       49.9878  | ms     |           2675.72 | 🔴          |
| 2026-10-03 16:43:54 |      7848612 | tsfresh_sliding_first_window         |      125.553   | ms     |            104.9  | 🔴          |
| 2026-10-03 16:43:54 |      7848612 | tsfresh_sliding_avg_next_50          |      119.272   | ms     |            115.11 | 🔴          |
| 2026-10-03 16:43:54 |      7848612 | tsfresh_expanding_first_window       |      117.211   | ms     |            114.62 | 🔴          |
| 2026-10-03 16:43:54 |      7848612 | tsfresh_expanding_avg_next_50        |      506.425   | ms     |             86.74 | 🔴          |
| 2026-10-03 16:43:54 |      7848612 | tsfel_sliding_first_window           |      218.473   | ms     |            161.04 | 🔴          |
| 2026-10-03 16:43:54 |      7848612 | tsfel_sliding_avg_next_50            |      217.177   | ms     |            166.11 | 🔴          |
| 2026-10-03 16:43:54 |      7848612 | tsfel_expanding_first_window         |      210.129   | ms     |            157.6  | 🔴          |
| 2026-10-03 16:43:54 |      7848612 | tsfel_expanding_avg_next_50          |      247.629   | ms     |            162.89 | 🔴          |
| 2026-10-03 16:43:54 |      7848612 | tsfast_compatible_features           |       46       | count  |              0    | ⚪          |
| 2026-10-03 16:43:54 |      7848612 | tsfresh_compatible_features          |       40       | count  |              0    | ⚪          |
| 2026-10-03 16:43:54 |      7848612 | tsfel_compatible_features            |       17       | count  |              0    | ⚪          |
