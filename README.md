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
| 2026-10-04 10:33:00 | 3c46fd3      | tsfast_sliding_first_window          |        8.22139 | ms     |            503.8  | 🔴          |
| 2026-10-04 10:33:00 | 3c46fd3      | tsfast_sliding_avg_next_50           |        7.52452 | ms     |           1534.44 | 🔴          |
| 2026-10-04 10:33:00 | 3c46fd3      | tsfast_expanding_first_window        |        3.52049 | ms     |            984.14 | 🔴          |
| 2026-10-04 10:33:00 | 3c46fd3      | tsfast_expanding_avg_next_50         |       13.6576  | ms     |           1090.19 | 🔴          |
| 2026-10-04 10:33:00 | 3c46fd3      | tsfast_static_sliding_first_window   |        5.68795 | ms     |           1197.99 | 🔴          |
| 2026-10-04 10:33:00 | 3c46fd3      | tsfast_static_sliding_avg_next_50    |        5.59887 | ms     |           1299.08 | 🔴          |
| 2026-10-04 10:33:00 | 3c46fd3      | tsfast_static_expanding_first_window |        5.78403 | ms     |           1375.67 | 🔴          |
| 2026-10-04 10:33:00 | 3c46fd3      | tsfast_static_expanding_avg_next_50  |       49.6946  | ms     |           2567.38 | 🔴          |
| 2026-10-04 10:33:00 | 3c46fd3      | tsfresh_sliding_first_window         |      124.864   | ms     |            104.55 | 🔴          |
| 2026-10-04 10:33:00 | 3c46fd3      | tsfresh_sliding_avg_next_50          |      125.169   | ms     |            128.25 | 🔴          |
| 2026-10-04 10:33:00 | 3c46fd3      | tsfresh_expanding_first_window       |      127.007   | ms     |            134.22 | 🔴          |
| 2026-10-04 10:33:00 | 3c46fd3      | tsfresh_expanding_avg_next_50        |      490.474   | ms     |             82.16 | 🔴          |
| 2026-10-04 10:33:00 | 3c46fd3      | tsfel_sliding_first_window           |      215.123   | ms     |            156.04 | 🔴          |
| 2026-10-04 10:33:00 | 3c46fd3      | tsfel_sliding_avg_next_50            |      211.037   | ms     |            158.54 | 🔴          |
| 2026-10-04 10:33:00 | 3c46fd3      | tsfel_expanding_first_window         |      207.78    | ms     |            154.82 | 🔴          |
| 2026-10-04 10:33:00 | 3c46fd3      | tsfel_expanding_avg_next_50          |      238.335   | ms     |            153.33 | 🔴          |
| 2026-10-04 10:33:00 | 3c46fd3      | tsfast_compatible_features           |       46       | count  |              0    | ⚪          |
| 2026-10-04 10:33:00 | 3c46fd3      | tsfresh_compatible_features          |       40       | count  |              0    | ⚪          |
| 2026-10-04 10:33:00 | 3c46fd3      | tsfel_compatible_features            |       17       | count  |              0    | ⚪          |
