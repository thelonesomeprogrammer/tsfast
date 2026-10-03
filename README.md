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
| 2026-10-03 21:01:26 |      7848612 | tsfast_sliding_first_window          |       47.0901  | ms     |           2822.18 | 🔴          |
| 2026-10-03 21:01:26 |      7848612 | tsfast_sliding_avg_next_50           |        7.47156 | ms     |           1598    | 🔴          |
| 2026-10-03 21:01:26 |      7848612 | tsfast_expanding_first_window        |        3.84808 | ms     |           1130.18 | 🔴          |
| 2026-10-03 21:01:26 |      7848612 | tsfast_expanding_avg_next_50         |       12.6884  | ms     |           1188.52 | 🔴          |
| 2026-10-03 21:01:26 |      7848612 | tsfast_static_sliding_first_window   |        6.61945 | ms     |           1083.46 | 🔴          |
| 2026-10-03 21:01:26 |      7848612 | tsfast_static_sliding_avg_next_50    |        5.27906 | ms     |           1210.84 | 🔴          |
| 2026-10-03 21:01:26 |      7848612 | tsfast_static_expanding_first_window |        5.46336 | ms     |           1303.25 | 🔴          |
| 2026-10-03 21:01:26 |      7848612 | tsfast_static_expanding_avg_next_50  |       43.1152  | ms     |           2294.1  | 🔴          |
| 2026-10-03 21:01:26 |      7848612 | tsfresh_sliding_first_window         |      125.64    | ms     |            105.04 | 🔴          |
| 2026-10-03 21:01:26 |      7848612 | tsfresh_sliding_avg_next_50          |      121.435   | ms     |            119.01 | 🔴          |
| 2026-10-03 21:01:26 |      7848612 | tsfresh_expanding_first_window       |      121.362   | ms     |            122.22 | 🔴          |
| 2026-10-03 21:01:26 |      7848612 | tsfresh_expanding_avg_next_50        |      523.781   | ms     |             93.14 | 🔴          |
| 2026-10-03 21:01:26 |      7848612 | tsfel_sliding_first_window           |      217.537   | ms     |            159.92 | 🔴          |
| 2026-10-03 21:01:26 |      7848612 | tsfel_sliding_avg_next_50            |      234.372   | ms     |            187.17 | 🔴          |
| 2026-10-03 21:01:26 |      7848612 | tsfel_expanding_first_window         |      209.358   | ms     |            156.65 | 🔴          |
| 2026-10-03 21:01:26 |      7848612 | tsfel_expanding_avg_next_50          |      297.96    | ms     |            216.32 | 🔴          |
| 2026-10-03 21:01:26 |      7848612 | tsfast_compatible_features           |       46       | count  |              0    | ⚪          |
| 2026-10-03 21:01:26 |      7848612 | tsfresh_compatible_features          |       40       | count  |              0    | ⚪          |
| 2026-10-03 21:01:26 |      7848612 | tsfel_compatible_features            |       17       | count  |              0    | ⚪          |
