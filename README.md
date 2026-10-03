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
| 2026-10-03 19:50:13 |      7848612 | tsfast_sliding_first_window          |       15.6658  | ms     |            -39.52 | 🟢          |
| 2026-10-03 19:50:13 |      7848612 | tsfast_sliding_avg_next_50           |        9.07289 | ms     |             28.11 | 🔴          |
| 2026-10-03 19:50:13 |      7848612 | tsfast_expanding_first_window        |        4.13704 | ms     |             38.72 | 🔴          |
| 2026-10-03 19:50:13 |      7848612 | tsfast_expanding_avg_next_50         |       15.2012  | ms     |             16.69 | 🔴          |
| 2026-10-03 19:50:13 |      7848612 | tsfast_static_sliding_first_window   |        7.57575 | ms     |             37.19 | 🔴          |
| 2026-10-03 19:50:13 |      7848612 | tsfast_static_sliding_avg_next_50    |        6.14693 | ms     |             14.23 | 🔴          |
| 2026-10-03 19:50:13 |      7848612 | tsfast_static_expanding_first_window |        7.37262 | ms     |             40.58 | 🔴          |
| 2026-10-03 19:50:13 |      7848612 | tsfast_static_expanding_avg_next_50  |       69.3775  | ms     |             48.33 | 🔴          |
| 2026-10-03 19:50:13 |      7848612 | tsfresh_sliding_first_window         |      261.893   | ms     |            103.95 | 🔴          |
| 2026-10-03 19:50:13 |      7848612 | tsfresh_sliding_avg_next_50          |      124.532   | ms     |              5.26 | 🔴          |
| 2026-10-03 19:50:13 |      7848612 | tsfresh_expanding_first_window       |      118.016   | ms     |              0.65 | ⚪          |
| 2026-10-03 19:50:13 |      7848612 | tsfresh_expanding_avg_next_50        |      495.838   | ms     |             -0.11 | ⚪          |
| 2026-10-03 19:50:13 |      7848612 | tsfel_sliding_first_window           |      211.906   | ms     |              0.49 | ⚪          |
| 2026-10-03 19:50:13 |      7848612 | tsfel_sliding_avg_next_50            |      230.422   | ms     |             10.55 | 🔴          |
| 2026-10-03 19:50:13 |      7848612 | tsfel_expanding_first_window         |      213.06    | ms     |              2.72 | ⚪          |
| 2026-10-03 19:50:13 |      7848612 | tsfel_expanding_avg_next_50          |      250.69    | ms     |              4.3  | ⚪          |
| 2026-10-03 19:50:13 |      7848612 | tsfast_compatible_features           |       46       | count  |              0    | ⚪          |
| 2026-10-03 19:50:13 |      7848612 | tsfresh_compatible_features          |       40       | count  |              0    | ⚪          |
| 2026-10-03 19:50:13 |      7848612 | tsfel_compatible_features            |       17       | count  |              0    | ⚪          |
