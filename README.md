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
| 2026-10-04 09:07:39 |      7848612 | tsfast_sliding_first_window          |       47.8637  | ms     |           2870.19 | 🔴          |
| 2026-10-04 09:07:39 |      7848612 | tsfast_sliding_avg_next_50           |        1.51978 | ms     |            245.39 | 🔴          |
| 2026-10-04 09:07:39 |      7848612 | tsfast_expanding_first_window        |        1.74403 | ms     |            457.55 | 🔴          |
| 2026-10-04 09:07:39 |      7848612 | tsfast_expanding_avg_next_50         |        3.16363 | ms     |            221.27 | 🔴          |
| 2026-10-04 09:07:39 |      7848612 | tsfast_static_sliding_first_window   |        2.83313 | ms     |            406.52 | 🔴          |
| 2026-10-04 09:07:39 |      7848612 | tsfast_static_sliding_avg_next_50    |        1.7323  | ms     |            330.15 | 🔴          |
| 2026-10-04 09:07:39 |      7848612 | tsfast_static_expanding_first_window |        1.43242 | ms     |            267.91 | 🔴          |
| 2026-10-04 09:07:39 |      7848612 | tsfast_static_expanding_avg_next_50  |        5.35526 | ms     |            197.37 | 🔴          |
| 2026-10-04 09:07:39 |      7848612 | tsfresh_sliding_first_window         |      143.348   | ms     |            133.94 | 🔴          |
| 2026-10-04 09:07:39 |      7848612 | tsfresh_sliding_avg_next_50          |      130.86    | ms     |            136.01 | 🔴          |
| 2026-10-04 09:07:39 |      7848612 | tsfresh_expanding_first_window       |      125.817   | ms     |            130.38 | 🔴          |
| 2026-10-04 09:07:39 |      7848612 | tsfresh_expanding_avg_next_50        |      524.359   | ms     |             93.35 | 🔴          |
| 2026-10-04 09:07:39 |      7848612 | tsfel_sliding_first_window           |      240.97    | ms     |            187.92 | 🔴          |
| 2026-10-04 09:07:39 |      7848612 | tsfel_sliding_avg_next_50            |      238.146   | ms     |            191.8  | 🔴          |
| 2026-10-04 09:07:39 |      7848612 | tsfel_expanding_first_window         |      245.813   | ms     |            201.34 | 🔴          |
| 2026-10-04 09:07:39 |      7848612 | tsfel_expanding_avg_next_50          |      278.302   | ms     |            195.45 | 🔴          |
| 2026-10-04 09:07:39 |      7848612 | tsfast_compatible_features           |       46       | count  |              0    | ⚪          |
| 2026-10-04 09:07:39 |      7848612 | tsfresh_compatible_features          |       40       | count  |              0    | ⚪          |
| 2026-10-04 09:07:39 |      7848612 | tsfel_compatible_features            |       17       | count  |              0    | ⚪          |
