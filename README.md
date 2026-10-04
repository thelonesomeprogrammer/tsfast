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
| 2026-10-04 10:33:05 | 3dba8a2      | tsfast_sliding_first_window          |        9.48191 | ms     |            596.38 | 🔴          |
| 2026-10-04 10:33:05 | 3dba8a2      | tsfast_sliding_avg_next_50           |        8.02908 | ms     |           1644.04 | 🔴          |
| 2026-10-04 10:33:05 | 3dba8a2      | tsfast_expanding_first_window        |        4.01402 | ms     |           1136.12 | 🔴          |
| 2026-10-04 10:33:05 | 3dba8a2      | tsfast_expanding_avg_next_50         |       14.3672  | ms     |           1152.03 | 🔴          |
| 2026-10-04 10:33:05 | 3dba8a2      | tsfast_static_sliding_first_window   |        5.69391 | ms     |           1199.35 | 🔴          |
| 2026-10-04 10:33:05 | 3dba8a2      | tsfast_static_sliding_avg_next_50    |        6.51446 | ms     |           1527.88 | 🔴          |
| 2026-10-04 10:33:05 | 3dba8a2      | tsfast_static_expanding_first_window |        6.08969 | ms     |           1453.65 | 🔴          |
| 2026-10-04 10:33:05 | 3dba8a2      | tsfast_static_expanding_avg_next_50  |       50.981   | ms     |           2636.43 | 🔴          |
| 2026-10-04 10:33:05 | 3dba8a2      | tsfresh_sliding_first_window         |      130.839   | ms     |            114.34 | 🔴          |
| 2026-10-04 10:33:05 | 3dba8a2      | tsfresh_sliding_avg_next_50          |      130.663   | ms     |            138.26 | 🔴          |
| 2026-10-04 10:33:05 | 3dba8a2      | tsfresh_expanding_first_window       |      124.784   | ms     |            130.12 | 🔴          |
| 2026-10-04 10:33:05 | 3dba8a2      | tsfresh_expanding_avg_next_50        |      502.392   | ms     |             86.59 | 🔴          |
| 2026-10-04 10:33:05 | 3dba8a2      | tsfel_sliding_first_window           |      228.92    | ms     |            172.46 | 🔴          |
| 2026-10-04 10:33:05 | 3dba8a2      | tsfel_sliding_avg_next_50            |      228.624   | ms     |            180.08 | 🔴          |
| 2026-10-04 10:33:05 | 3dba8a2      | tsfel_expanding_first_window         |      218.341   | ms     |            167.77 | 🔴          |
| 2026-10-04 10:33:05 | 3dba8a2      | tsfel_expanding_avg_next_50          |      294.209   | ms     |            212.72 | 🔴          |
| 2026-10-04 10:33:05 | 3dba8a2      | tsfast_compatible_features           |       46       | count  |              0    | ⚪          |
| 2026-10-04 10:33:05 | 3dba8a2      | tsfresh_compatible_features          |       40       | count  |              0    | ⚪          |
| 2026-10-04 10:33:05 | 3dba8a2      | tsfel_compatible_features            |       17       | count  |              0    | ⚪          |
