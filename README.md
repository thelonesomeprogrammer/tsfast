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
| 2026-10-04 00:00:50 | 65a51e7      | tsfast_sliding_first_window          |       1.36161  | ms     |            -15.51 | 🟢          |
| 2026-10-04 00:00:50 | 65a51e7      | tsfast_sliding_avg_next_50           |       0.460372 | ms     |              4.63 | ⚪          |
| 2026-10-04 00:00:50 | 65a51e7      | tsfast_expanding_first_window        |       0.324726 | ms     |              3.81 | ⚪          |
| 2026-10-04 00:00:50 | 65a51e7      | tsfast_expanding_avg_next_50         |       1.14751  | ms     |             16.53 | 🔴          |
| 2026-10-04 00:00:50 | 65a51e7      | tsfast_static_sliding_first_window   |       0.438213 | ms     |            -21.65 | 🟢          |
| 2026-10-04 00:00:50 | 65a51e7      | tsfast_static_sliding_avg_next_50    |       0.400181 | ms     |             -0.63 | ⚪          |
| 2026-10-04 00:00:50 | 65a51e7      | tsfast_static_expanding_first_window |       0.39196  | ms     |              0.67 | ⚪          |
| 2026-10-04 00:00:50 | 65a51e7      | tsfast_static_expanding_avg_next_50  |       1.86305  | ms     |              3.45 | ⚪          |
| 2026-10-04 00:00:50 | 65a51e7      | tsfresh_sliding_first_window         |      61.0423   | ms     |             -0.38 | ⚪          |
| 2026-10-04 00:00:50 | 65a51e7      | tsfresh_sliding_avg_next_50          |      54.8397   | ms     |             -1.1  | ⚪          |
| 2026-10-04 00:00:50 | 65a51e7      | tsfresh_expanding_first_window       |      54.2252   | ms     |             -0.71 | ⚪          |
| 2026-10-04 00:00:50 | 65a51e7      | tsfresh_expanding_avg_next_50        |     269.254    | ms     |             -0.72 | ⚪          |
| 2026-10-04 00:00:50 | 65a51e7      | tsfel_sliding_first_window           |      84.0206   | ms     |              0.39 | ⚪          |
| 2026-10-04 00:00:50 | 65a51e7      | tsfel_sliding_avg_next_50            |      81.6275   | ms     |              0.02 | ⚪          |
| 2026-10-04 00:00:50 | 65a51e7      | tsfel_expanding_first_window         |      81.5401   | ms     |             -0.04 | ⚪          |
| 2026-10-04 00:00:50 | 65a51e7      | tsfel_expanding_avg_next_50          |      94.0791   | ms     |             -0.12 | ⚪          |
| 2026-10-04 00:00:50 | 65a51e7      | tsfast_compatible_features           |      46        | count  |              0    | ⚪          |
| 2026-10-04 00:00:50 | 65a51e7      | tsfresh_compatible_features          |      40        | count  |              0    | ⚪          |
| 2026-10-04 00:00:50 | 65a51e7      | tsfel_compatible_features            |      17        | count  |              0    | ⚪          |
