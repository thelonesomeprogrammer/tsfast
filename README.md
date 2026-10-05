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
| 2026-10-05 15:22:58 | f405f4f      | tsfast_sliding_first_window          |       11.8015  | ms     |            784.7  | 🔴          |
| 2026-10-05 15:22:58 | f405f4f      | tsfast_sliding_avg_next_50           |        9.67526 | ms     |           2027.89 | 🔴          |
| 2026-10-05 15:22:58 | f405f4f      | tsfast_expanding_first_window        |        5.79381 | ms     |           1704.08 | 🔴          |
| 2026-10-05 15:22:58 | f405f4f      | tsfast_expanding_avg_next_50         |       36.1288  | ms     |           3669.97 | 🔴          |
| 2026-10-05 15:22:58 | f405f4f      | tsfast_static_sliding_first_window   |        9.68766 | ms     |           2376.11 | 🔴          |
| 2026-10-05 15:22:58 | f405f4f      | tsfast_static_sliding_avg_next_50    |        7.88785 | ms     |           1968.37 | 🔴          |
| 2026-10-05 15:22:58 | f405f4f      | tsfast_static_expanding_first_window |        8.73137 | ms     |           2171.84 | 🔴          |
| 2026-10-05 15:22:58 | f405f4f      | tsfast_static_expanding_avg_next_50  |       82.9776  | ms     |           4531.66 | 🔴          |
| 2026-10-05 15:22:58 | f405f4f      | tsfresh_sliding_first_window         |      135.806   | ms     |            123.07 | 🔴          |
| 2026-10-05 15:22:58 | f405f4f      | tsfresh_sliding_avg_next_50          |      144.137   | ms     |            158.89 | 🔴          |
| 2026-10-05 15:22:58 | f405f4f      | tsfresh_expanding_first_window       |      129.338   | ms     |            139    | 🔴          |
| 2026-10-05 15:22:58 | f405f4f      | tsfresh_expanding_avg_next_50        |      540.071   | ms     |             97.5  | 🔴          |
| 2026-10-05 15:22:58 | f405f4f      | tsfel_sliding_first_window           |      262.368   | ms     |            214.01 | 🔴          |
| 2026-10-05 15:22:58 | f405f4f      | tsfel_sliding_avg_next_50            |      269.656   | ms     |            231.12 | 🔴          |
| 2026-10-05 15:22:58 | f405f4f      | tsfel_expanding_first_window         |      261.781   | ms     |            221.97 | 🔴          |
| 2026-10-05 15:22:58 | f405f4f      | tsfel_expanding_avg_next_50          |      314.18    | ms     |            234.73 | 🔴          |
| 2026-10-05 15:22:58 | f405f4f      | tsfast_compatible_features           |       46       | count  |              0    | ⚪          |
| 2026-10-05 15:22:58 | f405f4f      | tsfresh_compatible_features          |       40       | count  |              0    | ⚪          |
| 2026-10-05 15:22:58 | f405f4f      | tsfel_compatible_features            |       17       | count  |              0    | ⚪          |
