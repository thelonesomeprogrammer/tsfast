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
| 2026-10-03 09:29:34 | 2f889b1      | tsfast_sliding_first_window          |       5.54824  | ms     |             51.25 | 🔴          |
| 2026-10-03 09:29:34 | 2f889b1      | tsfast_sliding_avg_next_50           |       1.29825  | ms     |             40.91 | 🔴          |
| 2026-10-03 09:29:34 | 2f889b1      | tsfast_expanding_first_window        |       0.354052 | ms     |            -51.8  | 🟢          |
| 2026-10-03 09:29:34 | 2f889b1      | tsfast_expanding_avg_next_50         |       2.52295  | ms     |            -19.12 | 🟢          |
| 2026-10-03 09:29:34 | 2f889b1      | tsfast_static_sliding_first_window   |       0.503063 | ms     |            -38.18 | 🟢          |
| 2026-10-03 09:29:34 | 2f889b1      | tsfast_static_sliding_avg_next_50    |       1.34129  | ms     |             35.67 | 🔴          |
| 2026-10-03 09:29:34 | 2f889b1      | tsfast_static_expanding_first_window |       1.24812  | ms     |             38.82 | 🔴          |
| 2026-10-03 09:29:34 | 2f889b1      | tsfast_static_expanding_avg_next_50  |       3.78785  | ms     |             -6.5  | 🟢          |
| 2026-10-03 09:29:34 | 2f889b1      | tsfresh_sliding_first_window         |      64.9228   | ms     |            -42.79 | 🟢          |
| 2026-10-03 09:29:34 | 2f889b1      | tsfresh_sliding_avg_next_50          |      58.4258   | ms     |            -45.9  | 🟢          |
| 2026-10-03 09:29:34 | 2f889b1      | tsfresh_expanding_first_window       |      58.3563   | ms     |            -45.71 | 🟢          |
| 2026-10-03 09:29:34 | 2f889b1      | tsfresh_expanding_avg_next_50        |     291.882    | ms     |            -33.52 | 🟢          |
| 2026-10-03 09:29:34 | 2f889b1      | tsfel_sliding_first_window           |     103.332    | ms     |            -45.02 | 🟢          |
| 2026-10-03 09:29:34 | 2f889b1      | tsfel_sliding_avg_next_50            |      89.6765   | ms     |            -52.93 | 🟢          |
| 2026-10-03 09:29:34 | 2f889b1      | tsfel_expanding_first_window         |      92.361    | ms     |            -49.96 | 🟢          |
| 2026-10-03 09:29:34 | 2f889b1      | tsfel_expanding_avg_next_50          |     102.707    | ms     |            -52.57 | 🟢          |
| 2026-10-03 09:29:34 | 2f889b1      | tsfast_compatible_features           |      46        | count  |              0    | ⚪          |
| 2026-10-03 09:29:34 | 2f889b1      | tsfresh_compatible_features          |      40        | count  |              0    | ⚪          |
| 2026-10-03 09:29:34 | 2f889b1      | tsfel_compatible_features            |      17        | count  |              0    | ⚪          |
