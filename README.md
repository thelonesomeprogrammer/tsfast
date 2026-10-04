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
| 2026-10-04 12:43:00 | 061ccde      | tsfast_sliding_first_window          |       1.28889  | ms     |            -20.02 | 🟢          |
| 2026-10-04 12:43:00 | 061ccde      | tsfast_sliding_avg_next_50           |       0.484266 | ms     |             10.06 | 🔴          |
| 2026-10-04 12:43:00 | 061ccde      | tsfast_expanding_first_window        |       0.336885 | ms     |              7.7  | 🔴          |
| 2026-10-04 12:43:00 | 061ccde      | tsfast_expanding_avg_next_50         |       1.03691  | ms     |              5.3  | 🔴          |
| 2026-10-04 12:43:00 | 061ccde      | tsfast_static_sliding_first_window   |       0.556231 | ms     |             -0.55 | ⚪          |
| 2026-10-04 12:43:00 | 061ccde      | tsfast_static_sliding_avg_next_50    |       0.436473 | ms     |              8.38 | 🔴          |
| 2026-10-04 12:43:00 | 061ccde      | tsfast_static_expanding_first_window |       0.436068 | ms     |             12    | 🔴          |
| 2026-10-04 12:43:00 | 061ccde      | tsfast_static_expanding_avg_next_50  |       1.94816  | ms     |              8.18 | 🔴          |
| 2026-10-04 12:43:00 | 061ccde      | tsfresh_sliding_first_window         |      57.1725   | ms     |             -6.7  | 🟢          |
| 2026-10-04 12:43:00 | 061ccde      | tsfresh_sliding_avg_next_50          |      54.6306   | ms     |             -1.47 | ⚪          |
| 2026-10-04 12:43:00 | 061ccde      | tsfresh_expanding_first_window       |      54.5239   | ms     |             -0.16 | ⚪          |
| 2026-10-04 12:43:00 | 061ccde      | tsfresh_expanding_avg_next_50        |     270.322    | ms     |             -0.32 | ⚪          |
| 2026-10-04 12:43:00 | 061ccde      | tsfel_sliding_first_window           |      83.9944   | ms     |              0.36 | ⚪          |
| 2026-10-04 12:43:00 | 061ccde      | tsfel_sliding_avg_next_50            |      81.8874   | ms     |              0.34 | ⚪          |
| 2026-10-04 12:43:00 | 061ccde      | tsfel_expanding_first_window         |      81.8408   | ms     |              0.33 | ⚪          |
| 2026-10-04 12:43:00 | 061ccde      | tsfel_expanding_avg_next_50          |      94.322    | ms     |              0.14 | ⚪          |
| 2026-10-04 12:43:00 | 061ccde      | tsfast_compatible_features           |      46        | count  |              0    | ⚪          |
| 2026-10-04 12:43:00 | 061ccde      | tsfresh_compatible_features          |      40        | count  |              0    | ⚪          |
| 2026-10-04 12:43:00 | 061ccde      | tsfel_compatible_features            |      17        | count  |              0    | ⚪          |
