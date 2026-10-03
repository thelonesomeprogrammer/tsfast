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
| 2026-10-03 17:26:13 |      7848612 | tsfast_sliding_first_window          |       4.35662  | ms     |            -33.16 | 🟢          |
| 2026-10-03 17:26:13 |      7848612 | tsfast_sliding_avg_next_50           |       1.10383  | ms     |             -0.47 | ⚪          |
| 2026-10-03 17:26:13 |      7848612 | tsfast_expanding_first_window        |       0.767469 | ms     |             -8.71 | 🟢          |
| 2026-10-03 17:26:13 |      7848612 | tsfast_expanding_avg_next_50         |       2.12699  | ms     |             -9.24 | 🟢          |
| 2026-10-03 17:26:13 |      7848612 | tsfast_static_sliding_first_window   |       1.12796  | ms     |             -3.63 | ⚪          |
| 2026-10-03 17:26:13 |      7848612 | tsfast_static_sliding_avg_next_50    |       1.0166   | ms     |             -8.72 | 🟢          |
| 2026-10-03 17:26:13 |      7848612 | tsfast_static_expanding_first_window |       0.992298 | ms     |            -12.19 | 🟢          |
| 2026-10-03 17:26:13 |      7848612 | tsfast_static_expanding_avg_next_50  |       4.01719  | ms     |             -2.65 | ⚪          |
| 2026-10-03 17:26:13 |      7848612 | tsfresh_sliding_first_window         |     122.807    | ms     |            -13.68 | 🟢          |
| 2026-10-03 17:26:13 |      7848612 | tsfresh_sliding_avg_next_50          |     124.791    | ms     |            -25.04 | 🟢          |
| 2026-10-03 17:26:13 |      7848612 | tsfresh_expanding_first_window       |     119.692    | ms     |             -5.99 | 🟢          |
| 2026-10-03 17:26:13 |      7848612 | tsfresh_expanding_avg_next_50        |     539.212    | ms     |             -0.23 | ⚪          |
| 2026-10-03 17:26:13 |      7848612 | tsfel_sliding_first_window           |     250.831    | ms     |             -3.16 | ⚪          |
| 2026-10-03 17:26:13 |      7848612 | tsfel_sliding_avg_next_50            |     222.916    | ms     |            -17.93 | 🟢          |
| 2026-10-03 17:26:13 |      7848612 | tsfel_expanding_first_window         |     220.77     | ms     |            -30.63 | 🟢          |
| 2026-10-03 17:26:13 |      7848612 | tsfel_expanding_avg_next_50          |     258.197    | ms     |            -10.03 | 🟢          |
| 2026-10-03 17:26:13 |      7848612 | tsfast_compatible_features           |      46        | count  |              0    | ⚪          |
| 2026-10-03 17:26:13 |      7848612 | tsfresh_compatible_features          |      40        | count  |              0    | ⚪          |
| 2026-10-03 17:26:13 |      7848612 | tsfel_compatible_features            |      17        | count  |              0    | ⚪          |
