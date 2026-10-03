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
| 2026-10-03 23:48:32 | 89ae2ab      | tsfast_sliding_first_window          |       1.28698  | ms     |            -20.14 | 🟢          |
| 2026-10-03 23:48:32 | 89ae2ab      | tsfast_sliding_avg_next_50           |       0.456114 | ms     |              3.66 | ⚪          |
| 2026-10-03 23:48:32 | 89ae2ab      | tsfast_expanding_first_window        |       0.305653 | ms     |             -2.29 | ⚪          |
| 2026-10-03 23:48:32 | 89ae2ab      | tsfast_expanding_avg_next_50         |       0.905447 | ms     |             -8.05 | 🟢          |
| 2026-10-03 23:48:32 | 89ae2ab      | tsfast_static_sliding_first_window   |       0.406027 | ms     |            -27.41 | 🟢          |
| 2026-10-03 23:48:32 | 89ae2ab      | tsfast_static_sliding_avg_next_50    |       0.400014 | ms     |             -0.67 | ⚪          |
| 2026-10-03 23:48:32 | 89ae2ab      | tsfast_static_expanding_first_window |       0.506401 | ms     |             30.07 | 🔴          |
| 2026-10-03 23:48:32 | 89ae2ab      | tsfast_static_expanding_avg_next_50  |       1.74778  | ms     |             -2.95 | ⚪          |
| 2026-10-03 23:48:32 | 89ae2ab      | tsfresh_sliding_first_window         |      61.7263   | ms     |              0.74 | ⚪          |
| 2026-10-03 23:48:32 | 89ae2ab      | tsfresh_sliding_avg_next_50          |      56.2812   | ms     |              1.5  | ⚪          |
| 2026-10-03 23:48:32 | 89ae2ab      | tsfresh_expanding_first_window       |      54.7371   | ms     |              0.23 | ⚪          |
| 2026-10-03 23:48:32 | 89ae2ab      | tsfresh_expanding_avg_next_50        |     270.707    | ms     |             -0.18 | ⚪          |
| 2026-10-03 23:48:32 | 89ae2ab      | tsfel_sliding_first_window           |      84.4741   | ms     |              0.93 | ⚪          |
| 2026-10-03 23:48:32 | 89ae2ab      | tsfel_sliding_avg_next_50            |      81.9598   | ms     |              0.42 | ⚪          |
| 2026-10-03 23:48:32 | 89ae2ab      | tsfel_expanding_first_window         |      81.9297   | ms     |              0.44 | ⚪          |
| 2026-10-03 23:48:32 | 89ae2ab      | tsfel_expanding_avg_next_50          |      94.9108   | ms     |              0.76 | ⚪          |
| 2026-10-03 23:48:32 | 89ae2ab      | tsfast_compatible_features           |      46        | count  |              0    | ⚪          |
| 2026-10-03 23:48:32 | 89ae2ab      | tsfresh_compatible_features          |      40        | count  |              0    | ⚪          |
| 2026-10-03 23:48:32 | 89ae2ab      | tsfel_compatible_features            |      17        | count  |              0    | ⚪          |
