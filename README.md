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
| 2026-10-03 14:02:30 | c224b58      | tsfast_sliding_first_window          |       1.61147  | ms     |            -70.96 | 🟢          |
| 2026-10-03 14:02:30 | c224b58      | tsfast_sliding_avg_next_50           |       0.440021 | ms     |            -66.11 | 🟢          |
| 2026-10-03 14:02:30 | c224b58      | tsfast_expanding_first_window        |       0.312805 | ms     |            -11.65 | 🟢          |
| 2026-10-03 14:02:30 | c224b58      | tsfast_expanding_avg_next_50         |       0.984726 | ms     |            -60.97 | 🟢          |
| 2026-10-03 14:02:30 | c224b58      | tsfast_static_sliding_first_window   |       0.55933  | ms     |             11.18 | 🔴          |
| 2026-10-03 14:02:30 | c224b58      | tsfast_static_sliding_avg_next_50    |       0.402722 | ms     |            -69.97 | 🟢          |
| 2026-10-03 14:02:30 | c224b58      | tsfast_static_expanding_first_window |       0.389338 | ms     |            -68.81 | 🟢          |
| 2026-10-03 14:02:30 | c224b58      | tsfast_static_expanding_avg_next_50  |       1.80089  | ms     |            -52.46 | 🟢          |
| 2026-10-03 14:02:30 | c224b58      | tsfresh_sliding_first_window         |      61.2752   | ms     |             -5.62 | 🟢          |
| 2026-10-03 14:02:30 | c224b58      | tsfresh_sliding_avg_next_50          |      55.4476   | ms     |             -5.1  | 🟢          |
| 2026-10-03 14:02:30 | c224b58      | tsfresh_expanding_first_window       |      54.6131   | ms     |             -6.41 | 🟢          |
| 2026-10-03 14:02:30 | c224b58      | tsfresh_expanding_avg_next_50        |     271.194    | ms     |             -7.09 | 🟢          |
| 2026-10-03 14:02:30 | c224b58      | tsfel_sliding_first_window           |      83.693    | ms     |            -19.01 | 🟢          |
| 2026-10-03 14:02:30 | c224b58      | tsfel_sliding_avg_next_50            |      81.613    | ms     |             -8.99 | 🟢          |
| 2026-10-03 14:02:30 | c224b58      | tsfel_expanding_first_window         |      81.573    | ms     |            -11.68 | 🟢          |
| 2026-10-03 14:02:30 | c224b58      | tsfel_expanding_avg_next_50          |      94.1946   | ms     |             -8.29 | 🟢          |
| 2026-10-03 14:02:30 | c224b58      | tsfast_compatible_features           |      46        | count  |              0    | ⚪          |
| 2026-10-03 14:02:30 | c224b58      | tsfresh_compatible_features          |      40        | count  |              0    | ⚪          |
| 2026-10-03 14:02:30 | c224b58      | tsfel_compatible_features            |      17        | count  |              0    | ⚪          |
