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
| 2026-10-03 16:14:05 |      7848612 | tsfast_sliding_first_window          |       3.76844  | ms     |            133.85 | 🔴          |
| 2026-10-03 16:14:05 |      7848612 | tsfast_sliding_avg_next_50           |       1.09058  | ms     |            147.85 | 🔴          |
| 2026-10-03 16:14:05 |      7848612 | tsfast_expanding_first_window        |       0.976562 | ms     |            212.2  | 🔴          |
| 2026-10-03 16:14:05 |      7848612 | tsfast_expanding_avg_next_50         |       1.96109  | ms     |             99.15 | 🔴          |
| 2026-10-03 16:14:05 |      7848612 | tsfast_static_sliding_first_window   |       1.10149  | ms     |             96.93 | 🔴          |
| 2026-10-03 16:14:05 |      7848612 | tsfast_static_sliding_avg_next_50    |       0.985045 | ms     |            144.6  | 🔴          |
| 2026-10-03 16:14:05 |      7848612 | tsfast_static_expanding_first_window |       0.969648 | ms     |            149.05 | 🔴          |
| 2026-10-03 16:14:05 |      7848612 | tsfast_static_expanding_avg_next_50  |       3.49561  | ms     |             94.1  | 🔴          |
| 2026-10-03 16:14:05 |      7848612 | tsfresh_sliding_first_window         |     123.044    | ms     |            100.81 | 🔴          |
| 2026-10-03 16:14:05 |      7848612 | tsfresh_sliding_avg_next_50          |     121.517    | ms     |            119.16 | 🔴          |
| 2026-10-03 16:14:05 |      7848612 | tsfresh_expanding_first_window       |     122.18     | ms     |            123.72 | 🔴          |
| 2026-10-03 16:14:05 |      7848612 | tsfresh_expanding_avg_next_50        |     505.197    | ms     |             86.29 | 🔴          |
| 2026-10-03 16:14:05 |      7848612 | tsfel_sliding_first_window           |     215.473    | ms     |            157.46 | 🔴          |
| 2026-10-03 16:14:05 |      7848612 | tsfel_sliding_avg_next_50            |     216.415    | ms     |            165.17 | 🔴          |
| 2026-10-03 16:14:05 |      7848612 | tsfel_expanding_first_window         |     214.971    | ms     |            163.53 | 🔴          |
| 2026-10-03 16:14:05 |      7848612 | tsfel_expanding_avg_next_50          |     246.866    | ms     |            162.08 | 🔴          |
| 2026-10-03 16:14:05 |      7848612 | tsfast_compatible_features           |      46        | count  |              0    | ⚪          |
| 2026-10-03 16:14:05 |      7848612 | tsfresh_compatible_features          |      40        | count  |              0    | ⚪          |
| 2026-10-03 16:14:05 |      7848612 | tsfel_compatible_features            |      17        | count  |              0    | ⚪          |
