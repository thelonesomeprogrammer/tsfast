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
| 2026-10-03 22:20:26 | bb24596      | tsfast_sliding_first_window          |       1.15156  | ms     |            -28.54 | 🟢          |
| 2026-10-03 22:20:26 | bb24596      | tsfast_sliding_avg_next_50           |       0.484982 | ms     |             10.22 | 🔴          |
| 2026-10-03 22:20:26 | bb24596      | tsfast_expanding_first_window        |       0.314713 | ms     |              0.61 | ⚪          |
| 2026-10-03 22:20:26 | bb24596      | tsfast_expanding_avg_next_50         |       1.05593  | ms     |              7.23 | 🔴          |
| 2026-10-03 22:20:26 | bb24596      | tsfast_static_sliding_first_window   |       0.452757 | ms     |            -19.05 | 🟢          |
| 2026-10-03 22:20:26 | bb24596      | tsfast_static_sliding_avg_next_50    |       0.451961 | ms     |             12.23 | 🔴          |
| 2026-10-03 22:20:26 | bb24596      | tsfast_static_expanding_first_window |       0.940561 | ms     |            141.58 | 🔴          |
| 2026-10-03 22:20:26 | bb24596      | tsfast_static_expanding_avg_next_50  |       1.76235  | ms     |             -2.14 | ⚪          |
| 2026-10-03 22:20:26 | bb24596      | tsfresh_sliding_first_window         |      67.2696   | ms     |              9.78 | 🔴          |
| 2026-10-03 22:20:26 | bb24596      | tsfresh_sliding_avg_next_50          |      56.582    | ms     |              2.05 | ⚪          |
| 2026-10-03 22:20:26 | bb24596      | tsfresh_expanding_first_window       |      54.7783   | ms     |              0.3  | ⚪          |
| 2026-10-03 22:20:26 | bb24596      | tsfresh_expanding_avg_next_50        |     270.009    | ms     |             -0.44 | ⚪          |
| 2026-10-03 22:20:26 | bb24596      | tsfel_sliding_first_window           |      84.367    | ms     |              0.81 | ⚪          |
| 2026-10-03 22:20:26 | bb24596      | tsfel_sliding_avg_next_50            |      82.2673   | ms     |              0.8  | ⚪          |
| 2026-10-03 22:20:26 | bb24596      | tsfel_expanding_first_window         |      82.5286   | ms     |              1.17 | ⚪          |
| 2026-10-03 22:20:26 | bb24596      | tsfel_expanding_avg_next_50          |      94.7859   | ms     |              0.63 | ⚪          |
| 2026-10-03 22:20:26 | bb24596      | tsfast_compatible_features           |      46        | count  |              0    | ⚪          |
| 2026-10-03 22:20:26 | bb24596      | tsfresh_compatible_features          |      40        | count  |              0    | ⚪          |
| 2026-10-03 22:20:26 | bb24596      | tsfel_compatible_features            |      17        | count  |              0    | ⚪          |
