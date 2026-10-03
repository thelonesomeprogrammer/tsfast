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
| 2026-10-03 12:20:06 | 3316ed4      | tsfast_sliding_first_window          |       4.24147  | ms     |              5.46 | 🔴          |
| 2026-10-03 12:20:06 | 3316ed4      | tsfast_sliding_avg_next_50           |       1.41592  | ms     |             29.29 | 🔴          |
| 2026-10-03 12:20:06 | 3316ed4      | tsfast_expanding_first_window        |       0.709534 | ms     |              9.09 | 🔴          |
| 2026-10-03 12:20:06 | 3316ed4      | tsfast_expanding_avg_next_50         |       2.47565  | ms     |             26.19 | 🔴          |
| 2026-10-03 12:20:06 | 3316ed4      | tsfast_static_sliding_first_window   |       1.13201  | ms     |             17.67 | 🔴          |
| 2026-10-03 12:20:06 | 3316ed4      | tsfast_static_sliding_avg_next_50    |       1.3775   | ms     |             42.6  | 🔴          |
| 2026-10-03 12:20:06 | 3316ed4      | tsfast_static_expanding_first_window |       0.922441 | ms     |             -0.92 | ⚪          |
| 2026-10-03 12:20:06 | 3316ed4      | tsfast_static_expanding_avg_next_50  |       4.30879  | ms     |             16.11 | 🔴          |
| 2026-10-03 12:20:06 | 3316ed4      | tsfresh_sliding_first_window         |     123.138    | ms     |              1.17 | ⚪          |
| 2026-10-03 12:20:06 | 3316ed4      | tsfresh_sliding_avg_next_50          |     119.755    | ms     |             -0.7  | ⚪          |
| 2026-10-03 12:20:06 | 3316ed4      | tsfresh_expanding_first_window       |     118.332    | ms     |              0.33 | ⚪          |
| 2026-10-03 12:20:06 | 3316ed4      | tsfresh_expanding_avg_next_50        |     534.93     | ms     |              6.83 | 🔴          |
| 2026-10-03 12:20:06 | 3316ed4      | tsfel_sliding_first_window           |     221.837    | ms     |              4.86 | ⚪          |
| 2026-10-03 12:20:06 | 3316ed4      | tsfel_sliding_avg_next_50            |     213.949    | ms     |             -1.24 | ⚪          |
| 2026-10-03 12:20:06 | 3316ed4      | tsfel_expanding_first_window         |     214.685    | ms     |              1.41 | ⚪          |
| 2026-10-03 12:20:06 | 3316ed4      | tsfel_expanding_avg_next_50          |     247.657    | ms     |             -2.25 | ⚪          |
| 2026-10-03 12:20:06 | 3316ed4      | tsfast_compatible_features           |      46        | count  |              0    | ⚪          |
| 2026-10-03 12:20:06 | 3316ed4      | tsfresh_compatible_features          |      40        | count  |              0    | ⚪          |
| 2026-10-03 12:20:06 | 3316ed4      | tsfel_compatible_features            |      17        | count  |              0    | ⚪          |
