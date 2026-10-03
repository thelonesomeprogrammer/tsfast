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
| 2026-10-03 20:48:51 | 144d4a8      | tsfast_sliding_first_window          |       4.07815  | ms     |              1.56 | ⚪          |
| 2026-10-03 20:48:51 | 144d4a8      | tsfast_sliding_avg_next_50           |       1.0496   | ms     |             -8.52 | 🟢          |
| 2026-10-03 20:48:51 | 144d4a8      | tsfast_expanding_first_window        |       0.922203 | ms     |             39.99 | 🔴          |
| 2026-10-03 20:48:51 | 144d4a8      | tsfast_expanding_avg_next_50         |       2.10071  | ms     |              6.9  | 🔴          |
| 2026-10-03 20:48:51 | 144d4a8      | tsfast_static_sliding_first_window   |       1.87135  | ms     |             82.79 | 🔴          |
| 2026-10-03 20:48:51 | 144d4a8      | tsfast_static_sliding_avg_next_50    |       1.04787  | ms     |              3.49 | ⚪          |
| 2026-10-03 20:48:51 | 144d4a8      | tsfast_static_expanding_first_window |       1.30916  | ms     |             35.15 | 🔴          |
| 2026-10-03 20:48:51 | 144d4a8      | tsfast_static_expanding_avg_next_50  |       3.70887  | ms     |             -4.8  | ⚪          |
| 2026-10-03 20:48:51 | 144d4a8      | tsfresh_sliding_first_window         |     127.96     | ms     |             -1.68 | ⚪          |
| 2026-10-03 20:48:51 | 144d4a8      | tsfresh_sliding_avg_next_50          |     133.766    | ms     |              0.14 | ⚪          |
| 2026-10-03 20:48:51 | 144d4a8      | tsfresh_expanding_first_window       |     193.148    | ms     |             39.25 | 🔴          |
| 2026-10-03 20:48:51 | 144d4a8      | tsfresh_expanding_avg_next_50        |     520.329    | ms     |             -5.29 | 🟢          |
| 2026-10-03 20:48:51 | 144d4a8      | tsfel_sliding_first_window           |     231.29     | ms     |              1.56 | ⚪          |
| 2026-10-03 20:48:51 | 144d4a8      | tsfel_sliding_avg_next_50            |     235.607    | ms     |            -10.26 | 🟢          |
| 2026-10-03 20:48:51 | 144d4a8      | tsfel_expanding_first_window         |     221.881    | ms     |             -3.09 | ⚪          |
| 2026-10-03 20:48:51 | 144d4a8      | tsfel_expanding_avg_next_50          |     257.317    | ms     |             -6.3  | 🟢          |
| 2026-10-03 20:48:51 | 144d4a8      | tsfast_compatible_features           |      46        | count  |              0    | ⚪          |
| 2026-10-03 20:48:51 | 144d4a8      | tsfresh_compatible_features          |      40        | count  |              0    | ⚪          |
| 2026-10-03 20:48:51 | 144d4a8      | tsfel_compatible_features            |      17        | count  |              0    | ⚪          |
