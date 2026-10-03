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
| 2026-10-03 20:59:56 | 3e263a4      | tsfast_sliding_first_window          |        8.99053 | ms     |             -4.55 | ⚪          |
| 2026-10-03 20:59:56 | 3e263a4      | tsfast_sliding_avg_next_50           |        7.25702 | ms     |              6.59 | 🔴          |
| 2026-10-03 20:59:56 | 3e263a4      | tsfast_expanding_first_window        |        3.25584 | ms     |             10.65 | 🔴          |
| 2026-10-03 20:59:56 | 3e263a4      | tsfast_expanding_avg_next_50         |       13.5018  | ms     |             -0.42 | ⚪          |
| 2026-10-03 20:59:56 | 3e263a4      | tsfast_static_sliding_first_window   |        5.68128 | ms     |              2.57 | ⚪          |
| 2026-10-03 20:59:56 | 3e263a4      | tsfast_static_sliding_avg_next_50    |        5.52238 | ms     |              4.43 | ⚪          |
| 2026-10-03 20:59:56 | 3e263a4      | tsfast_static_expanding_first_window |        5.16486 | ms     |             -7    | 🟢          |
| 2026-10-03 20:59:56 | 3e263a4      | tsfast_static_expanding_avg_next_50  |       47.2111  | ms     |              2.22 | ⚪          |
| 2026-10-03 20:59:56 | 3e263a4      | tsfresh_sliding_first_window         |      125.126   | ms     |              3.5  | ⚪          |
| 2026-10-03 20:59:56 | 3e263a4      | tsfresh_sliding_avg_next_50          |      119.823   | ms     |             -1.26 | ⚪          |
| 2026-10-03 20:59:56 | 3e263a4      | tsfresh_expanding_first_window       |      118.99    | ms     |              0.75 | ⚪          |
| 2026-10-03 20:59:56 | 3e263a4      | tsfresh_expanding_avg_next_50        |      542.721   | ms     |              0.52 | ⚪          |
| 2026-10-03 20:59:56 | 3e263a4      | tsfel_sliding_first_window           |      211.327   | ms     |              1.55 | ⚪          |
| 2026-10-03 20:59:56 | 3e263a4      | tsfel_sliding_avg_next_50            |      210.993   | ms     |              0.25 | ⚪          |
| 2026-10-03 20:59:56 | 3e263a4      | tsfel_expanding_first_window         |      206.596   | ms     |             -6.36 | 🟢          |
| 2026-10-03 20:59:56 | 3e263a4      | tsfel_expanding_avg_next_50          |      242.833   | ms     |              0.53 | ⚪          |
| 2026-10-03 20:59:56 | 3e263a4      | tsfast_compatible_features           |       46       | count  |              0    | ⚪          |
| 2026-10-03 20:59:56 | 3e263a4      | tsfresh_compatible_features          |       40       | count  |              0    | ⚪          |
| 2026-10-03 20:59:56 | 3e263a4      | tsfel_compatible_features            |       17       | count  |              0    | ⚪          |
