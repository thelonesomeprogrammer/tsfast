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
| 2026-10-03 20:31:23 |      7848612 | tsfast_sliding_first_window          |       3.68094  | ms     |            -41.44 | 🟢          |
| 2026-10-03 20:31:23 |      7848612 | tsfast_sliding_avg_next_50           |       1.05653  | ms     |            -57.53 | 🟢          |
| 2026-10-03 20:31:23 |      7848612 | tsfast_expanding_first_window        |       0.685453 | ms     |            -39.89 | 🟢          |
| 2026-10-03 20:31:23 |      7848612 | tsfast_expanding_avg_next_50         |       2.57237  | ms     |            -21.1  | 🟢          |
| 2026-10-03 20:31:23 |      7848612 | tsfast_static_sliding_first_window   |       1.69802  | ms     |            -21.53 | 🟢          |
| 2026-10-03 20:31:23 |      7848612 | tsfast_static_sliding_avg_next_50    |       1.40267  | ms     |            -24.11 | 🟢          |
| 2026-10-03 20:31:23 |      7848612 | tsfast_static_expanding_first_window |       1.00231  | ms     |            -59.9  | 🟢          |
| 2026-10-03 20:31:23 |      7848612 | tsfast_static_expanding_avg_next_50  |       4.5336   | ms     |            -38.5  | 🟢          |
| 2026-10-03 20:31:23 |      7848612 | tsfresh_sliding_first_window         |     128.152    | ms     |            -52.9  | 🟢          |
| 2026-10-03 20:31:23 |      7848612 | tsfresh_sliding_avg_next_50          |     122.334    | ms     |            -11.38 | 🟢          |
| 2026-10-03 20:31:23 |      7848612 | tsfresh_expanding_first_window       |     123.403    | ms     |              3.32 | ⚪          |
| 2026-10-03 20:31:23 |      7848612 | tsfresh_expanding_avg_next_50        |     513.309    | ms     |             -0.74 | ⚪          |
| 2026-10-03 20:31:23 |      7848612 | tsfel_sliding_first_window           |     224.993    | ms     |              4.18 | ⚪          |
| 2026-10-03 20:31:23 |      7848612 | tsfel_sliding_avg_next_50            |     228.508    | ms     |              6.55 | 🔴          |
| 2026-10-03 20:31:23 |      7848612 | tsfel_expanding_first_window         |     223.402    | ms     |              6.75 | 🔴          |
| 2026-10-03 20:31:23 |      7848612 | tsfel_expanding_avg_next_50          |     266.671    | ms     |              5.75 | 🔴          |
| 2026-10-03 20:31:23 |      7848612 | tsfast_compatible_features           |      46        | count  |              0    | ⚪          |
| 2026-10-03 20:31:23 |      7848612 | tsfresh_compatible_features          |      40        | count  |              0    | ⚪          |
| 2026-10-03 20:31:23 |      7848612 | tsfel_compatible_features            |      17        | count  |              0    | ⚪          |
