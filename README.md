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
| 2026-10-03 21:08:59 |      7848612 | tsfast_sliding_first_window          |       3.53265  | ms     |            119.22 | 🔴          |
| 2026-10-03 21:08:59 |      7848612 | tsfast_sliding_avg_next_50           |       0.94696  | ms     |            115.21 | 🔴          |
| 2026-10-03 21:08:59 |      7848612 | tsfast_expanding_first_window        |       0.678778 | ms     |            117    | 🔴          |
| 2026-10-03 21:08:59 |      7848612 | tsfast_expanding_avg_next_50         |       2.92809  | ms     |            197.35 | 🔴          |
| 2026-10-03 21:08:59 |      7848612 | tsfast_static_sliding_first_window   |       1.54662  | ms     |            176.51 | 🔴          |
| 2026-10-03 21:08:59 |      7848612 | tsfast_static_sliding_avg_next_50    |       1.54195  | ms     |            282.88 | 🔴          |
| 2026-10-03 21:08:59 |      7848612 | tsfast_static_expanding_first_window |       1.64199  | ms     |            321.74 | 🔴          |
| 2026-10-03 21:08:59 |      7848612 | tsfast_static_expanding_avg_next_50  |       5.46781  | ms     |            203.62 | 🔴          |
| 2026-10-03 21:08:59 |      7848612 | tsfresh_sliding_first_window         |     245.471    | ms     |            300.6  | 🔴          |
| 2026-10-03 21:08:59 |      7848612 | tsfresh_sliding_avg_next_50          |     141.345    | ms     |            154.92 | 🔴          |
| 2026-10-03 21:08:59 |      7848612 | tsfresh_expanding_first_window       |     117.053    | ms     |            114.33 | 🔴          |
| 2026-10-03 21:08:59 |      7848612 | tsfresh_expanding_avg_next_50        |     496.004    | ms     |             82.9  | 🔴          |
| 2026-10-03 21:08:59 |      7848612 | tsfel_sliding_first_window           |     209.635    | ms     |            150.48 | 🔴          |
| 2026-10-03 21:08:59 |      7848612 | tsfel_sliding_avg_next_50            |     209.932    | ms     |            157.23 | 🔴          |
| 2026-10-03 21:08:59 |      7848612 | tsfel_expanding_first_window         |     207.689    | ms     |            154.61 | 🔴          |
| 2026-10-03 21:08:59 |      7848612 | tsfel_expanding_avg_next_50          |     241.545    | ms     |            156.43 | 🔴          |
| 2026-10-03 21:08:59 |      7848612 | tsfast_compatible_features           |      46        | count  |              0    | ⚪          |
| 2026-10-03 21:08:59 |      7848612 | tsfresh_compatible_features          |      40        | count  |              0    | ⚪          |
| 2026-10-03 21:08:59 |      7848612 | tsfel_compatible_features            |      17        | count  |              0    | ⚪          |
