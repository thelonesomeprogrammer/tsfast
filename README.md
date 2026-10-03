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
| 2026-10-03 22:47:01 | c1208e9      | tsfast_sliding_first_window          |       1.29819  | ms     |            -19.44 | 🟢          |
| 2026-10-03 22:47:01 | c1208e9      | tsfast_sliding_avg_next_50           |       0.478539 | ms     |              8.75 | 🔴          |
| 2026-10-03 22:47:01 | c1208e9      | tsfast_expanding_first_window        |       0.303984 | ms     |             -2.82 | ⚪          |
| 2026-10-03 22:47:01 | c1208e9      | tsfast_expanding_avg_next_50         |       0.956411 | ms     |             -2.88 | ⚪          |
| 2026-10-03 22:47:01 | c1208e9      | tsfast_static_sliding_first_window   |       0.392675 | ms     |            -29.8  | 🟢          |
| 2026-10-03 22:47:01 | c1208e9      | tsfast_static_sliding_avg_next_50    |       0.38919  | ms     |             -3.36 | ⚪          |
| 2026-10-03 22:47:01 | c1208e9      | tsfast_static_expanding_first_window |       0.481844 | ms     |             23.76 | 🔴          |
| 2026-10-03 22:47:01 | c1208e9      | tsfast_static_expanding_avg_next_50  |       1.86884  | ms     |              3.77 | ⚪          |
| 2026-10-03 22:47:01 | c1208e9      | tsfresh_sliding_first_window         |      61.1355   | ms     |             -0.23 | ⚪          |
| 2026-10-03 22:47:01 | c1208e9      | tsfresh_sliding_avg_next_50          |      54.9161   | ms     |             -0.96 | ⚪          |
| 2026-10-03 22:47:01 | c1208e9      | tsfresh_expanding_first_window       |      54.3087   | ms     |             -0.56 | ⚪          |
| 2026-10-03 22:47:01 | c1208e9      | tsfresh_expanding_avg_next_50        |     268.138    | ms     |             -1.13 | ⚪          |
| 2026-10-03 22:47:01 | c1208e9      | tsfel_sliding_first_window           |      83.8373   | ms     |              0.17 | ⚪          |
| 2026-10-03 22:47:01 | c1208e9      | tsfel_sliding_avg_next_50            |      81.7383   | ms     |              0.15 | ⚪          |
| 2026-10-03 22:47:01 | c1208e9      | tsfel_expanding_first_window         |      81.43     | ms     |             -0.18 | ⚪          |
| 2026-10-03 22:47:01 | c1208e9      | tsfel_expanding_avg_next_50          |      94.0338   | ms     |             -0.17 | ⚪          |
| 2026-10-03 22:47:01 | c1208e9      | tsfast_compatible_features           |      46        | count  |              0    | ⚪          |
| 2026-10-03 22:47:01 | c1208e9      | tsfresh_compatible_features          |      40        | count  |              0    | ⚪          |
| 2026-10-03 22:47:01 | c1208e9      | tsfel_compatible_features            |      17        | count  |              0    | ⚪          |
