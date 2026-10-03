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
| 2026-10-03 19:49:41 | 3a7d582      | tsfast_sliding_first_window          |       1.32823  | ms     |            -64.75 | 🟢          |
| 2026-10-03 19:49:41 | 3a7d582      | tsfast_sliding_avg_next_50           |       0.497794 | ms     |            -54.36 | 🟢          |
| 2026-10-03 19:49:41 | 3a7d582      | tsfast_expanding_first_window        |       0.352621 | ms     |            -63.89 | 🟢          |
| 2026-10-03 19:49:41 | 3a7d582      | tsfast_expanding_avg_next_50         |       1.00093  | ms     |            -48.96 | 🟢          |
| 2026-10-03 19:49:41 | 3a7d582      | tsfast_static_sliding_first_window   |       0.455618 | ms     |            -58.64 | 🟢          |
| 2026-10-03 19:49:41 | 3a7d582      | tsfast_static_sliding_avg_next_50    |       0.411954 | ms     |            -58.18 | 🟢          |
| 2026-10-03 19:49:41 | 3a7d582      | tsfast_static_expanding_first_window |       0.391483 | ms     |            -59.63 | 🟢          |
| 2026-10-03 19:49:41 | 3a7d582      | tsfast_static_expanding_avg_next_50  |       1.85971  | ms     |            -46.8  | 🟢          |
| 2026-10-03 19:49:41 | 3a7d582      | tsfresh_sliding_first_window         |      61.3263   | ms     |            -50.16 | 🟢          |
| 2026-10-03 19:49:41 | 3a7d582      | tsfresh_sliding_avg_next_50          |      55.1082   | ms     |            -54.65 | 🟢          |
| 2026-10-03 19:49:41 | 3a7d582      | tsfresh_expanding_first_window       |      54.4949   | ms     |            -55.4  | 🟢          |
| 2026-10-03 19:49:41 | 3a7d582      | tsfresh_expanding_avg_next_50        |     269.972    | ms     |            -46.56 | 🟢          |
| 2026-10-03 19:49:41 | 3a7d582      | tsfel_sliding_first_window           |      84.9497   | ms     |            -60.58 | 🟢          |
| 2026-10-03 19:49:41 | 3a7d582      | tsfel_sliding_avg_next_50            |      82.7138   | ms     |            -61.78 | 🟢          |
| 2026-10-03 19:49:41 | 3a7d582      | tsfel_expanding_first_window         |      82.8414   | ms     |            -61.46 | 🟢          |
| 2026-10-03 19:49:41 | 3a7d582      | tsfel_expanding_avg_next_50          |      95.366    | ms     |            -61.37 | 🟢          |
| 2026-10-03 19:49:41 | 3a7d582      | tsfast_compatible_features           |      46        | count  |              0    | ⚪          |
| 2026-10-03 19:49:41 | 3a7d582      | tsfresh_compatible_features          |      40        | count  |              0    | ⚪          |
| 2026-10-03 19:49:41 | 3a7d582      | tsfel_compatible_features            |      17        | count  |              0    | ⚪          |
