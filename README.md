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
| Date                | CommitHash   | Benchmark_Name                       |   Metric_Value | Unit   |   Delta_From_Last |
|:--------------------|:-------------|:-------------------------------------|---------------:|:-------|------------------:|
| 2026-10-02 14:43:55 | dc89d37      | tsfast_sliding_first_window          |       1.58691  | ms     |                 0 |
| 2026-10-02 14:43:55 | dc89d37      | tsfast_sliding_avg_next_50           |       0.434408 | ms     |                 0 |
| 2026-10-02 14:43:55 | dc89d37      | tsfast_expanding_first_window        |       0.333309 | ms     |                 0 |
| 2026-10-02 14:43:55 | dc89d37      | tsfast_expanding_avg_next_50         |       1.84072  | ms     |                 0 |
| 2026-10-02 14:43:55 | dc89d37      | tsfast_static_sliding_first_window   |       0.328541 | ms     |                 0 |
| 2026-10-02 14:43:55 | dc89d37      | tsfast_static_sliding_avg_next_50    |       0.44282  | ms     |                 0 |
| 2026-10-02 14:43:55 | dc89d37      | tsfast_static_expanding_first_window |       0.397682 | ms     |                 0 |
| 2026-10-02 14:43:55 | dc89d37      | tsfast_static_expanding_avg_next_50  |       2.06037  | ms     |                 0 |
| 2026-10-02 14:43:55 | dc89d37      | tsfresh_sliding_first_window         |      62.5169   | ms     |                 0 |
| 2026-10-02 14:43:55 | dc89d37      | tsfresh_sliding_avg_next_50          |      56.3064   | ms     |                 0 |
| 2026-10-02 14:43:55 | dc89d37      | tsfresh_expanding_first_window       |      55.4194   | ms     |                 0 |
| 2026-10-02 14:43:55 | dc89d37      | tsfresh_expanding_avg_next_50        |     274.549    | ms     |                 0 |
| 2026-10-02 14:43:55 | dc89d37      | tsfel_sliding_first_window           |      84.6529   | ms     |                 0 |
| 2026-10-02 14:43:55 | dc89d37      | tsfel_sliding_avg_next_50            |      82.126    | ms     |                 0 |
| 2026-10-02 14:43:55 | dc89d37      | tsfel_expanding_first_window         |      81.9786   | ms     |                 0 |
| 2026-10-02 14:43:55 | dc89d37      | tsfel_expanding_avg_next_50          |      95.0517   | ms     |                 0 |
| 2026-10-02 14:43:55 | dc89d37      | tsfast_compatible_features           |      46        | count  |                 0 |
| 2026-10-02 14:43:55 | dc89d37      | tsfresh_compatible_features          |      40        | count  |                 0 |
| 2026-10-02 14:43:55 | dc89d37      | tsfel_compatible_features            |      17        | count  |                 0 |
