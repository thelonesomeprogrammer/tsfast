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
| 2026-10-02 17:31:39 | 998de26      | tsfast_sliding_first_window          |       3.54266  | ms     |            123.24 |
| 2026-10-02 17:31:39 | 998de26      | tsfast_sliding_avg_next_50           |       0.93616  | ms     |            115.5  |
| 2026-10-02 17:31:39 | 998de26      | tsfast_expanding_first_window        |       0.84877  | ms     |            154.65 |
| 2026-10-02 17:31:39 | 998de26      | tsfast_expanding_avg_next_50         |       3.96284  | ms     |            115.29 |
| 2026-10-02 17:31:39 | 998de26      | tsfast_static_sliding_first_window   |       0.96488  | ms     |            193.69 |
| 2026-10-02 17:31:39 | 998de26      | tsfast_static_sliding_avg_next_50    |       0.927234 | ms     |            109.39 |
| 2026-10-02 17:31:39 | 998de26      | tsfast_static_expanding_first_window |       0.854969 | ms     |            114.99 |
| 2026-10-02 17:31:39 | 998de26      | tsfast_static_expanding_avg_next_50  |       3.95062  | ms     |             91.74 |
| 2026-10-02 17:31:39 | 998de26      | tsfresh_sliding_first_window         |     127.333    | ms     |            103.68 |
| 2026-10-02 17:31:39 | 998de26      | tsfresh_sliding_avg_next_50          |     119.739    | ms     |            112.66 |
| 2026-10-02 17:31:39 | 998de26      | tsfresh_expanding_first_window       |     118.577    | ms     |            113.96 |
| 2026-10-02 17:31:39 | 998de26      | tsfresh_expanding_avg_next_50        |     507.048    | ms     |             84.68 |
| 2026-10-02 17:31:39 | 998de26      | tsfel_sliding_first_window           |     210.948    | ms     |            149.19 |
| 2026-10-02 17:31:39 | 998de26      | tsfel_sliding_avg_next_50            |     239.465    | ms     |            191.58 |
| 2026-10-02 17:31:39 | 998de26      | tsfel_expanding_first_window         |     212.967    | ms     |            159.78 |
| 2026-10-02 17:31:39 | 998de26      | tsfel_expanding_avg_next_50          |     244.491    | ms     |            157.22 |
| 2026-10-02 17:31:39 | 998de26      | tsfast_compatible_features           |      46        | count  |              0    |
| 2026-10-02 17:31:39 | 998de26      | tsfresh_compatible_features          |      40        | count  |              0    |
| 2026-10-02 17:31:39 | 998de26      | tsfel_compatible_features            |      17        | count  |              0    |
