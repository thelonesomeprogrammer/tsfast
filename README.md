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
| 2026-10-02 20:28:42 | 6ae4839      | tsfast_sliding_first_window          |       3.66831  | ms     |              3.55 | ⚪          |
| 2026-10-02 20:28:42 | 6ae4839      | tsfast_sliding_avg_next_50           |       0.921307 | ms     |             -1.59 | ⚪          |
| 2026-10-02 20:28:42 | 6ae4839      | tsfast_expanding_first_window        |       0.734568 | ms     |            -13.46 | 🟢          |
| 2026-10-02 20:28:42 | 6ae4839      | tsfast_expanding_avg_next_50         |       3.11921  | ms     |            -21.29 | 🟢          |
| 2026-10-02 20:28:42 | 6ae4839      | tsfast_static_sliding_first_window   |       0.813723 | ms     |            -15.67 | 🟢          |
| 2026-10-02 20:28:42 | 6ae4839      | tsfast_static_sliding_avg_next_50    |       0.988622 | ms     |              6.62 | 🔴          |
| 2026-10-02 20:28:42 | 6ae4839      | tsfast_static_expanding_first_window |       0.899076 | ms     |              5.16 | 🔴          |
| 2026-10-02 20:28:42 | 6ae4839      | tsfast_static_expanding_avg_next_50  |       4.05128  | ms     |              2.55 | ⚪          |
| 2026-10-02 20:28:42 | 6ae4839      | tsfresh_sliding_first_window         |     113.487    | ms     |            -10.87 | 🟢          |
| 2026-10-02 20:28:42 | 6ae4839      | tsfresh_sliding_avg_next_50          |     108.001    | ms     |             -9.8  | 🟢          |
| 2026-10-02 20:28:42 | 6ae4839      | tsfresh_expanding_first_window       |     107.482    | ms     |             -9.36 | 🟢          |
| 2026-10-02 20:28:42 | 6ae4839      | tsfresh_expanding_avg_next_50        |     439.075    | ms     |            -13.41 | 🟢          |
| 2026-10-02 20:28:42 | 6ae4839      | tsfel_sliding_first_window           |     187.951    | ms     |            -10.9  | 🟢          |
| 2026-10-02 20:28:42 | 6ae4839      | tsfel_sliding_avg_next_50            |     190.503    | ms     |            -20.45 | 🟢          |
| 2026-10-02 20:28:42 | 6ae4839      | tsfel_expanding_first_window         |     184.561    | ms     |            -13.34 | 🟢          |
| 2026-10-02 20:28:42 | 6ae4839      | tsfel_expanding_avg_next_50          |     216.524    | ms     |            -11.44 | 🟢          |
| 2026-10-02 20:28:42 | 6ae4839      | tsfast_compatible_features           |      46        | count  |              0    | ⚪          |
| 2026-10-02 20:28:42 | 6ae4839      | tsfresh_compatible_features          |      40        | count  |              0    | ⚪          |
| 2026-10-02 20:28:42 | 6ae4839      | tsfel_compatible_features            |      17        | count  |              0    | ⚪          |
