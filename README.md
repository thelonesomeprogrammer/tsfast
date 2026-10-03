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
| 2026-10-03 15:22:00 | b733353      | tsfast_sliding_first_window          |       1.28245  | ms     |            -76.89 | 🟢          |
| 2026-10-03 15:22:00 | b733353      | tsfast_sliding_avg_next_50           |       0.442362 | ms     |            -65.93 | 🟢          |
| 2026-10-03 15:22:00 | b733353      | tsfast_expanding_first_window        |       0.275373 | ms     |            -22.22 | 🟢          |
| 2026-10-03 15:22:00 | b733353      | tsfast_expanding_avg_next_50         |       0.81881  | ms     |            -67.55 | 🟢          |
| 2026-10-03 15:22:00 | b733353      | tsfast_static_sliding_first_window   |       1.14131  | ms     |            126.87 | 🔴          |
| 2026-10-03 15:22:00 | b733353      | tsfast_static_sliding_avg_next_50    |       0.661674 | ms     |            -50.67 | 🟢          |
| 2026-10-03 15:22:00 | b733353      | tsfast_static_expanding_first_window |       0.350952 | ms     |            -71.88 | 🟢          |
| 2026-10-03 15:22:00 | b733353      | tsfast_static_expanding_avg_next_50  |       1.75674  | ms     |            -53.62 | 🟢          |
| 2026-10-03 15:22:00 | b733353      | tsfresh_sliding_first_window         |      67.4357   | ms     |              3.87 | ⚪          |
| 2026-10-03 15:22:00 | b733353      | tsfresh_sliding_avg_next_50          |      57.2886   | ms     |             -1.95 | ⚪          |
| 2026-10-03 15:22:00 | b733353      | tsfresh_expanding_first_window       |      55.0122   | ms     |             -5.73 | 🟢          |
| 2026-10-03 15:22:00 | b733353      | tsfresh_expanding_avg_next_50        |     271.9      | ms     |             -6.85 | 🟢          |
| 2026-10-03 15:22:00 | b733353      | tsfel_sliding_first_window           |      97.8215   | ms     |             -5.33 | 🟢          |
| 2026-10-03 15:22:00 | b733353      | tsfel_sliding_avg_next_50            |      86.0257   | ms     |             -4.07 | ⚪          |
| 2026-10-03 15:22:00 | b733353      | tsfel_expanding_first_window         |      81.9993   | ms     |            -11.22 | 🟢          |
| 2026-10-03 15:22:00 | b733353      | tsfel_expanding_avg_next_50          |      99.4241   | ms     |             -3.2  | ⚪          |
| 2026-10-03 15:22:00 | b733353      | tsfast_compatible_features           |      46        | count  |              0    | ⚪          |
| 2026-10-03 15:22:00 | b733353      | tsfresh_compatible_features          |      40        | count  |              0    | ⚪          |
| 2026-10-03 15:22:00 | b733353      | tsfel_compatible_features            |      17        | count  |              0    | ⚪          |
