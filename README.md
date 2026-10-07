# TSFast: Ultra-Fast Time Series Feature Extraction

TSFast is a high-performance time-series feature extraction library written in Rust with Python bindings. It is designed for extreme speed, efficient memory usage, and interoperability with Apache Arrow.

## Purpose: Deploying TSFresh and TSFEL Features

TSFast is meant for **deploying** features, not for finding them. Use
[tsfresh](https://github.com/blue-yonder/tsfresh) or
[TSFEL](https://github.com/fraunhoferportugal/tsfel) during research to explore
and select features. Then pass the selected feature names to TSFast to compute
the same values in production.

- **Same names, same values**: features use the tsfresh/TSFEL names and are
  tested against both libraries, so a model trained on their output gets the
  same inputs at inference time.
- **Built for production**: the Rust engines are orders of magnitude faster
  (see the benchmarks below). The sliding and expanding engines update
  incrementally as new data arrives, which suits streaming and low-latency
  serving.
- **Small footprint**: the runtime doesn't need tsfresh, TSFEL or their
  dependency trees.

TSFast implements every tsfresh and TSFEL feature calculator.

**Sampling assumption**: every series is assumed to be regularly sampled.
TSFast takes no timestamps, so sample `i` is at time `i / fs`:

- TSFEL spectral and time-domain features use `fs = 100` Hz unless the feature
  takes the sampling frequency as a parameter (e.g. `human_range_energy-100`).
- tsfresh's `linear_trend_timewise` takes the sampling period in seconds:
  `linear_trend_timewise-slope-60` matches tsfresh on a regular
  `DatetimeIndex` with one sample per minute. Irregularly sampled series are
  not supported; resample them to a regular grid first.

## Key Features

- **Blazing Fast**: Core engine implemented in Rust with SIMD (Portable SIMD) for maximum performance.
- **O(n) Expanding Windows**: Highly optimized algorithms for expanding window feature extraction (prefix statistics).
- **NumPy In, NumPy Out**: Takes a 2-D array with one series per row (float32 is read in place, no copy) and returns a float32 feature array; `feature_names` labels its columns.
- **Selective Execution**: Only computes the features you request, using a bitmask-based engine to skip unnecessary calculations.
- **Python-Friendly**: Simple API based on `Extractor` and `ExpandingExtractor` classes.

## Benchmarks

![Benchmark Trends](.jules/benchmark_trends.png)

## Latest Results
| Date                | CommitHash   | Benchmark_Name                       |   Metric_Value | Unit   |   Delta_From_Last | Direction   |
|:--------------------|:-------------|:-------------------------------------|---------------:|:-------|------------------:|:------------|
| 2026-10-05 12:11:10 | f405f4f      | tsfast_sliding_first_window          |        2.72274 | ms     |            -14.01 | 🟢          |
| 2026-10-05 12:11:10 | f405f4f      | tsfast_sliding_avg_next_50           |        1.16132 | ms     |              2.05 | ⚪          |
| 2026-10-05 12:11:10 | f405f4f      | tsfast_expanding_first_window        |        1.10626 | ms     |              4.74 | ⚪          |
| 2026-10-05 12:11:10 | f405f4f      | tsfast_expanding_avg_next_50         |        3.87877 | ms     |              2.01 | ⚪          |
| 2026-10-05 12:11:10 | f405f4f      | tsfast_static_sliding_first_window   |        1.25408 | ms     |              6.67 | 🔴          |
| 2026-10-05 12:11:10 | f405f4f      | tsfast_static_sliding_avg_next_50    |        1.3604  | ms     |             10.98 | 🔴          |
| 2026-10-05 12:11:10 | f405f4f      | tsfast_static_expanding_first_window |        1.17159 | ms     |             -2.31 | ⚪          |
| 2026-10-05 12:11:10 | f405f4f      | tsfast_static_expanding_avg_next_50  |        5.09623 | ms     |              7.41 | 🔴          |
| 2026-10-05 12:11:10 | f405f4f      | tsfresh_sliding_first_window         |      137.904   | ms     |              5.7  | 🔴          |
| 2026-10-05 12:11:10 | f405f4f      | tsfresh_sliding_avg_next_50          |      130.325   | ms     |            -16.18 | 🟢          |
| 2026-10-05 12:11:10 | f405f4f      | tsfresh_expanding_first_window       |      125.321   | ms     |             -1.56 | ⚪          |
| 2026-10-05 12:11:10 | f405f4f      | tsfresh_expanding_avg_next_50        |      505.666   | ms     |              2    | ⚪          |
| 2026-10-05 12:11:10 | f405f4f      | tsfel_sliding_first_window           |      245.412   | ms     |              1.33 | ⚪          |
| 2026-10-05 12:11:10 | f405f4f      | tsfel_sliding_avg_next_50            |      263.305   | ms     |             10.62 | 🔴          |
| 2026-10-05 12:11:10 | f405f4f      | tsfel_expanding_first_window         |      424.004   | ms     |             79.11 | 🔴          |
| 2026-10-05 12:11:10 | f405f4f      | tsfel_expanding_avg_next_50          |      284.2     | ms     |              5.83 | 🔴          |
| 2026-10-05 12:11:10 | f405f4f      | tsfast_compatible_features           |       46       | count  |              0    | ⚪          |
| 2026-10-05 12:11:10 | f405f4f      | tsfresh_compatible_features          |       40       | count  |              0    | ⚪          |
| 2026-10-05 12:11:10 | f405f4f      | tsfel_compatible_features            |       17       | count  |              0    | ⚪          |
