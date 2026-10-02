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
| Date                | CommitHash                               | Benchmark_Name                                 |   Metric_Value | Unit    |   Delta_From_Last |
|:--------------------|:-----------------------------------------|:-----------------------------------------------|---------------:|:--------|------------------:|
| 2026-10-02 11:40:51 | 63fbb8e65b3cc5616d127913d5cbeb7703ca6c39 | test_bench_compare_tsfast                      |    0.000468616 | seconds |           8.92091 |
| 2026-10-02 11:40:51 | 63fbb8e65b3cc5616d127913d5cbeb7703ca6c39 | test_bench_compare_tsfresh                     |    0.0422207   | seconds |           2.42432 |
| 2026-10-02 11:40:51 | 63fbb8e65b3cc5616d127913d5cbeb7703ca6c39 | test_bench_compare_tsfel                       |    0.245524    | seconds |           8.20595 |
| 2026-10-02 11:40:51 | 63fbb8e65b3cc5616d127913d5cbeb7703ca6c39 | test_bench_expanding_initial_batch             |    0.000177956 | seconds |           4.52816 |
| 2026-10-02 11:40:51 | 63fbb8e65b3cc5616d127913d5cbeb7703ca6c39 | test_bench_expanding_incremental_update_10cols |    0.00299317  | seconds |         -27.3049  |
| 2026-10-02 11:40:51 | 63fbb8e65b3cc5616d127913d5cbeb7703ca6c39 | test_bench_expanding_full_stream_20_chunks     |    0.00789742  | seconds |           5.65518 |
| 2026-10-02 11:40:51 | 63fbb8e65b3cc5616d127913d5cbeb7703ca6c39 | test_bench_select_features_supervised          |    0.0544352   | seconds |           2.04158 |
| 2026-10-02 11:40:51 | 63fbb8e65b3cc5616d127913d5cbeb7703ca6c39 | test_bench_select_features_unsupervised        |    0.00054023  | seconds |           2.03453 |
| 2026-10-02 11:40:51 | 63fbb8e65b3cc5616d127913d5cbeb7703ca6c39 | test_bench_sliding_window200_stride50          |    0.00379654  | seconds |           3.64563 |
| 2026-10-02 11:40:51 | 63fbb8e65b3cc5616d127913d5cbeb7703ca6c39 | test_bench_sliding_single_chunk_stride50       |    0.000185085 | seconds |           4.9286  |
| 2026-10-02 11:40:51 | 63fbb8e65b3cc5616d127913d5cbeb7703ca6c39 | test_bench_static_single_series_basic          |    0.000174973 | seconds |           6.64218 |
| 2026-10-02 11:40:51 | 63fbb8e65b3cc5616d127913d5cbeb7703ca6c39 | test_bench_static_single_series_advanced       |    0.0002308   | seconds |           3.90216 |
| 2026-10-02 11:40:51 | 63fbb8e65b3cc5616d127913d5cbeb7703ca6c39 | test_bench_static_single_series_30_features    |    0.000318509 | seconds |           1.31772 |
| 2026-10-02 11:40:51 | 63fbb8e65b3cc5616d127913d5cbeb7703ca6c39 | test_bench_static_batched_100_series           |    0.00179463  | seconds |           0.42216 |
| 2026-10-02 11:40:51 | 63fbb8e65b3cc5616d127913d5cbeb7703ca6c39 | test_bench_static_batched_1000_series          |    0.0143248   | seconds |           4.69625 |
