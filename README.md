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
| 2026-10-02 10:05:33 | bb2d1514ff1982fa9236feb2148749bf386445d5 | test_bench_compare_tsfast                      |    0.000430235 | seconds |                 0 |
| 2026-10-02 10:05:33 | bb2d1514ff1982fa9236feb2148749bf386445d5 | test_bench_compare_tsfresh                     |    0.0412214   | seconds |                 0 |
| 2026-10-02 10:05:33 | bb2d1514ff1982fa9236feb2148749bf386445d5 | test_bench_compare_tsfel                       |    0.226905    | seconds |                 0 |
| 2026-10-02 10:05:33 | bb2d1514ff1982fa9236feb2148749bf386445d5 | test_bench_expanding_initial_batch             |    0.000170247 | seconds |                 0 |
| 2026-10-02 10:05:33 | bb2d1514ff1982fa9236feb2148749bf386445d5 | test_bench_expanding_incremental_update_10cols |    0.00411744  | seconds |                 0 |
| 2026-10-02 10:05:33 | bb2d1514ff1982fa9236feb2148749bf386445d5 | test_bench_expanding_full_stream_20_chunks     |    0.00747471  | seconds |                 0 |
| 2026-10-02 10:05:33 | bb2d1514ff1982fa9236feb2148749bf386445d5 | test_bench_select_features_supervised          |    0.0533461   | seconds |                 0 |
| 2026-10-02 10:05:33 | bb2d1514ff1982fa9236feb2148749bf386445d5 | test_bench_select_features_unsupervised        |    0.000529458 | seconds |                 0 |
| 2026-10-02 10:05:33 | bb2d1514ff1982fa9236feb2148749bf386445d5 | test_bench_sliding_window200_stride50          |    0.003663    | seconds |                 0 |
| 2026-10-02 10:05:33 | bb2d1514ff1982fa9236feb2148749bf386445d5 | test_bench_sliding_single_chunk_stride50       |    0.000176392 | seconds |                 0 |
| 2026-10-02 10:05:33 | bb2d1514ff1982fa9236feb2148749bf386445d5 | test_bench_static_single_series_basic          |    0.000164075 | seconds |                 0 |
| 2026-10-02 10:05:33 | bb2d1514ff1982fa9236feb2148749bf386445d5 | test_bench_static_single_series_advanced       |    0.000222132 | seconds |                 0 |
| 2026-10-02 10:05:33 | bb2d1514ff1982fa9236feb2148749bf386445d5 | test_bench_static_single_series_30_features    |    0.000314366 | seconds |                 0 |
| 2026-10-02 10:05:33 | bb2d1514ff1982fa9236feb2148749bf386445d5 | test_bench_static_batched_100_series           |    0.00178709  | seconds |                 0 |
| 2026-10-02 10:05:33 | bb2d1514ff1982fa9236feb2148749bf386445d5 | test_bench_static_batched_1000_series          |    0.0136822   | seconds |                 0 |