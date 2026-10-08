# Benchmark history

Written by `scripts/coach_benchmark.py`: all features in one extractor,
per engine, against tsfresh and TSFEL, tracked across commits. For the
headline comparison in the main README, see `scripts/readme_benchmark.py`;
per-feature timings are in `.jules/feature_benchmarks.md`.

![Benchmark trends](../.jules/benchmark_trends.png)

## Latest Results

| Date                | CommitHash   | Benchmark_Name                  |   Metric_Value | Unit   |   Delta_From_Last | Direction   |
|:--------------------|:-------------|:--------------------------------|---------------:|:-------|------------------:|:------------|
| 2026-10-07 11:34:57 | 851a2f0      | tsrocket_static_all_features    |        2.09651 | ms     |            -25.63 | 🟢          |
| 2026-10-07 11:34:57 | 851a2f0      | tsrocket_sliding_all_features   |        1.49502 | ms     |            -33.62 | 🟢          |
| 2026-10-07 11:34:57 | 851a2f0      | tsrocket_expanding_all_features |        2.04509 | ms     |            -24.93 | 🟢          |
| 2026-10-07 11:34:57 | 851a2f0      | reference_all_features          |      233.824   | ms     |             10.62 | 🔴          |
| 2026-10-07 11:34:57 | 851a2f0      | tsrocket_features               |      170       | count  |              8.97 | ⚪          |
| 2026-10-07 11:34:57 | 851a2f0      | referenced_features             |      161       | count  |              9.52 | ⚪          |
