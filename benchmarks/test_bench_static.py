import pytest
import tsrocket


@pytest.mark.benchmark(group="static_extraction")
def test_bench_static_single_series_basic(benchmark, single_batch, basic_features):
    extractor = tsrocket.Extractor(basic_features)
    # Warmup
    _ = extractor.process_2d_floats(single_batch)
    benchmark(extractor.process_2d_floats, single_batch)


@pytest.mark.benchmark(group="static_extraction")
def test_bench_static_single_series_advanced(
    benchmark, single_batch, advanced_features
):
    extractor = tsrocket.Extractor(advanced_features)
    _ = extractor.process_2d_floats(single_batch)
    benchmark(extractor.process_2d_floats, single_batch)


@pytest.mark.benchmark(group="static_extraction")
def test_bench_static_single_series_30_features(
    benchmark, single_batch, full_30_features
):
    extractor = tsrocket.Extractor(full_30_features)
    _ = extractor.process_2d_floats(single_batch)
    benchmark(extractor.process_2d_floats, single_batch)


@pytest.mark.benchmark(group="static_batch_scaling")
def test_bench_static_batched_100_series(benchmark, batch_100_series, full_30_features):
    extractor = tsrocket.Extractor(full_30_features)
    _ = extractor.process_2d_floats(batch_100_series)
    benchmark(extractor.process_2d_floats, batch_100_series)


@pytest.mark.benchmark(group="static_batch_scaling")
def test_bench_static_batched_1000_series(
    benchmark, batch_1000_series, full_30_features
):
    extractor = tsrocket.Extractor(full_30_features)
    _ = extractor.process_2d_floats(batch_1000_series)
    benchmark(extractor.process_2d_floats, batch_1000_series)
