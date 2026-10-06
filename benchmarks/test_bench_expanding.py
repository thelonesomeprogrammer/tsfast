import pytest
import tsfast


@pytest.mark.benchmark(group="expanding_extraction")
def test_bench_expanding_initial_batch(benchmark, single_batch, full_30_features):
    def init_and_update():
        ext = tsfast.ExpandingExtractor(full_30_features, 1)
        return ext.update(single_batch)

    benchmark(init_and_update)


@pytest.mark.benchmark(group="expanding_extraction")
def test_bench_expanding_incremental_update_10cols(
    benchmark, streaming_chunks, full_30_features
):
    ext = tsfast.ExpandingExtractor(full_30_features, 10)
    # Prime with initial 5 chunks
    for i in range(5):
        ext.update(streaming_chunks[i])

    # Benchmark a single incremental update step
    next_chunk = streaming_chunks[5]
    benchmark(ext.update, next_chunk)


@pytest.mark.benchmark(group="expanding_stream")
def test_bench_expanding_full_stream_20_chunks(
    benchmark, streaming_chunks, full_30_features
):
    def run_stream():
        ext = tsfast.ExpandingExtractor(full_30_features, 10)
        for chunk in streaming_chunks:
            ext.update(chunk)

    benchmark(run_stream)
