import pytest
import numpy as np
import pyarrow as pa
import tsfast

@pytest.fixture
def sliding_stream():
    np.random.seed(42)
    n_cols = 4
    total_len = 1000
    chunk_size = 50
    data = np.random.randn(n_cols, total_len).astype(np.float32)
    col_names = [f"c_{i}" for i in range(n_cols)]
    
    chunks = []
    for j in range(0, total_len, chunk_size):
        chunk = data[:, j:j + chunk_size]
        batch = pa.RecordBatch.from_arrays(
            [pa.array(chunk[i]) for i in range(n_cols)],
            names=col_names
        )
        chunks.append(batch)
    return chunks

@pytest.mark.benchmark(group="sliding_extraction")
def test_bench_sliding_window200_stride50(benchmark, sliding_stream, basic_features):
    def run_sliding():
        ext = tsfast.SlidingExtractor(basic_features, 4, window_size=200, stride=50)
        for chunk in sliding_stream:
            ext.update(chunk)

    benchmark(run_sliding)

@pytest.mark.benchmark(group="sliding_extraction")
def test_bench_sliding_single_chunk_stride50(benchmark, sliding_stream, basic_features):
    ext = tsfast.SlidingExtractor(basic_features, 4, window_size=200, stride=50)
    # Prime with initial 4 chunks (200 points)
    for i in range(4):
        ext.update(sliding_stream[i])
    
    next_chunk = sliding_stream[4]
    benchmark(ext.update, next_chunk)
