import pytest
import numpy as np
import pyarrow as pa
import tsfast
import tsfel
from tsfel.feature_extraction.features import median_abs_deviation as tsfel_median_abs_dev, mean_abs_deviation as tsfel_mean_abs_dev

def test_median_abs_deviation():
    np.random.seed(42)
    # Generate some random data
    data = np.random.randn(100).astype(np.float32)

    # tsfast
    batch = pa.RecordBatch.from_arrays([pa.array(data)], names=['col1'])
    ext = tsfast.Extractor(['median_abs_deviation'])
    tsfast_res = ext.process_2d_floats(batch).column(0)[0].as_py()

    # tsfel
    tsfel_res = tsfel_median_abs_dev(data)

    # Assert within 1%
    assert abs(tsfast_res - tsfel_res) <= 0.01 * abs(tsfel_res) + 1e-6

def test_mean_abs_deviation():
    np.random.seed(42)
    data = np.random.randn(100).astype(np.float32)

    # tsfast (mapped to mad)
    batch = pa.RecordBatch.from_arrays([pa.array(data)], names=['col1'])
    ext = tsfast.Extractor(['mean_abs_deviation'])
    tsfast_res = ext.process_2d_floats(batch).column(0)[0].as_py()

    # tsfel
    tsfel_res = tsfel_mean_abs_dev(data)

    # Assert within 1%
    assert abs(tsfast_res - tsfel_res) <= 0.01 * abs(tsfel_res) + 1e-6

def test_incremental():
    import tsfast
    import numpy as np
    import pyarrow as pa

    np.random.seed(42)
    data = np.random.randn(100).astype(np.float32)

    # Expanding extractor
    batch1 = pa.RecordBatch.from_arrays([pa.array(data[:50])], names=['col1'])
    batch2 = pa.RecordBatch.from_arrays([pa.array(data[50:])], names=['col1'])

    ext = tsfast.ExpandingExtractor(['median_abs_deviation'], 1)
    ext.update(batch1)
    tsfast_res_expanding = ext.update(batch2).column(0)[0].as_py()

    tsfel_res = tsfel_median_abs_dev(data)
    assert abs(tsfast_res_expanding - tsfel_res) <= 0.01 * abs(tsfel_res) + 1e-6
def test_sliding():
    import tsfast
    import numpy as np
    import pyarrow as pa

    np.random.seed(42)
    data = np.random.randn(100).astype(np.float32)

    batch = pa.RecordBatch.from_arrays([pa.array(data)], names=['col1'])
    ext = tsfast.SlidingExtractor(['median_abs_deviation'], 1, 10, 1)
    tsfast_res = ext.update(batch)


    # Check correctness
    window_size = 10
    col_data = tsfast_res.column(0)
    for i in range(len(col_data)):
        window_data = data[i:i+window_size]
        tsfel_res = tsfel_median_abs_dev(window_data)
        tsfast_val = col_data[i].as_py()
        assert abs(tsfast_val - tsfel_res) <= 0.01 * abs(tsfel_res) + 1e-6
