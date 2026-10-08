import pytest
import numpy as np
import tsrocket
import tsfel
from tsfel.feature_extraction.features import (
    median_abs_deviation as tsfel_median_abs_dev,
    mean_abs_deviation as tsfel_mean_abs_dev,
)


def test_median_abs_deviation():
    np.random.seed(42)
    # Generate some random data
    data = np.random.randn(100).astype(np.float32)

    # tsrocket
    batch = np.stack([data])
    ext = tsrocket.Extractor(["median_abs_deviation"])
    tsrocket_res = ext.process_2d_floats(batch)[0, 0]

    # tsfel
    tsfel_res = tsfel_median_abs_dev(data)

    # Assert within 1%
    assert abs(tsrocket_res - tsfel_res) <= 0.01 * abs(tsfel_res) + 1e-6


def test_mean_abs_deviation():
    np.random.seed(42)
    data = np.random.randn(100).astype(np.float32)

    # tsrocket (mapped to mad)
    batch = np.stack([data])
    ext = tsrocket.Extractor(["mean_abs_deviation"])
    tsrocket_res = ext.process_2d_floats(batch)[0, 0]

    # tsfel
    tsfel_res = tsfel_mean_abs_dev(data)

    # Assert within 1%
    assert abs(tsrocket_res - tsfel_res) <= 0.01 * abs(tsfel_res) + 1e-6


def test_incremental():
    import tsrocket
    import numpy as np

    np.random.seed(42)
    data = np.random.randn(100).astype(np.float32)

    # Expanding extractor
    batch1 = np.stack([data[:50]])
    batch2 = np.stack([data[50:]])

    ext = tsrocket.ExpandingExtractor(["median_abs_deviation"], 1)
    ext.update(batch1)
    tsrocket_res_expanding = ext.update(batch2)[0, 0]

    tsfel_res = tsfel_median_abs_dev(data)
    assert abs(tsrocket_res_expanding - tsfel_res) <= 0.01 * abs(tsfel_res) + 1e-6


def test_sliding():
    import tsrocket
    import numpy as np

    np.random.seed(42)
    data = np.random.randn(100).astype(np.float32)

    batch = np.stack([data])
    ext = tsrocket.SlidingExtractor(["median_abs_deviation"], 1, 10, 1)
    tsrocket_res = ext.update(batch)

    # Check correctness
    window_size = 10
    col_data = tsrocket_res[0, :, 0]
    for i in range(len(col_data)):
        window_data = data[i : i + window_size]
        tsfel_res = tsfel_median_abs_dev(window_data)
        tsrocket_val = col_data[i]
        assert abs(tsrocket_val - tsfel_res) <= 0.01 * abs(tsfel_res) + 1e-6
