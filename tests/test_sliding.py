import pytest
import pyarrow as pa
import numpy as np
from tsfast._tsfast import SlidingExtractor

def test_sliding_multiple_columns():
    features = ["mean", "total_sum"]
    n_cols = 2
    window_size = 2
    stride = 1
    extractor = SlidingExtractor(features, n_cols, window_size, stride)

    # Update 1: 3 rows
    # Col 1: [1.0, 2.0, 3.0]
    # Col 2: [10.0, 20.0, 30.0]
    data1 = pa.RecordBatch.from_arrays([
        pa.array([1.0, 2.0, 3.0], type=pa.float32()),
        pa.array([10.0, 20.0, 30.0], type=pa.float32())
    ], names=['col1', 'col2'])

    result1 = extractor.update(data1).to_pandas()
    # Output rows expected:
    # 1. Window [1.0, 2.0], [10.0, 20.0] -> col1 mean: 1.5, max: 2.0, sum: 3.0
    #                                      col2 mean: 15.0, max: 20.0, sum: 30.0
    # 2. Window [2.0, 3.0], [20.0, 30.0] -> col1 mean: 2.5, max: 3.0, sum: 5.0
    #                                      col2 mean: 25.0, max: 30.0, sum: 50.0

    # In sliding.rs, it creates a flattened output per column? Let's check how the output is formatted.
    # The output of SlidingExtractor gives results per column flattened or interleaved?
    # Actually, in sliding.rs: "let mut flat_data = Vec::with_capacity(n_cols * n_results);"
    # And it loops: for col_res in &column_results { for slide_res in col_res { flat_data.push(...) } }
    # So the output has size n_cols * n_results.
    # Col1 outputs then Col2 outputs?

    # Let's just calculate expected flattened values
    # result1 will have n_cols * 2 rows = 4 rows
    assert len(result1) == 4

    # Col1: 2 windows
    assert np.allclose(result1.iloc[0], [1.5, 3.0])
    assert np.allclose(result1.iloc[1], [2.5, 5.0])

    # Col2: 2 windows
    assert np.allclose(result1.iloc[2], [15.0, 30.0])
    assert np.allclose(result1.iloc[3], [25.0, 50.0])

def test_basic_features():
    features = [
        "mean", "total_sum",
        "energy", "root_mean_square",
        "length", "variance_larger_than_standard_deviation"
    ]
    n_cols = 1
    window_size = 3
    stride = 1
    extractor = SlidingExtractor(features, n_cols, window_size, stride)

    x = [1.0, 2.0, 3.0, 4.0]
    data = pa.RecordBatch.from_arrays([pa.array(x, type=pa.float32())], names=['c'])
    res = extractor.update(data).to_pandas()

    # Window 1: [1.0, 2.0, 3.0]
    # Window 2: [2.0, 3.0, 4.0]
    assert len(res) == 2

    w1 = np.array([1.0, 2.0, 3.0])
    assert np.allclose(res.iloc[0]['mean'], np.mean(w1))
    assert np.allclose(res.iloc[0]['total_sum'], np.sum(w1))
    # assert np.allclose(res.iloc[0]['min_value'], np.min(w1))
    # assert np.allclose(res.iloc[0]['max_value'], np.max(w1))
    assert np.allclose(res.iloc[0]['energy'], np.sum(w1**2))
    assert np.allclose(res.iloc[0]['root_mean_square'], np.sqrt(np.mean(w1**2)))

    w2 = np.array([2.0, 3.0, 4.0])
    assert np.allclose(res.iloc[1]['mean'], np.mean(w2))
    assert np.allclose(res.iloc[1]['total_sum'], np.sum(w2))
    # assert np.allclose(res.iloc[1]['min_value'], np.min(w2))
    # assert np.allclose(res.iloc[1]['max_value'], np.max(w2))
    assert np.allclose(res.iloc[1]['energy'], np.sum(w2**2))
    assert np.allclose(res.iloc[1]['root_mean_square'], np.sqrt(np.mean(w2**2)))

def test_empty_batch():
    features = ["mean"]
    extractor = SlidingExtractor(features, 1, 2, 1)
    empty_data = pa.RecordBatch.from_arrays([pa.array([], type=pa.float32())], names=['c'])
    result = extractor.update(empty_data)
    assert result.num_rows == 0

def test_stride_behavior():
    features = ["mean"]
    n_cols = 1
    window_size = 4
    stride = 2
    extractor = SlidingExtractor(features, n_cols, window_size, stride)

    # Input size 6: indices [0, 1, 2, 3, 4, 5]
    # Window 1 (len 4): [0, 1, 2, 3]
    # Window 2 (len 4, strided by 2): [2, 3, 4, 5]
    x = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    data = pa.RecordBatch.from_arrays([pa.array(x, type=pa.float32())], names=['c'])
    res = extractor.update(data).to_pandas()

    assert len(res) == 2
    assert np.allclose(res.iloc[0]['mean'], np.mean([0.0, 1.0, 2.0, 3.0]))
    assert np.allclose(res.iloc[1]['mean'], np.mean([2.0, 3.0, 4.0, 5.0]))

if __name__ == "__main__":
    pytest.main([__file__])

def test_sliding_invalid_feature():
    with pytest.raises(ValueError):
        SlidingExtractor(["invalid_feature"], 1, 3, 1)

def test_sliding_paa():
    features = ["paa-2-0", "paa-2-1"]
    n_cols = 1
    window_size = 4
    stride = 1
    extractor = SlidingExtractor(features, n_cols, window_size, stride)

    x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    data = pa.RecordBatch.from_arrays([pa.array(x, type=pa.float32())], names=['c'])
    res = extractor.update(data).to_pandas()

    # window 1: [1, 2, 3, 4] -> paa-2-0 is mean(1,2)=1.5, paa-2-1 is mean(3,4)=3.5
    # window 2: [2, 3, 4, 5] -> paa-2-0 is mean(2,3)=2.5, paa-2-1 is mean(4,5)=4.5
    assert len(res) == 3
    assert np.allclose(res.iloc[0]['paa-2-0'], 1.5)
    assert np.allclose(res.iloc[0]['paa-2-1'], 3.5)
    assert np.allclose(res.iloc[1]['paa-2-0'], 2.5)
    assert np.allclose(res.iloc[1]['paa-2-1'], 4.5)

def test_sliding_higher_moments():
    features = ["mean", "std_dev", "skewness", "kurtosis"]
    extractor = SlidingExtractor(features, 1, 5, 1)

    x = np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.float32)
    data = pa.RecordBatch.from_arrays([pa.array(x, type=pa.float32())], names=['c'])
    res = extractor.update(data).to_pandas()

    assert len(res) == 3
    # window 1: [1, 2, 3, 4, 5]
    w1 = x[:5]
    assert np.allclose(res.iloc[0]['mean'], np.mean(w1))
    assert np.allclose(res.iloc[0]['std_dev'], np.std(w1, ddof=1))

    from scipy.stats import kurtosis
    v = np.var(w1, ddof=1)
    m = np.mean(w1)
    m3 = np.mean((w1 - m)**3)
    expected_skew = m3 / (v**1.5)
    expected_kurt = kurtosis(w1, fisher=True, bias=False)

    assert np.allclose(res.iloc[0]['skewness'], expected_skew, atol=1e-5)
    assert np.allclose(res.iloc[0]['kurtosis'], expected_kurt, atol=1e-5)
