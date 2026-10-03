import pytest
import pyarrow as pa
import numpy as np
from tsfast._tsfast import ExpandingExtractor

def test_expanding_multiple_columns():
    features = ["mean", "max_value", "total_sum", 'mean_second_derivative_central', 'large_standard_deviation-0.05', 'symmetry_looking-0.05']
    n_cols = 2
    extractor = ExpandingExtractor(features, n_cols)
    
    # Update 1
    data1 = pa.RecordBatch.from_arrays([
        pa.array([1.0, 2.0], type=pa.float32()),
        pa.array([10.0, 20.0], type=pa.float32())
    ], names=['col1', 'col2'])
    
    result1 = extractor.update(data1).to_pandas()
    # col1 mean: 1.5, max: 2.0, sum: 3.0
    # col2 mean: 15.0, max: 20.0, sum: 30.0
    
    assert np.allclose(result1.iloc[0].iloc[:3], [1.5, 2.0, 3.0])
    assert np.allclose(result1.iloc[1].iloc[:3], [15.0, 20.0, 30.0])
    
    # Update 2
    data2 = pa.RecordBatch.from_arrays([
        pa.array([3.0], type=pa.float32()),
        pa.array([30.0], type=pa.float32())
    ], names=['col1', 'col2'])
    
    result2 = extractor.update(data2).to_pandas()
    # col1 mean: (1+2+3)/3 = 2.0, max: 3.0, sum: 6.0
    # col2 mean: (10+20+30)/3 = 20.0, max: 30.0, sum: 60.0
    
    assert np.allclose(result2.iloc[0].iloc[:3], [2.0, 3.0, 6.0])
    assert np.allclose(result2.iloc[1].iloc[:3], [20.0, 30.0, 60.0])

def test_expanding_all_basic_features():
    features = [
        "mean", "total_sum", "min_value", "max_value", 
        "energy", "root_mean_square", "mean_abs_change", "mean_change",
        "length", "variance_larger_than_standard_deviation"
    ]
    n_cols = 1
    extractor = ExpandingExtractor(features, n_cols)
    
    x = []
    for val in [1.0, 2.0, -1.0, 5.0]:
        x.append(val)
        data = pa.RecordBatch.from_arrays([pa.array([val], type=pa.float32())], names=['c'])
        res = extractor.update(data).to_pandas().iloc[0]
        
        arr = np.array(x)
        assert np.allclose(res['mean'], np.mean(arr))
        assert np.allclose(res['total_sum'], np.sum(arr))
        assert np.allclose(res['min_value'], np.min(arr))
        assert np.allclose(res['max_value'], np.max(arr))
        assert np.allclose(res['energy'], np.sum(arr**2))
        assert np.allclose(res['root_mean_square'], np.sqrt(np.mean(arr**2)))
        
        if len(x) > 1:
            diffs = np.diff(arr)
            assert np.allclose(res['mean_abs_change'], np.mean(np.abs(diffs)))
            assert np.allclose(res['mean_change'], np.mean(diffs))

def test_expanding_empty_batch():
    features = ["mean"]
    extractor = ExpandingExtractor(features, 1)
    empty_data = pa.RecordBatch.from_arrays([pa.array([], type=pa.float32())], names=['c'])
    result = extractor.update(empty_data)
    assert result.num_rows == 0

def test_expanding_paa():
    # PAA in expanding window is tricky because boundaries change.
    # Current implementation recalculates boundaries based on total N.
    features = ["paa-2-0", "paa-2-1"]
    extractor = ExpandingExtractor(features, 1)
    
    # Update 1: [1, 2] -> N=2. paa-2-0: [1], paa-2-1: [2]
    data1 = pa.RecordBatch.from_arrays([pa.array([1.0, 2.0], type=pa.float32())], names=['c'])
    res1 = extractor.update(data1).to_pandas().iloc[0]
    assert np.allclose(res1['paa-2-0'], 1.0)
    assert np.allclose(res1['paa-2-1'], 2.0)
    
    # Update 2: add [3, 4] -> N=4. paa-2-0: [1, 2] -> mean 1.5, paa-2-1: [3, 4] -> mean 3.5
    data2 = pa.RecordBatch.from_arrays([pa.array([3.0, 4.0], type=pa.float32())], names=['c'])
    res2 = extractor.update(data2).to_pandas().iloc[0]
    assert np.allclose(res2['paa-2-0'], 1.5)
    assert np.allclose(res2['paa-2-1'], 3.5)

def test_expanding_higher_moments():
    features = ["mean", "std_dev", "skewness", "kurtosis"]
    extractor = ExpandingExtractor(features, 1)
    
    # Use a larger dataset for skew/kurtosis to be more stable
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float32)
    
    # Update with whole array
    data = pa.RecordBatch.from_arrays([pa.array(x, type=pa.float32())], names=['c'])
    res = extractor.update(data).to_pandas().iloc[0]
    
    from scipy.stats import skew, kurtosis
    assert np.allclose(res['mean'], np.mean(x))
    assert np.allclose(res['std_dev'], np.std(x, ddof=1))
    # scipy skew/kurtosis might have different bias corrections, but let's check values are reasonable
    # tsfast skewness: (m3 / n) / var.powf(1.5)
    # tsfast kurtosis: (m4 / n) / (var * var) - 3.0
    
    m = np.mean(x)
    v = np.var(x, ddof=1)
    m3 = np.mean((x - m)**3)
    expected_skew = m3 / (v**1.5)
    expected_kurt = kurtosis(x, fisher=True, bias=False)
    
    # Wait, tsfast uses (m3/n) where m3 is sum of (x-mean)^3.
    # So (m3/n) is exactly np.mean((x-m)**3).
    assert np.allclose(res['skewness'], expected_skew, atol=1e-5)
    assert np.allclose(res["kurtosis"], expected_kurt, atol=1e-5)

def test_expanding_c3():
    # c3-lag: mean(x[t] * x[t-lag] * x[t-2*lag])
    features = ["c3-1", "c3-2"]
    extractor = ExpandingExtractor(features, 1)
    
    x = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)
    data = pa.RecordBatch.from_arrays([pa.array(x, type=pa.float32())], names=['c'])
    res = extractor.update(data).to_pandas().iloc[0]
    
    # c3-1: N=6. lags are at t=2, 3, 4, 5 (since we need t-2*lag >= 0)
    # t=2: x[2]*x[1]*x[0] = 3*2*1 = 6
    # t=3: x[3]*x[2]*x[1] = 4*3*2 = 24
    # t=4: x[4]*x[3]*x[2] = 5*4*3 = 60
    # t=5: x[5]*x[4]*x[3] = 6*5*4 = 120
    # mean: (6+24+60+120)/4 = 210/4 = 52.5
    assert np.allclose(res['c3-1'], 52.5)
    
    # c3-2: N=6. lag is 2. 2*lag=4. t=4, 5.
    # t=4: x[4]*x[2]*x[0] = 5*3*1 = 15
    # t=5: x[5]*x[3]*x[1] = 6*4*2 = 48
    # mean: (15+48)/2 = 63/2 = 31.5
    assert np.allclose(res['c3-2'], 31.5)

if __name__ == "__main__":
    pytest.main([__file__])

def test_expanding_duplicate_features():
    from tsfast._tsfast import ExpandingExtractor
    import pandas as pd
    from tsfresh.feature_extraction.feature_calculators import has_duplicate, has_duplicate_max, has_duplicate_min

    features = ["has_duplicate", "has_duplicate_max", "has_duplicate_min"]
    n_cols = 1
    extractor = ExpandingExtractor(features, n_cols)

    # Update 1
    x1 = np.array([1.0, 5.0, 3.0], dtype=np.float32)
    data1 = pa.RecordBatch.from_arrays([pa.array(x1)], names=['col1'])

    result1 = extractor.update(data1).to_pandas()

    # Update 2
    x2 = np.array([4.0, 5.0, 1.0, 2.0], dtype=np.float32)
    data2 = pa.RecordBatch.from_arrays([pa.array(x2)], names=['col1'])

    result2 = extractor.update(data2).to_pandas()

    full_series = np.concatenate([x1, x2])

    # Validate result1 (prefix chunks)
    prefix1 = x1
    series1 = pd.Series(prefix1)

    assert result1.iloc[0].iloc[0] == (1.0 if has_duplicate(series1) else 0.0)
    assert result1.iloc[0].iloc[1] == (1.0 if has_duplicate_max(series1) else 0.0)
    assert result1.iloc[0].iloc[2] == (1.0 if has_duplicate_min(series1) else 0.0)

    # Validate result2 (prefix chunks)
    series2 = pd.Series(full_series)

    assert result2.iloc[0].iloc[0] == (1.0 if has_duplicate(series2) else 0.0)
    assert result2.iloc[0].iloc[1] == (1.0 if has_duplicate_max(series2) else 0.0)
    assert result2.iloc[0].iloc[2] == (1.0 if has_duplicate_min(series2) else 0.0)

def test_reoccurring_ratios_expanding():
    import pyarrow as pa
    import tsfast
    import numpy as np

    x = np.array([1.0, 2.0, 2.0, 3.0, 3.0], dtype=np.float32)

    features = [
        "percentage_of_reoccurring_datapoints_to_all_datapoints",
        "percentage_of_reoccurring_values_to_all_values",
        "ratio_value_number_to_time_series_length"
    ]

    extractor = tsfast.ExpandingExtractor(features, 1)
    batch = pa.RecordBatch.from_arrays([pa.array(x)], names=['c1'])
    result_batch = extractor.update(batch)
    results = result_batch.to_pandas()

    # window 0 (idx 2): [1, 2, 2]
    # len=3, values={1,2} (2 unique), reoccur_dp=2 (two 2s), reoccur_val=1 (2)
    # res: [2/3, 1/2, 2/3]
    assert np.allclose(results.iloc[0].values, [4.0/5.0, 2.0/3.0, 3.0/5.0])

