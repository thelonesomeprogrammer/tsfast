import tsfast
import numpy as np
import pyarrow as pa
import pytest

def test_extract():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    features = ["mean", "std", "energy", "min", "max", "autocorr_lag1"]
    
    extractor = tsfast.Extractor(features)
    batch = pa.RecordBatch.from_arrays([pa.array(x)], names=['c1'])
    result_batch = extractor.process_2d_floats(batch)
    results = result_batch.to_pandas().iloc[0].values
    
    print(f"Extract results: {results}")
    
    assert np.allclose(results[0], 3.0)
    assert np.allclose(results[1], np.std(x, ddof=1)) # Rust uses ddof=1 for variance/std
    assert np.allclose(results[2], np.sum(x**2))
    assert results[3] == 1.0
    assert results[4] == 5.0
    
    # AutocorrLag1 parity with manual calculation (including the x[0]*x[0] start in Rust implementation)
    # Manual: ( (1*1 + 2*1 + 3*2 + 4*3 + 5*4) / 4 - 3*3 ) / 2.5 = (41/4 - 9) / 2.5 = 1.25 / 2.5 = 0.5
    assert np.allclose(results[5], 0.5)
    print("test_extract passed!")

def test_new_features():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    features = ["mad", "iqr", "entropy", "mean_abs_change", "mean_change", "cid_ce"]
    
    extractor = tsfast.Extractor(features)
    batch = pa.RecordBatch.from_arrays([pa.array(x)], names=['c1'])
    result_batch = extractor.process_2d_floats(batch)
    results = result_batch.to_pandas().iloc[0].values
    
    # MAD: mean(|x - mean(x)|)
    assert np.allclose(results[0], 1.5)
    # IQR: Q3 - Q1. x=[1, 2, 3, 4, 5, 6]. Q1=2.0, Q3=5.0. IQR=3.
    assert np.allclose(results[1], 3.0)
    assert results[2] > 0
    assert np.allclose(results[3], 1.0)
    assert np.allclose(results[4], 1.0)
    assert np.allclose(results[5], np.sqrt(5.0))
    print("test_new_features passed!")

def test_paa():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    features = ["paa-2-0", "paa-2-1"]
    
    extractor = tsfast.Extractor(features)
    batch = pa.RecordBatch.from_arrays([pa.array(x)], names=['c1'])
    result_batch = extractor.process_2d_floats(batch)
    results = result_batch.to_pandas().iloc[0].values
    
    assert np.allclose(results[0], 2.0)
    assert np.allclose(results[1], 5.0)
    print("test_paa passed!")

def test_advanced_features():
    x = np.array([1.0, 2.0, 1.0, 2.0, 1.0, 2.0], dtype=np.float32)
    # autocorr-2 for [1,2,1,2,1,2] mean=1.5, var=0.3
    # num: sum_{i=0}^{n-lag-1} (x[i]-mean)(x[i+lag]-mean)
    # i=0: (1-1.5)*(1-1.5) = 0.25
    # i=1: (2-1.5)*(2-1.5) = 0.25
    # i=2: (1-1.5)*(1-1.5) = 0.25
    # i=3: (2-1.5)*(2-1.5) = 0.25
    # sum = 1.0
    # den = var * (n-1) = 0.3 * 5 = 1.5
    # result = 1.0 / 1.5 = 0.666...
    features = ["fft_coeff-1-real", "fft_coeff-1-abs", "autocorr-2"]
    
    extractor = tsfast.Extractor(features)
    batch = pa.RecordBatch.from_arrays([pa.array(x)], names=['c1'])
    result_batch = extractor.process_2d_floats(batch)
    results = result_batch.to_pandas().iloc[0].values
    print(f"Advanced features results: {results}")
    
    # FFT real coeff 1 parity
    assert np.allclose(results[0], 0.0, atol=1e-5)
    
    # autocorr-2 parity
    assert np.allclose(results[2], 2/3, atol=1e-5)
    print("test_advanced_features passed!")

def test_2d_extraction():
    # Test processing multiple series at once
    x = np.array([
        [1, 2, 3, 4, 5],
        [5, 4, 3, 2, 1]
    ], dtype=np.float32)
    features = ["mean", "max_value"]
    
    extractor = tsfast.Extractor(features)
    batch = pa.RecordBatch.from_arrays([
        pa.array(x[0]),
        pa.array(x[1])
    ], names=['s1', 's2'])
    
    result_batch = extractor.process_2d_floats(batch)
    df = result_batch.to_pandas()
    
    assert np.allclose(df.iloc[0], [3.0, 5.0]) # s1
    assert np.allclose(df.iloc[1], [3.0, 5.0]) # s2
    print("test_2d_extraction passed!")

if __name__ == "__main__":
    test_extract()
    test_new_features()
    test_paa()
    test_advanced_features()
    test_2d_extraction()
