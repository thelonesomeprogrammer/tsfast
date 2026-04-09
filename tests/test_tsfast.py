import tsfast
import numpy as np

def test_extract():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    features = ["mean", "std", "energy", "min", "max", "autocorr_lag1"]
    results = tsfast.extract(x, features)
    print(f"Extract results: {results}")
    
    assert np.allclose(results[0], 3.0)
    assert np.allclose(results[1], np.std(x))
    assert np.allclose(results[2], np.sum(x**2))
    assert results[3] == 1.0
    assert results[4] == 5.0
    
    # Simple autocorr check
    m = np.mean(x)
    num = np.sum((x[1:] - m) * (x[:-1] - m))
    den = np.sum((x - m)**2)
    expected_autocorr = num / den if den != 0 else 0.0
    assert np.allclose(results[5], expected_autocorr)
    print("test_extract passed!")

def test_new_features():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    features = ["mad", "iqr", "entropy", "mean_abs_change", "mean_change", "cid_ce"]
    results = tsfast.extract(x, features)
    
    # MAD: median([2.5, 1.5, 0.5, 0.5, 1.5, 2.5]) = 1.5
    assert np.allclose(results[0], 1.5)
    
    # IQR: Q3 - Q1. x=[1, 2, 3, 4, 5, 6]. Q1=median([1, 2, 3])=2, Q3=median([4, 5, 6])=5. IQR=3.
    assert np.allclose(results[1], 3.0)
    
    # Entropy: non-zero for varying data
    assert results[2] > 0
    
    # mean_abs_change: mean(|x[i+1]-x[i]|) = 1.0
    assert np.allclose(results[3], 1.0)
    
    # mean_change: mean(x[i+1]-x[i]) = 1.0
    assert np.allclose(results[4], 1.0)
    
    # cid_ce: sqrt(sum((x[i+1]-x[i])^2)) = sqrt(5 * 1^2) = sqrt(5)
    assert np.allclose(results[5], np.sqrt(5.0))
    print("test_new_features passed!")

def test_npaa_npta():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    features = ["npaa-2-0", "npaa-2-1", "npta-2-0", "npta-2-1", "paa-2-0", "paa-2-1", "npaa-invalid"]
    results = tsfast.extract(x, features)
    
    # Normalization of x: mean=3.5, std ~ 1.7078
    m = np.mean(x)
    s = np.std(x)
    
    # npaa-2-0: segment 0 is x[0..3] = [1, 2, 3]. mean is 2. Normalized = (2 - 3.5)/s
    expected_npaa_0 = (2.0 - m) / s
    assert np.allclose(results[0], expected_npaa_0)
    
    # npaa-2-1: segment 1 is x[3..6] = [4, 5, 6]. mean is 5. Normalized = (5 - 3.5)/s
    expected_npaa_1 = (5.0 - m) / s
    assert np.allclose(results[1], expected_npaa_1)
    
    # npta-2-0: segment 0 is [1, 2, 3]. norm = (x-m)/s. delta = (norm_last - norm_first) / (len - 1)
    expected_npta_0 = ((3.0 - m)/s - (1.0 - m)/s) / 2.0
    assert np.allclose(results[2], expected_npta_0)
    
    # npta-2-1: segment 1 is [4, 5, 6]. delta = (norm_last - norm_first) / (len - 1)
    expected_npta_1 = ((6.0 - m)/s - (4.0 - m)/s) / 2.0
    assert np.allclose(results[3], expected_npta_1)

    # paa-2-0: [1, 2, 3] -> mean 2.0
    assert np.allclose(results[4], 2.0)
    # paa-2-1: [4, 5, 6] -> mean 5.0
    assert np.allclose(results[5], 5.0)
    
    # invalid feature returns NaN
    assert np.isnan(results[6])
    
    print("test_npaa_npta passed!")

def test_extract_expanding():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    features = ["mean", "energy", "min", "max", "autocorr_lag1", "mean_abs_change", "mean_change"]
    results = tsfast.extract_expanding(x, features)
    print(f"Extract expanding results:\n{results}")
    
    # results shape should be (len(x), len(features))
    assert results.shape == (5, 7)
    
    # Mean column
    expected_mean = np.cumsum(x) / np.arange(1, 6)
    assert np.allclose(results[:, 0], expected_mean)
    
    # Energy column
    expected_energy = np.cumsum(x**2)
    assert np.allclose(results[:, 1], expected_energy)
    
    # Min/Max
    assert np.all(results[:, 2] == 1.0)
    assert np.all(results[:, 3] == np.arange(1, 6))
    
    # Autocorr lag 1
    # Check last value
    m = np.mean(x)
    num = np.sum((x[1:] - m) * (x[:-1] - m))
    den = np.sum((x - m)**2)
    expected_autocorr = num / den if den != 0 else 0.0
    assert np.allclose(results[4, 4], expected_autocorr)

    # mean_abs_change
    # x=[1, 2, 3, 4, 5]. diffs=[1, 1, 1, 1]. mean_abs_change at each step: [0, 1, 1, 1, 1]
    assert np.allclose(results[:, 5], [0, 1, 1, 1, 1])

    # mean_change
    assert np.allclose(results[:, 6], [0, 1, 1, 1, 1])

    # Test O(n^2) fallback for some features
    features_fallback = ["mad", "iqr"]
    results_fb = tsfast.extract_expanding(x, features_fallback)
    assert results_fb.shape == (5, 2)
    # x=[1, 2, 3, 4, 5]
    # mad at each step:
    # [1]: 0
    # [1, 2]: median(|1-1.5|, |2-1.5|) = 0.5
    # [1, 2, 3]: median(1, 0, 1) = 1.0
    # [1, 2, 3, 4]: median(1.5, 0.5, 0.5, 1.5) = 1.0
    # [1, 2, 3, 4, 5]: median(2, 1, 0, 1, 2) = 1.0
    assert np.allclose(results_fb[:, 0], [0, 0.5, 1.0, 1.0, 1.0])
    
    print("test_extract_expanding passed!")

def test_extract_expanding_npaa_npta():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    features = ["npaa-2-0", "npaa-2-1", "npta-2-0", "npta-2-1", "paa-2-0", "paa-2-1", "npaa-invalid"]
    results = tsfast.extract_expanding(x, features)
    
    # Check shape
    assert results.shape == (6, 7)
    
    # Validate the last row matches the batch extraction for the full window
    batch_results = tsfast.extract(x, features)
    # Exclude invalid index from allclose due to NaNs
    assert np.allclose(results[-1, :6], batch_results[:6])
    assert np.isnan(results[-1, 6])
    
    print("test_extract_expanding_npaa_npta passed!")

if __name__ == "__main__":
    test_extract()
    test_new_features()
    test_npaa_npta()
    test_extract_expanding()
    test_extract_expanding_npaa_npta()
