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

def test_paa():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    features = ["paa-2-0", "paa-2-1"]
    results = tsfast.extract(x, features)
    
    # paa-2-0: [1, 2, 3] -> mean 2.0
    assert np.allclose(results[0], 2.0)
    # paa-2-1: [4, 5, 6] -> mean 5.0
    assert np.allclose(results[1], 5.0)
    
    print("test_paa passed!")


if __name__ == "__main__":
    test_extract()
    test_new_features()
    test_paa()
