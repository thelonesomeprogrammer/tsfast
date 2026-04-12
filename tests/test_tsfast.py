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


def test_m4():
    x = np.arange(100, dtype=np.float64)
    x[50] = 500.0
    x[20] = -500.0
    
    downsampled = tsfast.downsample(x, 10)
    assert 500 in downsampled
    assert -500 in downsampled
    print("test_m4 passed!")

def test_signal_features():
    # Signal with specific characteristics
    x = np.array([0, 1, 0, -1, 0, 1, 0, -1, 0], dtype=np.float64)
    features = [
        "auc", "ssc", "turning_points", 
        "zero_crossing_mean", "zero_crossing_std"
    ]
    results = tsfast.extract(x, features)
    
    # AUC: sum of trapezoids. 
    # [0, 1, 0, -1, 0, 1, 0, -1, 0]
    # Abs: [0, 1, 0, 1, 0, 1, 0, 1, 0]
    # Trapezoids: (0+1)/2=0.5, (1+0)/2=0.5, ... (1+0)/2=0.5 -> total 8 * 0.5 = 4.0
    assert np.allclose(results[0], 4.0)
    
    # SSC: local extrema. 
    # [0, 1, 0] -> 1 (max)
    # [1, 0, -1] -> 0
    # [0, -1, 0] -> 1 (min)
    # Total: 7 turning points (max, min, max, min, max, min, max)? 
    # [0, 1, 0, -1, 0, 1, 0, -1, 0]
    # idx 1: max (1 > 0 and 1 > 0)
    # idx 2: min (0 < 1 and 0 > -1) -> NO
    # Actually, peaks/valleys are at idx 1, 3, 5, 7.
    # Total 7 turning points: 
    # 1 (max), -1 (min), 1 (max), -1 (min) -> NO, wait.
    # 1, -1, 1, -1 are the values.
    # [0, 1, 0] -> 1 (idx 1)
    # [1, 0, -1] -> NO
    # [0, -1, 0] -> -1 (idx 3)
    # [ -1, 0, 1] -> NO
    # [0, 1, 0] -> 1 (idx 5)
    # [1, 0, -1] -> NO
    # [0, -1, 0] -> -1 (idx 7)
    # Total 4 turning points.
    assert np.allclose(results[2], 4.0)
    
    print("test_signal_features passed!")

if __name__ == "__main__":
    test_extract()
    test_new_features()
    test_paa()
    test_m4()
    test_signal_features()
