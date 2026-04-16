import numpy as np
import tsfast

def test_m4():
    print("Testing M4 Downsampling...")
    data = np.linspace(0, 100, 1000)
    # Add some peaks and valleys
    data[100] = 500
    data[500] = -500
    
    downsampled = tsfast.downsample(data, 10)
    print(f"Original size: {len(data)}, Downsampled size: {len(downsampled)}")
    assert 500 in downsampled
    assert -500 in downsampled
    print("M4 Downsampling OK")

def test_new_features():
    print("\nTesting New Features...")
    data = np.sin(np.linspace(0, 10 * np.pi, 1000))
    features = [
        "auc", "ssc", "turning_points", 
        "zero_crossing_mean", "zero_crossing_std"
    ]
    extracted = tsfast.extract(data, features)
    print(f"Extracted features: {dict(zip(features, extracted))}")
    assert extracted[0] > 0 # AUC
    assert extracted[1] > 0 # SSC
    assert extracted[2] > 0 # Turning Points
    print("New Features OK")

def test_extractor():
    print("\nTesting Extractor...")
    X = np.random.randn(20, 1000)
    y = np.array([0, 1] * 10)
    # Make the mean informative
    X[y == 1] += 5.0
    
    extractor = tsfast.Extractor(
        features="all",
        downsample=100,
        select_features=True
    )
    
    features = extractor.fit_transform(X, y)
    print(f"Feature matrix shape: {features.shape}")
    assert features.shape[1] > 0
    print("Extractor OK")

if __name__ == "__main__":
    test_m4()
    test_new_features()
    test_extractor()
