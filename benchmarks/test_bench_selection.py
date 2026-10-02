import pytest
import numpy as np
from tsfast.selection import select_features

@pytest.fixture(scope="module")
def selection_data():
    np.random.seed(42)
    n_samples = 200
    n_features = 50
    X = np.random.randn(n_samples, n_features)
    # Add 5 constant features
    X[:, 0:5] = 1.0
    # Add 5 identical duplicate features (correlation 1.0)
    X[:, 10:15] = X[:, 5:10]
    
    # Binary classification target
    y = np.random.randint(0, 2, size=n_samples)
    return X, y

@pytest.mark.benchmark(group="feature_selection")
def test_bench_select_features_supervised(benchmark, selection_data):
    X, y = selection_data
    benchmark(select_features, X, y=y)

@pytest.mark.benchmark(group="feature_selection")
def test_bench_select_features_unsupervised(benchmark, selection_data):
    X, _ = selection_data
    benchmark(select_features, X, y=None)
