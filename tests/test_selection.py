import numpy as np
import pytest
from tsfast.selection import select_features

def test_select_features_classification():
    # Set seed for reproducibility
    np.random.seed(42)

    # Target array: binary classification
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    # Feature 0: Highly informative (perfectly separates classes)
    f0 = np.where(y == 0, np.random.normal(0, 0.1, len(y)), np.random.normal(5, 0.1, len(y)))

    # Feature 1: Random/uninformative
    f1 = np.random.normal(0, 1, len(y))

    # Combine features into matrix X
    X = np.column_stack((f0, f1))

    # Select features
    X_selected, constant_mask = select_features(X, y)

    # Feature 0 should be selected, feature 1 should be dropped
    assert X_selected.shape[1] == 1
    assert np.allclose(X_selected[:, 0], f0)
    assert np.all(constant_mask)

def test_select_features_classification_small_groups():
    # Set seed for reproducibility
    np.random.seed(42)

    # Target array with one class having only 1 sample
    y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

    # Feature 0: Some feature
    f0 = np.random.normal(0, 1, len(y))

    # Feature 1: Another feature
    f1 = np.random.normal(0, 1, len(y))

    # Combine features into matrix X
    X = np.column_stack((f0, f1))

    # Select features
    X_selected, constant_mask = select_features(X, y)

    # Since one group has size 1, p-value is set to 1.0.
    # Therefore, no feature should be significant (FDR control should drop all).
    assert X_selected.shape[1] == 0
    assert np.all(constant_mask)
from tsfast.selection import select_features

def test_select_features_unsupervised():
    # Stage 1: Unsupervised filtering
    # Col 0: Constant feature -> should be removed
    # Col 1: Useful feature 1 -> keep
    # Col 2: Useful feature 1 copy -> highly correlated -> removed
    # Col 3: Useful feature 2 -> keep
    # Col 4: Useful feature 3 -> keep (not perfectly correlated with 1 or 3)
    X = np.array([
        [1, 1, 1, 0, 1],
        [1, 2, 2, 1, 8],
        [1, 3, 3, 0, 3],
        [1, 4, 4, 1, 9],
        [1, 5, 5, 0, 2]
    ], dtype=np.float64)

    X_out, constant_mask = select_features(X)

    # Constant mask should identify col 0 as false, others as true
    np.testing.assert_array_equal(constant_mask, [False, True, True, True, True])

    # After constant mask, cols 1, 2, 3, 4 are passed to correlation.
    # Col 1 and Col 2 have correlation 1.0 (threshold 0.98), so Col 2 is dropped.
    # Col 3 and Col 4 have correlation ~0.97, which is < 0.98, so both are kept.
    # We expect 3 columns remaining from the 4 non-constant columns.
    assert X_out.shape == (5, 3)

    # Checking output values matches expected features
    # Keeping Col 1, Col 3, Col 4.
    expected_X = np.array([
        [1, 0, 1],
        [2, 1, 8],
        [3, 0, 3],
        [4, 1, 9],
        [5, 0, 2]
    ], dtype=np.float64)

    np.testing.assert_array_almost_equal(X_out, expected_X)
