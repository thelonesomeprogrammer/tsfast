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
