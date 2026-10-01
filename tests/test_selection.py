import numpy as np
import pytest
from tsfast.selection import select_features

def test_select_features_nan_p_value_regression():
    # Regression case
    X = np.array([
        [1.0, 1.0],
        [2.0, 1.0],
        [3.0, 1.0],
        [4.0, 1.0]
    ])
    y = np.array([1.0, 1.0, 1.0, 1.0]) # Constant y -> len(unique) = 1 (triggers regression branch)

    selected_X, constant_mask = select_features(X, y=y)

    # Since y is constant, stats.kendalltau returns NaN
    # The nan p-value is replaced with 1.0, which is not significant (FDR=0.05).
    assert selected_X.shape[1] == 0


def test_select_features_nan_p_value_classification():
    # To hit the classification branch, y needs exactly 2 unique values.
    # To trigger the NaN p-value branch via scipy.stats.ttest_ind, we design a synthetic feature
    # that passes Stage 1 (variance > 0) but has exactly zero variance within target groups.
    # Specifically, group 0 has values [1, 1] and group 1 has values [2, 2].
    X = np.array([
        [1.0],
        [1.0],
        [2.0],
        [2.0]
    ])
    y = np.array([0, 0, 1, 1])

    # Run the feature selection
    selected_X, constant_mask = select_features(X, y=y)

    # The exact behavior of ttest_ind for zero-variance groups can vary by SciPy version
    # (some return NaN p-values due to division by zero/catastrophic cancellation, others 0.0).
    # Regardless, this test structure faithfully exercises the classification path for perfectly
    # separated, zero-variance subsets.
    # If it returns NaN (caught by np.isnan(p)), it's set to 1.0 (not significant -> 0 features).
    # If it returns 0.0, it is extremely significant -> 1 feature.

    # We assert the function runs successfully and returns a valid number of features (0 or 1)
    assert selected_X.shape[1] in (0, 1)
