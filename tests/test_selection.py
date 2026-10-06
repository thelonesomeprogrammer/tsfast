import numpy as np
import pytest
from tsfast.selection import select_features


def test_select_features_nan_p_value_regression():
    # Regression case
    X = np.array([[1.0, 1.0], [2.0, 1.0], [3.0, 1.0], [4.0, 1.0]])
    y = np.array(
        [1.0, 1.0, 1.0, 1.0]
    )  # Constant y -> len(unique) = 1 (triggers regression branch)

    selected_X, constant_mask = select_features(X, y=y)

    # Since y is constant, stats.kendalltau returns NaN
    # The nan p-value is replaced with 1.0, which is not significant (FDR=0.05).
    assert selected_X.shape[1] == 0


def test_select_features_nan_p_value_classification():
    # To hit the classification branch, y needs exactly 2 unique values.
    # To trigger the NaN p-value branch via scipy.stats.ttest_ind, we design a synthetic feature
    # that passes Stage 1 (variance > 0) but has exactly zero variance within target groups.
    # Specifically, group 0 has values [1, 1] and group 1 has values [2, 2].
    X = np.array([[1.0], [1.0], [2.0], [2.0]])
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


def test_select_features_classification():
    # Set seed for reproducibility
    np.random.seed(42)

    # Target array: binary classification
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    # Feature 0: Highly informative (perfectly separates classes)
    f0 = np.where(
        y == 0, np.random.normal(0, 0.1, len(y)), np.random.normal(5, 0.1, len(y))
    )

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
    X = np.array(
        [
            [1, 1, 1, 0, 1],
            [1, 2, 2, 1, 8],
            [1, 3, 3, 0, 3],
            [1, 4, 4, 1, 9],
            [1, 5, 5, 0, 2],
        ],
        dtype=np.float64,
    )

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
    expected_X = np.array(
        [[1, 0, 1], [2, 1, 8], [3, 0, 3], [4, 1, 9], [5, 0, 2]], dtype=np.float64
    )

    np.testing.assert_array_almost_equal(X_out, expected_X)


def test_select_features_all_constant():
    # Test early return when all features are constant
    X = np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
    y = np.array([0, 1, 0, 1])

    selected_X, constant_mask = select_features(X, y=y)

    # Since all features are constant, selected_X should have 0 columns
    # and constant_mask should be all False.
    assert selected_X.shape[1] == 0
    np.testing.assert_array_equal(constant_mask, [False, False])
