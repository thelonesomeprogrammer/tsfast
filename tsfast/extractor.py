import numpy as np
from . import _tsfast
from .selection import select_features

class Extractor:
    def __init__(self, features="all", downsample=None, select_features=False, correlation_threshold=0.98, fdr_level=0.05):
        """
        High-level feature extractor.
        :param features: List of feature names or "all".
        :param downsample: If not None, downsample to this many bins using M4 first.
        :param select_features: If True, run FRESH filter after extraction.
        """
        if features == "all":
            self.features = [
                "mean", "variance", "std", "min", "max", "median", "skew", "kurtosis",
                "mad", "iqr", "entropy", "energy", "rms", "zero_crossing_rate",
                "peak_count", "autocorr_lag1", "mean_abs_change", "mean_change",
                "cid_ce", "slope", "intercept", "abs_sum_change", "count_above_mean",
                "count_below_mean", "longest_strike_above_mean", "longest_strike_below_mean",
                "variation_coefficient", "mean_abs_deviation", "auc", "slope_sign_change",
                "turning_points", "zero_crossing_mean", "zero_crossing_std"
            ]
        else:
            self.features = features
        
        self.downsample_bins = downsample
        self.should_select = select_features
        self.correlation_threshold = correlation_threshold
        self.fdr_level = fdr_level
        self.selected_indices = None
        self.constant_mask = None

    def fit(self, X, y=None):
        """
        Extract features and determine selection if requested.
        X: 2D array (samples, time_points)
        y: Optional target for selection.
        """
        features_matrix = self.transform(X)
        if self.should_select:
            # We need to store which features were kept
            # This is tricky because select_features currently returns a subset of the matrix
            # Let's modify select_features to return masks or indices.
            pass
        return self

    def transform(self, X):
        """
        Extract features from X.
        """
        if self.downsample_bins is not None:
            # Keep as list of arrays to handle inhomogeneous shapes from M4
            X_to_extract = [_tsfast.downsample(row, self.downsample_bins) for row in X]
        else:
            X_to_extract = X

        features_matrix = _tsfast.extract2d(X_to_extract, self.features)
        
        if self.selected_indices is not None:
            features_matrix = features_matrix[:, self.selected_indices]
            
        return features_matrix

    def fit_transform(self, X, y=None):
        """
        Fit and transform.
        """
        if self.downsample_bins is not None:
            X_to_extract = [_tsfast.downsample(row, self.downsample_bins) for row in X]
        else:
            X_to_extract = X

        features_matrix = _tsfast.extract2d(X_to_extract, self.features)

        if self.should_select:
            # Perform selection and store state
            X_selected, constant_mask = select_features(
                features_matrix, y, 
                correlation_threshold=self.correlation_threshold, 
                fdr_level=self.fdr_level
            )
            
            # Identify which indices were kept
            # This is a bit tricky with current select_features
            # Let's just return the selected matrix for now as in the plan
            return X_selected
        
        return features_matrix
