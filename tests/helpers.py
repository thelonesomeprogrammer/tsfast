"""Shared test helpers."""
import pandas as pd


def frame(extractor, out):
    """Extractor output as a DataFrame, one column per feature.

    Sliding output (n_series, n_windows, n_features) is flattened series-major:
    every window of the first series, then every window of the second, ...
    """
    return pd.DataFrame(out.reshape(-1, out.shape[-1]), columns=extractor.feature_names)
