"""Shared test helpers."""

import pandas as pd


def frame(extractor, out):
    """Extractor output as a DataFrame, one column per feature.

    Sliding output (n_series, n_windows, n_features) is flattened series-major:
    every window of the first series, then every window of the second, ...
    """
    return pd.DataFrame(out.reshape(-1, out.shape[-1]), columns=extractor.feature_names)


def tsfresh_default_references():
    """References for tsfresh/TSFEL defaults tsrocket used to get wrong or lack:
    cid_ce(normalize=True), f_agg="median", partial autocorrelation at lag 0,
    agg_linear_trend's var (ddof=0 on the numpy chunks extract_features
    passes), and TSFEL's ECDF rank at a p that f32 can't hold exactly."""
    import numpy as np
    import tsfel.feature_extraction.features as F
    import tsfresh.feature_extraction.feature_calculators as fc

    def one(result):
        return list(result)[0][1]

    def ecdf_percentile(x, p):
        v = F.ecdf_percentile(x, [p])
        return float(v if np.isscalar(v) else v[0])

    return {
        "cid_ce_normalized": lambda x: fc.cid_ce(x, normalize=True),
        "agg_autocorrelation-median-10": lambda x: one(
            fc.agg_autocorrelation(x, [{"f_agg": "median", "maxlag": 10}])
        ),
        # numpy arrays have no .median(); tsfresh needs a Series for it.
        "agg_linear_trend-slope-5-median": lambda x: one(
            fc.agg_linear_trend(
                pd.Series(x), [{"attr": "slope", "chunk_len": 5, "f_agg": "median"}]
            )
        ),
        "agg_linear_trend-intercept-5-var": lambda x: one(
            fc.agg_linear_trend(x, [{"attr": "intercept", "chunk_len": 5, "f_agg": "var"}])
        ),
        "change_quantiles-0.2-0.8-False-median": lambda x: fc.change_quantiles(
            x, 0.2, 0.8, False, "median"
        ),
        # Alone, lag 0 is NaN in tsfresh; extract_features requests it with
        # other lags (0..9), which makes it 1.
        "partial_autocorr-0": lambda x: one(
            fc.partial_autocorrelation(x, [{"lag": 0}, {"lag": 1}])
        ),
        "ecdf_percentile-0.7": lambda x: ecdf_percentile(x, 0.7),
        "ecdf_percentile_count-0.7": lambda x: float(F.ecdf_percentile_count(x, [0.7])),
    }
