"""Reference implementation for every feature in feature_samples.txt.

REFERENCES maps a tsfast feature name to (library, fn) where fn(x: float64
array) -> float. tsfast must be within 1% of it (tests/test_references.py).
When tsfresh and TSFEL define the same quantity differently, tsfast has one
feature per definition, each mapped to its own library.

NO_REFERENCE lists tsfast-specific features, with the reason. Every name in
feature_samples.txt must be in exactly one of the two.

Also used by scripts/coach_benchmark.py to time tsfast against the reference
libraries per feature.
"""

import numpy as np
import stumpy
import tsfel.feature_extraction.features as F
import tsfresh.feature_extraction.feature_calculators as fc

FS = 100.0  # tsfast's sampling frequency for TSFEL spectral features


def _matrix_profile(x, window):
    return stumpy.stump(x, window)[:, 0].astype(float)


def _permutation_entropy(x, tau, dimension):
    """tsfresh.permutation_entropy with ties broken by position. tsfresh uses
    np.argsort's default kind, which numpy 2 runs as an unstable SIMD sort, so
    its result on tied values depends on the CPU."""
    chunks = fc._into_subchunks(x, dimension, tau)
    ranks = np.argsort(np.argsort(chunks, kind="stable"), kind="stable")
    _, counts = np.unique(ranks, axis=0, return_counts=True)
    probs = counts / len(ranks)
    return -np.sum(probs * np.log(probs))


def _one(result):
    """tsfresh "combiner" calculators return [(name, value)]; take the value."""
    return list(result)[0][1]


REFERENCES = {
    # ── statistics ──
    "total_sum": ("tsfresh", fc.sum_values),
    "mean": ("tsfresh", fc.mean),
    "variance": ("tsfresh", fc.variance),
    "std_dev": ("tsfresh", fc.standard_deviation),
    "min_value": ("tsfresh", fc.minimum),
    "max_value": ("tsfresh", fc.maximum),
    "median": ("tsfresh", fc.median),
    "median_abs_deviation": ("tsfel", F.median_abs_deviation),
    "skewness": ("tsfresh", fc.skewness),
    "biased_skewness": ("tsfel", F.skewness),
    "kurtosis": ("tsfresh", fc.kurtosis),
    "biased_fisher_kurtosis": ("tsfel", F.kurtosis),
    "mad": ("tsfel", F.mean_abs_deviation),
    "iqr": ("tsfel", F.interq_range),
    "entropy": ("tsfel", F.entropy),
    "energy": ("tsfresh", fc.abs_energy),
    "rms": ("tsfel", F.rms),
    "root_mean_square": ("tsfresh", fc.root_mean_square),
    "abs_max": ("tsfresh", fc.absolute_maximum),
    "pk_pk_distance": ("tsfel", F.pk_pk_distance),
    "variation_coefficient": ("tsfresh", fc.variation_coefficient),
    "length": ("tsfresh", fc.length),
    "lempel_ziv": ("tsfel", F.lempel_ziv),
    "lempel_ziv_complexity-3": ("tsfresh", lambda x: fc.lempel_ziv_complexity(x, 3)),
    "variance_larger_than_standard_deviation": (
        "tsfresh",
        fc.variance_larger_than_standard_deviation,
    ),
    "large_standard_deviation-0.05": (
        "tsfresh",
        lambda x: fc.large_standard_deviation(x, 0.05),
    ),
    "symmetry_looking-0.05": (
        "tsfresh",
        lambda x: _one(fc.symmetry_looking(x, [{"r": 0.05}])),
    ),
    "ratio_beyond_r_sigma-1.5": ("tsfresh", lambda x: fc.ratio_beyond_r_sigma(x, 1.5)),
    "quantile-0.25": ("tsfresh", lambda x: fc.quantile(x, 0.25)),
    "quantile-0.9": ("tsfresh", lambda x: fc.quantile(x, 0.9)),
    "index_mass_quantile-0.5": (
        "tsfresh",
        lambda x: _one(fc.index_mass_quantile(x, [{"q": 0.5}])),
    ),
    "index_mass_quantile-0.9": (
        "tsfresh",
        lambda x: _one(fc.index_mass_quantile(x, [{"q": 0.9}])),
    ),
    "mean_n_absolute_max-7": ("tsfresh", lambda x: fc.mean_n_absolute_max(x, 7)),
    "count_above_mean": ("tsfresh", fc.count_above_mean),
    "count_below_mean": ("tsfresh", fc.count_below_mean),
    "count_above-0.5": ("tsfresh", lambda x: fc.count_above(x, 0.5)),
    "count_below-0.5": ("tsfresh", lambda x: fc.count_below(x, 0.5)),
    "range_count-0-5": ("tsfresh", lambda x: fc.range_count(x, 0, 5)),
    "longest_strike_above_mean": ("tsfresh", fc.longest_strike_above_mean),
    "longest_strike_below_mean": ("tsfresh", fc.longest_strike_below_mean),
    "first_loc_max": ("tsfresh", fc.first_location_of_maximum),
    "last_loc_max": ("tsfresh", fc.last_location_of_maximum),
    "first_loc_min": ("tsfresh", fc.first_location_of_minimum),
    "last_loc_min": ("tsfresh", fc.last_location_of_minimum),
    "has_duplicate": ("tsfresh", fc.has_duplicate),
    "has_duplicate_max": ("tsfresh", fc.has_duplicate_max),
    "has_duplicate_min": ("tsfresh", fc.has_duplicate_min),
    "benford_correlation": ("tsfresh", fc.benford_correlation),
    "sum_of_reoccurring_values": ("tsfresh", fc.sum_of_reoccurring_values),
    "sum_of_reoccurring_data_points": ("tsfresh", fc.sum_of_reoccurring_data_points),
    "percentage_of_reoccurring_datapoints_to_all_datapoints": (
        "tsfresh",
        fc.percentage_of_reoccurring_datapoints_to_all_datapoints,
    ),
    "percentage_of_reoccurring_values_to_all_values": (
        "tsfresh",
        fc.percentage_of_reoccurring_values_to_all_values,
    ),
    "ratio_value_number_to_time_series_length": (
        "tsfresh",
        fc.ratio_value_number_to_time_series_length,
    ),
    "value_count-1": ("tsfresh", lambda x: fc.value_count(x, 1)),
    "value_count-0": ("tsfresh", lambda x: fc.value_count(x, 0)),
    "ecdf-10": ("tsfel", lambda x: F.ecdf(x, 10)[-1]),
    "ecdf-3": ("tsfel", lambda x: F.ecdf(x, 3)[-1]),
    "ecdf_percentile-0.5": ("tsfel", lambda x: float(F.ecdf_percentile(x, [0.5])[0]) if not np.isscalar(F.ecdf_percentile(x, [0.5])) else float(F.ecdf_percentile(x, [0.5]))),
    "ecdf_percentile_count-0.5": ("tsfel", lambda x: float(len(x)) if np.max(x) == np.min(x) else float(np.sum(x <= (F.ecdf_percentile(x, [0.5])[0] if not np.isscalar(F.ecdf_percentile(x, [0.5])) else F.ecdf_percentile(x, [0.5]))))),
    "ecdf_slope-0.2-0.5": ("tsfel", lambda x: F.ecdf_slope(x, 0.2, 0.5)),
    "binned_entropy__max_bins_5": ("tsfresh", lambda x: fc.binned_entropy(x, 5)),
    # ── changes / temporal ──
    "mean_abs_change": ("tsfresh", fc.mean_abs_change),
    "mean_change": ("tsfresh", fc.mean_change),
    "median_diff": ("tsfel", F.median_diff),
    "median_abs_diff": ("tsfel", F.median_abs_diff),
    "abs_sum_change": ("tsfresh", fc.absolute_sum_of_changes),
    "cid_ce": ("tsfresh", lambda x: fc.cid_ce(x, normalize=False)),
    "mean_second_derivative_central": ("tsfresh", fc.mean_second_derivative_central),
    "auc": ("tsfel", lambda x: F.auc(x, FS)),
    "max_frequency": ("tsfel", lambda x: F.max_frequency(x, FS)),
    "median_frequency": ("tsfel", lambda x: F.median_frequency(x, FS)),
    "fundamental_frequency": ("tsfel", lambda x: F.fundamental_frequency(x, FS)),
    "signal_distance": ("tsfel", F.distance),
    "slope": ("tsfel", F.slope),
    "intercept": (
        "tsfresh",
        lambda x: _one(fc.linear_trend(x, [{"attr": "intercept"}])),
    ),
    "zero_cross": ("tsfel", F.zero_cross),
    "negative_turning": ("tsfel", F.negative_turning),
    "positive_turning": ("tsfel", F.positive_turning),
    "peak_count": ("tsfresh", lambda x: fc.number_peaks(x, 1)),
    "number_peaks__n_1": ("tsfresh", lambda x: fc.number_peaks(x, 1)),
    "number_peaks__n_3": ("tsfresh", lambda x: fc.number_peaks(x, 3)),
    "number_crossing_m__m_0": ("tsfresh", lambda x: fc.number_crossing_m(x, 0)),
    "number_crossing_m__m_0.5": ("tsfresh", lambda x: fc.number_crossing_m(x, 0.5)),
    "c3-1": ("tsfresh", lambda x: fc.c3(x, 1)),
    "c3-2": ("tsfresh", lambda x: fc.c3(x, 2)),
    "time_reversal_asymmetry-1": (
        "tsfresh",
        lambda x: fc.time_reversal_asymmetry_statistic(x, 1),
    ),
    "time_reversal_asymmetry-2": (
        "tsfresh",
        lambda x: fc.time_reversal_asymmetry_statistic(x, 2),
    ),
    "energy_ratio_by_chunks_num_segments_3__segment_focus_1": (
        "tsfresh",
        lambda x: _one(
            fc.energy_ratio_by_chunks(x, [{"num_segments": 3, "segment_focus": 1}])
        ),
    ),
    "energy_ratio_by_chunks_num_segments_4__segment_focus_3": (
        "tsfresh",
        lambda x: _one(
            fc.energy_ratio_by_chunks(x, [{"num_segments": 4, "segment_focus": 3}])
        ),
    ),
    # ── autocorrelation ──
    "autocorr_lag1": ("tsfresh", lambda x: fc.autocorrelation(x, 1)),
    "autocorr-1": ("tsfresh", lambda x: fc.autocorrelation(x, 1)),
    "autocorr-3": ("tsfresh", lambda x: fc.autocorrelation(x, 3)),
    "autocorrelation": ("tsfel", F.autocorr),
    "agg_autocorrelation-mean-10": (
        "tsfresh",
        lambda x: _one(fc.agg_autocorrelation(x, [{"f_agg": "mean", "maxlag": 10}])),
    ),
    "agg_autocorrelation-max-5": (
        "tsfresh",
        lambda x: _one(fc.agg_autocorrelation(x, [{"f_agg": "max", "maxlag": 5}])),
    ),
    "agg_autocorrelation-var-5": (
        "tsfresh",
        lambda x: _one(fc.agg_autocorrelation(x, [{"f_agg": "var", "maxlag": 5}])),
    ),
    "partial_autocorr-1": (
        "tsfresh",
        lambda x: _one(fc.partial_autocorrelation(x, [{"lag": 1}])),
    ),
    "partial_autocorr-2": (
        "tsfresh",
        lambda x: _one(fc.partial_autocorrelation(x, [{"lag": 2}])),
    ),
    # ── linear trend ──
    **{
        f"linear_trend-{a}": (
            "tsfresh",
            (lambda a: lambda x: _one(fc.linear_trend(x, [{"attr": a}])))(a),
        )
        for a in ["slope", "intercept", "rvalue", "pvalue", "stderr"]
    },
    "agg_linear_trend-intercept-5-max": (
        "tsfresh",
        lambda x: _one(
            fc.agg_linear_trend(
                x, [{"attr": "intercept", "chunk_len": 5, "f_agg": "max"}]
            )
        ),
    ),
    "agg_linear_trend-slope-10-mean": (
        "tsfresh",
        lambda x: _one(
            fc.agg_linear_trend(
                x, [{"attr": "slope", "chunk_len": 10, "f_agg": "mean"}]
            )
        ),
    ),
    "agg_linear_trend-rvalue-5-var": (
        "tsfresh",
        lambda x: _one(
            fc.agg_linear_trend(x, [{"attr": "rvalue", "chunk_len": 5, "f_agg": "var"}])
        ),
    ),
    "change_quantiles-0.2-0.8-True-mean": (
        "tsfresh",
        lambda x: fc.change_quantiles(x, 0.2, 0.8, True, "mean"),
    ),
    "change_quantiles-0-1-False-var": (
        "tsfresh",
        lambda x: fc.change_quantiles(x, 0.0, 1.0, False, "var"),
    ),
    # ── complexity / dynamics ──
    "sample_entropy": ("tsfel", lambda x: F.sample_entropy(x, 2, 0.2 * np.std(x))),
    "approx_entropy-2-0.1": ("tsfresh", lambda x: fc.approximate_entropy(x, 2, 0.1)),
    "approx_entropy-2-0.5": ("tsfresh", lambda x: fc.approximate_entropy(x, 2, 0.5)),
    "permutation_entropy-1-3": ("tsfresh", lambda x: _permutation_entropy(x, 1, 3)),
    "permutation_entropy-1-5": ("tsfresh", lambda x: _permutation_entropy(x, 1, 5)),
    "higuchi_fd": ("tsfel", F.higuchi_fractal_dimension),
    "ar_coefficient-10-1": (
        "tsfresh",
        lambda x: _one(fc.ar_coefficient(x, [{"coeff": 1, "k": 10}])),
    ),
    "ar_coefficient-10-0": (
        "tsfresh",
        lambda x: _one(fc.ar_coefficient(x, [{"coeff": 0, "k": 10}])),
    ),
    "friedrich_coefficients-3-30-0": (
        "tsfresh",
        lambda x: _one(fc.friedrich_coefficients(x, [{"m": 3, "r": 30, "coeff": 0}])),
    ),
    "friedrich_coefficients-3-30-3": (
        "tsfresh",
        lambda x: _one(fc.friedrich_coefficients(x, [{"m": 3, "r": 30, "coeff": 3}])),
    ),
    "max_langevin_fixed_point-3-30": (
        "tsfresh",
        lambda x: fc.max_langevin_fixed_point(x, m=3, r=30),
    ),
    **{
        f"augmented_dickey_fuller-{a}": (
            "tsfresh",
            (lambda a: lambda x: _one(fc.augmented_dickey_fuller(x, [{"attr": a}])))(a),
        )
        for a in ["teststat", "pvalue", "usedlag"]
    },
    "dfa": ("tsfel", lambda x: getattr(__import__("tsfel.feature_extraction.features", fromlist=["dfa"]), "dfa")(x)),
    "hurst_exponent": ("tsfel", lambda x: getattr(__import__("tsfel.feature_extraction.features", fromlist=["hurst_exponent"]), "hurst_exponent")(x)),
    # tsfresh's matrix_profile needs the unmaintained `matrixprofile` package and
    # picks its own window; stumpy (tsfresh's backend for query_similarity_count)
    # computes the same z-normalised profile for tsfast's fixed window.
    "matrix_profile-10-min": ("stumpy", lambda x: _matrix_profile(x, 10).min()),
    "matrix_profile-10-max": ("stumpy", lambda x: _matrix_profile(x, 10).max()),
    "matrix_profile-10-mean": ("stumpy", lambda x: _matrix_profile(x, 10).mean()),
    "query_similarity_count-10-0.5": (
        "tsfresh",
        lambda x: _one(
            fc.query_similarity_count(x, [{"query": x[:10], "threshold": 0.5}])
        ),
    ),
    # ── frequency domain ──
    "fft_coeff-3-abs": (
        "tsfresh",
        lambda x: _one(fc.fft_coefficient(x, [{"coeff": 3, "attr": "abs"}])),
    ),
    "fft_coeff-0-real": (
        "tsfresh",
        lambda x: _one(fc.fft_coefficient(x, [{"coeff": 0, "attr": "real"}])),
    ),
    "fft_coeff-1-imag": (
        "tsfresh",
        lambda x: _one(fc.fft_coefficient(x, [{"coeff": 1, "attr": "imag"}])),
    ),
    "fft_coeff-2-angle": (
        "tsfresh",
        lambda x: _one(fc.fft_coefficient(x, [{"coeff": 2, "attr": "angle"}])),
    ),
    "spkt_welch_density__coeff_2": (
        "tsfresh",
        lambda x: _one(fc.spkt_welch_density(x, [{"coeff": 2}])),
    ),
    "spkt_welch_density__coeff_5": (
        "tsfresh",
        lambda x: _one(fc.spkt_welch_density(x, [{"coeff": 5}])),
    ),
    "fourier_entropy-5": ("tsfresh", lambda x: fc.fourier_entropy(x, 5)),
    "fft_aggregated-centroid": (
        "tsfresh",
        lambda x: _one(fc.fft_aggregated(x, [{"aggtype": "centroid"}])),
    ),
    "fft_aggregated-variance": (
        "tsfresh",
        lambda x: _one(fc.fft_aggregated(x, [{"aggtype": "variance"}])),
    ),
    "fft_aggregated-skew": (
        "tsfresh",
        lambda x: _one(fc.fft_aggregated(x, [{"aggtype": "skew"}])),
    ),
    "fft_aggregated-kurtosis": (
        "tsfresh",
        lambda x: _one(fc.fft_aggregated(x, [{"aggtype": "kurtosis"}])),
    ),
    "cwt_coefficients__coeff_3__w_5__widths_(2, 5, 10, 20)": (
        "tsfresh",
        lambda x: _one(
            fc.cwt_coefficients(x, [{"widths": (2, 5, 10, 20), "coeff": 3, "w": 5}])
        ),
    ),
    "number_cwt_peaks__n_1": ("tsfresh", lambda x: fc.number_cwt_peaks(x, 1)),
    "number_cwt_peaks__n_5": ("tsfresh", lambda x: fc.number_cwt_peaks(x, 5)),
    **{
        f"spectral_{s}": (
            "tsfel",
            (lambda fn: lambda x: fn(x, FS))(getattr(F, f"spectral_{s}")),
        )
        for s in [
            "centroid",
            "distance",
            "decrease",
            "slope",
            "spread",
            "entropy",
            "roll_on",
            "roll_off",
            "skewness",
            "kurtosis",
        ]
    },
    "max_power_spectrum": ("tsfel", lambda x: F.max_power_spectrum(x, FS)),
    "spectral_positive_turning": (
        "tsfel",
        lambda x: F.spectral_positive_turning(x, FS),
    ),
    "spectral_variation": ("tsfel", lambda x: F.spectral_variation(x, FS)),
    "power_bandwidth": ("tsfel", lambda x: F.power_bandwidth(x, FS)),
    "human_range_energy-100": ("tsfel", lambda x: F.human_range_energy(x, 100.0)),
    "calc_centroid-100": ("tsfel", lambda x: F.calc_centroid(x, 100.0)),
    "calc_centroid-50": ("tsfel", lambda x: F.calc_centroid(x, 50.0)),
    "mfcc-0": ("tsfel", lambda x: F.mfcc(x, FS)[0]),
    "mfcc-3": ("tsfel", lambda x: F.mfcc(x, FS)[3]),
    "lpcc-0": ("tsfel", lambda x: F.lpcc(x)[0]),
    "lpcc-3": ("tsfel", lambda x: F.lpcc(x)[3]),
    "wavelet_energy-0": ("tsfel", lambda x: F.wavelet_energy(x, FS)["values"][0]),
    "wavelet_energy-3": ("tsfel", lambda x: F.wavelet_energy(x, FS)["values"][3]),
    "wavelet_entropy": ("tsfel", lambda x: F.wavelet_entropy(x, FS)),
    "wavelet_abs_mean-3": ("tsfel", lambda x: F.wavelet_abs_mean(x, FS)["values"][3]),
    "wavelet_std-3": ("tsfel", lambda x: F.wavelet_std(x, FS)["values"][3]),
    "wavelet_var-3": ("tsfel", lambda x: F.wavelet_var(x, FS)["values"][3]),
    "mse-3": ("tsfel", lambda x: F.mse(x, m=3, maxscale=None, tolerance=0.2 * np.std(x))),
    "mse-2-10": ("tsfel", lambda x: F.mse(x, m=2, maxscale=10, tolerance=0.2 * np.std(x))),
    "maximum_fractal_length": ("tsfel", F.maximum_fractal_length),
    "petrosian_fractal_dimension": ("tsfel", F.petrosian_fractal_dimension),
    "average_power-100": ("tsfel", lambda x: F.average_power(x, 100.0)),
    "hist_mode-10": ("tsfel", lambda x: F.hist_mode(x, 10)),
    "hist_mode-3": ("tsfel", lambda x: F.hist_mode(x, 3)),
    "neighbourhood_peaks-10": ("tsfel", lambda x: F.neighbourhood_peaks(x, 10)),
}

# Lengths below which the reference returns NaN by policy rather than because
# the feature is undefined; tsfast still returns a value there.
MIN_LENGTH = {
    "higuchi_fd": 160,  # TSFEL FEATURES_MIN_SIZE
    "dfa": 160,
    "hurst_exponent": 160,
    "maximum_fractal_length": 160,  # TSFEL FEATURES_MIN_SIZE
}

NO_REFERENCE = {
    "zero_crossing_rate": "zero_cross / n; TSFEL dropped its zero-crossing-rate feature",
    "zero_crossing_mean": "tsfast-specific: mean spacing between zero crossings",
    "zero_crossing_std": "tsfast-specific: std of spacing between zero crossings",
    "slope_sign_change": "EMG feature (threshold 0); in neither library",
    "turning_points": "local maxima + minima; in neither library",
    "paa-4-1": "piecewise aggregate approximation; in neither library",
    "paa-3-2": "piecewise aggregate approximation; in neither library",
    "wavelet-0.5-1": "not a wavelet transform: Haar-like pairwise differences, width ignored",
    "spectrogram-2-0.5": "single spectrum bin; time parameter ignored; matches no TSFEL spectrogram feature",
}
