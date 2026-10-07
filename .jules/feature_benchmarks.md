# Per-feature benchmark

Microseconds per window per series. Window 256, 20 series, 32 new values per sliding/expanding update. Commit 851a2f0, 2026-10-07 11:34.
Sliding = per new window; expanding = per update. Reference = the tsfresh/TSFEL
function tsfast is tested against, called on one window; speed-up = reference / static.

| feature | static | sliding | expanding | reference | ref µs | speed-up |
|:--|--:|--:|--:|:--|--:|--:|
| `abs_max` | 1.92 | 0.58 | 2.58 | tsfresh | 5.7 | 3.0× |
| `abs_sum_change` | 1.98 | 0.62 | 2.52 | tsfresh | 10.3 | 5.2× |
| `agg_autocorrelation-max-5` | 2.97 | 1.07 | 3.70 | tsfresh | 116.2 | 39.1× |
| `agg_autocorrelation-mean-10` | 2.97 | 1.07 | 3.63 | tsfresh | 111.9 | 37.6× |
| `agg_autocorrelation-var-5` | 3.00 | 1.07 | 3.68 | tsfresh | 130.7 | 43.6× |
| `agg_linear_trend-intercept-5-max` | 2.45 | 0.78 | 2.66 | tsfresh | 693.0 | 283.3× |
| `agg_linear_trend-rvalue-5-var` | 2.51 | 0.78 | 2.61 | tsfresh | 1,425.2 | 568.3× |
| `agg_linear_trend-slope-10-mean` | 2.21 | 0.64 | 2.54 | tsfresh | 736.4 | 333.5× |
| `approx_entropy-2-0.1` | 95.08 | 64.16 | 86.63 | tsfresh | 10,925.7 | 114.9× |
| `approx_entropy-2-0.5` | 93.30 | 64.21 | 86.49 | tsfresh | 11,146.0 | 119.5× |
| `ar_coefficient-10-0` | 5.58 | 2.60 | 4.86 | tsfresh | 2,248.4 | 402.6× |
| `ar_coefficient-10-1` | 5.60 | 2.63 | 4.97 | tsfresh | 2,268.4 | 405.4× |
| `auc` | 2.03 | 0.62 | 2.58 | tsfel | 22.3 | 11.0× |
| `augmented_dickey_fuller-pvalue` | 121.66 | 84.32 | 107.12 | tsfresh | 8,904.3 | 73.2× |
| `augmented_dickey_fuller-teststat` | 121.98 | 83.99 | 107.32 | tsfresh | 8,429.2 | 69.1× |
| `augmented_dickey_fuller-usedlag` | 122.13 | 84.44 | 107.36 | tsfresh | 8,457.3 | 69.2× |
| `autocorr-1` | 2.99 | 1.05 | 3.67 | tsfresh | 76.6 | 25.7× |
| `autocorr-3` | 2.98 | 1.06 | 3.60 | tsfresh | 75.6 | 25.4× |
| `autocorr_lag1` | 2.10 | 0.63 | 2.58 | tsfresh | 73.6 | 35.0× |
| `autocorrelation` | 2.97 | 1.07 | 3.59 | tsfel | 88.8 | 29.9× |
| `benford_correlation` | 5.19 | 2.35 | 4.73 | tsfresh | 819.7 | 158.0× |
| `biased_fisher_kurtosis` | 2.02 | 0.57 | 2.57 | tsfel | 769.3 | 381.1× |
| `biased_skewness` | 2.03 | 0.59 | 2.57 | tsfel | 773.1 | 380.9× |
| `binned_entropy__max_bins_5` | 2.79 | 1.10 | 3.01 | tsfresh | 131.8 | 47.3× |
| `c3-1` | 1.99 | 0.64 | 2.45 | tsfresh | 15.3 | 7.7× |
| `c3-2` | 2.02 | 0.65 | 2.54 | tsfresh | 15.2 | 7.5× |
| `calc_centroid-100` | 1.90 | 0.58 | 2.54 | tsfel | 15.4 | 8.1× |
| `calc_centroid-50` | 1.96 | 0.59 | 2.45 | tsfel | 15.5 | 7.9× |
| `change_quantiles-0-1-False-var` | 3.33 | 1.51 | 3.35 | tsfresh | 1,101.7 | 331.1× |
| `change_quantiles-0.2-0.8-True-mean` | 3.27 | 1.46 | 3.41 | tsfresh | 1,052.2 | 321.3× |
| `cid_ce` | 2.01 | 0.61 | 2.51 | tsfresh | 6.3 | 3.1× |
| `count_above-0.5` | 2.04 | 0.56 | 2.56 | tsfresh | 6.8 | 3.3× |
| `count_above_mean` | 2.20 | 0.72 | 2.54 | tsfresh | 9.6 | 4.3× |
| `count_below-0.5` | 1.95 | 0.57 | 2.51 | tsfresh | 7.3 | 3.7× |
| `count_below_mean` | 2.23 | 0.73 | 2.52 | tsfresh | 9.9 | 4.4× |
| `cwt_coefficients__coeff_3__w_5__widths_(2, 5, 10, 20)` | 14.83 | 8.10 | 11.60 | tsfresh | 1,336.9 | 90.1× |
| `dfa` | 5.77 | 2.25 | 5.37 | tsfel | 59,536.6 | 10,316.5× |
| `ecdf-10` | 2.29 | 0.60 | 2.52 | tsfel | 12.9 | 5.6× |
| `ecdf-3` | 1.89 | 0.60 | 2.49 | tsfel | 12.4 | 6.6× |
| `ecdf_percentile-0.5` | 2.36 | 0.85 | 2.83 | tsfel | 69.3 | 29.4× |
| `ecdf_percentile_count-0.5` | 2.41 | 0.88 | 2.90 | tsfel | 86.9 | 36.1× |
| `ecdf_slope-0.2-0.5` | 2.55 | 0.99 | 2.96 | tsfel | 49.0 | 19.2× |
| `energy` | 1.96 | 0.58 | 2.50 | tsfresh | 1.6 | 0.8× |
| `energy_ratio_by_chunks_num_segments_3__segment_focus_1` | 2.04 | 0.64 | 2.54 | tsfresh | 29.8 | 14.6× |
| `energy_ratio_by_chunks_num_segments_4__segment_focus_3` | 2.05 | 0.64 | 2.57 | tsfresh | 30.4 | 14.9× |
| `entropy` | 4.81 | 2.35 | 4.79 | tsfel | 77.7 | 16.1× |
| `fft_coeff-0-real` | 2.36 | 0.87 | 4.32 | tsfresh | 19.9 | 8.4× |
| `fft_coeff-1-imag` | 2.47 | 0.86 | 4.36 | tsfresh | 19.2 | 7.8× |
| `fft_coeff-2-angle` | 2.50 | 0.85 | 4.39 | tsfresh | 23.6 | 9.4× |
| `fft_coeff-3-abs` | 2.49 | 0.86 | 4.48 | tsfresh | 21.8 | 8.8× |
| `first_loc_max` | 2.29 | 0.81 | 2.72 | tsfresh | 2.4 | 1.1× |
| `first_loc_min` | 2.25 | 0.81 | 2.52 | tsfresh | 2.4 | 1.1× |
| `friedrich_coefficients-3-30-0` | 6.96 | 3.19 | 5.93 | tsfresh | 7,237.7 | 1,039.4× |
| `friedrich_coefficients-3-30-3` | 6.66 | 3.18 | 6.04 | tsfresh | 7,588.2 | 1,140.1× |
| `fundamental_frequency` | 2.51 | 0.87 | 4.52 | tsfel | 122.4 | 48.7× |
| `has_duplicate` | 3.87 | 1.70 | 3.78 | tsfresh | 15.3 | 4.0× |
| `has_duplicate_max` | 2.02 | 0.66 | 2.53 | tsfresh | 10.9 | 5.4× |
| `has_duplicate_min` | 2.04 | 0.66 | 2.54 | tsfresh | 10.8 | 5.3× |
| `higuchi_fd` | 7.21 | 3.26 | 6.18 | tsfel | 6,286.3 | 872.1× |
| `human_range_energy-100` | 2.56 | 0.91 | 4.34 | tsfel | 50.8 | 19.8× |
| `hurst_exponent` | 9.69 | 4.20 | 7.95 | tsfel | 1,961.3 | 202.4× |
| `index_mass_quantile-0.5` | 2.05 | 0.67 | 2.58 | tsfresh | 21.9 | 10.7× |
| `index_mass_quantile-0.9` | 2.09 | 0.69 | 2.55 | tsfresh | 22.0 | 10.5× |
| `intercept` | 1.97 | 0.58 | 2.54 | tsfresh | 577.9 | 294.1× |
| `iqr` | 3.09 | 1.29 | 2.99 | tsfel | 173.3 | 56.2× |
| `kurtosis` | 2.07 | 0.58 | 2.56 | tsfresh | 108.0 | 52.3× |
| `large_standard_deviation-0.05` | 2.15 | 0.58 | 2.49 | tsfresh | 32.2 | 15.0× |
| `last_loc_max` | 2.27 | 0.81 | 2.57 | tsfresh | 3.5 | 1.5× |
| `last_loc_min` | 2.29 | 0.80 | 2.54 | tsfresh | 3.6 | 1.6× |
| `lempel_ziv` | 3.17 | 1.21 | 3.33 | tsfel | 115.3 | 36.4× |
| `lempel_ziv_complexity-3` | 5.75 | 2.62 | 6.57 | tsfresh | 463.7 | 80.6× |
| `length` | 1.96 | 0.58 | 2.54 | tsfresh | 0.2 | 0.1× |
| `linear_trend-intercept` | 1.98 | 0.60 | 2.54 | tsfresh | 571.6 | 289.1× |
| `linear_trend-pvalue` | 2.11 | 0.64 | 2.59 | tsfresh | 583.5 | 277.2× |
| `linear_trend-rvalue` | 2.02 | 0.61 | 2.56 | tsfresh | 580.3 | 287.8× |
| `linear_trend-slope` | 1.99 | 0.60 | 2.54 | tsfresh | 595.8 | 299.4× |
| `linear_trend-stderr` | 1.97 | 0.58 | 2.49 | tsfresh | 570.9 | 289.8× |
| `longest_strike_above_mean` | 2.16 | 0.73 | 2.64 | tsfresh | 520.3 | 240.9× |
| `longest_strike_below_mean` | 2.16 | 0.72 | 2.70 | tsfresh | 517.4 | 239.4× |
| `lpcc-0` | 5.44 | 2.25 | 6.24 | tsfel | 288.7 | 53.1× |
| `lpcc-3` | 5.44 | 2.24 | 5.91 | tsfel | 291.3 | 53.5× |
| `mad` | 2.04 | 0.58 | 2.59 | tsfel | 15.0 | 7.3× |
| `matrix_profile-10-max` | 505.80 | 381.38 | 498.43 | stumpy | 6,056.8 | 12.0× |
| `matrix_profile-10-mean` | 507.47 | 374.25 | 498.39 | stumpy | 6,013.8 | 11.9× |
| `matrix_profile-10-min` | 504.82 | 383.99 | 498.56 | stumpy | 6,363.2 | 12.6× |
| `max_frequency` | 2.71 | 0.96 | 4.47 | tsfel | 32.0 | 11.8× |
| `max_langevin_fixed_point-3-30` | 7.17 | 3.34 | 6.17 | tsfresh | 7,308.3 | 1,019.1× |
| `max_power_spectrum` | 3.50 | 1.45 | 3.78 | tsfel | 498.3 | 142.3× |
| `max_value` | 1.98 | 0.57 | 2.54 | tsfresh | 4.1 | 2.0× |
| `maximum_fractal_length` | 7.54 | 3.41 | 6.45 | tsfel | 6,409.5 | 850.1× |
| `mean` | 2.07 | 0.60 | 2.56 | tsfresh | 6.2 | 3.0× |
| `mean_abs_change` | 2.02 | 0.62 | 2.50 | tsfresh | 12.8 | 6.3× |
| `mean_change` | 1.99 | 0.60 | 2.61 | tsfresh | 0.8 | 0.4× |
| `mean_n_absolute_max-7` | 2.23 | 0.77 | 3.17 | tsfresh | 14.5 | 6.5× |
| `mean_second_derivative_central` | 1.96 | 0.57 | 2.50 | tsfresh | 1.2 | 0.6× |
| `median` | 2.35 | 0.83 | 2.93 | tsfresh | 31.7 | 13.5× |
| `median_abs_deviation` | 2.74 | 1.09 | 3.33 | tsfel | 589.8 | 215.1× |
| `median_abs_diff` | 2.45 | 0.90 | 2.95 | tsfel | 39.5 | 16.1× |
| `median_diff` | 2.38 | 0.91 | 2.91 | tsfel | 37.9 | 15.9× |
| `median_frequency` | 2.65 | 0.96 | 4.60 | tsfel | 32.6 | 12.3× |
| `mfcc-0` | 7.83 | 3.44 | 7.64 | tsfel | 1,039.2 | 132.8× |
| `mfcc-3` | 7.71 | 3.44 | 7.31 | tsfel | 1,048.1 | 136.0× |
| `min_value` | 1.99 | 0.58 | 2.53 | tsfresh | 4.0 | 2.0× |
| `mse-2-10` | 52.04 | 33.07 | 47.22 | tsfel | 9,654.1 | 185.5× |
| `mse-3` | 62.56 | 40.30 | 56.15 | tsfel | 14,600.0 | 233.4× |
| `negative_turning` | 2.09 | 0.58 | 2.53 | tsfel | 15.7 | 7.5× |
| `number_crossing_m__m_0` | 1.97 | 0.58 | 2.44 | tsfresh | 6.9 | 3.5× |
| `number_crossing_m__m_0.5` | 1.94 | 0.60 | 2.54 | tsfresh | 7.2 | 3.7× |
| `number_cwt_peaks__n_1` | 14.50 | 6.12 | 11.54 | tsfresh | 4,961.8 | 342.2× |
| `number_cwt_peaks__n_5` | 41.28 | 25.57 | 34.76 | tsfresh | 6,210.4 | 150.4× |
| `number_peaks__n_1` | 2.21 | 0.76 | 2.69 | tsfresh | 15.4 | 7.0× |
| `number_peaks__n_3` | 2.33 | 0.80 | 2.82 | tsfresh | 31.7 | 13.6× |
| `paa-3-2` | 1.98 | 0.59 | 2.42 | — |  | |
| `paa-4-1` | 2.00 | 0.58 | 2.54 | — |  | |
| `partial_autocorr-1` | 3.05 | 1.08 | 3.79 | tsfresh | 110.5 | 36.3× |
| `partial_autocorr-2` | 3.00 | 1.07 | 3.64 | tsfresh | 107.3 | 35.8× |
| `peak_count` | 2.02 | 0.57 | 2.52 | tsfresh | 14.8 | 7.3× |
| `percentage_of_reoccurring_datapoints_to_all_datapoints` | 3.86 | 0.58 | 2.68 | tsfresh | 489.5 | 126.8× |
| `percentage_of_reoccurring_values_to_all_values` | 3.81 | 0.59 | 2.71 | tsfresh | 45.0 | 11.8× |
| `permutation_entropy-1-3` | 6.95 | 2.94 | 5.59 | tsfresh | 342.8 | 49.3× |
| `permutation_entropy-1-5` | 13.29 | 5.24 | 9.80 | tsfresh | 452.5 | 34.0× |
| `pk_pk_distance` | 1.98 | 0.57 | 2.56 | tsfel | 8.5 | 4.3× |
| `positive_turning` | 2.06 | 0.56 | 2.53 | tsfel | 15.6 | 7.6× |
| `quantile-0.25` | 2.87 | 1.26 | 2.90 | tsfresh | 83.1 | 29.0× |
| `quantile-0.9` | 2.79 | 1.23 | 2.95 | tsfresh | 84.2 | 30.2× |
| `query_similarity_count-10-0.5` | 6.51 | 3.00 | 5.64 | tsfresh | 1,286.7 | 197.6× |
| `range_count-0-5` | 2.00 | 0.59 | 2.47 | tsfresh | 9.8 | 4.9× |
| `ratio_beyond_r_sigma-1.5` | 2.01 | 0.60 | 2.50 | tsfresh | 40.3 | 20.1× |
| `ratio_value_number_to_time_series_length` | 3.83 | 0.58 | 2.71 | tsfresh | 15.1 | 3.9× |
| `rms` | 1.95 | 0.59 | 2.50 | tsfel | 6.9 | 3.6× |
| `root_mean_square` | 1.99 | 0.59 | 2.48 | tsfresh | 8.0 | 4.0× |
| `sample_entropy` | 34.29 | 21.19 | 31.19 | tsfel | 1,566.9 | 45.7× |
| `signal_distance` | 2.08 | 0.64 | 2.66 | tsfel | 20.2 | 9.7× |
| `skewness` | 2.01 | 0.59 | 2.60 | tsfresh | 99.3 | 49.4× |
| `slope` | 1.98 | 0.58 | 2.60 | tsfel | 109.1 | 55.2× |
| `slope_sign_change` | 2.02 | 0.63 | 2.60 | — |  | |
| `spectral_centroid` | 2.47 | 0.86 | 4.37 | tsfel | 38.2 | 15.5× |
| `spectral_decrease` | 2.71 | 0.97 | 4.40 | tsfel | 47.6 | 17.6× |
| `spectral_distance` | 2.69 | 0.97 | 4.50 | tsfel | 62.0 | 23.0× |
| `spectral_entropy` | 2.99 | 1.13 | 4.76 | tsfel | 61.0 | 20.4× |
| `spectral_kurtosis` | 2.56 | 0.86 | 4.42 | tsfel | 231.2 | 90.2× |
| `spectral_roll_off` | 2.67 | 0.98 | 4.36 | tsfel | 44.2 | 16.6× |
| `spectral_roll_on` | 2.70 | 0.96 | 4.40 | tsfel | 42.6 | 15.8× |
| `spectral_skewness` | 2.55 | 0.86 | 4.27 | tsfel | 222.9 | 87.6× |
| `spectral_slope` | 2.73 | 0.96 | 4.38 | tsfel | 41.7 | 15.3× |
| `spectral_spread` | 2.60 | 0.90 | 4.33 | tsfel | 76.1 | 29.3× |
| `spectrogram-2-0.5` | 2.52 | 0.87 | 4.36 | — |  | |
| `spkt_welch_density__coeff_2` | 3.83 | 1.53 | 5.01 | tsfresh | 446.9 | 116.7× |
| `spkt_welch_density__coeff_5` | 3.83 | 1.53 | 5.16 | tsfresh | 450.1 | 117.6× |
| `std_dev` | 2.01 | 0.58 | 2.54 | tsfresh | 20.1 | 10.0× |
| `sum_of_reoccurring_data_points` | 3.95 | 1.77 | 3.93 | tsfresh | 43.0 | 10.9× |
| `sum_of_reoccurring_values` | 3.99 | 1.79 | 3.92 | tsfresh | 45.1 | 11.3× |
| `symmetry_looking-0.05` | 2.37 | 0.83 | 2.95 | tsfresh | 59.9 | 25.3× |
| `time_reversal_asymmetry-1` | 1.96 | 0.65 | 2.46 | tsfresh | 20.5 | 10.5× |
| `time_reversal_asymmetry-2` | 1.99 | 0.66 | 2.57 | tsfresh | 18.8 | 9.4× |
| `total_sum` | 2.03 | 0.61 | 2.55 | tsfresh | 4.1 | 2.0× |
| `turning_points` | 2.17 | 0.59 | 2.55 | — |  | |
| `value_count-0` | 1.92 | 0.58 | 2.50 | tsfresh | 3.4 | 1.8× |
| `value_count-1` | 2.14 | 0.60 | 2.51 | tsfresh | 3.4 | 1.6× |
| `variance` | 2.01 | 0.59 | 2.56 | tsfresh | 18.9 | 9.4× |
| `variance_larger_than_standard_deviation` | 2.00 | 0.57 | 2.51 | tsfresh | 19.2 | 9.6× |
| `variation_coefficient` | 1.96 | 0.58 | 2.51 | tsfresh | 26.7 | 13.6× |
| `wavelet-0.5-1` | 2.04 | 0.63 | 2.58 | — |  | |
| `wavelet_energy-0` | 92.68 | 64.60 | 78.38 | tsfel | 1,476.8 | 15.9× |
| `wavelet_energy-3` | 94.87 | 64.10 | 78.18 | tsfel | 1,514.6 | 16.0× |
| `wavelet_entropy` | 93.19 | 64.03 | 78.23 | tsfel | 1,446.3 | 15.5× |
| `zero_cross` | 1.99 | 0.60 | 2.63 | tsfel | 7.6 | 3.8× |
| `zero_crossing_mean` | 2.43 | 0.83 | 2.70 | — |  | |
| `zero_crossing_rate` | 2.08 | 0.61 | 2.51 | — |  | |
| `zero_crossing_std` | 2.47 | 0.85 | 2.67 | — |  | |
