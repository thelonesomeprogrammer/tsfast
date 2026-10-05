# Per-feature benchmark

Microseconds per window per series. Window 256, 20 series, 32 new values per sliding/expanding update. Commit 59748ce, 2026-10-05 20:20.
Sliding = per new window; expanding = per update. Reference = the tsfresh/TSFEL
function tsfast is tested against, called on one window; speed-up = reference / static.

| feature | static | sliding | expanding | reference | ref µs | speed-up |
|:--|--:|--:|--:|:--|--:|--:|
| `abs_max` | 0.55 | 0.38 | 1.61 | tsfresh | 3.0 | 5.5× |
| `abs_sum_change` | 0.76 | 0.43 | 1.17 | tsfresh | 5.4 | 7.1× |
| `agg_autocorrelation-max-5` | 1.58 | 0.86 | 2.18 | tsfresh | 48.8 | 30.9× |
| `agg_autocorrelation-mean-10` | 1.42 | 0.89 | 3.16 | tsfresh | 50.4 | 35.3× |
| `agg_autocorrelation-var-5` | 1.43 | 0.86 | 2.31 | tsfresh | 58.4 | 40.9× |
| `agg_linear_trend-intercept-5-max` | 1.19 | 0.58 | 1.42 | tsfresh | 311.2 | 262.1× |
| `agg_linear_trend-rvalue-5-var` | 1.09 | 0.59 | 1.43 | tsfresh | 728.1 | 665.3× |
| `agg_linear_trend-slope-10-mean` | 0.79 | 0.47 | 1.22 | tsfresh | 329.7 | 417.5× |
| `approx_entropy-2-0.1` | 42.45 | 33.63 | 54.93 | tsfresh | 5,562.2 | 131.0× |
| `approx_entropy-2-0.5` | 50.60 | 32.44 | 54.15 | tsfresh | 5,581.9 | 110.3× |
| `ar_coefficient-10-0` | 2.80 | 1.55 | 3.06 | tsfresh | 872.4 | 312.0× |
| `ar_coefficient-10-1` | 2.98 | 1.72 | 3.07 | tsfresh | 862.3 | 289.6× |
| `auc` | 0.73 | 0.48 | 1.21 | tsfel | 13.1 | 18.0× |
| `augmented_dickey_fuller-pvalue` | 74.26 | 50.65 | 64.76 | tsfresh | 3,631.5 | 48.9× |
| `augmented_dickey_fuller-teststat` | 68.24 | 44.40 | 60.23 | tsfresh | 3,592.7 | 52.6× |
| `augmented_dickey_fuller-usedlag` | 71.68 | 43.32 | 62.70 | tsfresh | 3,551.0 | 49.5× |
| `autocorr-1` | 1.38 | 0.88 | 2.42 | tsfresh | 37.8 | 27.4× |
| `autocorr-3` | 1.43 | 0.78 | 2.56 | tsfresh | 36.2 | 25.3× |
| `autocorr_lag1` | 0.77 | 0.46 | 1.33 | tsfresh | 44.5 | 58.0× |
| `autocorrelation` | 1.62 | 0.70 | 2.62 | tsfel | 36.5 | 22.6× |
| `benford_correlation` | 3.21 | 1.83 | 3.30 | tsfresh | 426.6 | 133.1× |
| `biased_fisher_kurtosis` | 0.56 | 0.41 | 1.45 | tsfel | 274.4 | 488.7× |
| `biased_skewness` | 0.58 | 0.41 | 1.66 | tsfel | 323.7 | 555.6× |
| `binned_entropy__max_bins_5` | 1.24 | 0.85 | 1.48 | tsfresh | 66.0 | 53.2× |
| `c3-1` | 0.73 | 0.44 | 1.58 | tsfresh | 8.1 | 11.1× |
| `c3-2` | 0.75 | 0.47 | 1.59 | tsfresh | 8.8 | 11.7× |
| `calc_centroid-100` | 0.59 | 0.41 | 1.35 | tsfel | 7.8 | 13.3× |
| `calc_centroid-50` | 0.58 | 0.41 | 1.44 | tsfel | 9.1 | 15.7× |
| `change_quantiles-0-1-False-var` | 1.97 | 1.14 | 2.64 | tsfresh | 431.5 | 219.4× |
| `change_quantiles-0.2-0.8-True-mean` | 1.59 | 1.36 | 2.06 | tsfresh | 414.1 | 259.9× |
| `cid_ce` | 0.77 | 0.42 | 1.13 | tsfresh | 4.0 | 5.2× |
| `count_above_mean` | 0.89 | 0.56 | 1.16 | tsfresh | 6.0 | 6.8× |
| `count_below_mean` | 0.93 | 0.52 | 1.36 | tsfresh | 5.4 | 5.8× |
| `cwt_coefficients__coeff_3__w_5__widths_(2, 5, 10, 20)` | 8.02 | 4.73 | 8.43 | tsfresh | 587.1 | 73.2× |
| `ecdf-10` | 0.52 | 0.40 | 1.15 | tsfel | 6.1 | 11.8× |
| `ecdf-3` | 0.51 | 0.40 | 1.12 | tsfel | 6.0 | 11.7× |
| `energy` | 0.55 | 0.41 | 1.35 | tsfresh | 1.1 | 2.0× |
| `energy_ratio_by_chunks_num_segments_3__segment_focus_1` | 0.78 | 0.48 | 1.29 | tsfresh | 13.7 | 17.5× |
| `energy_ratio_by_chunks_num_segments_4__segment_focus_3` | 0.75 | 0.47 | 1.20 | tsfresh | 16.1 | 21.4× |
| `entropy` | 2.54 | 1.95 | 3.29 | tsfel | 44.1 | 17.4× |
| `fft_coeff-0-real` | 1.23 | 0.78 | 2.87 | tsfresh | 7.9 | 6.5× |
| `fft_coeff-1-imag` | 1.25 | 0.72 | 3.12 | tsfresh | 7.9 | 6.3× |
| `fft_coeff-2-angle` | 1.23 | 0.78 | 2.65 | tsfresh | 9.7 | 7.9× |
| `fft_coeff-3-abs` | 1.16 | 0.76 | 2.51 | tsfresh | 8.9 | 7.7× |
| `first_loc_max` | 1.11 | 0.51 | 1.66 | tsfresh | 1.4 | 1.2× |
| `first_loc_min` | 1.05 | 0.61 | 1.31 | tsfresh | 1.7 | 1.6× |
| `friedrich_coefficients-3-30-0` | 4.00 | 3.04 | 4.35 | tsfresh | 2,744.1 | 685.8× |
| `friedrich_coefficients-3-30-3` | 4.35 | 2.61 | 4.36 | tsfresh | 2,760.7 | 634.4× |
| `has_duplicate` | 2.50 | 1.73 | 2.93 | tsfresh | 7.1 | 2.9× |
| `has_duplicate_max` | 0.73 | 0.49 | 1.36 | tsfresh | 6.7 | 9.2× |
| `has_duplicate_min` | 0.79 | 0.48 | 1.27 | tsfresh | 6.7 | 8.5× |
| `higuchi_fd` | 3.28 | 2.41 | 3.88 | tsfel | 4,478.6 | 1,365.1× |
| `human_range_energy-100` | 1.24 | 0.81 | 3.40 | tsfel | 23.5 | 18.9× |
| `index_mass_quantile-0.5` | 0.81 | 0.52 | 1.27 | tsfresh | 11.3 | 14.0× |
| `index_mass_quantile-0.9` | 0.89 | 0.54 | 1.61 | tsfresh | 11.6 | 13.1× |
| `intercept` | 0.60 | 0.42 | 1.09 | tsfresh | 218.3 | 366.5× |
| `iqr` | 1.59 | 1.12 | 2.22 | tsfel | 84.9 | 53.2× |
| `kurtosis` | 0.61 | 0.41 | 1.65 | tsfresh | 49.9 | 81.2× |
| `large_standard_deviation-0.05` | 0.62 | 0.41 | 1.25 | tsfresh | 16.5 | 26.4× |
| `last_loc_max` | 1.02 | 0.60 | 1.41 | tsfresh | 2.1 | 2.1× |
| `last_loc_min` | 1.06 | 0.62 | 1.19 | tsfresh | 2.1 | 2.0× |
| `length` | 0.50 | 0.41 | 1.11 | tsfresh | 0.1 | 0.3× |
| `linear_trend-intercept` | 0.64 | 0.41 | 1.07 | tsfresh | 236.8 | 372.7× |
| `linear_trend-pvalue` | 0.82 | 0.49 | 1.45 | tsfresh | 254.3 | 309.9× |
| `linear_trend-rvalue` | 0.63 | 0.42 | 1.10 | tsfresh | 214.0 | 338.7× |
| `linear_trend-slope` | 0.70 | 0.41 | 1.08 | tsfresh | 222.5 | 317.2× |
| `linear_trend-stderr` | 0.69 | 0.44 | 1.06 | tsfresh | 214.0 | 310.9× |
| `longest_strike_above_mean` | 0.83 | 0.55 | 1.33 | tsfresh | 437.4 | 525.5× |
| `longest_strike_below_mean` | 0.92 | 0.55 | 1.61 | tsfresh | 373.4 | 405.1× |
| `lpcc-0` | 5.05 | 3.68 | 5.76 | tsfel | 108.3 | 21.4× |
| `lpcc-3` | 6.23 | 3.17 | 6.41 | tsfel | 109.5 | 17.6× |
| `mad` | 0.57 | 0.41 | 1.61 | tsfel | 8.5 | 14.9× |
| `matrix_profile-10-max` | 349.38 | 239.77 | 328.71 | stumpy | 1,636.6 | 4.7× |
| `matrix_profile-10-mean` | 328.34 | 246.92 | 348.84 | stumpy | 1,547.0 | 4.7× |
| `matrix_profile-10-min` | 320.24 | 240.29 | 356.11 | stumpy | 1,646.9 | 5.1× |
| `max_langevin_fixed_point-3-30` | 4.38 | 3.09 | 4.70 | tsfresh | 2,837.7 | 647.8× |
| `max_power_spectrum` | 2.04 | 1.17 | 2.79 | tsfel | 219.8 | 107.5× |
| `max_value` | 0.54 | 0.40 | 1.43 | tsfresh | 2.3 | 4.3× |
| `mean` | 0.59 | 0.39 | 1.31 | tsfresh | 3.8 | 6.5× |
| `mean_abs_change` | 0.74 | 0.43 | 1.35 | tsfresh | 6.3 | 8.6× |
| `mean_change` | 0.73 | 0.43 | 1.60 | tsfresh | 0.5 | 0.7× |
| `mean_n_absolute_max-7` | 1.24 | 0.72 | 2.42 | tsfresh | 6.6 | 5.3× |
| `mean_second_derivative_central` | 0.51 | 0.40 | 1.26 | tsfresh | 0.8 | 1.5× |
| `median` | 1.15 | 0.82 | 2.05 | tsfresh | 13.4 | 11.7× |
| `median_abs_deviation` | 1.39 | 0.95 | 2.53 | tsfel | 238.3 | 171.8× |
| `median_abs_diff` | 1.41 | 0.82 | 1.47 | tsfel | 16.9 | 11.9× |
| `median_diff` | 1.42 | 0.72 | 1.58 | tsfel | 16.4 | 11.5× |
| `mfcc-0` | 5.64 | 3.28 | 5.89 | tsfel | 624.3 | 110.7× |
| `mfcc-3` | 5.20 | 2.94 | 5.98 | tsfel | 618.8 | 119.0× |
| `min_value` | 0.56 | 0.41 | 1.62 | tsfresh | 2.3 | 4.1× |
| `negative_turning` | 0.67 | 0.38 | 1.41 | tsfel | 10.7 | 16.0× |
| `number_crossing_m__m_0` | 0.55 | 0.42 | 1.17 | tsfresh | 4.2 | 7.6× |
| `number_crossing_m__m_0.5` | 0.55 | 0.41 | 1.29 | tsfresh | 4.0 | 7.2× |
| `number_cwt_peaks__n_1` | 10.25 | 4.16 | 9.81 | tsfresh | 2,491.5 | 243.1× |
| `number_cwt_peaks__n_5` | 29.49 | 16.74 | 29.20 | tsfresh | 3,298.8 | 111.9× |
| `number_peaks__n_1` | 0.87 | 0.57 | 1.32 | tsfresh | 8.0 | 9.2× |
| `number_peaks__n_3` | 1.10 | 0.59 | 1.80 | tsfresh | 19.1 | 17.3× |
| `paa-3-2` | 0.60 | 0.44 | 1.11 | — |  | |
| `paa-4-1` | 0.61 | 0.42 | 1.17 | — |  | |
| `partial_autocorr-1` | 1.54 | 0.85 | 1.94 | tsfresh | 38.6 | 25.0× |
| `partial_autocorr-2` | 1.55 | 0.86 | 2.32 | tsfresh | 44.3 | 28.6× |
| `peak_count` | 0.67 | 0.39 | 1.50 | tsfresh | 8.1 | 12.2× |
| `percentage_of_reoccurring_datapoints_to_all_datapoints` | 2.73 | 0.40 | 1.76 | tsfresh | 178.1 | 65.4× |
| `percentage_of_reoccurring_values_to_all_values` | 2.32 | 0.39 | 1.73 | tsfresh | 19.2 | 8.3× |
| `permutation_entropy-1-3` | 4.35 | 3.01 | 4.14 | tsfresh | 179.6 | 41.3× |
| `permutation_entropy-1-5` | 7.61 | 4.26 | 7.56 | tsfresh | 214.2 | 28.2× |
| `pk_pk_distance` | 0.57 | 0.42 | 1.27 | tsfel | 5.6 | 9.8× |
| `positive_turning` | 0.74 | 0.38 | 1.70 | tsfel | 10.5 | 14.2× |
| `quantile-0.25` | 1.33 | 0.98 | 2.01 | tsfresh | 44.8 | 33.6× |
| `quantile-0.9` | 1.49 | 1.12 | 2.04 | tsfresh | 41.1 | 27.6× |
| `query_similarity_count-10-0.5` | 3.53 | 2.54 | 4.39 | tsfresh | 618.2 | 175.4× |
| `ratio_beyond_r_sigma-1.5` | 0.65 | 0.43 | 1.29 | tsfresh | 21.3 | 33.0× |
| `ratio_value_number_to_time_series_length` | 2.34 | 0.38 | 1.63 | tsfresh | 7.2 | 3.1× |
| `rms` | 0.55 | 0.41 | 1.57 | tsfel | 3.9 | 7.1× |
| `root_mean_square` | 0.55 | 0.39 | 1.25 | tsfresh | 4.6 | 8.4× |
| `sample_entropy` | 11.18 | 6.26 | 12.28 | tsfresh | 8,376.8 | 749.4× |
| `signal_distance` | 0.84 | 0.54 | 1.36 | tsfel | 10.1 | 12.0× |
| `skewness` | 0.59 | 0.40 | 1.77 | tsfresh | 44.4 | 75.7× |
| `slope` | 0.60 | 0.41 | 1.38 | tsfel | 48.9 | 81.5× |
| `slope_sign_change` | 0.67 | 0.45 | 1.30 | — |  | |
| `spectral_centroid` | 1.39 | 0.78 | 2.44 | tsfel | 17.1 | 12.3× |
| `spectral_decrease` | 1.15 | 0.74 | 2.26 | tsfel | 22.8 | 19.8× |
| `spectral_distance` | 1.55 | 0.98 | 2.58 | tsfel | 26.8 | 17.3× |
| `spectral_entropy` | 1.65 | 0.91 | 2.58 | tsfel | 28.2 | 17.1× |
| `spectral_kurtosis` | 1.34 | 0.77 | 2.27 | tsfel | 110.3 | 82.1× |
| `spectral_roll_off` | 1.16 | 0.76 | 2.63 | tsfel | 17.9 | 15.5× |
| `spectral_roll_on` | 1.21 | 0.69 | 2.67 | tsfel | 17.8 | 14.7× |
| `spectral_skewness` | 1.21 | 0.77 | 2.37 | tsfel | 112.9 | 93.3× |
| `spectral_slope` | 1.19 | 0.77 | 2.21 | tsfel | 20.0 | 16.9× |
| `spectral_spread` | 1.29 | 0.78 | 2.39 | tsfel | 37.1 | 28.7× |
| `spectrogram-2-0.5` | 1.42 | 0.79 | 2.39 | — |  | |
| `spkt_welch_density__coeff_2` | 2.32 | 1.52 | 2.83 | tsfresh | 177.7 | 76.8× |
| `spkt_welch_density__coeff_5` | 2.16 | 1.32 | 3.88 | tsfresh | 178.5 | 82.7× |
| `std_dev` | 0.58 | 0.41 | 1.54 | tsfresh | 11.0 | 19.1× |
| `sum_of_reoccurring_data_points` | 2.61 | 1.76 | 3.27 | tsfresh | 18.8 | 7.2× |
| `sum_of_reoccurring_values` | 2.58 | 1.77 | 3.13 | tsfresh | 20.0 | 7.8× |
| `symmetry_looking-0.05` | 1.25 | 0.84 | 1.83 | tsfresh | 29.1 | 23.2× |
| `time_reversal_asymmetry-1` | 0.60 | 0.47 | 1.15 | tsfresh | 9.9 | 16.4× |
| `time_reversal_asymmetry-2` | 0.62 | 0.44 | 1.18 | tsfresh | 10.1 | 16.2× |
| `total_sum` | 0.53 | 0.43 | 1.25 | tsfresh | 2.7 | 5.1× |
| `turning_points` | 0.82 | 0.40 | 1.54 | — |  | |
| `value_count-0` | 0.56 | 0.44 | 1.18 | tsfresh | 2.4 | 4.2× |
| `value_count-1` | 0.61 | 0.43 | 1.46 | tsfresh | 2.6 | 4.3× |
| `variance` | 0.62 | 0.38 | 1.58 | tsfresh | 11.0 | 17.9× |
| `variance_larger_than_standard_deviation` | 0.57 | 0.40 | 1.09 | tsfresh | 10.6 | 18.7× |
| `variation_coefficient` | 0.59 | 0.41 | 1.45 | tsfresh | 16.8 | 28.6× |
| `wavelet-0.5-1` | 0.62 | 0.44 | 1.84 | — |  | |
| `wavelet_energy-0` | 54.99 | 34.79 | 53.86 | tsfel | 690.8 | 12.6× |
| `wavelet_energy-3` | 58.06 | 37.72 | 47.76 | tsfel | 691.6 | 11.9× |
| `wavelet_entropy` | 54.36 | 40.22 | 48.81 | tsfel | 679.9 | 12.5× |
| `zero_cross` | 0.98 | 0.41 | 1.70 | tsfel | 4.4 | 4.5× |
| `zero_crossing_mean` | 1.11 | 0.52 | 1.75 | — |  | |
| `zero_crossing_rate` | 0.89 | 0.40 | 1.20 | — |  | |
| `zero_crossing_std` | 1.30 | 0.54 | 1.81 | — |  | |
