# Per-feature benchmark

Microseconds per window per series. Window 256, 20 series, 32 new values per sliding/expanding update. Commit a168948, 2026-10-07 09:20.
Sliding = per new window; expanding = per update. Reference = the tsfresh/TSFEL
function tsfast is tested against, called on one window; speed-up = reference / static.

| feature | static | sliding | expanding | reference | ref µs | speed-up |
|:--|--:|--:|--:|:--|--:|--:|
| `abs_max` | 1.37 | 0.78 | 5.74 | tsfresh | 6.3 | 4.6× |
| `abs_sum_change` | 1.77 | 0.79 | 5.27 | tsfresh | 12.2 | 6.9× |
| `agg_autocorrelation-max-5` | 2.68 | 1.51 | 10.23 | tsfresh | 104.9 | 39.2× |
| `agg_autocorrelation-mean-10` | 2.63 | 1.36 | 9.37 | tsfresh | 107.0 | 40.8× |
| `agg_autocorrelation-var-5` | 3.10 | 1.50 | 10.28 | tsfresh | 117.3 | 37.8× |
| `agg_linear_trend-intercept-5-max` | 3.96 | 1.44 | 5.85 | tsfresh | 716.3 | 180.9× |
| `agg_linear_trend-rvalue-5-var` | 3.46 | 2.05 | 13.42 | tsfresh | 3,080.4 | 891.3× |
| `agg_linear_trend-slope-10-mean` | 3.04 | 1.17 | 6.40 | tsfresh | 767.8 | 252.3× |
| `approx_entropy-2-0.1` | 163.96 | 112.36 | 146.93 | tsfresh | 14,773.4 | 90.1× |
| `approx_entropy-2-0.5` | 110.55 | 83.98 | 108.48 | tsfresh | 10,072.1 | 91.1× |
| `ar_coefficient-10-0` | 6.10 | 3.31 | 9.05 | tsfresh | 2,216.3 | 363.3× |
| `ar_coefficient-10-1` | 7.24 | 3.27 | 8.63 | tsfresh | 2,180.8 | 301.2× |
| `auc` | 1.97 | 0.84 | 5.11 | tsfel | 24.0 | 12.2× |
| `augmented_dickey_fuller-pvalue` | 165.75 | 120.31 | 147.70 | tsfresh | 8,215.9 | 49.6× |
| `augmented_dickey_fuller-teststat` | 174.10 | 127.66 | 148.61 | tsfresh | 8,045.5 | 46.2× |
| `augmented_dickey_fuller-usedlag` | 162.08 | 120.79 | 147.53 | tsfresh | 8,253.9 | 50.9× |
| `autocorr-1` | 2.84 | 1.26 | 10.26 | tsfresh | 74.8 | 26.3× |
| `autocorr-3` | 2.56 | 1.29 | 10.27 | tsfresh | 75.8 | 29.6× |
| `autocorr_lag1` | 1.84 | 0.86 | 5.55 | tsfresh | 73.7 | 40.1× |
| `autocorrelation` | 2.52 | 1.29 | 9.29 | tsfel | 83.3 | 33.0× |
| `benford_correlation` | 7.55 | 3.67 | 7.26 | tsfresh | 887.8 | 117.5× |
| `biased_fisher_kurtosis` | 1.97 | 1.13 | 6.82 | tsfel | 726.1 | 369.0× |
| `biased_skewness` | 1.93 | 1.14 | 7.07 | tsfel | 712.2 | 369.6× |
| `binned_entropy__max_bins_5` | 2.53 | 1.37 | 5.62 | tsfresh | 116.7 | 46.1× |
| `c3-1` | 3.37 | 0.84 | 6.45 | tsfresh | 18.1 | 5.4× |
| `c3-2` | 1.86 | 0.90 | 6.57 | tsfresh | 18.3 | 9.8× |
| `calc_centroid-100` | 1.93 | 0.74 | 6.15 | tsfel | 15.4 | 8.0× |
| `calc_centroid-50` | 2.75 | 0.76 | 5.94 | tsfel | 15.6 | 5.7× |
| `change_quantiles-0-1-False-var` | 3.20 | 2.07 | 7.05 | tsfresh | 1,144.9 | 358.2× |
| `change_quantiles-0.2-0.8-True-mean` | 3.11 | 1.91 | 7.39 | tsfresh | 1,114.3 | 358.7× |
| `cid_ce` | 1.85 | 0.74 | 5.11 | tsfresh | 7.9 | 4.3× |
| `count_above-0.5` | 1.48 | 0.74 | 5.45 | tsfresh | 8.0 | 5.4× |
| `count_above_mean` | 2.12 | 0.92 | 5.42 | tsfresh | 11.6 | 5.4× |
| `count_below-0.5` | 1.78 | 0.73 | 5.54 | tsfresh | 7.9 | 4.4× |
| `count_below_mean` | 2.22 | 0.94 | 5.27 | tsfresh | 11.8 | 5.3× |
| `cwt_coefficients__coeff_3__w_5__widths_(2, 5, 10, 20)` | 15.88 | 9.05 | 15.69 | tsfresh | 1,176.5 | 74.1× |
| `ecdf-10` | 1.61 | 0.71 | 5.35 | tsfel | 12.4 | 7.7× |
| `ecdf-3` | 1.47 | 0.74 | 5.93 | tsfel | 12.1 | 8.2× |
| `energy` | 2.22 | 0.75 | 5.33 | tsfresh | 1.9 | 0.9× |
| `energy_ratio_by_chunks_num_segments_3__segment_focus_1` | 2.21 | 0.83 | 5.99 | tsfresh | 33.6 | 15.2× |
| `energy_ratio_by_chunks_num_segments_4__segment_focus_3` | 1.94 | 0.83 | 5.49 | tsfresh | 35.9 | 18.5× |
| `entropy` | 9.90 | 5.35 | 11.35 | tsfel | 74.8 | 7.6× |
| `fft_coeff-0-real` | 2.53 | 1.17 | 9.91 | tsfresh | 21.0 | 8.3× |
| `fft_coeff-1-imag` | 3.98 | 1.17 | 10.23 | tsfresh | 21.4 | 5.4× |
| `fft_coeff-2-angle` | 3.09 | 1.15 | 10.92 | tsfresh | 24.9 | 8.1× |
| `fft_coeff-3-abs` | 4.81 | 1.94 | 16.18 | tsfresh | 55.2 | 11.5× |
| `first_loc_max` | 1.99 | 1.07 | 5.57 | tsfresh | 3.5 | 1.8× |
| `first_loc_min` | 2.41 | 1.02 | 5.36 | tsfresh | 3.5 | 1.5× |
| `friedrich_coefficients-3-30-0` | 8.96 | 4.39 | 10.72 | tsfresh | 6,983.9 | 779.1× |
| `friedrich_coefficients-3-30-3` | 9.14 | 4.41 | 11.07 | tsfresh | 7,447.9 | 814.9× |
| `has_duplicate` | 4.06 | 2.49 | 7.46 | tsfresh | 15.1 | 3.7× |
| `has_duplicate_max` | 2.21 | 0.82 | 5.89 | tsfresh | 12.3 | 5.6× |
| `has_duplicate_min` | 2.26 | 0.83 | 5.63 | tsfresh | 12.2 | 5.4× |
| `higuchi_fd` | 16.59 | 6.72 | 13.18 | tsfel | 6,980.7 | 420.8× |
| `human_range_energy-100` | 6.59 | 1.22 | 10.22 | tsfel | 47.5 | 7.2× |
| `index_mass_quantile-0.5` | 2.03 | 0.95 | 5.44 | tsfresh | 23.2 | 11.4× |
| `index_mass_quantile-0.9` | 2.15 | 1.01 | 5.60 | tsfresh | 22.7 | 10.6× |
| `intercept` | 1.82 | 0.88 | 6.07 | tsfresh | 565.6 | 310.3× |
| `iqr` | 4.98 | 2.64 | 9.27 | tsfel | 168.4 | 33.8× |
| `kurtosis` | 2.16 | 1.14 | 7.00 | tsfresh | 116.2 | 53.8× |
| `large_standard_deviation-0.05` | 1.78 | 0.76 | 5.29 | tsfresh | 32.1 | 18.1× |
| `last_loc_max` | 2.07 | 1.03 | 5.44 | tsfresh | 4.8 | 2.3× |
| `last_loc_min` | 2.00 | 1.01 | 5.59 | tsfresh | 4.8 | 2.4× |
| `length` | 1.35 | 0.73 | 5.47 | tsfresh | 0.3 | 0.2× |
| `linear_trend-intercept` | 2.17 | 0.72 | 5.25 | tsfresh | 572.9 | 264.4× |
| `linear_trend-pvalue` | 4.60 | 0.85 | 5.53 | tsfresh | 567.5 | 123.4× |
| `linear_trend-rvalue` | 2.18 | 0.76 | 6.54 | tsfresh | 575.7 | 263.6× |
| `linear_trend-slope` | 3.80 | 1.21 | 11.18 | tsfresh | 1,098.0 | 289.1× |
| `linear_trend-stderr` | 2.41 | 0.75 | 5.71 | tsfresh | 579.3 | 240.8× |
| `longest_strike_above_mean` | 2.10 | 0.90 | 5.37 | tsfresh | 558.4 | 266.4× |
| `longest_strike_below_mean` | 2.04 | 0.94 | 5.43 | tsfresh | 554.0 | 271.9× |
| `lpcc-0` | 6.55 | 3.56 | 13.88 | tsfel | 256.2 | 39.1× |
| `lpcc-3` | 6.62 | 3.59 | 14.46 | tsfel | 260.5 | 39.3× |
| `mad` | 2.04 | 1.14 | 7.24 | tsfel | 18.2 | 8.9× |
| `matrix_profile-10-max` | 678.41 | 519.05 | 685.50 | stumpy | 2,024.0 | 3.0× |
| `matrix_profile-10-mean` | 633.34 | 508.00 | 663.35 | stumpy | 2,000.4 | 3.2× |
| `matrix_profile-10-min` | 690.09 | 508.73 | 662.47 | stumpy | 2,014.1 | 2.9× |
| `max_langevin_fixed_point-3-30` | 9.51 | 4.66 | 11.93 | tsfresh | 8,041.0 | 845.5× |
| `max_power_spectrum` | 3.49 | 2.00 | 7.93 | tsfel | 547.2 | 157.0× |
| `max_value` | 2.13 | 0.97 | 7.41 | tsfresh | 4.9 | 2.3× |
| `mean` | 2.16 | 1.09 | 6.93 | tsfresh | 7.9 | 3.7× |
| `mean_abs_change` | 2.07 | 0.77 | 5.35 | tsfresh | 15.6 | 7.5× |
| `mean_change` | 1.52 | 0.80 | 5.45 | tsfresh | 0.9 | 0.6× |
| `mean_n_absolute_max-7` | 2.20 | 0.93 | 7.26 | tsfresh | 14.3 | 6.5× |
| `mean_second_derivative_central` | 1.39 | 0.75 | 6.00 | tsfresh | 1.3 | 0.9× |
| `median` | 4.26 | 1.46 | 9.35 | tsfresh | 33.0 | 7.8× |
| `median_abs_deviation` | 4.62 | 2.21 | 9.37 | tsfel | 597.3 | 129.3× |
| `median_abs_diff` | 2.40 | 1.17 | 5.95 | tsfel | 50.9 | 21.2× |
| `median_diff` | 2.32 | 1.13 | 5.79 | tsfel | 40.6 | 17.5× |
| `mfcc-0` | 10.62 | 4.97 | 14.35 | tsfel | 1,148.9 | 108.2× |
| `mfcc-3` | 10.75 | 4.96 | 13.28 | tsfel | 1,130.4 | 105.1× |
| `min_value` | 2.45 | 0.92 | 7.47 | tsfresh | 4.9 | 2.0× |
| `negative_turning` | 1.82 | 0.74 | 5.28 | tsfel | 17.6 | 9.7× |
| `number_crossing_m__m_0` | 1.38 | 0.76 | 7.11 | tsfresh | 8.6 | 6.2× |
| `number_crossing_m__m_0.5` | 1.71 | 0.78 | 5.48 | tsfresh | 8.0 | 4.7× |
| `number_cwt_peaks__n_1` | 22.24 | 11.13 | 28.70 | tsfresh | 5,624.9 | 252.9× |
| `number_cwt_peaks__n_5` | 50.09 | 31.78 | 51.83 | tsfresh | 7,375.5 | 147.3× |
| `number_peaks__n_1` | 4.74 | 0.99 | 5.39 | tsfresh | 16.3 | 3.4× |
| `number_peaks__n_3` | 2.38 | 1.09 | 5.35 | tsfresh | 34.3 | 14.4× |
| `paa-3-2` | 3.04 | 0.77 | 7.98 | — |  | |
| `paa-4-1` | 2.26 | 0.81 | 7.50 | — |  | |
| `partial_autocorr-1` | 2.90 | 1.52 | 11.10 | tsfresh | 88.9 | 30.6× |
| `partial_autocorr-2` | 2.60 | 1.56 | 13.65 | tsfresh | 213.9 | 82.4× |
| `peak_count` | 1.86 | 0.74 | 5.35 | tsfresh | 16.4 | 8.8× |
| `percentage_of_reoccurring_datapoints_to_all_datapoints` | 4.31 | 0.74 | 5.32 | tsfresh | 519.9 | 120.7× |
| `percentage_of_reoccurring_values_to_all_values` | 4.10 | 0.75 | 5.58 | tsfresh | 39.7 | 9.7× |
| `permutation_entropy-1-3` | 13.73 | 6.95 | 19.36 | tsfresh | 375.7 | 27.4× |
| `permutation_entropy-1-5` | 20.23 | 10.31 | 26.67 | tsfresh | 488.2 | 24.1× |
| `pk_pk_distance` | 1.87 | 0.75 | 5.68 | tsfel | 9.8 | 5.2× |
| `positive_turning` | 1.86 | 0.72 | 5.47 | tsfel | 22.1 | 11.9× |
| `quantile-0.25` | 2.60 | 1.66 | 6.07 | tsfresh | 79.7 | 30.7× |
| `quantile-0.9` | 2.68 | 1.59 | 6.25 | tsfresh | 90.7 | 33.8× |
| `query_similarity_count-10-0.5` | 9.65 | 4.55 | 8.84 | tsfresh | 1,089.6 | 112.9× |
| `range_count-0.5-0.5` | 1.73 | 0.76 | 5.81 | tsfresh | 9.8 | 5.7× |
| `ratio_beyond_r_sigma-1.5` | 2.43 | 0.80 | 5.44 | tsfresh | 41.5 | 17.1× |
| `ratio_value_number_to_time_series_length` | 4.10 | 0.78 | 5.68 | tsfresh | 15.2 | 3.7× |
| `rms` | 1.69 | 0.75 | 5.92 | tsfel | 8.4 | 5.0× |
| `root_mean_square` | 1.64 | 0.76 | 5.52 | tsfresh | 10.1 | 6.1× |
| `sample_entropy` | 37.61 | 22.15 | 39.27 | tsfresh | 15,492.2 | 411.9× |
| `signal_distance` | 1.73 | 0.88 | 5.80 | tsfel | 21.8 | 12.6× |
| `skewness` | 2.63 | 1.12 | 6.76 | tsfresh | 101.8 | 38.7× |
| `slope` | 1.58 | 0.73 | 5.99 | tsfel | 99.0 | 62.7× |
| `slope_sign_change` | 1.73 | 0.82 | 5.53 | — |  | |
| `spectral_centroid` | 2.38 | 1.21 | 10.42 | tsfel | 36.3 | 15.3× |
| `spectral_decrease` | 2.24 | 1.13 | 10.29 | tsfel | 45.8 | 20.4× |
| `spectral_distance` | 2.61 | 1.40 | 11.48 | tsfel | 51.0 | 19.5× |
| `spectral_entropy` | 2.87 | 1.54 | 10.15 | tsfel | 55.5 | 19.3× |
| `spectral_kurtosis` | 2.22 | 1.16 | 10.09 | tsfel | 216.6 | 97.8× |
| `spectral_roll_off` | 2.31 | 1.20 | 10.07 | tsfel | 39.4 | 17.0× |
| `spectral_roll_on` | 2.48 | 1.26 | 10.36 | tsfel | 37.7 | 15.2× |
| `spectral_skewness` | 2.32 | 1.17 | 10.18 | tsfel | 216.3 | 93.1× |
| `spectral_slope` | 2.25 | 1.15 | 9.82 | tsfel | 38.3 | 17.0× |
| `spectral_spread` | 2.34 | 1.22 | 9.75 | tsfel | 70.9 | 30.2× |
| `spectrogram-2-0.5` | 4.81 | 1.26 | 9.83 | — |  | |
| `spkt_welch_density__coeff_2` | 3.83 | 2.21 | 11.03 | tsfresh | 441.0 | 115.0× |
| `spkt_welch_density__coeff_5` | 3.82 | 2.22 | 13.96 | tsfresh | 475.1 | 124.4× |
| `std_dev` | 2.49 | 0.92 | 6.84 | tsfresh | 21.2 | 8.5× |
| `sum_of_reoccurring_data_points` | 4.30 | 2.69 | 8.20 | tsfresh | 45.6 | 10.6× |
| `sum_of_reoccurring_values` | 4.55 | 2.82 | 7.64 | tsfresh | 42.0 | 9.2× |
| `symmetry_looking-0.05` | 2.54 | 1.09 | 6.32 | tsfresh | 60.0 | 23.6× |
| `time_reversal_asymmetry-1` | 3.28 | 0.81 | 6.25 | tsfresh | 20.5 | 6.2× |
| `time_reversal_asymmetry-2` | 3.55 | 1.20 | 11.29 | tsfresh | 50.2 | 14.2× |
| `total_sum` | 1.81 | 0.99 | 7.00 | tsfresh | 5.1 | 2.8× |
| `turning_points` | 1.77 | 0.72 | 5.34 | — |  | |
| `value_count-0` | 2.63 | 0.77 | 5.80 | tsfresh | 3.8 | 1.5× |
| `value_count-1` | 1.59 | 0.74 | 6.09 | tsfresh | 3.8 | 2.4× |
| `variance` | 2.25 | 0.76 | 7.67 | tsfresh | 20.4 | 9.1× |
| `variance_larger_than_standard_deviation` | 2.38 | 0.73 | 5.55 | tsfresh | 20.3 | 8.5× |
| `variation_coefficient` | 1.58 | 0.72 | 5.20 | tsfresh | 28.6 | 18.1× |
| `wavelet-0.5-1` | 1.98 | 0.82 | 5.94 | — |  | |
| `wavelet_energy-0` | 101.90 | 73.31 | 92.52 | tsfel | 1,371.7 | 13.5× |
| `wavelet_energy-3` | 104.08 | 74.01 | 91.37 | tsfel | 1,394.1 | 13.4× |
| `wavelet_entropy` | 103.39 | 73.87 | 91.50 | tsfel | 1,324.4 | 12.8× |
| `zero_cross` | 2.24 | 0.74 | 5.61 | tsfel | 8.9 | 4.0× |
| `zero_crossing_mean` | 2.16 | 1.10 | 5.80 | — |  | |
| `zero_crossing_rate` | 1.88 | 0.78 | 5.77 | — |  | |
| `zero_crossing_std` | 2.26 | 1.16 | 5.56 | — |  | |
