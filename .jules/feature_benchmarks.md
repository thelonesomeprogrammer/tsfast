# Per-feature benchmark

Microseconds per window per series. Window 256, 20 series, 32 new values per sliding/expanding update. Commit 83a16cf, 2026-10-06 17:08.
Sliding = per new window; expanding = per update. Reference = the tsfresh/TSFEL
function tsfast is tested against, called on one window; speed-up = reference / static.

| feature | static | sliding | expanding | reference | ref µs | speed-up |
|:--|--:|--:|--:|:--|--:|--:|
| `abs_max` | 1.41 | 0.71 | 5.18 | tsfresh | 19.7 | 14.0× |
| `abs_sum_change` | 2.40 | 0.84 | 6.06 | tsfresh | 13.6 | 5.7× |
| `agg_autocorrelation-max-5` | 3.17 | 1.84 | 10.81 | tsfresh | 120.9 | 38.1× |
| `agg_autocorrelation-mean-10` | 2.95 | 1.48 | 9.97 | tsfresh | 123.0 | 41.6× |
| `agg_autocorrelation-var-5` | 3.03 | 1.83 | 11.68 | tsfresh | 135.0 | 44.6× |
| `agg_linear_trend-intercept-5-max` | 2.13 | 0.98 | 5.40 | tsfresh | 782.4 | 367.0× |
| `agg_linear_trend-rvalue-5-var` | 2.08 | 1.21 | 6.43 | tsfresh | 1,552.2 | 745.2× |
| `agg_linear_trend-slope-10-mean` | 1.90 | 0.80 | 5.67 | tsfresh | 758.8 | 399.0× |
| `approx_entropy-2-0.1` | 115.48 | 80.61 | 110.12 | tsfresh | 10,468.8 | 90.7× |
| `approx_entropy-2-0.5` | 114.43 | 80.92 | 108.64 | tsfresh | 10,552.3 | 92.2× |
| `ar_coefficient-10-0` | 7.13 | 3.76 | 8.98 | tsfresh | 2,520.7 | 353.5× |
| `ar_coefficient-10-1` | 6.87 | 3.86 | 8.67 | tsfresh | 2,405.0 | 350.0× |
| `auc` | 2.26 | 0.83 | 6.02 | tsfel | 26.8 | 11.8× |
| `augmented_dickey_fuller-pvalue` | 166.45 | 110.87 | 138.69 | tsfresh | 9,268.6 | 55.7× |
| `augmented_dickey_fuller-teststat` | 164.24 | 139.42 | 160.04 | tsfresh | 9,541.8 | 58.1× |
| `augmented_dickey_fuller-usedlag` | 174.34 | 130.94 | 145.44 | tsfresh | 9,582.3 | 55.0× |
| `autocorr-1` | 2.94 | 1.67 | 11.44 | tsfresh | 83.5 | 28.4× |
| `autocorr-3` | 3.24 | 1.49 | 10.44 | tsfresh | 83.4 | 25.8× |
| `autocorr_lag1` | 2.31 | 0.83 | 5.32 | tsfresh | 83.7 | 36.2× |
| `autocorrelation` | 2.91 | 1.42 | 8.84 | tsfel | 87.0 | 29.9× |
| `benford_correlation` | 6.55 | 3.77 | 7.32 | tsfresh | 901.9 | 137.8× |
| `biased_fisher_kurtosis` | 1.96 | 0.76 | 6.44 | tsfel | 817.3 | 416.4× |
| `biased_skewness` | 1.57 | 0.72 | 5.25 | tsfel | 809.1 | 516.3× |
| `binned_entropy__max_bins_5` | 2.49 | 1.45 | 5.82 | tsfresh | 135.4 | 54.4× |
| `c3-1` | 1.96 | 0.86 | 6.38 | tsfresh | 21.2 | 10.8× |
| `c3-2` | 2.04 | 0.85 | 5.96 | tsfresh | 20.3 | 9.9× |
| `calc_centroid-100` | 2.57 | 1.07 | 9.83 | tsfel | 20.4 | 8.0× |
| `calc_centroid-50` | 4.62 | 1.30 | 12.34 | tsfel | 51.7 | 11.2× |
| `change_quantiles-0-1-False-var` | 3.45 | 2.34 | 7.11 | tsfresh | 1,272.5 | 368.7× |
| `change_quantiles-0.2-0.8-True-mean` | 3.43 | 2.17 | 8.41 | tsfresh | 1,220.0 | 355.2× |
| `cid_ce` | 2.27 | 0.79 | 5.65 | tsfresh | 9.0 | 3.9× |
| `count_above_mean` | 2.76 | 0.94 | 6.09 | tsfresh | 12.7 | 4.6× |
| `count_below_mean` | 2.43 | 0.94 | 6.38 | tsfresh | 13.5 | 5.5× |
| `cwt_coefficients__coeff_3__w_5__widths_(2, 5, 10, 20)` | 17.01 | 9.57 | 15.80 | tsfresh | 1,347.1 | 79.2× |
| `ecdf-10` | 3.30 | 1.19 | 9.44 | tsfel | 21.4 | 6.5× |
| `ecdf-3` | 1.31 | 0.73 | 5.67 | tsfel | 13.2 | 10.1× |
| `energy` | 1.39 | 0.71 | 6.19 | tsfresh | 2.1 | 1.5× |
| `energy_ratio_by_chunks_num_segments_3__segment_focus_1` | 2.12 | 0.87 | 6.41 | tsfresh | 34.1 | 16.1× |
| `energy_ratio_by_chunks_num_segments_4__segment_focus_3` | 2.08 | 0.85 | 5.62 | tsfresh | 35.6 | 17.2× |
| `entropy` | 5.24 | 3.73 | 7.53 | tsfel | 83.4 | 15.9× |
| `fft_coeff-0-real` | 2.63 | 1.45 | 11.48 | tsfresh | 20.8 | 7.9× |
| `fft_coeff-1-imag` | 2.56 | 1.38 | 11.19 | tsfresh | 20.8 | 8.1× |
| `fft_coeff-2-angle` | 2.56 | 1.30 | 12.16 | tsfresh | 24.3 | 9.5× |
| `fft_coeff-3-abs` | 2.69 | 1.40 | 10.92 | tsfresh | 22.7 | 8.4× |
| `first_loc_max` | 4.28 | 1.81 | 13.17 | tsfresh | 10.8 | 2.5× |
| `first_loc_min` | 5.12 | 1.88 | 7.94 | tsfresh | 3.9 | 0.8× |
| `friedrich_coefficients-3-30-0` | 9.79 | 5.16 | 11.64 | tsfresh | 7,410.8 | 756.8× |
| `friedrich_coefficients-3-30-3` | 10.11 | 4.94 | 11.55 | tsfresh | 7,946.2 | 785.9× |
| `has_duplicate` | 4.25 | 2.64 | 8.22 | tsfresh | 17.1 | 4.0× |
| `has_duplicate_max` | 2.18 | 0.82 | 5.47 | tsfresh | 14.0 | 6.4× |
| `has_duplicate_min` | 3.06 | 0.82 | 5.95 | tsfresh | 13.9 | 4.5× |
| `higuchi_fd` | 8.48 | 4.50 | 8.35 | tsfel | 6,807.9 | 802.6× |
| `human_range_energy-100` | 7.78 | 2.35 | 17.95 | tsfel | 95.7 | 12.3× |
| `index_mass_quantile-0.5` | 2.51 | 0.92 | 6.65 | tsfresh | 25.7 | 10.3× |
| `index_mass_quantile-0.9` | 2.16 | 0.95 | 5.35 | tsfresh | 25.3 | 11.7× |
| `intercept` | 1.21 | 0.74 | 6.14 | tsfresh | 643.7 | 529.9× |
| `iqr` | 2.71 | 1.78 | 5.51 | tsfel | 168.0 | 62.0× |
| `kurtosis` | 1.96 | 0.72 | 5.80 | tsfresh | 125.3 | 64.0× |
| `large_standard_deviation-0.05` | 3.52 | 1.18 | 11.02 | tsfresh | 45.8 | 13.0× |
| `last_loc_max` | 4.31 | 1.74 | 12.46 | tsfresh | 14.8 | 3.4× |
| `last_loc_min` | 2.26 | 1.01 | 6.51 | tsfresh | 5.3 | 2.3× |
| `length` | 1.59 | 0.74 | 5.66 | tsfresh | 0.3 | 0.2× |
| `linear_trend-intercept` | 1.39 | 0.71 | 5.65 | tsfresh | 677.5 | 488.2× |
| `linear_trend-pvalue` | 2.10 | 0.86 | 6.06 | tsfresh | 643.9 | 306.3× |
| `linear_trend-rvalue` | 1.49 | 0.73 | 5.81 | tsfresh | 650.5 | 436.8× |
| `linear_trend-slope` | 1.49 | 0.73 | 6.55 | tsfresh | 642.1 | 431.6× |
| `linear_trend-stderr` | 1.41 | 0.71 | 5.68 | tsfresh | 645.2 | 457.2× |
| `longest_strike_above_mean` | 2.86 | 0.88 | 5.48 | tsfresh | 612.6 | 214.3× |
| `longest_strike_below_mean` | 2.18 | 0.88 | 6.82 | tsfresh | 621.3 | 285.3× |
| `lpcc-0` | 6.41 | 3.83 | 15.60 | tsfel | 303.2 | 47.3× |
| `lpcc-3` | 8.87 | 4.26 | 17.85 | tsfel | 342.5 | 38.6× |
| `mad` | 1.82 | 0.77 | 5.89 | tsfel | 20.3 | 11.2× |
| `matrix_profile-10-max` | 640.11 | 514.68 | 667.36 | stumpy | 2,226.3 | 3.5× |
| `matrix_profile-10-mean` | 658.67 | 500.02 | 657.48 | stumpy | 2,130.5 | 3.2× |
| `matrix_profile-10-min` | 686.70 | 519.86 | 686.50 | stumpy | 3,085.7 | 4.5× |
| `max_langevin_fixed_point-3-30` | 10.19 | 5.37 | 12.83 | tsfresh | 7,396.7 | 726.1× |
| `max_power_spectrum` | 3.90 | 2.31 | 7.86 | tsfel | 560.0 | 143.6× |
| `max_value` | 1.33 | 0.75 | 5.42 | tsfresh | 5.4 | 4.1× |
| `mean` | 1.35 | 0.70 | 5.20 | tsfresh | 7.8 | 5.7× |
| `mean_abs_change` | 2.36 | 0.77 | 6.17 | tsfresh | 17.8 | 7.5× |
| `mean_change` | 2.01 | 0.80 | 7.08 | tsfresh | 0.9 | 0.4× |
| `mean_n_absolute_max-7` | 2.66 | 1.03 | 6.27 | tsfresh | 16.6 | 6.2× |
| `mean_second_derivative_central` | 1.27 | 0.71 | 5.42 | tsfresh | 1.2 | 1.0× |
| `median` | 2.42 | 1.05 | 5.31 | tsfresh | 32.1 | 13.3× |
| `median_abs_deviation` | 2.41 | 1.57 | 6.01 | tsfel | 676.6 | 281.3× |
| `median_abs_diff` | 2.66 | 1.24 | 5.59 | tsfel | 46.9 | 17.6× |
| `median_diff` | 2.47 | 1.22 | 6.43 | tsfel | 46.2 | 18.7× |
| `mfcc-0` | 21.67 | 9.93 | 28.15 | tsfel | 1,522.3 | 70.3× |
| `mfcc-3` | 15.63 | 8.45 | 22.74 | tsfel | 1,472.5 | 94.2× |
| `min_value` | 1.40 | 0.73 | 5.31 | tsfresh | 5.3 | 3.8× |
| `negative_turning` | 2.22 | 0.70 | 5.33 | tsfel | 19.6 | 8.8× |
| `number_crossing_m__m_0` | 1.52 | 0.71 | 5.90 | tsfresh | 9.8 | 6.5× |
| `number_crossing_m__m_0.5` | 2.19 | 0.75 | 5.86 | tsfresh | 9.3 | 4.3× |
| `number_cwt_peaks__n_1` | 17.77 | 8.78 | 23.22 | tsfresh | 6,006.9 | 338.0× |
| `number_cwt_peaks__n_5` | 57.97 | 36.25 | 60.86 | tsfresh | 8,075.5 | 139.3× |
| `number_peaks__n_1` | 2.37 | 1.04 | 6.32 | tsfresh | 19.2 | 8.1× |
| `number_peaks__n_3` | 2.64 | 1.07 | 5.49 | tsfresh | 38.9 | 14.7× |
| `paa-3-2` | 2.55 | 0.83 | 9.16 | — |  | |
| `paa-4-1` | 1.95 | 0.73 | 6.28 | — |  | |
| `partial_autocorr-1` | 3.10 | 1.82 | 12.16 | tsfresh | 103.3 | 33.3× |
| `partial_autocorr-2` | 3.12 | 1.84 | 12.56 | tsfresh | 117.4 | 37.6× |
| `peak_count` | 1.85 | 0.71 | 5.54 | tsfresh | 18.5 | 10.0× |
| `percentage_of_reoccurring_datapoints_to_all_datapoints` | 4.34 | 0.73 | 5.38 | tsfresh | 528.3 | 121.7× |
| `percentage_of_reoccurring_values_to_all_values` | 4.36 | 0.76 | 5.83 | tsfresh | 45.4 | 10.4× |
| `permutation_entropy-1-3` | 22.49 | 9.97 | 33.53 | tsfresh | 592.0 | 26.3× |
| `permutation_entropy-1-5` | 20.15 | 10.31 | 24.38 | tsfresh | 1,183.0 | 58.7× |
| `pk_pk_distance` | 1.65 | 0.72 | 6.61 | tsfel | 11.0 | 6.6× |
| `positive_turning` | 2.12 | 0.73 | 5.75 | tsfel | 19.3 | 9.1× |
| `quantile-0.25` | 2.93 | 1.85 | 6.99 | tsfresh | 82.5 | 28.1× |
| `quantile-0.9` | 2.79 | 1.88 | 6.19 | tsfresh | 82.0 | 29.4× |
| `query_similarity_count-10-0.5` | 8.59 | 4.69 | 8.75 | tsfresh | 1,207.5 | 140.6× |
| `ratio_beyond_r_sigma-1.5` | 3.45 | 1.14 | 9.20 | tsfresh | 130.6 | 37.8× |
| `ratio_value_number_to_time_series_length` | 4.25 | 0.72 | 6.16 | tsfresh | 17.2 | 4.0× |
| `rms` | 1.86 | 0.72 | 5.84 | tsfel | 7.8 | 4.2× |
| `root_mean_square` | 1.35 | 0.73 | 5.92 | tsfresh | 10.1 | 7.5× |
| `sample_entropy` | 33.17 | 17.83 | 30.80 | tsfresh | 17,159.9 | 517.3× |
| `signal_distance` | 1.88 | 0.87 | 5.52 | tsfel | 24.5 | 13.1× |
| `skewness` | 1.63 | 0.71 | 6.15 | tsfresh | 114.5 | 70.1× |
| `slope` | 1.21 | 0.74 | 5.59 | tsfel | 105.3 | 87.2× |
| `slope_sign_change` | 2.32 | 0.76 | 5.26 | — |  | |
| `spectral_centroid` | 2.97 | 1.50 | 10.84 | tsfel | 39.9 | 13.5× |
| `spectral_decrease` | 2.69 | 1.43 | 11.21 | tsfel | 51.7 | 19.2× |
| `spectral_distance` | 2.92 | 1.63 | 11.15 | tsfel | 57.6 | 19.7× |
| `spectral_entropy` | 3.17 | 1.83 | 10.65 | tsfel | 61.0 | 19.2× |
| `spectral_kurtosis` | 2.56 | 1.31 | 10.41 | tsfel | 238.7 | 93.1× |
| `spectral_roll_off` | 2.99 | 1.46 | 11.47 | tsfel | 42.4 | 14.1× |
| `spectral_roll_on` | 2.60 | 1.43 | 11.04 | tsfel | 43.4 | 16.7× |
| `spectral_skewness` | 2.67 | 1.39 | 10.66 | tsfel | 237.7 | 89.0× |
| `spectral_slope` | 2.82 | 1.44 | 11.08 | tsfel | 43.4 | 15.4× |
| `spectral_spread` | 2.70 | 1.47 | 12.03 | tsfel | 81.1 | 30.0× |
| `spectrogram-2-0.5` | 6.65 | 2.50 | 18.79 | — |  | |
| `spkt_welch_density__coeff_2` | 5.27 | 3.02 | 12.82 | tsfresh | 510.8 | 96.9× |
| `spkt_welch_density__coeff_5` | 4.34 | 2.59 | 12.72 | tsfresh | 502.8 | 115.8× |
| `std_dev` | 1.82 | 0.71 | 5.52 | tsfresh | 23.1 | 12.7× |
| `sum_of_reoccurring_data_points` | 4.87 | 2.81 | 7.67 | tsfresh | 45.6 | 9.4× |
| `sum_of_reoccurring_values` | 4.44 | 2.81 | 7.34 | tsfresh | 46.8 | 10.5× |
| `symmetry_looking-0.05` | 4.14 | 1.82 | 11.21 | tsfresh | 170.4 | 41.2× |
| `time_reversal_asymmetry-1` | 2.04 | 0.86 | 6.19 | tsfresh | 23.3 | 11.4× |
| `time_reversal_asymmetry-2` | 1.92 | 0.90 | 6.42 | tsfresh | 23.4 | 12.2× |
| `total_sum` | 1.42 | 0.70 | 5.47 | tsfresh | 4.8 | 3.4× |
| `turning_points` | 1.72 | 0.69 | 5.29 | — |  | |
| `value_count-0` | 4.03 | 1.33 | 12.70 | tsfresh | 11.7 | 2.9× |
| `value_count-1` | 2.58 | 1.04 | 10.12 | tsfresh | 5.6 | 2.2× |
| `variance` | 1.45 | 0.70 | 4.96 | tsfresh | 21.8 | 15.1× |
| `variance_larger_than_standard_deviation` | 1.47 | 0.72 | 5.65 | tsfresh | 22.8 | 15.6× |
| `variation_coefficient` | 1.61 | 0.72 | 6.07 | tsfresh | 33.1 | 20.6× |
| `wavelet-0.5-1` | 3.90 | 1.32 | 9.07 | — |  | |
| `wavelet_energy-0` | 107.77 | 75.72 | 95.59 | tsfel | 1,590.3 | 14.8× |
| `wavelet_energy-3` | 126.36 | 90.50 | 112.60 | tsfel | 1,628.8 | 12.9× |
| `wavelet_entropy` | 111.72 | 77.84 | 100.58 | tsfel | 1,533.4 | 13.7× |
| `zero_cross` | 2.44 | 0.72 | 6.39 | tsfel | 10.2 | 4.2× |
| `zero_crossing_mean` | 2.40 | 1.13 | 5.40 | — |  | |
| `zero_crossing_rate` | 1.97 | 0.76 | 5.62 | — |  | |
| `zero_crossing_std` | 2.42 | 1.17 | 5.29 | — |  | |
