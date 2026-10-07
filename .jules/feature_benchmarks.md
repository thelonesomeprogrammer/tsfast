# Per-feature benchmark

Microseconds per window per series. Window 256, 20 series, 32 new values per sliding/expanding update. Commit 83a16cf, 2026-10-07 07:01.
Sliding = per new window; expanding = per update. Reference = the tsfresh/TSFEL
function tsfast is tested against, called on one window; speed-up = reference / static.

| feature | static | sliding | expanding | reference | ref µs | speed-up |
|:--|--:|--:|--:|:--|--:|--:|
| `abs_max` | 1.42 | 0.75 | 5.48 | tsfresh | 7.1 | 5.0× |
| `abs_sum_change` | 1.73 | 0.81 | 5.79 | tsfresh | 13.6 | 7.9× |
| `agg_autocorrelation-max-5` | 2.65 | 1.52 | 11.87 | tsfresh | 115.5 | 43.6× |
| `agg_autocorrelation-mean-10` | 2.62 | 1.32 | 9.69 | tsfresh | 113.7 | 43.5× |
| `agg_autocorrelation-var-5` | 2.94 | 1.58 | 11.40 | tsfresh | 126.0 | 42.8× |
| `agg_linear_trend-intercept-5-max` | 2.27 | 1.06 | 6.18 | tsfresh | 760.0 | 335.0× |
| `agg_linear_trend-rvalue-5-var` | 2.15 | 1.30 | 18.65 | tsfresh | 3,461.4 | 1,611.8× |
| `agg_linear_trend-slope-10-mean` | 1.96 | 0.89 | 6.52 | tsfresh | 796.4 | 405.5× |
| `approx_entropy-2-0.1` | 110.03 | 80.43 | 108.25 | tsfresh | 9,919.8 | 90.2× |
| `approx_entropy-2-0.5` | 114.96 | 80.85 | 109.56 | tsfresh | 10,340.7 | 89.9× |
| `ar_coefficient-10-0` | 6.81 | 3.30 | 9.75 | tsfresh | 2,255.9 | 331.3× |
| `ar_coefficient-10-1` | 6.59 | 3.23 | 8.65 | tsfresh | 2,245.3 | 340.6× |
| `auc` | 1.88 | 0.87 | 5.81 | tsfel | 25.5 | 13.6× |
| `augmented_dickey_fuller-pvalue` | 213.18 | 111.16 | 135.85 | tsfresh | 8,702.4 | 40.8× |
| `augmented_dickey_fuller-teststat` | 160.62 | 116.77 | 290.03 | tsfresh | 46,952.7 | 292.3× |
| `augmented_dickey_fuller-usedlag` | 223.93 | 190.68 | 259.97 | tsfresh | 18,383.3 | 82.1× |
| `autocorr-1` | 2.63 | 1.36 | 11.15 | tsfresh | 82.8 | 31.5× |
| `autocorr-3` | 2.58 | 1.33 | 10.62 | tsfresh | 80.7 | 31.3× |
| `autocorr_lag1` | 1.94 | 0.88 | 6.05 | tsfresh | 83.2 | 42.9× |
| `autocorrelation` | 2.71 | 1.39 | 10.64 | tsfel | 81.4 | 30.0× |
| `benford_correlation` | 7.04 | 3.57 | 7.44 | tsfresh | 905.7 | 128.7× |
| `biased_fisher_kurtosis` | 1.51 | 0.79 | 6.03 | tsfel | 736.1 | 487.8× |
| `biased_skewness` | 4.18 | 1.21 | 10.97 | tsfel | 1,531.5 | 366.3× |
| `binned_entropy__max_bins_5` | 2.59 | 1.36 | 5.62 | tsfresh | 129.1 | 49.8× |
| `c3-1` | 1.94 | 0.84 | 7.30 | tsfresh | 19.6 | 10.1× |
| `c3-2` | 1.88 | 0.86 | 7.60 | tsfresh | 19.7 | 10.5× |
| `calc_centroid-100` | 1.44 | 0.74 | 5.78 | tsfel | 17.0 | 11.8× |
| `calc_centroid-50` | 1.88 | 0.80 | 7.00 | tsfel | 16.8 | 8.9× |
| `change_quantiles-0-1-False-var` | 3.72 | 2.14 | 8.05 | tsfresh | 1,164.3 | 313.0× |
| `change_quantiles-0.2-0.8-True-mean` | 3.24 | 1.93 | 7.60 | tsfresh | 1,125.3 | 347.5× |
| `cid_ce` | 2.02 | 0.77 | 6.25 | tsfresh | 8.0 | 4.0× |
| `count_above_mean` | 2.04 | 0.94 | 6.00 | tsfresh | 12.9 | 6.3× |
| `count_below_mean` | 1.94 | 0.97 | 5.85 | tsfresh | 13.3 | 6.9× |
| `cwt_coefficients__coeff_3__w_5__widths_(2, 5, 10, 20)` | 23.20 | 13.37 | 24.30 | tsfresh | 1,908.4 | 82.2× |
| `ecdf-10` | 1.19 | 0.71 | 6.02 | tsfel | 12.8 | 10.8× |
| `ecdf-3` | 1.17 | 0.74 | 6.84 | tsfel | 12.2 | 10.5× |
| `energy` | 1.63 | 0.76 | 5.90 | tsfresh | 2.0 | 1.2× |
| `energy_ratio_by_chunks_num_segments_3__segment_focus_1` | 2.02 | 0.87 | 6.11 | tsfresh | 36.7 | 18.2× |
| `energy_ratio_by_chunks_num_segments_4__segment_focus_3` | 2.01 | 0.85 | 6.03 | tsfresh | 38.1 | 18.9× |
| `entropy` | 5.41 | 3.73 | 8.30 | tsfel | 80.2 | 14.8× |
| `fft_coeff-0-real` | 2.37 | 1.17 | 10.50 | tsfresh | 22.0 | 9.3× |
| `fft_coeff-1-imag` | 2.25 | 1.24 | 11.21 | tsfresh | 22.2 | 9.9× |
| `fft_coeff-2-angle` | 2.51 | 1.26 | 11.34 | tsfresh | 25.9 | 10.3× |
| `fft_coeff-3-abs` | 2.46 | 1.22 | 10.13 | tsfresh | 23.8 | 9.6× |
| `first_loc_max` | 2.17 | 1.03 | 5.44 | tsfresh | 3.5 | 1.6× |
| `first_loc_min` | 2.05 | 1.04 | 6.40 | tsfresh | 3.5 | 1.7× |
| `friedrich_coefficients-3-30-0` | 8.87 | 4.55 | 10.97 | tsfresh | 7,613.7 | 858.5× |
| `friedrich_coefficients-3-30-3` | 9.23 | 4.66 | 11.59 | tsfresh | 7,851.2 | 851.0× |
| `has_duplicate` | 4.15 | 2.50 | 7.25 | tsfresh | 16.2 | 3.9× |
| `has_duplicate_max` | 2.41 | 0.86 | 6.29 | tsfresh | 14.1 | 5.8× |
| `has_duplicate_min` | 2.21 | 0.81 | 5.66 | tsfresh | 14.0 | 6.4× |
| `higuchi_fd` | 9.43 | 4.40 | 8.68 | tsfel | 8,370.0 | 887.5× |
| `human_range_energy-100` | 2.56 | 1.22 | 10.11 | tsfel | 51.6 | 20.1× |
| `index_mass_quantile-0.5` | 2.22 | 0.94 | 6.08 | tsfresh | 24.7 | 11.1× |
| `index_mass_quantile-0.9` | 1.92 | 0.99 | 5.38 | tsfresh | 25.2 | 13.1× |
| `intercept` | 1.17 | 0.76 | 6.01 | tsfresh | 576.2 | 492.1× |
| `iqr` | 3.03 | 1.75 | 6.75 | tsfel | 179.1 | 59.2× |
| `kurtosis` | 1.51 | 0.78 | 5.92 | tsfresh | 115.7 | 76.8× |
| `large_standard_deviation-0.05` | 1.74 | 0.74 | 5.99 | tsfresh | 37.4 | 21.5× |
| `last_loc_max` | 2.11 | 1.04 | 6.18 | tsfresh | 4.8 | 2.3× |
| `last_loc_min` | 2.09 | 1.06 | 5.92 | tsfresh | 4.9 | 2.4× |
| `lempel_ziv` | 3.97 | 1.79 | 8.04 | tsfel | 178.9 | 45.1× |
| `lempel_ziv_complexity-3` | 9.75 | 4.42 | 17.71 | tsfresh | 571.0 | 58.6× |
| `length` | 1.11 | 0.73 | 5.82 | tsfresh | 0.3 | 0.3× |
| `linear_trend-intercept` | 1.38 | 0.77 | 6.66 | tsfresh | 596.8 | 432.6× |
| `linear_trend-pvalue` | 2.11 | 0.90 | 6.55 | tsfresh | 610.9 | 290.2× |
| `linear_trend-rvalue` | 1.48 | 0.79 | 7.07 | tsfresh | 603.6 | 407.0× |
| `linear_trend-slope` | 1.31 | 0.77 | 5.91 | tsfresh | 587.1 | 447.3× |
| `linear_trend-stderr` | 1.66 | 0.75 | 5.98 | tsfresh | 599.1 | 362.0× |
| `longest_strike_above_mean` | 2.10 | 0.97 | 5.67 | tsfresh | 590.5 | 281.6× |
| `longest_strike_below_mean` | 2.14 | 0.88 | 5.66 | tsfresh | 585.0 | 273.9× |
| `lpcc-0` | 6.93 | 3.73 | 15.15 | tsfel | 270.1 | 39.0× |
| `lpcc-3` | 6.55 | 3.67 | 20.26 | tsfel | 411.7 | 62.9× |
| `mad` | 1.73 | 0.83 | 5.74 | tsfel | 20.5 | 11.9× |
| `matrix_profile-10-max` | 660.62 | 523.54 | 666.20 | stumpy | 2,130.1 | 3.2× |
| `matrix_profile-10-mean` | 676.25 | 572.57 | 722.79 | stumpy | 2,127.8 | 3.1× |
| `matrix_profile-10-min` | 674.61 | 567.90 | 672.78 | stumpy | 2,071.4 | 3.1× |
| `max_langevin_fixed_point-3-30` | 10.13 | 4.72 | 11.76 | tsfresh | 8,006.4 | 790.0× |
| `max_power_spectrum` | 3.52 | 2.09 | 7.80 | tsfel | 502.7 | 142.9× |
| `max_value` | 1.16 | 0.77 | 6.65 | tsfresh | 10.8 | 9.3× |
| `mean` | 1.26 | 0.76 | 6.26 | tsfresh | 9.2 | 7.3× |
| `mean_abs_change` | 2.19 | 0.82 | 6.23 | tsfresh | 17.0 | 7.8× |
| `mean_change` | 1.90 | 0.78 | 6.44 | tsfresh | 0.8 | 0.4× |
| `mean_n_absolute_max-7` | 2.08 | 1.07 | 6.64 | tsfresh | 16.0 | 7.7× |
| `mean_second_derivative_central` | 1.08 | 0.76 | 5.84 | tsfresh | 1.2 | 1.1× |
| `median` | 4.04 | 1.69 | 12.02 | tsfresh | 78.3 | 19.4× |
| `median_abs_deviation` | 15.27 | 3.15 | 13.97 | tsfel | 1,157.1 | 75.8× |
| `median_abs_diff` | 2.39 | 1.18 | 6.31 | tsfel | 43.8 | 18.4× |
| `median_diff` | 2.37 | 1.16 | 6.87 | tsfel | 43.2 | 18.2× |
| `mfcc-0` | 10.68 | 5.27 | 14.19 | tsfel | 1,197.3 | 112.1× |
| `mfcc-3` | 10.36 | 5.14 | 13.95 | tsfel | 1,174.5 | 113.4× |
| `min_value` | 1.24 | 0.79 | 6.20 | tsfresh | 5.6 | 4.5× |
| `negative_turning` | 2.04 | 0.77 | 6.23 | tsfel | 17.9 | 8.8× |
| `number_crossing_m__m_0` | 1.38 | 0.80 | 6.43 | tsfresh | 8.9 | 6.5× |
| `number_crossing_m__m_0.5` | 1.56 | 0.77 | 6.23 | tsfresh | 8.5 | 5.5× |
| `number_cwt_peaks__n_1` | 17.95 | 9.13 | 23.75 | tsfresh | 5,807.7 | 323.6× |
| `number_cwt_peaks__n_5` | 87.86 | 46.52 | 76.59 | tsfresh | 16,941.9 | 192.8× |
| `number_peaks__n_1` | 2.29 | 1.08 | 5.55 | tsfresh | 17.7 | 7.7× |
| `number_peaks__n_3` | 2.47 | 1.13 | 5.82 | tsfresh | 36.7 | 14.9× |
| `paa-3-2` | 1.94 | 0.77 | 8.24 | — |  | |
| `paa-4-1` | 2.27 | 0.82 | 7.09 | — |  | |
| `partial_autocorr-1` | 2.72 | 1.57 | 11.80 | tsfresh | 92.9 | 34.2× |
| `partial_autocorr-2` | 2.74 | 1.57 | 10.91 | tsfresh | 101.4 | 37.0× |
| `peak_count` | 1.93 | 0.75 | 5.88 | tsfresh | 18.0 | 9.3× |
| `percentage_of_reoccurring_datapoints_to_all_datapoints` | 4.19 | 0.69 | 5.59 | tsfresh | 514.5 | 122.9× |
| `percentage_of_reoccurring_values_to_all_values` | 4.10 | 0.79 | 6.07 | tsfresh | 42.5 | 10.4× |
| `permutation_entropy-1-3` | 13.64 | 7.12 | 19.03 | tsfresh | 371.1 | 27.2× |
| `permutation_entropy-1-5` | 20.93 | 10.65 | 27.41 | tsfresh | 501.7 | 24.0× |
| `pk_pk_distance` | 1.45 | 0.77 | 5.62 | tsfel | 11.4 | 7.8× |
| `positive_turning` | 1.83 | 0.75 | 6.68 | tsfel | 18.1 | 9.8× |
| `quantile-0.25` | 2.53 | 1.62 | 6.51 | tsfresh | 86.9 | 34.4× |
| `quantile-0.9` | 3.88 | 2.23 | 17.65 | tsfresh | 93.7 | 24.1× |
| `query_similarity_count-10-0.5` | 10.37 | 4.61 | 9.13 | tsfresh | 1,140.2 | 110.0× |
| `ratio_beyond_r_sigma-1.5` | 1.99 | 0.77 | 6.30 | tsfresh | 47.4 | 23.8× |
| `ratio_value_number_to_time_series_length` | 4.20 | 0.73 | 5.41 | tsfresh | 16.4 | 3.9× |
| `rms` | 1.29 | 0.79 | 6.54 | tsfel | 9.2 | 7.1× |
| `root_mean_square` | 1.23 | 0.78 | 6.12 | tsfresh | 11.5 | 9.4× |
| `sample_entropy` | 28.08 | 17.39 | 26.91 | tsfresh | 16,127.1 | 574.4× |
| `signal_distance` | 2.10 | 0.87 | 6.03 | tsfel | 23.6 | 11.3× |
| `skewness` | 7.80 | 1.65 | 11.70 | tsfresh | 222.7 | 28.6× |
| `slope` | 1.21 | 0.76 | 5.85 | tsfel | 94.9 | 78.5× |
| `slope_sign_change` | 1.81 | 0.81 | 5.51 | — |  | |
| `spectral_centroid` | 2.35 | 1.18 | 10.56 | tsfel | 37.7 | 16.1× |
| `spectral_decrease` | 2.32 | 1.17 | 10.89 | tsfel | 48.3 | 20.8× |
| `spectral_distance` | 2.56 | 1.38 | 11.02 | tsfel | 54.2 | 21.2× |
| `spectral_entropy` | 2.72 | 1.53 | 11.78 | tsfel | 59.4 | 21.8× |
| `spectral_kurtosis` | 2.34 | 1.25 | 10.85 | tsfel | 229.5 | 98.1× |
| `spectral_roll_off` | 2.26 | 1.12 | 10.28 | tsfel | 39.4 | 17.5× |
| `spectral_roll_on` | 2.35 | 1.17 | 10.63 | tsfel | 39.7 | 16.9× |
| `spectral_skewness` | 2.28 | 1.14 | 9.92 | tsfel | 229.4 | 100.8× |
| `spectral_slope` | 2.38 | 1.18 | 10.80 | tsfel | 41.0 | 17.3× |
| `spectral_spread` | 2.36 | 1.25 | 10.84 | tsfel | 75.5 | 32.0× |
| `spectrogram-2-0.5` | 2.18 | 1.16 | 10.27 | — |  | |
| `spkt_welch_density__coeff_2` | 9.19 | 3.57 | 17.44 | tsfresh | 822.1 | 89.4× |
| `spkt_welch_density__coeff_5` | 3.92 | 2.52 | 12.81 | tsfresh | 461.9 | 117.9× |
| `std_dev` | 1.31 | 0.80 | 5.70 | tsfresh | 24.1 | 18.3× |
| `sum_of_reoccurring_data_points` | 4.48 | 2.81 | 7.64 | tsfresh | 42.7 | 9.5× |
| `sum_of_reoccurring_values` | 4.40 | 2.73 | 7.82 | tsfresh | 44.5 | 10.1× |
| `symmetry_looking-0.05` | 2.89 | 1.07 | 6.33 | tsfresh | 65.4 | 22.6× |
| `time_reversal_asymmetry-1` | 1.89 | 0.85 | 7.50 | tsfresh | 22.8 | 12.1× |
| `time_reversal_asymmetry-2` | 1.50 | 0.87 | 6.65 | tsfresh | 22.4 | 14.9× |
| `total_sum` | 1.17 | 0.79 | 6.33 | tsfresh | 5.7 | 4.9× |
| `turning_points` | 1.79 | 0.74 | 5.52 | — |  | |
| `value_count-0` | 1.60 | 0.83 | 7.50 | tsfresh | 4.1 | 2.6× |
| `value_count-1` | 1.47 | 0.73 | 5.56 | tsfresh | 4.2 | 2.8× |
| `variance` | 1.48 | 0.76 | 5.90 | tsfresh | 25.6 | 17.3× |
| `variance_larger_than_standard_deviation` | 1.55 | 0.79 | 5.98 | tsfresh | 23.5 | 15.2× |
| `variation_coefficient` | 1.80 | 0.74 | 5.83 | tsfresh | 33.8 | 18.8× |
| `wavelet-0.5-1` | 2.00 | 0.81 | 5.91 | — |  | |
| `wavelet_energy-0` | 111.50 | 75.14 | 93.83 | tsfel | 1,412.9 | 12.7× |
| `wavelet_energy-3` | 149.91 | 106.62 | 127.05 | tsfel | 2,220.4 | 14.8× |
| `wavelet_entropy` | 107.91 | 75.31 | 93.86 | tsfel | 1,341.2 | 12.4× |
| `zero_cross` | 1.87 | 0.79 | 5.73 | tsfel | 9.4 | 5.0× |
| `zero_crossing_mean` | 2.26 | 1.06 | 6.17 | — |  | |
| `zero_crossing_rate` | 1.96 | 0.83 | 6.68 | — |  | |
| `zero_crossing_std` | 2.24 | 1.16 | 6.26 | — |  | |
