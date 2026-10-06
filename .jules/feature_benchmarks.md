# Per-feature benchmark

Microseconds per window per series. Window 256, 20 series, 32 new values per sliding/expanding update. Commit 83a16cf, 2026-10-06 09:49.
Sliding = per new window; expanding = per update. Reference = the tsfresh/TSFEL
function tsfast is tested against, called on one window; speed-up = reference / static.

| feature | static | sliding | expanding | reference | ref µs | speed-up |
|:--|--:|--:|--:|:--|--:|--:|
| `abs_max` | 1.59 | 0.81 | 5.92 | tsfresh | 6.2 | 3.9× |
| `abs_sum_change` | 2.32 | 0.79 | 5.28 | tsfresh | 12.2 | 5.3× |
| `agg_autocorrelation-max-5` | 2.76 | 1.50 | 10.33 | tsfresh | 102.9 | 37.3× |
| `agg_autocorrelation-mean-10` | 2.60 | 1.29 | 9.04 | tsfresh | 110.3 | 42.4× |
| `agg_autocorrelation-var-5` | 2.52 | 1.48 | 9.58 | tsfresh | 117.4 | 46.5× |
| `agg_linear_trend-intercept-5-max` | 2.27 | 1.04 | 5.78 | tsfresh | 714.7 | 314.4× |
| `agg_linear_trend-rvalue-5-var` | 2.20 | 1.07 | 6.35 | tsfresh | 1,806.6 | 819.9× |
| `agg_linear_trend-slope-10-mean` | 2.06 | 0.88 | 6.16 | tsfresh | 742.3 | 360.2× |
| `approx_entropy-2-0.1` | 111.86 | 80.01 | 107.76 | tsfresh | 9,763.5 | 87.3× |
| `approx_entropy-2-0.5` | 109.55 | 80.31 | 107.66 | tsfresh | 9,823.9 | 89.7× |
| `ar_coefficient-10-0` | 7.88 | 3.74 | 9.55 | tsfresh | 2,225.6 | 282.3× |
| `ar_coefficient-10-1` | 7.78 | 3.74 | 9.49 | tsfresh | 2,191.7 | 281.6× |
| `auc` | 4.26 | 0.86 | 5.83 | tsfel | 25.8 | 6.1× |
| `augmented_dickey_fuller-pvalue` | 161.80 | 119.83 | 147.80 | tsfresh | 8,140.6 | 50.3× |
| `augmented_dickey_fuller-teststat` | 163.78 | 119.12 | 146.91 | tsfresh | 9,179.9 | 56.1× |
| `augmented_dickey_fuller-usedlag` | 162.49 | 118.82 | 149.57 | tsfresh | 8,299.1 | 51.1× |
| `autocorr-1` | 2.51 | 1.28 | 10.15 | tsfresh | 74.1 | 29.5× |
| `autocorr-3` | 2.64 | 1.28 | 9.64 | tsfresh | 72.7 | 27.6× |
| `autocorr_lag1` | 2.02 | 0.90 | 5.80 | tsfresh | 74.0 | 36.7× |
| `autocorrelation` | 2.74 | 1.40 | 9.73 | tsfel | 79.6 | 29.1× |
| `benford_correlation` | 12.67 | 4.86 | 14.54 | tsfresh | 1,834.7 | 144.8× |
| `biased_fisher_kurtosis` | 2.05 | 0.78 | 5.56 | tsfel | 713.4 | 347.6× |
| `biased_skewness` | 2.96 | 0.82 | 7.11 | tsfel | 709.8 | 239.7× |
| `binned_entropy__max_bins_5` | 2.59 | 1.37 | 5.01 | tsfresh | 117.9 | 45.6× |
| `c3-1` | 1.89 | 0.80 | 6.49 | tsfresh | 18.6 | 9.8× |
| `c3-2` | 1.90 | 0.85 | 6.00 | tsfresh | 17.6 | 9.2× |
| `calc_centroid-100` | 2.19 | 0.75 | 5.83 | tsfel | 15.3 | 7.0× |
| `calc_centroid-50` | 2.25 | 0.74 | 6.32 | tsfel | 20.5 | 9.1× |
| `change_quantiles-0-1-False-var` | 3.14 | 2.04 | 7.29 | tsfresh | 1,158.9 | 368.7× |
| `change_quantiles-0.2-0.8-True-mean` | 3.12 | 1.88 | 7.53 | tsfresh | 1,108.5 | 355.1× |
| `cid_ce` | 2.02 | 0.79 | 5.88 | tsfresh | 8.5 | 4.2× |
| `count_above_mean` | 4.19 | 0.94 | 6.14 | tsfresh | 11.8 | 2.8× |
| `count_below_mean` | 2.27 | 0.95 | 5.78 | tsfresh | 13.3 | 5.9× |
| `cwt_coefficients__coeff_3__w_5__widths_(2, 5, 10, 20)` | 16.20 | 9.11 | 14.84 | tsfresh | 1,160.3 | 71.6× |
| `ecdf-10` | 1.97 | 0.76 | 6.08 | tsfel | 12.9 | 6.5× |
| `ecdf-3` | 1.50 | 0.73 | 5.56 | tsfel | 11.8 | 7.9× |
| `ecdf_percentile-0.5` | 2.68 | 1.14 | 7.71 | tsfel | 154.7 | 57.8× |
| `ecdf_percentile_count-0.5` | 4.75 | 1.84 | 13.15 | tsfel | 166.1 | 34.9× |
| `ecdf_slope-0.2-0.5` | 7.66 | 2.26 | 12.85 | tsfel | 122.5 | 16.0× |
| `energy` | 2.74 | 0.79 | 5.41 | tsfresh | 1.9 | 0.7× |
| `energy_ratio_by_chunks_num_segments_3__segment_focus_1` | 2.10 | 0.86 | 4.89 | tsfresh | 33.9 | 16.2× |
| `energy_ratio_by_chunks_num_segments_4__segment_focus_3` | 1.96 | 0.86 | 5.96 | tsfresh | 41.4 | 21.1× |
| `entropy` | 4.96 | 3.57 | 7.55 | tsfel | 73.3 | 14.8× |
| `fft_coeff-0-real` | 2.48 | 1.21 | 10.37 | tsfresh | 21.8 | 8.8× |
| `fft_coeff-1-imag` | 2.27 | 1.15 | 10.04 | tsfresh | 21.7 | 9.5× |
| `fft_coeff-2-angle` | 2.42 | 1.18 | 10.81 | tsfresh | 24.9 | 10.3× |
| `fft_coeff-3-abs` | 2.53 | 1.20 | 9.90 | tsfresh | 23.0 | 9.1× |
| `first_loc_max` | 2.35 | 1.07 | 5.16 | tsfresh | 3.5 | 1.5× |
| `first_loc_min` | 2.23 | 1.05 | 6.32 | tsfresh | 7.1 | 3.2× |
| `friedrich_coefficients-3-30-0` | 9.29 | 4.40 | 11.04 | tsfresh | 7,106.6 | 765.2× |
| `friedrich_coefficients-3-30-3` | 9.39 | 4.41 | 10.99 | tsfresh | 7,217.4 | 768.6× |
| `has_duplicate` | 4.10 | 2.48 | 6.95 | tsfresh | 15.2 | 3.7× |
| `has_duplicate_max` | 2.34 | 0.84 | 5.24 | tsfresh | 12.2 | 5.2× |
| `has_duplicate_min` | 2.37 | 0.83 | 5.36 | tsfresh | 12.2 | 5.2× |
| `higuchi_fd` | 10.32 | 4.91 | 9.14 | tsfel | 7,019.0 | 680.2× |
| `human_range_energy-100` | 4.91 | 1.25 | 10.60 | tsfel | 47.2 | 9.6× |
| `index_mass_quantile-0.5` | 2.10 | 0.93 | 6.42 | tsfresh | 22.8 | 10.8× |
| `index_mass_quantile-0.9` | 3.57 | 0.99 | 5.66 | tsfresh | 22.7 | 6.4× |
| `intercept` | 1.91 | 0.73 | 5.54 | tsfresh | 562.8 | 294.9× |
| `iqr` | 2.87 | 1.73 | 6.22 | tsfel | 164.4 | 57.3× |
| `kurtosis` | 2.92 | 0.74 | 6.17 | tsfresh | 120.8 | 41.4× |
| `large_standard_deviation-0.05` | 1.89 | 0.77 | 6.07 | tsfresh | 32.0 | 16.9× |
| `last_loc_max` | 2.24 | 1.05 | 5.79 | tsfresh | 6.4 | 2.9× |
| `last_loc_min` | 3.86 | 1.60 | 11.77 | tsfresh | 9.6 | 2.5× |
| `length` | 1.63 | 0.76 | 5.54 | tsfresh | 0.3 | 0.2× |
| `linear_trend-intercept` | 1.92 | 0.77 | 5.75 | tsfresh | 575.1 | 299.4× |
| `linear_trend-pvalue` | 2.21 | 0.84 | 5.79 | tsfresh | 571.4 | 259.0× |
| `linear_trend-rvalue` | 1.98 | 0.72 | 5.67 | tsfresh | 591.6 | 298.1× |
| `linear_trend-slope` | 2.34 | 0.74 | 5.63 | tsfresh | 562.4 | 240.7× |
| `linear_trend-stderr` | 1.86 | 0.73 | 5.98 | tsfresh | 572.2 | 308.3× |
| `longest_strike_above_mean` | 2.22 | 0.91 | 5.28 | tsfresh | 550.6 | 248.4× |
| `longest_strike_below_mean` | 4.81 | 0.94 | 5.61 | tsfresh | 590.5 | 122.8× |
| `lpcc-0` | 6.78 | 3.61 | 14.98 | tsfel | 267.9 | 39.5× |
| `lpcc-3` | 6.62 | 3.60 | 14.45 | tsfel | 263.4 | 39.8× |
| `mad` | 2.07 | 0.79 | 5.06 | tsfel | 17.7 | 8.5× |
| `matrix_profile-10-max` | 644.60 | 523.36 | 668.18 | stumpy | 1,997.4 | 3.1× |
| `matrix_profile-10-mean` | 657.18 | 501.39 | 657.59 | stumpy | 2,024.4 | 3.1× |
| `matrix_profile-10-min` | 659.81 | 503.11 | 662.02 | stumpy | 2,042.5 | 3.1× |
| `max_langevin_fixed_point-3-30` | 9.73 | 4.71 | 12.17 | tsfresh | 7,171.9 | 737.1× |
| `max_power_spectrum` | 3.76 | 2.01 | 7.40 | tsfel | 541.5 | 143.9× |
| `max_value` | 2.36 | 0.77 | 5.55 | tsfresh | 5.5 | 2.3× |
| `mean` | 2.05 | 0.78 | 7.42 | tsfresh | 9.8 | 4.8× |
| `mean_abs_change` | 1.98 | 0.81 | 5.83 | tsfresh | 15.3 | 7.7× |
| `mean_change` | 2.67 | 0.78 | 5.51 | tsfresh | 0.9 | 0.3× |
| `mean_n_absolute_max-7` | 2.59 | 1.09 | 6.59 | tsfresh | 14.9 | 5.7× |
| `mean_second_derivative_central` | 1.72 | 0.73 | 5.42 | tsfresh | 1.2 | 0.7× |
| `median` | 2.72 | 1.06 | 5.58 | tsfresh | 33.1 | 12.2× |
| `median_abs_deviation` | 3.75 | 1.39 | 6.77 | tsfel | 577.5 | 154.2× |
| `median_abs_diff` | 2.37 | 1.14 | 6.18 | tsfel | 40.9 | 17.3× |
| `median_diff` | 2.40 | 1.10 | 6.21 | tsfel | 40.7 | 16.9× |
| `mfcc-0` | 10.74 | 4.89 | 13.78 | tsfel | 1,128.3 | 105.0× |
| `mfcc-3` | 10.65 | 4.86 | 13.56 | tsfel | 1,176.5 | 110.5× |
| `min_value` | 1.72 | 0.76 | 5.60 | tsfresh | 5.3 | 3.1× |
| `negative_turning` | 1.96 | 0.74 | 5.51 | tsfel | 17.2 | 8.8× |
| `number_crossing_m__m_0` | 1.94 | 0.75 | 5.41 | tsfresh | 8.7 | 4.5× |
| `number_crossing_m__m_0.5` | 1.81 | 0.77 | 6.12 | tsfresh | 8.3 | 4.6× |
| `number_cwt_peaks__n_1` | 16.98 | 8.19 | 21.69 | tsfresh | 5,313.9 | 312.9× |
| `number_cwt_peaks__n_5` | 52.09 | 33.08 | 55.50 | tsfresh | 7,175.5 | 137.8× |
| `number_peaks__n_1` | 2.07 | 1.05 | 5.34 | tsfresh | 16.2 | 7.8× |
| `number_peaks__n_3` | 2.29 | 1.11 | 5.77 | tsfresh | 33.8 | 14.8× |
| `paa-3-2` | 1.88 | 0.80 | 7.64 | — |  | |
| `paa-4-1` | 1.86 | 0.75 | 6.97 | — |  | |
| `partial_autocorr-1` | 2.74 | 1.53 | 10.00 | tsfresh | 88.8 | 32.4× |
| `partial_autocorr-2` | 2.67 | 1.54 | 11.27 | tsfresh | 104.3 | 39.1× |
| `peak_count` | 2.04 | 0.74 | 5.48 | tsfresh | 16.2 | 7.9× |
| `percentage_of_reoccurring_datapoints_to_all_datapoints` | 4.30 | 0.78 | 4.96 | tsfresh | 526.3 | 122.3× |
| `percentage_of_reoccurring_values_to_all_values` | 4.21 | 0.77 | 5.87 | tsfresh | 39.7 | 9.4× |
| `permutation_entropy-1-3` | 12.66 | 7.11 | 18.76 | tsfresh | 367.2 | 29.0× |
| `permutation_entropy-1-5` | 19.94 | 10.17 | 24.82 | tsfresh | 494.8 | 24.8× |
| `pk_pk_distance` | 1.83 | 0.76 | 5.39 | tsfel | 9.6 | 5.2× |
| `positive_turning` | 2.34 | 0.76 | 5.51 | tsfel | 17.1 | 7.3× |
| `quantile-0.25` | 2.97 | 1.60 | 6.40 | tsfresh | 85.2 | 28.7× |
| `quantile-0.9` | 2.78 | 1.60 | 6.11 | tsfresh | 80.6 | 29.0× |
| `query_similarity_count-10-0.5` | 9.84 | 4.51 | 8.69 | tsfresh | 1,195.9 | 121.5× |
| `ratio_beyond_r_sigma-1.5` | 3.87 | 0.77 | 5.63 | tsfresh | 41.3 | 10.7× |
| `ratio_value_number_to_time_series_length` | 4.20 | 0.79 | 5.85 | tsfresh | 15.4 | 3.7× |
| `rms` | 1.61 | 0.77 | 5.58 | tsfel | 8.0 | 5.0× |
| `root_mean_square` | 1.95 | 0.74 | 5.22 | tsfresh | 13.5 | 6.9× |
| `sample_entropy` | 28.12 | 17.08 | 25.75 | tsfresh | 15,489.0 | 550.8× |
| `signal_distance` | 2.08 | 0.89 | 6.19 | tsfel | 21.8 | 10.5× |
| `skewness` | 3.11 | 0.79 | 5.76 | tsfresh | 101.3 | 32.6× |
| `slope` | 1.57 | 0.78 | 5.74 | tsfel | 90.6 | 57.9× |
| `slope_sign_change` | 2.37 | 0.84 | 6.02 | — |  | |
| `spectral_centroid` | 2.49 | 1.19 | 10.85 | tsfel | 34.9 | 14.0× |
| `spectral_decrease` | 2.34 | 1.18 | 10.11 | tsfel | 48.2 | 20.6× |
| `spectral_distance` | 2.55 | 1.41 | 11.00 | tsfel | 50.5 | 19.8× |
| `spectral_entropy` | 2.80 | 1.58 | 10.23 | tsfel | 61.6 | 22.0× |
| `spectral_kurtosis` | 2.37 | 1.17 | 10.12 | tsfel | 210.6 | 88.8× |
| `spectral_roll_off` | 2.31 | 1.17 | 10.22 | tsfel | 45.7 | 19.8× |
| `spectral_roll_on` | 2.48 | 1.15 | 9.95 | tsfel | 46.6 | 18.8× |
| `spectral_skewness` | 2.38 | 1.18 | 10.34 | tsfel | 239.6 | 100.6× |
| `spectral_slope` | 2.30 | 1.20 | 10.67 | tsfel | 45.0 | 19.6× |
| `spectral_spread` | 2.51 | 1.27 | 10.57 | tsfel | 70.4 | 28.1× |
| `spectrogram-2-0.5` | 3.24 | 1.20 | 11.06 | — |  | |
| `spkt_welch_density__coeff_2` | 4.29 | 2.17 | 11.72 | tsfresh | 440.3 | 102.6× |
| `spkt_welch_density__coeff_5` | 3.87 | 2.23 | 11.88 | tsfresh | 431.7 | 111.5× |
| `std_dev` | 1.40 | 0.74 | 5.84 | tsfresh | 20.7 | 14.8× |
| `sum_of_reoccurring_data_points` | 10.28 | 4.19 | 13.78 | tsfresh | 40.7 | 4.0× |
| `sum_of_reoccurring_values` | 10.60 | 4.29 | 14.37 | tsfresh | 94.9 | 9.0× |
| `symmetry_looking-0.05` | 4.05 | 1.09 | 6.39 | tsfresh | 59.5 | 14.7× |
| `time_reversal_asymmetry-1` | 2.02 | 0.85 | 7.21 | tsfresh | 20.8 | 10.3× |
| `time_reversal_asymmetry-2` | 2.41 | 0.82 | 5.61 | tsfresh | 20.7 | 8.6× |
| `total_sum` | 2.08 | 0.80 | 6.60 | tsfresh | 5.4 | 2.6× |
| `turning_points` | 1.82 | 0.73 | 5.39 | — |  | |
| `value_count-0` | 1.64 | 0.80 | 6.20 | tsfresh | 3.9 | 2.4× |
| `value_count-1` | 1.67 | 0.75 | 5.48 | tsfresh | 3.9 | 2.4× |
| `variance` | 2.49 | 1.20 | 6.63 | tsfresh | 19.4 | 7.8× |
| `variance_larger_than_standard_deviation` | 1.80 | 0.73 | 5.46 | tsfresh | 19.9 | 11.1× |
| `variation_coefficient` | 1.78 | 0.73 | 6.05 | tsfresh | 31.6 | 17.7× |
| `wavelet-0.5-1` | 2.28 | 0.81 | 5.83 | — |  | |
| `wavelet_energy-0` | 105.33 | 74.35 | 90.61 | tsfel | 1,415.6 | 13.4× |
| `wavelet_energy-3` | 103.43 | 74.08 | 94.35 | tsfel | 1,497.2 | 14.5× |
| `wavelet_entropy` | 103.11 | 73.89 | 93.83 | tsfel | 1,326.5 | 12.9× |
| `zero_cross` | 1.83 | 0.78 | 5.73 | tsfel | 10.7 | 5.8× |
| `zero_crossing_mean` | 2.28 | 1.10 | 5.83 | — |  | |
| `zero_crossing_rate` | 2.07 | 0.80 | 5.56 | — |  | |
| `zero_crossing_std` | 2.35 | 1.19 | 6.22 | — |  | |
