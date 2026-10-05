# Per-feature benchmark

Microseconds per window per series. Window 256, 20 series, 32 new values per sliding/expanding update. Commit c64693d, 2026-10-05 19:42.
Sliding = per new window; expanding = per update. Reference = the tsfresh/TSFEL
function tsfast is tested against, called on one window; speed-up = reference / static.

| feature | static | sliding | expanding | reference | ref µs | speed-up |
|:--|--:|--:|--:|:--|--:|--:|
| `abs_max` | 2.43 | 0.34 | 3.22 | tsfresh | 3.1 | 1.3× |
| `abs_sum_change` | 2.74 | 0.41 | 3.15 | tsfresh | 5.4 | 2.0× |
| `agg_autocorrelation-max-5` | 6.09 | 2.92 | 8.60 | tsfresh | 49.1 | 8.1× |
| `agg_autocorrelation-mean-10` | 6.58 | 3.18 | 8.25 | tsfresh | 50.7 | 7.7× |
| `agg_autocorrelation-var-5` | 7.33 | 3.22 | 8.90 | tsfresh | 60.8 | 8.3× |
| `agg_linear_trend-intercept-5-max` | 3.18 | 0.53 | 3.02 | tsfresh | 308.3 | 97.1× |
| `agg_linear_trend-rvalue-5-var` | 3.12 | 0.61 | 3.39 | tsfresh | 729.9 | 233.7× |
| `agg_linear_trend-slope-10-mean` | 2.88 | 0.43 | 2.96 | tsfresh | 297.1 | 103.2× |
| `approx_entropy-2-0.1` | 41.59 | 32.20 | 50.43 | tsfresh | 5,595.0 | 134.5× |
| `approx_entropy-2-0.5` | 37.66 | 32.07 | 46.92 | tsfresh | 5,573.8 | 148.0× |
| `ar_coefficient-10-0` | 4.54 | 1.60 | 4.82 | tsfresh | 922.7 | 203.2× |
| `ar_coefficient-10-1` | 4.39 | 1.51 | 4.68 | tsfresh | 920.5 | 209.5× |
| `auc` | 2.88 | 0.42 | 3.36 | tsfel | 12.0 | 4.1× |
| `augmented_dickey_fuller-pvalue` | 60.25 | 43.21 | 67.12 | tsfresh | 3,586.1 | 59.5× |
| `augmented_dickey_fuller-teststat` | 65.26 | 47.46 | 67.43 | tsfresh | 3,539.5 | 54.2× |
| `augmented_dickey_fuller-usedlag` | 60.15 | 44.43 | 72.73 | tsfresh | 3,497.1 | 58.1× |
| `autocorr-1` | 6.19 | 2.93 | 8.12 | tsfresh | 36.2 | 5.8× |
| `autocorr-3` | 6.58 | 3.03 | 8.23 | tsfresh | 37.4 | 5.7× |
| `autocorr_lag1` | 3.14 | 0.42 | 3.25 | tsfresh | 36.2 | 11.5× |
| `autocorrelation` | 6.14 | 2.92 | 8.64 | tsfel | 36.4 | 5.9× |
| `benford_correlation` | 4.82 | 2.09 | 4.85 | tsfresh | 403.6 | 83.8× |
| `biased_fisher_kurtosis` | 2.88 | 0.38 | 3.12 | tsfel | 277.4 | 96.3× |
| `biased_skewness` | 2.79 | 0.38 | 3.48 | tsfel | 278.6 | 100.0× |
| `binned_entropy__max_bins_5` | 3.08 | 0.67 | 3.58 | tsfresh | 65.0 | 21.1× |
| `c3-1` | 2.63 | 0.41 | 2.86 | tsfresh | 8.3 | 3.1× |
| `c3-2` | 2.82 | 0.43 | 2.78 | tsfresh | 8.1 | 2.9× |
| `calc_centroid-100` | 2.91 | 0.34 | 3.43 | tsfel | 7.8 | 2.7× |
| `calc_centroid-50` | 3.17 | 0.39 | 3.36 | tsfel | 7.7 | 2.4× |
| `change_quantiles-0-1-False-var` | 3.62 | 1.14 | 3.87 | tsfresh | 421.6 | 116.4× |
| `change_quantiles-0.2-0.8-True-mean` | 3.59 | 1.08 | 4.37 | tsfresh | 416.4 | 115.9× |
| `cid_ce` | 2.81 | 0.40 | 3.20 | tsfresh | 3.6 | 1.3× |
| `count_above_mean` | 2.90 | 0.56 | 3.21 | tsfresh | 5.3 | 1.8× |
| `count_below_mean` | 3.01 | 0.44 | 3.35 | tsfresh | 5.5 | 1.8× |
| `cwt_coefficients__coeff_3__w_5__widths_(2, 5, 10, 20)` | 8.07 | 3.72 | 8.35 | tsfresh | 597.9 | 74.1× |
| `ecdf-10` | 3.33 | 0.39 | 3.05 | tsfel | 6.1 | 1.8× |
| `ecdf-3` | 2.63 | 0.34 | 3.08 | tsfel | 5.8 | 2.2× |
| `energy` | 2.76 | 0.36 | 3.10 | tsfresh | 1.0 | 0.4× |
| `energy_ratio_by_chunks_num_segments_3__segment_focus_1` | 3.22 | 0.41 | 3.46 | tsfresh | 13.5 | 4.2× |
| `energy_ratio_by_chunks_num_segments_4__segment_focus_3` | 2.74 | 0.43 | 3.05 | tsfresh | 14.9 | 5.4× |
| `entropy` | 4.46 | 1.92 | 4.87 | tsfel | 37.7 | 8.4× |
| `fft_coeff-0-real` | 3.23 | 0.65 | 4.26 | tsfresh | 8.1 | 2.5× |
| `fft_coeff-1-imag` | 3.11 | 0.81 | 4.32 | tsfresh | 8.1 | 2.6× |
| `fft_coeff-2-angle` | 3.20 | 0.68 | 4.42 | tsfresh | 10.3 | 3.2× |
| `fft_coeff-3-abs` | 3.30 | 0.68 | 4.02 | tsfresh | 9.3 | 2.8× |
| `first_loc_max` | 3.00 | 0.50 | 3.43 | tsfresh | 1.4 | 0.5× |
| `first_loc_min` | 3.32 | 0.51 | 3.06 | tsfresh | 1.4 | 0.4× |
| `friedrich_coefficients-3-30-0` | 5.81 | 2.79 | 6.24 | tsfresh | 3,126.5 | 537.9× |
| `friedrich_coefficients-3-30-3` | 5.66 | 2.63 | 6.29 | tsfresh | 3,240.8 | 573.1× |
| `has_duplicate` | 4.62 | 1.52 | 5.46 | tsfresh | 9.3 | 2.0× |
| `has_duplicate_max` | 2.86 | 0.43 | 3.50 | tsfresh | 6.1 | 2.1× |
| `has_duplicate_min` | 2.84 | 0.42 | 3.54 | tsfresh | 6.1 | 2.1× |
| `higuchi_fd` | 5.02 | 1.99 | 5.41 | tsfel | 4,428.6 | 881.5× |
| `human_range_energy-100` | 3.19 | 0.64 | 5.30 | tsfel | 27.4 | 8.6× |
| `index_mass_quantile-0.5` | 2.76 | 0.52 | 3.69 | tsfresh | 11.4 | 4.1× |
| `index_mass_quantile-0.9` | 3.06 | 0.52 | 2.95 | tsfresh | 10.0 | 3.3× |
| `intercept` | 2.85 | 0.38 | 3.53 | tsfresh | 214.9 | 75.3× |
| `iqr` | 3.29 | 1.04 | 3.71 | tsfel | 86.1 | 26.2× |
| `kurtosis` | 2.73 | 0.37 | 2.80 | tsfresh | 51.2 | 18.8× |
| `large_standard_deviation-0.05` | 3.08 | 0.36 | 3.28 | tsfresh | 16.8 | 5.4× |
| `last_loc_max` | 2.84 | 0.48 | 3.41 | tsfresh | 2.1 | 0.8× |
| `last_loc_min` | 3.00 | 0.59 | 2.68 | tsfresh | 2.1 | 0.7× |
| `length` | 2.78 | 0.36 | 3.65 | tsfresh | 0.1 | 0.0× |
| `linear_trend-intercept` | 3.04 | 0.37 | 3.28 | tsfresh | 213.4 | 70.1× |
| `linear_trend-pvalue` | 3.00 | 0.41 | 3.36 | tsfresh | 213.5 | 71.1× |
| `linear_trend-rvalue` | 3.01 | 0.36 | 3.23 | tsfresh | 213.4 | 70.9× |
| `linear_trend-slope` | 2.62 | 0.44 | 2.83 | tsfresh | 223.7 | 85.4× |
| `linear_trend-stderr` | 2.92 | 0.41 | 3.19 | tsfresh | 213.1 | 73.0× |
| `longest_strike_above_mean` | 3.09 | 0.45 | 3.40 | tsfresh | 370.1 | 119.7× |
| `longest_strike_below_mean` | 2.91 | 0.53 | 3.04 | tsfresh | 378.2 | 129.8× |
| `lpcc-0` | 6.57 | 3.33 | 7.55 | tsfel | 107.0 | 16.3× |
| `lpcc-3` | 8.03 | 3.17 | 7.42 | tsfel | 108.1 | 13.5× |
| `mad` | 2.88 | 0.40 | 3.40 | tsfel | 8.3 | 2.9× |
| `matrix_profile-10-max` | 287.99 | 249.42 | 351.99 | stumpy | 1,690.8 | 5.9× |
| `matrix_profile-10-mean` | 283.11 | 247.13 | 317.50 | stumpy | 1,557.3 | 5.5× |
| `matrix_profile-10-min` | 270.06 | 242.76 | 357.25 | stumpy | 1,578.3 | 5.8× |
| `max_langevin_fixed_point-3-30` | 6.14 | 2.89 | 6.66 | tsfresh | 3,187.2 | 519.4× |
| `max_power_spectrum` | 4.98 | 1.07 | 5.94 | tsfel | 209.8 | 42.1× |
| `max_value` | 3.02 | 0.42 | 3.22 | tsfresh | 2.3 | 0.8× |
| `mean` | 2.94 | 0.36 | 3.44 | tsfresh | 3.9 | 1.3× |
| `mean_abs_change` | 3.11 | 0.39 | 3.59 | tsfresh | 6.9 | 2.2× |
| `mean_change` | 2.83 | 0.38 | 3.45 | tsfresh | 0.5 | 0.2× |
| `mean_n_absolute_max-7` | 2.72 | 0.68 | 4.19 | tsfresh | 6.7 | 2.4× |
| `mean_second_derivative_central` | 2.51 | 0.39 | 2.83 | tsfresh | 0.7 | 0.3× |
| `median` | 3.04 | 0.59 | 4.00 | tsfresh | 13.4 | 4.4× |
| `median_abs_deviation` | 3.26 | 0.77 | 4.23 | tsfel | 219.9 | 67.5× |
| `median_abs_diff` | 3.42 | 0.60 | 3.80 | tsfel | 17.6 | 5.2× |
| `median_diff` | 3.73 | 0.65 | 3.96 | tsfel | 16.4 | 4.4× |
| `mfcc-0` | 8.27 | 3.18 | 8.19 | tsfel | 653.5 | 79.0× |
| `mfcc-3` | 9.30 | 3.07 | 8.01 | tsfel | 631.9 | 68.0× |
| `min_value` | 2.65 | 0.39 | 3.52 | tsfresh | 2.3 | 0.9× |
| `negative_turning` | 2.76 | 0.34 | 3.07 | tsfel | 9.4 | 3.4× |
| `number_crossing_m__m_0` | 2.65 | 0.35 | 3.17 | tsfresh | 4.2 | 1.6× |
| `number_crossing_m__m_0.5` | 3.05 | 0.40 | 3.53 | tsfresh | 3.9 | 1.3× |
| `number_cwt_peaks__n_1` | 10.82 | 4.54 | 11.27 | tsfresh | 2,392.7 | 221.2× |
| `number_cwt_peaks__n_5` | 28.01 | 17.02 | 30.43 | tsfresh | 3,324.5 | 118.7× |
| `number_peaks__n_1` | 3.01 | 0.49 | 2.94 | tsfresh | 8.9 | 3.0× |
| `number_peaks__n_3` | 3.37 | 0.53 | 3.30 | tsfresh | 18.9 | 5.6× |
| `paa-3-2` | 3.05 | 0.41 | 3.12 | — |  | |
| `paa-4-1` | 2.70 | 0.42 | 3.19 | — |  | |
| `partial_autocorr-1` | 7.17 | 2.93 | 8.08 | tsfresh | 38.6 | 5.4× |
| `partial_autocorr-2` | 7.45 | 3.04 | 8.65 | tsfresh | 44.2 | 5.9× |
| `peak_count` | 2.87 | 0.33 | 3.19 | tsfresh | 8.1 | 2.8× |
| `percentage_of_reoccurring_datapoints_to_all_datapoints` | 5.01 | 0.41 | 3.21 | tsfresh | 184.0 | 36.8× |
| `percentage_of_reoccurring_values_to_all_values` | 4.43 | 0.39 | 3.03 | tsfresh | 19.0 | 4.3× |
| `permutation_entropy-1-3` | 5.24 | 2.83 | 5.70 | tsfresh | 157.4 | 30.0× |
| `permutation_entropy-1-5` | 8.77 | 3.67 | 9.50 | tsfresh | 218.6 | 24.9× |
| `pk_pk_distance` | 2.69 | 0.34 | 3.28 | tsfel | 4.9 | 1.8× |
| `positive_turning` | 2.66 | 0.37 | 3.30 | tsfel | 9.5 | 3.6× |
| `quantile-0.25` | 3.16 | 0.92 | 3.89 | tsfresh | 41.7 | 13.2× |
| `quantile-0.9` | 3.05 | 0.92 | 3.95 | tsfresh | 45.5 | 14.9× |
| `query_similarity_count-10-0.5` | 6.16 | 2.61 | 6.40 | tsfresh | 516.3 | 83.8× |
| `ratio_beyond_r_sigma-1.5` | 2.94 | 0.37 | 3.52 | tsfresh | 20.9 | 7.1× |
| `ratio_value_number_to_time_series_length` | 4.02 | 0.38 | 3.43 | tsfresh | 7.3 | 1.8× |
| `rms` | 2.93 | 0.37 | 3.53 | tsfel | 4.0 | 1.4× |
| `root_mean_square` | 3.14 | 0.37 | 3.38 | tsfresh | 4.4 | 1.4× |
| `sample_entropy` | 12.44 | 6.13 | 14.00 | tsfresh | 8,398.1 | 675.1× |
| `signal_distance` | 2.94 | 0.44 | 3.37 | tsfel | 9.4 | 3.2× |
| `skewness` | 2.82 | 0.35 | 3.35 | tsfresh | 45.7 | 16.2× |
| `slope` | 3.02 | 0.37 | 3.31 | tsfel | 49.3 | 16.3× |
| `slope_sign_change` | 2.63 | 0.40 | 3.45 | — |  | |
| `spectral_centroid` | 3.32 | 0.57 | 5.12 | tsfel | 17.6 | 5.3× |
| `spectral_decrease` | 3.09 | 0.70 | 5.07 | tsfel | 23.2 | 7.5× |
| `spectral_distance` | 3.29 | 0.71 | 5.05 | tsfel | 25.6 | 7.8× |
| `spectral_entropy` | 3.58 | 0.79 | 5.25 | tsfel | 28.6 | 8.0× |
| `spectral_kurtosis` | 3.40 | 0.64 | 5.05 | tsfel | 115.3 | 33.9× |
| `spectral_roll_off` | 3.34 | 0.68 | 5.06 | tsfel | 18.0 | 5.4× |
| `spectral_roll_on` | 3.14 | 0.62 | 5.31 | tsfel | 18.2 | 5.8× |
| `spectral_skewness` | 3.31 | 0.61 | 5.13 | tsfel | 113.0 | 34.1× |
| `spectral_slope` | 3.47 | 0.65 | 4.04 | tsfel | 20.1 | 5.8× |
| `spectral_spread` | 3.42 | 0.70 | 4.49 | tsfel | 37.4 | 11.0× |
| `spectrogram-2-0.5` | 3.67 | 0.63 | 5.06 | — |  | |
| `spkt_welch_density__coeff_2` | 4.77 | 1.18 | 5.55 | tsfresh | 178.8 | 37.5× |
| `spkt_welch_density__coeff_5` | 7.31 | 1.44 | 5.42 | tsfresh | 184.2 | 25.2× |
| `std_dev` | 2.78 | 0.36 | 3.53 | tsfresh | 11.0 | 4.0× |
| `sum_of_reoccurring_data_points` | 4.08 | 1.68 | 5.36 | tsfresh | 19.7 | 4.8× |
| `sum_of_reoccurring_values` | 4.23 | 1.60 | 5.30 | tsfresh | 20.3 | 4.8× |
| `symmetry_looking-0.05` | 3.19 | 0.58 | 3.73 | tsfresh | 25.6 | 8.0× |
| `time_reversal_asymmetry-1` | 3.01 | 0.39 | 3.19 | tsfresh | 9.8 | 3.3× |
| `time_reversal_asymmetry-2` | 2.67 | 0.42 | 2.89 | tsfresh | 10.2 | 3.8× |
| `total_sum` | 2.79 | 0.41 | 3.47 | tsfresh | 2.3 | 0.8× |
| `turning_points` | 2.84 | 0.37 | 3.29 | — |  | |
| `value_count-0` | 2.94 | 0.37 | 3.42 | tsfresh | 2.4 | 0.8× |
| `value_count-1` | 2.68 | 0.37 | 3.53 | tsfresh | 2.5 | 0.9× |
| `variance` | 2.74 | 0.41 | 3.13 | tsfresh | 10.4 | 3.8× |
| `variance_larger_than_standard_deviation` | 2.93 | 0.37 | 3.33 | tsfresh | 10.7 | 3.6× |
| `variation_coefficient` | 2.57 | 0.41 | 3.25 | tsfresh | 14.8 | 5.8× |
| `wavelet-0.5-1` | 3.01 | 0.40 | 3.61 | — |  | |
| `wavelet_energy-0` | 40.05 | 30.89 | 48.75 | tsfel | 693.2 | 17.3× |
| `wavelet_energy-3` | 41.16 | 30.58 | 49.67 | tsfel | 695.6 | 16.9× |
| `wavelet_entropy` | 43.88 | 30.38 | 49.06 | tsfel | 678.1 | 15.5× |
| `zero_cross` | 2.92 | 0.36 | 3.59 | tsfel | 4.5 | 1.5× |
| `zero_crossing_mean` | 2.97 | 0.57 | 3.15 | — |  | |
| `zero_crossing_rate` | 2.64 | 0.43 | 2.95 | — |  | |
| `zero_crossing_std` | 3.23 | 0.89 | 3.22 | — |  | |
