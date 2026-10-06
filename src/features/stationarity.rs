use super::dynamic::solve_ols_f64;
use crate::types::{AdfAttr, Feature};
use ndarray::{Array1, Array2};
use statrs::distribution::{ContinuousCDF, Normal};
use std::f32;
use std::simd::{f64x8, num::SimdFloat};

// MacKinnon (1994) critical values (regression "c")
const MAC_MAX: f64 = 2.74;
const MAC_MIN: f64 = -18.83;
const MAC_STAR: f64 = -1.61;
const TAU_SMALLPS: [f64; 3] = [2.1659, 1.4412, 0.038269];
const TAU_LARGEPS: [f64; 4] = [1.7339, 0.93202, -0.12745, -0.010368];

fn mackinnonp(teststat: f64) -> f64 {
    if teststat > MAC_MAX {
        return 1.0;
    } else if teststat < MAC_MIN {
        return 0.0;
    }
    let mut val = 0.0;
    if teststat <= MAC_STAR {
        for (_, &c) in TAU_SMALLPS.iter().rev().enumerate() {
            val = val * teststat + c;
        }
    } else {
        for (_, &c) in TAU_LARGEPS.iter().rev().enumerate() {
            val = val * teststat + c;
        }
    }
    let norm = Normal::new(0.0, 1.0).unwrap();
    norm.cdf(val)
}

pub fn eval_stationarity(
    feat: &Feature,
    context: &mut crate::context::FeatureContext,
) -> Option<f32> {
    let values = context.values;
    let n = values.len();

    let state = &mut *context.state;

    match feat {
        Feature::AugmentedDickeyFuller(attr) => {
            if state.adf_test_stat.is_nan() {
                if n < 3 {
                    return Some(f32::NAN);
                }

                // Default maxlag based on statsmodels: int(12 * (n / 100)^(1/4))
                let maxlag = (12.0 * ((n as f64) / 100.0).powf(0.25)) as usize;
                let maxlag = maxlag.min(n / 2 - 1); // Ensure we have enough data

                let mut xdiff = Vec::with_capacity(n - 1);
                for i in 1..n {
                    xdiff.push((values[i] - values[i - 1]) as f64);
                }

                let mut best_aic = f64::INFINITY;
                let mut best_lag = 0;

                let y_len = xdiff.len() - maxlag;
                if y_len <= 2 {
                    return Some(f32::NAN);
                }

                // Test lags from maxlag down to 0
                for lag in (0..=maxlag).rev() {
                    // Number of predictors: 1 (x_{t-1}) + lag (diffs) + 1 (const)
                    let k_vars = 1 + lag + 1;

                    // Create X matrix (y_len x k_vars)
                    let mut xtx = Array2::<f64>::zeros((k_vars, k_vars));
                    let mut xty = Array1::<f64>::zeros(k_vars);
                    let mut y_arr = Array1::<f64>::zeros(y_len);

                    let mut x_mat = Array2::<f64>::zeros((y_len, k_vars));

                    for t in 0..y_len {
                        let idx = t + maxlag;
                        let y_val = xdiff[idx];
                        y_arr[t] = y_val;

                        x_mat[[t, 0]] = values[idx] as f64; // x_{t-1}
                        for i in 0..lag {
                            x_mat[[t, 1 + i]] = xdiff[idx - 1 - i]; // diff terms
                        }
                        x_mat[[t, k_vars - 1]] = 1.0; // constant
                    }

                    // compute xtx and xty
                    for i in 0..k_vars {
                        let mut sum_xty = 0.0;
                        let mut t = 0;
                        let mut sum_vec_xty = f64x8::splat(0.0);
                        while t + 8 <= y_len {
                            let x_vec = f64x8::from_array([
                                x_mat[[t, i]],
                                x_mat[[t + 1, i]],
                                x_mat[[t + 2, i]],
                                x_mat[[t + 3, i]],
                                x_mat[[t + 4, i]],
                                x_mat[[t + 5, i]],
                                x_mat[[t + 6, i]],
                                x_mat[[t + 7, i]],
                            ]);
                            let y_vec = f64x8::from_slice(&y_arr.as_slice().unwrap_or(&[])[t..t + 8]);
                            sum_vec_xty += x_vec * y_vec;
                            t += 8;
                        }
                        sum_xty += sum_vec_xty.reduce_sum();
                        while t < y_len {
                            sum_xty += x_mat[[t, i]] * y_arr[t];
                            t += 1;
                        }
                        xty[i] = sum_xty;

                        for j in i..k_vars {
                            let mut sum = 0.0;
                            let mut t = 0;
                            let mut sum_vec = f64x8::splat(0.0);
                            while t + 8 <= y_len {
                                let xi_vec = f64x8::from_array([
                                    x_mat[[t, i]],
                                    x_mat[[t + 1, i]],
                                    x_mat[[t + 2, i]],
                                    x_mat[[t + 3, i]],
                                    x_mat[[t + 4, i]],
                                    x_mat[[t + 5, i]],
                                    x_mat[[t + 6, i]],
                                    x_mat[[t + 7, i]],
                                ]);
                                let xj_vec = f64x8::from_array([
                                    x_mat[[t, j]],
                                    x_mat[[t + 1, j]],
                                    x_mat[[t + 2, j]],
                                    x_mat[[t + 3, j]],
                                    x_mat[[t + 4, j]],
                                    x_mat[[t + 5, j]],
                                    x_mat[[t + 6, j]],
                                    x_mat[[t + 7, j]],
                                ]);
                                sum_vec += xi_vec * xj_vec;
                                t += 8;
                            }
                            sum += sum_vec.reduce_sum();
                            while t < y_len {
                                sum += x_mat[[t, i]] * x_mat[[t, j]];
                                t += 1;
                            }
                            xtx[[i, j]] = sum;
                            xtx[[j, i]] = sum;
                        }
                    }

                    if let Some(coeffs) = solve_ols_f64(&xtx, &xty) {
                        let mut rss = 0.0;
                        for t in 0..y_len {
                            let mut y_pred = 0.0;
                            for i in 0..k_vars {
                                y_pred += x_mat[[t, i]] * (coeffs[i] as f64);
                            }
                            let res = y_arr[t] - y_pred;
                            rss += res * res;
                        }

                        let n_obs = y_len as f64;
                        let llf = -n_obs / 2.0
                            * ((2.0 * std::f64::consts::PI).ln() + (rss / n_obs).ln() + 1.0);
                        let aic = -2.0 * llf + 2.0 * (k_vars as f64);

                        if aic < best_aic || best_aic.is_infinite() {
                            best_aic = aic;
                            best_lag = lag;
                        }
                    }
                }
                // Now re-run with best_lag using full available data
                let final_lag = best_lag;
                let mut best_t_stat = f32::NAN;

                let y_len = xdiff.len() - final_lag;
                if y_len > final_lag + 2 {
                    let k_vars = 1 + final_lag + 1;
                    let mut xtx = Array2::<f64>::zeros((k_vars, k_vars));
                    let mut xty = Array1::<f64>::zeros(k_vars);
                    let mut y_arr = Array1::<f64>::zeros(y_len);
                    let mut x_mat = Array2::<f64>::zeros((y_len, k_vars));

                    for t in 0..y_len {
                        let idx = t + final_lag;
                        y_arr[t] = xdiff[idx];
                        x_mat[[t, 0]] = values[idx] as f64;
                        for i in 0..final_lag {
                            x_mat[[t, 1 + i]] = xdiff[idx - 1 - i];
                        }
                        x_mat[[t, k_vars - 1]] = 1.0;
                    }

                    for i in 0..k_vars {
                        let mut sum_xty = 0.0;
                        let mut t = 0;
                        let mut sum_vec_xty = f64x8::splat(0.0);
                        while t + 8 <= y_len {
                            let x_vec = f64x8::from_array([
                                x_mat[[t, i]],
                                x_mat[[t + 1, i]],
                                x_mat[[t + 2, i]],
                                x_mat[[t + 3, i]],
                                x_mat[[t + 4, i]],
                                x_mat[[t + 5, i]],
                                x_mat[[t + 6, i]],
                                x_mat[[t + 7, i]],
                            ]);
                            let y_vec = f64x8::from_slice(&y_arr.as_slice().unwrap_or(&[])[t..t + 8]);
                            sum_vec_xty += x_vec * y_vec;
                            t += 8;
                        }
                        sum_xty += sum_vec_xty.reduce_sum();
                        while t < y_len {
                            sum_xty += x_mat[[t, i]] * y_arr[t];
                            t += 1;
                        }
                        xty[i] = sum_xty;

                        for j in i..k_vars {
                            let mut sum = 0.0;
                            let mut t = 0;
                            let mut sum_vec = f64x8::splat(0.0);
                            while t + 8 <= y_len {
                                let xi_vec = f64x8::from_array([
                                    x_mat[[t, i]],
                                    x_mat[[t + 1, i]],
                                    x_mat[[t + 2, i]],
                                    x_mat[[t + 3, i]],
                                    x_mat[[t + 4, i]],
                                    x_mat[[t + 5, i]],
                                    x_mat[[t + 6, i]],
                                    x_mat[[t + 7, i]],
                                ]);
                                let xj_vec = f64x8::from_array([
                                    x_mat[[t, j]],
                                    x_mat[[t + 1, j]],
                                    x_mat[[t + 2, j]],
                                    x_mat[[t + 3, j]],
                                    x_mat[[t + 4, j]],
                                    x_mat[[t + 5, j]],
                                    x_mat[[t + 6, j]],
                                    x_mat[[t + 7, j]],
                                ]);
                                sum_vec += xi_vec * xj_vec;
                                t += 8;
                            }
                            sum += sum_vec.reduce_sum();
                            while t < y_len {
                                sum += x_mat[[t, i]] * x_mat[[t, j]];
                                t += 1;
                            }
                            xtx[[i, j]] = sum;
                            xtx[[j, i]] = sum;
                        }
                    }

                    if let Some(coeffs) = solve_ols_f64(&xtx, &xty) {
                        let mut rss = 0.0;
                        for t in 0..y_len {
                            let mut y_pred = 0.0;
                            for i in 0..k_vars {
                                y_pred += x_mat[[t, i]] * (coeffs[i] as f64);
                            }
                            let res = y_arr[t] - y_pred;
                            rss += res * res;
                        }

                        let n_k = k_vars;
                        let mut a = xtx.clone();
                        let mut b = Array1::<f64>::zeros(n_k);
                        b[0] = 1.0;
                        let mut invertible = true;
                        for i in 0..n_k {
                            let mut max_row = i;
                            for row in i + 1..n_k {
                                if a[[row, i]].abs() > a[[max_row, i]].abs() {
                                    max_row = row;
                                }
                            }
                            if a[[max_row, i]].abs() < 1e-12 {
                                invertible = false;
                                break;
                            }
                            if max_row != i {
                                for j in i..n_k {
                                    let temp = a[[i, j]];
                                    a[[i, j]] = a[[max_row, j]];
                                    a[[max_row, j]] = temp;
                                }
                                let temp = b[i];
                                b[i] = b[max_row];
                                b[max_row] = temp;
                            }
                            for row in i + 1..n_k {
                                let factor = a[[row, i]] / a[[i, i]];
                                for j in i..n_k {
                                    a[[row, j]] -= factor * a[[i, j]];
                                }
                                b[row] -= factor * b[i];
                            }
                        }

                        if invertible {
                            let mut x_res = vec![0.0; n_k];
                            for i in (0..n_k).rev() {
                                let mut sum = 0.0;
                                for j in i + 1..n_k {
                                    sum += a[[i, j]] * x_res[j];
                                }
                                x_res[i] = (b[i] - sum) / a[[i, i]];
                            }
                            let inv_xtx00 = x_res[0];
                            let n_obs = y_len as f64;
                            let sigma2 = rss / (n_obs - k_vars as f64);
                            let var_coeff0 = sigma2 * inv_xtx00;

                            if var_coeff0 > 0.0 {
                                best_t_stat = (coeffs[0] as f64 / var_coeff0.sqrt()) as f32;
                            }
                        }
                    }
                }

                state.adf_test_stat = best_t_stat;
                state.adf_used_lag = best_lag as f32;
                if !best_t_stat.is_nan() {
                    state.adf_p_value = mackinnonp(best_t_stat as f64) as f32;
                }
            }

            match attr {
                AdfAttr::TestStat => Some(state.adf_test_stat),
                AdfAttr::PValue => Some(state.adf_p_value),
                AdfAttr::UsedLag => Some(state.adf_used_lag),
            }
        }
        _ => None,
    }
}
