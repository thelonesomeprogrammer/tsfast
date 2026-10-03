use crate::types::Feature;
use ndarray::{Array1, Array2};
use realfft::num_complex::Complex;
use std::f32;
use std::f64;
use std::simd::{f64x8, num::SimdFloat};

pub fn eval_dynamic(feat: &Feature, context: &mut crate::context::FeatureContext) -> Option<f32> {
    let values = context.values;
    let n = values.len();
    if n == 0 {
        return Some(f32::NAN);
    }

    let state = &mut *context.state;

    // Cache invalidation: Use pointer address for sliding checks
    let current_ptr = values.as_ptr() as usize;
    let mut offset = 0isize;
    if state.last_dynamic_ptr != 0 && state.last_dynamic_n != 0 {
        offset = (current_ptr as isize - state.last_dynamic_ptr as isize) / 4;
    }

    if state.last_dynamic_ptr != current_ptr || state.last_dynamic_n != n {
        state.ar_coeffs.clear();
        state.friedrich_coeffs.clear();
        state.max_langevin_fixed_point_cache.clear();

        if offset < 0 || offset > state.last_dynamic_n as isize || state.dynamic_val_cache.is_empty() {
            state.ar_xtx.clear();
            state.ar_xty.clear();
        }

        state.last_dynamic_ptr = current_ptr;
        state.last_dynamic_n = n;
    }

    let mut res = None;

    match feat {
        Feature::ArCoefficient(k, p) => {
            let k_val = *k as usize;
            let p_val = *p as usize;

            if p_val > k_val || n <= k_val {
                res = Some(f32::NAN);
            } else {
                let coeffs = state.ar_coeffs.entry(*k).or_insert_with(|| {
                    let mut xtx = state.ar_xtx.entry(*k).or_insert_with(|| Array2::<f64>::zeros((k_val + 1, k_val + 1))).clone();
                    let mut xty = state.ar_xty.entry(*k).or_insert_with(|| Array1::<f64>::zeros(k_val + 1)).clone();

                    let mut is_full_recalc = true;
                    if state.dynamic_val_cache.len() == state.last_dynamic_n && offset >= 0 && offset < state.last_dynamic_n as isize {
                        is_full_recalc = false;
                    }

                    if is_full_recalc {
                        let (new_xtx, new_xty) = compute_ar_matrices_full(values, k_val);
                        xtx = new_xtx;
                        xty = new_xty;
                    } else if offset == 0 {
                        // Expanding window: just add the new values
                        let old_n = state.dynamic_val_cache.len();
                        let added_count = n - old_n;
                        if added_count > 0 {
                            update_ar_matrices_add(&mut xtx, &mut xty, values, k_val, old_n, n);
                        }
                    } else {
                        // Sliding window: subtract dropped, add new
                        // The dropped elements correspond to indices `0..offset` in `dynamic_val_cache`.
                        if state.dynamic_val_cache.len() >= k_val + offset as usize {
                            update_ar_matrices_sub(&mut xtx, &mut xty, &state.dynamic_val_cache, k_val, offset as usize);
                            update_ar_matrices_add(&mut xtx, &mut xty, values, k_val, n - offset as usize, n);
                        } else {
                            // Fallback if cache is somehow too small
                            let (new_xtx, new_xty) = compute_ar_matrices_full(values, k_val);
                            xtx = new_xtx;
                            xty = new_xty;
                        }
                    }

                    state.ar_xtx.insert(*k, xtx.clone());
                    state.ar_xty.insert(*k, xty.clone());

                    if let Some(c) = solve_ols_f64(&xtx, &xty) {
                        c
                    } else {
                        vec![f32::NAN; k_val + 1]
                    }
                });

                if p_val < coeffs.len() {
                    res = Some(coeffs[p_val]);
                } else {
                    res = Some(0.0);
                }
            }
        },
        Feature::FriedrichCoefficients(m, r_bits, coeff) => {
            let m_val = *m as usize;
            let r_val = f32::from_bits(*r_bits);
            let coeff_val = *coeff as usize;

            let coeffs = state.friedrich_coeffs.entry((*m, *r_bits)).or_insert_with(|| {
                compute_friedrich_coeffs(values, m_val, r_val)
            });

            if coeff_val < coeffs.len() {
                res = Some(coeffs[coeff_val]);
            } else {
                res = Some(f32::NAN);
            }
        },
        Feature::MaxLangevinFixedPoint(m, r_bits) => {
            let m_val = *m as usize;
            let r_val = f32::from_bits(*r_bits);

            let max_root = *state.max_langevin_fixed_point_cache.entry((*m, *r_bits)).or_insert_with(|| {
                let coeffs = state.friedrich_coeffs.entry((*m, *r_bits)).or_insert_with(|| {
                    compute_friedrich_coeffs(values, m_val, r_val)
                });
                compute_max_langevin(coeffs)
            });
            res = Some(max_root);
        },
        _ => return None,
    }

    if state.dynamic_val_cache.len() != n || (n > 0 && state.dynamic_val_cache.last() != Some(&values[n-1])) {
        state.dynamic_val_cache.clear();
        state.dynamic_val_cache.extend_from_slice(values);
    }

    res
}

fn update_ar_matrices_add(xtx: &mut Array2<f64>, xty: &mut Array1<f64>, values: &[f32], k: usize, start_t: usize, end_t: usize) {
    let m = k + 1;
    let v64: Vec<f64> = values.iter().map(|&x| x as f64).collect();

    // Safety clamp
    let start_t = start_t.max(k);
    let end_t = end_t.max(k);

    for i in 0..m {
        let mut sum_y = 0.0;
        if i == 0 {
            for t in start_t..end_t {
                sum_y += v64[t];
            }
        } else {
            for t in start_t..end_t {
                sum_y += v64[t-i] * v64[t];
            }
        }
        xty[i] += sum_y;

        for j in i..m {
            let mut sum_x = 0.0;
            if i == 0 && j == 0 {
                sum_x = (end_t - start_t) as f64;
            } else if i == 0 {
                for t in start_t..end_t {
                    sum_x += v64[t-j];
                }
            } else {
                for t in start_t..end_t {
                    sum_x += v64[t-i] * v64[t-j];
                }
            }
            xtx[[i, j]] += sum_x;
            if i != j {
                xtx[[j, i]] += sum_x;
            }
        }
    }
}

fn update_ar_matrices_sub(xtx: &mut Array2<f64>, xty: &mut Array1<f64>, old_values: &[f32], k: usize, offset: usize) {
    let m = k + 1;
    let v64: Vec<f64> = old_values.iter().map(|&x| x as f64).collect();
    let end_t = k + offset;

    for i in 0..m {
        let mut sum_y = 0.0;
        if i == 0 {
            for t in k..end_t {
                sum_y += v64[t];
            }
        } else {
            for t in k..end_t {
                sum_y += v64[t-i] * v64[t];
            }
        }
        xty[i] -= sum_y;

        for j in i..m {
            let mut sum_x = 0.0;
            if i == 0 && j == 0 {
                sum_x = offset as f64;
            } else if i == 0 {
                for t in k..end_t {
                    sum_x += v64[t-j];
                }
            } else {
                for t in k..end_t {
                    sum_x += v64[t-i] * v64[t-j];
                }
            }
            xtx[[i, j]] -= sum_x;
            if i != j {
                xtx[[j, i]] -= sum_x;
            }
        }
    }
}

fn compute_ar_matrices_full(values: &[f32], k: usize) -> (Array2<f64>, Array1<f64>) {
    let m = k + 1;
    let mut xtx = Array2::<f64>::zeros((m, m));
    let mut xty = Array1::<f64>::zeros(m);

    let v64: Vec<f64> = values.iter().map(|&x| x as f64).collect();
    let n = values.len();

    // Use SIMD for full matrices
    for i in 0..m {
        if i == 0 {
            let mut sum = 0.0;
            let mut t = k;
            let mut sum_vec = f64x8::splat(0.0);
            while t + 8 <= n {
                let y_vec = f64x8::from_slice(&v64[t..t+8]);
                sum_vec += y_vec;
                t += 8;
            }
            sum += sum_vec.reduce_sum();
            while t < n {
                sum += v64[t];
                t += 1;
            }
            xty[i] = sum;
        } else {
            let mut sum = 0.0;
            let mut t = k;
            let mut sum_vec = f64x8::splat(0.0);
            while t + 8 <= n {
                let x_vec = f64x8::from_slice(&v64[t-i..t-i+8]);
                let y_vec = f64x8::from_slice(&v64[t..t+8]);
                sum_vec += x_vec * y_vec;
                t += 8;
            }
            sum += sum_vec.reduce_sum();
            while t < n {
                sum += v64[t-i] * v64[t];
                t += 1;
            }
            xty[i] = sum;
        }

        for j in i..m {
            let mut sum = 0.0;
            if i == 0 && j == 0 {
                sum = (n - k) as f64;
            } else if i == 0 {
                let mut t = k;
                let mut sum_vec = f64x8::splat(0.0);
                while t + 8 <= n {
                    let x_vec = f64x8::from_slice(&v64[t-j..t-j+8]);
                    sum_vec += x_vec;
                    t += 8;
                }
                sum += sum_vec.reduce_sum();
                while t < n {
                    sum += v64[t-j];
                    t += 1;
                }
            } else {
                let mut t = k;
                let mut sum_vec = f64x8::splat(0.0);
                while t + 8 <= n {
                    let xi_vec = f64x8::from_slice(&v64[t-i..t-i+8]);
                    let xj_vec = f64x8::from_slice(&v64[t-j..t-j+8]);
                    sum_vec += xi_vec * xj_vec;
                    t += 8;
                }
                sum += sum_vec.reduce_sum();
                while t < n {
                    sum += v64[t-i] * v64[t-j];
                    t += 1;
                }
            }
            xtx[[i, j]] = sum;
            xtx[[j, i]] = sum;
        }
    }

    (xtx, xty)
}

fn solve_ols_f64(xtx: &Array2<f64>, xty: &Array1<f64>) -> Option<Vec<f32>> {
    let n = xtx.nrows();
    let mut a = xtx.clone();
    let mut b = xty.clone();

    // Gaussian elimination with partial pivoting
    for i in 0..n {
        let mut max_row = i;
        for k in i + 1..n {
            if a[[k, i]].abs() > a[[max_row, i]].abs() {
                max_row = k;
            }
        }

        if a[[max_row, i]].abs() < 1e-12 {
            return None;
        }

        if max_row != i {
            for j in i..n {
                let temp = a[[i, j]];
                a[[i, j]] = a[[max_row, j]];
                a[[max_row, j]] = temp;
            }
            let temp = b[i];
            b[i] = b[max_row];
            b[max_row] = temp;
        }

        for k in i + 1..n {
            let factor = a[[k, i]] / a[[i, i]];
            for j in i..n {
                a[[k, j]] -= factor * a[[i, j]];
            }
            b[k] -= factor * b[i];
        }
    }

    let mut x_res = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = 0.0;
        for j in i + 1..n {
            sum += a[[i, j]] * x_res[j];
        }
        x_res[i] = (b[i] - sum) / a[[i, i]];
    }

    Some(x_res.iter().map(|&v| v as f32).collect())
}

fn solve_ols_polyfit(x: &Array2<f64>, y: &Array1<f64>) -> Option<Vec<f32>> {
    let xt = x.t();
    let xtx = xt.dot(x);
    let xty = xt.dot(y);
    solve_ols_f64(&xtx, &xty)
}

fn compute_ar_coeffs(values: &[f32], k: usize) -> Vec<f32> {
    let n = values.len();
    if n <= k {
        return vec![f32::NAN; k + 1];
    }

    let m = k + 1;
    let mut xtx = Array2::<f64>::zeros((m, m));
    let mut xty = Array1::<f64>::zeros(m);

    let v64: Vec<f64> = values.iter().map(|&x| x as f64).collect();

    for i in 0..m {
        if i == 0 {
            let mut sum = 0.0;
            let mut t = k;
            let mut sum_vec = f64x8::splat(0.0);
            while t + 8 <= n {
                let y_vec = f64x8::from_slice(&v64[t..t+8]);
                sum_vec += y_vec;
                t += 8;
            }
            sum += sum_vec.reduce_sum();
            while t < n {
                sum += v64[t];
                t += 1;
            }
            xty[i] = sum;
        } else {
            let mut sum = 0.0;
            let mut t = k;
            let mut sum_vec = f64x8::splat(0.0);
            while t + 8 <= n {
                let x_vec = f64x8::from_slice(&v64[t-i..t-i+8]);
                let y_vec = f64x8::from_slice(&v64[t..t+8]);
                sum_vec += x_vec * y_vec;
                t += 8;
            }
            sum += sum_vec.reduce_sum();
            while t < n {
                sum += v64[t-i] * v64[t];
                t += 1;
            }
            xty[i] = sum;
        }

        for j in i..m {
            let mut sum = 0.0;
            if i == 0 && j == 0 {
                sum = (n - k) as f64;
            } else if i == 0 {
                let mut t = k;
                let mut sum_vec = f64x8::splat(0.0);
                while t + 8 <= n {
                    let x_vec = f64x8::from_slice(&v64[t-j..t-j+8]);
                    sum_vec += x_vec;
                    t += 8;
                }
                sum += sum_vec.reduce_sum();
                while t < n {
                    sum += v64[t-j];
                    t += 1;
                }
            } else {
                let mut t = k;
                let mut sum_vec = f64x8::splat(0.0);
                while t + 8 <= n {
                    let xi_vec = f64x8::from_slice(&v64[t-i..t-i+8]);
                    let xj_vec = f64x8::from_slice(&v64[t-j..t-j+8]);
                    sum_vec += xi_vec * xj_vec;
                    t += 8;
                }
                sum += sum_vec.reduce_sum();
                while t < n {
                    sum += v64[t-i] * v64[t-j];
                    t += 1;
                }
            }
            xtx[[i, j]] = sum;
            xtx[[j, i]] = sum;
        }
    }

    if let Some(coeffs) = solve_ols_f64(&xtx, &xty) {
        coeffs
    } else {
        vec![f32::NAN; k + 1]
    }
}

fn compute_friedrich_coeffs(values: &[f32], m: usize, r: f32) -> Vec<f32> {
    let n = values.len();
    if n < 2 {
        return vec![f32::NAN; m + 1];
    }

    let mut signal = Vec::with_capacity(n - 1);
    let mut delta = Vec::with_capacity(n - 1);

    for i in 0..n - 1 {
        signal.push(values[i] as f64);
        delta.push((values[i + 1] - values[i]) as f64);
    }

    let r_int = r as usize;
    if r_int < 1 {
        return vec![f32::NAN; m + 1];
    }

    let mut pairs: Vec<(usize, f64)> = signal.iter().cloned().enumerate().collect();
    pairs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let total = pairs.len();
    let mut x_means = Vec::new();
    let mut y_means = Vec::new();

    for q in 0..r_int {
        let start = (q * total) / r_int;
        let end = ((q + 1) * total) / r_int;

        if end > start {
            let mut sum_x = 0.0;
            let mut sum_y = 0.0;
            let count = (end - start) as f64;

            for i in start..end {
                let idx = pairs[i].0;
                sum_x += signal[idx];
                sum_y += delta[idx];
            }

            x_means.push(sum_x / count);
            y_means.push(sum_y / count);
        }
    }

    if x_means.is_empty() {
        return vec![f32::NAN; m + 1];
    }

    let n_pts = x_means.len();
    let mut x_mat_vec = Vec::with_capacity(n_pts * (m + 1));

    for i in 0..n_pts {
        let x_val = x_means[i];
        for j in 0..=m {
            x_mat_vec.push(x_val.powi((m - j) as i32));
        }
    }

    let x = Array2::from_shape_vec((n_pts, m + 1), x_mat_vec).unwrap();
    let y = Array1::from(y_means);

    if let Some(coeffs) = solve_ols_polyfit(&x, &y) {
        coeffs
    } else {
        vec![f32::NAN; m + 1]
    }
}

fn compute_max_langevin(coeffs: &[f32]) -> f32 {
    if coeffs.iter().any(|c| c.is_nan()) {
        return f32::NAN;
    }

    let m = coeffs.len() - 1;
    if m == 0 {
        return f32::NAN;
    }

    let c0 = coeffs[0];
    if c0.abs() < 1e-7 {
        return f32::NAN;
    }

    let mut norm_coeffs = Vec::with_capacity(m + 1);
    for c in coeffs {
        norm_coeffs.push(c / c0);
    }

    let mut roots = vec![Complex::new(0.0, 0.0); m];
    let radius = 1.0;
    let angle_step = std::f32::consts::TAU / (m as f32);
    for i in 0..m {
        let angle = angle_step * (i as f32);
        roots[i] = Complex::new(radius * angle.cos(), radius * angle.sin());
    }

    for _iter in 0..100 {
        let mut max_diff = 0.0f32;
        for i in 0..m {
            let mut p = Complex::new(norm_coeffs[0], 0.0);
            for j in 1..=m {
                p = p * roots[i] + Complex::new(norm_coeffs[j], 0.0);
            }

            let mut prod = Complex::new(1.0, 0.0);
            for j in 0..m {
                if i != j {
                    prod = prod * (roots[i] - roots[j]);
                }
            }

            let change = p / prod;
            roots[i] = roots[i] - change;

            if change.norm() > max_diff {
                max_diff = change.norm();
            }
        }

        if max_diff < 1e-5 {
            break;
        }
    }

    let mut max_real = f32::NEG_INFINITY;
    let mut found = false;
    for r in roots {
        if r.im.abs() < 1e-4 {
            if r.re > max_real {
                max_real = r.re;
            }
            found = true;
        }
    }

    if found {
        max_real
    } else {
        f32::NAN
    }
}
