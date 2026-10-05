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

    let mut res = None;

    match feat {
        Feature::ArCoefficient(k, p) => {
            let k_val = *k as usize;
            let p_val = *p as usize;

            if p_val > k_val || n <= k_val {
                res = Some(f32::NAN);
            } else {
                let coeffs = state.ar_coeffs.entry(*k).or_insert_with(|| {
                    let (xtx, xty) = compute_ar_matrices_full(values, k_val);
                    solve_ols_f64(&xtx, &xty).unwrap_or_else(|| vec![f32::NAN; k_val + 1])
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

    res
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

pub(crate) fn solve_ols_f64(xtx: &Array2<f64>, xty: &Array1<f64>) -> Option<Vec<f32>> {
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

    // tsfresh bins with pd.qcut(signal, r): edges are linear-interpolated
    // quantiles, bins are right-closed (the first also includes its left edge),
    // and duplicate edges raise, which tsfresh turns into NaN coefficients.
    let r_int = r as usize;
    if r_int < 1 {
        return vec![f32::NAN; m + 1];
    }
    let mut sorted = signal.clone();
    sorted.sort_by(f64::total_cmp);
    let last = (sorted.len() - 1) as f64;
    let edges: Vec<f64> = (0..=r_int)
        .map(|q| {
            let pos = q as f64 / r_int as f64 * last;
            let lo = pos.floor() as usize;
            let hi = (lo + 1).min(sorted.len() - 1);
            sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo as f64)
        })
        .collect();
    if edges.windows(2).any(|w| w[0] == w[1]) {
        return vec![f32::NAN; m + 1];
    }

    let mut sum_x = vec![0.0f64; r_int];
    let mut sum_y = vec![0.0f64; r_int];
    let mut count = vec![0usize; r_int];
    for (&sx, &dy) in signal.iter().zip(&delta) {
        // First bin whose right edge is >= sx.
        let bin = edges[1..].partition_point(|&e| e < sx).min(r_int - 1);
        sum_x[bin] += sx;
        sum_y[bin] += dy;
        count[bin] += 1;
    }
    let mut x_means = Vec::with_capacity(r_int);
    let mut y_means = Vec::with_capacity(r_int);
    for q in 0..r_int {
        if count[q] > 0 {
            x_means.push(sum_x[q] / count[q] as f64);
            y_means.push(sum_y[q] / count[q] as f64);
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
