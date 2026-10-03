use crate::types::Feature;
use std::simd::num::SimdFloat;

#[inline(always)]
pub fn eval_autocorrelation(
    feat: &Feature,
    context: &mut crate::context::FeatureContext,
) -> Option<f32> {
    let values = context.values;
    let state = &mut *context.state;
    let n = context.n;
    let mean = context.mean;
    let m2 = context.m2;
    let _m3 = context.m3;
    let _m4 = context.m4;
    let var = context.var;
    let _std_dev = context.std_dev;
    let _mac_sum = context.mac_sum;
    let _mc_sum = context.mc_sum;
    let _first_max_idx = context.first_max_idx;
    let _last_max_idx = context.last_max_idx;
    let _first_min_idx = context.first_min_idx;
    let _last_min_idx = context.last_min_idx;
    let _median = context.median;
    let _iqr = context.iqr;
    let _entropy = context.entropy;
    let _mad_sum = context.mad_sum;
    let _count_a = context.count_a;
    let _count_b = context.count_b;
    let _max_strike_a = context.max_strike_a;
    let _max_strike_b = context.max_strike_b;
    let _zc_mean = context.zc_mean;
    let _zc_std = context.zc_std;
    let _freq_centroid = context.freq_centroid;
    let _spectral_decrease = context.spectral_decrease;
    let _spectral_slope = context.spectral_slope;
    let fft_autocorr = context.fft_autocorr;
    let _fft_complex = context.fft_complex;
    let _spectrum = context.spectrum;
    let _unique_c3_lags = context.unique_c3_lags;
    let unique_tra_lags = context.unique_tra_lags;
    let _unique_paa_totals = context.unique_paa_totals;
    let _paa_boundaries = context.paa_boundaries;

    let res = match feat {
        Feature::AutocorrLag1 if var > 1e-9 && n > 1.0 => {
            let x0 = values[0];
            let xn = values[values.len() - 1];
            let cov =
                state.sum_prod - mean * (2.0 * state.total_sum - x0 - xn) + (n - 1.0) * mean * mean;
            cov / m2
        }
        Feature::AutocorrFirst1e => {
            if !fft_autocorr.is_empty() {
                let threshold = 0.36787944;
                let mut found = false;
                let mut first_lag = 0.0;
                for (l, &val) in fft_autocorr.iter().enumerate().skip(1) {
                    if val < threshold {
                        first_lag = l as f32;
                        found = true;
                        break;
                    }
                }
                if found { first_lag } else { 0.0 }
            } else {
                0.0
            }
        }
        Feature::Autocorr(lag) if var > 1e-9 && values.len() > *lag as usize => {
            let l = *lag as usize;
            if !fft_autocorr.is_empty() && l < fft_autocorr.len() {
                fft_autocorr[l]
            } else {
                0.0
            }
        }
        Feature::TimeReversalAsymmetry(lag) if values.len() > 2 * *lag as usize => {
            let l = *lag as usize;
            let l_idx = unique_tra_lags.iter().position(|&l| l == *lag);
            if let Some(idx) = l_idx {
                let n_iters = values.len() - 2 * l;
                state.tra_sums[idx] / n_iters as f32
            } else {
                let mut sum = 0.0;
                let n_iters = values.len() - 2 * l;

                // SIMD optimization
                let mut i = 0;
                let mut sum_simd = std::simd::f32x4::splat(0.0);
                while i + 3 < n_iters {
                    let v_i = std::simd::f32x4::from_slice(&values[i..i + 4]);
                    let v_il = std::simd::f32x4::from_slice(&values[i + l..i + l + 4]);
                    let v_i2l = std::simd::f32x4::from_slice(&values[i + 2 * l..i + 2 * l + 4]);
                    sum_simd += v_i2l * v_i2l * v_il - v_il * v_i * v_i;
                    i += 4;
                }
                sum += sum_simd.reduce_sum();

                // Remainder
                for j in i..n_iters {
                    sum += values[j + 2 * l].powi(2) * values[j + l] - values[j + l] * values[j].powi(2);
                }

                sum / n_iters as f32
            }
        }
        Feature::PartialAutocorr(lag) if var > 1e-9 && values.len() > *lag as usize => {
            let l = *lag as usize;
            if fft_autocorr.is_empty() || fft_autocorr.len() <= l {
                0.0
            } else {
                let r = &fft_autocorr;
                let stride = l + 1;
                let mut phi = vec![0.0; stride * stride];
                let mut error = r[0] as f64;
                if error.abs() < 1e-9 {
                    0.0
                } else {
                    phi[stride + 1] = r[1] / r[0];
                    error *= 1.0 - (phi[stride + 1] * phi[stride + 1]) as f64;
                    for k in 1..l {
                        let mut sum = 0.0;
                        for i in 1..=k {
                            sum += phi[k * stride + i] * r[k + 1 - i];
                        }
                        phi[(k + 1) * stride + (k + 1)] = (r[k + 1] - sum) / error as f32;
                        for i in 1..=k {
                            phi[(k + 1) * stride + i] = phi[k * stride + i]
                                - phi[(k + 1) * stride + (k + 1)] * phi[k * stride + (k + 1 - i)];
                        }
                        error *= 1.0
                            - (phi[(k + 1) * stride + (k + 1)] * phi[(k + 1) * stride + (k + 1)])
                                as f64;
                        if error.abs() < 1e-9 {
                            break;
                        }
                    }
                    phi[l * stride + l]
                }
            }
        }
        _ => return None,
    };
    Some(res)
}
