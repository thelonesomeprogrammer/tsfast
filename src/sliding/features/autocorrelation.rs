use crate::types::{Feature, AggAttr, AggFunc, FftAttr};
use crate::common::ColumnState;

#[inline(always)]
pub fn eval_autocorrelation(
    feat: &Feature,
    values: &[f32],
    state: &mut ColumnState,
    n: f32,
    mean: f32,
    m2: f32,
    m3: f32,
    m4: f32,
    mad_sum: f32,
    iqr: f32,
    entropy: f32,
    count_a: usize,
    count_b: usize,
    max_strike_a: usize,
    max_strike_b: usize,
    zc_mean: f32,
    zc_std: f32,
    freq_centroid: f32,
    spectral_decrease: f32,
    spectral_slope: f32,
    first_max_idx: usize,
    last_max_idx: usize,
    first_min_idx: usize,
    last_min_idx: usize,
    fft_autocorr: &[f32],
    fft_complex: &[realfft::num_complex::Complex<f32>],
    spectrum: &[f32],
    unique_c3_lags: &[u16],
    unique_paa_totals: &[u16],
    paa_boundaries: &[Vec<usize>],
    var: f32,
    std_dev: f32,
    mac_sum: f32,
    mc_sum: f32,
    median: f32,
) -> Option<f32> {
    let res = match feat {
                Feature::AutocorrLag1 if var > 1e-9 && n > 1.0 => {
                    let x0 = values[0];
                    let xn = values[values.len() - 1];
                    let cov = state.sum_prod - mean * (2.0 * state.total_sum - x0 - xn)
                        + (n - 1.0) * mean * mean;
                    (cov / m2) as f32
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
                    let mut sum = 0.0;
                    for i in 0..values.len() - 2 * l {
                        sum += values[i + 2 * l].powi(2) * values[i + l]
                            - values[i + l] * values[i].powi(2);
                    }
                    sum / (values.len() - 2 * l) as f32
                }
                Feature::PartialAutocorr(lag) if var > 1e-9 && values.len() > *lag as usize => {
                    let l = *lag as usize;
                    if fft_autocorr.is_empty() || fft_autocorr.len() <= l {
                        0.0
                    } else {
                        let r = &fft_autocorr;
                        let mut phi = vec![vec![0.0; l + 1]; l + 1];
                        let mut error = r[0] as f64;
                        if error.abs() < 1e-9 {
                            0.0
                        } else {
                            phi[1][1] = r[1] / r[0];
                            error *= 1.0 - (phi[1][1] * phi[1][1]) as f64;
                            for k in 1..l {
                                let mut sum = 0.0;
                                for i in 1..=k {
                                    sum += phi[k][i] * r[k + 1 - i];
                                }
                                phi[k + 1][k + 1] = (r[k + 1] - sum) / error as f32;
                                for i in 1..=k {
                                    phi[k + 1][i] =
                                        phi[k][i] - phi[k + 1][k + 1] * phi[k][k + 1 - i];
                                }
                                error *= 1.0 - (phi[k + 1][k + 1] * phi[k + 1][k + 1]) as f64;
                                if error.abs() < 1e-9 {
                                    break;
                                }
                            }
                            phi[l][l]
                        }
                    }
                }
        _ => return None,
    };
    Some(res)
}
