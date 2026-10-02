use crate::types::{Feature, AggAttr, AggFunc, FftAttr};
use crate::common::ColumnState;

#[inline(always)]
pub fn eval_changes(
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
                Feature::MeanAbsChange => (mac_sum / n) as f32,
                Feature::MeanChange => (mc_sum / n) as f32,
                Feature::CidCe => state.sum_sq_diff.sqrt() as f32,
                Feature::Slope => {
                    let mean_i = (n - 1.0) * 0.5;
                    let s_xx = (n * (n * n - 1.0)) / 12.0;
                    let s_xy = state.sum_ix - n * mean_i * mean;
                    if s_xx.abs() > 1e-9 {
                        (s_xy / s_xx) as f32
                    } else {
                        0.0
                    }
                }
                Feature::Intercept => {
                    let mean_i = (n - 1.0) * 0.5;
                    let s_xx = (n * (n * n - 1.0)) / 12.0;
                    let s_xy = state.sum_ix - n * mean_i * mean;
                    let slope = if s_xx.abs() > 1e-9 { s_xy / s_xx } else { 0.0 };
                    (mean - slope * mean_i) as f32
                }
                Feature::AbsSumChange => mac_sum as f32,
                Feature::Auc => state.auc_sum as f32,
                Feature::AggLinearTrend(attr, chunk_len, func)
                    if values.len() >= *chunk_len as usize =>
                {
                    let cl = *chunk_len as usize;
                    let mut agg_series = std::mem::take(&mut state.agg_linear_trend_buffer);
                    agg_series.clear();
                    for chunk in values.chunks_exact(cl) {
                        let val = match func {
                            crate::types::AggFunc::Max => {
                                chunk.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b))
                            }
                            crate::types::AggFunc::Min => {
                                chunk.iter().fold(f32::INFINITY, |a, &b| a.min(b))
                            }
                            crate::types::AggFunc::Mean => chunk.iter().sum::<f32>() / cl as f32,
                            crate::types::AggFunc::Var => {
                                let m = chunk.iter().sum::<f32>() / cl as f32;
                                chunk.iter().map(|&v| (v - m).powi(2)).sum::<f32>() / cl as f32
                            }
                        };
                        agg_series.push(val);
                    }
                    let m_n = agg_series.len() as f32;
                    if m_n < 2.0 {
                        0.0
                    } else {
                        let m_sum_x: f32 = (0..agg_series.len()).map(|i| i as f32).sum();
                        let m_sum_y: f32 = agg_series.iter().sum();
                        let m_sum_xx: f32 = (0..agg_series.len()).map(|i| (i as f32).powi(2)).sum();
                        let m_sum_xy: f32 = agg_series
                            .iter()
                            .enumerate()
                            .map(|(i, &v)| i as f32 * v)
                            .sum();
                        let s_xx = m_sum_xx - (m_sum_x * m_sum_x) / m_n;
                        let s_xy = m_sum_xy - (m_sum_x * m_sum_y) / m_n;
                        let slope = if s_xx.abs() > 1e-9 { s_xy / s_xx } else { 0.0 };
                        let intercept = (m_sum_y - slope * m_sum_x) / m_n;
                        let res = match attr {
                            crate::types::AggAttr::Slope => slope,
                            crate::types::AggAttr::Intercept => intercept,
                            crate::types::AggAttr::Stderr | crate::types::AggAttr::RValue => {
                                let mut ss_res = 0.0;
                                let mut ss_tot = 0.0;
                                let m_y = m_sum_y / m_n;
                                for (i, &y) in agg_series.iter().enumerate() {
                                    let y_hat = intercept + slope * i as f32;
                                    ss_res += (y - y_hat).powi(2);
                                    ss_tot += (y - m_y).powi(2);
                                }
                                if matches!(attr, crate::types::AggAttr::Stderr) {
                                    if m_n > 2.0 && s_xx.abs() > 1e-9 {
                                        (ss_res / (m_n - 2.0) / s_xx).sqrt()
                                    } else {
                                        0.0
                                    }
                                } else {
                                    if ss_tot > 1e-9 {
                                        (1.0 - ss_res / ss_tot).sqrt() * slope.signum()
                                    } else {
                                        0.0
                                    }
                                }
                            }
                            _ => 0.0,
                        };
                        state.agg_linear_trend_buffer = agg_series;
                        res
                    }
                }
                Feature::SignalDistance => {
                    let mut dist = 0.0;
                    for i in 1..values.len() {
                        dist += ((values[i] - values[i - 1]).powi(2) + 1.0).sqrt();
                    }
                    dist
                }
        _ => return None,
    };
    Some(res)
}
