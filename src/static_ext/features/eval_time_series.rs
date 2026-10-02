use crate::common::ColumnState;
use crate::types::Feature;

#[inline(always)]
pub fn eval_time_series(
    feat: &Feature,
    n: f32,
    mean: f32,
    var: f32,
    m2: f32,
    mac_sum: f32,
    mc_sum: f32,
    state: &mut ColumnState,
    unique_c3_lags: &[u16],
    unique_paa_totals: &[u16],
    paa_boundaries: &[Vec<usize>],
    fft_autocorr: &[f32],
    values: &[f32],
    signal_dist: f32,
) -> Option<f32> {
    match feat {
        Feature::PeakCount => Some(state.peaks as f32),
        Feature::AutocorrLag1 if var > 1e-9 && n > 1.0 => {
            let x0 = values[0];
            let xn = values[values.len() - 1];
            let cov = state.sum_prod - mean * (2.0 * state.total_sum - x0 - xn)
                + (n - 1.0) * mean * mean;
            Some(cov / m2)
        }
        Feature::AutocorrFirst1e => {
            let threshold = 0.36787944;
            if !fft_autocorr.is_empty() {
                let mut found = false;
                let mut first_lag = 0.0;
                for (l, &val) in fft_autocorr.iter().enumerate().skip(1) {
                    if val < threshold {
                        first_lag = l as f32;
                        found = true;
                        break;
                    }
                }
                Some(if found { first_lag } else { 0.0 })
            } else if var > 1e-9 && n > 1.0 {
                let mut first_lag = 0.0;
                let m2_val = m2;
                for l in 1..values.len() {
                    let mut sum = 0.0;
                    for i in 0..values.len() - l {
                        sum += (values[i] - mean) * (values[i + l] - mean);
                    }
                    if sum / m2_val < threshold {
                        first_lag = l as f32;
                        break;
                    }
                }
                Some(first_lag)
            } else {
                Some(0.0)
            }
        }
        Feature::MeanAbsChange if n > 1.0 => Some(mac_sum / (n - 1.0)),
        Feature::MeanChange if n > 1.0 => Some(mc_sum / (n - 1.0)),
        Feature::CidCe => Some(state.sum_sq_diff.sqrt()),
        Feature::Slope => {
            let mean_i = (n - 1.0) * 0.5;
            let s_xx = (n * (n * n - 1.0)) / 12.0;
            let s_xy = state.sum_ix - n * mean_i * mean;
            Some(if s_xx.abs() > 1e-9 { s_xy / s_xx } else { 0.0 })
        }
        Feature::Intercept => {
            let mean_i = (n - 1.0) * 0.5;
            let s_xx = (n * (n * n - 1.0)) / 12.0;
            let s_xy = state.sum_ix - n * mean_i * mean;
            let slope = if s_xx.abs() > 1e-9 { s_xy / s_xx } else { 0.0 };
            Some(mean - slope * mean_i)
        }
        Feature::AbsSumChange => Some(mac_sum),
        Feature::Auc => Some(state.auc_sum),
        Feature::C3(lag) => {
            let l_idx = unique_c3_lags.iter().position(|&l| l == *lag).unwrap();
            let l = *lag as usize;
            if values.len() > 2 * l {
                Some(state.c3_sums[l_idx] / (values.len() - 2 * l) as f32)
            } else {
                Some(0.0)
            }
        }
        Feature::Paa(total, index) => {
            let t_idx = unique_paa_totals
                .iter()
                .position(|&t| t == *total)
                .unwrap();
            let b = &paa_boundaries[t_idx];
            let start = b[*index as usize];
            let end = b[*index as usize + 1];
            if start < end {
                Some(state.paa_sums[t_idx][*index as usize] / (end - start) as f32)
            } else {
                Some(0.0)
            }
        }
        Feature::Autocorr(lag) => {
            let l = *lag as usize;
            if var > 1e-9 && values.len() > l {
                let mut sum = 0.0;
                for i in 0..values.len() - l {
                    sum += (values[i] - mean) * (values[i + l] - mean);
                }
                let m2 = var * (n - 1.0);
                Some(if m2.abs() > 1e-9 { sum / m2 } else { 0.0 })
            } else {
                Some(0.0)
            }
        }
        Feature::TimeReversalAsymmetry(lag) if values.len() > 2 * *lag as usize => {
            let l = *lag as usize;
            let mut sum = 0.0;
            for i in 0..values.len() - 2 * l {
                sum += values[i + 2 * l].powi(2) * values[i + l]
                    - values[i + l] * values[i].powi(2);
            }
            Some(sum / (values.len() - 2 * l) as f32)
        }
        Feature::PartialAutocorr(lag) if var > 1e-9 && values.len() > *lag as usize => {
            let l = *lag as usize;
            let mut r = std::mem::take(&mut state.pacf_buffer);
            r.clear();
            let m2 = var * (n - 1.0);
            for k in 0..=l {
                let mut sum = 0.0;
                for i in 0..values.len() - k {
                    sum += (values[i] - mean) * (values[i + k] - mean);
                }
                r.push(sum / m2);
            }
            let mut phi = vec![vec![0.0; l + 1]; l + 1];
            let mut error = r[0];
            if error.abs() < 1e-9 {
                return Some(0.0);
            }
            phi[1][1] = r[1] / r[0];
            error *= 1.0 - phi[1][1] * phi[1][1];
            for k in 1..l {
                let mut sum = 0.0;
                for i in 1..=k {
                    sum += phi[k][i] * r[k + 1 - i];
                }
                phi[k + 1][k + 1] = (r[k + 1] - sum) / error;
                for i in 1..=k {
                    phi[k + 1][i] = phi[k][i] - phi[k + 1][k + 1] * phi[k][k + 1 - i];
                }
                error *= 1.0 - phi[k + 1][k + 1] * phi[k + 1][k + 1];
                if error.abs() < 1e-9 {
                    break;
                }
            }
            let res = phi[l][l];
            state.pacf_buffer = r;
            Some(res)
        }
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
                return Some(0.0);
            }
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
            Some(res)
        }
        Feature::ApproxEntropy(m, r_bits) if values.len() > *m as usize + 1 => {
            let m_val = *m as usize;
            let r = f32::from_bits(*r_bits);
            let mut buffer = std::mem::take(&mut state.approx_entropy_buffer);
            let res = crate::common::approx_entropy_phi(m_val, r, values, &mut buffer)
                - crate::common::approx_entropy_phi(m_val + 1, r, values, &mut buffer);
            state.approx_entropy_buffer = buffer;
            Some(res)
        }
        Feature::MaxLangevinFixedPoint(_m, _r_bits) => {
            Some(0.0)
        }
        Feature::SignalDistance => Some(signal_dist),
        _ => None,
    }
}
