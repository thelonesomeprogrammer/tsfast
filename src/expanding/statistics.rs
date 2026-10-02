use crate::common::ColumnState;
use crate::types::Feature;

#[inline(always)]
pub fn eval_statistics(
    feat: &Feature,
    n: f32,
    mean: f32,
    var: f32,
    std_dev: f32,
    m2: f32,
    m3: f32,
    m4: f32,
    mad_sum: f32,
    median: f32,
    iqr: f32,
    entropy: f32,
    count_a: usize,
    count_b: usize,
    max_strike_a: usize,
    max_strike_b: usize,
    zc_mean: f32,
    zc_std: f32,
    state: &mut ColumnState,
    running_sorted: &Vec<f32>,
    full_series: &[f32],
) -> Option<f32> {
    match feat {
        Feature::Median => Some(median),
        Feature::Skew if var > 1e-9 => {
            let mu2 = m2 / n;
            Some((m3 / n) / mu2.powf(1.5))
        }
        Feature::UnbiasedFisherKurtosis if var > 1e-9 && n > 3.0 => {
            let mu2 = m2 / n;
            let g2 = (m4 / n) / (mu2 * mu2) - 3.0;
            Some(((n - 1.0) / ((n - 2.0) * (n - 3.0))) * ((n + 1.0) * g2 + 6.0))
        }
        Feature::BiasedFisherKurtosis if var > 1e-9 => {
            let mu2 = m2 / n;
            Some((m4 / n) / (mu2 * mu2) - 3.0)
        }
        Feature::Mad => Some(mad_sum / n),
        Feature::Iqr => Some(iqr),
        Feature::Entropy => Some(entropy),
        Feature::PeakCount => Some(state.peaks as f32),
        Feature::Slope => {
            let mean_i = (n - 1.0) * 0.5;
            let s_xx = (n * (n * n - 1.0)) / 12.0;
            let s_xy = state.sum_ix - n * mean_i * mean;
            if s_xx.abs() > 1e-9 { Some(s_xy / s_xx) } else { Some(0.0) }
        }
        Feature::Intercept => {
            let mean_i = (n - 1.0) * 0.5;
            let s_xx = (n * (n * n - 1.0)) / 12.0;
            let s_xy = state.sum_ix - n * mean_i * mean;
            let slope = if s_xx.abs() > 1e-9 { s_xy / s_xx } else { 0.0 };
            Some(mean - slope * mean_i)
        }
        Feature::CountAboveMean => Some(count_a as f32),
        Feature::CountBelowMean => Some(count_b as f32),
        Feature::LongestStrikeAboveMean => Some(max_strike_a as f32),
        Feature::LongestStrikeBelowMean => Some(max_strike_b as f32),
        Feature::VariationCoefficient if mean.abs() > 1e-9 => Some(std_dev / mean),
        Feature::ZeroCrossingMean => Some(zc_mean),
        Feature::ZeroCrossingStd => Some(zc_std),
        Feature::AggLinearTrend(attr, chunk_len, func)
            if full_series.len() >= *chunk_len as usize =>
        {
            let cl = *chunk_len as usize;
            let mut agg_series = std::mem::take(&mut state.agg_linear_trend_buffer);
            agg_series.clear();
            for chunk in full_series.chunks_exact(cl) {
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
        Feature::Quantile(q_bits) => {
            let q = f32::from_bits(*q_bits);
            if running_sorted.is_empty() {
                Some(0.0)
            } else if running_sorted.len() == 1 {
                Some(running_sorted[0])
            } else {
                let n_len = running_sorted.len();
                let idx = q * (n_len as f32 - 1.0);
                let i = idx.floor() as usize;
                let f = idx - i as f32;
                if i >= n_len - 1 {
                    Some(running_sorted[n_len - 1])
                } else {
                    Some((1.0 - f) * running_sorted[i] + f * running_sorted[i + 1])
                }
            }
        }
        Feature::BenfordCorrelation => {
            let mut counts = [0.0; 9];
            for &v in full_series {
                let abs_v = v.abs();
                if abs_v > 0.0 {
                    let first_digit =
                        (abs_v / 10.0_f32.powf(abs_v.log10().floor())).floor() as usize;
                    if (1..=9).contains(&first_digit) {
                        counts[first_digit - 1] += 1.0;
                    }
                }
            }
            let total: f32 = counts.iter().sum();
            if total > 0.0 {
                let p: Vec<f32> = counts.iter().map(|&c| c / total).collect();
                let b: Vec<f32> = (1..10).map(|i| (1.0 + 1.0 / i as f32).log10()).collect();
                let mu_p = p.iter().sum::<f32>() / 9.0;
                let mu_b = b.iter().sum::<f32>() / 9.0;
                let mut num = 0.0;
                let mut den_p = 0.0;
                let mut den_b = 0.0;
                for i in 0..9 {
                    num += (p[i] - mu_p) * (b[i] - mu_b);
                    den_p += (p[i] - mu_p).powi(2);
                    den_b += (b[i] - mu_b).powi(2);
                }
                if den_p > 0.0 && den_b > 0.0 {
                    Some(num / (den_p * den_b).sqrt())
                } else {
                    Some(0.0)
                }
            } else {
                Some(0.0)
            }
        }
        Feature::VarianceLargerThanStandardDeviation => {
            if var > 1.0 {
                Some(1.0)
            } else {
                Some(0.0)
            }
        }
        Feature::MeanNAbsoluteMax(n_max) => {
            let mut abs_vals: Vec<f32> = full_series.iter().map(|v| v.abs()).collect();
            abs_vals.sort_unstable_by(|a, b| {
                b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
            });
            let count = (*n_max as usize).min(abs_vals.len());
            if count > 0 {
                Some(abs_vals.iter().take(count).sum::<f32>() / count as f32)
            } else {
                Some(0.0)
            }
        }
        _ => None,
    }
}
