use crate::types::{Feature, AggAttr, AggFunc, FftAttr};
use crate::common::ColumnState;

#[inline(always)]
pub fn eval_distribution(
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
                Feature::Median => median,
                Feature::Entropy => entropy,
                Feature::Quantile(q_bits) => {
                    let q = f32::from_bits(*q_bits);
                    // ⚡ Bolt Optimization: Reuse sort_buffer to prevent inner loop memory allocations
                    let mut copy: Vec<f32> = std::mem::take(&mut state.sort_buffer);
                    copy.clear();
                    copy.extend_from_slice(values);
                    let res = if copy.is_empty() {
                        0.0
                    } else if copy.len() == 1 {
                        copy[0]
                    } else {
                        let n_len = copy.len();
                        let idx = q * (n_len as f32 - 1.0);
                        let i = idx.floor() as usize;
                        let f = idx - i as f32;
                        copy.sort_unstable_by(|a: &f32, b: &f32| {
                            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                        });
                        if i >= n_len - 1 {
                            copy[n_len - 1]
                        } else {
                            (1.0 - f) * copy[i] + f * copy[i + 1]
                        }
                    };
                    state.sort_buffer = copy;
                    res
                }
                Feature::BenfordCorrelation => {
                    let mut counts = [0.0; 9];
                    for &v in values {
                        let mut abs_v = v.abs();
                        if abs_v > 0.0 {
                            while abs_v < 1.0 {
                                abs_v *= 10.0;
                            }
                            while abs_v >= 10.0 {
                                abs_v /= 10.0;
                            }
                            let first_digit = abs_v.floor() as usize;
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
                            num / (den_p * den_b).sqrt()
                        } else {
                            0.0
                        }
                    } else {
                        0.0
                    }
                }
                Feature::SumOfReoccurringValues => {
                    let mut counts = std::collections::HashMap::new();
                    for &v in values {
                        let bits = v.to_bits();
                        *counts.entry(bits).or_insert(0) += 1;
                    }
                    counts
                        .iter()
                        .filter(|&(_, &count)| count > 1)
                        .map(|(&bits, _)| f32::from_bits(bits))
                        .sum()
                }
                Feature::SumOfReoccurringDataPoints => {
                    let mut counts = std::collections::HashMap::new();
                    for &v in values {
                        let bits = v.to_bits();
                        *counts.entry(bits).or_insert(0) += 1;
                    }
                    counts
                        .iter()
                        .filter(|&(_, &count)| count > 1)
                        .map(|(&bits, &count)| f32::from_bits(bits) * count as f32)
                        .sum()
                }
        _ => return None,
    };
    Some(res)
}
