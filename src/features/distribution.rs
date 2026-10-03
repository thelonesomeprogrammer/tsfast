use crate::types::Feature;

#[inline(always)]
pub fn eval_distribution(
    feat: &Feature,
    context: &mut crate::context::FeatureContext,
) -> Option<f32> {
    let values = context.values;
    let state = &mut *context.state;
    let n = context.n;
    let _mean = context.mean;
    let _m2 = context.m2;
    let _m3 = context.m3;
    let _m4 = context.m4;
    let _var = context.var;
    let std_dev = context.std_dev;
    let _mac_sum = context.mac_sum;
    let _mc_sum = context.mc_sum;
    let _first_max_idx = context.first_max_idx;
    let _last_max_idx = context.last_max_idx;
    let _first_min_idx = context.first_min_idx;
    let _last_min_idx = context.last_min_idx;
    let median = context.median;
    let _iqr = context.iqr;
    let entropy = context.entropy;
    let min_val = state.min_value;
    let max_val = state.max_value;
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
    let _fft_autocorr = context.fft_autocorr;
    let _fft_complex = context.fft_complex;
    let _spectrum = context.spectrum;
    let _unique_c3_lags = context.unique_c3_lags;
    let _unique_paa_totals = context.unique_paa_totals;
    let _paa_boundaries = context.paa_boundaries;

    let res = match feat {
        Feature::Median => median,
        Feature::Entropy => entropy,
        Feature::BinnedEntropy(max_bins) => {
            let max_bins = *max_bins as usize;
            if max_bins == 0 || n == 0.0 {
                return Some(f32::NAN);
            }
            if min_val == max_val {
                return Some(0.0);
            }

            let mut hist = std::mem::take(&mut state.binned_entropy_buffer);
            hist.clear();
            hist.resize(max_bins, 0.0);

            let bin_width = (max_val - min_val) / (max_bins as f32);

            for &val in values {
                let mut bin = ((val - min_val) / bin_width).floor() as usize;
                if bin >= max_bins {
                    bin = max_bins - 1;
                }
                hist[bin] += 1.0;
            }

            let mut entropy = 0.0_f32;
            for &count in &hist {
                if count > 0.0 {
                    let p: f32 = count / n;
                    entropy -= p * p.ln();
                }
            }
            state.binned_entropy_buffer = hist;
            entropy
        }
        Feature::RatioBeyondRSigma(r_bits) => {
            let r = f32::from_bits(*r_bits);
            let boundary = r * std_dev;
            let mean = context.mean;
            let mut count = 0;
            // We can use vectorized logic to speed up processing
            use std::simd::{cmp::SimdPartialOrd, f32x4, num::SimdFloat};

            let chunks = values.chunks_exact(4);
            let rem = chunks.remainder();
            let boundary_simd = f32x4::splat(boundary);
            let mean_simd = f32x4::splat(mean);

            for chunk in chunks {
                let v = f32x4::from_slice(chunk);
                let diff = (v - mean_simd).abs();
                let mask = diff.simd_gt(boundary_simd);
                count += mask.to_bitmask().count_ones();
            }

            for &v in rem {
                if (v - mean).abs() > boundary {
                    count += 1;
                }
            }

            if values.is_empty() {
                0.0
            } else {
                count as f32 / values.len() as f32
            }
        }
        Feature::IndexMassQuantile(q_bits) => {
            let q = f32::from_bits(*q_bits) as f64;
            let target_mass = q * state.abs_sum as f64;

            if target_mass <= 0.0 || values.is_empty() {
                0.0
            } else {
                // Check if we can resume (only if the array hasn't shrunk, e.g. expanding)
                if state.mass_pointer < values.len() && state.mass_cum_sum < target_mass {
                    let mut cum_sum = state.mass_cum_sum;
                    let mut pointer = state.mass_pointer;
                    while pointer < values.len() {
                        cum_sum += values[pointer].abs() as f64;
                        if cum_sum >= target_mass {
                            state.mass_cum_sum = cum_sum;
                            state.mass_pointer = pointer;
                            break;
                        }
                        pointer += 1;
                    }
                    if pointer >= values.len() {
                        pointer = values.len() - 1;
                        state.mass_cum_sum = cum_sum;
                        state.mass_pointer = pointer;
                    }
                    (state.mass_pointer + 1) as f32 / values.len() as f32
                } else {
                    // Sliding window or mass center shifted left
                    let mut cum_sum = 0.0;
                    let mut pointer = 0;
                    while pointer < values.len() {
                        cum_sum += values[pointer].abs() as f64;
                        if cum_sum >= target_mass {
                            break;
                        }
                        pointer += 1;
                    }
                    if pointer >= values.len() {
                        pointer = values.len() - 1;
                    }
                    state.mass_cum_sum = cum_sum;
                    state.mass_pointer = pointer;
                    (pointer + 1) as f32 / values.len() as f32
                }
            }
        }
        Feature::Quantile(q_bits) => {
            let q = f32::from_bits(*q_bits);

            if let Some(sorted) = context.running_sorted {
                let res = if sorted.is_empty() {
                    0.0
                } else if sorted.len() == 1 {
                    sorted[0]
                } else {
                    let n_len = sorted.len();
                    let idx = q * (n_len as f32 - 1.0);
                    let i = idx.floor() as usize;
                    let f = idx - i as f32;
                    if i >= n_len - 1 {
                        sorted[n_len - 1]
                    } else {
                        (1.0 - f) * sorted[i] + f * sorted[i + 1]
                    }
                };
                res
            } else {
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

                    let (val_i, val_i_plus_1) = if i >= n_len - 1 {
                        let (_, &mut val, _) = copy.select_nth_unstable_by(n_len - 1, |a, b| {
                            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                        });
                        (val, val)
                    } else {
                        let (_, &mut val1, _) = copy.select_nth_unstable_by(i + 1, |a, b| {
                            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                        });
                        let val0 = *copy[..=i]
                            .iter()
                            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                            .unwrap();
                        (val0, val1)
                    };

                    if i >= n_len - 1 {
                        val_i
                    } else {
                        (1.0 - f) * val_i + f * val_i_plus_1
                    }
                };
                state.sort_buffer = copy;
                res
            }
        }
        Feature::BenfordCorrelation => {
            let mut counts = [0.0; 9];
            for &v in values {
                let mut abs_v = v.abs();
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
                    num / (den_p * den_b).sqrt()
                } else {
                    0.0
                }
            } else {
                0.0
            }
        }
        Feature::SumOfReoccurringValues => {
            let mut counts = rustc_hash::FxHashMap::default();
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
            let mut counts = rustc_hash::FxHashMap::default();
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
