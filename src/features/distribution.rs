use crate::types::Feature;

#[inline(always)]

fn compute_quantile(sorted: &[f32], q: f32) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let n_len = sorted.len();
    let idx = q * (n_len as f32 - 1.0);
    let i = idx.floor() as usize;
    let f = idx - i as f32;
    if i >= n_len - 1 {
        sorted[n_len - 1]
    } else {
        (1.0 - f) * sorted[i] + f * sorted[i + 1]
    }
}

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
        // tsfresh: |mean - median| < r * (max - min)
        Feature::SymmetryLooking(r_bits) => {
            let r = f32::from_bits(*r_bits);
            ((context.mean - median).abs() < r * (max_val - min_val)) as u8 as f32
        }
        Feature::MedianAbsDeviation => context.median_abs_dev,
        // TSFEL: entropy of the distinct-value distribution, normalised by log2(n).
        Feature::Entropy => {
            let mut sorted = std::mem::take(&mut state.sort_buffer);
            sorted.clear();
            sorted.extend_from_slice(values);
            sorted.sort_unstable_by(f32::total_cmp);
            let n_f = sorted.len() as f64;
            let mut h = 0.0f64;
            for run in sorted.chunk_by(|a, b| a == b) {
                let p = run.len() as f64 / n_f;
                h -= p * p.log2();
            }
            state.sort_buffer = sorted;
            if values.len() <= 2 { 0.0 } else { (h / n_f.log2()) as f32 }
        }
        Feature::Ecdf(d) => {
            let d_idx = *d as f32;
            if d_idx >= n { 1.0 } else { d_idx / n }
        }

        Feature::EcdfPercentile(p_bits) => {
            let p = f32::from_bits(*p_bits);
            let k = (p * n).floor() as usize;
            if min_val == max_val && !values.is_empty() {
                return Some(values[0]);
            }
            if k == 0 || values.is_empty() {
                return Some(f32::NAN);
            }
            if let Some(sorted) = context.running_sorted.filter(|s| !s.is_empty()) {
                sorted[k - 1]
            } else {
                let mut copy: Vec<f32> = std::mem::take(&mut state.sort_buffer);
                copy.clear();
                copy.extend_from_slice(values);
                let (_, val, _) = copy.select_nth_unstable_by(k - 1, f32::total_cmp);
                let res = *val;
                state.sort_buffer = copy;
                res
            }
        }
        Feature::EcdfPercentileCount(p_bits) => {
            let p = f32::from_bits(*p_bits);
            let k = (p * n).floor() as usize;
            if min_val == max_val && !values.is_empty() {
                return Some(n); // If constant, the count is the total length
            }
            if k == 0 || values.is_empty() {
                return Some(f32::NAN);
            }
            let v = if let Some(sorted) = context.running_sorted.filter(|s| !s.is_empty()) {
                sorted[k - 1]
            } else {
                let mut copy: Vec<f32> = std::mem::take(&mut state.sort_buffer);
                copy.clear();
                copy.extend_from_slice(values);
                let (_, val, _) = copy.select_nth_unstable_by(k - 1, f32::total_cmp);
                let res = *val;
                state.sort_buffer = copy;
                res
            };

            use std::simd::{cmp::SimdPartialOrd, f32x4};
            let mut count = 0;
            let chunks = values.chunks_exact(4);
            let rem = chunks.remainder();
            let v_simd = f32x4::splat(v);

            for chunk in chunks {
                let chunk_simd = f32x4::from_slice(chunk);
                let mask = chunk_simd.simd_le(v_simd);
                count += mask.to_bitmask().count_ones();
            }

            for &val in rem {
                if val <= v {
                    count += 1;
                }
            }
            count as f32
        }
        Feature::EcdfSlope(p_init_bits, p_end_bits) => {
            if values.is_empty() {
                return Some(f32::NAN);
            }
            if min_val == max_val {
                return Some(f32::INFINITY);
            }
            let p_init = f32::from_bits(*p_init_bits);
            let p_end = f32::from_bits(*p_end_bits);
            let k_init = (p_init * n).floor() as usize;
            let k_end = (p_end * n).floor() as usize;
            if k_init == 0 || k_end == 0 {
                return Some(f32::NAN);
            }

            let (x_init, x_end) = if let Some(sorted) = context.running_sorted.filter(|s| !s.is_empty()) {
                (sorted[k_init - 1], sorted[k_end - 1])
            } else {
                let mut copy: Vec<f32> = std::mem::take(&mut state.sort_buffer);
                copy.clear();
                copy.extend_from_slice(values);

                let (_, val_end, _) = copy.select_nth_unstable_by(k_end - 1, f32::total_cmp);
                let x_e = *val_end;
                let (_, val_init, _) = copy[0..k_end].select_nth_unstable_by(k_init - 1, f32::total_cmp);
                let x_i = *val_init;

                state.sort_buffer = copy;
                (x_i, x_e)
            };

            (p_end - p_init) / (x_end - x_init)
        }
        Feature::ValueCount(val_bits) => {
            let val = f32::from_bits(*val_bits);
            let count = if val.is_nan() {
                values.iter().filter(|&&v| v.is_nan()).count()
            } else {
                values.iter().filter(|&&v| v == val).count()
            };
            return Some(count as f32);
        }
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
            // tsfresh: first index where the cumulative |x| mass reaches q.
            let q = f32::from_bits(*q_bits) as f64;
            let total: f64 = values.iter().map(|v| v.abs() as f64).sum();
            let target_mass = q * total;
            if target_mass <= 0.0 || values.is_empty() {
                0.0
            } else {
                let mut cum_sum = 0.0;
                let idx = values
                    .iter()
                    .position(|v| {
                        cum_sum += v.abs() as f64;
                        cum_sum >= target_mass
                    })
                    .unwrap_or(values.len() - 1);
                (idx + 1) as f32 / values.len() as f32
            }
        }

        Feature::ChangeQuantiles(ql_bits, qh_bits, isabs, f_agg) => {
            let ql = f32::from_bits(*ql_bits);
            let qh = f32::from_bits(*qh_bits);

            if ql >= qh || values.len() < 2 {
                return Some(0.0);
            }

            let (ql_val, qh_val) = if let Some(sorted) = context.running_sorted.filter(|s| !s.is_empty()) {
                (compute_quantile(sorted, ql), compute_quantile(sorted, qh))
            } else {
                let mut copy: Vec<f32> = std::mem::take(&mut state.sort_buffer);
                copy.clear();
                copy.extend_from_slice(values);
                copy.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let vals = (compute_quantile(&copy, ql), compute_quantile(&copy, qh));
                state.sort_buffer = copy;
                vals
            };

            let mut diffs = Vec::with_capacity(values.len());
            let mut prev_in = false;
            let mut prev_val = 0.0;

            for (i, &v) in values.iter().enumerate() {
                let in_corridor = v >= ql_val && v <= qh_val;

                if in_corridor {
                    if prev_in {
                        let diff = v - prev_val;
                        diffs.push(if *isabs { diff.abs() } else { diff });
                    }
                }
                prev_in = in_corridor;
                prev_val = v;
            }

            if diffs.is_empty() {
                return Some(0.0);
            }

            let res = match f_agg {
                crate::types::AggFunc::Mean => diffs.iter().sum::<f32>() / diffs.len() as f32,
                crate::types::AggFunc::Var => {
                    let n = diffs.len() as f32;
                    if n < 2.0 {
                        0.0
                    } else {
                        let mean = diffs.iter().sum::<f32>() / n;
                        let var: f32 = diffs.iter().map(|&x| (x - mean).powi(2)).sum();
                        var / n
                    }
                }
                crate::types::AggFunc::Max => {
                    diffs.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
                }
                crate::types::AggFunc::Min => diffs.iter().cloned().fold(f32::INFINITY, f32::min),
            };
            res
        }

        Feature::Quantile(q_bits) => {
            let q = f32::from_bits(*q_bits);

            if let Some(sorted) = context.running_sorted.filter(|s| !s.is_empty()) {
                compute_quantile(sorted, q)
            } else {
                let mut copy: Vec<f32> = std::mem::take(&mut state.sort_buffer);
                copy.clear();
                copy.extend_from_slice(values);
                copy.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let res = compute_quantile(&copy, q);
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
