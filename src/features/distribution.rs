use crate::types::Feature;

#[inline(always)]
pub fn eval_distribution(
    feat: &Feature,
    context: &mut crate::context::FeatureContext,
) -> Option<f32> {
    let values = context.values;
    let state = &mut *context.state;
    let median = context.median;
    let entropy = context.entropy;

    let res = match feat {
        Feature::Median => median,
        Feature::Entropy => entropy,
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
