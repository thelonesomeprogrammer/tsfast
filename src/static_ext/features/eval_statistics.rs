use crate::common::ColumnState;
use crate::types::Feature;
use rustc_hash::FxHashMap;

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
    values: &[f32],
    sorted_copy: &mut Option<Vec<f32>>,
    benford_corr: f32,
    first_max_idx: usize,
    last_max_idx: usize,
    first_min_idx: usize,
    last_min_idx: usize,
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
        Feature::Energy => Some(state.energy),
        Feature::Rms | Feature::RootMeanSquare => Some((state.energy / n).sqrt()),
        Feature::ZeroCrossingRate => Some(state.zcr_count as f32 / n),
        Feature::ZeroCrossingMean => Some(zc_mean),
        Feature::ZeroCrossingStd => Some(zc_std),
        Feature::CountAboveMean => Some(count_a as f32),
        Feature::CountBelowMean => Some(count_b as f32),
        Feature::LongestStrikeAboveMean => Some(max_strike_a as f32),
        Feature::LongestStrikeBelowMean => Some(max_strike_b as f32),
        Feature::VariationCoefficient if mean.abs() > 1e-9 => Some(std_dev / mean),
        Feature::Quantile(q_bits) => {
            let q = f32::from_bits(*q_bits);
            let mut copy = if let Some(c) = sorted_copy.take() {
                c
            } else {
                let mut c = std::mem::take(&mut state.sort_buffer);
                c.clear();
                c.extend_from_slice(values);
                c
            };
            let res = if copy.is_empty() {
                0.0
            } else if copy.len() == 1 {
                copy[0]
            } else {
                let n_len = copy.len();
                let idx = q * (n_len as f32 - 1.0);
                let i = idx.floor() as usize;
                let f = idx - i as f32;

                copy.sort_unstable_by(|a, b| {
                    a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                });

                if i >= n_len - 1 {
                    copy[n_len - 1]
                } else {
                    (1.0 - f) * copy[i] + f * copy[i + 1]
                }
            };
            state.sort_buffer = copy;
            Some(res)
        }
        Feature::IndexMassQuantile(q_bits) => {
            let q = f32::from_bits(*q_bits);
            let mut current_abs_sum = 0.0;
            let target = q * state.abs_sum;
            let mut res = 1.0;
            for (i, &v) in values.iter().enumerate() {
                current_abs_sum += v.abs();
                if current_abs_sum >= target {
                    res = (i + 1) as f32 / n;
                    break;
                }
            }
            Some(res)
        }
        Feature::BenfordCorrelation => Some(benford_corr),
        Feature::SumOfReoccurringValues => {
            let mut counts = FxHashMap::default();
            for &v in values {
                let bits = v.to_bits();
                *counts.entry(bits).or_insert(0) += 1;
            }
            Some(counts
                .iter()
                .filter(|&(_, &count)| count > 1)
                .map(|(&bits, _)| f32::from_bits(bits))
                .sum())
        }
        Feature::SumOfReoccurringDataPoints => {
            let mut counts = FxHashMap::default();
            for &v in values {
                let bits = v.to_bits();
                *counts.entry(bits).or_insert(0) += 1;
            }
            Some(counts
                .iter()
                .filter(|&(_, &count)| count > 1)
                .map(|(&bits, &count)| f32::from_bits(bits) * count as f32)
                .sum())
        }
        Feature::VarianceLargerThanStandardDeviation => {
            Some(if var > 1.0 { 1.0 } else { 0.0 })
        }
        Feature::MeanNAbsoluteMax(n_max) => {
            let mut abs_vals = std::mem::take(&mut state.sort_buffer);
            abs_vals.clear();
            abs_vals.extend(values.iter().map(|v| v.abs()));
            abs_vals.sort_unstable_by(|a, b| {
                b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
            });
            let count = (*n_max as usize).min(abs_vals.len());
            let res = if count > 0 {
                abs_vals.iter().take(count).sum::<f32>() / count as f32
            } else {
                0.0
            };
            state.sort_buffer = abs_vals;
            Some(res)
        }
        Feature::HasDuplicateMax => {
            let mut count = 0;
            for &v in values {
                if v == state.max_value {
                    count += 1;
                    if count > 1 {
                        break;
                    }
                }
            }
            Some(if count > 1 { 1.0 } else { 0.0 })
        }
        Feature::HasDuplicateMin => {
            let mut count = 0;
            for &v in values {
                if v == state.min_value {
                    count += 1;
                    if count > 1 {
                        break;
                    }
                }
            }
            Some(if count > 1 { 1.0 } else { 0.0 })
        }
        Feature::HasDuplicate => {
            let mut unique = rustc_hash::FxHashSet::default();
            let mut has_dup = false;
            for &v in values {
                if !unique.insert(v.to_bits()) {
                    has_dup = true;
                    break;
                }
            }
            Some(if has_dup { 1.0 } else { 0.0 })
        }
        Feature::FirstLocMax => Some(first_max_idx as f32 / n),
        Feature::LastLocMax => Some((last_max_idx + 1) as f32 / n),
        Feature::FirstLocMin => Some(first_min_idx as f32 / n),
        Feature::LastLocMin => Some((last_min_idx + 1) as f32 / n),
        _ => None,
    }
}
