use crate::types::{Feature, AggAttr, AggFunc, FftAttr};
use crate::common::ColumnState;

#[inline(always)]
pub fn eval_min_max(
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
                Feature::Min => state.min_value,
                Feature::Max => state.max_value,
                Feature::AbsMax => state.abs_max,
                Feature::FirstLocMax => first_max_idx as f32 / n,
                Feature::LastLocMax => (last_max_idx + 1) as f32 / n,
                Feature::FirstLocMin => first_min_idx as f32 / n,
                Feature::LastLocMin => (last_min_idx + 1) as f32 / n,
                Feature::MeanNAbsoluteMax(n_max) => {
                    let mut abs_vals: Vec<f32> = values.iter().map(|v| v.abs()).collect();
                    abs_vals.sort_unstable_by(|a, b| {
                        b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
                    });
                    let count = (*n_max as usize).min(abs_vals.len());
                    if count > 0 {
                        abs_vals.iter().take(count).sum::<f32>() / count as f32
                    } else {
                        0.0
                    }
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
                    if count > 1 { 1.0 } else { 0.0 }
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
                    if count > 1 { 1.0 } else { 0.0 }
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
                    if has_dup { 1.0 } else { 0.0 }
                }
        _ => return None,
    };
    Some(res)
}
