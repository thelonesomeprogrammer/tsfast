use crate::common::ColumnState;
use crate::types::Compute;

use crate::metrics::ZcMetrics;
use std::simd::num::SimdFloat;
use std::simd::f32x4;

pub struct DiffProcessor;

impl DiffProcessor {
    #[inline(always)]
    pub fn reset_state(state: &mut ColumnState, values: &[f32], is_incremental: bool) {
        if !is_incremental {
            state.mac_sum = 0.0;
            state.mc_sum = 0.0;
            state.sum_sq_diff = 0.0;
            state.sum_prod = 0.0;
            state.sum_ix = 0.0;
            state.auc_sum = 0.0;
            state.zcr_count = 0;
        }

        state.mac_sum_vec = std::simd::f32x4::splat(0.0);
        state.mc_sum_vec = std::simd::f32x4::splat(0.0);

        state.peaks = 0;
        // Recount peaks for the whole window
        for i in 1..values.len().saturating_sub(1) {
            if values[i] > values[i - 1] && values[i] > values[i + 1] {
                state.peaks += 1;
            }
        }

        state.zc_indices.clear();
        if !values.is_empty() {
            state.prev_last = values[0];
        }
    }

    #[inline(always)]
    pub fn update_batch(
        compute: Compute,
        old_slice: &[f32],
        new_slice: &[f32],
        value_before_old: Option<f32>,
        value_after_old: f32,
        value_before_new: f32,
        state: &mut ColumnState,
    ) {
        let y = old_slice.len();
        if y == 0 {
            return;
        }

        if compute.intersects(
            Compute::MAC
                | Compute::MC
                | Compute::CID_CE
                | Compute::AUTOCORR_LAG1
                | Compute::AUC
                | Compute::ZERO_CROSS,
        ) {
            // Remove effect of diffs starting within or at boundary of old_slice
            // Boundary diff: (old_slice[0], value_before_old)
            if let Some(prev) = value_before_old {
                let diff = old_slice[0] - prev;
                if compute.contains(Compute::MAC) {
                    state.mac_sum -= diff.abs();
                }
                if compute.contains(Compute::MC) {
                    state.mc_sum -= diff;
                }
                if compute.contains(Compute::CID_CE) {
                    state.sum_sq_diff -= diff * diff;
                }
                if compute.contains(Compute::AUTOCORR_LAG1) {
                    state.sum_prod -= old_slice[0] * prev;
                }
                if compute.contains(Compute::AUC) {
                    state.auc_sum -= (old_slice[0] + prev) * 0.5;
                }
                if compute.contains(Compute::ZERO_CROSS) && (old_slice[0] < 0.0) != (prev < 0.0) {
                    state.zcr_count -= 1;
                }
            }

            // Internal diffs in old_slice
            for i in 1..y {
                let diff = old_slice[i] - old_slice[i - 1];
                if compute.contains(Compute::MAC) {
                    state.mac_sum -= diff.abs();
                }
                if compute.contains(Compute::MC) {
                    state.mc_sum -= diff;
                }
                if compute.contains(Compute::CID_CE) {
                    state.sum_sq_diff -= diff * diff;
                }
                if compute.contains(Compute::AUTOCORR_LAG1) {
                    state.sum_prod -= old_slice[i] * old_slice[i - 1];
                }
                if compute.contains(Compute::AUC) {
                    state.auc_sum -= (old_slice[i] + old_slice[i - 1]) * 0.5;
                }
                if compute.contains(Compute::ZERO_CROSS) && (old_slice[i] < 0.0) != (old_slice[i - 1] < 0.0) {
                    state.zcr_count -= 1;
                }
            }

            // Boundary diff between old_slice and remaining window: (value_after_old, old_slice[y-1])
            let diff_after = value_after_old - old_slice[y - 1];
            if compute.contains(Compute::MAC) {
                state.mac_sum -= diff_after.abs();
            }
            if compute.contains(Compute::MC) {
                state.mc_sum -= diff_after;
            }
            if compute.contains(Compute::CID_CE) {
                state.sum_sq_diff -= diff_after * diff_after;
            }
            if compute.contains(Compute::AUTOCORR_LAG1) {
                state.sum_prod -= value_after_old * old_slice[y - 1];
            }
            if compute.contains(Compute::AUC) {
                state.auc_sum -= (value_after_old + old_slice[y - 1]) * 0.5;
            }
            if compute.contains(Compute::ZERO_CROSS) && (value_after_old < 0.0) != (old_slice[y - 1] < 0.0) {
                state.zcr_count -= 1;
            }

            // Add effect of diffs in new_slice
            // Boundary diff: (new_slice[0], value_before_new)
            let diff_new_start = new_slice[0] - value_before_new;
            if compute.contains(Compute::MAC) {
                state.mac_sum += diff_new_start.abs();
            }
            if compute.contains(Compute::MC) {
                state.mc_sum += diff_new_start;
            }
            if compute.contains(Compute::CID_CE) {
                state.sum_sq_diff += diff_new_start * diff_new_start;
            }
            if compute.contains(Compute::AUTOCORR_LAG1) {
                state.sum_prod += new_slice[0] * value_before_new;
            }
            if compute.contains(Compute::AUC) {
                state.auc_sum += (new_slice[0] + value_before_new) * 0.5;
            }
            if compute.contains(Compute::ZERO_CROSS) && (new_slice[0] < 0.0) != (value_before_new < 0.0) {
                state.zcr_count += 1;
            }

            // Internal diffs in new_slice
            for i in 1..y {
                let diff = new_slice[i] - new_slice[i - 1];
                if compute.contains(Compute::MAC) {
                    state.mac_sum += diff.abs();
                }
                if compute.contains(Compute::MC) {
                    state.mc_sum += diff;
                }
                if compute.contains(Compute::CID_CE) {
                    state.sum_sq_diff += diff * diff;
                }
                if compute.contains(Compute::AUTOCORR_LAG1) {
                    state.sum_prod += new_slice[i] * new_slice[i - 1];
                }
                if compute.contains(Compute::AUC) {
                    state.auc_sum += (new_slice[i] + new_slice[i - 1]) * 0.5;
                }
                if compute.contains(Compute::ZERO_CROSS) && (new_slice[i] < 0.0) != (new_slice[i - 1] < 0.0) {
                    state.zcr_count += 1;
                }
            }
        }
    }

    #[inline(always)]
    pub fn update_incremental(
        compute: Compute,
        old_val: f32,
        new_val: f32,
        old_val_next: f32,
        old_last: f32,
        state: &mut ColumnState,
    ) {
        if compute.intersects(
            Compute::MAC
                | Compute::MC
                | Compute::CID_CE
                | Compute::AUTOCORR_LAG1
                | Compute::AUC
        ) {
            // Remove effect of (old_val, old_val_next)
            let old_diff = old_val_next - old_val;
            if compute.contains(Compute::MAC) {
                state.mac_sum -= old_diff.abs();
            }
            if compute.contains(Compute::MC) {
                state.mc_sum -= old_diff;
            }
            if compute.contains(Compute::CID_CE) {
                state.sum_sq_diff -= old_diff * old_diff;
            }
            if compute.contains(Compute::AUTOCORR_LAG1) {
                state.sum_prod -= old_val_next * old_val;
            }
            if compute.contains(Compute::AUC) {
                state.auc_sum -= (old_val + old_val_next) * 0.5;
            }

            // Add effect of (old_last, new_val)
            let new_diff = new_val - old_last;
            if compute.contains(Compute::MAC) {
                state.mac_sum += new_diff.abs();
            }
            if compute.contains(Compute::MC) {
                state.mc_sum += new_diff;
            }
            if compute.contains(Compute::CID_CE) {
                state.sum_sq_diff += new_diff * new_diff;
            }
            if compute.contains(Compute::AUTOCORR_LAG1) {
                state.sum_prod += new_val * old_last;
            }
            if compute.contains(Compute::AUC) {
                state.auc_sum += (new_val + old_last) * 0.5;
            }
        }

        // Zero Crossing Rate
        if compute.contains(Compute::ZERO_CROSS) {
            if (old_val < 0.0) != (old_val_next < 0.0) {
                state.zcr_count -= 1;
            }
            if (old_last < 0.0) != (new_val < 0.0) {
                state.zcr_count += 1;
            }
        }
    }

    #[inline(always)]
    pub fn process_simd(
        compute: Compute,
        chunk: std::simd::f32x4,
        values: &[f32],
        global_idx: usize,
        is_incremental: bool,
        state: &mut ColumnState,
    ) {
        use std::simd::num::SimdFloat;
        use std::simd::cmp::SimdPartialOrd;

        if compute.intersects(Compute::ZERO_CROSS | Compute::AUTOCORR_LAG1 | Compute::MAC | Compute::MC | Compute::CID_CE | Compute::AUC) {
            let shifted = if global_idx == 0 {
                std::simd::f32x4::from_array([
                    state.prev_last,
                    chunk[0],
                    chunk[1],
                    chunk[2]
                ])
            } else {
                std::simd::f32x4::from_array([
                    values[global_idx - 1],
                    chunk[0],
                    chunk[1],
                    chunk[2]
                ])
            };

            state.prev_last = chunk[3];

            if !is_incremental {
                let diff = chunk - shifted;

                if compute.contains(Compute::MAC) {
                    state.mac_sum_vec += diff.abs();
                }
                if compute.contains(Compute::MC) {
                    state.mc_sum_vec += diff;
                }
                if compute.contains(Compute::CID_CE) {
                    state.sum_sq_diff_vec += diff * diff;
                }
                if compute.contains(Compute::AUTOCORR_LAG1) {
                    state.sum_prod_vec += chunk * shifted;
                }
                if compute.contains(Compute::AUC) {
                    state.auc_sum_vec += (chunk + shifted) * f32x4::splat(0.5);
                }
                if compute.contains(Compute::ZERO_CROSS) {
                    let signs = chunk.simd_lt(std::simd::f32x4::splat(0.0));
                    let prev_signs = shifted.simd_lt(std::simd::f32x4::splat(0.0));
                    let mask = (signs ^ prev_signs).to_bitmask();
                    state.zcr_count += mask.count_ones();
                }
            }

            if compute.contains(Compute::ZC_INDICES) && compute.contains(Compute::ZERO_CROSS) {
                let signs = chunk.simd_lt(std::simd::f32x4::splat(0.0));
                let prev_signs = shifted.simd_lt(std::simd::f32x4::splat(0.0));
                let mask = (signs ^ prev_signs).to_bitmask();
                for bit in 0..4 {
                    if (mask >> bit) & 1 == 1 {
                        state.zc_indices.push(global_idx as f32 + bit as f32);
                    }
                }
            }
        }
    }

    #[inline(always)]
    pub fn process_remainder(
        compute: Compute,
        val: f32,
        prev: f32,
        offset: f32,
        is_incremental: bool,
        state: &mut ColumnState,
    ) {
        if !is_incremental {
            let diff = val - prev;
            if compute.contains(Compute::MAC) {
                state.mac_sum += diff.abs();
            }
            if compute.contains(Compute::MC) {
                state.mc_sum += diff;
            }
            if compute.contains(Compute::CID_CE) {
                state.sum_sq_diff += diff * diff;
            }
            if compute.contains(Compute::AUTOCORR_LAG1) {
                state.sum_prod += val * prev;
            }
            if compute.contains(Compute::AUC) {
                state.auc_sum += (val + prev) * 0.5;
            }
            if compute.contains(Compute::ZERO_CROSS) && (val < 0.0) != (prev < 0.0) {
                state.zcr_count += 1;
            }
        }

        if compute.contains(Compute::ZC_INDICES) && compute.contains(Compute::ZERO_CROSS) && (val < 0.0) != (prev < 0.0) {
            state.zc_indices.push(offset);
        }
    }

    #[inline(always)]
    pub fn finalize_simd(compute: Compute, state: &mut ColumnState) {
        if compute.contains(Compute::MAC) {
            state.mac_sum += state.mac_sum_vec.reduce_sum();
            state.mac_sum_vec = f32x4::splat(0.0);
        }
        if compute.contains(Compute::MC) {
            state.mc_sum += state.mc_sum_vec.reduce_sum();
            state.mc_sum_vec = f32x4::splat(0.0);
        }
        if compute.contains(Compute::CID_CE) {
            state.sum_sq_diff += state.sum_sq_diff_vec.reduce_sum();
            state.sum_sq_diff_vec = f32x4::splat(0.0);
        }
        if compute.contains(Compute::AUTOCORR_LAG1) {
            state.sum_prod += state.sum_prod_vec.reduce_sum();
            state.sum_prod_vec = f32x4::splat(0.0);
        }
        if compute.contains(Compute::AUC) {
            state.auc_sum += state.auc_sum_vec.reduce_sum();
            state.auc_sum_vec = f32x4::splat(0.0);
        }
    }

    pub fn finalize(
        compute: Compute,
        state: &ColumnState,
    ) -> ZcMetrics {
        let mut zc_mean = 0.0;
        let mut zc_std = 0.0;

        if compute.contains(Compute::ZC_STATS) && !state.zc_indices.is_empty() {
            zc_mean = state.zc_indices.iter().sum::<f32>() / state.zc_indices.len() as f32;
            if compute.contains(Compute::ZC_STD) {
                let zc_m2 = state
                    .zc_indices
                    .iter()
                    .map(|&idx| (idx - zc_mean).powi(2))
                    .sum::<f32>();
                zc_std = (zc_m2 / state.zc_indices.len() as f32).sqrt();
            }
        }

        ZcMetrics {
            zc_mean,
            zc_std,
        }
    }
}
