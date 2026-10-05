use crate::common::ColumnState;
use crate::types::Compute;
use std::simd::cmp::SimdPartialOrd;
use std::simd::f32x4;
use std::simd::num::SimdFloat;

pub struct DiffProcessor;

impl DiffProcessor {
    #[inline(always)]
    pub fn process_simd(
        compute: Compute,
        chunk: f32x4,
        shifted: f32x4,
        offset: f32,
        state: &mut ColumnState,
    ) {
        if compute.intersects(Compute::ZERO_CROSS | Compute::MAC | Compute::MC | Compute::CID_CE) {
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
            if compute.contains(Compute::ZERO_CROSS) {
                let signs = chunk.simd_lt(f32x4::splat(0.0));
                let prev_signs = shifted.simd_lt(f32x4::splat(0.0));
                let mask = (signs ^ prev_signs).to_bitmask();
                state.zcr_count += mask.count_ones();
                if compute.contains(Compute::ZC_INDICES) {
                    for bit in 0..4 {
                        if (mask >> bit) & 1 == 1 {
                            state.zc_indices.push(offset + bit as f32);
                        }
                    }
                }
            }
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
    }

    #[inline(always)]
    pub fn process_remainder(
        compute: Compute,
        val: f32,
        prev: f32,
        global_idx: usize,
        state: &mut ColumnState,
    ) {
        let diff = val - prev;
        if compute.contains(Compute::MAC) {
            state.mac_sum_vec += f32x4::from_array([diff.abs(), 0.0, 0.0, 0.0]);
        }
        if compute.contains(Compute::MC) {
            state.mc_sum_vec += f32x4::from_array([diff, 0.0, 0.0, 0.0]);
        }
        if compute.contains(Compute::CID_CE) {
            state.sum_sq_diff += diff * diff;
        }
        if compute.contains(Compute::ZERO_CROSS) && (val < 0.0) != (prev < 0.0) {
            state.zcr_count += 1;
            if compute.contains(Compute::ZC_INDICES) {
                state.zc_indices.push(global_idx as f32);
            }
        }
    }

    pub fn finalize(
        compute: Compute,
        state: &ColumnState,
    ) -> crate::metrics::ZcMetrics {
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

        crate::metrics::ZcMetrics {
            zc_mean,
            zc_std,
        }
    }
}
