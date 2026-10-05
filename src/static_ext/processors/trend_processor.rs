use crate::common::{ColumnState, LANES};
use crate::types::Compute;
use std::simd::cmp::SimdPartialOrd;
use std::simd::f32x4;
use std::simd::num::SimdFloat;

pub struct TrendProcessor;

impl TrendProcessor {
    #[inline(always)]
    pub fn process_simd(
        compute: Compute,
        chunk: f32x4,
        i: &[f32],
        offset: f32,
        global_idx: usize,
        values: &[f32],
        state: &mut ColumnState,
        unique_paa_totals: &[u16],
        paa_boundaries: &[Vec<usize>],
        unique_c3_lags: &[u16],
    ) {
        if compute.intersects(Compute::PEAKS | Compute::TROUGHS) {
            let left = f32x4::from_array([
                if global_idx > 0 {
                    values[global_idx - 1]
                } else {
                    values[0]
                },
                values[global_idx],
                values[global_idx + 1],
                values[global_idx + 2],
            ]);
            let right = f32x4::from_array([
                values[global_idx + 1],
                values[global_idx + 2],
                values[global_idx + 3],
                if global_idx + 4 < values.len() {
                    values[global_idx + 4]
                } else {
                    values[values.len() - 1]
                },
            ]);
            if compute.contains(Compute::PEAKS) {
                let mask = chunk.simd_gt(left) & chunk.simd_gt(right);
                state.peaks += mask.to_bitmask().count_ones();
            }
            if compute.contains(Compute::TROUGHS) {
                let mask = chunk.simd_lt(left) & chunk.simd_lt(right);
                state.troughs += mask.to_bitmask().count_ones();
            }
        }

        if compute.contains(Compute::SLOPE) {
            let indices = f32x4::from_array([offset, offset + 1.0, offset + 2.0, offset + 3.0]);
            state.sum_ix_vec += indices * chunk;
        }

        if compute.contains(Compute::CALC_CENTROID) {
            let indices = f32x4::from_array([offset, offset + 1.0, offset + 2.0, offset + 3.0]);
            state.t_energy_vec += indices * (chunk * chunk);
        }

        if compute.contains(Compute::PAA) {
            for (t_idx, _total) in unique_paa_totals.iter().enumerate() {
                let b = &paa_boundaries[t_idx];
                let seg_idx = &mut state.current_paa_segs[t_idx];
                if global_idx + LANES <= b[*seg_idx + 1] {
                    state.paa_sums[t_idx][*seg_idx] += chunk.reduce_sum();
                } else {
                    for (j, v) in i.iter().enumerate().take(LANES) {
                        let idx = global_idx + j;
                        while *seg_idx < state.paa_sums[t_idx].len() - 1
                            && idx >= b[*seg_idx + 1]
                        {
                            *seg_idx += 1;
                        }
                        state.paa_sums[t_idx][*seg_idx] += v;
                    }
                }
            }
        }

        if compute.contains(Compute::C3) {
            for (l_idx, &lag) in unique_c3_lags.iter().enumerate() {
                let l = lag as usize;
                if global_idx >= 2 * l {
                    let chunk_il =
                        f32x4::from_slice(&values[global_idx - l..global_idx - l + 4]);
                    let chunk_i2l =
                        f32x4::from_slice(&values[global_idx - 2 * l..global_idx - 2 * l + 4]);
                    state.c3_sums_vec[l_idx] += chunk * chunk_il * chunk_i2l;
                } else {
                    for j in 0..LANES {
                        let idx = global_idx + j;
                        if idx >= 2 * l {
                            state.c3_sums[l_idx] +=
                                values[idx] * values[idx - l] * values[idx - 2 * l];
                        }
                    }
                }
            }
        }
    }

    #[inline(always)]
    pub fn finalize_simd(compute: Compute, state: &mut ColumnState) {
        if compute.contains(Compute::SLOPE) {
            state.sum_ix += state.sum_ix_vec.reduce_sum();
            state.sum_ix_vec = f32x4::splat(0.0);
        }
        if compute.contains(Compute::CALC_CENTROID) {
            state.t_energy += state.t_energy_vec.reduce_sum();
            state.t_energy_vec = f32x4::splat(0.0);
        }
        if compute.contains(Compute::C3) {
            for l_idx in 0..state.c3_sums.len() {
                state.c3_sums[l_idx] += state.c3_sums_vec[l_idx].reduce_sum();
                state.c3_sums_vec[l_idx] = f32x4::splat(0.0);
            }
        }
    }

    #[inline(always)]
    pub fn process_sequential(
        compute: Compute,
        val: f32,
        global_idx: usize,
        values: &[f32],
        state: &mut ColumnState,
        unique_paa_totals: &[u16],
        paa_boundaries: &[Vec<usize>],
        unique_c3_lags: &[u16],
    ) {
        if global_idx > 0 && global_idx < values.len() - 1 {
            if compute.contains(Compute::PEAKS)
                && val > values[global_idx - 1]
                && val > values[global_idx + 1]
            {
                state.peaks += 1;
            }
            if compute.contains(Compute::TROUGHS)
                && val < values[global_idx - 1]
                && val < values[global_idx + 1]
            {
                state.troughs += 1;
            }
        }
        if compute.contains(Compute::SLOPE) {
            state.sum_ix += (global_idx as f32) * val;
        }
        if compute.contains(Compute::CALC_CENTROID) {
            state.t_energy += (global_idx as f32) * val * val;
        }
        if compute.contains(Compute::PAA) {
            for (t_idx, _total) in unique_paa_totals.iter().enumerate() {
                let b = &paa_boundaries[t_idx];
                let seg_idx = &mut state.current_paa_segs[t_idx];
                while *seg_idx < state.paa_sums[t_idx].len() - 1 && global_idx >= b[*seg_idx + 1] {
                    *seg_idx += 1;
                }
                state.paa_sums[t_idx][*seg_idx] += val;
            }
        }
        if compute.contains(Compute::C3) {
            for (l_idx, &lag) in unique_c3_lags.iter().enumerate() {
                let l = lag as usize;
                if global_idx >= 2 * l {
                    state.c3_sums[l_idx] += val * values[global_idx - l] * values[global_idx - 2 * l];
                }
            }
        }
    }
}
