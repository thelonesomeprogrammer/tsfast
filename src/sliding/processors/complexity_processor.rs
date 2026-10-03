use crate::common::{ColumnState, LANES};
use crate::types::Compute;
use std::simd::f32x4;
use std::simd::num::SimdFloat;

pub struct ComplexityProcessor;

impl ComplexityProcessor {
    #[inline(always)]
    pub fn reset_state(state: &mut ColumnState) {
        for s in &mut state.paa_sums {
            s.fill(0.0);
        }
        for s in &mut state.current_paa_segs {
            *s = 0;
        }
        for s in &mut state.c3_sums {
            *s = 0.0;
        }
    }

    #[inline(always)]
    pub fn process_simd(
        compute: Compute,
        chunk: std::simd::f32x4,
        global_idx: usize,
        values: &[f32],
        unique_paa_totals: &[u16],
        paa_boundaries: &[Vec<usize>],
        unique_c3_lags: &[u16],
        state: &mut ColumnState,
    ) {
        if compute.contains(Compute::PAA) {
            for (t_idx, _total) in unique_paa_totals.iter().enumerate() {
                let b = &paa_boundaries[t_idx];
                let seg_idx = &mut state.current_paa_segs[t_idx];
                if global_idx + LANES <= b[*seg_idx + 1] {
                    state.paa_sums[t_idx][*seg_idx] += chunk.reduce_sum();
                } else {
                    for (j, v) in chunk.as_array().iter().enumerate().take(LANES) {
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
        if compute.contains(Compute::C3) {
            for l_idx in 0..state.c3_sums.len() {
                state.c3_sums[l_idx] += state.c3_sums_vec[l_idx].reduce_sum();
                state.c3_sums_vec[l_idx] = f32x4::splat(0.0);
            }
        }
    }

    #[inline(always)]
    pub fn process_remainder(
        compute: Compute,
        val: f32,
        global_idx: usize,
        values: &[f32],
        unique_paa_totals: &[u16],
        paa_boundaries: &[Vec<usize>],
        unique_c3_lags: &[u16],
        state: &mut ColumnState,
    ) {
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
