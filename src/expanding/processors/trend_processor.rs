use crate::common::ColumnState;
use crate::types::Compute;
use std::simd::f32x4;
use std::simd::num::SimdFloat;

pub struct TrendProcessor;

impl TrendProcessor {
    #[inline(always)]
    pub fn process_simd(
        compute: Compute,
        chunk: f32x4,
        offset: f32,
        state: &mut ColumnState,
    ) {
        if compute.contains(Compute::SLOPE) {
            let indices = f32x4::from_array([offset, offset + 1.0, offset + 2.0, offset + 3.0]);
            state.sum_ix += (indices * chunk).reduce_sum();
        }
    }

    #[inline(always)]
    pub fn process_sequential(
        compute: Compute,
        val: f32,
        global_idx: usize,
        global_start_idx: usize,
        state: &mut ColumnState,
        full_series: &[f32],
        values: &[f32],
        unique_c3_lags: &[u16],
        unique_autocorr_lags: &[u16],
    ) {
        if compute.contains(Compute::PEAKS) {
            if state.prev_val > state.prev_prev_val && state.prev_val > val {
                state.peaks += 1;
            }
            state.prev_prev_val = state.prev_val;
            state.prev_val = val;
        }

        if compute.contains(Compute::C3) {
            for (l_idx, &lag) in unique_c3_lags.iter().enumerate() {
                let l = lag as usize;
                if global_idx >= 2 * l {
                    let v_l = if global_idx - l < global_start_idx {
                        full_series[global_idx - l]
                    } else {
                        values[global_idx - l - global_start_idx]
                    };
                    let v_2l = if global_idx - 2 * l < global_start_idx {
                        full_series[global_idx - 2 * l]
                    } else {
                        values[global_idx - 2 * l - global_start_idx]
                    };
                    state.c3_sums[l_idx] += val * v_l * v_2l;
                }
            }
        }

        if compute.intersects(Compute::FULL_AUTOCORR | Compute::PACF) {
            for (l_idx, &lag) in unique_autocorr_lags.iter().enumerate() {
                let l = lag as usize;
                if global_idx >= l {
                    let v_l = if global_idx - l < global_start_idx {
                        full_series[global_idx - l]
                    } else {
                        values[global_idx - l - global_start_idx]
                    };
                    state.autocorr_sums[l_idx] += val * v_l;
                }
            }
        }

        if compute.contains(Compute::SLOPE) {
            state.sum_ix += (global_idx as f32) * val;
        }
    }
}
