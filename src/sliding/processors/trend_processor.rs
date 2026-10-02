use crate::common::ColumnState;
use crate::types::Compute;
use std::simd::f32x4;
use std::simd::num::SimdFloat;

pub struct TrendProcessor;

impl TrendProcessor {
    #[inline(always)]
    pub fn update_batch(
        compute: Compute,
        old_slice: &[f32],
        new_slice: &[f32],
        window_size: usize,
        state: &mut ColumnState,
    ) {
        if compute.contains(Compute::SLOPE) {
            let y = old_slice.len();
            let mut sum_old = 0.0;
            let mut sum_ix_old = 0.0;
            for (i, &v) in old_slice.iter().enumerate() {
                sum_old += v;
                sum_ix_old += (i as f32) * v;
            }

            let mut sum_ix_new = 0.0;
            let w_f = window_size as f32;
            let y_f = y as f32;
            for (j, &v) in new_slice.iter().enumerate() {
                sum_ix_new += (w_f - y_f + j as f32) * v;
            }

            let old_total_sum = state.total_sum - (new_slice.iter().sum::<f32>() - sum_old);
            let rem_sum = old_total_sum - sum_old;

            state.sum_ix = (state.sum_ix - sum_ix_old) - (y_f * rem_sum) + sum_ix_new;
        }
    }

    #[inline(always)]
    pub fn update_incremental(
        compute: Compute,
        new_val: f32,
        window_size: usize,
        state: &mut ColumnState,
    ) {
        if compute.contains(Compute::SLOPE) {
            state.sum_ix =
                state.sum_ix - (state.total_sum - new_val) + (window_size as f32 - 1.0) * new_val;
        }
    }

    #[inline(always)]
    pub fn process_simd(
        compute: Compute,
        chunk: std::simd::f32x4,
        global_idx: usize,
        state: &mut ColumnState,
    ) {
        if compute.contains(Compute::SLOPE) {
            let offset = global_idx as f32;
            let indices = f32x4::from_array([offset, offset + 1.0, offset + 2.0, offset + 3.0]);
            state.sum_ix += (indices * chunk).reduce_sum();
        }
    }

    #[inline(always)]
    pub fn process_remainder(
        compute: Compute,
        val: f32,
        global_idx: usize,
        state: &mut ColumnState,
    ) {
        if compute.contains(Compute::SLOPE) {
            state.sum_ix += (global_idx as f32) * val;
        }
    }
}
