use crate::common::ColumnState;
use crate::types::Compute;
use std::simd::f32x4;
use std::simd::num::SimdFloat;

pub struct StatsProcessor;

impl StatsProcessor {
    #[inline(always)]
    pub fn process_simd(
        compute: Compute,
        chunk: f32x4,
        i: &[f32],
        offset_usize: usize,
        state: &mut ColumnState,
    ) {
        state.total_sum += chunk.reduce_sum();

        let c_min = chunk.reduce_min();
        if c_min < state.min_value {
            state.min_value = c_min;
            for bit in 0..4 {
                if i[bit] == c_min {
                    state.first_min_idx = offset_usize + bit;
                    break;
                }
            }
        }
        if c_min <= state.min_value {
            for bit in (0..4).rev() {
                if i[bit] == state.min_value {
                    state.last_min_idx = offset_usize + bit;
                    break;
                }
            }
        }

        let c_max = chunk.reduce_max();
        if c_max > state.max_value {
            state.max_value = c_max;
            for bit in 0..4 {
                if i[bit] == c_max {
                    state.first_max_idx = offset_usize + bit;
                    break;
                }
            }
        }
        if c_max >= state.max_value {
            for bit in (0..4).rev() {
                if i[bit] == state.max_value {
                    state.last_max_idx = offset_usize + bit;
                    break;
                }
            }
        }

        if compute.contains(Compute::ABS_MAX) {
            state.abs_max = state.abs_max.max(chunk.abs().reduce_max());
        }
        let sq = chunk * chunk;
        state.energy += sq.reduce_sum();
        state.sum_cubes += (sq * chunk).reduce_sum();
        state.sum_quads += (sq * sq).reduce_sum();
    }

    #[inline(always)]
    pub fn process_remainder(
        compute: Compute,
        val: f32,
        global_idx: usize,
        state: &mut ColumnState,
    ) {
        if compute.contains(Compute::SUM) {
            state.total_sum += val;
        }
        if compute.contains(Compute::MIN) {
            if val < state.min_value {
                state.min_value = val;
                state.first_min_idx = global_idx;
            }
            if val <= state.min_value {
                state.last_min_idx = global_idx;
            }
        }
        if compute.contains(Compute::MAX) {
            if val > state.max_value {
                state.max_value = val;
                state.first_max_idx = global_idx;
            }
            if val >= state.max_value {
                state.last_max_idx = global_idx;
            }
        }
        if compute.contains(Compute::ABS_MAX) {
            state.abs_max = state.abs_max.max(val.abs());
        }
        if compute.contains(Compute::ENERGY) {
            let sq = val * val;
            state.energy += sq;
            if compute.contains(Compute::SKEW) {
                state.sum_cubes += sq * val;
            }
            if compute.contains(Compute::KURTOSIS) {
                state.sum_quads += sq * sq;
            }
        }
    }
}
