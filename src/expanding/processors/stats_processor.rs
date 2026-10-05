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
        if compute.contains(Compute::SUM) {
            state.sum_vec += chunk;
        }

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
            state.abs_max_vec = state.abs_max_vec.simd_max(chunk.abs());
        }
        let sq = chunk * chunk;
        state.energy_vec += sq;
        state.sum_cubes_vec += sq * chunk;
        state.sum_quads_vec += sq * sq;
    }

    #[inline(always)]
    pub fn finalize_simd(compute: Compute, state: &mut ColumnState) {
        if compute.contains(Compute::SUM) {
            state.total_sum += state.sum_vec.reduce_sum();
            state.sum_vec = f32x4::splat(0.0);
        }
        if compute.contains(Compute::ENERGY) {
            state.energy += state.energy_vec.reduce_sum();
            state.energy_vec = f32x4::splat(0.0);
            if compute.contains(Compute::SKEW) {
                state.sum_cubes += state.sum_cubes_vec.reduce_sum();
                state.sum_cubes_vec = f32x4::splat(0.0);
            }
            if compute.contains(Compute::KURTOSIS) {
                state.sum_quads += state.sum_quads_vec.reduce_sum();
                state.sum_quads_vec = f32x4::splat(0.0);
            }
        }
        if compute.contains(Compute::ABS_MAX) {
            state.abs_max = state.abs_max.max(state.abs_max_vec.reduce_max());
            state.abs_max_vec = f32x4::splat(0.0);
        }
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

    pub fn finalize_base_metrics(state: &mut ColumnState, n: f32) -> crate::metrics::BaseMetrics {
        let mean = state.total_sum / n;
        let m2 = state.energy - (state.total_sum * state.total_sum) / n;
        let m3 = state.sum_cubes - 3.0 * mean * state.energy + 2.0 * mean * mean * state.total_sum;
        let m4 = state.sum_quads - 4.0 * mean * state.sum_cubes + 6.0 * mean * mean * state.energy
            - 3.0 * mean * mean * mean * state.total_sum;

        // Population variance (ddof=0), as tsfresh and TSFEL.
        let var = if n > 0.0 { m2 / n } else { 0.0 };
        let std_dev = var.sqrt();

        crate::metrics::BaseMetrics {
            mean,
            m2,
            m3,
            m4,
            var,
            std_dev,
        }
    }
}
