use crate::common::ColumnState;
use crate::types::Compute;
use std::simd::cmp::SimdPartialOrd;
use std::simd::f32x4;
use std::simd::num::SimdFloat;

pub struct StatsProcessor;

impl StatsProcessor {
    #[inline(always)]
    pub fn process_simd(compute: Compute, chunk: f32x4, state: &mut ColumnState) {
        if compute.contains(Compute::SUM) {
            state.total_sum += chunk.reduce_sum();
        }
        if compute.contains(Compute::MIN) {
            state.min_value = state.min_value.min(chunk.reduce_min());
        }
        if compute.contains(Compute::MAX) {
            state.max_value = state.max_value.max(chunk.reduce_max());
        }
        if compute.contains(Compute::ENERGY) {
            let sq = chunk * chunk;
            state.energy += sq.reduce_sum();
            if compute.contains(Compute::SKEW) {
                state.sum_cubes += (sq * chunk).reduce_sum();
            }
            if compute.contains(Compute::KURTOSIS) {
                state.sum_quads += (sq * sq).reduce_sum();
            }
        }
        if compute.contains(Compute::ABS_MAX) {
            state.abs_max = state.abs_max.max(chunk.abs().reduce_max());
        }
        if compute.contains(Compute::ABS_SUM) {
            state.abs_sum += chunk.abs().reduce_sum();
        }
    }

    #[inline(always)]
    pub fn process_remainder(compute: Compute, val: f32, state: &mut ColumnState) {
        if compute.contains(Compute::SUM) {
            state.total_sum += val;
        }
        if compute.contains(Compute::MIN) {
            state.min_value = state.min_value.min(val);
        }
        if compute.contains(Compute::MAX) {
            state.max_value = state.max_value.max(val);
        }
        if compute.contains(Compute::ABS_MAX) {
            state.abs_max = state.abs_max.max(val.abs());
        }
        if compute.contains(Compute::ABS_SUM) {
            state.abs_sum += val.abs();
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
