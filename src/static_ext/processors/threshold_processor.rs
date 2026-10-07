use crate::common::ColumnState;
use crate::types::Compute;
use std::simd::cmp::SimdPartialOrd;
use std::simd::f32x4;

pub struct ThresholdProcessor;

impl ThresholdProcessor {
    #[inline(always)]
    pub fn process_simd(
        compute: Compute,
        chunk: f32x4,
        unique_count_above_thresholds: &[u32],
        unique_count_below_thresholds: &[u32],
        unique_range_counts: &[(u32, u32)],
        state: &mut ColumnState,
    ) {
        if compute.contains(Compute::COUNT_ABOVE) {
            for (idx, &t_bits) in unique_count_above_thresholds.iter().enumerate() {
                let t = f32x4::splat(f32::from_bits(t_bits));
                let mask = chunk.simd_ge(t);
                state.count_above_counts[idx] += mask.to_bitmask().count_ones();
            }
        }
        if compute.contains(Compute::COUNT_BELOW) {
            for (idx, &t_bits) in unique_count_below_thresholds.iter().enumerate() {
                let t = f32x4::splat(f32::from_bits(t_bits));
                let mask = chunk.simd_le(t);
                state.count_below_counts[idx] += mask.to_bitmask().count_ones();
            }
        }
        if compute.contains(Compute::RANGE_COUNT) {
            for (idx, &(min_bits, max_bits)) in unique_range_counts.iter().enumerate() {
                let min_v = f32x4::splat(f32::from_bits(min_bits));
                let max_v = f32x4::splat(f32::from_bits(max_bits));
                let mask = chunk.simd_ge(min_v) & chunk.simd_lt(max_v);
                state.range_counts[idx] += mask.to_bitmask().count_ones();
            }
        }
    }

    #[inline(always)]
    pub fn process_remainder(
        compute: Compute,
        val: f32,
        unique_count_above_thresholds: &[u32],
        unique_count_below_thresholds: &[u32],
        unique_range_counts: &[(u32, u32)],
        state: &mut ColumnState,
    ) {
        if compute.contains(Compute::COUNT_ABOVE) {
            for (idx, &t_bits) in unique_count_above_thresholds.iter().enumerate() {
                if val >= f32::from_bits(t_bits) {
                    state.count_above_counts[idx] += 1;
                }
            }
        }
        if compute.contains(Compute::COUNT_BELOW) {
            for (idx, &t_bits) in unique_count_below_thresholds.iter().enumerate() {
                if val <= f32::from_bits(t_bits) {
                    state.count_below_counts[idx] += 1;
                }
            }
        }
        if compute.contains(Compute::RANGE_COUNT) {
            for (idx, &(min_bits, max_bits)) in unique_range_counts.iter().enumerate() {
                if val >= f32::from_bits(min_bits) && val < f32::from_bits(max_bits) {
                    state.range_counts[idx] += 1;
                }
            }
        }
    }
}
