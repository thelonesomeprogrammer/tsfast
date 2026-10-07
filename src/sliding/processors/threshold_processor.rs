use crate::common::ColumnState;
use crate::types::Compute;
use std::simd::cmp::SimdPartialOrd;
use std::simd::f32x4;

pub struct ThresholdProcessor;

impl ThresholdProcessor {
    #[inline(always)]
    pub fn reset_state(state: &mut ColumnState) {
        for c in &mut state.count_above_counts {
            *c = 0;
        }
        for c in &mut state.count_below_counts {
            *c = 0;
        }
        for c in &mut state.range_counts {
            *c = 0;
        }
    }

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

    /// Values leaving the window decrement their counters; values entering it
    /// increment theirs. No rescan: each stride costs O(#thresholds).
    #[inline(always)]
    pub fn update_batch(
        compute: Compute,
        old_slice: &[f32],
        new_slice: &[f32],
        unique_count_above_thresholds: &[u32],
        unique_count_below_thresholds: &[u32],
        unique_range_counts: &[(u32, u32)],
        state: &mut ColumnState,
    ) {
        if compute.contains(Compute::COUNT_ABOVE) {
            for (idx, &t_bits) in unique_count_above_thresholds.iter().enumerate() {
                let t = f32::from_bits(t_bits);
                let mut delta: i64 = 0;
                for &v in new_slice {
                    if v >= t {
                        delta += 1;
                    }
                }
                for &v in old_slice {
                    if v >= t {
                        delta -= 1;
                    }
                }
                state.count_above_counts[idx] =
                    (state.count_above_counts[idx] as i64 + delta) as u32;
            }
        }
        if compute.contains(Compute::COUNT_BELOW) {
            for (idx, &t_bits) in unique_count_below_thresholds.iter().enumerate() {
                let t = f32::from_bits(t_bits);
                let mut delta: i64 = 0;
                for &v in new_slice {
                    if v <= t {
                        delta += 1;
                    }
                }
                for &v in old_slice {
                    if v <= t {
                        delta -= 1;
                    }
                }
                state.count_below_counts[idx] =
                    (state.count_below_counts[idx] as i64 + delta) as u32;
            }
        }
        if compute.contains(Compute::RANGE_COUNT) {
            for (idx, &(min_bits, max_bits)) in unique_range_counts.iter().enumerate() {
                let min_v = f32::from_bits(min_bits);
                let max_v = f32::from_bits(max_bits);
                let mut delta: i64 = 0;
                for &v in new_slice {
                    if v >= min_v && v < max_v {
                        delta += 1;
                    }
                }
                for &v in old_slice {
                    if v >= min_v && v < max_v {
                        delta -= 1;
                    }
                }
                state.range_counts[idx] = (state.range_counts[idx] as i64 + delta) as u32;
            }
        }
    }

    #[inline(always)]
    pub fn update_incremental(
        compute: Compute,
        old_val: f32,
        new_val: f32,
        unique_count_above_thresholds: &[u32],
        unique_count_below_thresholds: &[u32],
        unique_range_counts: &[(u32, u32)],
        state: &mut ColumnState,
    ) {
        if compute.contains(Compute::COUNT_ABOVE) {
            for (idx, &t_bits) in unique_count_above_thresholds.iter().enumerate() {
                let t = f32::from_bits(t_bits);
                if new_val >= t {
                    state.count_above_counts[idx] += 1;
                }
                if old_val >= t {
                    state.count_above_counts[idx] -= 1;
                }
            }
        }
        if compute.contains(Compute::COUNT_BELOW) {
            for (idx, &t_bits) in unique_count_below_thresholds.iter().enumerate() {
                let t = f32::from_bits(t_bits);
                if new_val <= t {
                    state.count_below_counts[idx] += 1;
                }
                if old_val <= t {
                    state.count_below_counts[idx] -= 1;
                }
            }
        }
        if compute.contains(Compute::RANGE_COUNT) {
            for (idx, &(min_bits, max_bits)) in unique_range_counts.iter().enumerate() {
                let min_v = f32::from_bits(min_bits);
                let max_v = f32::from_bits(max_bits);
                if new_val >= min_v && new_val < max_v {
                    state.range_counts[idx] += 1;
                }
                if old_val >= min_v && old_val < max_v {
                    state.range_counts[idx] -= 1;
                }
            }
        }
    }
}
