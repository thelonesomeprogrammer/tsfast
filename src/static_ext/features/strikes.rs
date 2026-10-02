use crate::common::LANES;
use std::simd::f32x4;
use std::simd::num::SimdFloat;

#[inline(always)]
pub fn compute_strike_features(
    compute: &crate::types::FastBitArray,
    values: &[f32],
    mean: f32,
    mad_sum: &mut f32,
    count_a: &mut usize,
    count_b: &mut usize,
    max_strike_a: &mut usize,
    max_strike_b: &mut usize,
) {
    if compute.any([9, 25, 26, 27, 28]) {
        let mean_vec = f32x4::splat(mean);
        let mut mad_sum_vec = f32x4::splat(0.0);

        for chunk in values.chunks_exact(LANES) {
            let c = f32x4::from_slice(chunk);
            if compute[9] {
                mad_sum_vec += (c - mean_vec).abs();
            }
        }
        *mad_sum = mad_sum_vec.reduce_sum();

        let rem_start = (values.len() / LANES) * LANES;
        for &val in &values[rem_start..] {
            if compute[9] {
                *mad_sum += (val - mean).abs();
            }
        }

        if compute.any([25, 26, 27, 28]) {
            let mut current_strike_a = 0;
            let mut current_strike_b = 0;
            for &val in values {
                if val > mean {
                    *count_a += 1;
                    current_strike_a += 1;
                    *max_strike_a = (*max_strike_a).max(current_strike_a);
                    current_strike_b = 0;
                } else if val < mean {
                    *count_b += 1;
                    current_strike_b += 1;
                    *max_strike_b = (*max_strike_b).max(current_strike_b);
                    current_strike_a = 0;
                } else {
                    current_strike_a = 0;
                    current_strike_b = 0;
                }
            }
        }
    }
}
