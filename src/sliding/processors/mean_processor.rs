use crate::types::Compute;
use std::simd::f32x4;
use std::simd::num::SimdFloat;

use crate::metrics::MeanMetrics;

pub struct MeanProcessor;

impl MeanProcessor {
    pub fn finalize(compute: Compute, values: &[f32], n: f32, mean: f32) -> MeanMetrics {
        let mut mad_sum = 0.0;
        let mut count_a = 0;
        let mut count_b = 0;
        let mut max_strike_a = 0;
        let mut max_strike_b = 0;

        if compute.intersects(
            Compute::MAD
                | Compute::CNT_ABOVE_MEAN
                | Compute::CNT_BELOW_MEAN
                | Compute::STRIKE_ABOVE
                | Compute::STRIKE_BELOW,
        ) {
            let mut mad_sum_val = 0.0;

            // Use common lanes without importing directly
            const LANES: usize = 4;
            let mean_vec = f32x4::splat(mean);

            let mut mad_sum_vec = std::simd::f32x4::splat(0.0);
            for chunk in values.as_chunks::<LANES>().0 {
                let c = std::simd::f32x4::from_slice(chunk);
                if compute.contains(Compute::MAD) {
                    mad_sum_vec += (c - mean_vec).abs();
                }
            }
            mad_sum_val += mad_sum_vec.reduce_sum();

            let rem_start = (values.len() / LANES) * LANES;
            for &val in &values[rem_start..] {
                if compute.contains(Compute::MAD) {
                    mad_sum_val += (val - mean).abs();
                }
            }
            mad_sum = mad_sum_val / n;

            if compute.intersects(
                Compute::CNT_ABOVE_MEAN
                    | Compute::CNT_BELOW_MEAN
                    | Compute::STRIKE_ABOVE
                    | Compute::STRIKE_BELOW,
            ) {
                let mut current_strike_a = 0;
                let mut current_strike_b = 0;
                for &val in values {
                    if val > mean {
                        count_a += 1;
                        current_strike_a += 1;
                        max_strike_a = max_strike_a.max(current_strike_a);
                        current_strike_b = 0;
                    } else if val < mean {
                        count_b += 1;
                        current_strike_b += 1;
                        max_strike_b = max_strike_b.max(current_strike_b);
                        current_strike_a = 0;
                    } else {
                        current_strike_a = 0;
                        current_strike_b = 0;
                    }
                }
            }
        }

        MeanMetrics {
            mad_sum,
            count_a,
            count_b,
            max_strike_a,
            max_strike_b,
        }
    }
}
