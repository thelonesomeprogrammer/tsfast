use crate::common::ColumnState;
use crate::common::LANES;
use crate::types::Compute;
use std::simd::cmp::SimdPartialOrd;
use std::simd::f32x4;
use std::simd::num::SimdFloat;

use crate::metrics::MeanMetrics;

pub struct MeanProcessor;

impl MeanProcessor {
    pub fn finalize(
        compute: Compute,
        full_series: &[f32],
        n: f32,
        mean: f32,
        state: &ColumnState,
    ) -> (MeanMetrics, f32) {
        let mut mad_sum = 0.0;
        let mut count_a = 0;
        let mut count_b = 0;
        let mut max_strike_a = 0;
        let mut current_strike_a = 0;
        let mut max_strike_b = 0;
        let mut current_strike_b = 0;
        let mut entropy = 0.0;

        if compute.intersects(Compute::MAD | Compute::ENTROPY | Compute::CNT_ABOVE_MEAN | Compute::CNT_BELOW_MEAN | Compute::STRIKE_ABOVE | Compute::STRIKE_BELOW) {
            let range = state.max_value - state.min_value;
            let bins = 10;
            let mut counts: [usize; 10] = [0; 10];

            let mean_vec = f32x4::splat(mean);
            let mut mad_sum_vec = f32x4::splat(0.0);

            for chunk in full_series.chunks_exact(LANES) {
                let c = f32x4::from_slice(chunk);
                if compute.contains(Compute::MAD) {
                    mad_sum_vec += (c - mean_vec).abs();
                }

                if compute.contains(Compute::CNT_ABOVE_MEAN) {
                    count_a += c.simd_gt(mean_vec).to_bitmask().count_ones() as usize;
                }
                if compute.contains(Compute::CNT_BELOW_MEAN) {
                    count_b += c.simd_lt(mean_vec).to_bitmask().count_ones() as usize;
                }

                for &v in chunk {
                    if compute.contains(Compute::ENTROPY) && range > 1e-9 {
                        let b = (((v - state.min_value) / range) * (bins as f32 - 1.0)) as usize;
                        counts[b.min(bins - 1)] += 1;
                    }

                    if compute.intersects(Compute::STRIKE_ABOVE | Compute::STRIKE_BELOW) {
                        if v > mean {
                            current_strike_a += 1;
                            max_strike_a = max_strike_a.max(current_strike_a);
                            current_strike_b = 0;
                        } else if v < mean {
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
            mad_sum = mad_sum_vec.reduce_sum();

            let rem_start = (full_series.len() / LANES) * LANES;
            for &v in &full_series[rem_start..] {
                if compute.contains(Compute::MAD) {
                    mad_sum += (v - mean).abs();
                }
                if compute.contains(Compute::ENTROPY) && range > 1e-9 {
                    let b = (((v - state.min_value) / range) * (bins as f32 - 1.0)) as usize;
                    counts[b.min(bins - 1)] += 1;
                }
                if compute.contains(Compute::CNT_ABOVE_MEAN) && v > mean {
                    count_a += 1;
                }
                if compute.contains(Compute::CNT_BELOW_MEAN) && v < mean {
                    count_b += 1;
                }
                if compute.intersects(Compute::STRIKE_ABOVE | Compute::STRIKE_BELOW) {
                    if v > mean {
                        current_strike_a += 1;
                        max_strike_a = max_strike_a.max(current_strike_a);
                        current_strike_b = 0;
                    } else if v < mean {
                        current_strike_b += 1;
                        max_strike_b = max_strike_b.max(current_strike_b);
                        current_strike_a = 0;
                    } else {
                        current_strike_a = 0;
                        current_strike_b = 0;
                    }
                }
            }
            mad_sum /= n;

            if compute.contains(Compute::ENTROPY) && range > 1e-9 {
                for &c in &counts {
                    if c > 0 {
                        let p = c as f32 / n;
                        entropy -= p * p.ln();
                    }
                }
            }
        }

        (MeanMetrics {
            mad_sum,
            count_a,
            count_b,
            max_strike_a,
            max_strike_b,
        }, entropy)
    }
}
