use crate::common::ColumnState;
use crate::types::Compute;

use crate::metrics::BaseMetrics;
use std::simd::f32x4;
use std::simd::num::SimdFloat;

pub struct StatsProcessor;

impl StatsProcessor {
    #[inline(always)]
    pub fn reset_state(state: &mut ColumnState, is_incremental: bool) {
        if !is_incremental {
            state.total_sum = 0.0;
            state.min_value = f32::INFINITY;
            state.max_value = f32::NEG_INFINITY;
            state.energy = 0.0;
            state.sum_cubes = 0.0;
            state.sum_quads = 0.0;
            // Maintained incrementally by TrendProcessor; only a full recompute rebuilds it.
            state.t_energy = 0.0;

            state.min_q_head = 0;
            state.min_q_tail = 0;
            state.min_q_len = 0;
            state.max_q_head = 0;
            state.max_q_tail = 0;
            state.max_q_len = 0;
            state.abs_max = 0.0;
        }
    }

    #[inline(always)]
    pub fn update_batch(
        compute: Compute,
        old_slice: &[f32],
        new_slice: &[f32],
        state: &mut ColumnState,
    ) {
        if compute.intersects(
            Compute::SUM
                | Compute::MEAN
                | Compute::VARIANCE
                | Compute::STD
                | Compute::SKEW
                | Compute::KURTOSIS
                | Compute::MAD
                | Compute::ENERGY
                | Compute::RMS
                | Compute::ROOT_MEAN_SQ
                | Compute::CNT_ABOVE_MEAN
                | Compute::CNT_BELOW_MEAN
                | Compute::STRIKE_ABOVE
                | Compute::STRIKE_BELOW
                | Compute::VAR_COEFF
                | Compute::FULL_AUTOCORR
                | Compute::PACF,
        ) {
            let mut sum_old = 0.0;
            let mut energy_old = 0.0;
            let mut cubes_old = 0.0;
            let mut quads_old = 0.0;
            for &v in old_slice {
                let sq = v * v;
                sum_old += v;
                energy_old += sq;
                if compute.contains(Compute::SKEW) {
                    cubes_old += sq * v;
                }
                if compute.contains(Compute::KURTOSIS) {
                    quads_old += sq * sq;
                }
            }

            let mut sum_new = 0.0;
            let mut energy_new = 0.0;
            let mut cubes_new = 0.0;
            let mut quads_new = 0.0;
            for &v in new_slice {
                let sq = v * v;
                sum_new += v;
                energy_new += sq;
                if compute.contains(Compute::SKEW) {
                    cubes_new += sq * v;
                }
                if compute.contains(Compute::KURTOSIS) {
                    quads_new += sq * sq;
                }
            }

            state.total_sum += sum_new - sum_old;
            state.energy += energy_new - energy_old;
            if compute.contains(Compute::SKEW) {
                state.sum_cubes += cubes_new - cubes_old;
            }
            if compute.contains(Compute::KURTOSIS) {
                state.sum_quads += quads_new - quads_old;
            }
        }
    }

    #[inline(always)]
    pub fn update_incremental(
        compute: Compute,
        old_val: f32,
        new_val: f32,
        state: &mut ColumnState,
    ) {
        if compute.intersects(
            Compute::SUM
                | Compute::MEAN
                | Compute::VARIANCE
                | Compute::STD
                | Compute::SKEW
                | Compute::KURTOSIS
                | Compute::MAD
                | Compute::ENERGY
                | Compute::RMS
                | Compute::ROOT_MEAN_SQ
                | Compute::CNT_ABOVE_MEAN
                | Compute::CNT_BELOW_MEAN
                | Compute::STRIKE_ABOVE
                | Compute::STRIKE_BELOW
                | Compute::VAR_COEFF
                | Compute::FULL_AUTOCORR
                | Compute::PACF,
        ) {
            state.total_sum += new_val - old_val;
            let old_sq = old_val * old_val;
            let new_sq = new_val * new_val;
            state.energy += new_sq - old_sq;
            if compute.contains(Compute::SKEW) {
                state.sum_cubes += new_sq * new_val - old_sq * old_val;
            }
            if compute.contains(Compute::KURTOSIS) {
                state.sum_quads += new_sq * new_sq - old_sq * old_sq;
            }
        }
    }

    #[inline(always)]
    pub fn process_simd(
        compute: Compute,
        chunk: std::simd::f32x4,
        global_idx: usize,
        window_size: usize,
        state: &mut ColumnState,
    ) {
        if compute.contains(Compute::SUM) {
            state.sum_vec += chunk;
        }
        if compute.contains(Compute::MIN) {
            state.min_vec = state.min_vec.simd_min(chunk);
        }
        if compute.contains(Compute::MAX) {
            state.max_vec = state.max_vec.simd_max(chunk);
        }
        if compute.contains(Compute::ENERGY) {
            let sq = chunk * chunk;
            state.energy_vec += sq;
            if compute.contains(Compute::SKEW) {
                state.sum_cubes_vec += sq * chunk;
            }
            if compute.contains(Compute::KURTOSIS) {
                state.sum_quads_vec += sq * sq;
            }
        }

        if compute.contains(Compute::ABS_MAX) {
            state.abs_max_vec = state.abs_max_vec.simd_max(chunk.abs());
        }

        if compute.intersects(Compute::MIN | Compute::MAX | Compute::IQR | Compute::ENTROPY) {
            if state.min_queue.is_empty() {
                state.min_queue = vec![(0, 0.0); window_size];
                state.max_queue = vec![(0, 0.0); window_size];
            }
            for (j, &val) in chunk.as_array().iter().enumerate() {
                let idx = global_idx + j;
                // Min Queue
                while state.min_q_len > 0 {
                    let prev_tail = if state.min_q_tail == 0 {
                        window_size - 1
                    } else {
                        state.min_q_tail - 1
                    };
                    if state.min_queue[prev_tail].1 >= val {
                        state.min_q_tail = prev_tail;
                        state.min_q_len -= 1;
                    } else {
                        break;
                    }
                }
                state.min_queue[state.min_q_tail] = (idx, val);
                state.min_q_tail += 1;
                if state.min_q_tail >= window_size {
                    state.min_q_tail -= window_size;
                }
                state.min_q_len += 1;

                // Max Queue
                while state.max_q_len > 0 {
                    let prev_tail = if state.max_q_tail == 0 {
                        window_size - 1
                    } else {
                        state.max_q_tail - 1
                    };
                    if state.max_queue[prev_tail].1 <= val {
                        state.max_q_tail = prev_tail;
                        state.max_q_len -= 1;
                    } else {
                        break;
                    }
                }
                state.max_queue[state.max_q_tail] = (idx, val);
                state.max_q_tail += 1;
                if state.max_q_tail >= window_size {
                    state.max_q_tail -= window_size;
                }
                state.max_q_len += 1;
            }
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

        if compute.contains(Compute::ABS_MAX) {
            state.abs_max = state.abs_max.max(val.abs());
        }
    }

    #[inline(always)]
    pub fn finalize_simd(compute: Compute, state: &mut ColumnState) {
        if compute.contains(Compute::SUM) {
            state.total_sum += state.sum_vec.reduce_sum();
            state.sum_vec = f32x4::splat(0.0);
        }
        if compute.contains(Compute::MIN) {
            state.min_value = state.min_value.min(state.min_vec.reduce_min());
            state.min_vec = f32x4::splat(f32::INFINITY);
        }
        if compute.contains(Compute::MAX) {
            state.max_value = state.max_value.max(state.max_vec.reduce_max());
            state.max_vec = f32x4::splat(f32::NEG_INFINITY);
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

    pub fn finalize_base_metrics(state: &mut ColumnState, n: f32) -> BaseMetrics {
        let mean = state.total_sum / n;
        let m2 = state.energy - (state.total_sum * state.total_sum) / n;
        let m3 = state.sum_cubes - 3.0 * mean * state.energy + 2.0 * mean * mean * state.total_sum;
        let m4 = state.sum_quads - 4.0 * mean * state.sum_cubes + 6.0 * mean * mean * state.energy
            - 3.0 * mean * mean * mean * state.total_sum;

        let var = if n > 1.0 { m2 / (n - 1.0) } else { 0.0 };
        let std_dev = var.sqrt();

        BaseMetrics {
            mean,
            m2,
            m3,
            m4,
            var,
            std_dev,
        }
    }
}
