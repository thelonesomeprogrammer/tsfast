use crate::common::ColumnState;
use crate::types::Compute;
use std::simd::f32x4;
use std::simd::num::SimdFloat;

pub struct TrendProcessor;

impl TrendProcessor {
    #[inline(always)]
    pub fn process_simd(
        compute: Compute,
        chunk: f32x4,
        offset: f32,
        state: &mut ColumnState,
    ) {
        if compute.contains(Compute::SLOPE) {
            let indices = f32x4::from_array([offset, offset + 1.0, offset + 2.0, offset + 3.0]);
            state.sum_ix_vec += indices * chunk;
        }
        if compute.contains(Compute::CALC_CENTROID) {
            let indices = f32x4::from_array([offset, offset + 1.0, offset + 2.0, offset + 3.0]);
            state.t_energy_vec += indices * (chunk * chunk);
        }
    }

    #[inline(always)]
    pub fn finalize_simd(compute: Compute, state: &mut ColumnState) {
        if compute.contains(Compute::SLOPE) {
            state.sum_ix += state.sum_ix_vec.reduce_sum();
            state.sum_ix_vec = f32x4::splat(0.0);
        }
        if compute.contains(Compute::CALC_CENTROID) {
            state.t_energy += state.t_energy_vec.reduce_sum();
            state.t_energy_vec = f32x4::splat(0.0);
        }
    }

    #[inline(always)]
    pub fn process_sequential(
        compute: Compute,
        val: f32,
        global_idx: usize,
        global_start_idx: usize,
        state: &mut ColumnState,
        full_series: &[f32],
        values: &[f32],
        unique_c3_lags: &[u16],
        unique_autocorr_lags: &[u16],
        unique_tra_lags: &[u16],
    ) {
        if compute.contains(Compute::PEAKS) {
            if state.prev_val > state.prev_prev_val && state.prev_val > val {
                state.peaks += 1;
            }
            state.prev_prev_val = state.prev_val;
            state.prev_val = val;
        }

        if compute.contains(Compute::C3) {
            for (l_idx, &lag) in unique_c3_lags.iter().enumerate() {
                let l = lag as usize;
                if global_idx >= 2 * l {
                    let v_l = if global_idx - l < global_start_idx {
                        full_series[global_idx - l]
                    } else {
                        values[global_idx - l - global_start_idx]
                    };
                    let v_2l = if global_idx - 2 * l < global_start_idx {
                        full_series[global_idx - 2 * l]
                    } else {
                        values[global_idx - 2 * l - global_start_idx]
                    };
                    state.c3_sums[l_idx] += val * v_l * v_2l;
                }
            }
        }

        if compute.contains(Compute::TRA) {
            for (l_idx, &lag) in unique_tra_lags.iter().enumerate() {
                let l = lag as usize;
                if global_idx >= 2 * l {
                    let v_l = if global_idx - l < global_start_idx {
                        full_series[global_idx - l]
                    } else {
                        values[global_idx - l - global_start_idx]
                    };
                    let v_2l = if global_idx - 2 * l < global_start_idx {
                        full_series[global_idx - 2 * l]
                    } else {
                        values[global_idx - 2 * l - global_start_idx]
                    };
                    state.tra_sums[l_idx] += val * val * v_l - v_l * v_2l * v_2l;
                }
            }
        }

        if compute.intersects(Compute::FULL_AUTOCORR | Compute::PACF) {
            for (l_idx, &lag) in unique_autocorr_lags.iter().enumerate() {
                let l = lag as usize;
                if global_idx >= l {
                    let v_l = if global_idx - l < global_start_idx {
                        full_series[global_idx - l]
                    } else {
                        values[global_idx - l - global_start_idx]
                    };
                    state.autocorr_sums[l_idx] += val * v_l;
                }
            }
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
        if compute.contains(Compute::CALC_CENTROID) {
            state.t_energy += (global_idx as f32) * val * val;
        }
    }

    #[inline(always)]
    pub fn finalize_paa(
        compute: Compute,
        unique_paa_totals: &[u16],
        paa_boundaries: &[Vec<usize>],
        full_series: &[f32],
        state: &mut ColumnState,
    ) {
        if compute.contains(Compute::PAA) {
            for (t_idx, _total) in unique_paa_totals.iter().enumerate() {
                let b = &paa_boundaries[t_idx];
                for seg_idx in 0..state.paa_sums[t_idx].len() {
                    let start = b[seg_idx];
                    let end = b[seg_idx + 1];
                    let mut sum = 0.0;
                    if start < end {
                        for v in &full_series[start..end] {
                            sum += v;
                        }
                    }
                    state.paa_sums[t_idx][seg_idx] = sum;
                }
            }
        }
    }
}
