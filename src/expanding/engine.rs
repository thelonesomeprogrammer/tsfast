use crate::common::ColumnState;
use crate::common::LANES;
use crate::types::{Compute, Feature};
use realfft::RealToComplex;
use std::simd::f32x4;
use std::simd::num::SimdFloat;
use std::sync::Arc;

use super::processors::stats_processor::StatsProcessor;
use super::processors::diff_processor::DiffProcessor;
use super::processors::trend_processor::TrendProcessor;
use super::processors::sort_processor::SortProcessor;
use super::processors::mean_processor::MeanProcessor;
use super::processors::fft_processor::FftProcessor;

pub(crate) struct ExpandingEngine<'a> {
    pub(crate) compute: Compute,
    pub(crate) features: &'a [Feature],
    pub(crate) unique_paa_totals: &'a [u16],
    pub(crate) unique_c3_lags: &'a [u16],
    pub(crate) unique_autocorr_lags: &'a [u16],
    pub(crate) paa_boundaries: &'a [Vec<usize>],
    pub(crate) r2c: Option<Arc<dyn RealToComplex<f32>>>,
    pub(crate) fft_size: usize,
    pub(crate) fft_update_period: usize,
}

impl<'a> ExpandingEngine<'a> {
    #[inline(always)]
    pub(crate) fn process_expanding(
        &mut self,
        values: &[f32],
        global_idx: usize,
        state: &mut ColumnState,
        full_series: &mut Vec<f32>,
        running_sorted: &mut Vec<f32>,
    ) -> Vec<f32> {
        let rem_start = self.process_simd_chunks(values, global_idx, state, full_series);
        self.process_remainder(values, rem_start, global_idx, state, full_series);

        full_series.extend_from_slice(values);

        if state.prefix_sums.is_empty() && !full_series.is_empty() {
            let mut s = 0.0;
            state.prefix_sums.reserve(full_series.len());
            for &v in full_series.iter() {
                s += v;
                state.prefix_sums.push(s);
            }
        } else {
            let mut s = state.prefix_sums.last().copied().unwrap_or(0.0);
            for &v in values {
                s += v;
                state.prefix_sums.push(s);
            }
        }

        let total_n = full_series.len() as f32;
        self.finalize_results(total_n, state, running_sorted, full_series)
    }

    #[inline(always)]
    fn process_simd_chunks(
        &self,
        values: &[f32],
        global_start_idx: usize,
        state: &mut ColumnState,
        full_series: &[f32],
    ) -> usize {
        let chunks = values.chunks_exact(LANES);
        let rem_start = (values.len() / LANES) * LANES;

        for (c_idx, i) in chunks.enumerate() {
            let offset_usize = global_start_idx + c_idx * LANES;
            let offset = offset_usize as f32;
            let chunk = f32x4::from_slice(i);
            let shifted = f32x4::from_array([state.prev_last, i[0], i[1], i[2]]);

            StatsProcessor::process_simd(self.compute, chunk, i, offset_usize, state);
            DiffProcessor::process_simd(self.compute, chunk, shifted, offset, state);
            TrendProcessor::process_simd(self.compute, chunk, offset, state);

            state.prev_last = i[LANES - 1];
        }

        for (i, &val) in values[..rem_start].iter().enumerate() {
            let global_idx = global_start_idx + i;
            TrendProcessor::process_sequential(
                self.compute, val, global_idx, global_start_idx, state, full_series, values,
                self.unique_c3_lags, self.unique_autocorr_lags
            );
        }
        rem_start
    }

    #[inline(always)]
    fn process_remainder(
        &self,
        values: &[f32],
        rem_start: usize,
        global_start_idx: usize,
        state: &mut ColumnState,
        full_series: &[f32],
    ) {
        for i in rem_start..values.len() {
            let val = values[i];
            let global_idx = global_start_idx + i;

            StatsProcessor::process_remainder(self.compute, val, global_idx, state);

            if global_idx > 0 {
                let prev = if i > 0 {
                    values[i - 1]
                } else {
                    state.prev_last
                };
                DiffProcessor::process_remainder(self.compute, val, prev, global_idx, state);
            }

            TrendProcessor::process_sequential(
                self.compute, val, global_idx, global_start_idx, state, full_series, values,
                self.unique_c3_lags, self.unique_autocorr_lags
            );

            state.prev_last = val;
        }
    }

    #[inline(always)]
    fn finalize_results(
        &self,
        n: f32,
        state: &mut ColumnState,
        running_sorted: &mut Vec<f32>,
        full_series: &[f32],
    ) -> Vec<f32> {
        let mean = state.total_sum / n;
        let mac_sum = state.mac_sum_vec.reduce_sum();
        let mc_sum = state.mc_sum_vec.reduce_sum();

        let m2 = state.energy - (state.total_sum * state.total_sum) / n;
        let m3 = state.sum_cubes - 3.0 * mean * state.energy + 2.0 * mean * mean * state.total_sum;
        let m4 = state.sum_quads - 4.0 * mean * state.sum_cubes + 6.0 * mean * mean * state.energy
            - 3.0 * mean * mean * mean * state.total_sum;

        let (median, iqr) = SortProcessor::process_running_sorted(self.compute, full_series, running_sorted);

        let fft_res = FftProcessor::finalize(
            self.compute,
            full_series,
            &self.r2c,
            self.fft_size,
            self.fft_update_period,
            state,
        );

        let mean_res = MeanProcessor::finalize(
            self.compute,
            full_series,
            n,
            mean,
            state,
        );

        let mut zc_mean = 0.0;
        let mut zc_std = 0.0;
        if self.compute.contains(Compute::ZC_STATS) && !state.zc_indices.is_empty() {
            zc_mean = state.zc_indices.iter().sum::<f32>() / state.zc_indices.len() as f32;
            if self.compute.contains(Compute::ZC_STD) {
                let zc_m2 = state
                    .zc_indices
                    .iter()
                    .map(|&idx| (idx - zc_mean).powi(2))
                    .sum::<f32>();
                zc_std = (zc_m2 / state.zc_indices.len() as f32).sqrt();
            }
        }

        let var = if n > 1.0 { m2 / (n - 1.0) } else { 0.0 };
        let std_dev = var.sqrt();

        self.features
            .iter()
            .map(|feat| {
                if let Some(val) = crate::expanding::basic::eval_basic(
                    feat, n, mean, var, std_dev, state,
                ) {
                    return val;
                }
                if let Some(val) = crate::expanding::statistics::eval_statistics(
                    feat,
                    n,
                    mean,
                    var,
                    std_dev,
                    m2,
                    m3,
                    m4,
                    mean_res.mad_sum,
                    median,
                    iqr,
                    mean_res.entropy,
                    mean_res.count_a,
                    mean_res.count_b,
                    mean_res.max_strike_a,
                    mean_res.max_strike_b,
                    zc_mean,
                    zc_std,
                    state,
                    running_sorted,
                    full_series,
                ) {
                    return val;
                }
                if let Some(val) = crate::expanding::time_series::eval_time_series(
                    feat,
                    n,
                    mean,
                    var,
                    m2,
                    mac_sum,
                    mc_sum,
                    state,
                    &self.unique_c3_lags,
                    &self.unique_paa_totals,
                    &self.paa_boundaries,
                    &self.unique_autocorr_lags,
                    full_series,
                ) {
                    return val;
                }
                if let Some(val) = crate::expanding::fft::eval_fft(
                    feat,
                    n,
                    &fft_res.spectrum,
                    &fft_res.fft_complex,
                    fft_res.freq_centroid,
                    fft_res.spectral_decrease,
                    fft_res.spectral_slope,
                    full_series,
                ) {
                    return val;
                }
                0.0
            })
            .collect()
    }
}
