use crate::common::{ColumnState, LANES};
use crate::types::{Compute, Feature};
use realfft::RealToComplex;
use std::simd::f32x4;
use std::simd::num::SimdFloat;
use std::sync::Arc;

use super::processors::stats_processor::StatsProcessor;
use super::processors::diff_processor::DiffProcessor;
use super::processors::trend_processor::TrendProcessor;

pub(crate) struct StaticEngine<'a> {
    pub(crate) compute: Compute,
    pub(crate) features: &'a [Feature],
    pub(crate) unique_paa_totals: &'a [u16],
    pub(crate) unique_c3_lags: &'a [u16],
    pub(crate) paa_boundaries: &'a [Vec<usize>],
    pub(crate) r2c: Option<Arc<dyn RealToComplex<f32>>>,
    pub(crate) fft_size: usize,
}

impl<'a> StaticEngine<'a> {
    #[inline(always)]
    pub(crate) fn process_column(&self, values: &[f32]) -> Vec<f32> {
        let n = values.len() as f32;
        if n == 0.0 {
            return vec![0.0; self.features.len()];
        }

        let mut state =
            ColumnState::new(self.unique_paa_totals, self.unique_c3_lags, &[], values[0]);

        let rem_start = self.process_simd_chunks(values, &mut state);
        self.process_remainder(values, rem_start, &mut state);
        self.finalize_results(values, n, state)
    }

    #[inline(always)]
    fn process_simd_chunks(&self, values: &[f32], state: &mut ColumnState) -> usize {
        let chunks = values.chunks_exact(LANES);
        let rem_start = (values.len() / LANES) * LANES;

        for (chunk_idx, i) in chunks.enumerate() {
            let chunk = f32x4::from_slice(i);
            let global_idx = chunk_idx * LANES;
            let offset = global_idx as f32;
            let shifted = f32x4::from_array([state.prev_last, i[0], i[1], i[2]]);

            StatsProcessor::process_simd(self.compute, chunk, state);
            DiffProcessor::process_simd(self.compute, chunk, shifted, offset, state);
            TrendProcessor::process_simd(
                self.compute,
                chunk,
                i,
                offset,
                global_idx,
                values,
                state,
                self.unique_paa_totals,
                self.paa_boundaries,
                self.unique_c3_lags,
            );

            state.prev_last = i[LANES - 1];
        }
        rem_start
    }

    #[inline(always)]
    fn process_remainder(&self, values: &[f32], rem_start: usize, state: &mut ColumnState) {
        for i in rem_start..values.len() {
            let val = values[i];

            StatsProcessor::process_remainder(self.compute, val, state);
            
            if i > 0 {
                let prev = values[i - 1];
                DiffProcessor::process_remainder(self.compute, val, prev, i, state);
            }

            TrendProcessor::process_sequential(
                self.compute,
                val,
                i,
                values,
                state,
                self.unique_paa_totals,
                self.paa_boundaries,
                self.unique_c3_lags,
            );
        }
    }

    #[inline(always)]
    fn finalize_results(&self, values: &[f32], n: f32, mut state: ColumnState) -> Vec<f32> {
        let mean = state.total_sum / n;
        let mac_sum = state.mac_sum_vec.reduce_sum();
        let mc_sum = state.mc_sum_vec.reduce_sum();

        let m2 = state.energy - (state.total_sum * state.total_sum) / n;
        let m3 = state.sum_cubes - 3.0 * mean * state.energy + 2.0 * mean * mean * state.total_sum;
        let m4 = state.sum_quads - 4.0 * mean * state.sum_cubes + 6.0 * mean * mean * state.energy
            - 3.0 * mean * mean * mean * state.total_sum;

        let mut mad_sum = 0.0;
        let mut count_a = 0;
        let mut count_b = 0;
        let mut max_strike_a = 0;
        let mut max_strike_b = 0;
        let mut median = 0.0;
        let mut iqr = 0.0;
        let mut entropy = 0.0;
        let mut zc_mean = 0.0;
        let mut zc_std = 0.0;
        let mut first_max_idx = 0;
        let mut last_max_idx = 0;
        let mut first_min_idx = 0;
        let mut last_min_idx = 0;

        let mut sorted_copy = crate::static_ext::features::distribution::compute_distribution_features(
            &self.compute,
            values,
            &mut state,
            n,
            &mut median,
            &mut iqr,
            &mut entropy,
        );

        let benford_corr = crate::static_ext::features::benford::compute_benford_correlation(&self.compute, values);

        let mut spectrum = Vec::new();
        let mut fft_complex = Vec::new();
        let mut freq_centroid = 0.0;
        let mut spectral_decrease = 0.0;
        let mut spectral_slope = 0.0;

        crate::static_ext::features::spectral::compute_spectral_features(
            &self.compute,
            values,
            &mut state,
            self.fft_size,
            &self.r2c,
            &mut spectrum,
            &mut fft_complex,
            &mut freq_centroid,
            &mut spectral_decrease,
            &mut spectral_slope,
        );

        let mut signal_dist = 0.0;
        if self.compute.contains(Compute::SIG_DISTANCE) {
            for i in 1..values.len() {
                signal_dist += ((values[i] - values[i - 1]).powi(2) + 1.0).sqrt();
            }
        }

        let fft_autocorr = crate::static_ext::features::autocorr::compute_fft_autocorr(&self.compute, values, n, mean, m2);

        if self.compute.contains(Compute::NEEDS_SORT) {
            crate::static_ext::features::extrema::compute_extrema_features(
                &self.compute,
                values,
                state.max_value,
                state.min_value,
                &mut first_max_idx,
                &mut last_max_idx,
                &mut first_min_idx,
                &mut last_min_idx,
            );

            crate::static_ext::features::strikes::compute_strike_features(
                &self.compute,
                values,
                mean,
                &mut mad_sum,
                &mut count_a,
                &mut count_b,
                &mut max_strike_a,
                &mut max_strike_b,
            );

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
        }

        let var = if n > 1.0 { m2 / (n - 1.0) } else { 0.0 };
        let std_dev = var.sqrt();

        self.features
            .iter()
            .map(|feat| {
                if let Some(val) = crate::static_ext::features::eval_basic::eval_basic(
                    feat, n, mean, var, std_dev, &state,
                ) {
                    return val;
                }
                if let Some(val) = crate::static_ext::features::eval_statistics::eval_statistics(
                    feat,
                    n,
                    mean,
                    var,
                    std_dev,
                    m2,
                    m3,
                    m4,
                    mad_sum,
                    median,
                    iqr,
                    entropy,
                    count_a,
                    count_b,
                    max_strike_a,
                    max_strike_b,
                    zc_mean,
                    zc_std,
                    &mut state,
                    values,
                    &mut sorted_copy,
                    benford_corr,
                    first_max_idx,
                    last_max_idx,
                    first_min_idx,
                    last_min_idx,
                ) {
                    return val;
                }
                if let Some(val) = crate::static_ext::features::eval_time_series::eval_time_series(
                    feat,
                    n,
                    mean,
                    var,
                    m2,
                    mac_sum,
                    mc_sum,
                    &mut state,
                    &self.unique_c3_lags,
                    &self.unique_paa_totals,
                    &self.paa_boundaries,
                    &fft_autocorr,
                    values,
                    signal_dist,
                ) {
                    return val;
                }
                if let Some(val) = crate::static_ext::features::eval_fft::eval_fft(
                    feat,
                    n,
                    &spectrum,
                    &fft_complex,
                    freq_centroid,
                    spectral_decrease,
                    spectral_slope,
                    values,
                ) {
                    return val;
                }
                0.0
            })
            .collect()
    }
}
