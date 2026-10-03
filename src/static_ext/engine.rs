use crate::common::{ColumnState, LANES};
use crate::types::{Compute, Feature};
use realfft::RealToComplex;
use std::simd::f32x4;
use std::sync::Arc;

use super::processors::stats_processor::StatsProcessor;
use super::processors::diff_processor::DiffProcessor;
use super::processors::trend_processor::TrendProcessor;
use super::processors::sort_processor::SortProcessor;
use super::processors::mean_processor::MeanProcessor;
use super::processors::fft_processor::FftProcessor;

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
        StatsProcessor::finalize_simd(self.compute, &mut state);
        DiffProcessor::finalize_simd(self.compute, &mut state);
        TrendProcessor::finalize_simd(self.compute, &mut state);

        let base_metrics = StatsProcessor::finalize_base_metrics(&mut state, n);

        let sort_metrics = SortProcessor::finalize(self.compute, values, n, &mut state);
        let mean_metrics = MeanProcessor::finalize(self.compute, values, n, base_metrics.mean);
        let fft_res = FftProcessor::finalize(
            self.compute,
            values,
            n,
            base_metrics.mean,
            base_metrics.m2,
            self.fft_size,
            &self.r2c,
            &mut state,
        ).unwrap_or(crate::metrics::FftResult {
            spectrum: Vec::new(),
            fft_complex: Vec::new(),
            freq_centroid: 0.0,
            spectral_decrease: 0.0,
            spectral_slope: 0.0,
            fft_autocorr: Vec::new(),
        });

        let zc_metrics = DiffProcessor::finalize(self.compute, &state);

        let mut context = crate::context::FeatureContext::new(
            values,
            None,
            &mut state,
            n,
            base_metrics,
            sort_metrics,
            mean_metrics,
            zc_metrics,
            &fft_res,
            &self.unique_c3_lags,
            &self.unique_paa_totals,
            &self.paa_boundaries,
        );

        let mut feats = Vec::with_capacity(self.features.len());
        for feat in self.features {
            let val = {
                if let Some(v) = crate::features::moments::eval_moments(feat, &mut context) { v }
                else if let Some(v) = crate::features::min_max::eval_min_max(feat, &mut context) { v }
                else if let Some(v) = crate::features::distribution::eval_distribution(feat, &mut context) { v }
                else if let Some(v) = crate::features::energy::eval_energy(feat, &mut context) { v }
                else if let Some(v) = crate::features::crossings_peaks::eval_crossings_peaks(feat, &mut context) { v }
                else if let Some(v) = crate::features::autocorrelation::eval_autocorrelation(feat, &mut context) { v }
                else if let Some(v) = crate::features::changes::eval_changes(feat, &mut context) { v }
                else if let Some(v) = crate::features::runs::eval_runs(feat, &mut context) { v }
                else if let Some(v) = crate::features::transform::eval_transform(feat, &mut context) { v }
                else if let Some(v) = crate::features::complexity::eval_complexity(feat, &mut context) { v }
                else if let Some(v) = crate::features::misc::eval_misc(feat, &mut context) { v }
                else { 0.0 }
            };
            feats.push(val);
        }
        feats
    }
}
