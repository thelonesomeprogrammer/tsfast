use crate::common::ColumnState;
use crate::types::{Compute, Feature};
use std::sync::Arc;

use super::processors::stats_processor::StatsProcessor;
use super::processors::diff_processor::DiffProcessor;
use super::processors::complexity_processor::ComplexityProcessor;
use super::processors::trend_processor::TrendProcessor;
use super::processors::fft_processor::FftProcessor;
use super::processors::queue_processor::QueueProcessor;
use super::processors::sort_processor::SortProcessor;
use super::processors::mean_processor::MeanProcessor;

pub(crate) struct SlidingEngine<'a> {
    pub(crate) compute: Compute,
    pub(crate) features: &'a [Feature],
    pub(crate) unique_paa_totals: &'a [u16],
    pub(crate) unique_c3_lags: &'a [u16],
    pub(crate) paa_boundaries: &'a [Vec<usize>],
    pub(crate) r2c: Option<Arc<dyn realfft::RealToComplex<f32>>>,
    pub(crate) fft_size: usize,
}

impl<'a> SlidingEngine<'a> {
    #[inline(always)]
    pub(crate) fn update_batch(
        &self,
        old_slice: &[f32],
        new_slice: &[f32],
        value_before_old: Option<f32>,
        value_after_old: f32,
        value_before_new: f32,
        global_start_idx: usize,
        window_size: usize,
        state: &mut ColumnState,
    ) {
        let y = old_slice.len();
        if y == 0 {
            return;
        }

        StatsProcessor::update_batch(self.compute, old_slice, new_slice, state);
        DiffProcessor::update_batch(self.compute, old_slice, new_slice, value_before_old, value_after_old, value_before_new, state);

        TrendProcessor::update_batch(self.compute, old_slice, new_slice, window_size, state);


        if self.compute.intersects(crate::types::Compute::REOCCUR_RATIOS) {
            for &val in new_slice {
                let bits = val.to_bits();
                let count = state.value_counts.entry(bits).or_insert(0);
                *count += 1;
                if *count == 2 {
                    state.reoccurring_values += 1;
                    state.reoccurring_datapoints += 2;
                } else if *count > 2 {
                    state.reoccurring_datapoints += 1;
                }
            }

            for &val in old_slice {
                let old_bits = val.to_bits();
                let count = state.value_counts.entry(old_bits).or_insert(0);
                if *count == 2 {
                    state.reoccurring_values -= 1;
                    state.reoccurring_datapoints -= 2;
                } else if *count > 2 {
                    state.reoccurring_datapoints -= 1;
                }
                *count -= 1;
                if *count == 0 {
                    state.value_counts.remove(&old_bits);
                }
            }
        }
QueueProcessor::update_batch(self.compute, new_slice, global_start_idx, window_size, state);
    }

    #[inline(always)]
    pub(crate) fn update_incremental(
        &self,
        old_val: f32,
        new_val: f32,
        old_val_next: f32,
        old_last: f32,
        global_idx: usize,
        window_size: usize,
        state: &mut ColumnState,
    ) {
        StatsProcessor::update_incremental(self.compute, old_val, new_val, state);
        DiffProcessor::update_incremental(self.compute, old_val, new_val, old_val_next, old_last, state);

        TrendProcessor::update_incremental(self.compute, new_val, window_size, state);


        if self.compute.intersects(crate::types::Compute::REOCCUR_RATIOS) {
            let bits = new_val.to_bits();
            let count = state.value_counts.entry(bits).or_insert(0);
            *count += 1;
            if *count == 2 {
                state.reoccurring_values += 1;
                state.reoccurring_datapoints += 2;
            } else if *count > 2 {
                state.reoccurring_datapoints += 1;
            }

            let old_bits = old_val.to_bits();
            let count = state.value_counts.entry(old_bits).or_insert(0);
            if *count == 2 {
                state.reoccurring_values -= 1;
                state.reoccurring_datapoints -= 2;
            } else if *count > 2 {
                state.reoccurring_datapoints -= 1;
            }
            *count -= 1;
            if *count == 0 {
                state.value_counts.remove(&old_bits);
            }
        }
QueueProcessor::update_incremental(self.compute, new_val, global_idx, window_size, state);
    }

    #[inline(always)]
    pub(crate) fn process_column(
        &self,
        values: &[f32],
        state: &mut ColumnState,
        is_incremental: bool,
    ) -> Result<Vec<f32>, String> {
        let n = values.len() as f32;
        if n == 0.0 {
            return Ok(vec![0.0; self.features.len()]);
        }

        StatsProcessor::reset_state(state, is_incremental);
        DiffProcessor::reset_state(state, values, is_incremental);
        ComplexityProcessor::reset_state(state);
        QueueProcessor::reset_state(state, values.len());

        if !is_incremental && self.compute.intersects(crate::types::Compute::REOCCUR_RATIOS) {
            state.value_counts.clear();
            state.reoccurring_datapoints = 0;
            state.reoccurring_values = 0;
        }

        // SIMD Pass 1
        let rem_start = self.process_simd_chunks(values, state, is_incremental);

        // Remainder Pass 1
        self.process_remainder(values, rem_start, state, is_incremental);

        // Post-processing and Pass 2
        self.finalize_results(values, n, state)
    }

    #[inline(always)]
    fn process_simd_chunks(
        &self,
        values: &[f32],
        state: &mut ColumnState,
        is_incremental: bool,
    ) -> usize {
        let chunks = values.chunks_exact(crate::common::LANES);
        let mut i = 0;
        for chunk_slice in chunks {
            let chunk = std::simd::f32x4::from_slice(chunk_slice);
            let global_idx = i;

            if !is_incremental {
                StatsProcessor::process_simd(self.compute, chunk, global_idx, values.len(), state);
                TrendProcessor::process_simd(self.compute, chunk, global_idx, state);

                if self.compute.intersects(crate::types::Compute::REOCCUR_RATIOS) {
                    let arr = chunk.to_array();
                    for &val in &arr {
                        let bits = val.to_bits();
                        let count = state.value_counts.entry(bits).or_insert(0);
                        *count += 1;
                        if *count == 2 {
                            state.reoccurring_values += 1;
                            state.reoccurring_datapoints += 2;
                        } else if *count > 2 {
                            state.reoccurring_datapoints += 1;
                        }
                    }
                }
            }

            DiffProcessor::process_simd(self.compute, chunk, values, global_idx, is_incremental, state);

            ComplexityProcessor::process_simd(self.compute, chunk, global_idx, values, self.unique_paa_totals, self.paa_boundaries, self.unique_c3_lags, state);

            i += crate::common::LANES;
        }

        i
    }

    #[inline(always)]
    fn process_remainder(
        &self,
        values: &[f32],
        rem_start: usize,
        state: &mut ColumnState,
        is_incremental: bool,
    ) {
        let window_size = values.len();

        for i in rem_start..values.len() {
            let val = values[i];

            if !is_incremental {
                StatsProcessor::process_remainder(self.compute, val, state);
                TrendProcessor::process_remainder(self.compute, val, i, state);
            }

            if !is_incremental && self.compute.intersects(crate::types::Compute::REOCCUR_RATIOS) {
                let bits = val.to_bits();
                let count = state.value_counts.entry(bits).or_insert(0);
                *count += 1;
                if *count == 2 {
                    state.reoccurring_values += 1;
                    state.reoccurring_datapoints += 2;
                } else if *count > 2 {
                    state.reoccurring_datapoints += 1;
                }
            }
QueueProcessor::process_remainder(self.compute, val, i, window_size, state);

            if i > 0 {
                let prev = values[i - 1];
                DiffProcessor::process_remainder(self.compute, val, prev, i as f32, is_incremental, state);
            } else {
                DiffProcessor::process_remainder(self.compute, val, state.prev_last, i as f32, is_incremental, state);
            }
            state.prev_last = val;

            ComplexityProcessor::process_remainder(self.compute, val, i, values, self.unique_paa_totals, self.paa_boundaries, self.unique_c3_lags, state);
        }
    }

    #[inline(always)]
    pub(crate) fn finalize_results(
        &self,
        values: &[f32],
        n: f32,
        state: &mut ColumnState,
    ) -> Result<Vec<f32>, String> {
        StatsProcessor::finalize_simd(self.compute, state);
        DiffProcessor::finalize_simd(self.compute, state);
        TrendProcessor::finalize_simd(self.compute, state);
        ComplexityProcessor::finalize_simd(self.compute, state);
        let base_metrics = StatsProcessor::finalize_base_metrics(state, n);
        let fft_res = FftProcessor::finalize(
            self.compute,
            values,
            n,
            base_metrics.mean,
            base_metrics.m2,
            self.fft_size,
            &self.r2c,
            state,
        )?;

        let sort_metrics = SortProcessor::finalize(self.compute, values, n, state);
        let mean_metrics = MeanProcessor::finalize(self.compute, values, n, base_metrics.mean);
        let zc_metrics = DiffProcessor::finalize(self.compute, state);

        let mut context = crate::context::FeatureContext::new(
            values,
            None,
            state,
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
        Ok(feats)
    }
}
