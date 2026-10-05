use crate::common::{ColumnState, LANES};
use crate::types::{Compute, Feature};
use realfft::RealToComplex;
use std::simd::f32x4;
use std::sync::Arc;

use super::processors::diff_processor::DiffProcessor;
use super::processors::fft_processor::FftProcessor;
use super::processors::mean_processor::MeanProcessor;
use super::processors::sort_processor::SortProcessor;
use super::processors::stats_processor::StatsProcessor;
use super::processors::trend_processor::TrendProcessor;

pub(crate) struct StaticEngine<'a> {
    pub(crate) compute: Compute,
    pub(crate) features: &'a [Feature],
    pub(crate) unique_paa_totals: &'a [u16],
    pub(crate) unique_c3_lags: &'a [u16],
    pub(crate) unique_tra_lags: &'a [u16],
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

        let mut state = ColumnState::new(
            self.unique_paa_totals,
            self.unique_c3_lags,
            &[],
            self.unique_tra_lags,
            values[0],
        );

        let rem_start = self.process_simd_chunks(values, &mut state);
        self.process_remainder(values, rem_start, &mut state);

        if self
            .compute
            .intersects(crate::types::Compute::REOCCUR_RATIOS)
        {
            for &val in values {
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
        self.finalize_results(values, n, state)
    }

    #[inline(always)]
    fn process_simd_chunks(&self, values: &[f32], state: &mut ColumnState) -> usize {
        let chunks = values.as_chunks::<LANES>().0;
        let rem_start = (values.len() / LANES) * LANES;

        for (chunk_idx, i) in chunks.iter().enumerate() {
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
        let mut fft_res = FftProcessor::finalize(
            self.compute,
            values,
            n,
            base_metrics.mean,
            base_metrics.m2,
            self.fft_size,
            &self.r2c,
            &mut state,
        )
        .unwrap_or(crate::metrics::FftResult {
            spectrum: Vec::new(),
            fft_complex: Vec::new(),
            freq_centroid: 0.0,
            spectral_decrease: 0.0,
            spectral_slope: 0.0,
            spectral_spread: 0.0,
            spectral_entropy: 0.0,
            spectral_roll_on: 0.0,
            spectral_roll_off: 0.0,
            spectral_skewness: 0.0,
            spectral_kurtosis: 0.0,
            fft_autocorr: Vec::new(),
            mfcc: Vec::new(),
            lpcc: Vec::new(),
            cwt_energy: Vec::new(),
            cwt_entropy: 0.0,
            cwt_peaks: 0,
            welch_density: Vec::new(),
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
            self.unique_c3_lags,
            self.unique_tra_lags,
            self.unique_paa_totals,
            self.paa_boundaries,
        );

        let mut feats = Vec::with_capacity(self.features.len());
        for feat in self.features {
            let val = crate::features::eval(feat, &mut context);
            feats.push(val);
        }

        if self.compute.intersects(Compute::ANY_FFT) {
            state.spectrum_buffer = std::mem::take(&mut fft_res.spectrum);
        }

        feats
    }
}
