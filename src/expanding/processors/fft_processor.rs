use std::sync::Arc;

use realfft::RealToComplex;

use crate::common::ColumnState;
use crate::metrics::FftResult;
use crate::types::Compute;

pub struct FftProcessor;

impl FftProcessor {
    /// Expanding windows: a DFT can't be extended to a longer length, so this
    /// is a fresh exact-length FFT, recomputed only once `fft_update_period`
    /// new values have arrived (period 1 = every update, exact).
    pub fn finalize(
        compute: Compute,
        full_series: &[f32],
        mean: f32,
        m2: f32,
        r2c: &Option<Arc<dyn RealToComplex<f32>>>,
        fft_update_period: usize,
        state: &mut ColumnState,
    ) -> Result<FftResult, String> {
        let n_total = full_series.len();
        let mut fft_complex = Vec::new();
        if let Some(r2c) = r2c
            .as_ref()
            .filter(|_| compute.intersects(Compute::ANY_FFT))
        {
            let should_update =
                state.last_fft_n == 0 || n_total - state.last_fft_n >= fft_update_period;
            if should_update {
                state.last_fft_complex = crate::spectral::rfft(full_series, r2c.as_ref(), state)?;
                state.last_fft_n = n_total;
            }
            fft_complex = state.last_fft_complex.clone();
        }
        let dft_len = state.last_fft_n;
        crate::spectral::finalize(
            compute,
            full_series,
            n_total as f32,
            mean,
            m2,
            fft_complex,
            dft_len,
            state,
        )
    }
}
