use std::sync::Arc;

use realfft::RealToComplex;

use crate::common::{ColumnState, SlidingDFT};
use crate::metrics::FftResult;
use crate::types::Compute;

pub struct FftProcessor;

impl FftProcessor {
    /// Sliding windows: the DFT is updated in O(window) per step by the sliding
    /// DFT (see `SlidingDFT::update` in sliding.rs) instead of an O(n log n)
    /// FFT. It is rebuilt from a fresh FFT on the first window and after every
    /// `window` updates, which bounds the f32 drift of repeated twiddle products.
    pub fn finalize(
        compute: Compute,
        values: &[f32],
        n: f32,
        mean: f32,
        m2: f32,
        r2c: &Option<Arc<dyn RealToComplex<f32>>>,
        state: &mut ColumnState,
    ) -> Result<FftResult, String> {
        let mut fft_complex = Vec::new();
        if let Some(r2c) = r2c
            .as_ref()
            .filter(|_| compute.intersects(Compute::ANY_FFT))
        {
            let stale = state
                .sliding_dft
                .as_ref()
                .is_none_or(|s| s.updates >= values.len());
            if stale {
                let bins = crate::spectral::rfft(values, r2c.as_ref(), state)?;
                state.sliding_dft = Some(SlidingDFT::from_fft(bins, values.len()));
            }
            if let Some(sdft) = &state.sliding_dft {
                fft_complex = sdft.bins.clone();
            }
        }
        crate::spectral::finalize(
            compute,
            values,
            n,
            mean,
            m2,
            fft_complex,
            values.len(),
            state,
        )
    }
}
