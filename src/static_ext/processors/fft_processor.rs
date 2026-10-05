use std::sync::Arc;

use realfft::RealToComplex;

use crate::common::ColumnState;
use crate::metrics::FftResult;
use crate::types::Compute;

pub struct FftProcessor;

impl FftProcessor {
    /// Static windows: one exact-length FFT per column.
    pub fn finalize(
        compute: Compute,
        values: &[f32],
        n: f32,
        mean: f32,
        m2: f32,
        r2c: &Option<Arc<dyn RealToComplex<f32>>>,
        state: &mut ColumnState,
    ) -> Result<FftResult, String> {
        let fft_complex = match r2c {
            Some(r2c) if compute.intersects(Compute::ANY_FFT) => {
                crate::spectral::rfft(values, r2c.as_ref(), state)?
            }
            _ => Vec::new(),
        };
        crate::spectral::finalize(compute, values, n, mean, m2, fft_complex, values.len(), state)
    }
}
