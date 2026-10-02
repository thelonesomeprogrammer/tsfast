use crate::common::ColumnState;
use realfft::{RealFftPlanner, RealToComplex};
use std::sync::Arc;

#[inline(always)]
pub fn compute_spectral_features(
    compute: &crate::types::Compute,
    values: &[f32],
    state: &mut ColumnState,
    fft_size: usize,
    r2c_opt: &Option<Arc<dyn RealToComplex<f32>>>,
    spectrum: &mut Vec<f32>,
    fft_complex: &mut Vec<realfft::num_complex::Complex<f32>>,
    freq_centroid: &mut f32,
    spectral_decrease: &mut f32,
    spectral_slope: &mut f32,
) {
    if compute.intersects(crate::types::Compute::ANY_FFT) {
        let mut indata = std::mem::take(&mut state.fft_in_buffer);
        let mut outdata = std::mem::take(&mut state.fft_out_buffer);

        let (complex_data, spec) = if let Some(r2c) = r2c_opt {
            if indata.len() < fft_size {
                indata.resize(fft_size, 0.0);
            }
            indata.fill(0.0);
            indata[..values.len()].copy_from_slice(values);

            let out_len = r2c.complex_len();
            if outdata.len() < out_len {
                outdata.resize(out_len, realfft::num_complex::Complex::new(0.0, 0.0));
            }

            r2c.process(&mut indata, &mut outdata[..out_len]).unwrap();
            let s = outdata[..out_len].iter().map(|c| c.norm()).collect();
            (outdata[..out_len].to_vec(), s)
        } else if compute.intersects(crate::types::Compute::SPEC_CENTROID | crate::types::Compute::SPEC_DISTANCE | crate::types::Compute::SPEC_DECREASE | crate::types::Compute::SPEC_SLOPE | crate::types::Compute::SPECTROGRAM) {
            let mut planner = RealFftPlanner::<f32>::new();
            let r2c = planner.plan_fft_forward(values.len());

            if indata.len() < values.len() {
                indata.resize(values.len(), 0.0);
            }
            indata[..values.len()].copy_from_slice(values);

            let out_len = r2c.complex_len();
            if outdata.len() < out_len {
                outdata.resize(out_len, realfft::num_complex::Complex::new(0.0, 0.0));
            }

            r2c.process(&mut indata[..values.len()], &mut outdata[..out_len])
                .unwrap();
            let s = outdata[..out_len].iter().map(|c| c.norm()).collect();
            (outdata[..out_len].to_vec(), s)
        } else {
            (Vec::new(), Vec::new())
        };

        state.fft_in_buffer = indata;
        state.fft_out_buffer = outdata;

        *fft_complex = complex_data;
        *spectrum = spec;

        if !spectrum.is_empty() {
            let spec_sum: f32 = spectrum.iter().sum();
            if spec_sum > 0.0 {
                *freq_centroid = spectrum
                    .iter()
                    .enumerate()
                    .map(|(i, &mag)| i as f32 * mag)
                    .sum::<f32>()
                    / spec_sum;

                if spectrum.len() > 1 {
                    let spec_sum_no_first: f32 = spectrum[1..].iter().sum();
                    if spec_sum_no_first > 0.0 {
                        *spectral_decrease = spectrum[1..]
                            .iter()
                            .enumerate()
                            .map(|(i, &mag)| (mag - spectrum[0]) / (i + 1) as f32)
                            .sum::<f32>()
                            / spec_sum_no_first;
                    }

                    let m_n = spectrum.len() as f32;
                    let sum_x: f32 = (0..spectrum.len()).map(|i| i as f32).sum();
                    let sum_y: f32 = spectrum.iter().sum();
                    let sum_xx: f32 = (0..spectrum.len()).map(|i| (i as f32).powi(2)).sum();
                    let sum_xy: f32 = spectrum
                        .iter()
                        .enumerate()
                        .map(|(i, &mag)| i as f32 * mag)
                        .sum();
                    let s_xx = sum_xx - (sum_x * sum_x) / m_n;
                    let s_xy = sum_xy - (sum_x * sum_y) / m_n;
                    if s_xx.abs() > 1e-9 {
                        *spectral_slope = s_xy / s_xx;
                    }
                }
            }
        }
    }
}
