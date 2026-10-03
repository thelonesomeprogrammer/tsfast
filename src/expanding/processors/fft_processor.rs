use crate::common::ColumnState;
use crate::types::Compute;
use realfft::{RealToComplex, num_complex};
use std::sync::Arc;

use crate::metrics::FftResult;

pub struct FftProcessor;

impl FftProcessor {
    pub fn finalize(
        compute: Compute,
        full_series: &[f32],
        r2c_opt: &Option<Arc<dyn RealToComplex<f32>>>,
        fft_size: usize,
        fft_update_period: usize,
        state: &mut ColumnState,
    ) -> FftResult {
        let mut spectrum = Vec::new();
        let mut fft_complex = Vec::new();
        let mut freq_centroid = 0.0;
        let mut spectral_decrease = 0.0;
        let mut spectral_slope = 0.0;
        let mut spectral_spread = 0.0;
        let mut spectral_entropy = 0.0;

        if compute.intersects(Compute::ANY_FFT) {
            let n_total = full_series.len();
            let should_update =
                state.last_fft_n == 0 || (n_total - state.last_fft_n) >= fft_update_period;

            if should_update {
                if let Some(r2c) = r2c_opt {
                    if state.fft_in_buffer.len() < fft_size {
                        state.fft_in_buffer.resize(fft_size, 0.0);
                    }
                    state.fft_in_buffer.fill(0.0);
                    state.fft_in_buffer[..n_total].copy_from_slice(full_series);
                    let complex_len = r2c.complex_len();
                    if state.fft_out_buffer.len() < complex_len {
                        state
                            .fft_out_buffer
                            .resize(complex_len, num_complex::Complex::new(0.0, 0.0));
                    }
                    r2c.process(
                        &mut state.fft_in_buffer,
                        &mut state.fft_out_buffer[..complex_len],
                    )
                    .unwrap();
                    spectrum = state.fft_out_buffer[..complex_len]
                        .iter()
                        .map(|c| c.norm())
                        .collect();
                    fft_complex = state.fft_out_buffer[..complex_len].to_vec();
                } else if compute.intersects(Compute::SPEC_CENTROID | Compute::SPEC_DISTANCE | Compute::SPEC_DECREASE | Compute::SPEC_SLOPE | Compute::SPECTROGRAM | Compute::HUMAN_RANGE_E) {
                    use realfft::RealFftPlanner;
                    let mut planner = RealFftPlanner::<f32>::new();
                    let r2c = planner.plan_fft_forward(n_total);
                    let complex_len = r2c.complex_len();
                    state.fft_in_buffer.clear();
                    state.fft_in_buffer.extend_from_slice(full_series);
                    if state.fft_out_buffer.len() < complex_len {
                        state
                            .fft_out_buffer
                            .resize(complex_len, num_complex::Complex::new(0.0, 0.0));
                    }
                    r2c.process(
                        &mut state.fft_in_buffer,
                        &mut state.fft_out_buffer[..complex_len],
                    )
                    .unwrap();
                    spectrum = state.fft_out_buffer[..complex_len]
                        .iter()
                        .map(|c| c.norm())
                        .collect();
                    fft_complex = state.fft_out_buffer[..spectrum.len()].to_vec();
                }
                // Cache it
                state.last_fft_n = n_total;
                state.last_spectrum = spectrum.clone();
                state.last_fft_complex = fft_complex.clone();
            } else {
                // Use cached
                spectrum = state.last_spectrum.clone();
                fft_complex = state.last_fft_complex.clone();
            }

            if !spectrum.is_empty() {
                let spec_sum: f32 = spectrum.iter().sum();
                if spec_sum > 0.0 {
                    freq_centroid = spectrum
                        .iter()
                        .enumerate()
                        .map(|(i, &mag)| i as f32 * mag)
                        .sum::<f32>()
                        / spec_sum;

                    if spectrum.len() > 1 {
                        let spec_sum_no_first: f32 = spectrum[1..].iter().sum();
                        if spec_sum_no_first > 0.0 {
                            spectral_decrease = spectrum[1..]
                                .iter()
                                .enumerate()
                                .map(|(i, &mag)| (mag - spectrum[0]) / (i + 1) as f32)
                                .sum::<f32>()
                                / spec_sum_no_first;
                        }

                        let m_n = spectrum.len() as f32;
                        let sum_x = m_n * (m_n - 1.0) / 2.0;
                        let sum_y: f32 = spectrum.iter().sum();
                        let sum_xx = m_n * (m_n - 1.0) * (2.0 * m_n - 1.0) / 6.0;
                        let sum_xy: f32 = spectrum
                            .iter()
                            .enumerate()
                            .map(|(i, &mag)| i as f32 * mag)
                            .sum();
                        let s_xx = sum_xx - (sum_x * sum_x) / m_n;
                        let s_xy = sum_xy - (sum_x * sum_y) / m_n;
                        if s_xx.abs() > 1e-9 {
                            spectral_slope = s_xy / s_xx;
                        }
                    }

                    if compute.intersects(Compute::SPEC_SPREAD) {
                        let spread_sum: f32 = spectrum
                            .iter()
                            .enumerate()
                            .map(|(i, &mag)| (i as f32 - freq_centroid).powi(2) * mag)
                            .sum();
                        spectral_spread = (spread_sum / spec_sum).sqrt();
                    }

                    if compute.intersects(Compute::SPEC_ENTROPY) {
                        let mut p_sum = 0.0;
                        let mut power_vals = Vec::with_capacity(spectrum.len());
                        for (i, &mag) in spectrum.iter().enumerate() {
                            let power = if i == 0 { 0.0 } else { mag * mag };
                            power_vals.push(power);
                            p_sum += power;
                        }

                        if p_sum > 0.0 {
                            let mut entropy_sum = 0.0;
                            for &power in &power_vals {
                                if power > 0.0 {
                                    let p = power / p_sum;
                                    entropy_sum += p * p.log2();
                                }
                            }
                            if spectrum.len() > 1 {
                                spectral_entropy = -entropy_sum / (spectrum.len() as f32).log2();
                            }
                        }
                    }
                }
            }
        }

        FftResult {
            spectrum,
            fft_complex,
            freq_centroid,
            spectral_decrease,
            spectral_slope,
            spectral_spread,
            spectral_entropy,
            fft_autocorr: Vec::new(),
        }
    }
}
