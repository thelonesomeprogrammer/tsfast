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
                } else if compute.intersects(
                    Compute::SPEC_CENTROID
                        | Compute::SPEC_DISTANCE
                        | Compute::SPEC_DECREASE
                        | Compute::SPEC_SLOPE
                        | Compute::SPECTROGRAM
                        | Compute::HUMAN_RANGE_E,
                ) {
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
                }
            }
        }

        let mut welch_density = Vec::new();
        if compute.intersects(Compute::WELCH) && !full_series.is_empty() {
            let nperseg = full_series.len().min(256);
            if nperseg > 0 {
                let step = nperseg / 2;
                let mut num_segments = 0;

                if state.welch_planner.is_none() {
                    state.welch_planner = Some(std::sync::Arc::new(std::sync::Mutex::new(realfft::RealFftPlanner::<f32>::new())));
                }

                let mut planner_lock = state.welch_planner.as_ref().unwrap().lock().unwrap();
                let r2c_welch = planner_lock.plan_fft_forward(nperseg);
                let complex_len = r2c_welch.complex_len();
                welch_density = vec![0.0; complex_len];
                let mut indata = vec![0.0; nperseg];
                let mut outdata = r2c_welch.make_output_vec();

                let mut window = vec![0.0; nperseg];
                let mut window_sum_sq = 0.0;
                let pi2 = 2.0 * std::f32::consts::PI;
                for i in 0..nperseg {
                    let w = 0.5 * (1.0 - (pi2 * i as f32 / nperseg as f32).cos());
                    window[i] = w;
                    window_sum_sq += w * w;
                }

                let scale = if window_sum_sq > 0.0 { 1.0 / window_sum_sq } else { 1.0 };

                let mut start = 0;
                while start < full_series.len() {
                    let end = (start + nperseg).min(full_series.len());
                    let actual_len = end - start;
                    if actual_len < nperseg {
                        if num_segments == 0 {
                        } else {
                            break;
                        }
                    }

                    for i in 0..nperseg {
                        if start + i < full_series.len() {
                            indata[i] = full_series[start + i] * window[i];
                        } else {
                            indata[i] = 0.0;
                        }
                    }

                    if r2c_welch.process(&mut indata, &mut outdata).is_ok() {
                        for (i, c) in outdata.iter().enumerate() {
                            let mag_sq = c.norm_sqr();
                            if i == 0 || i == complex_len - 1 {
                                welch_density[i] += mag_sq * scale;
                            } else {
                                welch_density[i] += mag_sq * scale * 2.0;
                            }
                        }
                        num_segments += 1;
                    }

                    if start + nperseg >= full_series.len() {
                        break;
                    }
                    start += step;
                }

                if num_segments > 0 {
                    for v in welch_density.iter_mut() {
                        *v /= num_segments as f32;
                    }
                }
            }
        }

        let cwt_peaks = 0;

        FftResult {
            cwt_peaks,
            welch_density,

            spectrum,
            fft_complex,
            freq_centroid,
            spectral_decrease,
            spectral_slope,
            fft_autocorr: Vec::new(),
        }
    }
}
