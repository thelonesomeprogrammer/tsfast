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
        let mut spectral_roll_on = 0.0;
        let mut spectral_roll_off = 0.0;

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
                    let n_fft = (spectrum.len() - 1) * 2;
                    let freq_step = 100.0 / n_fft as f32;

                    use std::simd::f32x4;
                    use std::simd::num::SimdFloat;

                    let mut sum_y_vec = f32x4::splat(0.0);
                    let mut sum_xy_vec = f32x4::splat(0.0);
                    let mut sum_f_vec = f32x4::splat(0.0);
                    let mut dot_ff_vec = f32x4::splat(0.0);

                    let mut i = 0;
                    while i + 4 <= spectrum.len() {
                        let mag = f32x4::from_slice(&spectrum[i..i+4]);
                        let f = f32x4::from_array([
                            i as f32 * freq_step,
                            (i+1) as f32 * freq_step,
                            (i+2) as f32 * freq_step,
                            (i+3) as f32 * freq_step
                        ]);

                        sum_y_vec += mag;
                        sum_xy_vec += f * mag;
                        sum_f_vec += f;
                        dot_ff_vec += f * f;
                        i += 4;
                    }

                    let mut sum_y = sum_y_vec.reduce_sum();
                    let mut sum_xy = sum_xy_vec.reduce_sum();
                    let mut sum_f = sum_f_vec.reduce_sum();
                    let mut dot_ff = dot_ff_vec.reduce_sum();

                    while i < spectrum.len() {
                        let f = i as f32 * freq_step;
                        let mag = spectrum[i];
                        sum_y += mag;
                        sum_xy += f * mag;
                        sum_f += f;
                        dot_ff += f * f;
                        i += 1;
                    }

                    let denom = m_n * dot_ff - (sum_f * sum_f);
                    if denom.abs() > 1e-9 {
                        let num = (1.0 / sum_y) * (m_n * sum_xy - sum_f * sum_y);
                        spectral_slope = num / denom;
                    }

                    let roll_on_thresh = 0.05 * sum_y;
                    let roll_off_thresh = 0.95 * sum_y;

                    let mut cumsum = 0.0;
                    let mut roll_on_idx = -1.0;
                    let mut roll_off_idx = -1.0;

                    let mut j = 0;
                    while j + 4 <= spectrum.len() {
                        let mag = f32x4::from_slice(&spectrum[j..j+4]);
                        let block_sum = mag.reduce_sum();

                        // We check if either roll_on or roll_off is hit in this block
                        let hit_on = roll_on_idx < 0.0 && cumsum + block_sum >= roll_on_thresh;
                        let hit_off = roll_off_idx < 0.0 && cumsum + block_sum >= roll_off_thresh;

                        if hit_on || hit_off {
                            for k in 0..4 {
                                cumsum += spectrum[j+k];
                                if roll_on_idx < 0.0 && cumsum >= roll_on_thresh {
                                    roll_on_idx = (j+k) as f32;
                                }
                                if roll_off_idx < 0.0 && cumsum >= roll_off_thresh {
                                    roll_off_idx = (j+k) as f32;
                                }
                            }
                            if roll_off_idx >= 0.0 {
                                break;
                            }
                        } else {
                            cumsum += block_sum;
                        }

                        j += 4;
                    }

                    while j < spectrum.len() {
                        if roll_off_idx >= 0.0 {
                            break;
                        }
                        cumsum += spectrum[j];
                        if roll_on_idx < 0.0 && cumsum >= roll_on_thresh {
                            roll_on_idx = j as f32;
                        }
                        if roll_off_idx < 0.0 && cumsum >= roll_off_thresh {
                            roll_off_idx = j as f32;
                            break;
                        }
                        j += 1;
                    }

                    if roll_on_idx >= 0.0 {
                        spectral_roll_on = roll_on_idx * freq_step;
                    }
                    if roll_off_idx >= 0.0 {
                        spectral_roll_off = roll_off_idx * freq_step;
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
            spectral_roll_on,
            spectral_roll_off,
            fft_autocorr: Vec::new(),
        }
    }
}
