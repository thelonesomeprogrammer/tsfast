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
        mean: f32,
        m2: f32,
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
        let mut spectral_roll_on = 0.0;
        let mut spectral_roll_off = 0.0;
        let mut spectral_spread = 0.0;
        let mut spectral_skewness = 0.0;
        let mut spectral_kurtosis = 0.0;

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
                        &mut state.fft_in_buffer[..r2c.len()],
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
                        &mut state.fft_in_buffer[..r2c.len()],
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
                            let mag = f32x4::from_slice(&spectrum[i..i + 4]);
                            let f = f32x4::from_array([
                                i as f32 * freq_step,
                                (i + 1) as f32 * freq_step,
                                (i + 2) as f32 * freq_step,
                                (i + 3) as f32 * freq_step,
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
                            let mag = f32x4::from_slice(&spectrum[j..j + 4]);
                            let block_sum = mag.reduce_sum();

                            // We check if either roll_on or roll_off is hit in this block
                            let hit_on = roll_on_idx < 0.0 && cumsum + block_sum >= roll_on_thresh;
                            let hit_off =
                                roll_off_idx < 0.0 && cumsum + block_sum >= roll_off_thresh;

                            if hit_on || hit_off {
                                for k in 0..4 {
                                    cumsum += spectrum[j + k];
                                    if roll_on_idx < 0.0 && cumsum >= roll_on_thresh {
                                        roll_on_idx = (j + k) as f32;
                                    }
                                    if roll_off_idx < 0.0 && cumsum >= roll_off_thresh {
                                        roll_off_idx = (j + k) as f32;
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


        let mut fft_autocorr = Vec::new();
        let n = full_series.len() as f32;

        if compute.intersects(Compute::FULL_AUTOCORR | Compute::PACF) && n > 1.0 {
            let n2 = full_series.len() * 2;
            let fft_size_ac = crate::common::next_good_fft_size(n2);
            let mut planner = realfft::RealFftPlanner::<f32>::new();
            let r2c_ac = planner.plan_fft_forward(fft_size_ac);
            let c2r_ac = planner.plan_fft_inverse(fft_size_ac);

            let mut indata = std::mem::take(&mut state.fft_in_buffer);
            if indata.len() < fft_size_ac {
                indata.resize(fft_size_ac, 0.0);
            }
            if fft_size_ac > full_series.len() {
                indata[full_series.len()..fft_size_ac].fill(0.0);
            }
            for (i, &v) in full_series.iter().enumerate() {
                indata[i] = v - mean;
            }

            let mut outdata = std::mem::take(&mut state.fft_out_buffer);
            let complex_len = r2c_ac.complex_len();
            if outdata.len() < complex_len {
                outdata.resize(complex_len, realfft::num_complex::Complex::new(0.0, 0.0));
            }

            if r2c_ac
                .process(&mut indata[..fft_size_ac], &mut outdata[..complex_len])
                .is_ok()
            {
                for c in &mut outdata[..complex_len] {
                    *c = realfft::num_complex::Complex::new(c.norm_sqr(), 0.0);
                }

                let mut outdata_inv = std::mem::take(&mut state.fft_inv_buffer);
                if outdata_inv.len() < fft_size_ac {
                    outdata_inv.resize(fft_size_ac, 0.0);
                }

                if c2r_ac
                    .process(&mut outdata[..complex_len], &mut outdata_inv[..fft_size_ac])
                    .is_ok()
                {
                    let var_ac = if n > 1.0 { m2 / (n - 1.0) } else { 0.0 };
                    let m2_val = var_ac * (n - 1.0);
                    if m2_val.abs() > 1e-9 {
                        let scale = 1.0 / (fft_size_ac as f32);
                        fft_autocorr = outdata_inv[..full_series.len()]
                            .iter()
                            .map(|&v| (v * scale) / m2_val)
                            .collect();
                    } else {
                        let scale = 1.0 / (fft_size_ac as f32);
                        fft_autocorr = outdata_inv[..full_series.len()]
                            .iter()
                            .map(|&v| (v * scale))
                            .collect();
                    }
                }
                state.fft_inv_buffer = outdata_inv;
            }
            state.fft_in_buffer = indata;
            state.fft_out_buffer = outdata;
        }

        let mut mfcc = Vec::new();
        let mut lpcc = Vec::new();
        let mut cwt_energy = Vec::new();
        let mut cwt_entropy = 0.0;

        if compute.intersects(Compute::LPCC) {
            lpcc = crate::features::lpc::compute_lpcc(full_series);
        }

        if compute.intersects(Compute::MFCC) {
            let nfilt = 40;
            let num_ceps = 12;

            let values = full_series;
            if values.len() > 1 && !spectrum.is_empty() {
                let nfft_f32 = (spectrum.len() - 1) as f32 * 2.0;
                let fs = 100.0_f32;

                if state.mfcc_filter_banks.is_empty()
                    || state.mfcc_filter_banks[0].len() != spectrum.len()
                {
                    let low_freq_mel = 0.0_f32;
                    let high_freq_mel = 2595.0_f32 * (1.0_f32 + (fs / 2.0_f32) / 700.0_f32).log10();

                    let mut mel_points = Vec::with_capacity(nfilt + 2);
                    for i in 0..(nfilt + 2) {
                        mel_points.push(
                            low_freq_mel
                                + i as f32 * (high_freq_mel - low_freq_mel)
                                    / (nfilt as f32 + 1.0_f32),
                        );
                    }

                    let hz_points: Vec<f32> = mel_points
                        .iter()
                        .map(|m| 700.0_f32 * (10.0_f32.powf(m / 2595.0_f32) - 1.0_f32))
                        .collect();
                    let filter_bin: Vec<usize> = hz_points
                        .iter()
                        .map(|h| {
                            (((nfft_f32 + 1.0) * h / fs).floor() as usize).min(spectrum.len() - 1)
                        })
                        .collect();

                    state.mfcc_filter_banks = vec![vec![0.0; spectrum.len()]; nfilt];
                    for m in 1..=nfilt {
                        let f_m_minus = filter_bin[m - 1];
                        let f_m = filter_bin[m];
                        let f_m_plus = filter_bin[m + 1];

                        let enorm =
                            2.0_f32 / (hz_points[m + 1] - hz_points[m - 1]).max(f32::EPSILON);

                        let denom1 = (f_m as f32 - f_m_minus as f32).max(1.0);
                        for k in f_m_minus..f_m {
                            state.mfcc_filter_banks[m - 1][k] =
                                enorm * (k as f32 - f_m_minus as f32) / denom1;
                        }

                        let denom2 = (f_m_plus as f32 - f_m as f32).max(1.0);
                        for k in f_m..f_m_plus {
                            state.mfcc_filter_banks[m - 1][k] =
                                enorm * (f_m_plus as f32 - k as f32) / denom2;
                        }
                    }

                    state.mfcc_dct_matrix = vec![vec![0.0; nfilt]; num_ceps];
                    let ortho_factor = (2.0_f32 / nfilt as f32).sqrt();
                    for k in 1..=num_ceps {
                        for n in 0..nfilt {
                            state.mfcc_dct_matrix[k - 1][n] = ortho_factor
                                * 2.0_f32
                                * (std::f32::consts::PI * k as f32 * (2.0 * n as f32 + 1.0)
                                    / (2.0 * nfilt as f32))
                                    .cos();
                        }
                    }
                }

                let mut pow_frames: Vec<f32> = std::mem::take(&mut state.fft_in_buffer);
                pow_frames.clear();
                for k in 0..spectrum.len() {
                    let w = 2.0 * std::f32::consts::PI * (k as f32) / nfft_f32;
                    let factor = 1.0 + 0.97 * 0.97 - 2.0 * 0.97 * w.cos();
                    pow_frames.push(spectrum[k] * spectrum[k] * factor / nfft_f32);
                }

                let mut filter_banks = vec![0.0_f32; nfilt];

                for m in 0..nfilt {
                    let filter = &state.mfcc_filter_banks[m];
                    let bank_sum: f32 = pow_frames
                        .iter()
                        .zip(filter.iter())
                        .map(|(p, f)| p * f)
                        .sum();
                    let val = if bank_sum <= 0.0 {
                        f32::EPSILON
                    } else {
                        bank_sum
                    };
                    filter_banks[m] = 20.0_f32 * val.log10();
                }

                let mut dct = vec![0.0_f32; num_ceps];
                for k in 0..num_ceps {
                    let dct_row = &state.mfcc_dct_matrix[k];
                    dct[k] = filter_banks
                        .iter()
                        .zip(dct_row.iter())
                        .map(|(f, d)| f * d)
                        .sum();
                }

                let cep_lifter = 22.0_f32;
                for i in 0..num_ceps {
                    let lift = 1.0_f32
                        + (cep_lifter / 2.0_f32)
                            * (std::f32::consts::PI * i as f32 / cep_lifter).sin();
                    dct[i] *= lift;
                }

                mfcc = dct;
                state.fft_in_buffer = pow_frames;
            } else {
                mfcc = vec![0.0_f32; num_ceps];
            }
        }

        if compute.intersects(Compute::CWT_MEXH) {
            let max_width = 10;
            let dt = 1.0_f32;

            if state.cwt_final_energy.is_empty() {
                state.cwt_final_energy = vec![0.0_f32; max_width - 1];
                state.cwt_final_sum_abs = vec![0.0_f32; max_width - 1];
                state.cwt_final_clnc = vec![0.0_f32; max_width - 1];
            }

            if full_series.len() > 0 {
                if state.cwt_kernels.is_empty() {
                    state.cwt_kernels = Vec::with_capacity(max_width - 1);
                    for scale in 1..max_width {
                        let s = scale as f32;
                        let bound = (8.0_f32 * s) as i32;
                        let const_val =
                            2.0_f32 / (3.0_f32.sqrt() * std::f32::consts::PI.powf(0.25));
                        let norm = 1.0_f32 / s.sqrt();

                        let kernel_len = (2 * bound + 1) as usize;
                        let mut kernel = Vec::with_capacity(kernel_len);
                        for n in -bound..=bound {
                            let t_val = n as f32 * dt / s;
                            let psi = const_val
                                * (1.0_f32 - t_val * t_val)
                                * (-t_val * t_val / 2.0_f32).exp();
                            kernel.push(psi * norm);
                        }
                        kernel.reverse();
                        state.cwt_kernels.push(kernel);
                    }
                }

                let mut energy_sum = 0.0_f32;
                let mut current_sum_abs = Vec::with_capacity(max_width - 1);
                let mut current_clnc = Vec::with_capacity(max_width - 1);

                for scale in 1..max_width {
                    let kernel = &state.cwt_kernels[scale - 1];
                    let bound = (kernel.len() / 2) as i32;
                    let kernel_len = kernel.len();

                    let n_idx = full_series.len() - 1;

                    if n_idx as i32 - bound >= 0 {
                        let final_i = (n_idx as i32 - bound) as usize;
                        let mut sum = 0.0_f32;
                        let start_data_idx = final_i as i32 - bound;

                        let start_k = if start_data_idx < 0 {
                            (-start_data_idx) as usize
                        } else {
                            0
                        };
                        let end_k = if start_data_idx + kernel_len as i32 > full_series.len() as i32
                        {
                            (full_series.len() as i32 - start_data_idx) as usize
                        } else {
                            kernel_len
                        };
                        if start_k < end_k {
                            let data_slice = &full_series[(start_data_idx + start_k as i32) as usize
                                ..(start_data_idx + end_k as i32) as usize];
                            let kernel_slice = &kernel[start_k..end_k];
                            sum = data_slice
                                .iter()
                                .zip(kernel_slice.iter())
                                .map(|(d, k)| d * k)
                                .sum();
                        }

                        state.cwt_final_energy[scale - 1] += sum * sum;
                        let sum_abs = sum.abs();
                        state.cwt_final_sum_abs[scale - 1] += sum_abs;
                        if sum_abs > 0.0 {
                            state.cwt_final_clnc[scale - 1] += sum_abs * sum_abs.ln();
                        }
                    }

                    let mut tail_energy = 0.0_f32;
                    let mut tail_sum_abs = 0.0_f32;
                    let mut tail_clnc = 0.0_f32;

                    let start_tail = (n_idx as i32 - bound + 1).max(0) as usize;
                    for i in start_tail..full_series.len() {
                        let mut sum = 0.0_f32;
                        let start_data_idx = i as i32 - bound;

                        let start_k = if start_data_idx < 0 {
                            (-start_data_idx) as usize
                        } else {
                            0
                        };
                        let end_k = if start_data_idx + kernel_len as i32 > full_series.len() as i32
                        {
                            (full_series.len() as i32 - start_data_idx) as usize
                        } else {
                            kernel_len
                        };
                        if start_k < end_k {
                            let data_slice = &full_series[(start_data_idx + start_k as i32) as usize
                                ..(start_data_idx + end_k as i32) as usize];
                            let kernel_slice = &kernel[start_k..end_k];
                            sum = data_slice
                                .iter()
                                .zip(kernel_slice.iter())
                                .map(|(d, k)| d * k)
                                .sum();
                        }

                        tail_energy += sum * sum;
                        let sum_abs = sum.abs();
                        tail_sum_abs += sum_abs;
                        if sum_abs > 0.0 {
                            tail_clnc += sum_abs * sum_abs.ln();
                        }
                    }

                    let scale_energy = state.cwt_final_energy[scale - 1] + tail_energy;
                    let scale_sum_abs = state.cwt_final_sum_abs[scale - 1] + tail_sum_abs;
                    let scale_clnc = state.cwt_final_clnc[scale - 1] + tail_clnc;

                    cwt_energy.push((scale_energy / full_series.len() as f32).sqrt());
                    current_sum_abs.push(scale_sum_abs);
                    current_clnc.push(scale_clnc);
                    energy_sum += scale_sum_abs;
                }

                if energy_sum > 0.0_f32 {
                    for scale in 1..max_width {
                        let _clnc = current_clnc[scale - 1];
                        let sum_abs = current_sum_abs[scale - 1];
                        let prob = sum_abs / energy_sum;
                        if prob > 0.0 {
                            cwt_entropy -= prob * prob.ln();
                        }
                    }
                }
            } else {
                cwt_energy = vec![0.0_f32; max_width - 1];
            }
        }

        let mut welch_density = Vec::new();
        if compute.intersects(Compute::WELCH) && !full_series.is_empty() {
            let nperseg = full_series.len().min(256);
            if nperseg > 0 {
                let step = nperseg / 2;
                let mut num_segments = 0;

                if state.welch_planner.is_none() {
                    state.welch_planner = Some(std::sync::Arc::new(std::sync::Mutex::new(
                        realfft::RealFftPlanner::<f32>::new(),
                    )));
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

                let scale = if window_sum_sq > 0.0 {
                    1.0 / window_sum_sq
                } else {
                    1.0
                };

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
            spectral_spread,
            spectral_entropy,
            spectral_roll_on,
            spectral_roll_off,
            spectral_skewness,
            spectral_kurtosis,
            fft_autocorr,
            mfcc,
            lpcc,
            cwt_energy,
            cwt_entropy,
        }
    }
}
