use crate::common::ColumnState;
use crate::common::SlidingDFT;
use crate::types::Compute;
use std::simd::f32x4;
use std::simd::num::SimdFloat;

use crate::metrics::FftResult;

pub struct FftProcessor;

impl FftProcessor {
    pub fn finalize(
        compute: Compute,
        values: &[f32],
        n: f32,
        mean: f32,
        m2: f32,
        fft_size: usize,
        r2c: &Option<std::sync::Arc<dyn realfft::RealToComplex<f32>>>,
        state: &mut ColumnState,
    ) -> Result<FftResult, String> {
        let mut fft_complex = Vec::new();
        let mut spectrum = Vec::new();

        if compute.intersects(Compute::ANY_FFT) {
            if state.sliding_dft.is_none() {
                if let Some(r2c_ref) = r2c {
                    let mut indata = vec![0.0; fft_size];
                    indata[..values.len()].copy_from_slice(values);
                    let mut outdata = r2c_ref.make_output_vec();
                    r2c_ref
                        .process(&mut indata, &mut outdata)
                        .map_err(|e| e.to_string())?;
                    state.sliding_dft = Some(SlidingDFT::from_fft(outdata, values.len()));
                }
            }

            if let Some(ref sdft) = state.sliding_dft {
                fft_complex = sdft.bins.clone();
                let mut buf = std::mem::take(&mut state.spectrum_buffer);
                buf.clear();
                buf.extend(fft_complex.iter().map(|c| c.norm()));
                spectrum = buf;
            }
        }

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

        let mut fft_autocorr = Vec::new();

        if compute.intersects(Compute::FULL_AUTOCORR | Compute::PACF | Compute::LPCC) && n > 1.0 {
            let n2 = values.len() * 2;
            let fft_size_ac = crate::common::next_good_fft_size(n2);
            let mut planner = realfft::RealFftPlanner::<f32>::new();
            let r2c_ac = planner.plan_fft_forward(fft_size_ac);
            let c2r_ac = planner.plan_fft_inverse(fft_size_ac);

            let mut indata = std::mem::take(&mut state.fft_in_buffer);
            if indata.len() < fft_size_ac {
                indata.resize(fft_size_ac, 0.0);
            }
            if fft_size_ac > values.len() {
                indata[values.len()..fft_size_ac].fill(0.0);
            }
            for (i, &v) in values.iter().enumerate() {
                indata[i] = v - mean;
            }

            let mut outdata = std::mem::take(&mut state.fft_out_buffer);
            let complex_len = r2c_ac.complex_len();
            if outdata.len() < complex_len {
                outdata.resize(complex_len, realfft::num_complex::Complex::new(0.0, 0.0));
            }

            r2c_ac
                .process(&mut indata[..fft_size_ac], &mut outdata[..complex_len])
                .map_err(|e| e.to_string())?;

            for c in &mut outdata[..complex_len] {
                *c = realfft::num_complex::Complex::new(c.norm_sqr(), 0.0);
            }

            let mut outdata_inv = std::mem::take(&mut state.fft_inv_buffer);
            if outdata_inv.len() < fft_size_ac {
                outdata_inv.resize(fft_size_ac, 0.0);
            }

            c2r_ac
                .process(&mut outdata[..complex_len], &mut outdata_inv[..fft_size_ac])
                .map_err(|e| e.to_string())?;

            let var_ac = if n > 1.0 { m2 / (n - 1.0) } else { 0.0 };
            let m2_val = var_ac * (n - 1.0);
            if m2_val.abs() > 1e-9 {
                let scale = 1.0 / (fft_size_ac as f32);
                fft_autocorr = outdata_inv[..values.len()]
                    .iter()
                    .map(|&v| (v * scale) / m2_val)
                    .collect();
            } else {
                let scale = 1.0 / (fft_size_ac as f32);
                fft_autocorr = outdata_inv[..values.len()]
                    .iter()
                    .map(|&v| (v * scale))
                    .collect();
            }

            state.fft_in_buffer = indata;
            state.fft_out_buffer = outdata;
            state.fft_inv_buffer = outdata_inv;
        }

        if !spectrum.is_empty() {
            let mut m0 = 0.0;
            let mut m1 = 0.0;
            let mut m2 = 0.0;
            let mut m3 = 0.0;
            let mut m4 = 0.0;

            for (i, &mag) in spectrum.iter().enumerate() {
                let f = i as f32;
                m0 += mag;
                let f_mag = f * mag;
                m1 += f_mag;
                let f2_mag = f * f_mag;
                m2 += f2_mag;
                let f3_mag = f * f2_mag;
                m3 += f3_mag;
                m4 += f * f3_mag;
            }

            if m0 > 0.0 {
                let c = m1 / m0;
                freq_centroid = c;

                // algebraic expansion of spread
                let spread_sq = (m2 / m0) - (c * c);
                if spread_sq > 0.0 {
                    spectral_spread = spread_sq.sqrt();
                    let spread_cube = spectral_spread * spread_sq;
                    let spread_quad = spread_sq * spread_sq;

                    let skew_num = m3 - 3.0 * c * m2 + 3.0 * c * c * m1 - c * c * c * m0;
                    spectral_skewness = skew_num / (m0 * spread_cube);

                    let kurt_num = m4 - 4.0 * c * m3 + 6.0 * c * c * m2 - 4.0 * c * c * c * m1
                        + c * c * c * c * m0;
                    spectral_kurtosis = kurt_num / (m0 * spread_quad);
                }

                if spectrum.len() > 1 {
                    let spec_sum_no_first = m0 - spectrum[0];
                    if spec_sum_no_first > 0.0 {
                        let mut sd_num = 0.0;
                        for (i, &mag) in spectrum[1..].iter().enumerate() {
                            sd_num += (mag - spectrum[0]) / (i + 1) as f32;
                        }
                        spectral_decrease = sd_num / spec_sum_no_first;
                    }

                    let m_n = spectrum.len() as f32;
                    let n_fft = (spectrum.len() - 1) * 2;
                    let freq_step = 100.0 / n_fft as f32;

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
                        let hit_off = roll_off_idx < 0.0 && cumsum + block_sum >= roll_off_thresh;

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
                    spectral_spread = (spread_sum / m0).sqrt();
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

        let mut mfcc = Vec::new();
        let mut lpcc = Vec::new();
        let mut cwt_energy = Vec::new();
        let mut cwt_entropy = 0.0;

        if compute.intersects(Compute::LPCC) {
            lpcc = crate::features::lpc::compute_lpcc(&fft_autocorr, values.len());
        }

        if compute.intersects(Compute::MFCC) {
            let nfilt = 40;
            let num_ceps = 12;

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

            if values.len() > 0 {
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

                let mut buffer: Vec<f32> = std::mem::take(&mut state.fft_in_buffer);
                let mut cwt_coeffs = Vec::with_capacity(max_width - 1);

                for scale in 1..max_width {
                    let kernel = &state.cwt_kernels[scale - 1];
                    let bound = (kernel.len() / 2) as i32;
                    let kernel_len = kernel.len();

                    buffer.clear();
                    buffer.resize(values.len(), 0.0);

                    for i in 0..values.len() {
                        let mut sum = 0.0_f32;
                        let start_data_idx = i as i32 - bound;

                        let start_k = if start_data_idx < 0 {
                            (-start_data_idx) as usize
                        } else {
                            0
                        };
                        let end_k = if start_data_idx + kernel_len as i32 > values.len() as i32 {
                            (values.len() as i32 - start_data_idx) as usize
                        } else {
                            kernel_len
                        };

                        if start_k < end_k {
                            let data_slice = &values[(start_data_idx + start_k as i32) as usize
                                ..(start_data_idx + end_k as i32) as usize];
                            let kernel_slice = &kernel[start_k..end_k];
                            sum = data_slice
                                .iter()
                                .zip(kernel_slice.iter())
                                .map(|(d, k)| d * k)
                                .sum();
                        }
                        buffer[i] = sum;
                    }
                    cwt_coeffs.push(buffer.clone());
                }
                state.fft_in_buffer = buffer;

                let mut energy_sum = 0.0_f32;
                for conv in &cwt_coeffs {
                    let scale_energy: f32 = conv.iter().map(|c| c * c).sum();
                    let scale_sum_abs: f32 = conv.iter().map(|c| c.abs()).sum();

                    cwt_energy.push((scale_energy / values.len() as f32).sqrt());
                    energy_sum += scale_sum_abs;
                }

                if energy_sum > 0.0_f32 {
                    for conv in &cwt_coeffs {
                        let scale_sum_abs: f32 = conv.iter().map(|c| c.abs()).sum();
                        let p = scale_sum_abs / energy_sum;
                        if p > 0.0_f32 {
                            cwt_entropy -= p * p.ln();
                        }
                    }
                }
            } else {
                cwt_energy = vec![0.0_f32; max_width - 1];
            }
        }

        let mut welch_density = Vec::new();
        if compute.intersects(Compute::WELCH) && !values.is_empty() {
            let nperseg = values.len().min(256);
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
                while start < values.len() {
                    let end = (start + nperseg).min(values.len());
                    let actual_len = end - start;
                    if actual_len < nperseg {
                        if num_segments == 0 {
                        } else {
                            break;
                        }
                    }

                    for i in 0..nperseg {
                        if start + i < values.len() {
                            indata[i] = values[start + i] * window[i];
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

                    if start + nperseg >= values.len() {
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

        Ok(FftResult {
            cwt_peaks,
            welch_density,

            fft_complex,
            spectrum,
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
        })
    }
}
