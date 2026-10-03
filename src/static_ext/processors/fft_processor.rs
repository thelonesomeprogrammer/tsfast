use crate::common::ColumnState;
use crate::common::SlidingDFT;
use crate::types::Compute;

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
                    r2c_ref.process(&mut indata, &mut outdata)
                        .map_err(|e| e.to_string())?;
                    state.sliding_dft = Some(SlidingDFT::from_fft(outdata, values.len()));
                }
            }

            if let Some(ref sdft) = state.sliding_dft {
                fft_complex = sdft.bins.clone();
                spectrum = fft_complex.iter().map(|c| c.norm()).collect();
            }
        }

        let mut freq_centroid = 0.0;
        let mut spectral_decrease = 0.0;
        let mut spectral_slope = 0.0;

        let mut fft_autocorr = Vec::new();
        if compute.intersects(Compute::FULL_AUTOCORR | Compute::PACF) && n > 1.0 {
            let n2 = values.len() * 2;
            let fft_size_ac = crate::common::next_good_fft_size(n2);
            let mut planner = realfft::RealFftPlanner::<f32>::new();
            let r2c_ac = planner.plan_fft_forward(fft_size_ac);
            let c2r_ac = planner.plan_fft_inverse(fft_size_ac);

            let mut indata = vec![0.0; fft_size_ac];
            for (i, &v) in values.iter().enumerate() {
                indata[i] = v - mean;
            }
            let mut outdata = r2c_ac.make_output_vec();
            r2c_ac
                .process(&mut indata, &mut outdata)
                .map_err(|e| e.to_string())?;

            for c in &mut outdata {
                *c = realfft::num_complex::Complex::new(c.norm_sqr(), 0.0);
            }

            let mut outdata_inv = c2r_ac.make_output_vec();
            c2r_ac
                .process(&mut outdata, &mut outdata_inv)
                .map_err(|e| e.to_string())?;

            let var_ac = if n > 1.0 { m2 / (n - 1.0) } else { 0.0 };
            let m2_val = var_ac * (n - 1.0);
            if m2_val.abs() > 1e-9 {
                let scale = 1.0 / (fft_size_ac as f32);
                fft_autocorr = outdata_inv
                    .into_iter()
                    .take(values.len())
                    .map(|v| (v * scale) / m2_val)
                    .collect();
            }
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


        let mut mfcc = Vec::new();
        let mut cwt_energy = Vec::new();
        let mut cwt_entropy = 0.0;

        if compute.intersects(Compute::MFCC) {
            let nfilt = 40;
            let num_ceps = 12;

            if values.len() > 1 && !spectrum.is_empty() {
                let nfft_f32 = (spectrum.len() - 1) as f32 * 2.0;
                let fs = 100.0_f32;

                if state.mfcc_filter_banks.is_empty() || state.mfcc_filter_banks[0].len() != spectrum.len() {
                    let low_freq_mel = 0.0_f32;
                    let high_freq_mel = 2595.0_f32 * (1.0_f32 + (fs / 2.0_f32) / 700.0_f32).log10();

                    let mut mel_points = Vec::with_capacity(nfilt + 2);
                    for i in 0..(nfilt + 2) {
                        mel_points.push(low_freq_mel + i as f32 * (high_freq_mel - low_freq_mel) / (nfilt as f32 + 1.0_f32));
                    }

                    let hz_points: Vec<f32> = mel_points.iter().map(|m| 700.0_f32 * (10.0_f32.powf(m / 2595.0_f32) - 1.0_f32)).collect();
                    let filter_bin: Vec<usize> = hz_points.iter().map(|h| (((nfft_f32 + 1.0) * h / fs).floor() as usize).min(spectrum.len() - 1)).collect();

                    state.mfcc_filter_banks = vec![vec![0.0; spectrum.len()]; nfilt];
                    for m in 1..=nfilt {
                        let f_m_minus = filter_bin[m - 1];
                        let f_m = filter_bin[m];
                        let f_m_plus = filter_bin[m + 1];

                        let enorm = 2.0_f32 / (hz_points[m + 1] - hz_points[m - 1]).max(f32::EPSILON);

                        let denom1 = (f_m as f32 - f_m_minus as f32).max(1.0);
                        for k in f_m_minus..f_m {
                            state.mfcc_filter_banks[m - 1][k] = enorm * (k as f32 - f_m_minus as f32) / denom1;
                        }

                        let denom2 = (f_m_plus as f32 - f_m as f32).max(1.0);
                        for k in f_m..f_m_plus {
                            state.mfcc_filter_banks[m - 1][k] = enorm * (f_m_plus as f32 - k as f32) / denom2;
                        }
                    }

                    state.mfcc_dct_matrix = vec![vec![0.0; nfilt]; num_ceps];
                    let ortho_factor = (2.0_f32 / nfilt as f32).sqrt();
                    for k in 1..=num_ceps {
                        for n in 0..nfilt {
                            state.mfcc_dct_matrix[k - 1][n] = ortho_factor * 2.0_f32 *
                                (std::f32::consts::PI * k as f32 * (2.0 * n as f32 + 1.0) / (2.0 * nfilt as f32)).cos();
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
                    let bank_sum: f32 = pow_frames.iter().zip(filter.iter()).map(|(p, f)| p * f).sum();
                    let val = if bank_sum <= 0.0 { f32::EPSILON } else { bank_sum };
                    filter_banks[m] = 20.0_f32 * val.log10();
                }

                let mut dct = vec![0.0_f32; num_ceps];
                for k in 0..num_ceps {
                    let dct_row = &state.mfcc_dct_matrix[k];
                    dct[k] = filter_banks.iter().zip(dct_row.iter()).map(|(f, d)| f * d).sum();
                }

                let cep_lifter = 22.0_f32;
                for i in 0..num_ceps {
                    let lift = 1.0_f32 + (cep_lifter / 2.0_f32) * (std::f32::consts::PI * i as f32 / cep_lifter).sin();
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
                        let const_val = 2.0_f32 / (3.0_f32.sqrt() * std::f32::consts::PI.powf(0.25));
                        let norm = 1.0_f32 / s.sqrt();

                        let kernel_len = (2 * bound + 1) as usize;
                        let mut kernel = Vec::with_capacity(kernel_len);
                        for n in -bound..=bound {
                            let t_val = n as f32 * dt / s;
                            let psi = const_val * (1.0_f32 - t_val * t_val) * (-t_val * t_val / 2.0_f32).exp();
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

                        let start_k = if start_data_idx < 0 { (-start_data_idx) as usize } else { 0 };
                        let end_k = if start_data_idx + kernel_len as i32 > values.len() as i32 {
                            (values.len() as i32 - start_data_idx) as usize
                        } else {
                            kernel_len
                        };

                        if start_k < end_k {
                            let data_slice = &values[(start_data_idx + start_k as i32) as usize .. (start_data_idx + end_k as i32) as usize];
                            let kernel_slice = &kernel[start_k..end_k];
                            sum = data_slice.iter().zip(kernel_slice.iter()).map(|(d, k)| d * k).sum();
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

        Ok(FftResult {
            fft_complex,
            spectrum,
            freq_centroid,
            spectral_decrease,
            spectral_slope,
            fft_autocorr,
            mfcc,
            cwt_energy,
            cwt_entropy,
        })
    }
}
