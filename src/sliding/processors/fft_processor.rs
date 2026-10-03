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
        let mut spectral_skewness = 0.0;
        let mut spectral_kurtosis = 0.0;
        let mut spectral_roll_on = 0.0;
        let mut spectral_roll_off = 0.0;
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
        })
    }
}
