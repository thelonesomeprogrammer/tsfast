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

        let mut welch_density = Vec::new();
        if compute.intersects(Compute::WELCH) && !values.is_empty() {
            let nperseg = values.len().min(256);
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
            fft_autocorr,
        })
    }
}
