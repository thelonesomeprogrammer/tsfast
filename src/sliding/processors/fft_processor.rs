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
        let mut spectral_skewness = 0.0;
        let mut spectral_kurtosis = 0.0;

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

                    let kurt_num = m4 - 4.0 * c * m3 + 6.0 * c * c * m2 - 4.0 * c * c * c * m1 + c * c * c * c * m0;
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
                    let sum_x = m_n * (m_n - 1.0) / 2.0;
                    let sum_y = m0;
                    let sum_xx = m_n * (m_n - 1.0) * (2.0 * m_n - 1.0) / 6.0;
                    let sum_xy = m1;
                    let s_xx = sum_xx - (sum_x * sum_x) / m_n;
                    let s_xy = sum_xy - (sum_x * sum_y) / m_n;
                    if s_xx.abs() > 1e-9 {
                        spectral_slope = s_xy / s_xx;
                    }
                }
            }
        }

        Ok(FftResult {
            fft_complex,
            spectrum,
            freq_centroid,
            spectral_decrease,
            spectral_slope,
            spectral_spread,
            spectral_skewness,
            spectral_kurtosis,
            fft_autocorr,
        })
    }
}
