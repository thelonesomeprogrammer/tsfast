use crate::types::{Feature, AggAttr, AggFunc, FftAttr};
use crate::common::ColumnState;

#[inline(always)]
pub fn eval_transform(
    feat: &Feature,
    values: &[f32],
    state: &mut ColumnState,
    n: f32,
    mean: f32,
    m2: f32,
    m3: f32,
    m4: f32,
    mad_sum: f32,
    iqr: f32,
    entropy: f32,
    count_a: usize,
    count_b: usize,
    max_strike_a: usize,
    max_strike_b: usize,
    zc_mean: f32,
    zc_std: f32,
    freq_centroid: f32,
    spectral_decrease: f32,
    spectral_slope: f32,
    first_max_idx: usize,
    last_max_idx: usize,
    first_min_idx: usize,
    last_min_idx: usize,
    fft_autocorr: &[f32],
    fft_complex: &[realfft::num_complex::Complex<f32>],
    spectrum: &[f32],
    unique_c3_lags: &[u16],
    unique_paa_totals: &[u16],
    paa_boundaries: &[Vec<usize>],
    var: f32,
    std_dev: f32,
    mac_sum: f32,
    mc_sum: f32,
    median: f32,
) -> Option<f32> {
    let res = match feat {
                Feature::C3(lag) => {
                    let l_idx = unique_c3_lags.iter().position(|&l| l == *lag).unwrap();
                    let l = *lag as usize;
                    if values.len() > 2 * l {
                        state.c3_sums[l_idx] / (values.len() - 2 * l) as f32
                    } else {
                        0.0
                    }
                }
                Feature::Paa(total, index) => {
                    let t_idx = unique_paa_totals
                        .iter()
                        .position(|&t| t == *total)
                        .unwrap();
                    let b = &paa_boundaries[t_idx];
                    let start = b[*index as usize];
                    let end = b[*index as usize + 1];
                    if start < end {
                        state.paa_sums[t_idx][*index as usize] / (end - start) as f32
                    } else {
                        0.0
                    }
                }
                Feature::FftCoefficient(coeff, attr) => {
                    let k = *coeff as usize;
                    let (re, im) = if !fft_complex.is_empty() && k < fft_complex.len() {
                        (fft_complex[k].re, fft_complex[k].im)
                    } else {
                        (0.0, 0.0) // Fallback if no FFT computed
                    };
                    match attr {
                        crate::types::FftAttr::Real => re,
                        crate::types::FftAttr::Imag => im,
                        crate::types::FftAttr::Abs => (re * re + im * im).sqrt(),
                        crate::types::FftAttr::Angle => im.atan2(re).to_degrees(),
                    }
                }
                Feature::SpectralCentroid => freq_centroid,
                Feature::SpectralDistance => {
                    if !spectrum.is_empty() {
                        let m = spectrum.iter().sum::<f32>() / spectrum.len() as f32;
                        spectrum
                            .iter()
                            .map(|&s| (s - m).powi(2))
                            .sum::<f32>()
                            .sqrt()
                    } else {
                        0.0
                    }
                }
                Feature::SpectralDecrease => spectral_decrease,
                Feature::SpectralSlope => spectral_slope,
                Feature::SpectrogramCoefficients(_, f_bits) => {
                    if !spectrum.is_empty() {
                        let target_freq = f32::from_bits(*f_bits);
                        let fs = 100.0;
                        let n_fft = (spectrum.len() - 1) * 2;
                        let freq_step = fs / n_fft as f32;
                        let idx = (target_freq / freq_step).round() as usize;
                        let idx = idx.min(spectrum.len() - 1);
                        spectrum[idx]
                    } else {
                        0.0
                    }
                }
                Feature::WaveletFeatures(_w_bits, f_type) => {
                    if values.len() >= 2 {
                        let mut sum = 0.0;
                        for i in (0..values.len() - 1).step_by(2) {
                            if *f_type == 0 {
                                sum += (values[i] - values[i + 1]).abs();
                            } else {
                                sum += (values[i] - values[i + 1]).powi(2);
                            }
                        }
                        if *f_type == 0 {
                            sum / (values.len() / 2) as f32
                        } else {
                            (sum / (values.len() / 2) as f32).sqrt()
                        }
                    } else {
                        0.0
                    }
                }
        _ => return None,
    };
    Some(res)
}
