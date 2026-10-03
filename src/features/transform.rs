use crate::types::Feature;

#[inline(always)]
pub fn eval_transform(
    feat: &Feature,
    context: &mut crate::context::FeatureContext,
) -> Option<f32> {
    let values = context.values;
    let state = &mut *context.state;
    let _n = context.n;
    let _mean = context.mean;
    let _m2 = context.m2;
    let _m3 = context.m3;
    let _m4 = context.m4;
    let _var = context.var;
    let _std_dev = context.std_dev;
    let _mac_sum = context.mac_sum;
    let _mc_sum = context.mc_sum;
    let _first_max_idx = context.first_max_idx;
    let _last_max_idx = context.last_max_idx;
    let _first_min_idx = context.first_min_idx;
    let _last_min_idx = context.last_min_idx;
    let _median = context.median;
    let _iqr = context.iqr;
    let _entropy = context.entropy;
    let _mad_sum = context.mad_sum;
    let _count_a = context.count_a;
    let _count_b = context.count_b;
    let _max_strike_a = context.max_strike_a;
    let _max_strike_b = context.max_strike_b;
    let _zc_mean = context.zc_mean;
    let _zc_std = context.zc_std;
    let freq_centroid = context.freq_centroid;
    let spectral_decrease = context.spectral_decrease;
    let spectral_slope = context.spectral_slope;
    let spectral_spread = context.spectral_spread;
    let spectral_entropy = context.spectral_entropy;
    let _fft_autocorr = context.fft_autocorr;
    let fft_complex = context.fft_complex;
    let spectrum = context.spectrum;
    let unique_c3_lags = context.unique_c3_lags;
    let unique_paa_totals = context.unique_paa_totals;
    let paa_boundaries = context.paa_boundaries;

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
                Feature::MaxPowerSpectrum => {
            if spectrum.is_empty() {
                0.0
            } else {
                spectrum.iter().map(|&s| s * s).fold(0.0, f32::max)
            }
        },
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
                Feature::SpectralSpread => spectral_spread,
                Feature::SpectralEntropy => spectral_entropy,
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
