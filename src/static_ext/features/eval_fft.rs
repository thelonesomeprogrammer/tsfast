use crate::types::Feature;
use realfft::num_complex::Complex;

#[inline(always)]
pub fn eval_fft(
    feat: &Feature,
    n: f32,
    spectrum: &[f32],
    fft_complex: &[Complex<f32>],
    freq_centroid: f32,
    spectral_decrease: f32,
    spectral_slope: f32,
    values: &[f32],
) -> Option<f32> {
    match feat {
        Feature::SpectralCentroid => Some(freq_centroid),
        Feature::SpectralDecrease => Some(spectral_decrease),
        Feature::SpectralSlope => Some(spectral_slope),
        Feature::SpectralDistance if !spectrum.is_empty() => {
            let m = spectrum.iter().sum::<f32>() / spectrum.len() as f32;
            Some(spectrum
                .iter()
                .map(|&s| (s - m).powi(2))
                .sum::<f32>()
                .sqrt())
        }
        Feature::HumanRangeEnergy(fs_bits) if !spectrum.is_empty() => {
            let fs = f32::from_bits(*fs_bits);
            let n_fft = (spectrum.len() - 1) * 2;
            let freq_step = fs / n_fft as f32;
            let start_idx = (0.6 / freq_step).ceil() as usize;
            let end_idx = (2.5 / freq_step).floor() as usize;

            let total_energy: f32 = spectrum.iter().map(|&s| s * s).sum();
            if total_energy > 0.0 {
                let range_energy: f32 = spectrum
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i >= start_idx && *i <= end_idx)
                    .map(|(_, &s)| s * s)
                    .sum();
                Some(range_energy / total_energy)
            } else {
                Some(0.0)
            }
        }
        Feature::SpectrogramCoefficients(_t, f_bits) if !spectrum.is_empty() => {
            let target_freq = f32::from_bits(*f_bits);
            let fs = 100.0;
            let n_fft = (spectrum.len() - 1) * 2;
            let freq_step = fs / n_fft as f32;
            let idx = (target_freq / freq_step).round() as usize;
            let idx = idx.min(spectrum.len() - 1);
            Some(spectrum[idx])
        }
        Feature::FftCoefficient(coeff, attr) => {
            let k = *coeff as usize;
            let (re, im) = if !fft_complex.is_empty() && k < fft_complex.len() {
                (fft_complex[k].re, fft_complex[k].im)
            } else {
                let mut re = 0.0;
                let mut im = 0.0;
                let pi2 = 2.0 * std::f32::consts::PI;
                for (n_idx, &v) in values.iter().enumerate() {
                    let angle = pi2 * k as f32 * n_idx as f32 / n;
                    re += v * angle.cos();
                    im -= v * angle.sin();
                }
                (re, im)
            };
            Some(match attr {
                crate::types::FftAttr::Real => re,
                crate::types::FftAttr::Imag => im,
                crate::types::FftAttr::Abs => (re * re + im * im).sqrt(),
                crate::types::FftAttr::Angle => im.atan2(re).to_degrees(),
            })
        }
        Feature::WaveletFeatures(_w_bits, f_type) if values.len() >= 2 => {
            let mut sum = 0.0;
            for i in (0..values.len() - 1).step_by(2) {
                if *f_type == 0 {
                    sum += (values[i] - values[i + 1]).abs();
                } else {
                    sum += (values[i] - values[i + 1]).powi(2);
                }
            }
            if *f_type == 0 {
                Some(sum / (values.len() / 2) as f32)
            } else {
                Some((sum / (values.len() / 2) as f32).sqrt())
            }
        }
        _ => None,
    }
}
