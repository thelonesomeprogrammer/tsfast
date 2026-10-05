use crate::types::Feature;

#[inline(always)]
pub fn eval_transform(feat: &Feature, context: &mut crate::context::FeatureContext) -> Option<f32> {
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
    let spectral_roll_on = context.spectral_roll_on;
    let spectral_roll_off = context.spectral_roll_off;
    let spectral_spread = context.spectral_spread;
    let spectral_skewness = context.spectral_skewness;
    let spectral_kurtosis = context.spectral_kurtosis;
    let _fft_autocorr = context.fft_autocorr;
    let mfcc = context.mfcc;
    let lpcc = context.lpcc;
    let cwt_energy = context.cwt_energy;
    let cwt_entropy = context.cwt_entropy;
    let fft_complex = context.fft_complex;
    let spectrum = context.spectrum;
    let unique_c3_lags = context.unique_c3_lags;
    let _unique_paa_totals = context.unique_paa_totals;
    let _paa_boundaries = context.paa_boundaries;

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
        // Mean of segment `index` of `total` equal-width segments; segment i
        // covers [i * n / total, (i + 1) * n / total). Same in every engine.
        Feature::Paa(total, index) => {
            let n_vals = values.len();
            let start = *index as usize * n_vals / *total as usize;
            let end = (*index as usize + 1) * n_vals / *total as usize;
            if start < end {
                values[start..end].iter().sum::<f32>() / (end - start) as f32
            } else {
                0.0
            }
        }

        Feature::SpktWelchDensity(coeff) => {
            let k = *coeff as usize;
            if k < context.welch_density.len() {
                context.welch_density[k]
            } else {
                0.0
            }
        }
        // tsfresh: pywt.cwt(x, widths, "mexh")[widths.index(w), coeff].
        Feature::CwtCoefficients(_widths, _len, coeff, w) => {
            let c = *coeff as usize;
            if c >= values.len() {
                return Some(f32::NAN);
            }
            super::cwt::mexh_cwt(values, *w as f64)[c] as f32
        }
        Feature::NumberCwtPeaks(n_val) => super::cwt::number_cwt_peaks(values, *n_val as usize) as f32,
        // TSFEL: max(scipy.signal.welch(x / std(x), fs, nperseg=len(x))[1]).
        Feature::MaxPowerSpectrum => {
            let n_vals = values.len() as f64;
            let mean = values.iter().map(|&v| v as f64).sum::<f64>() / n_vals;
            let std = (values.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n_vals).sqrt();
            let scaled: Vec<f32> = if std > 0.0 {
                values.iter().map(|&v| (v as f64 / std) as f32).collect()
            } else {
                values.to_vec()
            };
            let psd = crate::spectral::welch_psd(&scaled, scaled.len(), crate::spectral::FS)
                .unwrap_or_default();
            psd.into_iter().fold(0.0, f32::max)
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
        // TSFEL: sum(linspace(0, cumsum[-1], len) - cumsum(|X|)).
        Feature::SpectralDistance => {
            let len = spectrum.len();
            if len > 1 {
                let mut cum = 0.0f64;
                let cums: Vec<f64> = spectrum.iter().map(|&s| { cum += s as f64; cum }).collect();
                let total = cum;
                cums.iter()
                    .enumerate()
                    .map(|(i, &c)| total * i as f64 / (len - 1) as f64 - c)
                    .sum::<f64>() as f32
            } else {
                0.0
            }
        }
        Feature::SpectralDecrease => spectral_decrease,
        Feature::SpectralSlope => spectral_slope,
        Feature::SpectralRollOn => spectral_roll_on,
        Feature::SpectralRollOff => spectral_roll_off,
        Feature::SpectralSpread => spectral_spread,
        Feature::SpectralSkewness => spectral_skewness,
        Feature::SpectralKurtosis => spectral_kurtosis,
        Feature::SpectrogramCoefficients(_, f_bits) => {
            if !spectrum.is_empty() {
                let target_freq = f32::from_bits(*f_bits);
                let freq_step = crate::spectral::FS / context.dft_len as f32;
                let idx = (target_freq / freq_step).round() as usize;
                let idx = idx.min(spectrum.len() - 1);
                spectrum[idx]
            } else {
                0.0
            }
        }
        Feature::SpectralEntropy => spectral_entropy,
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
        Feature::CalcCentroid(fs_bits) => {
            let fs = f32::from_bits(*fs_bits);
            if state.energy == 0.0 || fs == 0.0 {
                0.0
            } else {
                (state.t_energy / fs) / state.energy
            }
        }

        Feature::Mfcc(idx) => {
            if !mfcc.is_empty() && (*idx as usize) < mfcc.len() {
                mfcc[*idx as usize]
            } else {
                0.0
            }
        }
        Feature::Lpcc(idx) => {
            if !lpcc.is_empty() && (*idx as usize) < lpcc.len() {
                lpcc[*idx as usize]
            } else {
                0.0
            }
        }
        Feature::WaveletEnergy(idx) => {
            if !cwt_energy.is_empty() && (*idx as usize) < cwt_energy.len() {
                cwt_energy[*idx as usize]
            } else {
                0.0
            }
        }
        Feature::WaveletEntropy => cwt_entropy,
        _ => return None,
    };
    Some(res)
}
