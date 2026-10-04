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
            let t_idx = unique_paa_totals.iter().position(|&t| t == *total).unwrap();
            let b = &paa_boundaries[t_idx];
            let start = b[*index as usize];
            let end = b[*index as usize + 1];
            if start < end {
                state.paa_sums[t_idx][*index as usize] / (end - start) as f32
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
        Feature::CwtCoefficients(_widths, _len, coeff, w) => {
            let scale = *w as f32;
            let n = values.len();
            if n == 0 {
                return Some(std::f32::NAN);
            }

            // Check if wavelet is cached
            let mut int_psi_scale =
                context
                    .state
                    .cwt_wavelets
                    .get(w)
                    .cloned()
                    .unwrap_or_else(|| {
                        let num_points = 4096;
                        let x_start = -8.0_f32;
                        let step = 16.0 / (num_points as f32);
                        let constant = 2.0 / (3.0_f32.sqrt() * std::f32::consts::PI.powf(0.25));
                        let mut int_psi = vec![0.0; num_points];
                        let mut integral = 0.0;
                        for i in 0..num_points {
                            let x = x_start + (i as f32) * step;
                            let x_sq = x * x;
                            let val = constant * (1.0 - x_sq) * (-x_sq / 2.0).exp();
                            integral += val * step;
                            int_psi[i] = integral;
                        }
                        let j_max = (scale * 16.0).floor() as usize + 1;
                        let mut scale_arr = Vec::with_capacity(j_max);
                        for i in 0..j_max {
                            let j_val = (i as f32 / (scale * step)).floor() as usize;
                            if j_val < int_psi.len() {
                                scale_arr.push(int_psi[j_val]);
                            } else {
                                break;
                            }
                        }
                        scale_arr.reverse();

                        scale_arr
                    });
            // We cache it (state is mutable inside eval_transform conceptually but we can't mutate it easily without RefCell.
            // Wait, we have `&'a mut ColumnState`. BUT context.state is `&mut ColumnState`.
            // We can mutate it!
            if !context.state.cwt_wavelets.contains_key(w) {
                context.state.cwt_wavelets.insert(*w, int_psi_scale.clone());
            }

            let c = *coeff as usize;
            if c >= n {
                return Some(std::f32::NAN);
            }

            let conv_len = n + int_psi_scale.len() - 1;
            let coef_len = conv_len - 1;
            let d = (coef_len as f32 - n as f32) / 2.0;
            let start_idx = d.floor() as usize;
            let target_coef_idx = start_idx + c;

            let mut conv_k = 0.0;
            let mut conv_k_plus_1 = 0.0;
            let k = target_coef_idx;

            for m in 0..n {
                let j_k = k as isize - m as isize;
                if j_k >= 0 && (j_k as usize) < int_psi_scale.len() {
                    conv_k += values[m] * int_psi_scale[j_k as usize];
                }
                let j_kp1 = k as isize + 1 - m as isize;
                if j_kp1 >= 0 && (j_kp1 as usize) < int_psi_scale.len() {
                    conv_k_plus_1 += values[m] * int_psi_scale[j_kp1 as usize];
                }
            }

            let diff = conv_k_plus_1 - conv_k;
            let coef = -scale.sqrt() * diff;
            coef * 1.421711
        }
        Feature::NumberCwtPeaks(n_val) => {
            let n = values.len();
            if n == 0 {
                return Some(0.0);
            }

            let max_w = *n_val as usize;
            let mut all_peaks = Vec::new();

            for w_idx in 1..=max_w {
                let w = w_idx as f32;
                let vec_len = (10.0 * w).min(n as f32) as usize;
                let vec_len = if vec_len == 0 { 1 } else { vec_len };
                let wavelet_len = 2 * vec_len + 1;
                let mut ricker = vec![0.0; wavelet_len];
                let constant = 2.0 / ((3.0 * w).sqrt() * std::f32::consts::PI.powf(0.25));
                for i in 0..wavelet_len {
                    let x = (i as f32) - (vec_len as f32);
                    let x_a_sq = (x / w) * (x / w);
                    ricker[i] = constant * (1.0 - x_a_sq) * (-x_a_sq / 2.0).exp();
                }
                ricker.reverse();

                let mut conv = vec![0.0; n];
                for i in 0..n {
                    let mut sum = 0.0;
                    for j in 0..wavelet_len {
                        let data_idx = i as isize + j as isize - vec_len as isize;
                        if data_idx >= 0 && data_idx < n as isize {
                            sum += values[data_idx as usize] * ricker[j];
                        }
                    }
                    conv[i] = sum;
                }

                let mut peaks = Vec::new();
                let mut i = 1;
                while i < n - 1 {
                    if conv[i] > conv[i - 1] {
                        let mut j = i;
                        while j < n - 1 && conv[j] == conv[i] {
                            j += 1;
                        }
                        if conv[i] > conv[j] {
                            peaks.push((i + j - 1) / 2);
                        }
                        i = j;
                    } else {
                        i += 1;
                    }
                }
                all_peaks.push(peaks);
            }

            if all_peaks.is_empty() {
                return Some(0.0);
            }
            let base_peaks = &all_peaks[0];
            let mut final_count = 0;
            for &p in base_peaks {
                let mut found_in_other_scales = 0;
                for peaks in all_peaks.iter().skip(1) {
                    if peaks
                        .iter()
                        .any(|&p2| (p as isize - p2 as isize).abs() <= max_w as isize)
                    {
                        found_in_other_scales += 1;
                    }
                }
                if found_in_other_scales > 0 {
                    final_count += 1;
                }
            }

            final_count as f32
        }
        Feature::MaxPowerSpectrum => {
            if spectrum.is_empty() {
                0.0
            } else {
                spectrum.iter().map(|&s| s * s).fold(0.0, f32::max)
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
        Feature::SpectralRollOn => spectral_roll_on,
        Feature::SpectralRollOff => spectral_roll_off,
        Feature::SpectralSpread => spectral_spread,
        Feature::SpectralSkewness => spectral_skewness,
        Feature::SpectralKurtosis => spectral_kurtosis,
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
        Feature::SpectralCentroid => freq_centroid,
        Feature::CalcCentroid(fs_bits) => {
            let fs = f32::from_bits(*fs_bits);
            if state.energy == 0.0 || fs == 0.0 {
                0.0
            } else {
                (state.t_energy / fs) / state.energy
            }
        }
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
