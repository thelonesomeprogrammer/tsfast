//! Everything computed from a window's DFT or raw values for the frequency-
//! domain features. Shared by all three engines; each engine only decides how
//! it obtains the DFT (fresh FFT, sliding DFT, or a cached one).

use std::simd::cmp::SimdPartialOrd;
use std::simd::f32x4;
use std::simd::num::SimdFloat;

use realfft::num_complex::Complex;

use crate::common::ColumnState;
use crate::metrics::FftResult;
use crate::types::Compute;

/// Sampling frequency TSFEL's spectral features are evaluated at.
pub const FS: f32 = 100.0;

/// Exact-length real DFT of `values`, as `np.fft.rfft` (no zero padding: padding
/// changes the bins and every TSFEL spectral feature with them).
pub fn rfft(
    values: &[f32],
    r2c: &dyn realfft::RealToComplex<f32>,
    state: &mut ColumnState,
) -> Result<Vec<Complex<f32>>, String> {
    debug_assert_eq!(r2c.len(), values.len());
    let mut indata = std::mem::take(&mut state.fft_in_buffer);
    indata.clear();
    indata.extend_from_slice(values);
    let mut out = r2c.make_output_vec();
    let res = r2c
        .process(&mut indata, &mut out)
        .map_err(|e| e.to_string());
    state.fft_in_buffer = indata;
    res.map(|_| out)
}

pub fn finalize(
    compute: Compute,
    values: &[f32],
    n: f32,
    mean: f32,
    m2: f32,
    fft_complex: Vec<Complex<f32>>,
    dft_len: usize,
    state: &mut ColumnState,
) -> Result<FftResult, String> {
    // TSFEL frequencies: np.fft.rfftfreq(dft_len, 1 / fs), fs = 100.
    let freq_step = if dft_len > 0 {
        FS / dft_len as f32
    } else {
        0.0
    };

    let mut spectrum = std::mem::take(&mut state.spectrum_buffer);
    spectrum.clear();
    spectrum.extend(fft_complex.iter().map(|c| c.norm()));

    let mut freq_centroid = 0.0;
    let mut spectral_decrease = 0.0;
    let mut spectral_slope = 0.0;
    let mut spectral_spread = 0.0;
    let mut spectral_entropy = 0.0;
    let mut spectral_roll_on = 0.0;
    let mut spectral_roll_off = 0.0;
    let mut spectral_skewness = 0.0;
    let mut spectral_kurtosis = 0.0;
    let mut max_frequency = 0.0;
    let mut median_frequency = 0.0;
    let mut fundamental_frequency = 0.0;

    let mut fft_autocorr = Vec::new();

    if compute.intersects(Compute::FULL_AUTOCORR | Compute::PACF) && n > 1.0 {
        let n2 = values.len() * 2;
        let fft_size_ac = crate::common::next_good_fft_size(n2);
        let (r2c_ac, c2r_ac) = PLANNER.with_borrow_mut(|p| {
            (
                p.plan_fft_forward(fft_size_ac),
                p.plan_fft_inverse(fft_size_ac),
            )
        });

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
                .map(|&v| v * scale)
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

            if compute.intersects(Compute::SPEC_ROLLON | Compute::SPEC_ROLLOFF | Compute::MAX_FREQ | Compute::MEDIAN_FREQ | Compute::SPEC_DECREASE | Compute::SPEC_SLOPE) && spectrum.len() > 1 {
                let spec_sum_no_first = m0 - spectrum[0];
                if spec_sum_no_first > 0.0 {
                    let mut sd_num = 0.0;
                    for (i, &mag) in spectrum[1..].iter().enumerate() {
                        sd_num += (mag - spectrum[0]) / (i + 1) as f32;
                    }
                    spectral_decrease = sd_num / spec_sum_no_first;
                }

                let m_n = spectrum.len() as f32;

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
                let median_thresh = 0.50 * sum_y;

                let mut cumsum = 0.0;
                let mut roll_on_idx = -1.0;
                let mut roll_off_idx = -1.0;
                let mut median_idx = -1.0;

                let mut j = 0;
                while j + 4 <= spectrum.len() {
                    let mag = f32x4::from_slice(&spectrum[j..j + 4]);
                    let block_sum = mag.reduce_sum();

                    // We check if any threshold is hit in this block
                    let hit_on = roll_on_idx < 0.0 && cumsum + block_sum >= roll_on_thresh;
                    let hit_med = median_idx < 0.0 && cumsum + block_sum > median_thresh;
                    let hit_off = roll_off_idx < 0.0 && cumsum + block_sum >= roll_off_thresh;

                    if hit_on || hit_med || hit_off {
                        for k in 0..4 {
                            cumsum += spectrum[j + k];
                            if roll_on_idx < 0.0 && cumsum >= roll_on_thresh {
                                roll_on_idx = (j + k) as f32;
                            }
                            if median_idx < 0.0 && cumsum > median_thresh {
                                median_idx = (j + k) as f32;
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
                    if median_idx < 0.0 && cumsum > median_thresh {
                        median_idx = j as f32;
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
                if median_idx >= 0.0 {
                    median_frequency = median_idx * freq_step;
                }
                if roll_off_idx >= 0.0 {
                    spectral_roll_off = roll_off_idx * freq_step;
                    // match TSFEL logic for max frequency (which uses identical cumsum 0.95 crossing threshold fallback)
                    max_frequency = roll_off_idx * freq_step;
                }
            }

            if compute.intersects(Compute::FUNDAMENTAL_FREQ) && spectrum.len() > 2 {
                // Find max ignoring DC component using SIMD max reduction
                let mut max_mag_vec = f32x4::splat(0.0);
                let mut i = 1;
                while i + 4 <= spectrum.len() {
                    max_mag_vec = max_mag_vec.simd_max(f32x4::from_slice(&spectrum[i..i + 4]));
                    i += 4;
                }
                let mut max_mag = max_mag_vec.reduce_max();
                while i < spectrum.len() {
                    if spectrum[i] > max_mag {
                        max_mag = spectrum[i];
                    }
                    i += 1;
                }

                let threshold = max_mag * 0.3;
                let mut min_bp = usize::MAX;

                let thresh_vec = f32x4::splat(threshold);

                // For i=1 edge case
                if spectrum[1] > threshold && 0.0 < spectrum[1] && spectrum[1] >= spectrum[2] {
                    min_bp = 1;
                } else {
                    let mut i = 2;
                    while i + 4 <= spectrum.len() - 1 {
                        let prev = f32x4::from_slice(&spectrum[i - 1..i + 3]);
                        let mag = f32x4::from_slice(&spectrum[i..i + 4]);
                        let next = f32x4::from_slice(&spectrum[i + 1..i + 5]);

                        let mask = mag.simd_gt(thresh_vec) & prev.simd_lt(mag) & mag.simd_ge(next);
                        if mask.any() {
                            min_bp = i + mask.first_set().unwrap();
                            break;
                        }
                        i += 4;
                    }

                    if min_bp == usize::MAX {
                        while i < spectrum.len() - 1 {
                            let mag = spectrum[i];
                            if mag > threshold && spectrum[i - 1] < mag && mag >= spectrum[i + 1] {
                                min_bp = i;
                                break;
                            }
                            i += 1;
                        }
                    }
                }

                if min_bp != usize::MAX {
                    fundamental_frequency = min_bp as f32 * freq_step;
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

                // TSFEL: entropy of the power probabilities of the mean-removed
                // signal, normalised by log2(count of non-zero probabilities).
                // Its DC bin is float residue (~1e-30) rather than exactly 0 for
                // almost all inputs, so it adds nothing to the sum but counts.
                if p_sum > 0.0 {
                    let mut entropy_sum = 0.0;
                    let mut nonzero = 1usize; // the DC residue
                    for &power in &power_vals {
                        if power > 0.0 {
                            let p = power / p_sum;
                            entropy_sum += p * p.log2();
                            nonzero += 1;
                        }
                    }
                    if nonzero > 1 {
                        spectral_entropy = -entropy_sum / (nonzero as f32).log2();
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
        lpcc = crate::features::lpc::compute_lpcc(values);
    }

    if compute.intersects(Compute::MFCC) {
        mfcc = tsfel_mfcc(values)?;
    }

    if compute.intersects(Compute::CWT_MEXH) {
        // TSFEL wavelet_energy / wavelet_entropy: pywt.cwt(x, 1..10, "mexh").
        cwt_energy = vec![0.0; 9];
        if !values.is_empty() {
            let n_vals = values.len() as f64;
            let mut abs_sums = Vec::with_capacity(9);
            for (i, scale) in (1..10).enumerate() {
                let row = crate::features::cwt::mexh_cwt(values, scale as f64);
                cwt_energy[i] = (row.iter().map(|c| c * c).sum::<f64>() / n_vals).sqrt() as f32;
                abs_sums.push(row.iter().map(|c| c.abs()).sum::<f64>());
            }
            let total: f64 = abs_sums.iter().sum();
            // TSFEL returns 0 when sum(signal) == 0.
            if total > 0.0 && values.iter().map(|&v| v as f64).sum::<f64>() != 0.0 {
                cwt_entropy = -abs_sums
                    .iter()
                    .map(|&e| e / total)
                    .filter(|&p| p > 0.0)
                    .map(|p| p * p.ln())
                    .sum::<f64>() as f32;
            }
        }
    }

    let mut welch_density = Vec::new();
    if compute.intersects(Compute::WELCH) && !values.is_empty() {
        // tsfresh spkt_welch_density: scipy.signal.welch(x, nperseg=min(n, 256)), fs = 1.
        welch_density = welch_psd(values, values.len().min(256), 1.0)?;
    }

    let cwt_peaks = 0;

    // Moments above are in bins; TSFEL reports Hz.
    freq_centroid *= freq_step;
    spectral_spread *= freq_step;

    Ok(FftResult {
        dft_len,
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
        max_frequency,
        median_frequency,
        fundamental_frequency,
        fft_autocorr,
        mfcc,
        lpcc,
        cwt_energy,
        cwt_entropy,
    })
}

thread_local! {
    /// Planning an FFT costs far more than running a short one, and the planner
    /// caches its plans, so keep one per (rayon worker) thread for the process
    /// lifetime instead of one per column.
    static PLANNER: std::cell::RefCell<realfft::RealFftPlanner<f32>> =
        std::cell::RefCell::new(realfft::RealFftPlanner::new());
}

fn plan_forward(len: usize) -> std::sync::Arc<dyn realfft::RealToComplex<f32>> {
    PLANNER.with_borrow_mut(|p| p.plan_fft_forward(len))
}

/// scipy.signal.welch(values, fs, nperseg) with its defaults: periodic Hann
/// window, 50% overlap, constant detrend per segment, one-sided density.
pub fn welch_psd(values: &[f32], nperseg: usize, fs: f32) -> Result<Vec<f32>, String> {
    let n = values.len();
    if nperseg == 0 || n < nperseg {
        return Ok(Vec::new());
    }
    let r2c = plan_forward(nperseg);
    let window: Vec<f32> = (0..nperseg)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / nperseg as f32).cos())
        .collect();
    let scale = 1.0 / (fs * window.iter().map(|w| w * w).sum::<f32>());
    let step = nperseg - nperseg / 2;
    let n_segments = (n - nperseg / 2) / step;

    let mut psd = vec![0.0f32; nperseg / 2 + 1];
    let mut indata = vec![0.0f32; nperseg];
    let mut outdata = r2c.make_output_vec();
    for seg in 0..n_segments {
        let chunk = &values[seg * step..seg * step + nperseg];
        let mean = chunk.iter().sum::<f32>() / nperseg as f32;
        for ((d, &v), &w) in indata.iter_mut().zip(chunk).zip(&window) {
            *d = (v - mean) * w;
        }
        r2c.process(&mut indata, &mut outdata)
            .map_err(|e| e.to_string())?;
        for (p, c) in psd.iter_mut().zip(&outdata) {
            *p += c.norm_sqr() * scale;
        }
    }
    // One-sided: double all bins except DC and (for even nperseg) Nyquist.
    let last = if nperseg % 2 == 0 {
        psd.len() - 1
    } else {
        psd.len()
    };
    for p in &mut psd[1..last] {
        *p *= 2.0;
    }
    for p in &mut psd {
        *p /= n_segments.max(1) as f32;
    }
    Ok(psd)
}

/// TSFEL mfcc(signal, fs=100): pre-emphasis 0.97, 512-point power spectrum,
/// 40 mel filters in dB, orthonormal DCT-II coefficients 1..=12, mean-centred,
/// sinusoidal lifter 22.
fn tsfel_mfcc(values: &[f32]) -> Result<Vec<f32>, String> {
    const NFFT: usize = 512;
    const NFILT: usize = 40;
    const NUM_CEPS: usize = 12;
    if values.is_empty() {
        return Ok(vec![0.0; NUM_CEPS]);
    }
    // np.fft.rfft(emphasized, 512): truncates or zero-pads to 512.
    let mut indata = vec![0.0f32; NFFT];
    for (i, d) in indata.iter_mut().enumerate().take(values.len()) {
        *d = if i == 0 {
            values[0]
        } else {
            values[i] - 0.97 * values[i - 1]
        };
    }
    let r2c = plan_forward(NFFT);
    let mut out = r2c.make_output_vec();
    r2c.process(&mut indata, &mut out)
        .map_err(|e| e.to_string())?;
    let pow: Vec<f64> = out
        .iter()
        .map(|c| c.norm_sqr() as f64 / NFFT as f64)
        .collect();

    let fs = FS as f64;
    let high_mel = 2595.0 * (1.0 + (fs / 2.0) / 700.0).log10();
    let hz: Vec<f64> = (0..NFILT + 2)
        .map(|i| {
            let mel = high_mel * i as f64 / (NFILT + 1) as f64;
            700.0 * (10f64.powf(mel / 2595.0) - 1.0)
        })
        .collect();
    let bin: Vec<f64> = hz
        .iter()
        .map(|h| ((NFFT + 1) as f64 * h / fs).floor())
        .collect();

    let mut banks = [0.0f64; NFILT];
    for m in 1..=NFILT {
        let (lo, mid, hi) = (bin[m - 1], bin[m], bin[m + 1]);
        let enorm = 2.0 / (hz[m + 1] - hz[m - 1]);
        let mut acc = 0.0;
        for k in lo as usize..mid as usize {
            acc += pow[k] * (k as f64 - lo) / (mid - lo);
        }
        for k in mid as usize..hi as usize {
            acc += pow[k] * (hi - k as f64) / (hi - mid);
        }
        let e = acc * enorm;
        banks[m - 1] = 20.0 * (if e == 0.0 { f64::EPSILON } else { e }).log10();
    }

    let ortho = (2.0 / NFILT as f64).sqrt();
    let mut ceps: Vec<f64> = (1..=NUM_CEPS)
        .map(|k| {
            ortho
                * banks
                    .iter()
                    .enumerate()
                    .map(|(i, b)| {
                        b * (std::f64::consts::PI * k as f64 * (2 * i + 1) as f64
                            / (2 * NFILT) as f64)
                            .cos()
                    })
                    .sum::<f64>()
        })
        .collect();
    let mean = ceps.iter().sum::<f64>() / NUM_CEPS as f64 + 1e-8;
    for (i, c) in ceps.iter_mut().enumerate() {
        let lift = 1.0 + 11.0 * (std::f64::consts::PI * i as f64 / 22.0).sin();
        *c = (*c - mean) * lift;
    }
    Ok(ceps.into_iter().map(|c| c as f32).collect())
}
