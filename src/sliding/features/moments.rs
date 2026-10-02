use crate::types::{Feature, AggAttr, AggFunc, FftAttr};
use crate::common::ColumnState;

#[inline(always)]
pub fn eval_moments(
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
                Feature::TotalSum => state.total_sum as f32,
                Feature::Mean => mean as f32,
                Feature::Variance => var as f32,
                Feature::Std => std_dev as f32,
                Feature::Skew if var > 1e-9 => {
                    let mu2 = m2 / n;
                    (m3 / n) / mu2.powf(1.5)
                }
                Feature::UnbiasedFisherKurtosis if var > 1e-9 && n > 3.0 => {
                    let mu2 = m2 / n;
                    let g2 = (m4 / n) / (mu2 * mu2) - 3.0;
                    ((n - 1.0) / ((n - 2.0) * (n - 3.0))) * ((n + 1.0) * g2 + 6.0)
                }
                Feature::BiasedFisherKurtosis if var > 1e-9 => {
                    let mu2 = m2 / n;
                    (m4 / n) / (mu2 * mu2) - 3.0
                }
                Feature::Mad => mad_sum as f32,
                Feature::Iqr => iqr,
                Feature::VariationCoefficient if mean.abs() > 1e-9 => (std_dev / mean) as f32,
        _ => return None,
    };
    Some(res)
}
