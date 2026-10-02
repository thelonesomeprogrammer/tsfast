use crate::types::{Feature, AggAttr, AggFunc, FftAttr};
use crate::common::ColumnState;

#[inline(always)]
pub fn eval_crossings_peaks(
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
                Feature::ZeroCrossingRate => state.zcr_count as f32 / n,
                Feature::PeakCount => state.peaks as f32,
                Feature::ZeroCrossingMean => zc_mean,
                Feature::ZeroCrossingStd => zc_std,
        _ => return None,
    };
    Some(res)
}
