use crate::types::Feature;

#[inline(always)]
pub fn eval_energy(
    feat: &Feature,
    context: &mut crate::context::FeatureContext,
) -> Option<f32> {
    let _values = context.values;
    let state = &mut *context.state;
    let n = context.n;
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
    let _freq_centroid = context.freq_centroid;
    let _spectral_decrease = context.spectral_decrease;
    let _spectral_slope = context.spectral_slope;
    let _fft_autocorr = context.fft_autocorr;
    let _fft_complex = context.fft_complex;
    let spectrum = context.spectrum;
    let _unique_c3_lags = context.unique_c3_lags;
    let _unique_paa_totals = context.unique_paa_totals;
    let _paa_boundaries = context.paa_boundaries;

    let res = match feat {
                Feature::Energy => state.energy as f32,
                Feature::Rms | Feature::RootMeanSquare => (state.energy / n).sqrt(),
                Feature::HumanRangeEnergy(fs_bits) => {
                    if !spectrum.is_empty() {
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
                            range_energy / total_energy
                        } else {
                            0.0
                        }
                    } else {
                        0.0
                    }
                }
        _ => return None,
    };
    Some(res)
}
