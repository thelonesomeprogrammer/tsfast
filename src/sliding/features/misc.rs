use crate::types::Feature;

#[inline(always)]
pub fn eval_misc(
    feat: &Feature,
    context: &mut crate::sliding::context::FeatureContext,
) -> Option<f32> {
    let _values = context.values;
    let _state = &mut *context.state;
    let n = context.n;
    let _mean = context.mean;
    let _m2 = context.m2;
    let _m3 = context.m3;
    let _m4 = context.m4;
    let var = context.var;
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
    let _spectrum = context.spectrum;
    let _unique_c3_lags = context.unique_c3_lags;
    let _unique_paa_totals = context.unique_paa_totals;
    let _paa_boundaries = context.paa_boundaries;
    
    let res = match feat {
                Feature::Length => n as f32,
                Feature::VarianceLargerThanStandardDeviation => {
                    if var > 1.0 {
                        1.0
                    } else {
                        0.0
                    }
                }
        _ => return None,
    };
    Some(res)
}
