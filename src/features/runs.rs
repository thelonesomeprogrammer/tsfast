use crate::types::Feature;

#[inline(always)]
pub fn eval_runs(feat: &Feature, context: &mut crate::context::FeatureContext) -> Option<f32> {
    let _values = context.values;
    let _state = &mut *context.state;
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
    let count_a = context.count_a;
    let count_b = context.count_b;
    let max_strike_a = context.max_strike_a;
    let max_strike_b = context.max_strike_b;
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
    let unique_count_above_thresholds = context.unique_count_above_thresholds;
    let unique_count_below_thresholds = context.unique_count_below_thresholds;
    let unique_range_counts = context.unique_range_counts;

    let res = match feat {
        Feature::CountAboveMean => count_a as f32,
        Feature::CountBelowMean => count_b as f32,
        Feature::LongestStrikeAboveMean => max_strike_a as f32,
        Feature::LongestStrikeBelowMean => max_strike_b as f32,
        Feature::CountAbove(t_bits) => {
            let idx = if let Some(idx) = unique_count_above_thresholds
                .iter()
                .position(|&b| b == *t_bits)
            {
                idx
            } else {
                return Some(0.0);
            };
            context.state.count_above_counts[idx] as f32 / context.n
        }
        Feature::CountBelow(t_bits) => {
            let idx = if let Some(idx) = unique_count_below_thresholds
                .iter()
                .position(|&b| b == *t_bits)
            {
                idx
            } else {
                return Some(0.0);
            };
            context.state.count_below_counts[idx] as f32 / context.n
        }
        Feature::RangeCount(min_bits, max_bits) => {
            let idx = if let Some(idx) = unique_range_counts
                .iter()
                .position(|&(mn, mx)| mn == *min_bits && mx == *max_bits)
            {
                idx
            } else {
                return Some(0.0);
            };
            context.state.range_counts[idx] as f32
        }
        _ => return None,
    };
    Some(res)
}
