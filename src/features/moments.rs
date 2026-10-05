use crate::types::Feature;

#[inline(always)]
pub fn eval_moments(feat: &Feature, context: &mut crate::context::FeatureContext) -> Option<f32> {
    let _values = context.values;
    let state = &mut *context.state;
    let n = context.n;
    let mean = context.mean;
    let m2 = context.m2;
    let m3 = context.m3;
    let m4 = context.m4;
    let var = context.var;
    let std_dev = context.std_dev;
    let _mac_sum = context.mac_sum;
    let _mc_sum = context.mc_sum;
    let _first_max_idx = context.first_max_idx;
    let _last_max_idx = context.last_max_idx;
    let _first_min_idx = context.first_min_idx;
    let _last_min_idx = context.last_min_idx;
    let _median = context.median;
    let iqr = context.iqr;
    let _entropy = context.entropy;
    let mad_sum = context.mad_sum;
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
        Feature::TotalSum => state.total_sum,
        Feature::Mean => mean,
        Feature::Variance => var,
        Feature::Std => std_dev,
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
        Feature::Mad => mad_sum,
        Feature::Iqr => iqr,
        Feature::VariationCoefficient if mean.abs() > 1e-9 => std_dev / mean,
        // tsfresh: np.std(x) (ddof=0) > r * (max - min)
        Feature::LargeStandardDeviation(r_bits) => {
            let r = f32::from_bits(*r_bits);
            let pop_std = (m2 / n).max(0.0).sqrt();
            (pop_std > r * (state.max_value - state.min_value)) as u8 as f32
        }
        _ => return None,
    };
    Some(res)
}
