use crate::types::Feature;

/// Number of points strictly greater than every neighbour within `p` on both
/// sides (tsfresh `number_peaks`, TSFEL `neighbourhood_peaks`).
///
/// Linear-ish instead of O(len * p): a failed right scan at offset `j` means
/// everything between `i` and `i + j` is below `values[i]` with `i` in its
/// neighbourhood, so none of it can be a peak and we jump straight to `i + j`.
/// `known` remembers the stretch left of `i` already proven smaller than
/// `values[i]`, so monotone runs don't rescan the left side each step.
fn count_n_peaks(values: &[f32], p: usize) -> usize {
    let len = values.len();
    if p == 0 || len <= 2 * p {
        return 0;
    }
    let end = len - p;
    let mut count = 0;
    let mut i = p;
    // Invariant: values[known..i] < values[i].
    let mut known = i;
    while i < end {
        let v = values[i];
        let lo = i - p;
        // Nearest neighbours first: they're the likeliest to fail.
        if !values[lo..known.clamp(lo, i)].iter().rev().all(|&x| x < v) {
            i += 1;
            known = i;
            continue;
        }
        // Now values[lo..i] < v.
        match values[i + 1..=i + p].iter().position(|&x| !(x < v)) {
            Some(off) => {
                let next = i + 1 + off;
                known = if values[next] > v { lo } else { next };
                i = next;
            }
            None => {
                count += 1;
                // values[i+1..=i+p] < v, and v is in each one's neighbourhood.
                i += p + 1;
                known = i;
            }
        }
    }
    count
}

#[inline(always)]
pub fn eval_crossings_peaks(
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
    let _mad_sum = context.mad_sum;
    let _count_a = context.count_a;
    let _count_b = context.count_b;
    let _max_strike_a = context.max_strike_a;
    let _max_strike_b = context.max_strike_b;
    let zc_mean = context.zc_mean;
    let zc_std = context.zc_std;
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
        Feature::ZeroCrossingRate => state.zcr_count as f32 / n,
        Feature::ZeroCross => state.zcr_count as f32,
        Feature::PeakCount => state.peaks as f32,
        Feature::NegativeTurning => state.troughs as f32,
        Feature::PositiveTurning => state.peaks as f32,
        // Strict local maxima + strict local minima.
        Feature::TurningPoints => (state.peaks + state.troughs) as f32,
        // EMG slope sign change (threshold 0): points where the slope changes
        // sign or flattens, (x[i] - x[i-1]) * (x[i] - x[i+1]) >= 0.
        Feature::SlopeSignChange => {
            let values = context.values;
            values
                .windows(3)
                .filter(|w| (w[1] - w[0]) * (w[1] - w[2]) >= 0.0)
                .count() as f32
        }
        Feature::NumberCrossingM(m_bits) => {
            let mut count = 0;
            let m = f32::from_bits(*m_bits);
            let values = context.values;
            if values.len() > 1 {
                for i in 1..values.len() {
                    if (values[i] > m) != (values[i - 1] > m) {
                        count += 1;
                    }
                }
            }
            count as f32
        }
        Feature::NumberPeaks(peak_n) => count_n_peaks(context.values, *peak_n as usize) as f32,
        Feature::ZeroCrossingMean => zc_mean,
        Feature::ZeroCrossingStd => zc_std,
        _ => return None,
    };
    Some(res)
}
