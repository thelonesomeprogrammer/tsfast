use crate::types::Feature;

#[inline(always)]
pub fn eval_complexity(
    feat: &Feature,
    context: &mut crate::context::FeatureContext,
) -> Option<f32> {
    let values = context.values;
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
        Feature::ApproxEntropy(m, r_bits) if values.len() > *m as usize + 1 => {
            let m_val = *m as usize;
            // tsfresh: r is relative to the population std.
            let r = f32::from_bits(*r_bits) * population_std(values);
            (crate::common::approx_entropy_simd(m_val, r, values)
                - crate::common::approx_entropy_simd(m_val + 1, r, values))
            .abs()
        }
        Feature::SampleEntropy => {
            crate::common::sample_entropy_simd(values, population_std(values))
        }
        Feature::PermutationEntropy(tau, dimension) => {
            crate::common::permutation_entropy(values, *tau, *dimension)
        }
        Feature::HiguchiFd => calc_higuchi_fd(values),
        _ => return None,
    };
    Some(res)
}

#[inline(always)]
fn calc_higuchi_fd(values: &[f32]) -> f32 {
    let n = values.len();
    if n < 10 {
        // matching tsfel FEATURES_MIN_SIZE
        return f32::NAN;
    }

    // TSFEL: k in np.arange(1, n // 10), i.e. n / 10 excluded; polyfit needs 2 points.
    let k_max = n / 10 - 1;
    if k_max < 2 {
        return f32::NAN;
    }

    let mut log_k_inv_sum = 0.0f64;
    let mut log_lk_sum = 0.0f64;
    let mut log_k_inv_vals = Vec::with_capacity(k_max);
    let mut log_lk_vals = Vec::with_capacity(k_max);

    for k in 1..=k_max {
        let mut lmk_sum = 0.0f64;
        for m in 1..=k {
            let mut sum_length = 0.0f64;
            let iters = (n - m) / k;
            for i in 1..=iters {
                let idx1 = m + i * k - 1;
                let idx2 = m + (i - 1) * k - 1;
                sum_length += (values[idx1] - values[idx2]).abs() as f64;
            }
            let norm_factor = (n - 1) as f64 / (iters * k) as f64;
            let lmk = (sum_length * norm_factor) / k as f64;
            lmk_sum += lmk;
        }
        let lk = lmk_sum / k as f64;
        let log_lk = lk.ln();
        let log_k_inv = (1.0 / k as f64).ln();

        log_k_inv_vals.push(log_k_inv);
        log_lk_vals.push(log_lk);
        log_k_inv_sum += log_k_inv;
        log_lk_sum += log_lk;
    }

    let count = k_max as f64;
    let x_mean = log_k_inv_sum / count;
    let y_mean = log_lk_sum / count;

    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for i in 0..k_max {
        let dx = log_k_inv_vals[i] - x_mean;
        let dy = log_lk_vals[i] - y_mean;
        num += dx * dy;
        den += dx * dx;
    }

    if den == 0.0 {
        return f32::NAN;
    }

    (num / den) as f32
}

/// np.std(x): computed here so the O(n^2) entropies don't depend on which
/// accumulation flags other requested features happen to enable.
fn population_std(values: &[f32]) -> f32 {
    let n = values.len() as f64;
    if n == 0.0 {
        return 0.0;
    }
    let mean = values.iter().map(|&v| v as f64).sum::<f64>() / n;
    let var = values
        .iter()
        .map(|&v| (v as f64 - mean).powi(2))
        .sum::<f64>()
        / n;
    var.sqrt() as f32
}
