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
        Feature::Dfa => calc_dfa(values, context.state),
        Feature::HurstExponent => calc_hurst(values, context.state),
        Feature::HiguchiFd => calc_higuchi_fd(values, context.state),
        Feature::MaximumFractalLength => calc_maximum_fractal_length(values, context.state),
        _ => return None,
    };
    Some(res)
}

/// Calculates the Detrended Fluctuation Analysis (DFA) of the signal.
///
/// This implementation matches the TSFEL logic exactly:
/// - Windows scales: `linspace(4, n // 10, n // 2, dtype=int)` deduplicated.
/// - The windows do not overlap; each chunk is evaluated independently.
/// - The profile is built as the cumulative sum of the demeaned signal (`cumsum(signal - mean)`).
/// - Linear detrending (order 1) is applied to each chunk to compute the residual sum of squares.
/// - A log-log fit is performed to determine the scaling exponent after identifying any potential plateau.
/// - Minimum size behavior requires 160 elements; otherwise, NaN is returned.
#[inline(always)]
fn calc_dfa(values: &[f32], state: &mut crate::common::ColumnState) -> f32 {
    let n = values.len();
    if n < 160 {
        return f32::NAN;
    }

    let first = values[0];
    if values.iter().all(|&v| v == first) {
        return f32::NAN;
    }

    let mean = values.iter().map(|&v| v as f64).sum::<f64>() / (n as f64);

    let mut acc_sig = std::mem::take(&mut state.workspace_f64_1);
    acc_sig.resize(n, 0.0);

    let mut y_pref = std::mem::take(&mut state.workspace_f64_2);
    y_pref.resize(n + 1, 0.0);

    let mut y2_pref = std::mem::take(&mut state.workspace_f64_3);
    y2_pref.resize(n + 1, 0.0);

    let mut iy_pref = std::mem::take(&mut state.workspace_f64_4);
    iy_pref.resize(n + 1, 0.0);

    let mut current_acc = 0.0;
    for i in 0..n {
        current_acc += (values[i] as f64) - mean;
        acc_sig[i] = current_acc;

        y_pref[i + 1] = y_pref[i] + current_acc;
        y2_pref[i + 1] = y2_pref[i] + current_acc * current_acc;
        iy_pref[i + 1] = iy_pref[i] + (i as f64) * current_acc;
    }

    let min_scale = 4;
    let max_scale = n / 10;
    let num_scales = n / 2;

    let mut scales = std::mem::take(&mut state.approx_entropy_buffer);
    scales.clear();

    if num_scales > 1 {
        for i in 0..num_scales {
            let scale = (min_scale as f64) + ((max_scale - min_scale) as f64) * (i as f64) / ((num_scales - 1) as f64);
            scales.push(scale.floor() as usize);
        }
    } else {
        scales.push(min_scale);
    }

    scales.sort_unstable();
    scales.dedup();

    let mut log_scales = std::mem::take(&mut state.workspace_f64_5);
    log_scales.clear();
    let mut log_flucts = std::mem::take(&mut state.sort_buffer);
    let mut log_flucts_f64 = Vec::with_capacity(scales.len());

    for &s in &scales {
        let num_windows = n / s;
        if num_windows == 0 {
            continue;
        }

        let mut rms_sum = 0.0;
        let x_bar = (s - 1) as f64 / 2.0;
        let ss_xx = (s as f64) * (((s * s) as f64) - 1.0) / 12.0;

        for idx in 0..num_windows {
            let start = idx * s;
            let end = start + s;

            let sum_y = y_pref[end] - y_pref[start];
            let sum_y2 = y2_pref[end] - y2_pref[start];
            let sum_iy_global = iy_pref[end] - iy_pref[start];

            let sum_jy = sum_iy_global - (start as f64) * sum_y;

            let y_bar = sum_y / (s as f64);
            let ss_xy = sum_jy - (s as f64) * x_bar * y_bar;
            let mut beta1 = 0.0;
            if ss_xx != 0.0 {
                beta1 = ss_xy / ss_xx;
            }
            let beta0 = y_bar - beta1 * x_bar;

            let mut rss = sum_y2 - beta0 * sum_y - beta1 * sum_jy;
            if rss < 0.0 {
                rss = 0.0;
            }

            rms_sum += rss / (s as f64);
        }

        let fluct = (rms_sum / (num_windows as f64)).sqrt();
        log_scales.push((s as f64).ln());
        log_flucts_f64.push(fluct.ln());
    }

    let mut i_plateau = log_flucts_f64.len();
    if log_flucts_f64.len() > 5 {
        let mut dy = Vec::with_capacity(log_flucts_f64.len() - 1);
        for i in 0..log_flucts_f64.len() - 1 {
            dy.push(log_flucts_f64[i + 1] - log_flucts_f64[i]);
        }

        let mean_y = log_flucts_f64.iter().sum::<f64>() / (log_flucts_f64.len() as f64);
        let consecutive_points = 5;

        for i in 0..=(dy.len() - consecutive_points) {
            let mut all_below = true;
            for j in 0..consecutive_points {
                if dy[i + j].abs() >= 0.1 {
                    all_below = false;
                    break;
                }
            }
            if all_below {
                let plateau_val = log_flucts_f64[i..i + consecutive_points].iter().sum::<f64>() / (consecutive_points as f64);
                if plateau_val > mean_y {
                    i_plateau = i;
                    break;
                }
            }
        }
    }

    state.workspace_f64_1 = acc_sig;
    state.workspace_f64_2 = y_pref;
    state.workspace_f64_3 = y2_pref;
    state.workspace_f64_4 = iy_pref;
    state.approx_entropy_buffer = scales;

    let res = if i_plateau > 1 {
        let x = &log_scales[0..i_plateau];
        let y = &log_flucts_f64[0..i_plateau];

        let mean_x = x.iter().sum::<f64>() / (x.len() as f64);
        let mean_y = y.iter().sum::<f64>() / (y.len() as f64);

        let mut num = 0.0;
        let mut den = 0.0;
        for i in 0..x.len() {
            let dx = x[i] - mean_x;
            let dy = y[i] - mean_y;
            num += dx * dy;
            den += dx * dx;
        }
        if den == 0.0 {
            f32::NAN
        } else {
            (num / den) as f32
        }
    } else {
        f32::NAN
    };

    state.workspace_f64_5 = log_scales;
    state.sort_buffer = log_flucts;

    res
}

/// Calculates the Hurst exponent of the signal through Rescaled Range (R/S) analysis.
///
/// This implementation matches the TSFEL logic exactly:
/// - Windows scales: `linspace(4, n // 10, n // 2, dtype=int)` deduplicated.
/// - The windows do not overlap; each chunk is evaluated independently.
/// - The profile is built as the cumulative sum of the demeaned chunk (`cumsum(chunk - mean)`).
/// - The chunk standard deviation is calculated over the raw chunk values.
/// - Minimum size behavior requires 160 elements; otherwise, NaN is returned.
#[inline(always)]
fn calc_hurst(values: &[f32], state: &mut crate::common::ColumnState) -> f32 {
    let n = values.len();
    if n < 160 {
        return f32::NAN;
    }

    let first = values[0];
    if values.iter().all(|&v| v == first) {
        return f32::NAN;
    }

    let min_scale = 4;
    let max_scale = n / 10;
    let num_scales = n / 2;

    let mut scales = std::mem::take(&mut state.approx_entropy_buffer);
    scales.clear();

    if num_scales > 1 {
        for i in 0..num_scales {
            let scale = (min_scale as f64) + ((max_scale - min_scale) as f64) * (i as f64) / ((num_scales - 1) as f64);
            scales.push(scale.floor() as usize);
        }
    } else {
        scales.push(min_scale);
    }

    scales.sort_unstable();
    scales.dedup();

    let mut log_scales = std::mem::take(&mut state.workspace_f64_5);
    log_scales.clear();
    let mut log_rs = std::mem::take(&mut state.workspace_f64_4);
    log_rs.clear();

    let mut y_pref = std::mem::take(&mut state.workspace_f64_1);
    y_pref.resize(n + 1, 0.0);

    let mut y2_pref = std::mem::take(&mut state.workspace_f64_2);
    y2_pref.resize(n + 1, 0.0);

    for i in 0..n {
        let v = values[i] as f64;
        y_pref[i + 1] = y_pref[i] + v;
        y2_pref[i + 1] = y2_pref[i] + v * v;
    }

    for &s in &scales {
        let num_windows = n / s;
        if num_windows == 0 {
            continue;
        }

        let mut rs_sum = 0.0;
        let mut valid_windows = 0;

        for idx in 0..num_windows {
            let start = idx * s;
            let end = start + s;

            let sum_y = y_pref[end] - y_pref[start];
            let sum_y2 = y2_pref[end] - y2_pref[start];
            let mean = sum_y / (s as f64);

            let mut var = (sum_y2 - 2.0 * mean * sum_y + (s as f64) * mean * mean) / (s as f64);
            if var < 0.0 { var = 0.0; }
            let std = var.sqrt();

            let mut acc = 0.0;
            let mut max_acc = f64::NEG_INFINITY;
            let mut min_acc = f64::INFINITY;

            for i in start..end {
                acc += (values[i] as f64) - mean;
                if acc > max_acc { max_acc = acc; }
                if acc < min_acc { min_acc = acc; }
            }

            let r = max_acc - min_acc;
            if std > 0.0 {
                rs_sum += r / std;
                valid_windows += 1;
            }
        }

        if valid_windows > 0 {
            let mean_rs = rs_sum / (valid_windows as f64);
            log_scales.push((s as f64).log10());
            log_rs.push(mean_rs.log10());
        }
    }

    state.workspace_f64_1 = y_pref;
    state.workspace_f64_2 = y2_pref;
    state.approx_entropy_buffer = scales;

    let res = if log_scales.len() > 1 {
        let x = &log_scales;
        let y = &log_rs;

        let mean_x = x.iter().sum::<f64>() / (x.len() as f64);
        let mean_y = y.iter().sum::<f64>() / (y.len() as f64);

        let mut num = 0.0;
        let mut den = 0.0;
        for i in 0..x.len() {
            let dx = x[i] - mean_x;
            let dy = y[i] - mean_y;
            num += dx * dy;
            den += dx * dx;
        }
        if den == 0.0 {
            f32::NAN
        } else {
            (num / den) as f32
        }
    } else {
        f32::NAN
    };

    state.workspace_f64_5 = log_scales;
    state.workspace_f64_4 = log_rs;

    res
}

#[inline(always)]
fn ensure_higuchi_lengths(values: &[f32], state: &mut crate::common::ColumnState) {
    if !state.higuchi_lk.is_empty() {
        return; // Already computed
    }
    let n = values.len();
    if n < 10 {
        return;
    }
    let k_max = n / 10 - 1;
    if k_max < 2 {
        return;
    }

    let mut higuchi_lk = std::mem::take(&mut state.higuchi_lk);
    let mut higuchi_k_values = std::mem::take(&mut state.higuchi_k_values);
    higuchi_lk.clear();
    higuchi_k_values.clear();
    let mut log_k_inv_sum = 0.0f64;
    let mut log_lk_sum = 0.0f64;

    // Opt: Compute linear regression standard covariance/variance dynamically in one pass
    // to avoid allocating two vectors for `log_k_inv` and `log_lk`
    let mut sum_x_sq = 0.0f64;
    let mut sum_xy = 0.0f64;

    // TSFEL normalization logic requires evaluating sum for each m individually.
    // Instead of using a vector of size k, we iterate directly using O(1) space to avoid allocations.
    for k in 1..=k_max {
        let mut lmk_sum = 0.0f64;

        for m in 1..=k {
            let iters = (n - m) / k;
            let mut sum_length = 0.0f64;

            // Vectorize by iterating through contiguous slices where possible or just manually striding
            // The compiler typically unrolls this strided iteration
            let mut i = 1;
            while i <= iters {
                let idx1 = m + i * k - 1;
                let idx2 = m + (i - 1) * k - 1;
                sum_length += (values[idx1] - values[idx2]).abs() as f64;
                i += 1;
            }

            let norm_factor = (n - 1) as f64 / (iters * k) as f64;
            let lmk = (sum_length * norm_factor) / k as f64;
            lmk_sum += lmk;
        }

        let lk = lmk_sum / k as f64;
        higuchi_lk.push(lk);
        higuchi_k_values.push(k as f64);
    }

    state.higuchi_lk = higuchi_lk;
    state.higuchi_k_values = higuchi_k_values;
}

#[inline(always)]
fn calc_higuchi_fd(values: &[f32], state: &mut crate::common::ColumnState) -> f32 {
    let n = values.len();
    if n < 10 {
        return f32::NAN;
    }
    let k_max = n / 10 - 1;
    if k_max < 2 {
        return f32::NAN;
    }

    ensure_higuchi_lengths(values, state);

    let lk = &state.higuchi_lk;
    let k_values = &state.higuchi_k_values;

    if lk.len() < 2 {
        return f32::NAN;
    }

    let count = lk.len() as f64;
    let mut log_k_inv_sum = 0.0f64;
    let mut log_lk_sum = 0.0f64;

    for i in 0..lk.len() {
        log_k_inv_sum += (1.0 / k_values[i]).ln();
        log_lk_sum += lk[i].ln();
        log_k_inv_sum += log_k_inv;
        log_lk_sum += log_lk;
        sum_x_sq += log_k_inv * log_k_inv;
        sum_xy += log_k_inv * log_lk;
    }

    let x_mean = log_k_inv_sum / count;
    let y_mean = log_lk_sum / count;

    let mut num = 0.0f64;
    let mut den = 0.0f64;

    for i in 0..lk.len() {
        let dx = (1.0 / k_values[i]).ln() - x_mean;
        let dy = lk[i].ln() - y_mean;
        num += dx * dy;
        den += dx * dx;
    }
    let num = sum_xy - count * x_mean * y_mean;
    let den = sum_x_sq - count * x_mean * x_mean;

    if den == 0.0 {
        return f32::NAN;
    }

    (num / den) as f32
}

#[inline(always)]
fn calc_maximum_fractal_length(values: &[f32], state: &mut crate::common::ColumnState) -> f32 {
    let n = values.len();
    if n < 10 {
        return f32::NAN;
    }
    let k_max = n / 10 - 1;
    if k_max < 2 {
        return f32::NAN;
    }

    ensure_higuchi_lengths(values, state);

    let lk = &state.higuchi_lk;
    let k_values = &state.higuchi_k_values;

    if lk.len() < 2 {
        return f32::NAN;
    }

    let count = lk.len() as f64;
    let mut log_k_inv_sum = 0.0f64;
    let mut log_lk_sum = 0.0f64;

    for i in 0..lk.len() {
        log_k_inv_sum += (1.0 / k_values[i]).log10();
        log_lk_sum += lk[i].log10();
    }

    let x_mean = log_k_inv_sum / count;
    let y_mean = log_lk_sum / count;

    let mut num = 0.0f64;
    let mut den = 0.0f64;

    for i in 0..lk.len() {
        let dx = (1.0 / k_values[i]).log10() - x_mean;
        let dy = lk[i].log10() - y_mean;
        num += dx * dy;
        den += dx * dx;
    }

    if den == 0.0 {
        return f32::NAN;
    }

    let slope = num / den;
    let intercept = y_mean - slope * x_mean;

    intercept as f32
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
