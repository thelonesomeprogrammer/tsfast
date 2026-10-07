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
        Feature::LempelZiv => {
            eval_lempel_ziv(context)
        }
        Feature::LempelZivComplexity(bins) => {
            eval_lempel_ziv_complexity(context, *bins)
        }
        Feature::ApproxEntropy(m, r_bits) if values.len() > *m as usize + 1 => {
            let m_val = *m as usize;
            // tsfresh: r is relative to the population std.
            let r = f32::from_bits(*r_bits) * population_std(values);
            (crate::common::approx_entropy_simd(m_val, r, values)
                - crate::common::approx_entropy_simd(m_val + 1, r, values))
            .abs()
        }
        Feature::SampleEntropy => crate::common::sample_entropy_simd(values, population_std(values)),
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
    let var = values.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
    var.sqrt() as f32
}


use std::simd::prelude::*;
use std::simd::f32x8;

#[inline(always)]
fn lz_complexity_bits(state: &mut crate::common::ColumnState, n: usize) -> f32 {
    if n == 0 {
        return 0.0;
    }

    state.lz_binary_trie.clear();
    state.lz_binary_trie.push([0, 0]); // root node
    let mut num_substrings = 0;

    let mut ind = 0;
    let mut inc = 1;

    while ind + inc <= n {
        let mut curr_node = 0;
        let mut found = true;

        for i in 0..inc {
            let symbol_idx = ind + i;
            let bit = ((state.lz_bit_buffer[symbol_idx / 64] >> (symbol_idx % 64)) & 1) as usize;

            let next_node = state.lz_binary_trie[curr_node][bit];
            if next_node != 0 {
                curr_node = next_node;
            } else {
                found = false;
                break;
            }
        }

        if found {
            inc += 1;
        } else {
            let mut curr = 0;
            for i in 0..inc {
                let symbol_idx = ind + i;
                let bit = ((state.lz_bit_buffer[symbol_idx / 64] >> (symbol_idx % 64)) & 1) as usize;

                let next_node = state.lz_binary_trie[curr][bit];
                if next_node != 0 {
                    curr = next_node;
                } else {
                    let new_node = state.lz_binary_trie.len();
                    state.lz_binary_trie.push([0, 0]);
                    state.lz_binary_trie[curr][bit] = new_node;
                    curr = new_node;
                }
            }
            num_substrings += 1;
            ind += inc;
            inc = 1;
        }
    }

    num_substrings as f32 / n as f32
}

#[inline(always)]
fn lz_complexity(state: &mut crate::common::ColumnState) -> f32 {
    let sequence = &state.lz_symbol_buffer;
    let n = sequence.len();
    if n == 0 {
        return 0.0;
    }

    state.lz_trie_nodes.clear();
    state.lz_trie_nodes.push(Vec::new()); // root node
    let mut num_substrings = 0;

    let mut ind = 0;
    let mut inc = 1;

    while ind + inc <= n {
        let mut curr_node = 0;
        let mut found = true;

        for i in 0..inc {
            let symbol = sequence[ind + i] as u16;

            // Search children
            let mut next_node = None;
            for &(s, child_idx) in &state.lz_trie_nodes[curr_node] {
                if s == symbol {
                    next_node = Some(child_idx);
                    break;
                }
            }

            if let Some(child_idx) = next_node {
                curr_node = child_idx;
            } else {
                found = false;
                break;
            }
        }

        if found {
            inc += 1;
        } else {
            // Add new substring to trie
            let mut curr = 0;
            for i in 0..inc {
                let symbol = sequence[ind + i] as u16;

                let mut next_node = None;
                for &(s, child_idx) in &state.lz_trie_nodes[curr] {
                    if s == symbol {
                        next_node = Some(child_idx);
                        break;
                    }
                }

                if let Some(child_idx) = next_node {
                    curr = child_idx;
                } else {
                    let new_node = state.lz_trie_nodes.len();
                    state.lz_trie_nodes.push(Vec::new());
                    state.lz_trie_nodes[curr].push((symbol, new_node));
                    curr = new_node;
                }
            }
            num_substrings += 1;
            ind += inc;
            inc = 1;
        }
    }

    num_substrings as f32 / n as f32
}

/// Computes the Lempel-Ziv's (LZ) complexity index, normalized by the signal's length.
/// TSFEL definition: binarises around the mean threshold, and counts LZ76 dictionary entries.
/// Optimizaton: binarise 8 symbols per instruction into a u64 buffer.
fn eval_lempel_ziv(context: &mut crate::context::FeatureContext) -> f32 {
    let values = context.values;
    let n = values.len();
    let state = &mut *context.state;
    if n == 0 {
        return 0.0;
    }

    let threshold = context.mean;
    let threshold_vec = f32x8::splat(threshold);

    state.lz_bit_buffer.clear();
    state.lz_bit_buffer.resize((n + 63) / 64, 0);

    let mut chunks = values.chunks_exact(8);
    let mut idx = 0;
    for chunk in &mut chunks {
        let vals = f32x8::from_slice(chunk);
        let mask = vals.simd_gt(threshold_vec);
        let bitmask = mask.to_bitmask() as u64;
        state.lz_bit_buffer[idx / 64] |= bitmask << (idx % 64);
        idx += 8;
    }
    for &v in chunks.remainder() {
        if v > threshold {
            state.lz_bit_buffer[idx / 64] |= 1 << (idx % 64);
        }
        idx += 1;
    }

    lz_complexity_bits(state, n)
}

/// Calculate a complexity estimate based on the Lempel-Ziv compression algorithm.
/// tsfresh definition: discretises into `bins` equal-width bins and then runs an LZ76-style dictionary parse.
fn eval_lempel_ziv_complexity(context: &mut crate::context::FeatureContext, bins: u16) -> f32 {
    let values = context.values;
    let state = &mut *context.state;
    if values.is_empty() || bins < 2 {
        return 0.0;
    }

    let min_val = state.min_value;
    let max_val = state.max_value;

    state.lz_symbol_buffer.clear();

    if max_val <= min_val {
        for _ in values {
            state.lz_symbol_buffer.push(0);
        }
        return lz_complexity(state);
    }

    let bin_width = (max_val - min_val) / bins as f32;

    for &v in values {
        // Compute searchsorted equivalent
        if v <= min_val {
            state.lz_symbol_buffer.push(0);
        } else if v >= max_val {
            state.lz_symbol_buffer.push((bins - 1) as u8);
        } else {
            let mut bin_idx = ((v - min_val) / bin_width) as u8;
            if bin_idx >= bins as u8 {
                bin_idx = bins as u8 - 1;
            }
            state.lz_symbol_buffer.push(bin_idx);
        }
    }

    lz_complexity(state)
}
