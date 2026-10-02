use crate::common::ColumnState;
use crate::common::LANES;
use crate::types::{FastBitArray, Feature};
use realfft::{RealToComplex, num_complex};
use std::simd::cmp::SimdPartialOrd;
use std::simd::f32x4;
use std::simd::num::SimdFloat;
use std::sync::Arc;

const ENTROPY_BINS: usize = 10;

pub(crate) struct ExpandingEngine<'a> {
    pub(crate) compute: FastBitArray,
    pub(crate) features: &'a [Feature],
    pub(crate) unique_paa_totals: &'a [u16],
    pub(crate) unique_c3_lags: &'a [u16],
    pub(crate) unique_autocorr_lags: &'a [u16],
    pub(crate) paa_boundaries: &'a [Vec<usize>],
    pub(crate) r2c: Option<Arc<dyn RealToComplex<f32>>>,
    pub(crate) fft_size: usize,
    pub(crate) fft_update_period: usize,
}

impl<'a> ExpandingEngine<'a> {
    #[inline(always)]
    pub(crate) fn process_expanding(
        &mut self,
        values: &[f32],
        global_idx: usize,
        state: &mut ColumnState,
        full_series: &mut Vec<f32>,
        running_sorted: &mut Vec<f32>,
    ) -> Vec<f32> {
        // 1. Pass 1: Update incremental state with NEW values
        let rem_start = self.process_simd_chunks(values, global_idx, state, full_series);
        self.process_remainder(values, rem_start, global_idx, state, full_series);

        // Update prefix sums before extending full_series or after?
        // Let's do it after extending full_series to be consistent.
        full_series.extend_from_slice(values);

        if state.prefix_sums.is_empty() && !full_series.is_empty() {
            let mut s = 0.0;
            state.prefix_sums.reserve(full_series.len());
            for &v in full_series.iter() {
                s += v;
                state.prefix_sums.push(s);
            }
        } else {
            let mut s = state.prefix_sums.last().copied().unwrap_or(0.0);
            for &v in values {
                s += v;
                state.prefix_sums.push(s);
            }
        }

        let total_n = full_series.len() as f32;

        // 3. Pass 2: Finalize results over FULL series for mean-dependent features
        self.finalize_results(total_n, state, running_sorted, full_series)
    }

    #[inline(always)]
    fn process_simd_chunks(
        &self,
        values: &[f32],
        global_start_idx: usize,
        state: &mut ColumnState,
        full_series: &[f32],
    ) -> usize {
        let chunks = values.chunks_exact(LANES);
        let rem_start = (values.len() / LANES) * LANES;

        for (c_idx, i) in chunks.enumerate() {
            let offset_usize = global_start_idx + c_idx * LANES;
            let offset = offset_usize as f32;
            let chunk = f32x4::from_slice(i);
            let shifted = f32x4::from_array([state.prev_last, i[0], i[1], i[2]]);
            let indices = f32x4::from_array([offset, offset + 1.0, offset + 2.0, offset + 3.0]);
            let simd_zero = f32x4::splat(0.0);

            state.total_sum += chunk.reduce_sum();

            let c_min = chunk.reduce_min();
            if c_min < state.min_value {
                state.min_value = c_min;
                // Find first occurrence in this chunk
                for bit in 0..4 {
                    if i[bit] == c_min {
                        state.first_min_idx = offset_usize + bit;
                        break;
                    }
                }
            }
            if c_min <= state.min_value {
                // Update last occurrence
                for bit in (0..4).rev() {
                    if i[bit] == state.min_value {
                        state.last_min_idx = offset_usize + bit;
                        break;
                    }
                }
            }

            let c_max = chunk.reduce_max();
            if c_max > state.max_value {
                state.max_value = c_max;
                for bit in 0..4 {
                    if i[bit] == c_max {
                        state.first_max_idx = offset_usize + bit;
                        break;
                    }
                }
            }
            if c_max >= state.max_value {
                for bit in (0..4).rev() {
                    if i[bit] == state.max_value {
                        state.last_max_idx = offset_usize + bit;
                        break;
                    }
                }
            }

            if self.compute[38] {
                state.abs_max = state.abs_max.max(chunk.abs().reduce_max());
            }
            let sq = chunk * chunk;
            state.energy += sq.reduce_sum();
            state.sum_cubes += (sq * chunk).reduce_sum();
            state.sum_quads += (sq * sq).reduce_sum();
            let diff = chunk - shifted;
            state.mac_sum_vec += diff.abs();
            state.mc_sum_vec += diff;
            state.sum_sq_diff += (diff * diff).reduce_sum();
            state.sum_prod += (chunk * shifted).reduce_sum();
            state.auc_sum += (chunk + shifted).reduce_sum() * 0.5;
            let signs = chunk.simd_lt(simd_zero);
            let prev_signs = shifted.simd_lt(simd_zero);
            let mask = (signs ^ prev_signs).to_bitmask();
            state.zcr_count += mask.count_ones();
            state.sum_ix += (indices * chunk).reduce_sum();
            for bit in 0..4 {
                if (mask >> bit) & 1 == 1 {
                    state.zc_indices.push(offset + bit as f32);
                }
            }
            state.prev_last = i[LANES - 1];
        }

        // Scalar pass for features that are hard to SIMD or need more context
        for (i, &val) in values[..rem_start].iter().enumerate() {
            let global_idx = global_start_idx + i;

            if self.compute[16] {
                if state.prev_val > state.prev_prev_val && state.prev_val > val {
                    state.peaks += 1;
                }
                state.prev_prev_val = state.prev_val;
                state.prev_val = val;
            }

            if self.compute[30] {
                for (l_idx, &lag) in self.unique_c3_lags.iter().enumerate() {
                    let l = lag as usize;
                    if global_idx >= 2 * l {
                        let v_l = if global_idx - l < global_start_idx {
                            full_series[global_idx - l]
                        } else {
                            values[global_idx - l - global_start_idx]
                        };
                        let v_2l = if global_idx - 2 * l < global_start_idx {
                            full_series[global_idx - 2 * l]
                        } else {
                            values[global_idx - 2 * l - global_start_idx]
                        };
                        state.c3_sums[l_idx] += val * v_l * v_2l;
                    }
                }
            }

            if self.compute[43] || self.compute[44] {
                for (l_idx, &lag) in self.unique_autocorr_lags.iter().enumerate() {
                    let l = lag as usize;
                    if global_idx >= l {
                        let v_l = if global_idx - l < global_start_idx {
                            full_series[global_idx - l]
                        } else {
                            values[global_idx - l - global_start_idx]
                        };
                        state.autocorr_sums[l_idx] += val * v_l;
                    }
                }
            }
        }
        rem_start
    }

    #[inline(always)]
    fn process_remainder(
        &self,
        values: &[f32],
        rem_start: usize,
        global_start_idx: usize,
        state: &mut ColumnState,
        full_series: &[f32],
    ) {
        for i in rem_start..values.len() {
            let val = values[i];
            let global_idx = global_start_idx + i;
            if self.compute[0] {
                state.total_sum += val;
            }
            if self.compute[4] {
                if val < state.min_value {
                    state.min_value = val;
                    state.first_min_idx = global_idx;
                }
                if val <= state.min_value {
                    state.last_min_idx = global_idx;
                }
            }
            if self.compute[5] {
                if val > state.max_value {
                    state.max_value = val;
                    state.first_max_idx = global_idx;
                }
                if val >= state.max_value {
                    state.last_max_idx = global_idx;
                }
            }
            if self.compute[38] {
                state.abs_max = state.abs_max.max(val.abs());
            }
            if self.compute[12] {
                let sq = val * val;
                state.energy += sq;
                if self.compute[7] {
                    state.sum_cubes += sq * val;
                }
                if self.compute[8] {
                    state.sum_quads += sq * sq;
                }
            }
            if global_idx > 0 {
                let prev = if i > 0 {
                    values[i - 1]
                } else {
                    state.prev_last
                };
                let diff = val - prev;
                if self.compute[18] {
                    state.mac_sum_vec += f32x4::from_array([diff.abs(), 0.0, 0.0, 0.0]);
                }
                if self.compute[19] {
                    state.mc_sum_vec += f32x4::from_array([diff, 0.0, 0.0, 0.0]);
                }
                if self.compute[20] {
                    state.sum_sq_diff += diff * diff;
                }
                if self.compute[17] {
                    state.sum_prod += val * prev;
                }
                if self.compute[31] {
                    state.auc_sum += (val + prev) * 0.5;
                }
                if self.compute[15] && (val < 0.0) != (prev < 0.0) {
                    state.zcr_count += 1;
                    if self.compute[36] {
                        state.zc_indices.push(global_idx as f32);
                    }
                }
            }

            if self.compute[16] {
                if state.prev_val > state.prev_prev_val && state.prev_val > val {
                    state.peaks += 1;
                }
                state.prev_prev_val = state.prev_val;
                state.prev_val = val;
            }

            if self.compute[30] {
                for (l_idx, &lag) in self.unique_c3_lags.iter().enumerate() {
                    let l = lag as usize;
                    if global_idx >= 2 * l {
                        let v_l = if global_idx - l < global_start_idx {
                            full_series[global_idx - l]
                        } else {
                            values[global_idx - l - global_start_idx]
                        };
                        let v_2l = if global_idx - 2 * l < global_start_idx {
                            full_series[global_idx - 2 * l]
                        } else {
                            values[global_idx - 2 * l - global_start_idx]
                        };
                        state.c3_sums[l_idx] += val * v_l * v_2l;
                    }
                }
            }

            if self.compute[43] || self.compute[44] {
                for (l_idx, &lag) in self.unique_autocorr_lags.iter().enumerate() {
                    let l = lag as usize;
                    if global_idx >= l {
                        let v_l = if global_idx - l < global_start_idx {
                            full_series[global_idx - l]
                        } else {
                            values[global_idx - l - global_start_idx]
                        };
                        state.autocorr_sums[l_idx] += val * v_l;
                    }
                }
            }

            if self.compute[21] {
                state.sum_ix += (global_idx as f32) * val;
            }
            state.prev_last = val;
        }
    }

    #[inline(always)]
    fn finalize_results(
        &self,
        n: f32,
        state: &mut ColumnState,
        running_sorted: &mut Vec<f32>,
        full_series: &[f32],
    ) -> Vec<f32> {
        let mean = state.total_sum / n;
        let mac_sum = state.mac_sum_vec.reduce_sum();
        let mc_sum = state.mc_sum_vec.reduce_sum();

        // Moments from power sums (more efficient for expanding window)
        let m2 = state.energy - (state.total_sum * state.total_sum) / n;
        let m3 = state.sum_cubes - 3.0 * mean * state.energy + 2.0 * mean * mean * state.total_sum;
        let m4 = state.sum_quads - 4.0 * mean * state.sum_cubes + 6.0 * mean * mean * state.energy
            - 3.0 * mean * mean * mean * state.total_sum;

        let mut mad_sum = 0.0;
        let mut count_a = 0;
        let mut count_b = 0;
        let mut max_strike_a = 0;
        let mut current_strike_a = 0;
        let mut max_strike_b = 0;
        let mut current_strike_b = 0;
        let mut median = 0.0;
        let mut iqr = 0.0;
        let mut entropy = 0.0;
        let mut zc_mean = 0.0;
        let mut zc_std = 0.0;

        // 1. Median/IQR Optimization: Merge instead of Sort
        if self.compute.any([6, 10, 49, 63]) {
            if running_sorted.len() < full_series.len() {
                let mut new_elements: Vec<f32> = full_series[running_sorted.len()..].to_vec();
                new_elements
                    .sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                if running_sorted.is_empty() {
                    *running_sorted = new_elements;
                } else {
                    let mut merged = Vec::with_capacity(running_sorted.len() + new_elements.len());
                    let mut i = 0;
                    let mut j = 0;
                    while i < running_sorted.len() && j < new_elements.len() {
                        if running_sorted[i] <= new_elements[j] {
                            merged.push(running_sorted[i]);
                            i += 1;
                        } else {
                            merged.push(new_elements[j]);
                            j += 1;
                        }
                    }
                    merged.extend_from_slice(&running_sorted[i..]);
                    merged.extend_from_slice(&new_elements[j..]);
                    *running_sorted = merged;
                }
            }
            let n_size = running_sorted.len();
            if n_size > 0 {
                if self.compute[6] {
                    if n_size % 2 == 1 {
                        median = running_sorted[n_size / 2];
                    } else {
                        let mid = n_size / 2;
                        median = (running_sorted[mid] + running_sorted[mid - 1]) / 2.0;
                    }
                }
                if self.compute[10] {
                    let n_f = n_size as f32;
                    let get_q = |q: f32, data: &[f32]| -> f32 {
                        if data.is_empty() {
                            return 0.0;
                        }
                        let idx = q * (n_f - 1.0);
                        let i = idx.floor() as usize;
                        let f = idx - i as f32;
                        if i >= data.len() - 1 {
                            data[data.len() - 1]
                        } else {
                            (1.0 - f) * data[i] + f * data[i + 1]
                        }
                    };
                    iqr = get_q(0.75, running_sorted) - get_q(0.25, running_sorted);
                }
            }
        }

        let mut spectrum = Vec::new();
        let mut fft_complex = Vec::new();
        let mut freq_centroid = 0.0;
        let mut spectral_decrease = 0.0;
        let mut spectral_slope = 0.0;

        if self.compute.any_fft() {
            let n_total = full_series.len();
            let should_update =
                state.last_fft_n == 0 || (n_total - state.last_fft_n) >= self.fft_update_period;

            if should_update {
                if let Some(r2c) = &self.r2c {
                    if state.fft_in_buffer.len() < self.fft_size {
                        state.fft_in_buffer.resize(self.fft_size, 0.0);
                    }
                    state.fft_in_buffer.fill(0.0);
                    state.fft_in_buffer[..n_total].copy_from_slice(full_series);
                    let complex_len = r2c.complex_len();
                    if state.fft_out_buffer.len() < complex_len {
                        state
                            .fft_out_buffer
                            .resize(complex_len, num_complex::Complex::new(0.0, 0.0));
                    }
                    r2c.process(
                        &mut state.fft_in_buffer,
                        &mut state.fft_out_buffer[..complex_len],
                    )
                    .unwrap();
                    spectrum = state.fft_out_buffer[..complex_len]
                        .iter()
                        .map(|c| c.norm())
                        .collect();
                    fft_complex = state.fft_out_buffer[..complex_len].to_vec();
                } else if self.compute.any([54, 55, 56, 57, 60, 64]) {
                    use realfft::RealFftPlanner;
                    let mut planner = RealFftPlanner::<f32>::new();
                    let r2c = planner.plan_fft_forward(n_total);
                    let complex_len = r2c.complex_len();
                    state.fft_in_buffer.clear();
                    state.fft_in_buffer.extend_from_slice(full_series);
                    if state.fft_out_buffer.len() < complex_len {
                        state
                            .fft_out_buffer
                            .resize(complex_len, num_complex::Complex::new(0.0, 0.0));
                    }
                    r2c.process(
                        &mut state.fft_in_buffer,
                        &mut state.fft_out_buffer[..complex_len],
                    )
                    .unwrap();
                    spectrum = state.fft_out_buffer[..complex_len]
                        .iter()
                        .map(|c| c.norm())
                        .collect();
                    fft_complex = state.fft_out_buffer[..spectrum.len()].to_vec();
                }
                // Cache it
                state.last_fft_n = n_total;
                state.last_spectrum = spectrum.clone();
                state.last_fft_complex = fft_complex.clone();
            } else {
                // Use cached
                spectrum = state.last_spectrum.clone();
                fft_complex = state.last_fft_complex.clone();
            }

            if !spectrum.is_empty() {
                let spec_sum: f32 = spectrum.iter().sum();
                if spec_sum > 0.0 {
                    freq_centroid = spectrum
                        .iter()
                        .enumerate()
                        .map(|(i, &mag)| i as f32 * mag)
                        .sum::<f32>()
                        / spec_sum;

                    if spectrum.len() > 1 {
                        let spec_sum_no_first: f32 = spectrum[1..].iter().sum();
                        if spec_sum_no_first > 0.0 {
                            spectral_decrease = spectrum[1..]
                                .iter()
                                .enumerate()
                                .map(|(i, &mag)| (mag - spectrum[0]) / (i + 1) as f32)
                                .sum::<f32>()
                                / spec_sum_no_first;
                        }

                        let m_n = spectrum.len() as f32;
                        let sum_x: f32 = (0..spectrum.len()).map(|i| i as f32).sum();
                        let sum_y: f32 = spectrum.iter().sum();
                        let sum_xx: f32 = (0..spectrum.len()).map(|i| (i as f32).powi(2)).sum();
                        let sum_xy: f32 = spectrum
                            .iter()
                            .enumerate()
                            .map(|(i, &mag)| i as f32 * mag)
                            .sum();
                        let s_xx = sum_xx - (sum_x * sum_x) / m_n;
                        let s_xy = sum_xy - (sum_x * sum_y) / m_n;
                        if s_xx.abs() > 1e-9 {
                            spectral_slope = s_xy / s_xx;
                        }
                    }
                }
            }
        }

        // 2. Single Pass for mean-dependent and range-dependent features
        if self.compute.any([9, 11, 25, 26, 27, 28]) {
            let range = state.max_value - state.min_value;
            let bins = ENTROPY_BINS;
            let mut counts: [usize; ENTROPY_BINS] = [0; ENTROPY_BINS];

            let mean_vec = f32x4::splat(mean);
            let mut mad_sum_vec = f32x4::splat(0.0);

            for chunk in full_series.chunks_exact(LANES) {
                let c = f32x4::from_slice(chunk);
                if self.compute[9] {
                    mad_sum_vec += (c - mean_vec).abs();
                }

                if self.compute[25] {
                    count_a += c.simd_gt(mean_vec).to_bitmask().count_ones() as usize;
                }
                if self.compute[26] {
                    count_b += c.simd_lt(mean_vec).to_bitmask().count_ones() as usize;
                }

                for &v in chunk {
                    if self.compute[11] && range > 1e-9 {
                        let b = (((v - state.min_value) / range) * (bins as f32 - 1.0)) as usize;
                        counts[b.min(bins - 1)] += 1;
                    }

                    if self.compute.any([27, 28]) {
                        if v > mean {
                            current_strike_a += 1;
                            max_strike_a = max_strike_a.max(current_strike_a);
                            current_strike_b = 0;
                        } else if v < mean {
                            current_strike_b += 1;
                            max_strike_b = max_strike_b.max(current_strike_b);
                            current_strike_a = 0;
                        } else {
                            current_strike_a = 0;
                            current_strike_b = 0;
                        }
                    }
                }
            }
            mad_sum = mad_sum_vec.reduce_sum();

            let rem_start = (full_series.len() / LANES) * LANES;
            for &v in &full_series[rem_start..] {
                if self.compute[9] {
                    mad_sum += (v - mean).abs();
                }
                if self.compute[11] && range > 1e-9 {
                    let b = (((v - state.min_value) / range) * (bins as f32 - 1.0)) as usize;
                    counts[b.min(bins - 1)] += 1;
                }
                if self.compute[25] && v > mean {
                    count_a += 1;
                }
                if self.compute[26] && v < mean {
                    count_b += 1;
                }
                if self.compute.any([27, 28]) {
                    if v > mean {
                        current_strike_a += 1;
                        max_strike_a = max_strike_a.max(current_strike_a);
                        current_strike_b = 0;
                    } else if v < mean {
                        current_strike_b += 1;
                        max_strike_b = max_strike_b.max(current_strike_b);
                        current_strike_a = 0;
                    } else {
                        current_strike_a = 0;
                        current_strike_b = 0;
                    }
                }
            }

            if self.compute[11] && range > 1e-9 {
                for &c in &counts {
                    if c > 0 {
                        let p = c as f32 / n;
                        entropy -= p * p.ln();
                    }
                }
            }
        }

        if self.compute[34] && !state.zc_indices.is_empty() {
            zc_mean = state.zc_indices.iter().sum::<f32>() / state.zc_indices.len() as f32;
            if self.compute[35] {
                let zc_m2 = state
                    .zc_indices
                    .iter()
                    .map(|&idx| (idx - zc_mean).powi(2))
                    .sum::<f32>();
                zc_std = (zc_m2 / state.zc_indices.len() as f32).sqrt();
            }
        }

        let var = if n > 1.0 { m2 / (n - 1.0) } else { 0.0 };
        let std_dev = var.sqrt();

        self.features
            .iter()
            .map(|feat| {
                if let Some(val) =
                    crate::expanding::basic::eval_basic(feat, n, mean, var, std_dev, state)
                {
                    return val;
                }
                if let Some(val) = crate::expanding::statistics::eval_statistics(
                    feat,
                    n,
                    mean,
                    var,
                    std_dev,
                    m2,
                    m3,
                    m4,
                    mad_sum,
                    median,
                    iqr,
                    entropy,
                    count_a,
                    count_b,
                    max_strike_a,
                    max_strike_b,
                    zc_mean,
                    zc_std,
                    state,
                    running_sorted,
                    full_series,
                ) {
                    return val;
                }
                if let Some(val) = crate::expanding::time_series::eval_time_series(
                    feat,
                    n,
                    mean,
                    var,
                    m2,
                    mac_sum,
                    mc_sum,
                    state,
                    &self.unique_c3_lags,
                    &self.unique_paa_totals,
                    &self.paa_boundaries,
                    &self.unique_autocorr_lags,
                    full_series,
                ) {
                    return val;
                }
                if let Some(val) = crate::expanding::fft::eval_fft(
                    feat,
                    n,
                    &spectrum,
                    &fft_complex,
                    freq_centroid,
                    spectral_decrease,
                    spectral_slope,
                    full_series,
                ) {
                    return val;
                }
                0.0
            })
            .collect()
    }
}
