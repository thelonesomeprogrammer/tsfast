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
                        if data.is_empty() { return 0.0; }
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
                    fft_complex = state.fft_out_buffer[..complex_len].to_vec();
                    spectrum = fft_complex.iter().map(|c| c.norm()).collect();
                } else if self.compute.any([54, 55, 56, 57, 60, 64]) {
                    use realfft::RealFftPlanner;
                    let mut planner = RealFftPlanner::<f32>::new();
                    let r2c = planner.plan_fft_forward(n_total);
                    let mut indata = full_series.to_vec();
                    let mut outdata = r2c.make_output_vec();
                    r2c.process(&mut indata, &mut outdata).unwrap();
                    fft_complex = outdata;
                    spectrum = fft_complex.iter().map(|c| c.norm()).collect();
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
            .map(|feat| match feat {
                Feature::TotalSum => state.total_sum,
                Feature::Mean => mean,
                Feature::Variance => var,
                Feature::Std => std_dev,
                Feature::Min => state.min_value,
                Feature::Max => state.max_value,
                Feature::Median => median,
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
                Feature::Mad => mad_sum / n,
                Feature::Iqr => iqr,
                Feature::Entropy => entropy,
                Feature::Energy => state.energy,
                Feature::Rms | Feature::RootMeanSquare => (state.energy / n).sqrt(),
                Feature::ZeroCrossingRate => state.zcr_count as f32 / n,
                Feature::PeakCount => state.peaks as f32,
                Feature::AutocorrLag1 if var > 1e-9 && n > 1.0 => {
                    let x0 = full_series[0];
                    let xn = full_series[full_series.len() - 1];
                    let cov = state.sum_prod - mean * (2.0 * state.total_sum - x0 - xn) + (n - 1.0) * mean * mean;
                    cov / m2
                }
                Feature::AutocorrFirst1e => {
                    let n_val = full_series.len();
                    if n_val > 1 {
                        let n2 = n_val * 2;
                        let fft_size_ac = crate::common::next_good_fft_size(n2);
                        let mut planner = realfft::RealFftPlanner::<f32>::new();
                        let r2c_ac = planner.plan_fft_forward(fft_size_ac);
                        let c2r_ac = planner.plan_fft_inverse(fft_size_ac);

                        let mut indata = vec![0.0; fft_size_ac];
                        for (i, &v) in full_series.iter().enumerate() {
                            indata[i] = v - mean;
                        }
                        let mut outdata = r2c_ac.make_output_vec();
                        r2c_ac.process(&mut indata, &mut outdata).unwrap();

                        for c in &mut outdata {
                            *c = realfft::num_complex::Complex::new(c.norm_sqr(), 0.0);
                        }

                        let mut outdata_inv = c2r_ac.make_output_vec();
                        c2r_ac.process(&mut outdata, &mut outdata_inv).unwrap();

                        let m2_val = if n > 1.0 { state.energy - (state.total_sum * state.total_sum) / n } else { 0.0 };
                        if m2_val.abs() > 1e-9 {
                            let scale = 1.0 / (fft_size_ac as f32);
                            let threshold = 0.36787944;
                            let mut found = false;
                            let mut first_lag = 0.0;
                            for l in 1..n_val {
                                let val = (outdata_inv[l] * scale) / m2_val;
                                if val < threshold {
                                    first_lag = l as f32;
                                    found = true;
                                    break;
                                }
                            }
                            if found { first_lag } else { 0.0 }
                        } else { 0.0 }
                    } else { 0.0 }
                }
                Feature::MeanAbsChange if n > 1.0 => mac_sum / (n - 1.0),
                Feature::MeanChange if n > 1.0 => mc_sum / (n - 1.0),
                Feature::CidCe => state.sum_sq_diff.sqrt(),
                Feature::Slope => {
                    let mean_i = (n - 1.0) * 0.5;
                    let s_xx = (n * (n * n - 1.0)) / 12.0;
                    let s_xy = state.sum_ix - n * mean_i * mean;
                    if s_xx.abs() > 1e-9 { s_xy / s_xx } else { 0.0 }
                }
                Feature::Intercept => {
                    let mean_i = (n - 1.0) * 0.5;
                    let s_xx = (n * (n * n - 1.0)) / 12.0;
                    let s_xy = state.sum_ix - n * mean_i * mean;
                    let slope = if s_xx.abs() > 1e-9 { s_xy / s_xx } else { 0.0 };
                    mean - slope * mean_i
                }
                Feature::AbsSumChange => mac_sum,
                Feature::CountAboveMean => count_a as f32,
                Feature::CountBelowMean => count_b as f32,
                Feature::LongestStrikeAboveMean => max_strike_a as f32,
                Feature::LongestStrikeBelowMean => max_strike_b as f32,
                Feature::VariationCoefficient if mean.abs() > 1e-9 => std_dev / mean,
                Feature::Auc => state.auc_sum,
                Feature::ZeroCrossingMean => zc_mean,
                Feature::ZeroCrossingStd => zc_std,
                Feature::C3(lag) => {
                    let l_idx = self.unique_c3_lags.iter().position(|&l| l == *lag).unwrap();
                    let l = *lag as usize;
                    if full_series.len() > 2 * l {
                        state.c3_sums[l_idx] / (full_series.len() - 2 * l) as f32
                    } else {
                        0.0
                    }
                }
                Feature::Paa(total, index) => {
                    let t_idx = self
                        .unique_paa_totals
                        .iter()
                        .position(|&t| t == *total)
                        .unwrap();
                    let b = &self.paa_boundaries[t_idx];
                    let start = b[*index as usize];
                    let end = b[*index as usize + 1];
                    if start < end {
                        let sum = if start == 0 {
                            state.prefix_sums[end - 1]
                        } else {
                            state.prefix_sums[end - 1] - state.prefix_sums[start - 1]
                        };
                        sum / (end - start) as f32
                    } else {
                        0.0
                    }
                }
                Feature::AbsMax => state.abs_max,
                Feature::FirstLocMax => state.first_max_idx as f32 / n,
                Feature::LastLocMax => (state.last_max_idx + 1) as f32 / n,
                Feature::FirstLocMin => state.first_min_idx as f32 / n,
                Feature::LastLocMin => (state.last_min_idx + 1) as f32 / n,
                Feature::Autocorr(lag) if var > 1e-9 && full_series.len() > *lag as usize => {
                    let l = *lag as usize;
                    let l_idx = self
                        .unique_autocorr_lags
                        .iter()
                        .position(|&lg| lg == *lag)
                        .unwrap();
                    let n_l = full_series.len() - l;

                    let sum_xi = state.prefix_sums[n_l - 1];
                    let sum_xil =
                        state.prefix_sums[full_series.len() - 1] - state.prefix_sums[l - 1];
                    let cov = state.autocorr_sums[l_idx] as f32 - mean * (sum_xi + sum_xil)
                        + n_l as f32 * mean * mean;

                    if var.abs() > 1e-9 {
                        cov / (n_l as f32 * var)
                    } else {
                        0.0
                    }
                }
                Feature::TimeReversalAsymmetry(lag) if full_series.len() > 2 * *lag as usize => {
                    let l = *lag as usize;
                    let mut sum = 0.0;
                    for i in 0..full_series.len() - 2 * l {
                        sum += full_series[i + 2 * l].powi(2) * full_series[i + l]
                            - full_series[i + l] * full_series[i].powi(2);
                    }
                    sum / (full_series.len() - 2 * l) as f32
                }
                Feature::FftCoefficient(coeff, attr) => {
                    let k = *coeff as usize;
                    let (re, im) = if !fft_complex.is_empty() && k < fft_complex.len() {
                        (fft_complex[k].re, fft_complex[k].im)
                    } else {
                        let mut re = 0.0;
                        let mut im = 0.0;
                        let pi2 = 2.0 * std::f32::consts::PI;
                        for (n_idx, &v) in full_series.iter().enumerate() {
                            let angle = pi2 * k as f32 * n_idx as f32 / n;
                            re += v * angle.cos();
                            im -= v * angle.sin();
                        }
                        (re, im)
                    };
                    match attr {
                        crate::types::FftAttr::Real => re,
                        crate::types::FftAttr::Imag => im,
                        crate::types::FftAttr::Abs => (re * re + im * im).sqrt(),
                        crate::types::FftAttr::Angle => im.atan2(re).to_degrees(),
                    }
                }
                Feature::PartialAutocorr(lag)
                    if var > 1e-9 && full_series.len() > *lag as usize =>
                {
                    let l = *lag as usize;
                    let mut r = Vec::with_capacity(l + 1);
                    for k in 0..=l {
                        if k == 0 {
                            r.push(1.0);
                            continue;
                        }
                        let k_idx = self
                            .unique_autocorr_lags
                            .iter()
                            .position(|&lg| lg == k as u16)
                            .unwrap();
                        let n_k = full_series.len() - k;
                        let sum_xi = state.prefix_sums[n_k - 1];
                        let sum_xk =
                            state.prefix_sums[full_series.len() - 1] - state.prefix_sums[k - 1];
                        let cov = state.autocorr_sums[k_idx] as f32 - mean * (sum_xi + sum_xk)
                            + n_k as f32 * mean * mean;
                        r.push(cov / (n_k as f32 * var));
                    }
                    let mut phi = vec![vec![0.0; l + 1]; l + 1];
                    let mut error = r[0];
                    if error.abs() < 1e-9 {
                        return 0.0;
                    }
                    phi[1][1] = r[1] / r[0];
                    error *= 1.0 - phi[1][1] * phi[1][1];
                    for k in 1..l {
                        let mut sum = 0.0;
                        for i in 1..=k {
                            sum += phi[k][i] * r[k + 1 - i];
                        }
                        phi[k + 1][k + 1] = (r[k + 1] - sum) / error;
                        for i in 1..=k {
                            phi[k + 1][i] = phi[k][i] - phi[k + 1][k + 1] * phi[k][k + 1 - i];
                        }
                        error *= 1.0 - phi[k + 1][k + 1] * phi[k + 1][k + 1];
                        if error.abs() < 1e-9 {
                            break;
                        }
                    }
                    phi[l][l]
                }
                Feature::AggLinearTrend(attr, chunk_len, func)
                    if full_series.len() >= *chunk_len as usize =>
                {
                    let cl = *chunk_len as usize;
                    let mut agg_series = Vec::new();
                    for chunk in full_series.chunks_exact(cl) {
                        let val = match func {
                            crate::types::AggFunc::Max => {
                                chunk.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b))
                            }
                            crate::types::AggFunc::Min => {
                                chunk.iter().fold(f32::INFINITY, |a, &b| a.min(b))
                            }
                            crate::types::AggFunc::Mean => chunk.iter().sum::<f32>() / cl as f32,
                            crate::types::AggFunc::Var => {
                                let m = chunk.iter().sum::<f32>() / cl as f32;
                                chunk.iter().map(|&v| (v - m).powi(2)).sum::<f32>() / cl as f32
                            }
                        };
                        agg_series.push(val);
                    }
                    let m_n = agg_series.len() as f32;
                    if m_n < 2.0 {
                        return 0.0;
                    }
                    let m_sum_x: f32 = (0..agg_series.len()).map(|i| i as f32).sum();
                    let m_sum_y: f32 = agg_series.iter().sum();
                    let m_sum_xx: f32 = (0..agg_series.len()).map(|i| (i as f32).powi(2)).sum();
                    let m_sum_xy: f32 = agg_series
                        .iter()
                        .enumerate()
                        .map(|(i, &v)| i as f32 * v)
                        .sum();
                    let s_xx = m_sum_xx - (m_sum_x * m_sum_x) / m_n;
                    let s_xy = m_sum_xy - (m_sum_x * m_sum_y) / m_n;
                    let slope = if s_xx.abs() > 1e-9 { s_xy / s_xx } else { 0.0 };
                    let intercept = (m_sum_y - slope * m_sum_x) / m_n;
                    match attr {
                        crate::types::AggAttr::Slope => slope,
                        crate::types::AggAttr::Intercept => intercept,
                        crate::types::AggAttr::Stderr | crate::types::AggAttr::RValue => {
                            let mut ss_res = 0.0;
                            let mut ss_tot = 0.0;
                            let m_y = m_sum_y / m_n;
                            for (i, &y) in agg_series.iter().enumerate() {
                                let y_hat = intercept + slope * i as f32;
                                ss_res += (y - y_hat).powi(2);
                                ss_tot += (y - m_y).powi(2);
                            }
                            if matches!(attr, crate::types::AggAttr::Stderr) {
                                if m_n > 2.0 && s_xx.abs() > 1e-9 {
                                    (ss_res / (m_n - 2.0) / s_xx).sqrt()
                                } else {
                                    0.0
                                }
                            } else {
                                if ss_tot > 1e-9 {
                                    (1.0 - ss_res / ss_tot).sqrt() * slope.signum()
                                } else {
                                    0.0
                                }
                            }
                        }
                        _ => 0.0,
                    }
                }
                Feature::ApproxEntropy(m, r_bits) if full_series.len() > *m as usize + 1 => {
                    let m_val = *m as usize;
                    let r = f32::from_bits(*r_bits);
                    fn phi(m: usize, r: f32, data: &[f32]) -> f32 {
                        let n = data.len();
                        let mut result = 0.0;
                        for i in 0..n - m + 1 {
                            let mut count = 0;
                            for j in 0..n - m + 1 {
                                let mut max_diff: f32 = 0.0;
                                for k in 0..m {
                                    max_diff = max_diff.max((data[i + k] - data[j + k]).abs());
                                }
                                if max_diff <= r {
                                    count += 1;
                                }
                            }
                            result += (count as f32 / (n - m + 1) as f32).ln();
                        }
                        result / (n - m + 1) as f32
                    }
                    phi(m_val, r, full_series) - phi(m_val + 1, r, full_series)
                }
                Feature::Quantile(q_bits) => {
                    let q = f32::from_bits(*q_bits);
                    if running_sorted.is_empty() {
                        0.0
                    } else if running_sorted.len() == 1 {
                        running_sorted[0]
                    } else {
                        let n_len = running_sorted.len();
                        let idx = q * (n_len as f32 - 1.0);
                        let i = idx.floor() as usize;
                        let f = idx - i as f32;
                        if i >= n_len - 1 {
                            running_sorted[n_len - 1]
                        } else {
                            (1.0 - f) * running_sorted[i] + f * running_sorted[i + 1]
                        }
                    }
                }
                Feature::BenfordCorrelation => {
                    let mut counts = [0.0; 9];
                    for &v in full_series {
                        let mut abs_v = v.abs();
                        if abs_v > 0.0 {
                            while abs_v < 1.0 { abs_v *= 10.0; }
                            while abs_v >= 10.0 { abs_v /= 10.0; }
                            let first_digit = abs_v.floor() as usize;
                            if (1..=9).contains(&first_digit) {
                                counts[first_digit - 1] += 1.0;
                            }
                        }
                    }
                    let total: f32 = counts.iter().sum();
                    if total > 0.0 {
                        let p: Vec<f32> = counts.iter().map(|&c| c / total).collect();
                        let b: Vec<f32> = (1..10).map(|i| (1.0 + 1.0 / i as f32).log10()).collect();
                        let mu_p = p.iter().sum::<f32>() / 9.0;
                        let mu_b = b.iter().sum::<f32>() / 9.0;
                        let mut num = 0.0;
                        let mut den_p = 0.0;
                        let mut den_b = 0.0;
                        for i in 0..9 {
                            num += (p[i] - mu_p) * (b[i] - mu_b);
                            den_p += (p[i] - mu_p).powi(2);
                            den_b += (b[i] - mu_b).powi(2);
                        }
                        if den_p > 0.0 && den_b > 0.0 {
                            num / (den_p * den_b).sqrt()
                        } else { 0.0 }
                    } else { 0.0 }
                }
                Feature::SumOfReoccurringValues => {
                    use std::collections::HashMap;
                    let mut counts = HashMap::new();
                    for &v in full_series {
                        let bits = v.to_bits();
                        *counts.entry(bits).or_insert(0) += 1;
                    }
                    counts
                        .iter()
                        .filter(|&(_, &count)| count > 1)
                        .map(|(&bits, _)| f32::from_bits(bits))
                        .sum()
                }
                Feature::SumOfReoccurringDataPoints => {
                    use std::collections::HashMap;
                    let mut counts = HashMap::new();
                    for &v in full_series {
                        let bits = v.to_bits();
                        *counts.entry(bits).or_insert(0) += 1;
                    }
                    counts
                        .iter()
                        .filter(|&(_, &count)| count > 1)
                        .map(|(&bits, &count)| f32::from_bits(bits) * count as f32)
                        .sum()
                }
                Feature::MeanNAbsoluteMax(n_max) => {
                    let mut abs_vals: Vec<f32> = full_series.iter().map(|v| v.abs()).collect();
                    abs_vals.sort_unstable_by(|a, b| {
                        b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
                    });
                    let count = (*n_max as usize).min(abs_vals.len());
                    if count > 0 {
                        abs_vals.iter().take(count).sum::<f32>() / count as f32
                    } else {
                        0.0
                    }
                }
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
                Feature::SpectralCentroid => freq_centroid,
                Feature::SpectralDecrease => spectral_decrease,
                Feature::SpectralSlope => spectral_slope,
                Feature::SignalDistance => {
                    let mut dist = 0.0;
                    for i in 1..full_series.len() {
                        dist += ((full_series[i] - full_series[i - 1]).powi(2) + 1.0).sqrt();
                    }
                    dist
                }
                Feature::SpectralDistance => {
                    if !spectrum.is_empty() {
                        let m = spectrum.iter().sum::<f32>() / spectrum.len() as f32;
                        spectrum
                            .iter()
                            .map(|&s| (s - m).powi(2))
                            .sum::<f32>()
                            .sqrt()
                    } else {
                        0.0
                    }
                }
                Feature::WaveletFeatures(_w_bits, f_type) => {
                    if full_series.len() >= 2 {
                        let mut sum = 0.0;
                        for i in (0..full_series.len() - 1).step_by(2) {
                            if *f_type == 0 {
                                sum += (full_series[i] - full_series[i + 1]).abs();
                            } else {
                                sum += (full_series[i] - full_series[i + 1]).powi(2);
                            }
                        }
                        if *f_type == 0 {
                            sum / (full_series.len() / 2) as f32
                        } else {
                            (sum / (full_series.len() / 2) as f32).sqrt()
                        }
                    } else {
                        0.0
                    }
                }
                Feature::SpectrogramCoefficients(_t, f_bits) => {
                    if !spectrum.is_empty() {
                        let target_freq = f32::from_bits(*f_bits as u32);
                        let fs = 100.0;
                        let n_fft = (spectrum.len() - 1) * 2;
                        let freq_step = fs / n_fft as f32;
                        let idx = (target_freq / freq_step).round() as usize;
                        let idx = idx.min(spectrum.len() - 1);
                        spectrum[idx]
                    } else {
                        0.0
                    }
                }
                _ => 0.0,
            })
            .collect()
    }
}
