use crate::common::{ColumnState, LANES};
use crate::types::{FastBitArray, Feature};
use realfft::{RealFftPlanner, RealToComplex};
use std::collections::HashMap;
use std::simd::cmp::SimdPartialOrd;
use std::simd::f32x4;
use std::simd::num::SimdFloat;
use std::sync::Arc;

pub(crate) struct StaticEngine<'a> {
    pub(crate) compute: FastBitArray,
    pub(crate) features: &'a [Feature],
    pub(crate) unique_paa_totals: &'a [u16],
    pub(crate) unique_c3_lags: &'a [u16],
    pub(crate) paa_boundaries: &'a [Vec<usize>],
    pub(crate) r2c: Option<Arc<dyn RealToComplex<f32>>>,
    pub(crate) fft_size: usize,
}

impl<'a> StaticEngine<'a> {
    #[inline(always)]
    pub(crate) fn process_column(&self, values: &[f32]) -> Vec<f32> {
        let n = values.len() as f32;
        if n == 0.0 {
            return vec![0.0; self.features.len()];
        }

        let mut state =
            ColumnState::new(self.unique_paa_totals, self.unique_c3_lags, &[], values[0]);

        // SIMD Pass 1
        let rem_start = self.process_simd_chunks(values, &mut state);

        // Remainder Pass 1
        self.process_remainder(values, rem_start, &mut state);

        // Post-processing and Pass 2
        self.finalize_results(values, n, state)
    }

    #[inline(always)]
    fn process_simd_chunks(&self, values: &[f32], state: &mut ColumnState) -> usize {
        let chunks = values.chunks_exact(LANES);
        let rem_start = (values.len() / LANES) * LANES;

        for (chunk_idx, i) in chunks.enumerate() {
            let chunk = f32x4::from_slice(i);
            let global_idx = chunk_idx * LANES;
            let offset = global_idx as f32;

            if self.compute[0] {
                state.total_sum += chunk.reduce_sum();
            }
            if self.compute[4] {
                state.min_value = state.min_value.min(chunk.reduce_min());
            }
            if self.compute[5] {
                state.max_value = state.max_value.max(chunk.reduce_max());
            }
            if self.compute[12] {
                let sq = chunk * chunk;
                state.energy += sq.reduce_sum();
                if self.compute[7] {
                    state.sum_cubes += (sq * chunk).reduce_sum();
                }
                if self.compute[8] {
                    state.sum_quads += (sq * sq).reduce_sum();
                }
            }

            if self.compute[38] {
                state.abs_max = state.abs_max.max(chunk.abs().reduce_max());
            }

            if self.compute[61] {
                state.abs_sum += chunk.abs().reduce_sum();
            }

            if self.compute.any([15, 17, 18, 19, 20, 31]) {
                let shifted = f32x4::from_array([state.prev_last, i[0], i[1], i[2]]);
                let diff = chunk - shifted;
                if self.compute[18] {
                    state.mac_sum_vec += diff.abs();
                }
                if self.compute[19] {
                    state.mc_sum_vec += diff;
                }
                if self.compute[20] {
                    state.sum_sq_diff += (diff * diff).reduce_sum();
                }
                if self.compute[17] {
                    state.sum_prod += (chunk * shifted).reduce_sum();
                }
                if self.compute[31] {
                    state.auc_sum += (chunk + shifted).reduce_sum() * 0.5;
                }
                if self.compute[15] {
                    let signs = chunk.simd_lt(f32x4::splat(0.0));
                    let prev_signs = shifted.simd_lt(f32x4::splat(0.0));
                    let mask = (signs ^ prev_signs).to_bitmask();
                    state.zcr_count += mask.count_ones();
                    if self.compute[36] {
                        for bit in 0..4 {
                            if (mask >> bit) & 1 == 1 {
                                state.zc_indices.push(offset + bit as f32);
                            }
                        }
                    }
                }
                state.prev_last = i[LANES - 1];
            }

            if self.compute[16] {
                let left = f32x4::from_array([
                    if global_idx > 0 {
                        values[global_idx - 1]
                    } else {
                        values[0]
                    },
                    values[global_idx],
                    values[global_idx + 1],
                    values[global_idx + 2],
                ]);
                let right = f32x4::from_array([
                    values[global_idx + 1],
                    values[global_idx + 2],
                    values[global_idx + 3],
                    if global_idx + 4 < values.len() {
                        values[global_idx + 4]
                    } else {
                        values[values.len() - 1]
                    },
                ]);
                let mask = chunk.simd_gt(left) & chunk.simd_gt(right);
                state.peaks += mask.to_bitmask().count_ones();
            }

            if self.compute[21] {
                let indices = f32x4::from_array([offset, offset + 1.0, offset + 2.0, offset + 3.0]);
                state.sum_ix += (indices * chunk).reduce_sum();
            }

            if self.compute[23] {
                for (t_idx, _total) in self.unique_paa_totals.iter().enumerate() {
                    let b = &self.paa_boundaries[t_idx];
                    let seg_idx = &mut state.current_paa_segs[t_idx];
                    if global_idx + LANES <= b[*seg_idx + 1] {
                        state.paa_sums[t_idx][*seg_idx] += chunk.reduce_sum();
                    } else {
                        for (j, v) in i.iter().enumerate().take(LANES) {
                            let idx = global_idx + j;
                            while *seg_idx < state.paa_sums[t_idx].len() - 1
                                && idx >= b[*seg_idx + 1]
                            {
                                *seg_idx += 1;
                            }
                            state.paa_sums[t_idx][*seg_idx] += v;
                        }
                    }
                }
            }

            if self.compute[30] {
                for (l_idx, &lag) in self.unique_c3_lags.iter().enumerate() {
                    let l = lag as usize;
                    if global_idx >= 2 * l {
                        let chunk_il =
                            f32x4::from_slice(&values[global_idx - l..global_idx - l + 4]);
                        let chunk_i2l =
                            f32x4::from_slice(&values[global_idx - 2 * l..global_idx - 2 * l + 4]);
                        state.c3_sums[l_idx] += (chunk * chunk_il * chunk_i2l).reduce_sum();
                    } else {
                        for j in 0..LANES {
                            let idx = global_idx + j;
                            if idx >= 2 * l {
                                state.c3_sums[l_idx] +=
                                    values[idx] * values[idx - l] * values[idx - 2 * l];
                            }
                        }
                    }
                }
            }
        }
        rem_start
    }

    #[inline(always)]
    fn process_remainder(&self, values: &[f32], rem_start: usize, state: &mut ColumnState) {
        for i in rem_start..values.len() {
            let val = values[i];
            if self.compute[0] {
                state.total_sum += val;
            }
            if self.compute[4] {
                state.min_value = state.min_value.min(val);
            }
            if self.compute[5] {
                state.max_value = state.max_value.max(val);
            }
            if self.compute[38] {
                state.abs_max = state.abs_max.max(val.abs());
            }
            if self.compute[61] {
                state.abs_sum += val.abs();
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
            if i > 0 {
                let diff = val - values[i - 1];
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
                    state.sum_prod += val * values[i - 1];
                }
                if self.compute[31] {
                    state.auc_sum += (val + values[i - 1]) * 0.5;
                }
                if self.compute[15] && (val < 0.0) != (values[i - 1] < 0.0) {
                    state.zcr_count += 1;
                    if self.compute[36] {
                        state.zc_indices.push(i as f32);
                    }
                }
            }
            if self.compute[16]
                && i > 0
                && i < values.len() - 1
                && val > values[i - 1]
                && val > values[i + 1]
            {
                state.peaks += 1;
            }
            if self.compute[21] {
                state.sum_ix += (i as f32) * val;
            }
            if self.compute[23] {
                for (t_idx, _total) in self.unique_paa_totals.iter().enumerate() {
                    let b = &self.paa_boundaries[t_idx];
                    let seg_idx = &mut state.current_paa_segs[t_idx];
                    while *seg_idx < state.paa_sums[t_idx].len() - 1 && i >= b[*seg_idx + 1] {
                        *seg_idx += 1;
                    }
                    state.paa_sums[t_idx][*seg_idx] += val;
                }
            }
            if self.compute[30] {
                for (l_idx, &lag) in self.unique_c3_lags.iter().enumerate() {
                    let l = lag as usize;
                    if i >= 2 * l {
                        state.c3_sums[l_idx] += val * values[i - l] * values[i - 2 * l];
                    }
                }
            }
        }
    }

    #[inline(always)]
    fn finalize_results(&self, values: &[f32], n: f32, state: ColumnState) -> Vec<f32> {
        let mean = state.total_sum / n;
        let mac_sum = state.mac_sum_vec.reduce_sum();
        let mc_sum = state.mc_sum_vec.reduce_sum();

        // One-pass moments from power sums
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
        let mut first_max_idx = 0;
        let mut last_max_idx = 0;
        let mut first_min_idx = 0;
        let mut last_min_idx = 0;

        let mut sorted_copy: Option<Vec<f32>> = None;
        if self.compute.any([6, 10, 11, 49]) {
            let mut copy = values.to_vec();
            copy.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            
            if self.compute[6] {
                let n_len = copy.len();
                if n_len > 0 {
                    if n_len % 2 == 1 {
                        median = copy[n_len / 2];
                    } else {
                        median = (copy[n_len / 2] + copy[n_len / 2 - 1]) / 2.0;
                    }
                }
            }
            if self.compute[10] {
                let n_f = n as f32;
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
                iqr = get_q(0.75, &copy) - get_q(0.25, &copy);
            }
            if self.compute[11] {
                let range = state.max_value - state.min_value;
                if range > 1e-9 {
                    let bins = 10;
                    let mut counts = vec![0usize; bins];
                    for &v in values {
                        let b = (((v - state.min_value) / range) * (bins as f32 - 1.0)) as usize;
                        counts[b.min(bins - 1)] += 1;
                    }
                    for &c in &counts {
                        if c > 0 {
                            let p = c as f32 / n;
                            entropy -= p * p.ln();
                        }
                    }
                }
            }
            if self.compute[49] {
                sorted_copy = Some(copy);
            }
        }

        let mut benford_corr = 0.0;
        if self.compute[51] {
            let mut counts = [0.0; 9];
            for &v in values {
                let mut abs_v = v.abs();
                if abs_v > 0.0 {
                    while abs_v < 1.0 {
                        abs_v *= 10.0;
                    }
                    while abs_v >= 10.0 {
                        abs_v /= 10.0;
                    }
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
                    benford_corr = num / (den_p * den_b).sqrt();
                }
            }
        }

        let mut spectrum = Vec::new();
        let mut fft_complex = Vec::new();
        let mut freq_centroid = 0.0;
        let mut spectral_decrease = 0.0;
        let mut spectral_slope = 0.0;

        if self.compute.any_fft() {
            let (complex_data, spec) = if let Some(r2c) = &self.r2c {
                let mut indata = vec![0.0; self.fft_size];
                indata[..values.len()].copy_from_slice(values);
                let mut outdata = r2c.make_output_vec();
                r2c.process(&mut indata, &mut outdata).unwrap();
                let s = outdata.iter().map(|c| c.norm()).collect();
                (outdata, s)
            } else if self.compute.any([54, 55, 56, 57, 60]) {
                let mut planner = RealFftPlanner::<f32>::new();
                let r2c = planner.plan_fft_forward(values.len());
                let mut indata = values.to_vec();
                let mut outdata = r2c.make_output_vec();
                r2c.process(&mut indata, &mut outdata).unwrap();
                let s = outdata.iter().map(|c| c.norm()).collect();
                (outdata, s)
            } else {
                (Vec::new(), Vec::new())
            };
            fft_complex = complex_data;
            spectrum = spec;

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

        let mut signal_dist = 0.0;
        if self.compute[58] {
            for i in 1..values.len() {
                signal_dist += ((values[i] - values[i - 1]).powi(2) + 1.0).sqrt();
            }
        }

        let mut fft_autocorr = Vec::new();
        if self.compute.any([43, 44]) && n > 1.0 {
            let n2 = values.len() * 2;
            let fft_size_ac = crate::common::next_good_fft_size(n2);
            let mut planner = realfft::RealFftPlanner::<f32>::new();
            let r2c_ac = planner.plan_fft_forward(fft_size_ac);
            let c2r_ac = planner.plan_fft_inverse(fft_size_ac);

            let mut indata = vec![0.0; fft_size_ac];
            for (i, &v) in values.iter().enumerate() {
                indata[i] = v - mean;
            }
            let mut outdata = r2c_ac.make_output_vec();
            r2c_ac.process(&mut indata, &mut outdata).unwrap();

            for c in &mut outdata {
                *c = realfft::num_complex::Complex::new(c.norm_sqr(), 0.0);
            }

            let mut outdata_inv = c2r_ac.make_output_vec();
            c2r_ac.process(&mut outdata, &mut outdata_inv).unwrap();

            let m2_ac = m2; // var * (n - 1.0)
            if m2_ac.abs() > 1e-9 {
                let scale = 1.0 / (fft_size_ac as f32);
                fft_autocorr = outdata_inv
                    .into_iter()
                    .take(values.len())
                    .map(|v| (v * scale) / m2_ac)
                    .collect();
            }
        }

        if self.compute[37] {
            if self.compute.any([39, 40, 41, 42]) {
                let mut found_max = false;
                let mut found_min = false;
                for (i, &v) in values.iter().enumerate() {
                    if v == state.max_value {
                        if !found_max {
                            first_max_idx = i;
                            found_max = true;
                        }
                        last_max_idx = i;
                    }
                    if v == state.min_value {
                        if !found_min {
                            first_min_idx = i;
                            found_min = true;
                        }
                        last_min_idx = i;
                    }
                }
            }

            if self.compute.any([9, 25, 26, 27, 28]) {
                let mean_vec = f32x4::splat(mean);
                let mut mad_sum_vec = f32x4::splat(0.0);

                for chunk in values.chunks_exact(LANES) {
                    let c = f32x4::from_slice(chunk);
                    if self.compute[9] {
                        mad_sum_vec += (c - mean_vec).abs();
                    }
                }
                mad_sum = mad_sum_vec.reduce_sum();

                let rem_start = (values.len() / LANES) * LANES;
                for &val in &values[rem_start..] {
                    if self.compute[9] {
                        mad_sum += (val - mean).abs();
                    }
                }

                if self.compute.any([25, 26, 27, 28]) {
                    for &val in values {
                        if val > mean {
                            count_a += 1;
                            current_strike_a += 1;
                            max_strike_a = max_strike_a.max(current_strike_a);
                            current_strike_b = 0;
                        } else if val < mean {
                            count_b += 1;
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
                    let x0 = values[0];
                    let xn = values[values.len() - 1];
                    let cov = state.sum_prod - mean * (2.0 * state.total_sum - x0 - xn) + (n - 1.0) * mean * mean;
                    cov / m2
                }
                Feature::AutocorrFirst1e => {
                    let threshold = 0.36787944;
                    if !fft_autocorr.is_empty() {
                        let mut found = false;
                        let mut first_lag = 0.0;
                        // Skip lag 0 which is always 1.0
                        for (l, &val) in fft_autocorr.iter().enumerate().skip(1) {
                            if val < threshold {
                                first_lag = l as f32;
                                found = true;
                                break;
                            }
                        }
                        if found { first_lag } else { 0.0 }
                    } else if var > 1e-9 && n > 1.0 {
                        // Efficient non-FFT loop
                        let mut first_lag = 0.0;
                        let m2_val = m2;
                        for l in 1..values.len() {
                            let mut sum = 0.0;
                            for i in 0..values.len() - l {
                                sum += (values[i] - mean) * (values[i + l] - mean);
                            }
                            if sum / m2_val < threshold {
                                first_lag = l as f32;
                                break;
                            }
                        }
                        first_lag
                    } else {
                        0.0
                    }
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
                    if values.len() > 2 * l {
                        state.c3_sums[l_idx] / (values.len() - 2 * l) as f32
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
                        state.paa_sums[t_idx][*index as usize] / (end - start) as f32
                    } else {
                        0.0
                    }
                }
                Feature::AbsMax => state.abs_max,
                Feature::FirstLocMax => first_max_idx as f32 / n,
                Feature::LastLocMax => (last_max_idx + 1) as f32 / n,
                Feature::FirstLocMin => first_min_idx as f32 / n,
                Feature::LastLocMin => (last_min_idx + 1) as f32 / n,
                Feature::Autocorr(lag) => {
                    let l = *lag as usize;
                    if var > 1e-9 && values.len() > l {
                        let mut sum = 0.0;
                        for i in 0..values.len() - l {
                            sum += (values[i] - mean) * (values[i + l] - mean);
                        }
                        let m2 = var * (n - 1.0);
                        if m2.abs() > 1e-9 { sum / m2 } else { 0.0 }
                    } else {
                        0.0
                    }
                }
                Feature::TimeReversalAsymmetry(lag) if values.len() > 2 * *lag as usize => {
                    let l = *lag as usize;
                    let mut sum = 0.0;
                    for i in 0..values.len() - 2 * l {
                        sum += values[i + 2 * l].powi(2) * values[i + l]
                            - values[i + l] * values[i].powi(2);
                    }
                    sum / (values.len() - 2 * l) as f32
                }
                Feature::FftCoefficient(coeff, attr) => {
                    let k = *coeff as usize;
                    let (re, im) = if !fft_complex.is_empty() && k < fft_complex.len() {
                        (fft_complex[k].re, fft_complex[k].im)
                    } else {
                        let mut re = 0.0;
                        let mut im = 0.0;
                        let pi2 = 2.0 * std::f32::consts::PI;
                        for (n_idx, &v) in values.iter().enumerate() {
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
                Feature::PartialAutocorr(lag) if var > 1e-9 && values.len() > *lag as usize => {
                    let l = *lag as usize;
                    let mut r = Vec::with_capacity(l + 1);
                    let m2 = var * (n - 1.0);
                    for k in 0..=l {
                        let mut sum = 0.0;
                        for i in 0..values.len() - k {
                            sum += (values[i] - mean) * (values[i + k] - mean);
                        }
                        r.push(sum / m2);
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
                    if values.len() >= *chunk_len as usize =>
                {
                    let cl = *chunk_len as usize;
                    let mut agg_series = Vec::new();
                    for chunk in values.chunks_exact(cl) {
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
                Feature::ApproxEntropy(m, r_bits) if values.len() > *m as usize + 1 => {
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
                    phi(m_val, r, values) - phi(m_val + 1, r, values)
                }
                Feature::Quantile(q_bits) => {
                    let q = f32::from_bits(*q_bits);
                    let mut copy = if let Some(c) = sorted_copy.take() {
                        c
                    } else {
                        values.to_vec()
                    };
                    if copy.is_empty() {
                        0.0
                    } else if copy.len() == 1 {
                        copy[0]
                    } else {
                        let n_len = copy.len();
                        let idx = q * (n_len as f32 - 1.0);
                        let i = idx.floor() as usize;
                        let f = idx - i as f32;
                        
                        // We need i and i+1 to be valid.
                        // select_nth_unstable only gives us one.
                        // For linear interpolation, we need to sort or at least find two.
                        // Since we already might have a sorted copy, let's just sort if not.
                        copy.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                        
                        if i >= n_len - 1 {
                            copy[n_len - 1]
                        } else {
                            (1.0 - f) * copy[i] + f * copy[i + 1]
                        }
                    }
                }
                Feature::IndexMassQuantile(q_bits) => {
                    let q = f32::from_bits(*q_bits);
                    let mut current_abs_sum = 0.0;
                    let target = q * state.abs_sum;
                    let mut res = 1.0;
                    for (i, &v) in values.iter().enumerate() {
                        current_abs_sum += v.abs();
                        if current_abs_sum >= target {
                            res = (i + 1) as f32 / n;
                            break;
                        }
                    }
                    res
                }
                Feature::BenfordCorrelation => benford_corr,
                Feature::SumOfReoccurringValues => {
                    let mut counts = HashMap::new();
                    for &v in values {
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
                    let mut counts = HashMap::new();
                    for &v in values {
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
                    let mut abs_vals: Vec<f32> = values.iter().map(|v| v.abs()).collect();
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
                Feature::HumanRangeEnergy(fs_bits) if !spectrum.is_empty() => {
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
                }
                Feature::MaxLangevinFixedPoint(m, r_bits) => {
                    let _m_val = *m as usize;
                    let _r = f32::from_bits(*r_bits);
                    0.0
                }
                Feature::SpectralCentroid => freq_centroid,
                Feature::SpectralDecrease => spectral_decrease,
                Feature::SpectralSlope => spectral_slope,
                Feature::SignalDistance => signal_dist,
                Feature::SpectralDistance if !spectrum.is_empty() => {
                    let m = spectrum.iter().sum::<f32>() / spectrum.len() as f32;
                    spectrum
                        .iter()
                        .map(|&s| (s - m).powi(2))
                        .sum::<f32>()
                        .sqrt()
                }
                Feature::WaveletFeatures(_w_bits, f_type) if values.len() >= 2 => {
                    let mut sum = 0.0;
                    for i in (0..values.len() - 1).step_by(2) {
                        if *f_type == 0 {
                            sum += (values[i] - values[i + 1]).abs();
                        } else {
                            sum += (values[i] - values[i + 1]).powi(2);
                        }
                    }
                    if *f_type == 0 {
                        sum / (values.len() / 2) as f32
                    } else {
                        (sum / (values.len() / 2) as f32).sqrt()
                    }
                }
                Feature::SpectrogramCoefficients(_t, f_bits) if !spectrum.is_empty() => {
                    let target_freq = f32::from_bits(*f_bits as u32);
                    // We don't have fs here easily unless we passed it.
                    // Assuming default fs=100 for now if it looks like a frequency.
                    let fs = 100.0;
                    let n_fft = (spectrum.len() - 1) * 2;
                    let freq_step = fs / n_fft as f32;
                    let idx = (target_freq / freq_step).round() as usize;
                    let idx = idx.min(spectrum.len() - 1);
                    spectrum[idx]
                }
                _ => 0.0,
            })
            .collect()
    }
}
