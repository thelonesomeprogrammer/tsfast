use crate::common::{ColumnState, LANES};
use crate::types::{FastBitArray, Feature};
use std::simd::cmp::SimdPartialOrd;
use std::simd::f32x4;
use std::simd::num::SimdFloat;

pub(crate) struct StaticEngine<'a> {
    pub(crate) compute: FastBitArray,
    pub(crate) features: &'a [Feature],
    pub(crate) unique_paa_totals: &'a [u16],
    pub(crate) unique_c3_lags: &'a [u16],
    pub(crate) paa_boundaries: &'a [Vec<usize>],
}

impl<'a> StaticEngine<'a> {
    #[inline(always)]
    pub(crate) fn process_column(&self, values: &[f32]) -> Vec<f32> {
        let n = values.len() as f32;
        if n == 0.0 {
            return vec![0.0; self.features.len()];
        }

        let mut state = ColumnState::new(self.unique_paa_totals, self.unique_c3_lags, values[0]);

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
                        state.c3_sums[l_idx] += (chunk * chunk_il * chunk_i2l).reduce_sum() as f64;
                    } else {
                        for j in 0..LANES {
                            let idx = global_idx + j;
                            if idx >= 2 * l {
                                state.c3_sums[l_idx] +=
                                    (values[idx] * values[idx - l] * values[idx - 2 * l]) as f64;
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
                        state.c3_sums[l_idx] += (val * values[i - l] * values[i - 2 * l]) as f64;
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

        let mut m2 = 0.0;
        let mut m3 = 0.0;
        let mut m4 = 0.0;
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

        if self.compute[37] {
            if self.compute[6] {
                let mut copy = values.to_vec();
                let n_size = copy.len();
                let mid = n_size / 2;
                median = *copy
                    .select_nth_unstable_by(mid, |a, b| {
                        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .1;
                if self.compute[10] {
                    let q1_idx = (n_size as f32 * 0.25) as usize;
                    let q3_idx = (n_size as f32 * 0.75) as usize;
                    let q1 = *copy
                        .select_nth_unstable_by(q1_idx, |a, b| {
                            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .1;
                    let q3 = *copy
                        .select_nth_unstable_by(q3_idx, |a, b| {
                            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .1;
                    iqr = q3 - q1;
                }
                if self.compute[11] {
                    let range = state.max_value - state.min_value;
                    if range > 1e-9 {
                        let bins = 10;
                        let mut counts = vec![0usize; bins];
                        for &v in values {
                            let b =
                                (((v - state.min_value) / range) * (bins as f32 - 1.0)) as usize;
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
            }

            let mean_vec = f32x4::splat(mean);
            let mut m2_vec = f32x4::splat(0.0);
            let mut m3_vec = f32x4::splat(0.0);
            let mut m4_vec = f32x4::splat(0.0);
            let mut mad_sum_vec = f32x4::splat(0.0);

            for chunk in values.chunks_exact(LANES) {
                let c = f32x4::from_slice(chunk);
                let diff = c - mean_vec;
                if self.compute[2] {
                    let d2 = diff * diff;
                    m2_vec += d2;
                    if self.compute[7] {
                        m3_vec += d2 * diff;
                    }
                    if self.compute[8] {
                        m4_vec += d2 * d2;
                    }
                }
                if self.compute[9] {
                    mad_sum_vec += diff.abs();
                }
            }
            m2 = m2_vec.reduce_sum();
            m3 = m3_vec.reduce_sum();
            m4 = m4_vec.reduce_sum();
            mad_sum = mad_sum_vec.reduce_sum();

            let rem_start = (values.len() / LANES) * LANES;
            for &val in &values[rem_start..] {
                let diff = val - mean;
                if self.compute[2] {
                    let d2 = diff * diff;
                    m2 += d2;
                    if self.compute[7] {
                        m3 += d2 * diff;
                    }
                    if self.compute[8] {
                        m4 += d2 * d2;
                    }
                }
                if self.compute[9] {
                    mad_sum += diff.abs();
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
                Feature::Skew if var > 1e-9 => (m3 / n) / var.powf(1.5),
                Feature::Kurtosis if var > 1e-9 => (m4 / n) / (var * var) - 3.0,
                Feature::Mad => mad_sum / n,
                Feature::Iqr => iqr,
                Feature::Entropy => entropy,
                Feature::Energy => state.energy,
                Feature::Rms | Feature::RootMeanSquare => (state.energy / n).sqrt(),
                Feature::ZeroCrossingRate => state.zcr_count as f32 / n,
                Feature::PeakCount => state.peaks as f32,
                Feature::AutocorrLag1 if var > 1e-9 => {
                    (state.sum_prod / (n - 1.0) - mean * mean) / var
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
                        (state.c3_sums[l_idx] / (values.len() - 2 * l) as f64) as f32
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
                _ => 0.0,
            })
            .collect()
    }
}
