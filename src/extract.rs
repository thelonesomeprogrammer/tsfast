use arrow::array::{ArrayRef, Float32Array, RecordBatch};
use arrow::datatypes::DataType;
use arrow::datatypes::{Field, Schema};
use arrow::pyarrow::PyArrowType;
use bitvec::array::BitArray;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::simd::cmp::SimdPartialOrd;
use std::simd::f32x4;
use std::simd::num::SimdFloat;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Copy)]
pub enum ArrowFeature {
    TotalSum,
    Mean,
    Variance,
    Std,
    Min,
    Max,
    Median,
    Skew,
    Kurtosis,
    Mad,
    Iqr,
    Entropy,
    Energy,
    Rms,
    RootMeanSquare,
    ZeroCrossingRate,
    PeakCount,
    AutocorrLag1,
    MeanAbsChange,
    MeanChange,
    CidCe,
    Slope,
    Intercept,
    Paa(u16, u16),
    AbsSumChange,
    CountAboveMean,
    CountBelowMean,
    LongestStrikeAboveMean,
    LongestStrikeBelowMean,
    VariationCoefficient,
    C3(u16),
    Auc,
    SlopeSignChange,
    TurningPoints,
    ZeroCrossingMean,
    ZeroCrossingStd,
}

#[pyclass]
pub struct ArrowExtractor {
    pub features: Vec<ArrowFeature>,
    pub compute: BitArray<[u64; 1]>,
    pub paa_args: Vec<(u16, u16)>,
    pub c3_args: Vec<u16>,
    pub unique_paa_totals: Vec<u16>,
    pub unique_c3_lags: Vec<u16>,
}

#[pymethods]
impl ArrowExtractor {
    #[new]
    pub fn new(feature_str: Vec<String>) -> Self {
        let mut features = Vec::new();
        let mut paa_args = Vec::new();
        let mut c3_args = Vec::new();
        let mut unique_paa_totals = std::collections::BTreeSet::new();
        let mut unique_c3_lags = std::collections::BTreeSet::new();

        for i in feature_str {
            match i.as_str() {
                "total_sum" => features.push(ArrowFeature::TotalSum),
                "mean" => features.push(ArrowFeature::Mean),
                "variance" => features.push(ArrowFeature::Variance),
                "std" | "std_dev" => features.push(ArrowFeature::Std),
                "min" | "min_value" => features.push(ArrowFeature::Min),
                "max" | "max_value" => features.push(ArrowFeature::Max),
                "median" => features.push(ArrowFeature::Median),
                "skew" | "skewness" => features.push(ArrowFeature::Skew),
                "kurtosis" => features.push(ArrowFeature::Kurtosis),
                "mad" => features.push(ArrowFeature::Mad),
                "iqr" => features.push(ArrowFeature::Iqr),
                "entropy" => features.push(ArrowFeature::Entropy),
                "energy" => features.push(ArrowFeature::Energy),
                "rms" => features.push(ArrowFeature::Rms),
                "root_mean_square" => features.push(ArrowFeature::RootMeanSquare),
                "zero_crossing_rate" => features.push(ArrowFeature::ZeroCrossingRate),
                "peak_count" => features.push(ArrowFeature::PeakCount),
                "autocorr_lag1" => features.push(ArrowFeature::AutocorrLag1),
                "mean_abs_change" => features.push(ArrowFeature::MeanAbsChange),
                "mean_change" => features.push(ArrowFeature::MeanChange),
                "cid_ce" => features.push(ArrowFeature::CidCe),
                "slope" => features.push(ArrowFeature::Slope),
                "intercept" => features.push(ArrowFeature::Intercept),
                "abs_sum_change" => features.push(ArrowFeature::AbsSumChange),
                "count_above_mean" => features.push(ArrowFeature::CountAboveMean),
                "count_below_mean" => features.push(ArrowFeature::CountBelowMean),
                "longest_strike_above_mean" => features.push(ArrowFeature::LongestStrikeAboveMean),
                "longest_strike_below_mean" => features.push(ArrowFeature::LongestStrikeBelowMean),
                "variation_coefficient" => features.push(ArrowFeature::VariationCoefficient),
                "auc" => features.push(ArrowFeature::Auc),
                "slope_sign_change" => features.push(ArrowFeature::SlopeSignChange),
                "turning_points" => features.push(ArrowFeature::TurningPoints),
                "zero_crossing_mean" => features.push(ArrowFeature::ZeroCrossingMean),
                "zero_crossing_std" => features.push(ArrowFeature::ZeroCrossingStd),
                e => {
                    if let Some(arg) = e.strip_prefix("paa-") {
                        let params: Vec<&str> = arg.split('-').collect();
                        if params.len() == 2
                            && let (Ok(n), Ok(m)) =
                                (params[0].parse::<u16>(), params[1].parse::<u16>())
                        {
                            features.push(ArrowFeature::Paa(n, m));
                            paa_args.push((n, m));
                            unique_paa_totals.insert(n);
                            continue;
                        }
                    } else if let Some(arg) = e.strip_prefix("c3-") {
                        let param = arg.trim();
                        if let Ok(n) = param.parse::<u16>() {
                            features.push(ArrowFeature::C3(n));
                            c3_args.push(n);
                            unique_c3_lags.insert(n);
                            continue;
                        }
                    }
                    panic!("Unknown feature: {}", e);
                }
            }
        }
        let compute = build_calculation_order(&features);

        Self {
            features,
            compute,
            paa_args,
            c3_args,
            unique_paa_totals: unique_paa_totals.into_iter().collect(),
            unique_c3_lags: unique_c3_lags.into_iter().collect(),
        }
    }

    pub fn process_2d_floats(
        &self,
        batch: PyArrowType<RecordBatch>,
    ) -> PyResult<PyArrowType<RecordBatch>> {
        let record_batch = batch.0;
        let n_cols = record_batch.num_columns();
        let n_rows = record_batch.num_rows();
        let compute = self.compute;
        let features = &self.features;
        let unique_paa_totals = &self.unique_paa_totals;
        let unique_c3_lags = &self.unique_c3_lags;

        let do_m2 = compute[2] || compute[3] || compute[7] || compute[8] || compute[29] || compute[35];
        let do_m3 = compute[7];
        let do_m4 = compute[8];
        let do_mean = compute[1]
            || do_m2
            || compute[17]
            || compute[18]
            || compute[19]
            || compute[25]
            || compute[26]
            || compute[27]
            || compute[28]
            || compute[34]
            || compute[35];
        let do_total_sum = compute[0] || do_mean;
        let do_min = compute[4] || compute[10] || compute[11];
        let do_max = compute[5] || compute[10] || compute[11];
        let do_energy = compute[12] || compute[13] || compute[14];
        let do_mac = compute[18] || compute[24];
        let do_mc = compute[19];
        let do_zcr = compute[15] || compute[34] || compute[35];
        let do_autocorr = compute[17];
        let do_cidce = compute[20];
        let do_slope = compute[21] || compute[22];
        let do_paa = compute[23];
        let do_auc = compute[31];
        let do_peaks = compute[16] || compute[32] || compute[33];
        let do_c3 = compute[30];

        let needs_sorted = compute[6] || compute[10] || compute[11];
        let do_second_pass = do_m2
            || compute[9]
            || compute[25]
            || compute[26]
            || compute[27]
            || compute[28]
            || compute[34]
            || compute[35]
            || needs_sorted;

        const LANES: usize = 4;

        let paa_boundaries: Vec<Vec<usize>> = unique_paa_totals
            .iter()
            .map(|&total| {
                (0..=total)
                    .map(|i| (i as f32 * n_rows as f32 / total as f32).round() as usize)
                    .collect()
            })
            .collect();

        let column_results: Vec<Vec<f32>> = record_batch
            .columns()
            .par_iter()
            .map(|col| {
                let float_array = col
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .expect("Failed to downcast column to Float32Array");

                let values = float_array.values();
                let n = values.len() as f32;
                if n == 0.0 {
                    return vec![0.0; features.len()];
                }

                let mut total_sum: f32 = 0.0;
                let mut min_value: f32 = f32::INFINITY;
                let mut max_value: f32 = f32::NEG_INFINITY;
                let mut energy: f32 = 0.0;
                let mut mac_sum_vec = f32x4::splat(0.0);
                let mut mc_sum_vec = f32x4::splat(0.0);
                let mut zcr_count: u32 = 0;
                let mut sum_prod: f32 = 0.0;
                let mut sum_sq_diff: f32 = 0.0;
                let mut sum_ix: f32 = 0.0;
                let mut auc_sum: f32 = 0.0;
                let mut peaks: u32 = 0;
                let mut zc_indices = Vec::new();

                let mut paa_sums: Vec<Vec<f32>> = unique_paa_totals
                    .iter()
                    .map(|&total| vec![0.0; total as usize])
                    .collect();
                let mut current_paa_segs = vec![0usize; unique_paa_totals.len()];

                let mut c3_sums = vec![0.0f64; unique_c3_lags.len()];

                let mut prev_last: f32 = values[0];

                for (chunk_idx, i) in values.chunks_exact(LANES).enumerate() {
                    let chunk = f32x4::from_slice(i);
                    let global_idx = chunk_idx * LANES;
                    let offset = global_idx as f32;

                    if do_total_sum {
                        total_sum += chunk.reduce_sum();
                    }
                    if do_min {
                        min_value = min_value.min(chunk.reduce_min());
                    }
                    if do_max {
                        max_value = max_value.max(chunk.reduce_max());
                    }
                    if do_energy {
                        energy += (chunk * chunk).reduce_sum();
                    }

                    if do_mac || do_mc || do_zcr || do_autocorr || do_cidce || do_auc {
                        let shifted = f32x4::from_array([prev_last, i[0], i[1], i[2]]);
                        let diff = chunk - shifted;
                        if do_mac {
                            mac_sum_vec += diff.abs();
                        }
                        if do_mc {
                            mc_sum_vec += diff;
                        }
                        if do_cidce {
                            sum_sq_diff += (diff * diff).reduce_sum();
                        }
                        if do_autocorr {
                            sum_prod += (chunk * shifted).reduce_sum();
                        }
                        if do_auc {
                            auc_sum += (chunk + shifted).reduce_sum() * 0.5;
                        }
                        if do_zcr {
                            let signs = chunk.simd_lt(f32x4::splat(0.0));
                            let prev_signs = shifted.simd_lt(f32x4::splat(0.0));
                            let mask = (signs ^ prev_signs).to_bitmask();
                            zcr_count += mask.count_ones();
                            if compute[34] || compute[35] {
                                for bit in 0..4 {
                                    if (mask >> bit) & 1 == 1 {
                                        zc_indices.push(offset + bit as f32);
                                    }
                                }
                            }
                        }
                        prev_last = i[LANES - 1];
                    }

                    if do_peaks {
                        // SIMD Peak Detection
                        let left = f32x4::from_array([
                            if global_idx > 0 { values[global_idx - 1] } else { values[0] },
                            values[global_idx],
                            values[global_idx + 1],
                            values[global_idx + 2],
                        ]);
                        let right = f32x4::from_array([
                            values[global_idx + 1],
                            values[global_idx + 2],
                            values[global_idx + 3],
                            if global_idx + 4 < values.len() { values[global_idx + 4] } else { values[values.len() - 1] },
                        ]);
                        let mask = chunk.simd_gt(left) & chunk.simd_gt(right);
                        peaks += mask.to_bitmask().count_ones();
                    }

                    if do_slope {
                        let indices =
                            f32x4::from_array([offset, offset + 1.0, offset + 2.0, offset + 3.0]);
                        sum_ix += (indices * chunk).reduce_sum();
                    }

                    if do_paa {
                        for (t_idx, _total) in unique_paa_totals.iter().enumerate() {
                            let b = &paa_boundaries[t_idx];
                            let seg_idx = &mut current_paa_segs[t_idx];
                            if global_idx + LANES <= b[*seg_idx + 1] {
                                paa_sums[t_idx][*seg_idx] += chunk.reduce_sum();
                            } else {
                                for j in 0..LANES {
                                    let idx = global_idx + j;
                                    while *seg_idx < paa_sums[t_idx].len() - 1 && idx >= b[*seg_idx + 1] {
                                        *seg_idx += 1;
                                    }
                                    paa_sums[t_idx][*seg_idx] += i[j];
                                }
                            }
                        }
                    }

                    if do_c3 {
                        for (l_idx, &lag) in unique_c3_lags.iter().enumerate() {
                            let l = lag as usize;
                            if global_idx >= 2 * l {
                                let chunk_il =
                                    f32x4::from_slice(&values[global_idx - l..global_idx - l + 4]);
                                let chunk_i2l = f32x4::from_slice(
                                    &values[global_idx - 2 * l..global_idx - 2 * l + 4],
                                );
                                c3_sums[l_idx] += (chunk * chunk_il * chunk_i2l).reduce_sum() as f64;
                            } else {
                                for j in 0..LANES {
                                    let idx = global_idx + j;
                                    if idx >= 2 * l {
                                        c3_sums[l_idx] += (values[idx]
                                            * values[idx - l]
                                            * values[idx - 2 * l])
                                            as f64;
                                    }
                                }
                            }
                        }
                    }
                }

                let rem_start = (values.len() / LANES) * LANES;
                for i in rem_start..values.len() {
                    let val = values[i];
                    if do_total_sum {
                        total_sum += val;
                    }
                    if do_min {
                        min_value = min_value.min(val);
                    }
                    if do_max {
                        max_value = max_value.max(val);
                    }
                    if do_energy {
                        energy += val * val;
                    }
                    if i > 0 {
                        let diff = val - values[i - 1];
                        if do_mac {
                            mac_sum_vec += f32x4::from_array([diff.abs(), 0.0, 0.0, 0.0]);
                        }
                        if do_mc {
                            mc_sum_vec += f32x4::from_array([diff, 0.0, 0.0, 0.0]);
                        }
                        if do_cidce {
                            sum_sq_diff += diff * diff;
                        }
                        if do_autocorr {
                            sum_prod += val * values[i - 1];
                        }
                        if do_auc {
                            auc_sum += (val + values[i - 1]) * 0.5;
                        }
                        if do_zcr && (val < 0.0) != (values[i - 1] < 0.0) {
                            zcr_count += 1;
                            if compute[34] || compute[35] {
                                zc_indices.push(i as f32);
                            }
                        }
                    }
                    if do_peaks && i > 0 && i < values.len() - 1 && val > values[i - 1] && val > values[i + 1] {
                        peaks += 1;
                    }
                    if do_slope {
                        sum_ix += (i as f32) * val;
                    }
                    if do_paa {
                        for (t_idx, _total) in unique_paa_totals.iter().enumerate() {
                            let b = &paa_boundaries[t_idx];
                            let seg_idx = &mut current_paa_segs[t_idx];
                            while *seg_idx < paa_sums[t_idx].len() - 1 && i >= b[*seg_idx + 1] {
                                *seg_idx += 1;
                            }
                            paa_sums[t_idx][*seg_idx] += val;
                        }
                    }
                    if do_c3 {
                        for (l_idx, &lag) in unique_c3_lags.iter().enumerate() {
                            let l = lag as usize;
                            if i >= 2 * l {
                                c3_sums[l_idx] += (val * values[i - l] * values[i - 2 * l]) as f64;
                            }
                        }
                    }
                }

                let mean = total_sum / n;
                let mac_sum = mac_sum_vec.reduce_sum();
                let mc_sum = mc_sum_vec.reduce_sum();

                let mut m2: f32 = 0.0;
                let mut m3: f32 = 0.0;
                let mut m4: f32 = 0.0;
                let mut mad_sum: f32 = 0.0;
                let mut count_a: u32 = 0;
                let mut count_b: u32 = 0;
                let mut max_strike_a: u32 = 0;
                let mut current_strike_a: u32 = 0;
                let mut max_strike_b: u32 = 0;
                let mut current_strike_b: u32 = 0;
                let mut median: f32 = 0.0;
                let mut iqr: f32 = 0.0;
                let mut entropy: f32 = 0.0;
                let mut zc_mean: f32 = 0.0;
                let mut zc_std: f32 = 0.0;

                if do_second_pass {
                    if needs_sorted {
                        let mut copy = values.to_vec();
                        let n_size = copy.len();
                        if compute[6] {
                            let mid = n_size / 2;
                            copy.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                            if n_size % 2 == 0 {
                                let max_low = copy[..mid].iter().max_by(|a, b| a.partial_cmp(b).unwrap()).copied().unwrap_or(copy[mid]);
                                median = (max_low + copy[mid]) * 0.5;
                            } else {
                                median = copy[mid];
                            }
                        }
                        if compute[10] {
                            let q1_idx = (n_size as f32 * 0.25) as usize;
                            let q3_idx = (n_size as f32 * 0.75) as usize;
                            copy.select_nth_unstable_by(q1_idx, |a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                            let q1 = copy[q1_idx];
                            copy.select_nth_unstable_by(q3_idx, |a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                            let q3 = copy[q3_idx];
                            iqr = q3 - q1;
                        }
                        if compute[11] {
                            let min = min_value;
                            let max = max_value;
                            let range = max - min;
                            if range > 1e-9 {
                                let bins = 10;
                                let mut counts = vec![0usize; bins];
                                for &v in values {
                                    let b = (((v - min) / range) * (bins as f32 - 1.0)) as usize;
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

                    // Combined second pass for moments and strikes
                    let mean_vec = f32x4::splat(mean);
                    let mut m2_vec = f32x4::splat(0.0);
                    let mut m3_vec = f32x4::splat(0.0);
                    let mut m4_vec = f32x4::splat(0.0);
                    let mut mad_sum_vec = f32x4::splat(0.0);

                    for chunk in values.chunks_exact(LANES) {
                        let c = f32x4::from_slice(chunk);
                        let diff = c - mean_vec;
                        if do_m2 {
                            let d2 = diff * diff;
                            m2_vec += d2;
                            if do_m3 { m3_vec += d2 * diff; }
                            if do_m4 { m4_vec += d2 * d2; }
                        }
                        if compute[9] {
                            mad_sum_vec += diff.abs();
                        }
                    }
                    m2 = m2_vec.reduce_sum();
                    m3 = m3_vec.reduce_sum();
                    m4 = m4_vec.reduce_sum();
                    mad_sum = mad_sum_vec.reduce_sum();

                    for &val in &values[rem_start..] {
                        let diff = val - mean;
                        if do_m2 {
                            let d2 = diff * diff;
                            m2 += d2;
                            if do_m3 { m3 += d2 * diff; }
                            if do_m4 { m4 += d2 * d2; }
                        }
                        if compute[9] {
                            mad_sum += diff.abs();
                        }
                    }

                    // Strike count and count above/below mean (still scalar as it's highly conditional)
                    if compute[25] || compute[26] || compute[27] || compute[28] {
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

                    if compute[34] || compute[35] {
                        if !zc_indices.is_empty() {
                            zc_mean = zc_indices.iter().sum::<f32>() / zc_indices.len() as f32;
                            if compute[35] {
                                let zc_m2 = zc_indices.iter().map(|&idx| (idx - zc_mean).powi(2)).sum::<f32>();
                                zc_std = (zc_m2 / zc_indices.len() as f32).sqrt();
                            }
                        }
                    }
                }

                let variance = if n > 1.0 { m2 / (n - 1.0) } else { 0.0 };
                let std_dev = variance.sqrt();

                let mut res = Vec::with_capacity(features.len());
                for feat in features {
                    let val = match feat {
                        ArrowFeature::TotalSum => total_sum,
                        ArrowFeature::Mean => mean,
                        ArrowFeature::Variance => variance,
                        ArrowFeature::Std => std_dev,
                        ArrowFeature::Min => min_value,
                        ArrowFeature::Max => max_value,
                        ArrowFeature::Median => median,
                        ArrowFeature::Skew => if variance > 1e-9 { (m3 / n) / variance.powf(1.5) } else { 0.0 },
                        ArrowFeature::Kurtosis => if variance > 1e-9 { (m4 / n) / (variance * variance) - 3.0 } else { 0.0 },
                        ArrowFeature::Mad => mad_sum / n,
                        ArrowFeature::Iqr => iqr,
                        ArrowFeature::Entropy => entropy,
                        ArrowFeature::Energy => energy,
                        ArrowFeature::Rms | ArrowFeature::RootMeanSquare => (energy / n).sqrt(),
                        ArrowFeature::ZeroCrossingRate => zcr_count as f32 / n,
                        ArrowFeature::PeakCount => peaks as f32,
                        ArrowFeature::AutocorrLag1 => if variance > 1e-9 { (sum_prod / (n - 1.0) - mean * mean) / variance } else { 0.0 },
                        ArrowFeature::MeanAbsChange => mac_sum / n,
                        ArrowFeature::MeanChange => mc_sum / n,
                        ArrowFeature::CidCe => sum_sq_diff.sqrt(),
                        ArrowFeature::Slope => {
                            let mean_i = (n - 1.0) * 0.5;
                            let s_xx = (n * (n * n - 1.0)) / 12.0;
                            let s_xy = sum_ix - n * mean_i * mean;
                            if s_xx.abs() > 1e-9 { s_xy / s_xx } else { 0.0 }
                        }
                        ArrowFeature::Intercept => {
                            let mean_i = (n - 1.0) * 0.5;
                            let s_xx = (n * (n * n - 1.0)) / 12.0;
                            let s_xy = sum_ix - n * mean_i * mean;
                            let slope = if s_xx.abs() > 1e-9 { s_xy / s_xx } else { 0.0 };
                            mean - slope * mean_i
                        }
                        ArrowFeature::AbsSumChange => mac_sum,
                        ArrowFeature::CountAboveMean => count_a as f32,
                        ArrowFeature::CountBelowMean => count_b as f32,
                        ArrowFeature::LongestStrikeAboveMean => max_strike_a as f32,
                        ArrowFeature::LongestStrikeBelowMean => max_strike_b as f32,
                        ArrowFeature::VariationCoefficient => if mean.abs() > 1e-9 { std_dev / mean } else { 0.0 },
                        ArrowFeature::Auc => auc_sum,
                        ArrowFeature::ZeroCrossingMean => zc_mean,
                        ArrowFeature::ZeroCrossingStd => zc_std,
                        ArrowFeature::C3(lag) => {
                            let l_idx = unique_c3_lags.iter().position(|&l| l == *lag).unwrap();
                            let l = *lag as usize;
                            if values.len() > 2 * l { (c3_sums[l_idx] / (values.len() - 2 * l) as f64) as f32 } else { 0.0 }
                        }
                        ArrowFeature::Paa(total, index) => {
                            let t_idx = unique_paa_totals.iter().position(|&t| t == *total).unwrap();
                            let b = &paa_boundaries[t_idx];
                            let start = b[*index as usize];
                            let end = b[*index as usize + 1];
                            if start < end { paa_sums[t_idx][*index as usize] / (end - start) as f32 } else { 0.0 }
                        }
                        _ => 0.0,
                    };
                    res.push(val);
                }
                res
            })
            .collect();

        let mut fields = Vec::with_capacity(features.len());
        let results: Vec<ArrayRef> = (0..features.len())
            .into_par_iter()
            .map(|feat_idx| {
                let mut col_data = Vec::with_capacity(n_cols);
                for col_idx in 0..n_cols {
                    col_data.push(column_results[col_idx][feat_idx]);
                }
                Arc::new(Float32Array::from(col_data)) as ArrayRef
            })
            .collect();

        for feat in features {
            let name = match feat {
                ArrowFeature::TotalSum => "total_sum".to_string(),
                ArrowFeature::Mean => "mean".to_string(),
                ArrowFeature::Variance => "variance".to_string(),
                ArrowFeature::Std => "std_dev".to_string(),
                ArrowFeature::Min => "min_value".to_string(),
                ArrowFeature::Max => "max_value".to_string(),
                ArrowFeature::Median => "median".to_string(),
                ArrowFeature::Skew => "skewness".to_string(),
                ArrowFeature::Kurtosis => "kurtosis".to_string(),
                ArrowFeature::Mad => "mad".to_string(),
                ArrowFeature::Iqr => "iqr".to_string(),
                ArrowFeature::Entropy => "entropy".to_string(),
                ArrowFeature::Energy => "energy".to_string(),
                ArrowFeature::Rms => "rms".to_string(),
                ArrowFeature::RootMeanSquare => "root_mean_square".to_string(),
                ArrowFeature::ZeroCrossingRate => "zero_crossing_rate".to_string(),
                ArrowFeature::PeakCount => "peak_count".to_string(),
                ArrowFeature::AutocorrLag1 => "autocorr_lag1".to_string(),
                ArrowFeature::MeanAbsChange => "mean_abs_change".to_string(),
                ArrowFeature::MeanChange => "mean_change".to_string(),
                ArrowFeature::CidCe => "cid_ce".to_string(),
                ArrowFeature::Slope => "slope".to_string(),
                ArrowFeature::Intercept => "intercept".to_string(),
                ArrowFeature::AbsSumChange => "abs_sum_change".to_string(),
                ArrowFeature::CountAboveMean => "count_above_mean".to_string(),
                ArrowFeature::CountBelowMean => "count_below_mean".to_string(),
                ArrowFeature::LongestStrikeAboveMean => "longest_strike_above_mean".to_string(),
                ArrowFeature::LongestStrikeBelowMean => "longest_strike_below_mean".to_string(),
                ArrowFeature::VariationCoefficient => "variation_coefficient".to_string(),
                ArrowFeature::Auc => "auc".to_string(),
                ArrowFeature::ZeroCrossingMean => "zero_crossing_mean".to_string(),
                ArrowFeature::ZeroCrossingStd => "zero_crossing_std".to_string(),
                ArrowFeature::C3(lag) => format!("c3-{}", lag),
                ArrowFeature::Paa(total, index) => format!("paa-{}-{}", total, index),
                _ => "unknown".to_string(),
            };
            fields.push(Field::new(name, DataType::Float32, false));
        }

        let return_batch =
            RecordBatch::try_new(Arc::new(Schema::new(fields)), results).map_err(|e| {
                PyTypeError::new_err(format!("Failed to create return RecordBatch: {}", e))
            })?;

        Ok(PyArrowType(return_batch))
    }
}

fn build_calculation_order(features: &Vec<ArrowFeature>) -> BitArray<[u64; 1]> {
    let mut bits = BitArray::<[u64; 1]>::ZERO;
    for feat in features {
        match feat {
            ArrowFeature::TotalSum => { bits.set(0, true); }
            ArrowFeature::Mean => { bits.set(1, true); }
            ArrowFeature::Variance => { bits.set(2, true); bits.set(1, true); }
            ArrowFeature::Std => { bits.set(3, true); bits.set(2, true); bits.set(1, true); }
            ArrowFeature::Min => { bits.set(4, true); }
            ArrowFeature::Max => { bits.set(5, true); }
            ArrowFeature::Median => { bits.set(6, true); }
            ArrowFeature::Skew => { bits.set(7, true); bits.set(3, true); bits.set(2, true); bits.set(1, true); }
            ArrowFeature::Kurtosis => { bits.set(8, true); bits.set(3, true); bits.set(2, true); bits.set(1, true); }
            ArrowFeature::Mad => { bits.set(9, true); bits.set(1, true); }
            ArrowFeature::Iqr => { bits.set(10, true); bits.set(4, true); bits.set(5, true); }
            ArrowFeature::Entropy => { bits.set(11, true); bits.set(4, true); bits.set(5, true); bits.set(10, true); }
            ArrowFeature::Energy => { bits.set(12, true); }
            ArrowFeature::Rms => { bits.set(13, true); bits.set(12, true); }
            ArrowFeature::RootMeanSquare => { bits.set(14, true); bits.set(12, true); }
            ArrowFeature::ZeroCrossingRate => { bits.set(15, true); }
            ArrowFeature::PeakCount => { bits.set(16, true); }
            ArrowFeature::AutocorrLag1 => { bits.set(17, true); bits.set(1, true); bits.set(2, true); }
            ArrowFeature::MeanAbsChange => { bits.set(18, true); bits.set(1, true); }
            ArrowFeature::MeanChange => { bits.set(19, true); bits.set(1, true); }
            ArrowFeature::CidCe => { bits.set(20, true); bits.set(1, true); }
            ArrowFeature::Slope => { bits.set(21, true); bits.set(1, true); }
            ArrowFeature::Intercept => { bits.set(22, true); bits.set(21, true); bits.set(1, true); }
            ArrowFeature::Paa(_, _) => { bits.set(23, true); }
            ArrowFeature::AbsSumChange => { bits.set(24, true); }
            ArrowFeature::CountAboveMean => { bits.set(25, true); bits.set(1, true); }
            ArrowFeature::CountBelowMean => { bits.set(26, true); bits.set(1, true); }
            ArrowFeature::LongestStrikeAboveMean => { bits.set(27, true); bits.set(1, true); }
            ArrowFeature::LongestStrikeBelowMean => { bits.set(28, true); bits.set(1, true); }
            ArrowFeature::VariationCoefficient => { bits.set(29, true); bits.set(3, true); bits.set(2, true); bits.set(1, true); }
            ArrowFeature::C3(_) => { bits.set(30, true); }
            ArrowFeature::Auc => { bits.set(31, true); }
            ArrowFeature::SlopeSignChange => { bits.set(32, true); bits.set(16, true); }
            ArrowFeature::TurningPoints => { bits.set(33, true); bits.set(16, true); }
            ArrowFeature::ZeroCrossingMean => { bits.set(34, true); bits.set(15, true); bits.set(1, true); }
            ArrowFeature::ZeroCrossingStd => { bits.set(35, true); bits.set(15, true); bits.set(1, true); }
        }
    }
    bits
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Float32Array;
    use arrow::array::RecordBatch;
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    fn create_test_batch(data: Vec<f32>) -> PyArrowType<RecordBatch> {
        let schema = Schema::new(vec![Field::new("c1", DataType::Float32, false)]);
        let array = Float32Array::from(data);
        let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(array)]).unwrap();
        PyArrowType(batch)
    }

    #[test]
    fn test_all_features() {
        let data = vec![-2.0, -1.0, 0.5, 2.0, 1.0, -0.5, -1.0, 1.0, 2.0, 3.0];
        let features = vec![
            "total_sum".to_string(),
            "mean".to_string(),
            "variance".to_string(),
            "std_dev".to_string(),
            "min_value".to_string(),
            "max_value".to_string(),
            "median".to_string(),
            "skewness".to_string(),
            "kurtosis".to_string(),
            "mad".to_string(),
            "iqr".to_string(),
            "entropy".to_string(),
            "energy".to_string(),
            "rms".to_string(),
            "zero_crossing_rate".to_string(),
            "peak_count".to_string(),
            "autocorr_lag1".to_string(),
            "mean_abs_change".to_string(),
            "mean_change".to_string(),
            "cid_ce".to_string(),
            "slope".to_string(),
            "intercept".to_string(),
            "abs_sum_change".to_string(),
            "count_above_mean".to_string(),
            "count_below_mean".to_string(),
            "longest_strike_above_mean".to_string(),
            "longest_strike_below_mean".to_string(),
            "variation_coefficient".to_string(),
            "auc".to_string(),
            "zero_crossing_mean".to_string(),
            "zero_crossing_std".to_string(),
            "c3-1".to_string(),
            "paa-2-0".to_string(),
            "paa-2-1".to_string(),
        ];
        let extractor = ArrowExtractor::new(features);
        let batch = create_test_batch(data.clone());
        let result = extractor.process_2d_floats(batch).unwrap().0;

        assert_eq!(result.num_columns(), 34);

        for i in 0..result.num_columns() {
            let col = result.column(i).as_any().downcast_ref::<Float32Array>().unwrap();
            let val = col.value(0);
            println!("{}: {}", result.schema().field(i).name(), val);
            assert!(!val.is_nan(), "Feature {} is NaN", result.schema().field(i).name());
        }
    }

    #[test]
    fn test_simd_vs_scalar_comprehensive() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let features = vec![
            "mean".to_string(),
            "variance".to_string(),
            "total_sum".to_string(),
            "min_value".to_string(),
            "max_value".to_string(),
            "std_dev".to_string(),
            "energy".to_string(),
            "rms".to_string(),
            "mad".to_string(),
        ];
        let extractor = ArrowExtractor::new(features);
        let batch = create_test_batch(data.clone());
        let result = extractor.process_2d_floats(batch).unwrap().0;

        let n = data.len() as f32;
        let scalar_sum: f32 = data.iter().sum();
        let scalar_mean = scalar_sum / n;
        let scalar_var = data.iter().map(|&x| (x - scalar_mean).powi(2)).sum::<f32>() / (n - 1.0);
        let scalar_energy = data.iter().map(|&x| x * x).sum::<f32>();
        let scalar_mad = data.iter().map(|&x| (x - scalar_mean).abs()).sum::<f32>() / n;

        let col_mean = result.column(0).as_any().downcast_ref::<Float32Array>().unwrap().value(0);
        let col_var = result.column(1).as_any().downcast_ref::<Float32Array>().unwrap().value(0);
        let col_sum = result.column(2).as_any().downcast_ref::<Float32Array>().unwrap().value(0);
        let col_energy = result.column(6).as_any().downcast_ref::<Float32Array>().unwrap().value(0);
        let col_mad = result.column(8).as_any().downcast_ref::<Float32Array>().unwrap().value(0);

        assert!((col_mean - scalar_mean).abs() < 1e-5);
        assert!((col_var - scalar_var).abs() < 1e-5);
        assert!((col_sum - scalar_sum).abs() < 1e-5);
        assert!((col_energy - scalar_energy).abs() < 1e-5);
        assert!((col_mad - scalar_mad).abs() < 1e-5);
    }
}
