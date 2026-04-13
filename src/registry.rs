use crate::features::{
    energy::{energy, rms},
    signal::{
        abs_sum_change, auc, autocorr_lag1, c3, cid_ce, mean_abs_change, mean_change, paa,
        peak_count, slope_sign_change, turning_points, zero_crossing_derivations,
        zero_crossing_rate,
    },
    statistics::{
        count_above_mean, count_below_mean, entropy, iqr, kurtosis, longest_strike_above_mean,
        longest_strike_below_mean, mad, max, mean, median, min, skew, variance,
        variation_coefficient,
    },
    trend::{intercept, slope},
};
use ndarray::Axis;
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;
use std::simd::prelude::*;

struct SendPtr(*const f64);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Copy)]
pub enum Feature {
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

pub struct Cache {
    pub mean: Option<f64>,
    pub variance: Option<f64>,
    pub std: Option<f64>,
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub median: Option<f64>,
    pub skew: Option<f64>,
    pub kurtosis: Option<f64>,
    pub mad: Option<f64>,
    pub iqr: Option<f64>,
    pub entropy: Option<f64>,
    pub energy: Option<f64>,
    pub rms: Option<f64>,
    pub autocorr_lag1: Option<f64>,
    pub mean_abs_change: Option<f64>,
    pub mean_change: Option<f64>,
    pub cid_ce: Option<f64>,
    pub slope: Option<f64>,
    pub intercept: Option<f64>,
    pub zero_crossing_rate: Option<f64>,
    pub peak_count: Option<f64>,
    pub abs_sum_change: Option<f64>,
    pub count_above_mean: Option<f64>,
    pub count_below_mean: Option<f64>,
    pub longest_strike_above_mean: Option<f64>,
    pub longest_strike_below_mean: Option<f64>,
    pub variation_coefficient: Option<f64>,
    pub auc: Option<f64>,
    pub slope_sign_change: Option<f64>,
    pub turning_points: Option<f64>,
    pub zero_crossing_mean: Option<f64>,
    pub zero_crossing_std: Option<f64>,
    pub c3: HashMap<u16, f64>,
    pub paa: HashMap<(u16, u16), f64>,
}

impl Cache {
    pub fn new() -> Self {
        Self {
            mean: None,
            variance: None,
            std: None,
            min: None,
            max: None,
            median: None,
            skew: None,
            kurtosis: None,
            mad: None,
            iqr: None,
            entropy: None,
            energy: None,
            rms: None,
            autocorr_lag1: None,
            mean_abs_change: None,
            mean_change: None,
            cid_ce: None,
            slope: None,
            intercept: None,
            zero_crossing_rate: None,
            peak_count: None,
            abs_sum_change: None,
            count_above_mean: None,
            count_below_mean: None,
            longest_strike_above_mean: None,
            longest_strike_below_mean: None,
            variation_coefficient: None,
            auc: None,
            slope_sign_change: None,
            turning_points: None,
            zero_crossing_mean: None,
            zero_crossing_std: None,
            c3: HashMap::new(),
            paa: HashMap::new(),
        }
    }
    fn fill_vec(&self, features: &[Feature], out: &mut [f64]) {
        for (i, feat) in features.iter().enumerate() {
            out[i] = match feat {
                Feature::Mean => self.mean.unwrap_or(f64::NAN),
                Feature::Variance => self.variance.unwrap_or(f64::NAN),
                Feature::Std => self.std.unwrap_or(f64::NAN),
                Feature::Min => self.min.unwrap_or(f64::NAN),
                Feature::Max => self.max.unwrap_or(f64::NAN),
                Feature::Median => self.median.unwrap_or(f64::NAN),
                Feature::Skew => self.skew.unwrap_or(f64::NAN),
                Feature::Kurtosis => self.kurtosis.unwrap_or(f64::NAN),
                Feature::Mad => self.mad.unwrap_or(f64::NAN),
                Feature::Iqr => self.iqr.unwrap_or(f64::NAN),
                Feature::Entropy => self.entropy.unwrap_or(f64::NAN),
                Feature::Energy => self.energy.unwrap_or(f64::NAN),
                Feature::Rms => self.rms.unwrap_or(f64::NAN),
                Feature::RootMeanSquare => self.rms.unwrap_or(f64::NAN),
                Feature::ZeroCrossingRate => self.zero_crossing_rate.unwrap_or(f64::NAN),
                Feature::PeakCount => self.peak_count.unwrap_or(f64::NAN),
                Feature::AutocorrLag1 => self.autocorr_lag1.unwrap_or(f64::NAN),
                Feature::MeanAbsChange => self.mean_abs_change.unwrap_or(f64::NAN),
                Feature::MeanChange => self.mean_change.unwrap_or(f64::NAN),
                Feature::CidCe => self.cid_ce.unwrap_or(f64::NAN),
                Feature::Slope => self.slope.unwrap_or(f64::NAN),
                Feature::Intercept => self.intercept.unwrap_or(f64::NAN),
                Feature::AbsSumChange => self.abs_sum_change.unwrap_or(f64::NAN),
                Feature::CountAboveMean => self.count_above_mean.unwrap_or(f64::NAN),
                Feature::CountBelowMean => self.count_below_mean.unwrap_or(f64::NAN),
                Feature::LongestStrikeAboveMean => {
                    self.longest_strike_above_mean.unwrap_or(f64::NAN)
                }
                Feature::LongestStrikeBelowMean => {
                    self.longest_strike_below_mean.unwrap_or(f64::NAN)
                }
                Feature::VariationCoefficient => self.variation_coefficient.unwrap_or(f64::NAN),
                Feature::Auc => self.auc.unwrap_or(f64::NAN),
                Feature::SlopeSignChange => self.slope_sign_change.unwrap_or(f64::NAN),
                Feature::TurningPoints => self.turning_points.unwrap_or(f64::NAN),
                Feature::ZeroCrossingMean => self.zero_crossing_mean.unwrap_or(f64::NAN),
                Feature::ZeroCrossingStd => self.zero_crossing_std.unwrap_or(f64::NAN),
                Feature::C3(lag) => *self.c3.get(lag).unwrap_or(&f64::NAN),
                Feature::Paa(sec, bin) => *self.paa.get(&(*sec, *bin)).unwrap_or(&f64::NAN),
            };
        }
    }
}

#[pyclass]
pub struct FeatureExtractor {
    pub features: Vec<Feature>,
    pub calculation_order: Vec<Feature>,
}

#[pymethods]
impl FeatureExtractor {
    #[new]
    pub fn new(features: Vec<String>) -> Self {
        let mut feature_enums = Vec::new();
        for feature in features {
            match feature.to_lowercase().as_str() {
                "mean" => feature_enums.push(Feature::Mean),
                "variance" => feature_enums.push(Feature::Variance),
                "std" => feature_enums.push(Feature::Std),
                "min" => feature_enums.push(Feature::Min),
                "max" => feature_enums.push(Feature::Max),
                "median" => feature_enums.push(Feature::Median),
                "skew" => feature_enums.push(Feature::Skew),
                "kurtosis" => feature_enums.push(Feature::Kurtosis),
                "mad" => feature_enums.push(Feature::Mad),
                "iqr" => feature_enums.push(Feature::Iqr),
                "entropy" => feature_enums.push(Feature::Entropy),
                "energy" => feature_enums.push(Feature::Energy),
                "rms" | "rootmeansquare" => feature_enums.push(Feature::RootMeanSquare),
                "zerocrossingrate" | "zero_crossing_rate" => {
                    feature_enums.push(Feature::ZeroCrossingRate)
                }
                "peakcount" | "peak_count" => feature_enums.push(Feature::PeakCount),
                "autocorrlag1" | "autocorr_lag1" => feature_enums.push(Feature::AutocorrLag1),
                "meanabschange" | "mean_abs_change" => feature_enums.push(Feature::MeanAbsChange),
                "meanchange" | "mean_change" => feature_enums.push(Feature::MeanChange),
                "cidce" | "cid_ce" => feature_enums.push(Feature::CidCe),
                "slope" => feature_enums.push(Feature::Slope),
                "intercept" => feature_enums.push(Feature::Intercept),
                "abssumchange" | "abs_sum_change" => feature_enums.push(Feature::AbsSumChange),
                "countabovemean" | "count_above_mean" => {
                    feature_enums.push(Feature::CountAboveMean)
                }
                "countbelowmean" | "count_below_mean" => {
                    feature_enums.push(Feature::CountBelowMean)
                }
                "longeststrikeabovemean" | "longest_strike_above_mean" => {
                    feature_enums.push(Feature::LongestStrikeAboveMean)
                }
                "longeststrikebelowmean" | "longest_strike_below_mean" => {
                    feature_enums.push(Feature::LongestStrikeBelowMean)
                }
                "variationcoefficient" | "variation_coefficient" => {
                    feature_enums.push(Feature::VariationCoefficient)
                }
                "auc" => feature_enums.push(Feature::Auc),
                "slopesignchange" | "slope_sign_change" | "ssc" => {
                    feature_enums.push(Feature::SlopeSignChange)
                }
                "turning_points" | "turningpoints" => feature_enums.push(Feature::TurningPoints),
                "zero_crossing_mean" | "zerocrossingmean" => {
                    feature_enums.push(Feature::ZeroCrossingMean)
                }
                "zero_crossing_std" | "zerocrossingstd" => {
                    feature_enums.push(Feature::ZeroCrossingStd)
                }
                st => {
                    if st.contains("paa") {
                        let parts: Vec<&str> = st.split('-').collect();
                        if parts.len() == 3
                            && parts[0] == "paa"
                            && let (Ok(n_segments), Ok(n_bins)) =
                                (parts[1].parse::<u16>(), parts[2].parse::<u16>())
                        {
                            feature_enums.push(Feature::Paa(n_segments, n_bins));
                            continue;
                        }
                    }
                    if st.contains("c3") {
                        let parts: Vec<&str> = st.split('-').collect();
                        if parts.len() == 2
                            && parts[0] == "c3"
                            && let Ok(lag) = parts[1].parse::<u16>()
                        {
                            feature_enums.push(Feature::C3(lag));
                            continue;
                        } else if st == "c3" {
                            feature_enums.push(Feature::C3(1));
                            continue;
                        }
                    }
                    panic!("Unknown feature: {}", st);
                }
            }
        }
        let features = feature_enums.clone();
        feature_enums.dedup();
        let calculation_order = build_calculation_order(feature_enums);

        Self {
            features,
            calculation_order,
        }
    }

    pub fn extract1d(
        &self,
        data: PyReadonlyArray1<'_, f64>,
        _py: Python<'_>,
    ) -> PyResult<Py<PyArray1<f64>>> {
        let data_slice = data.as_slice()?;
        let mut cache = Cache::new();
        self.extract_to_cache(data_slice, &mut cache);
        let mut featu = vec![0.0; self.features.len()];
        cache.fill_vec(&self.features, &mut featu);
        Ok(PyArray1::from_vec(_py, featu).to_owned().into())
    }

    pub fn extract2d(&self, input_data: Bound<'_, PyAny>) -> PyResult<Py<PyArray2<f64>>> {
        let py = input_data.py();

        if let Ok(arr) = input_data.extract::<PyReadonlyArray2<'_, f64>>() {
            let view = arr.as_array();
            let n_samples = view.shape()[0];
            let n_features = self.features.len();
            let mut results = ndarray::Array2::zeros((n_samples, n_features));

            results
                .axis_iter_mut(Axis(0))
                .into_par_iter()
                .zip(view.axis_iter(Axis(0)))
                .for_each(|(mut res_row, sig_row)| {
                    let mut cache = Cache::new();
                    let data_slice = sig_row.as_slice().unwrap_or_else(|| &[]);
                    if !data_slice.is_empty() {
                        self.extract_to_cache(data_slice, &mut cache);
                    }
                    cache.fill_vec(&self.features, res_row.as_slice_mut().unwrap());
                });

            return Ok(results.into_pyarray(py).unbind());
        }

        // Fallback for list of arrays - Optimized to avoid to_vec()
        if let Ok(list) = input_data.cast::<pyo3::types::PyList>() {
            let n_samples = list.len();
            let n_features = self.features.len();
            let mut results = ndarray::Array2::zeros((n_samples, n_features));

            let mut signals = Vec::with_capacity(n_samples);
            for i in 0..n_samples {
                let item = list.get_item(i)?;
                let arr: Bound<'_, PyArray1<f64>> = item.extract()?;
                let ptr = unsafe { arr.as_array().as_ptr() };
                signals.push((SendPtr(ptr), arr.len(), arr.clone().unbind()));
            }

            results
                .axis_iter_mut(Axis(0))
                .into_par_iter()
                .zip(signals.par_iter())
                .for_each(|(mut res_row, (ptr_wrapper, len, _))| {
                    let mut cache = Cache::new();
                    // SAFETY: The numpy array is kept alive by the Py object in the tuple.
                    let data_slice = unsafe { std::slice::from_raw_parts(ptr_wrapper.0, *len) };
                    self.extract_to_cache(data_slice, &mut cache);
                    cache.fill_vec(&self.features, res_row.as_slice_mut().unwrap());
                });

            return Ok(results.into_pyarray(py).unbind());
        }

        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "Expected a list of 1D arrays or a 2D array",
        ))
    }
}

// Internal methods moved out of #[pymethods]
impl FeatureExtractor {
    fn extract_to_cache(&self, data_slice: &[f64], cache: &mut Cache) {
        let n = data_slice.len();
        if n == 0 {
            return;
        }

        // 1. Identify which features can be computed in the fused pass
        let mut fused_needed = false;
        for feat in &self.calculation_order {
            match feat {
                Feature::Mean
                | Feature::Variance
                | Feature::Min
                | Feature::Max
                | Feature::Energy
                | Feature::Rms
                | Feature::RootMeanSquare
                | Feature::ZeroCrossingRate
                | Feature::PeakCount
                | Feature::MeanAbsChange
                | Feature::MeanChange
                | Feature::CidCe
                | Feature::Slope
                | Feature::Intercept
                | Feature::AbsSumChange
                | Feature::CountAboveMean
                | Feature::CountBelowMean
                | Feature::LongestStrikeAboveMean
                | Feature::LongestStrikeBelowMean
                | Feature::VariationCoefficient
                | Feature::Auc
                | Feature::SlopeSignChange
                | Feature::TurningPoints
                | Feature::ZeroCrossingMean
                | Feature::ZeroCrossingStd => {
                    fused_needed = true;
                }

                _ => {}
            }
        }

        if fused_needed {
            self.fused_pass_vectorized(data_slice, cache);
        }

        // 2. Compute remaining features (order statistics, etc.)
        let mut needs_sort = false;
        for feat in &self.calculation_order {
            match feat {
                Feature::Median | Feature::Mad | Feature::Iqr | Feature::Entropy => {
                    needs_sort = true;
                    break;
                }
                _ => {}
            }
        }

        if needs_sort {
            use crate::features::statistics::{iqr_from_sorted, median_from_sorted};
            let mut sorted_data = data_slice.to_vec();
            sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            if cache.median.is_none() {
                cache.median = Some(median_from_sorted(&sorted_data));
            }
            if cache.iqr.is_none() {
                cache.iqr = Some(iqr_from_sorted(&sorted_data));
            }
            if cache.mad.is_none() {
                if let Some(med) = cache.median {
                    let mut deviations: Vec<f64> =
                        data_slice.iter().map(|&x| (x - med).abs()).collect();
                    deviations
                        .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    cache.mad = Some(median_from_sorted(&deviations));
                }
            }
        }

        // 3. Any other features (Autocorr, PAA, etc.)
        for feat_name in &self.calculation_order {
            match feat_name {
                Feature::Skew => {
                    if cache.skew.is_none() {
                        skew(data_slice, cache);
                    }
                }
                Feature::Kurtosis => {
                    if cache.kurtosis.is_none() {
                        kurtosis(data_slice, cache);
                    }
                }
                Feature::Entropy => {
                    if cache.entropy.is_none() {
                        entropy(data_slice, cache);
                    }
                }
                Feature::AutocorrLag1 => {
                    if cache.autocorr_lag1.is_none() {
                        autocorr_lag1(data_slice, cache);
                    }
                }
                Feature::Paa(sec, bin) => {
                    if !cache.paa.contains_key(&(*sec, *bin)) {
                        paa(data_slice, *sec as usize, *bin as usize, cache);
                    }
                }
                Feature::C3(lag) => {
                    if !cache.c3.contains_key(lag) {
                        c3(data_slice, *lag as usize, cache);
                    }
                }
                _ => {} // already handled
            }
        }
    }

    fn fused_pass_vectorized(&self, data: &[f64], cache: &mut Cache) {
        let n = data.len();
        if n == 0 {
            return;
        }
        let n_f = n as f64;

        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        let mut min_val = f64::INFINITY;
        let mut max_val = f64::NEG_INFINITY;

        // SIMD for sum, sum_sq, min, max
        let (prefix, chunks, remainder) = data.as_simd::<8>();

        for &x in prefix {
            sum += x;
            sum_sq += x * x;
            if x < min_val {
                min_val = x;
            }
            if x > max_val {
                max_val = x;
            }
        }

        let mut sums = f64x8::splat(0.0);
        let mut sum_sqs = f64x8::splat(0.0);
        let mut mins = f64x8::splat(f64::INFINITY);
        let mut maxs = f64x8::splat(f64::NEG_INFINITY);

        for chunk in chunks {
            let c = *chunk;
            sums += c;
            sum_sqs += c * c;
            mins = mins.simd_min(c);
            maxs = maxs.simd_max(c);
        }

        sum += sums.reduce_sum();
        sum_sq += sum_sqs.reduce_sum();
        min_val = min_val.min(mins.reduce_min());
        max_val = max_val.max(maxs.reduce_max());

        for &x in remainder {
            sum += x;
            sum_sq += x * x;
            if x < min_val {
                min_val = x;
            }
            if x > max_val {
                max_val = x;
            }
        }

        // Post-loop derivations
        let m = sum / n_f;
        cache.mean = Some(m);
        cache.min = Some(min_val);
        cache.max = Some(max_val);
        cache.energy = Some(sum_sq);
        cache.rms = Some((sum_sq / n_f).sqrt());

        let var = (sum_sq / n_f) - (m * m);
        let variance_val = if var < 0.0 { 0.0 } else { var };
        cache.variance = Some(variance_val);
        let s = variance_val.sqrt();
        cache.std = Some(s);

        // Optional second SIMD pass for Skew and Kurtosis
        if self.calculation_order.contains(&Feature::Skew)
            || self.calculation_order.contains(&Feature::Kurtosis)
        {
            let m_v = f64x8::splat(m);
            let mut sums_3 = f64x8::splat(0.0);
            let mut sums_4 = f64x8::splat(0.0);
            for chunk in chunks {
                let diff = *chunk - m_v;
                let diff2 = diff * diff;
                sums_3 += diff2 * diff;
                sums_4 += diff2 * diff2;
            }
            let mut sum_3 = sums_3.reduce_sum();
            let mut sum_4 = sums_4.reduce_sum();
            for &x in prefix {
                let diff = x - m;
                let diff2 = diff * diff;
                sum_3 += diff2 * diff;
                sum_4 += diff2 * diff2;
            }
            for &x in remainder {
                let diff = x - m;
                let diff2 = diff * diff;
                sum_3 += diff2 * diff;
                sum_4 += diff2 * diff2;
            }

            if s != 0.0 {
                let m3 = sum_3 / n_f;
                cache.skew = Some(m3 / s.powi(3));
                let m4 = sum_4 / n_f;
                cache.kurtosis = Some(m4 / s.powi(4) - 3.0);
            } else {
                cache.skew = Some(0.0);
                cache.kurtosis = Some(0.0);
            }
        }

        // Second part: dependencies like sum_xy, sum_diff etc.
        let mut sum_xy = 0.0;
        let mut sum_diff = 0.0;
        let mut sum_abs_diff = 0.0;
        let mut sum_diff_sq = 0.0;
        let mut zero_crossings = 0;

        let mut count_above = 0;
        let mut count_below = 0;
        let mut max_strike_above = 0;
        let mut max_strike_below = 0;
        let mut current_strike_above = 0;
        let mut current_strike_below = 0;
        let mut sum_abs_dev = 0.0;

        // sum_xy can be SIMD-ified too
        let (prefix_xy, chunks_xy, remainder_xy) = data.as_simd::<8>();
        let mut idx_s = 0.0;
        for &x in prefix_xy {
            sum_xy += idx_s * x;
            idx_s += 1.0;
            sum_abs_dev += (x - m).abs();
            if x > m {
                count_above += 1;
                current_strike_above += 1;
                max_strike_below = max_strike_below.max(current_strike_below);
                current_strike_below = 0;
            } else if x < m {
                count_below += 1;
                current_strike_below += 1;
                max_strike_above = max_strike_above.max(current_strike_above);
                current_strike_above = 0;
            } else {
                max_strike_above = max_strike_above.max(current_strike_above);
                max_strike_below = max_strike_below.max(current_strike_below);
                current_strike_above = 0;
                current_strike_below = 0;
            }
        }
        let mut sums_xy = f64x8::splat(0.0);
        let mut indices = f64x8::from_array([
            idx_s,
            idx_s + 1.0,
            idx_s + 2.0,
            idx_s + 3.0,
            idx_s + 4.0,
            idx_s + 5.0,
            idx_s + 6.0,
            idx_s + 7.0,
        ]);
        let step = f64x8::splat(8.0);
        for chunk in chunks_xy {
            sums_xy += indices * *chunk;
            indices += step;
            // For strike counts and mean abs deviation, we stay scalar for now as it's complex to SIMD
            for &x in chunk.as_array() {
                sum_abs_dev += (x - m).abs();
                if x > m {
                    count_above += 1;
                    current_strike_above += 1;
                    max_strike_below = max_strike_below.max(current_strike_below);
                    current_strike_below = 0;
                } else if x < m {
                    count_below += 1;
                    current_strike_below += 1;
                    max_strike_above = max_strike_above.max(current_strike_above);
                    current_strike_above = 0;
                } else {
                    max_strike_above = max_strike_above.max(current_strike_above);
                    max_strike_below = max_strike_below.max(current_strike_below);
                    current_strike_above = 0;
                    current_strike_below = 0;
                }
            }
        }
        sum_xy += sums_xy.reduce_sum();

        let mut idx_scalar = prefix_xy.len() + chunks_xy.len() * 8;
        for &x in remainder_xy {
            sum_xy += (idx_scalar as f64) * x;
            idx_scalar += 1;
            sum_abs_dev += (x - m).abs();
            if x > m {
                count_above += 1;
                current_strike_above += 1;
                max_strike_below = max_strike_below.max(current_strike_below);
                current_strike_below = 0;
            } else if x < m {
                count_below += 1;
                current_strike_below += 1;
                max_strike_above = max_strike_above.max(current_strike_above);
                current_strike_above = 0;
            } else {
                max_strike_above = max_strike_above.max(current_strike_above);
                max_strike_below = max_strike_below.max(current_strike_below);
                current_strike_above = 0;
                current_strike_below = 0;
            }
        }
        max_strike_above = max_strike_above.max(current_strike_above);
        max_strike_below = max_strike_below.max(current_strike_below);

        // Zero crossings and peak counts need a scalar pass
        let mut peak_count_val = 0;
        let mut ssc_count = 0;
        let mut turning_points_count = 0;
        let mut area_sum = 0.0;
        let mut last_crossing = None;
        let mut zc_intervals_sum = 0.0;
        let mut zc_intervals_sq_sum = 0.0;
        let mut zc_intervals_count = 0;

        for i in 1..n {
            let x = data[i];
            let prev = data[i - 1];
            let diff = x - prev;
            sum_diff += diff;
            sum_abs_diff += diff.abs();
            sum_diff_sq += diff * diff;
            area_sum += (x.abs() + prev.abs()) / 2.0;

            if (prev < 0.0 && x >= 0.0) || (prev >= 0.0 && x < 0.0) {
                zero_crossings += 1;
                if let Some(last) = last_crossing {
                    let interval = (i - last) as f64;
                    zc_intervals_sum += interval;
                    zc_intervals_sq_sum += interval * interval;
                    zc_intervals_count += 1;
                }
                last_crossing = Some(i);
            }
            if i < n - 1 {
                let next = data[i + 1];
                let diff2 = next - x;
                if (diff > 0.0 && diff2 < 0.0) || (diff < 0.0 && diff2 > 0.0) {
                    ssc_count += 1;
                }
                if (x > prev && x > next) || (x < prev && x < next) {
                    turning_points_count += 1;
                }
                if x > prev && x > next {
                    peak_count_val += 1;
                }
            }
        }

        // Post-loop derivations
        cache.count_above_mean = Some(count_above as f64);
        cache.count_below_mean = Some(count_below as f64);
        cache.longest_strike_above_mean = Some(max_strike_above as f64);
        cache.longest_strike_below_mean = Some(max_strike_below as f64);
        if m != 0.0 {
            cache.variation_coefficient = Some(cache.std.unwrap() / m);
        } else {
            cache.variation_coefficient = Some(0.0);
        }

        if n > 1 {
            cache.mean_change = Some(sum_diff / (n_f - 1.0));
            cache.mean_abs_change = Some(sum_abs_diff / (n_f - 1.0));
            cache.abs_sum_change = Some(sum_abs_diff);
            cache.cid_ce = Some(sum_diff_sq.sqrt());
            cache.zero_crossing_rate = Some(zero_crossings as f64 / (n_f - 1.0));
        } else {
            cache.mean_change = Some(0.0);
            cache.mean_abs_change = Some(0.0);
            cache.abs_sum_change = Some(0.0);
            cache.cid_ce = Some(0.0);
            cache.zero_crossing_rate = Some(0.0);
        }

        if zc_intervals_count > 0 {
            let zc_mean = zc_intervals_sum / zc_intervals_count as f64;
            cache.zero_crossing_mean = Some(zc_mean);
            let zc_var = (zc_intervals_sq_sum / zc_intervals_count as f64) - (zc_mean * zc_mean);
            cache.zero_crossing_std = Some(zc_var.max(0.0).sqrt());
        } else {
            cache.zero_crossing_mean = Some(0.0);
            cache.zero_crossing_std = Some(0.0);
        }

        cache.peak_count = Some(peak_count_val as f64);
        cache.auc = Some(area_sum);
        cache.slope_sign_change = Some(ssc_count as f64);
        cache.turning_points = Some(turning_points_count as f64);

        let sum_x = n_f * (n_f - 1.0) / 2.0;
        let sum_x2 = n_f * (n_f - 1.0) * (2.0 * n_f - 1.0) / 6.0;
        let denominator = n_f * sum_x2 - sum_x * sum_x;
        if denominator != 0.0 {
            let slope_val = (n_f * sum_xy - sum_x * sum) / denominator;
            cache.slope = Some(slope_val);
            cache.intercept = Some((sum - slope_val * sum_x) / n_f);
        } else {
            cache.slope = Some(0.0);
            cache.intercept = Some(m);
        }
    }
}

fn build_calculation_order(features: Vec<Feature>) -> Vec<Feature> {
    if features.is_empty() {
        return Vec::new();
    }
    let mut calculation_order = Vec::new();
    for feat in features {
        match feat {
            Feature::Mean
            | Feature::Variance
            | Feature::Min
            | Feature::Max
            | Feature::Median
            | Feature::Energy => {
                calculation_order.push(feat);
            }
            Feature::Std => {
                calculation_order.push(Feature::Variance);
                calculation_order.push(Feature::Std);
            }
            Feature::Skew => {
                calculation_order.push(Feature::Std);
                calculation_order.push(Feature::Skew);
            }
            Feature::Kurtosis => {
                calculation_order.push(Feature::Std);
                calculation_order.push(Feature::Kurtosis);
            }
            Feature::Mad => {
                calculation_order.push(Feature::Median);
                calculation_order.push(Feature::Mad);
            }
            Feature::Iqr => {
                calculation_order.push(Feature::Min);
                calculation_order.push(Feature::Max);
                calculation_order.push(Feature::Iqr);
            }
            Feature::Entropy => {
                calculation_order.push(Feature::Min);
                calculation_order.push(Feature::Max);
                calculation_order.push(Feature::Iqr);
                calculation_order.push(Feature::Entropy);
            }
            Feature::Rms => {
                calculation_order.push(Feature::Energy);
                calculation_order.push(Feature::Rms);
            }
            Feature::AutocorrLag1 => {
                calculation_order.push(Feature::Mean);
                calculation_order.push(Feature::AutocorrLag1);
            }
            Feature::MeanAbsChange | Feature::MeanChange | Feature::CidCe => {
                calculation_order.push(Feature::Mean);
                calculation_order.push(feat);
            }
            Feature::Slope | Feature::Intercept => {
                calculation_order.push(Feature::Slope);
                calculation_order.push(Feature::Intercept);
            }
            Feature::Paa(_, _) => {
                calculation_order.push(feat);
            }
            Feature::RootMeanSquare => {
                calculation_order.push(Feature::Energy);
                calculation_order.push(Feature::RootMeanSquare);
            }
            Feature::ZeroCrossingRate | Feature::PeakCount => {
                calculation_order.push(feat);
            }
            Feature::AbsSumChange => {
                calculation_order.push(feat);
            }
            Feature::CountAboveMean
            | Feature::CountBelowMean
            | Feature::LongestStrikeAboveMean
            | Feature::LongestStrikeBelowMean => {
                calculation_order.push(Feature::Mean);
                calculation_order.push(feat);
            }
            Feature::VariationCoefficient => {
                calculation_order.push(Feature::Mean);
                calculation_order.push(Feature::Std);
                calculation_order.push(feat);
            }
            Feature::C3(_) => {
                calculation_order.push(feat);
            }
            Feature::Auc
            | Feature::SlopeSignChange
            | Feature::TurningPoints
            | Feature::ZeroCrossingMean
            | Feature::ZeroCrossingStd => {
                calculation_order.push(feat);
            }
        }
    }

    calculation_order.dedup();
    if calculation_order.contains(&Feature::Variance) {
        let idx = calculation_order.iter().position(|&f| f == Feature::Mean);
        if let Some(idx) = idx {
            calculation_order.remove(idx);
        }
    }
    calculation_order
}

fn extract(data: &[f64], feature: &Feature, cache: &mut Cache) {
    match feature {
        Feature::Mean => mean(data, cache),
        Feature::Variance => variance(data, cache),
        Feature::Std => cache.std = cache.variance.map(|v| v.sqrt()),
        Feature::Min => min(data, cache),
        Feature::Max => max(data, cache),
        Feature::Median => median(data, cache),
        Feature::Skew => skew(data, cache),
        Feature::Kurtosis => kurtosis(data, cache),
        Feature::Mad => mad(data, cache),
        Feature::Iqr => iqr(data, cache),
        Feature::Entropy => entropy(data, cache),
        Feature::Energy => energy(data, cache),
        Feature::Rms | Feature::RootMeanSquare => rms(data, cache),
        Feature::AutocorrLag1 => autocorr_lag1(data, cache),
        Feature::MeanAbsChange => mean_abs_change(data, cache),
        Feature::MeanChange => mean_change(data, cache),
        Feature::CidCe => cid_ce(data, cache),
        Feature::Slope => slope(data, cache),
        Feature::Intercept => intercept(data, cache),
        Feature::ZeroCrossingRate => zero_crossing_rate(data, cache),
        Feature::PeakCount => peak_count(data, cache),
        Feature::Paa(sec, bin) => paa(data, *sec as usize, *bin as usize, cache),
        Feature::AbsSumChange => abs_sum_change(data, cache),
        Feature::CountAboveMean => count_above_mean(data, cache),
        Feature::CountBelowMean => count_below_mean(data, cache),
        Feature::LongestStrikeAboveMean => longest_strike_above_mean(data, cache),
        Feature::LongestStrikeBelowMean => longest_strike_below_mean(data, cache),
        Feature::VariationCoefficient => variation_coefficient(data, cache),
        Feature::Auc => auc(data, cache),
        Feature::SlopeSignChange => slope_sign_change(data, cache),
        Feature::TurningPoints => turning_points(data, cache),
        Feature::ZeroCrossingMean | Feature::ZeroCrossingStd => {
            zero_crossing_derivations(data, cache)
        }
        Feature::C3(lag) => c3(data, *lag as usize, cache),
    }
}
