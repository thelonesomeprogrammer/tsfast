use crate::features::{
    energy::{energy, rms},
    signal::{
        autocorr_lag1, cid_ce, mean_abs_change, mean_change, paa, peak_count, zero_crossing_rate,
    },
    statistics::{entropy, iqr, kurtosis, mad, max, mean, median, min, skew, variance},
    trend::{intercept, slope},
};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;

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
            paa: HashMap::new(),
        }
    }
    fn to_vec_with(&self, features: &[Feature]) -> Vec<f64> {
        features
            .iter()
            .map(|feat| match feat {
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
                Feature::Paa(sec, bin) => *self.paa.get(&(*sec, *bin)).unwrap_or(&f64::NAN),
            })
            .collect()
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
        let featu = cache.to_vec_with(&self.features);
        Ok(PyArray1::from_vec(_py, featu).to_owned().into())
    }

    pub fn extract2d(
        &self,
        input_data: Bound<'_, PyAny>,
    ) -> PyResult<Py<PyArray2<f64>>> {
        let py = input_data.py();
        let mut signals = Vec::new();

        if let Ok(list) = input_data.cast::<pyo3::types::PyList>() {
            for i in 0..list.len() {
                let item = list.get_item(i)?;
                let arr: PyReadonlyArray1<'_, f64> = item.extract()?;
                signals.push(arr.to_vec()?);
            }
        } else if let Ok(arr) = input_data.extract::<PyReadonlyArray2<'_, f64>>() {
            let view = arr.as_array();
            for i in 0..view.shape()[0] {
                let row = view.index_axis(ndarray::Axis(0), i);
                signals.push(row.to_vec());
            }
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Expected a list of 1D arrays or a 2D array",
            ));
        }

        let n_samples = signals.len();
        let n_features = self.features.len();

        // Process in parallel
        let results_vec: Vec<Vec<f64>> = signals
            .par_iter()
            .map(|sig| {
                let mut cache = Cache::new();
                self.extract_to_cache(sig, &mut cache);
                cache.to_vec_with(&self.features)
            })
            .collect();

        // Convert to ndarray Matrix
        let mut results = ndarray::Array2::zeros((n_samples, n_features));
        for (i, row_vec) in results_vec.into_iter().enumerate() {
            for (j, val) in row_vec.into_iter().enumerate() {
                results[[i, j]] = val;
            }
        }

        Ok(results.into_pyarray(py).unbind())
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
                | Feature::Intercept => {
                    fused_needed = true;
                }
                _ => {}
            }
        }

        if fused_needed {
            self.fused_pass(data_slice, cache);
        }

        // 2. Compute remaining features (order statistics, etc.)
        // We handle features that need sorting (Median, MAD, IQR, Entropy) together
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
                    deviations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    cache.mad = Some(median_from_sorted(&deviations));
                }
            }
            // Entropy uses iqr which is already in cache now
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
                _ => {} // already handled
            }
        }
    }

    fn fused_pass(&self, data: &[f64], cache: &mut Cache) {
        let n = data.len();
        if n == 0 {
            return;
        }
        let n_f = n as f64;

        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        let mut min_val = f64::INFINITY;
        let mut max_val = f64::NEG_INFINITY;
        let mut sum_xy = 0.0;

        // 1. Basic stats & linear regression sums
        // Splitting these into simpler loops can help autovectorization
        for i in 0..n {
            let x = data[i];
            sum += x;
            sum_sq += x * x;
            if x < min_val {
                min_val = x;
            }
            if x > max_val {
                max_val = x;
            }
            sum_xy += (i as f64) * x;
        }

        // 2. Differences (mean change, cid_ce, zero crossings)
        let mut sum_diff = 0.0;
        let mut sum_abs_diff = 0.0;
        let mut sum_diff_sq = 0.0;
        let mut zero_crossings = 0;
        if n > 1 {
            for i in 1..n {
                let x = data[i];
                let prev = data[i - 1];
                let diff = x - prev;
                sum_diff += diff;
                sum_abs_diff += diff.abs();
                sum_diff_sq += diff * diff;

                if (prev < 0.0 && x >= 0.0) || (prev >= 0.0 && x < 0.0) {
                    zero_crossings += 1;
                }
            }
        }

        // 3. Peak count
        let mut peak_count_val = 0;
        if n > 2 {
            for i in 1..n - 1 {
                let x = data[i];
                if x > data[i - 1] && x > data[i + 1] {
                    peak_count_val += 1;
                }
            }
        }

        // Post-loop derivations
        let m = sum / n_f;
        cache.mean = Some(m);
        cache.min = Some(min_val);
        cache.max = Some(max_val);
        cache.energy = Some(sum_sq);
        cache.rms = Some((sum_sq / n_f).sqrt());

        // Variance
        let var = (sum_sq / n_f) - (m * m);
        cache.variance = Some(if var < 0.0 { 0.0 } else { var });
        cache.std = Some(cache.variance.unwrap().sqrt());

        if n > 1 {
            cache.mean_change = Some(sum_diff / (n_f - 1.0));
            cache.mean_abs_change = Some(sum_abs_diff / (n_f - 1.0));
            cache.cid_ce = Some(sum_diff_sq.sqrt());
            cache.zero_crossing_rate = Some(zero_crossings as f64 / (n_f - 1.0));
        } else {
            cache.mean_change = Some(0.0);
            cache.mean_abs_change = Some(0.0);
            cache.cid_ce = Some(0.0);
            cache.zero_crossing_rate = Some(0.0);
        }
        cache.peak_count = Some(peak_count_val as f64);

        // Linear Regression for Slope/Intercept
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
    }
}
