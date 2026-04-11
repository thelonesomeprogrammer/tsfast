use crate::features::{
    energy::{energy, rms},
    signal::{
        autocorr_lag1, cid_ce, mean_abs_change, mean_change, paa, peak_count, zero_crossing_rate,
    },
    statistics::{entropy, iqr, kurtosis, mad, max, mean, median, min, skew, variance},
    trend::{intercept, slope},
};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
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

        for feat_name in &self.calculation_order {
            match feat_name {
                Feature::Mean => mean(data_slice, &mut cache),
                Feature::Variance => variance(data_slice, &mut cache),
                Feature::Std => cache.std = cache.variance.map(|v| v.sqrt()),
                Feature::Min => min(data_slice, &mut cache),
                Feature::Max => max(data_slice, &mut cache),
                Feature::Median => median(data_slice, &mut cache),
                Feature::Skew => skew(data_slice, &mut cache),
                Feature::Kurtosis => kurtosis(data_slice, &mut cache),
                Feature::Mad => mad(data_slice, &mut cache),
                Feature::Iqr => iqr(data_slice, &mut cache),
                Feature::Entropy => entropy(data_slice, &mut cache),
                Feature::Energy => energy(data_slice, &mut cache),
                Feature::Rms | Feature::RootMeanSquare => rms(data_slice, &mut cache),
                Feature::AutocorrLag1 => autocorr_lag1(data_slice, &mut cache),
                Feature::MeanAbsChange => mean_abs_change(data_slice, &mut cache),
                Feature::MeanChange => mean_change(data_slice, &mut cache),
                Feature::CidCe => cid_ce(data_slice, &mut cache),
                Feature::Slope => slope(data_slice, &mut cache),
                Feature::Intercept => intercept(data_slice, &mut cache),
                Feature::ZeroCrossingRate => zero_crossing_rate(data_slice, &mut cache),
                Feature::PeakCount => peak_count(data_slice, &mut cache),
                Feature::Paa(sec, bin) => paa(data_slice, *sec as usize, *bin as usize, &mut cache),
            }
        }

        let featu = cache.to_vec_with(&self.features);
        Ok(PyArray1::from_vec(_py, featu).to_owned().into())
    }

    pub fn extract2d(
        &self,
        data: Vec<PyReadonlyArray2<'_, f64>>,
        py: Python<'_>,
    ) -> PyResult<Py<PyArray2<f64>>> {
        let mut results = Vec::new();
        let slice = data.as_slice();
        for arr in slice {
            let data_slice = arr.as_slice()?;
            let mut cache = Cache::new();
            for feat_name in &self.calculation_order {
                extract(data_slice, feat_name, &mut cache);
            }
            results.push(cache.to_vec_with(&self.features));
        }
        let arr = PyArray2::from_vec2(py, &results);
        if let Err(e) = arr {
            eprintln!("Error creating 2D array: {:?}", e);
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Failed to create 2D array",
            ));
        }

        Ok(arr.unwrap().to_owned().into())
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
