use crate::features::signal::paa;
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use std::collections::HashMap;

pub type FeatureFn = fn(&[f64], &mut HashMap<&'static str, f64>) -> f64;
pub type ExpandingFeatureFn = fn(&[f64], &mut [f64]);

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

#[pyclass]
pub struct FeatureExtractor {
    pub features: Vec<Feature>,
    pub calculation_order: Vec<Feature>,
}

#[pymethods]
impl FeatureExtractor {
    #[new]
    pub fn new(features: &str) -> Self {
        let features_vec = features.split(',').map(|s| s.trim()).collect::<Vec<&str>>();
        let mut feature_enums = Vec::new();
        for feature in features_vec {
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
                "zerocrossingrate" => feature_enums.push(Feature::ZeroCrossingRate),
                "peakcount" => feature_enums.push(Feature::PeakCount),
                "autocorrlag1" => feature_enums.push(Feature::AutocorrLag1),
                "meanabschange" => feature_enums.push(Feature::MeanAbsChange),
                "meanchange" => feature_enums.push(Feature::MeanChange),
                "cidce" => feature_enums.push(Feature::CidCe),
                "slope" => feature_enums.push(Feature::Slope),
                "intercept" => feature_enums.push(Feature::Intercept),
                st => {
                    if st.contains("paa") {
                        let parts: Vec<&str> = st.split('-').collect();
                        if parts.len() == 3 && parts[0] == "paa" {
                            if let (Ok(n_segments), Ok(n_bins)) =
                                (parts[1].parse::<u16>(), parts[2].parse::<u16>())
                            {
                                feature_enums.push(Feature::Paa(n_segments, n_bins));
                            } else {
                                panic!("Invalid PAA parameters in feature: {}", st);
                            }
                        } else {
                            panic!("Invalid feature name: {}", st);
                        }
                    } else {
                        panic!("Unknown feature: {}", st);
                    }
                }
            }
        }
        let features = feature_enums.clone();
        feature_enums.dedup();
        build_calculation_order(&features);

        Self { features }
    }
    fn extract(
        &self,
        data: PyReadonlyArray1<'_, f64>,
        features: Vec<String>,
        _py: Python<'_>,
    ) -> PyResult<Py<PyArray1<f64>>> {
        let data_slice = data.as_slice()?;

        pub fn extract(data: &[f64], features: &[&str], registry: &Registry) -> Vec<f64> {
            let mut sorted_features = features.to_vec();

            let mut computed = HashMap::new();

            for feat_name in &sorted_features {
                if computed.contains_key(feat_name) {
                    continue;
                }

                if let Some(feat_fn) = registry.batch_features.get(feat_name) {
                    feat_fn(data, &mut computed);
                } else if feat_name.starts_with("paa-") {
                    let parts: Vec<&str> = feat_name.split('-').collect();
                    if parts.len() == 3
                        && let (Ok(segments), Ok(idx)) =
                            (parts[1].parse::<usize>(), parts[2].parse::<usize>())
                    {
                        paa(data, segments, idx, &mut computed);
                    }
                }
            }

            // Return results in the original requested order
            features
                .iter()
                .map(|&f| *computed.get(f).unwrap_or(&f64::NAN))
                .collect()
        }

        Ok(result.into_pyarray(_py).unbind())
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
                calculation_order.push(feat.clone());
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
                calculation_order.push(feat.clone());
            }
            Feature::Slope | Feature::Intercept => {
                calculation_order.push(Feature::Slope);
                calculation_order.push(Feature::Intercept);
            }
            Feature::Paa(_, _) => {
                calculation_order.push(feat.clone());
            }
            Feature::RootMeanSquare => {
                calculation_order.push(Feature::Energy);
                calculation_order.push(Feature::RootMeanSquare);
            }
            Feature::ZeroCrossingRate | Feature::PeakCount => {
                calculation_order.push(feat.clone());
            }
        }
    }

    if calculation_order.contains(&Feature::Variance) {
        let idx = calculation_order.iter().position(|&f| f == Feature::Mean);
        if let Some(idx) = idx {
            calculation_order.remove(idx);
        }
    }
    calculation_order
}
