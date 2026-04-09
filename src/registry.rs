use std::collections::HashMap;
use crate::features::{statistics, energy, signal, trend};

pub type FeatureFn = fn(&[f64]) -> f64;
pub type ExpandingFeatureFn = fn(&[f64], &mut [f64]);

pub struct Registry {
    pub batch_features: HashMap<&'static str, FeatureFn>,
    pub expanding_features: HashMap<&'static str, ExpandingFeatureFn>,
}

impl Registry {
    pub fn new() -> Self {
        let mut batch = HashMap::new();
        batch.insert("mean", statistics::mean as FeatureFn);
        batch.insert("variance", statistics::variance as FeatureFn);
        batch.insert("std", statistics::std as FeatureFn);
        batch.insert("min", statistics::min as FeatureFn);
        batch.insert("max", statistics::max as FeatureFn);
        batch.insert("median", statistics::median as FeatureFn);
        batch.insert("skew", statistics::skew as FeatureFn);
        batch.insert("kurtosis", statistics::kurtosis as FeatureFn);
        batch.insert("mad", statistics::mad as FeatureFn);
        batch.insert("iqr", statistics::iqr as FeatureFn);
        batch.insert("entropy", statistics::entropy as FeatureFn);
        batch.insert("energy", energy::energy as FeatureFn);
        batch.insert("rms", energy::rms as FeatureFn);
        batch.insert("root_mean_square", energy::rms as FeatureFn);
        batch.insert("zero_crossing_rate", signal::zero_crossing_rate as FeatureFn);
        batch.insert("peak_count", signal::peak_count as FeatureFn);
        batch.insert("autocorr_lag1", signal::autocorr_lag1 as FeatureFn);
        batch.insert("mean_abs_change", signal::mean_abs_change as FeatureFn);
        batch.insert("mean_change", signal::mean_change as FeatureFn);
        batch.insert("cid_ce", signal::cid_ce as FeatureFn);
        batch.insert("slope", trend::slope as FeatureFn);
        batch.insert("intercept", trend::intercept as FeatureFn);

        let mut expanding = HashMap::new();
        expanding.insert("mean", crate::expanding_inner::mean as ExpandingFeatureFn);
        expanding.insert("variance", crate::expanding_inner::variance as ExpandingFeatureFn);
        expanding.insert("std", crate::expanding_inner::std as ExpandingFeatureFn);
        expanding.insert("energy", crate::expanding_inner::energy as ExpandingFeatureFn);
        expanding.insert("rms", crate::expanding_inner::rms as ExpandingFeatureFn);
        expanding.insert("min", crate::expanding_inner::min as ExpandingFeatureFn);
        expanding.insert("max", crate::expanding_inner::max as ExpandingFeatureFn);
        expanding.insert("autocorr_lag1", crate::expanding_inner::autocorr_lag1 as ExpandingFeatureFn);
        expanding.insert("mean_abs_change", crate::expanding_inner::mean_abs_change as ExpandingFeatureFn);
        expanding.insert("mean_change", crate::expanding_inner::mean_change as ExpandingFeatureFn);

        Self {
            batch_features: batch,
            expanding_features: expanding,
        }
    }
}
