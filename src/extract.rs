use arrow::array::{Float32Array, RecordBatch};
use arrow::datatypes::DataType;
use arrow::datatypes::{Field, Schema};
use arrow::pyarrow::PyArrowType;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use std::simd::f32x4;
use std::simd::num::SimdFloat;
use std::sync::Arc;

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
    MeanAbsDeviation,
    C3(u16),
    Auc,
    SlopeSignChange,
    TurningPoints,
    ZeroCrossingMean,
    ZeroCrossingStd,
}

#[pyclass]
pub struct FeatureExtractor {
    pub features: Vec<Feature>,
}

#[pymethods]
impl FeatureExtractor {
    #[new]
    pub fn new(feature_str: Vec<String>) -> Self {
        let mut features = Vec::new();
        for i in feature_str {
            match i.as_str() {
                "mean" => features.push(Feature::Mean),
                "variance" => features.push(Feature::Variance),
                "std" => features.push(Feature::Std),
                "min" => features.push(Feature::Min),
                "max" => features.push(Feature::Max),
                "median" => features.push(Feature::Median),
                "skew" => features.push(Feature::Skew),
                "kurtosis" => features.push(Feature::Kurtosis),
                "mad" => features.push(Feature::Mad),
                "iqr" => features.push(Feature::Iqr),
                "entropy" => features.push(Feature::Entropy),
                "energy" => features.push(Feature::Energy),
                "rms" => features.push(Feature::Rms),
                "root_mean_square" => features.push(Feature::RootMeanSquare),
                "zero_crossing_rate" => features.push(Feature::ZeroCrossingRate),
                "peak_count" => features.push(Feature::PeakCount),
                "autocorr_lag1" => features.push(Feature::AutocorrLag1),
                "mean_abs_change" => features.push(Feature::MeanAbsChange),
                "mean_change" => features.push(Feature::MeanChange),
                "cid_ce" => features.push(Feature::CidCe),
                "slope" => features.push(Feature::Slope),
                "intercept" => features.push(Feature::Intercept),
                "abs_sum_change" => features.push(Feature::AbsSumChange),
                "count_above_mean" => features.push(Feature::CountAboveMean),
                "count_below_mean" => features.push(Feature::CountBelowMean),
                "longest_strike_above_mean" => features.push(Feature::LongestStrikeAboveMean),
                "longest_strike_below_mean" => features.push(Feature::LongestStrikeBelowMean),
                "variation_coefficient" => features.push(Feature::VariationCoefficient),
                "mean_abs_deviation" => features.push(Feature::MeanAbsDeviation),
                "auc" => features.push(Feature::Auc),
                "slope_sign_change" => features.push(Feature::SlopeSignChange),
                "turning_points" => features.push(Feature::TurningPoints),
                "zero_crossing_mean" => features.push(Feature::ZeroCrossingMean),
                "zero_crossing_std" => features.push(Feature::ZeroCrossingStd),
                e => {
                    if let Some(arg) = e.strip_prefix("paa-") {
                        let params: Vec<&str> = arg.split('-').collect();
                        if params.len() == 2
                            && let (Ok(n), Ok(m)) =
                                (params[0].parse::<u16>(), params[1].parse::<u16>())
                        {
                            features.push(Feature::Paa(n, m));
                            continue;
                        }
                    } else if let Some(arg) = e.strip_prefix("c3-") {
                        let param = arg.trim();
                        if let Ok(n) = param.parse::<u16>() {
                            features.push(Feature::C3(n));
                            continue;
                        }
                    }
                    panic!("Unknown feature: {}", e);
                }
            }
        }

        Self { features }
    }

    pub fn process_2d_floats(
        &self,
        batch: PyArrowType<RecordBatch>,
    ) -> PyResult<PyArrowType<RecordBatch>> {
        // PyArrowType acts as a transparent wrapper. Extract the underlying RecordBatch.
        let record_batch = batch.0;

        let mut results = Vec::with_capacity(record_batch.num_columns());

        const LANES: usize = 4; // f32x4 processes 4 floats at a time

        // Treat the RecordBatch as a 2D grid: iterate over columns, then rows.
        for col in record_batch.columns() {
            // Enforce the f64 constraint
            if col.data_type() != &DataType::Float32 && col.data_type() != &DataType::Float64 {
                return Err(PyTypeError::new_err("All columns must be Float64."));
            }

            // Safely downcast the generic array into a specific Float64Array
            let float_array = col
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| PyTypeError::new_err("Failed to downcast column to Float64Array"))?;

            let mut total_sum: f32 = 0.0;
            let mut min_value: f32 = f32::INFINITY;
            let mut max_value: f32 = f32::NEG_INFINITY;
            let mut mean: f32 = 0.0;
            let mut m2: f32 = 0.0;
            let mut m3: f32 = 0.0;
            let mut m4: f32 = 0.0;
            let n = float_array.len() as f32;
            let mut energy: f32 = 0.0;
            let mut mac_sum_vec = f32x4::splat(0.0);
            let mut mc_sum_vec = f32x4::splat(0.0);

            let mut iter = float_array.values().chunks_exact(LANES).peekable();

            while let Some(i) = iter.next() {
                let chunk = f32x4::from_slice(i);
                let chunk2 = if let Some(next_chunk) = iter.peek() {
                    if next_chunk.len() < LANES {
                        f32x4::from_slice(&[0.0; LANES])
                    } else {
                        f32x4::from_slice(next_chunk)
                    }
                } else {
                    f32x4::from_slice(&[0.0; LANES])
                };
                mac_sum_vec += (chunk2 - chunk).abs();
                mc_sum_vec += chunk2 - chunk;

                total_sum += chunk.reduce_sum();
                min_value = min_value.min(chunk.reduce_min());
                max_value = max_value.max(chunk.reduce_max());
                let delta = chunk - f32x4::splat(mean);
                mean += delta.reduce_sum() / n;
                m2 += (delta * (chunk - f32x4::splat(mean))).reduce_sum();
                m3 += (delta * delta * (chunk - f32x4::splat(mean))).reduce_sum();
                m4 += (delta * delta * delta * (chunk - f32x4::splat(mean))).reduce_sum();
                energy += (chunk * chunk).reduce_sum();
            }

            let mac_sum = mac_sum_vec.reduce_sum();
            let mc_sum = mc_sum_vec.reduce_sum();

            let mac = mac_sum / n;
            let mean_change = mc_sum / n;

            let variance = m2 / (n - 1.0);
            let std_dev = variance.sqrt();
            let skewness = (m3 / n) / variance.powf(1.5);
            let kurtosis = (m4 / n) / (variance * variance);
            let vc = std_dev / mean;
            let rms = (energy / n).sqrt();
            results.push(Arc::new(Float32Array::from_iter([
                total_sum,
                min_value,
                max_value,
                mean,
                std_dev,
                skewness,
                kurtosis,
                vc,
                rms,
                energy,
                variance,
                mac,
                mean_change,
                mac_sum,
            ])) as Arc<dyn arrow::array::Array>);
        }

        let return_batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("total_sum", DataType::Float32, false),
                Field::new("min_value", DataType::Float32, false),
                Field::new("max_value", DataType::Float32, false),
                Field::new("mean", DataType::Float32, false),
                Field::new("std_dev", DataType::Float32, false),
                Field::new("skewness", DataType::Float32, false),
                Field::new("kurtosis", DataType::Float32, false),
                Field::new("variation_coefficient", DataType::Float32, false),
                Field::new("rms", DataType::Float32, false),
                Field::new("energy", DataType::Float32, false),
                Field::new("variance", DataType::Float32, false),
                Field::new("mean_abs_change", DataType::Float32, false),
                Field::new("mean_change", DataType::Float32, false),
                Field::new("abs_sum_change", DataType::Float32, false),
            ])),
            results,
        )
        .map_err(|e| PyTypeError::new_err(format!("Failed to create return RecordBatch: {}", e)))?;
        Ok(PyArrowType(return_batch))
    }
}
