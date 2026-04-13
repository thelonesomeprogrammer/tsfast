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
    MeanAbsDeviation,
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
}

#[pymethods]
impl ArrowExtractor {
    #[new]
    pub fn new(feature_str: Vec<String>) -> Self {
        let mut features = Vec::new();
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
                "mean_abs_deviation" => features.push(ArrowFeature::MeanAbsDeviation),
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
                            continue;
                        }
                    } else if let Some(arg) = e.strip_prefix("c3-") {
                        let param = arg.trim();
                        if let Ok(n) = param.parse::<u16>() {
                            features.push(ArrowFeature::C3(n));
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
        let record_batch = batch.0;
        let n_cols = record_batch.num_columns();

        // 14 fixed features for now
        let mut total_sums = Vec::with_capacity(n_cols);
        let mut min_values = Vec::with_capacity(n_cols);
        let mut max_values = Vec::with_capacity(n_cols);
        let mut means = Vec::with_capacity(n_cols);
        let mut std_devs = Vec::with_capacity(n_cols);
        let mut skewnesses = Vec::with_capacity(n_cols);
        let mut kurtosises = Vec::with_capacity(n_cols);
        let mut vcs = Vec::with_capacity(n_cols);
        let mut rmss = Vec::with_capacity(n_cols);
        let mut energies = Vec::with_capacity(n_cols);
        let mut variances = Vec::with_capacity(n_cols);
        let mut macs = Vec::with_capacity(n_cols);
        let mut mean_changes = Vec::with_capacity(n_cols);
        let mut mac_sums = Vec::with_capacity(n_cols);

        const LANES: usize = 4;

        for col in record_batch.columns() {
            if col.data_type() != &DataType::Float32 && col.data_type() != &DataType::Float64 {
                return Err(PyTypeError::new_err("All columns must be Float32 or Float64."));
            }

            let float_array = col
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| PyTypeError::new_err("Failed to downcast column to Float32Array"))?;

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
            let variance = m2 / (n - 1.0);
            let std_dev = variance.sqrt();

            total_sums.push(total_sum);
            min_values.push(min_value);
            max_values.push(max_value);
            means.push(mean);
            std_devs.push(std_dev);
            skewnesses.push((m3 / n) / variance.powf(1.5));
            kurtosises.push((m4 / n) / (variance * variance));
            vcs.push(std_dev / mean);
            rmss.push((energy / n).sqrt());
            energies.push(energy);
            variances.push(variance);
            macs.push(mac_sum / n);
            mean_changes.push(mc_sum / n);
            mac_sums.push(mac_sum);
        }

        let schema = Schema::new(vec![
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
        ]);

        let results: Vec<Arc<dyn arrow::array::Array>> = vec![
            Arc::new(Float32Array::from(total_sums)),
            Arc::new(Float32Array::from(min_values)),
            Arc::new(Float32Array::from(max_values)),
            Arc::new(Float32Array::from(means)),
            Arc::new(Float32Array::from(std_devs)),
            Arc::new(Float32Array::from(skewnesses)),
            Arc::new(Float32Array::from(kurtosises)),
            Arc::new(Float32Array::from(vcs)),
            Arc::new(Float32Array::from(rmss)),
            Arc::new(Float32Array::from(energies)),
            Arc::new(Float32Array::from(variances)),
            Arc::new(Float32Array::from(macs)),
            Arc::new(Float32Array::from(mean_changes)),
            Arc::new(Float32Array::from(mac_sums)),
        ];

        let return_batch = RecordBatch::try_new(Arc::new(schema), results)
            .map_err(|e| PyTypeError::new_err(format!("Failed to create return RecordBatch: {}", e)))?;

        Ok(PyArrowType(return_batch))
    }
}
