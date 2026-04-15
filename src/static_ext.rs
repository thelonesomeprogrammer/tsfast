use crate::types::FastBitArray;
use crate::types::Feature;
use arrow::array::{ArrayRef, Float32Array, RecordBatch};
use arrow::datatypes::DataType;
use arrow::datatypes::{Field, Schema};
use arrow::pyarrow::PyArrowType;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::sync::Arc;

pub mod extractor;

use crate::common::map_features_to_indices;
use extractor::StaticEngine;

#[pyclass]
pub struct Extractor {
    pub features: Vec<Feature>,
    pub compute: FastBitArray,
    pub paa_args: Vec<(u16, u16)>,
    pub c3_args: Vec<u16>,
    pub unique_paa_totals: Vec<u16>,
    pub unique_c3_lags: Vec<u16>,
}

#[pymethods]
impl Extractor {
    #[new]
    pub fn new(feature_str: Vec<String>) -> Self {
        let mut features = Vec::new();
        let mut paa_args = Vec::new();
        let mut c3_args = Vec::new();
        let mut unique_paa_totals = std::collections::BTreeSet::new();
        let mut unique_c3_lags = std::collections::BTreeSet::new();

        for i in feature_str {
            let feat = Feature::from(i);
            if let Feature::Paa(total, index) = feat {
                paa_args.push((total, index));
                unique_paa_totals.insert(total);
            } else if let Feature::C3(lag) = feat {
                c3_args.push(lag);
                unique_c3_lags.insert(lag);
            }
            features.push(feat);
        }
        let compute = map_features_to_indices(&features);

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

                let processor = StaticEngine {
                    compute,
                    features,
                    unique_paa_totals,
                    unique_c3_lags,
                    paa_boundaries: &paa_boundaries,
                };
                processor.process_column(float_array.values())
            })
            .collect();

        let mut fields = Vec::with_capacity(features.len());
        let results: Vec<ArrayRef> = (0..features.len())
            .into_par_iter()
            .map(|feat_idx| {
                let mut col_data = Vec::with_capacity(n_cols);
                for col in column_results.iter().take(n_cols) {
                    col_data.push(col[feat_idx]);
                }
                Arc::new(Float32Array::from(col_data)) as ArrayRef
            })
            .collect();

        for feat in features {
            fields.push(Field::new(feat.name(), DataType::Float32, false));
        }

        let return_batch =
            RecordBatch::try_new(Arc::new(Schema::new(fields)), results).map_err(|e| {
                PyTypeError::new_err(format!("Failed to create return RecordBatch: {}", e))
            })?;

        Ok(PyArrowType(return_batch))
    }
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
        let extractor = Extractor::new(features);
        let batch = create_test_batch(data.clone());
        let result = extractor.process_2d_floats(batch).unwrap().0;

        assert_eq!(result.num_columns(), 34);

        for i in 0..result.num_columns() {
            let col = result
                .column(i)
                .as_any()
                .downcast_ref::<Float32Array>()
                .unwrap();
            let val = col.value(0);
            println!("{}: {}", result.schema().field(i).name(), val);
            assert!(
                !val.is_nan(),
                "Feature {} is NaN",
                result.schema().field(i).name()
            );
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
        let extractor = Extractor::new(features);
        let batch = create_test_batch(data.clone());
        let result = extractor.process_2d_floats(batch).unwrap().0;

        let n = data.len() as f32;
        let scalar_sum: f32 = data.iter().sum();
        let scalar_mean = scalar_sum / n;
        let scalar_var = data.iter().map(|&x| (x - scalar_mean).powi(2)).sum::<f32>() / (n - 1.0);
        let scalar_energy = data.iter().map(|&x| x * x).sum::<f32>();
        let scalar_mad = data.iter().map(|&x| (x - scalar_mean).abs()).sum::<f32>() / n;

        let col_mean = result
            .column(0)
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap()
            .value(0);
        let col_var = result
            .column(1)
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap()
            .value(0);
        let col_sum = result
            .column(2)
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap()
            .value(0);
        let col_energy = result
            .column(6)
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap()
            .value(0);
        let col_mad = result
            .column(8)
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap()
            .value(0);

        assert!((col_mean - scalar_mean).abs() < 1e-5);
        assert!((col_var - scalar_var).abs() < 1e-5);
        assert!((col_sum - scalar_sum).abs() < 1e-5);
        assert!((col_energy - scalar_energy).abs() < 1e-5);
        assert!((col_mad - scalar_mad).abs() < 1e-5);
    }
}
