use crate::common::{ColumnState, map_features_to_indices};
use crate::types::{FastBitArray, Feature};
use arrow::array::{ArrayRef, Float32Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::pyarrow::PyArrowType;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use std::sync::Arc;

pub mod engine;
use engine::ExpandingEngine;

#[pyclass]
pub struct ExpandingExtractor {
    pub features: Vec<Feature>,
    pub compute: FastBitArray,
    pub unique_paa_totals: Vec<u16>,
    pub unique_c3_lags: Vec<u16>,
    pub paa_boundaries: Vec<Vec<usize>>,
    // State per column
    pub states: Vec<ColumnState>,
    pub histories: Vec<Vec<f32>>,
    pub sorted_histories: Vec<Vec<f32>>,
}

#[pymethods]
impl ExpandingExtractor {
    #[new]
    pub fn new(feature_str: Vec<String>, n_cols: usize) -> Self {
        let mut features = Vec::new();
        let mut unique_paa_totals = std::collections::BTreeSet::new();
        let mut unique_c3_lags = std::collections::BTreeSet::new();

        for i in feature_str {
            let feat = Feature::from(i);
            if let Feature::Paa(total, _) = feat {
                unique_paa_totals.insert(total);
            } else if let Feature::C3(lag) = feat {
                unique_c3_lags.insert(lag);
            }
            features.push(feat);
        }
        let compute = map_features_to_indices(&features);
        let unique_paa_totals: Vec<u16> = unique_paa_totals.into_iter().collect();
        let unique_c3_lags: Vec<u16> = unique_c3_lags.into_iter().collect();

        // We can't pre-calculate paa_boundaries if we don't know the total N.
        // But the static extractor seems to use them. 
        // In expanding window, N changes every step.
        // If the user expects PAA, it might be tricky.
        // For now, let's keep it empty and see.

        Self {
            features,
            compute,
            unique_paa_totals,
            unique_c3_lags,
            paa_boundaries: Vec::new(),
            states: (0..n_cols).map(|_| ColumnState::new(&[], &[], 0.0)).collect(), // Initial placeholder
            histories: vec![Vec::new(); n_cols],
            sorted_histories: vec![Vec::new(); n_cols],
        }
    }

    pub fn update(&mut self, batch: PyArrowType<RecordBatch>) -> PyResult<PyArrowType<RecordBatch>> {
        let record_batch = batch.0;
        let n_cols = record_batch.num_columns();
        let n_rows = record_batch.num_rows();

        if n_rows == 0 {
            return Ok(PyArrowType(record_batch));
        }

        // Initialize states if it's the first call or first time for these columns
        if self.states.len() < n_cols {
             // This shouldn't happen if n_cols is fixed at creation, but let's be safe
             for _ in self.states.len()..n_cols {
                 self.states.push(ColumnState::new(&self.unique_paa_totals, &self.unique_c3_lags, 0.0));
                 self.histories.push(Vec::new());
                 self.sorted_histories.push(Vec::new());
             }
        }

        let mut column_results = Vec::with_capacity(n_cols);
        
        // PAA boundaries need N, which is histories[col].len() + n_rows
        // We might need to recalculate them per column if they have different lengths.
        // But usually expanding window is used on synchronised streams.
        
        for col_idx in 0..n_cols {
            let array = record_batch
                .column(col_idx)
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| PyTypeError::new_err("Expected Float32Array"))?;
            
            let values = array.values();
            
            // If this is the first update for this column, initialize prev_last
            if self.histories[col_idx].is_empty() {
                self.states[col_idx] = ColumnState::new(&self.unique_paa_totals, &self.unique_c3_lags, values[0]);
            }

            // PAA boundaries recalculation
            let total_n = self.histories[col_idx].len() + values.len();
            let mut current_paa_boundaries = Vec::new();
            for &total in &self.unique_paa_totals {
                let mut b = Vec::with_capacity(total as usize + 1);
                for i in 0..=total {
                    b.push((i as f32 * total_n as f32 / total as f32) as usize);
                }
                current_paa_boundaries.push(b);
            }

            let mut engine = ExpandingEngine {
                compute: self.compute,
                features: &self.features,
                unique_paa_totals: &self.unique_paa_totals,
                unique_c3_lags: &self.unique_c3_lags,
                paa_boundaries: &current_paa_boundaries,
            };

            let res = engine.process_expanding(
                values,
                self.histories[col_idx].len(),
                &mut self.states[col_idx],
                &mut self.histories[col_idx],
                &mut self.sorted_histories[col_idx],
            );
            column_results.push(res);
        }

        // Format result as RecordBatch
        // This returns 1 row (the latest feature values) for each column.
        // Wait, usually people want a 2D result: (features x columns)
        // Static extractor: n_cols input -> n_cols rows in result, each row has features as columns.
        // Let's follow that.

        let mut fields = Vec::with_capacity(self.features.len());
        for feat in &self.features {
            fields.push(Field::new(feat.name(), DataType::Float32, false));
        }
        let schema = Arc::new(Schema::new(fields));

        let results: Vec<ArrayRef> = (0..self.features.len())
            .map(|feat_idx| {
                let col_data: Vec<f32> = column_results.iter().map(|res| res[feat_idx]).collect();
                Arc::new(Float32Array::from(col_data)) as ArrayRef
            })
            .collect();

        let return_batch = RecordBatch::try_new(schema, results)
            .map_err(|e| PyTypeError::new_err(format!("Failed to create RecordBatch: {}", e)))?;

        Ok(PyArrowType(return_batch))
    }
}
