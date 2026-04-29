use crate::common::{ColumnState, map_features_to_indices, next_good_fft_size};
use crate::types::{FastBitArray, Feature};
use arrow::array::{ArrayRef, Float32Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::pyarrow::PyArrowType;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use std::sync::{Arc, Mutex};
use realfft::RealFftPlanner;

pub mod engine;
use engine::ExpandingEngine;

#[pyclass]
pub struct ExpandingExtractor {
    pub features: Vec<Feature>,
    pub compute: FastBitArray,
    pub unique_paa_totals: Vec<u16>,
    pub unique_c3_lags: Vec<u16>,
    pub unique_autocorr_lags: Vec<u16>,
    pub paa_boundaries: Vec<Vec<usize>>,
    // State per column
    pub states: Vec<ColumnState>,
    pub histories: Vec<Vec<f32>>,
    pub sorted_histories: Vec<Vec<f32>>,
    pub planner: Arc<Mutex<RealFftPlanner<f32>>>,
    pub max_size: Option<usize>,
    pub fft_update_period: usize,
}

#[pymethods]
impl ExpandingExtractor {
    #[new]
    #[pyo3(signature = (feature_str, n_cols, max_size=None, fft_update_period=1))]
    pub fn new(feature_str: Vec<String>, n_cols: usize, max_size: Option<usize>, fft_update_period: usize) -> Self {
        let mut features = Vec::new();
        let mut unique_paa_totals = std::collections::BTreeSet::new();
        let mut unique_c3_lags = std::collections::BTreeSet::new();
        let mut unique_autocorr_lags = std::collections::BTreeSet::new();

        for i in feature_str {
            let feat = Feature::from(i);
            match feat {
                Feature::Paa(total, _) => {
                    unique_paa_totals.insert(total);
                }
                Feature::C3(lag) => {
                    unique_c3_lags.insert(lag);
                }
                Feature::Autocorr(lag) => {
                    unique_autocorr_lags.insert(lag);
                }
                Feature::PartialAutocorr(lag) => {
                    // PACF also needs autocorrelations up to lag
                    for l in 1..=lag {
                        unique_autocorr_lags.insert(l);
                    }
                }
                _ => {}
            }
            features.push(feat);
        }
        let compute = map_features_to_indices(&features);
        let unique_paa_totals: Vec<u16> = unique_paa_totals.into_iter().collect();
        let unique_c3_lags: Vec<u16> = unique_c3_lags.into_iter().collect();
        let unique_autocorr_lags: Vec<u16> = unique_autocorr_lags.into_iter().collect();

        let planned_size = max_size.map(next_good_fft_size);
        let planner = RealFftPlanner::<f32>::new();
        let planner_arc = Arc::new(Mutex::new(planner));

        if let Some(size) = planned_size {
            if compute.any_fft() {
                let mut p = planner_arc.lock().unwrap();
                p.plan_fft_forward(size);
            }
        }

        Self {
            features,
            compute,
            unique_paa_totals: unique_paa_totals.clone(),
            unique_c3_lags: unique_c3_lags.clone(),
            unique_autocorr_lags: unique_autocorr_lags.clone(),
            paa_boundaries: Vec::new(),
            states: (0..n_cols)
                .map(|_| {
                    ColumnState::new(
                        &unique_paa_totals,
                        &unique_c3_lags,
                        &unique_autocorr_lags,
                        0.0,
                    )
                })
                .collect(), // Initial placeholder
            histories: vec![Vec::new(); n_cols],
            sorted_histories: vec![Vec::new(); n_cols],
            planner: planner_arc,
            max_size,
            fft_update_period,
        }
    }

    pub fn update(
        &mut self,
        batch: PyArrowType<RecordBatch>,
    ) -> PyResult<PyArrowType<RecordBatch>> {
        let record_batch = batch.0;
        let n_cols = record_batch.num_columns();
        let n_rows = record_batch.num_rows();

        if n_rows == 0 {
            return Ok(PyArrowType(record_batch));
        }

        if self.states.len() < n_cols {
            for _ in self.states.len()..n_cols {
                self.states.push(ColumnState::new(
                    &self.unique_paa_totals,
                    &self.unique_c3_lags,
                    &self.unique_autocorr_lags,
                    0.0,
                ));
                self.histories.push(Vec::new());
                self.sorted_histories.push(Vec::new());
            }
        }

        // Pre-calculate PAA boundaries once for all columns
        let total_n = self.histories[0].len() + n_rows;
        let current_paa_boundaries: Vec<Vec<usize>> = self
            .unique_paa_totals
            .iter()
            .map(|&total| {
                let mut b = Vec::with_capacity(total as usize + 1);
                for i in 0..=total {
                    b.push((i as f32 * total_n as f32 / total as f32) as usize);
                }
                b
            })
            .collect();

        // Ensure we have a plan for the current total_n or max_size
        let fft_size = if let Some(ms) = self.max_size {
            next_good_fft_size(ms.max(total_n))
        } else {
            total_n
        };

        let r2c = if self.compute.any_fft() && fft_size > 0 {
            let mut p = self.planner.lock().unwrap();
            Some(p.plan_fft_forward(fft_size))
        } else {
            None
        };

        use rayon::prelude::*;

        let column_results: Vec<Vec<f32>> = self.states[..n_cols]
            .par_iter_mut()
            .zip(self.histories[..n_cols].par_iter_mut())
            .zip(self.sorted_histories[..n_cols].par_iter_mut())
            .zip(record_batch.columns().par_iter())
            .map(|(((state, history), sorted_history), column)| {
                let array = column
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .expect("Expected Float32Array");

                let values = array.values();

                if history.is_empty() {
                    *state = ColumnState::new(
                        &self.unique_paa_totals,
                        &self.unique_c3_lags,
                        &self.unique_autocorr_lags,
                        values[0],
                    );
                }

                let mut engine = ExpandingEngine {
                    compute: self.compute,
                    features: &self.features,
                    unique_paa_totals: &self.unique_paa_totals,
                    unique_c3_lags: &self.unique_c3_lags,
                    unique_autocorr_lags: &self.unique_autocorr_lags,
                    paa_boundaries: &current_paa_boundaries,
                    r2c: r2c.as_ref().cloned(),
                    fft_size,
                    fft_update_period: self.fft_update_period,
                };

                engine.process_expanding(
                    values,
                    history.len(),
                    state,
                    history,
                    sorted_history,
                )
            })
            .collect();

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
