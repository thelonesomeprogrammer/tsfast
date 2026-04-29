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
use engine::SlidingEngine;

#[pyclass]
pub struct SlidingExtractor {
    pub features: Vec<Feature>,
    pub compute: FastBitArray,
    pub unique_paa_totals: Vec<u16>,
    pub unique_c3_lags: Vec<u16>,
    pub unique_autocorr_lags: Vec<u16>,
    pub window_size: usize,
    // State per column
    pub states: Vec<ColumnState>,
    pub histories: Vec<Vec<f32>>,
    pub planner: Arc<Mutex<RealFftPlanner<f32>>>,
}

#[pymethods]
impl SlidingExtractor {
    #[new]
    #[pyo3(signature = (feature_str, n_cols, window_size))]
    pub fn new(feature_str: Vec<String>, n_cols: usize, window_size: usize) -> Self {
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

        let planner = RealFftPlanner::<f32>::new();
        let planner_arc = Arc::new(Mutex::new(planner));

        if compute.any_fft() {
            let mut p = planner_arc.lock().unwrap();
            p.plan_fft_forward(next_good_fft_size(window_size));
        }

        Self {
            features,
            compute,
            unique_paa_totals,
            unique_c3_lags,
            unique_autocorr_lags,
            window_size,
            states: (0..n_cols)
                .map(|_| ColumnState::new(&[], &[], &[], 0.0))
                .collect(),
            histories: vec![Vec::with_capacity(window_size); n_cols],
            planner: planner_arc,
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
                self.histories.push(Vec::with_capacity(self.window_size));
            }
        }

        let paa_boundaries: Vec<Vec<usize>> = self
            .unique_paa_totals
            .iter()
            .map(|&total| {
                let mut b = Vec::with_capacity(total as usize + 1);
                for i in 0..=total {
                    b.push((i as f32 * self.window_size as f32 / total as f32) as usize);
                }
                b
            })
            .collect();

        let fft_size = next_good_fft_size(self.window_size);
        let r2c = if self.compute.any_fft() {
            let mut p = self.planner.lock().unwrap();
            Some(p.plan_fft_forward(fft_size))
        } else {
            None
        };

        use rayon::prelude::*;

        let column_results: Vec<Vec<Vec<f32>>> = self.states[..n_cols]
            .par_iter_mut()
            .zip(self.histories[..n_cols].par_iter_mut())
            .zip(record_batch.columns().par_iter())
            .map(|((state, history), column)| {
                let array = column
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .expect("Expected Float32Array");

                let values = array.values();
                let mut engine = SlidingEngine {
                    compute: self.compute,
                    features: &self.features,
                    unique_paa_totals: &self.unique_paa_totals,
                    unique_c3_lags: &self.unique_c3_lags,
                    paa_boundaries: &paa_boundaries,
                    r2c: r2c.as_ref().cloned(),
                    fft_size,
                };
                
                let mut batch_res = Vec::with_capacity(values.len());

                for &val in values {
                    if history.len() < self.window_size {
                        history.push(val);
                        if history.len() == self.window_size {
                            // First time window is full, initialize state
                            let res = engine.process_column(history, state, false);
                            batch_res.push(res);
                        } else {
                            batch_res.push(vec![0.0; self.features.len()]);
                        }
                    } else {
                        // Slide window incrementally
                        let old_val = history[0];
                        let old_val_next = history[1];
                        let old_last = *history.last().unwrap();
                        engine.update_incremental(old_val, val, old_val_next, old_last, state.n as usize + self.window_size, self.window_size, state);
                        
                        history.copy_within(1.., 0);
                        *history.last_mut().unwrap() = val;
                        
                        // Sliding DFT update
                        if let Some(ref mut sdft) = state.sliding_dft {
                            sdft.update(old_val, val);
                        }

                        batch_res.push(engine.process_column(history, state, true));
                    }
                    state.n += 1.0;
                }
                batch_res
            })
            .collect();

        // Convert nested Vec results to Arrow RecordBatch
        // This is complex because we have multiple results per input row (sliding window)
        // Usually, for streaming, we return one result row per input row.
        
        let mut fields = Vec::with_capacity(self.features.len());
        for feat in &self.features {
            fields.push(Field::new(feat.name(), DataType::Float32, false));
        }
        let schema = Arc::new(Schema::new(fields));

        let last_results: Vec<ArrayRef> = (0..self.features.len())
            .map(|feat_idx| {
                let col_data: Vec<f32> = column_results.iter().map(|res| res.last().unwrap()[feat_idx]).collect();
                Arc::new(Float32Array::from(col_data)) as ArrayRef
            })
            .collect();

        let return_batch = RecordBatch::try_new(schema, last_results)
            .map_err(|e| PyTypeError::new_err(format!("Failed to create RecordBatch: {}", e)))?;

        Ok(PyArrowType(return_batch))
    }
}
