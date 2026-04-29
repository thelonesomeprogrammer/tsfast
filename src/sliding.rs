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
    pub stride: usize,
    // State per column
    pub states: Vec<ColumnState>,
    pub histories: Vec<Vec<f32>>,
    pub planner: Arc<Mutex<RealFftPlanner<f32>>>,
}

#[pymethods]
impl SlidingExtractor {
    #[new]
    #[pyo3(signature = (feature_str, n_cols, window_size, stride=1))]
    pub fn new(feature_str: Vec<String>, n_cols: usize, window_size: usize, stride: usize) -> Self {
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
            unique_paa_totals: unique_paa_totals.clone(),
            unique_c3_lags: unique_c3_lags.clone(),
            unique_autocorr_lags: unique_autocorr_lags.clone(),
            window_size,
            stride,
            states: (0..n_cols)
                .map(|_| {
                    ColumnState::new(
                        &unique_paa_totals,
                        &unique_c3_lags,
                        &unique_autocorr_lags,
                        0.0,
                    )
                })
                .collect(),
            histories: vec![Vec::with_capacity(window_size + stride); n_cols],
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
                self.histories.push(Vec::with_capacity(self.window_size + self.stride));
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

        let window_size = self.window_size;
        let stride = self.stride;

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
                let engine = SlidingEngine {
                    compute: self.compute,
                    features: &self.features,
                    unique_paa_totals: &self.unique_paa_totals,
                    unique_c3_lags: &self.unique_c3_lags,
                    paa_boundaries: &paa_boundaries,
                    r2c: r2c.as_ref().cloned(),
                    fft_size,
                };
                
                let mut batch_res = Vec::new();
                
                for &val in values {
                    history.push(val);
                    
                    if history.len() == window_size {
                        // First time window is full
                        batch_res.push(engine.process_column(history, state, false));
                    } else if history.len() == window_size + stride {
                        // We have reached a stride boundary
                        let old_slice_vec: Vec<f32> = history[..stride].to_vec();
                        let new_slice_vec: Vec<f32> = history[window_size..window_size + stride].to_vec();
                        
                        let value_after_old = history[stride];
                        let value_before_new = history[window_size - 1];
                        
                        engine.update_batch(
                            &old_slice_vec,
                            &new_slice_vec,
                            None,
                            value_after_old,
                            value_before_new,
                            state.n as usize + window_size,
                            window_size,
                            state
                        );
                        
                        history.drain(..stride);
                        
                        if let Some(ref mut sdft) = state.sliding_dft {
                            for (i, &v) in old_slice_vec.iter().enumerate() {
                                sdft.update(v, new_slice_vec[i]);
                            }
                        }
                        
                        batch_res.push(engine.process_column(history, state, true));
                        state.n += stride as f32;
                    }
                }
                batch_res
            })
            .collect();

        let n_results = column_results[0].len();
        let mut fields = Vec::with_capacity(self.features.len());
        for feat in &self.features {
            fields.push(Field::new(feat.name(), DataType::Float32, false));
        }
        let schema = Arc::new(Schema::new(fields));

        if n_results == 0 {
            return Ok(PyArrowType(RecordBatch::new_empty(schema)));
        }

        let results: Vec<ArrayRef> = (0..self.features.len())
            .map(|feat_idx| {
                let mut flat_data = Vec::with_capacity(n_cols * n_results);
                for col_res in &column_results {
                    for slide_res in col_res {
                        flat_data.push(slide_res[feat_idx]);
                    }
                }
                Arc::new(Float32Array::from(flat_data)) as ArrayRef
            })
            .collect();

        let return_batch = RecordBatch::try_new(schema, results)
            .map_err(|e| PyTypeError::new_err(format!("Failed to create RecordBatch: {}", e)))?;

        Ok(PyArrowType(return_batch))
    }
}
