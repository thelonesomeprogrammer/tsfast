use crate::common::{ColumnState, map_features_to_indices, next_good_fft_size};
use crate::types::{Compute, Feature};
use crate::numpy_io;
use numpy::PyArray2;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use realfft::RealFftPlanner;
use std::sync::{Arc, Mutex};

pub mod engine;
pub mod processors;
use engine::ExpandingEngine;

#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub struct ExpandingExtractor {
    pub features: Vec<Feature>,
    pub compute: Compute,
    pub unique_paa_totals: Vec<u16>,
    pub unique_c3_lags: Vec<u16>,
    pub unique_autocorr_lags: Vec<u16>,
    pub unique_tra_lags: Vec<u16>,
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
    pub fn new(
        feature_str: Vec<String>,
        n_cols: usize,
        max_size: Option<usize>,
        fft_update_period: usize,
    ) -> PyResult<Self> {
        let mut features = Vec::new();
        let mut unique_paa_totals = std::collections::BTreeSet::new();
        let mut unique_c3_lags = std::collections::BTreeSet::new();
        let mut unique_autocorr_lags = std::collections::BTreeSet::new();
        let mut unique_tra_lags = std::collections::BTreeSet::new();

        for i in feature_str {
            let feat = std::str::FromStr::from_str(&i)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
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
                Feature::TimeReversalAsymmetry(lag) => {
                    unique_tra_lags.insert(lag);
                }
                _ => {}
            }
            features.push(feat);
        }
        let compute = map_features_to_indices(&features);
        let unique_paa_totals: Vec<u16> = unique_paa_totals.into_iter().collect();
        let unique_c3_lags: Vec<u16> = unique_c3_lags.into_iter().collect();
        let unique_autocorr_lags: Vec<u16> = unique_autocorr_lags.into_iter().collect();
        let unique_tra_lags: Vec<u16> = unique_tra_lags.into_iter().collect();

        let planned_size = max_size.map(next_good_fft_size);
        let planner = RealFftPlanner::<f32>::new();
        let planner_arc = Arc::new(Mutex::new(planner));

        if let Some(size) = planned_size {
            if compute.intersects(Compute::ANY_FFT) {
                let mut p = planner_arc.lock().unwrap_or_else(|e| e.into_inner());
                p.plan_fft_forward(size);
            }
        }

        Ok(Self {
            features,
            compute,
            unique_paa_totals: unique_paa_totals.clone(),
            unique_c3_lags: unique_c3_lags.clone(),
            unique_autocorr_lags: unique_autocorr_lags.clone(),
            unique_tra_lags: unique_tra_lags.clone(),
            paa_boundaries: Vec::new(),
            states: (0..n_cols)
                .map(|_| {
                    ColumnState::new(
                        &unique_paa_totals,
                        &unique_c3_lags,
                        &unique_autocorr_lags,
                        &unique_tra_lags,
                        0.0,
                    )
                })
                .collect(), // Initial placeholder
            histories: vec![Vec::new(); n_cols],
            sorted_histories: vec![Vec::new(); n_cols],
            planner: planner_arc,
            max_size,
            fft_update_period,
        })
    }

    /// Canonical names of the output columns, in order.
    #[getter]
    pub fn feature_names(&self) -> Vec<String> {
        self.features.iter().map(Feature::name).collect()
    }

    /// Append `values`, a 2-D float32/float64 numpy array of shape
    /// (n_series, n_new_samples), to each series. Returns a float32 array of
    /// shape (n_series, n_features): the features of everything seen so far.
    /// No new samples gives zero rows.
    pub fn update<'py>(
        &mut self,
        py: Python<'py>,
        values: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let (n_series, data) = numpy_io::with_rows(values, |rows, n_samples| {
            let n_series = if n_samples == 0 { 0 } else { rows.len() };
            let data = py
                .detach(|| self.update_rows(&rows[..n_series]))
                .map_err(PyTypeError::new_err)?;
            Ok((n_series, data))
        })?;
        numpy_io::to_numpy_2d(py, n_series, self.features.len(), data)
    }
}

impl ExpandingExtractor {
    /// Append one row of new samples per series (all the same length). Returns
    /// each series' features, flattened row-major.
    pub fn update_rows(&mut self, rows: &[&[f32]]) -> Result<Vec<f32>, String> {
        let n_cols = rows.len();
        if n_cols == 0 || rows[0].is_empty() {
            return Ok(Vec::new());
        }

        if self.states.len() < n_cols {
            for _ in self.states.len()..n_cols {
                self.states.push(ColumnState::new(
                    &self.unique_paa_totals,
                    &self.unique_c3_lags,
                    &self.unique_autocorr_lags,
                    &self.unique_tra_lags,
                    0.0,
                ));
                self.histories.push(Vec::new());
                self.sorted_histories.push(Vec::new());
            }
        }

        // Pre-calculate PAA boundaries once for all columns
        let total_n = self.histories[0].len() + rows[0].len();
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

        // Exact-length FFT: padding would change the spectrum (see spectral::rfft).
        let fft_size = total_n;

        let r2c = if self.compute.intersects(Compute::ANY_FFT) && fft_size > 0 {
            let mut p = self.planner.lock().unwrap_or_else(|e| e.into_inner());
            Some(p.plan_fft_forward(fft_size))
        } else {
            None
        };

        use rayon::prelude::*;

        let column_results: Result<Vec<Vec<f32>>, String> = self.states[..n_cols]
            .par_iter_mut()
            .zip(self.histories[..n_cols].par_iter_mut())
            .zip(self.sorted_histories[..n_cols].par_iter_mut())
            .zip(rows.par_iter())
            .map(|(((state, history), sorted_history), &values)| {

                if history.is_empty() {
                    *state = ColumnState::new(
                        &self.unique_paa_totals,
                        &self.unique_c3_lags,
                        &self.unique_autocorr_lags,
                        &self.unique_tra_lags,
                        values[0],
                    );
                }

                let mut engine = ExpandingEngine {
                    compute: self.compute,
                    features: &self.features,
                    unique_paa_totals: &self.unique_paa_totals,
                    unique_c3_lags: &self.unique_c3_lags,
                    unique_autocorr_lags: &self.unique_autocorr_lags,
                    unique_tra_lags: &self.unique_tra_lags,
                    paa_boundaries: &current_paa_boundaries,
                    r2c: r2c.as_ref().cloned(),
                    fft_update_period: self.fft_update_period,
                };

                Ok(engine.process_expanding(values, history.len(), state, history, sorted_history))
            })
            .collect();

        Ok(column_results?.concat())
    }
}
