use crate::numpy_io;
use crate::types::Compute;
use crate::types::Feature;
use numpy::PyArray2;
use pyo3::prelude::*;
use rayon::prelude::*;
use realfft::RealFftPlanner;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub mod engine;

pub mod processors;

use crate::common::{map_features_to_indices, next_good_fft_size};
use engine::StaticEngine;

#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub struct Extractor {
    pub features: Vec<Feature>,
    pub compute: Compute,
    pub paa_args: Vec<(u16, u16)>,
    pub c3_args: Vec<u16>,
    pub unique_paa_totals: Vec<u16>,
    pub unique_c3_lags: Vec<u16>,
    pub unique_tra_lags: Vec<u16>,
    pub unique_count_above_thresholds: Vec<u32>,
    pub unique_count_below_thresholds: Vec<u32>,
    pub unique_range_counts: Vec<(u32, u32)>,
    pub planner: Arc<Mutex<RealFftPlanner<f32>>>,
    pub max_size: Option<usize>,
    pub fs: f32,
}

#[pymethods]
impl Extractor {
    #[new]
    #[pyo3(signature = (feature_str, max_size=None, fs=100.0))]
    pub fn new(feature_str: Vec<String>, max_size: Option<usize>, fs: f32) -> PyResult<Self> {
        if !(fs.is_finite() && fs > 0.0) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "fs must be a positive, finite sampling frequency, got {fs}"
            )));
        }
        let mut features = Vec::new();
        let mut paa_args = Vec::new();
        let mut c3_args = Vec::new();
        let mut unique_paa_totals = std::collections::BTreeSet::new();
        let mut unique_c3_lags = std::collections::BTreeSet::new();
        let mut unique_tra_lags = std::collections::BTreeSet::new();
        let mut unique_count_above_thresholds = std::collections::BTreeSet::new();
        let mut unique_count_below_thresholds = std::collections::BTreeSet::new();
        let mut unique_range_counts = std::collections::BTreeSet::new();

        for i in feature_str {
            let feat = std::str::FromStr::from_str(&i)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
            if let Feature::Paa(total, index) = feat {
                paa_args.push((total, index));
                unique_paa_totals.insert(total);
            } else if let Feature::C3(lag) = feat {
                c3_args.push(lag);
                unique_c3_lags.insert(lag);
            } else if let Feature::CountAbove(t) = feat {
                unique_count_above_thresholds.insert(t);
            } else if let Feature::CountBelow(t) = feat {
                unique_count_below_thresholds.insert(t);
            } else if let Feature::RangeCount(min, max) = feat {
                unique_range_counts.insert((min, max));
            }
            features.push(feat);
        }
        let compute = map_features_to_indices(&features);

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
            paa_args,
            c3_args,
            unique_paa_totals: unique_paa_totals.into_iter().collect(),
            unique_c3_lags: unique_c3_lags.into_iter().collect(),
            unique_tra_lags: unique_tra_lags.into_iter().collect(),
            unique_count_above_thresholds: unique_count_above_thresholds.into_iter().collect(),
            unique_count_below_thresholds: unique_count_below_thresholds.into_iter().collect(),
            unique_range_counts: unique_range_counts.into_iter().collect(),
            planner: planner_arc,
            max_size,
            fs,
        })
    }

    /// Canonical names of the output columns, in order.
    #[getter]
    pub fn feature_names(&self) -> Vec<String> {
        self.features.iter().map(Feature::name).collect()
    }

    /// Features of every row of `values`, a 2-D float32/float64 numpy array of
    /// shape (n_series, n_samples): one series per row. Returns a float32 array
    /// of shape (n_series, n_features), columns in `feature_names` order. An
    /// array with no samples gives zero rows.
    pub fn process_2d_floats<'py>(
        &self,
        py: Python<'py>,
        values: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let (n_series, data) = numpy_io::with_rows(values, |rows, n_samples| {
            let n_series = if n_samples == 0 { 0 } else { rows.len() };
            // Python can still write to the array meanwhile; like any extension
            // releasing the GIL, that only garbles this call's features.
            Ok((
                n_series,
                py.detach(|| self.extract(&rows[..n_series], n_samples)),
            ))
        })?;
        numpy_io::to_numpy_2d(py, n_series, self.features.len(), data)
    }
}

/// Estimated work below which waking rayon's workers (about 5 µs per fan-out)
/// costs more than splitting the columns over them saves.
const PARALLEL_MIN_WORK: Duration = Duration::from_micros(25);

impl Extractor {
    /// Features of each column, all `n_rows` long, flattened row-major: one row
    /// of `features.len()` values per column.
    pub fn extract(&self, columns: &[&[f32]], n_rows: usize) -> Vec<f32> {
        let compute = self.compute;
        let paa_boundaries: Vec<Vec<usize>> = self
            .unique_paa_totals
            .iter()
            .map(|&total| {
                (0..=total)
                    .map(|i| (i as f32 * n_rows as f32 / total as f32).round() as usize)
                    .collect()
            })
            .collect();

        // Exact-length FFT: padding would change the spectrum (see spectral::rfft).
        let fft_size = n_rows;

        let r2c = if compute.intersects(Compute::ANY_FFT) && fft_size > 0 {
            let mut p = self.planner.lock().unwrap_or_else(|e| e.into_inner());
            Some(p.plan_fft_forward(fft_size))
        } else {
            None
        };

        let processor = StaticEngine {
            compute,
            features: &self.features,
            unique_paa_totals: &self.unique_paa_totals,
            unique_c3_lags: &self.unique_c3_lags,
            unique_tra_lags: &self.unique_tra_lags,
            unique_count_above_thresholds: &self.unique_count_above_thresholds,
            unique_count_below_thresholds: &self.unique_count_below_thresholds,
            unique_range_counts: &self.unique_range_counts,
            paa_boundaries: &paa_boundaries,
            r2c,
            fs: self.fs,
        };

        // Time the first column to decide whether the rest are worth spreading
        // over threads: cheap features on a few short series finish before the
        // workers would even wake up.
        let mut column_results = Vec::with_capacity(columns.len());
        if let Some((first, rest)) = columns.split_first() {
            let start = Instant::now();
            column_results.push(processor.process_column(first));
            let estimate = start.elapsed() * rest.len() as u32;
            if estimate < PARALLEL_MIN_WORK {
                column_results.extend(rest.iter().map(|col| processor.process_column(col)));
            } else {
                column_results.par_extend(rest.par_iter().map(|col| processor.process_column(col)));
            }
        }
        column_results.concat()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let extractor = Extractor::new(features, None, crate::spectral::DEFAULT_FS).unwrap();
        let result = extractor.extract(&[&data], data.len());

        assert_eq!(result.len(), 34);

        let names = extractor.feature_names();
        for (i, &val) in result.iter().enumerate() {
            println!("{}: {}", names[i], val);
            assert!(!val.is_nan(), "Feature {} is NaN", names[i]);
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
        let extractor = Extractor::new(features, None, crate::spectral::DEFAULT_FS).unwrap();
        let result = extractor.extract(&[&data], data.len());

        let n = data.len() as f32;
        let scalar_sum: f32 = data.iter().sum();
        let scalar_mean = scalar_sum / n;
        let scalar_var = data.iter().map(|&x| (x - scalar_mean).powi(2)).sum::<f32>() / n; // ddof=0
        let scalar_energy = data.iter().map(|&x| x * x).sum::<f32>();
        let scalar_mad = data.iter().map(|&x| (x - scalar_mean).abs()).sum::<f32>() / n;

        let col_mean = result[0];
        let col_var = result[1];
        let col_sum = result[2];
        let col_energy = result[6];
        let col_mad = result[8];

        assert!((col_mean - scalar_mean).abs() < 1e-5);
        assert!((col_var - scalar_var).abs() < 1e-5);
        assert!((col_sum - scalar_sum).abs() < 1e-5);
        assert!((col_energy - scalar_energy).abs() < 1e-5);
        assert!((col_mad - scalar_mad).abs() < 1e-5);
    }
}
