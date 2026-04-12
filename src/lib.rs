#![allow(dead_code)]
#![feature(portable_simd)]
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;

mod downsampling;
mod features;
mod registry;
mod sliding;

#[pyfunction]
fn downsample(
    _py: Python<'_>,
    data: PyReadonlyArray1<'_, f64>,
    n_bins: usize,
) -> PyResult<Py<PyArray1<f64>>> {
    let data_slice = data.as_slice()?;
    let result = downsampling::m4_downsample(data_slice, n_bins);
    Ok(result.into_pyarray(_py).unbind())
}

#[pyfunction]
fn extract(
    _py: Python<'_>,
    data: PyReadonlyArray1<'_, f64>,
    features: Vec<String>,
) -> PyResult<Py<PyArray1<f64>>> {
    let extractor = registry::FeatureExtractor::new(features);
    extractor.extract1d(data, _py)
}

#[pyfunction]
fn extract2d(
    _py: Python<'_>,
    data: Bound<'_, PyAny>,
    features: Vec<String>,
) -> PyResult<Py<PyArray2<f64>>> {
    let extractor = registry::FeatureExtractor::new(features);
    extractor.extract2d(data)
}

#[pymodule]
fn _tsfast(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<registry::FeatureExtractor>()?;
    m.add_function(wrap_pyfunction!(downsample, m)?)?;
    m.add_function(wrap_pyfunction!(extract, m)?)?;
    m.add_function(wrap_pyfunction!(extract2d, m)?)?;
    Ok(())
}
