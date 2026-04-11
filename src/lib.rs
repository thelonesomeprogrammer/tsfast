use numpy::IntoPyArray;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;

mod features;
mod registry;
mod sliding;

#[pyfunction]
fn extract(
    _py: Python<'_>,
    data: PyReadonlyArray1<'_, f64>,
    features: Vec<String>,
) -> PyResult<Py<PyArray1<f64>>> {
    let extractor = registry::FeatureExtractor::new(features);
    extractor.extract1d(data, _py)
}

#[pymodule]
fn _tsfast(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<registry::FeatureExtractor>()?;
    m.add_function(wrap_pyfunction!(extract, m)?)?;
    Ok(())
}
