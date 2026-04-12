#![allow(dead_code)]
#![feature(portable_simd)]
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

mod downsampling;
mod features;
mod registry;
mod sliding;

mod extract;

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

#[pymodule]
fn _tsfast(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<registry::FeatureExtractor>()?;
    m.add_function(wrap_pyfunction!(downsample, m)?)?;
    Ok(())
}
