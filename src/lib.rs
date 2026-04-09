use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1};
use numpy::IntoPyArray;

mod features;
mod registry;
mod extract_inner;
mod expanding_inner;
mod sliding;

#[pyfunction]
#[pyo3(name = "extract")]
fn extract_py(
    _py: Python<'_>,
    data: PyReadonlyArray1<'_, f64>,
    features: Vec<String>,
) -> PyResult<Py<PyArray1<f64>>> {
    let registry = registry::Registry::new();
    let data_slice = data.as_slice()?;
    let features_refs: Vec<&str> = features.iter().map(|s| s.as_str()).collect();
    let result = extract_inner::extract(data_slice, &features_refs, &registry);
    Ok(result.into_pyarray(_py).unbind())
}

#[pyfunction]
#[pyo3(name = "extract_expanding")]
fn extract_expanding_py(
    _py: Python<'_>,
    data: PyReadonlyArray1<'_, f64>,
    features: Vec<String>,
) -> PyResult<Py<PyArray2<f64>>> {
    let registry = registry::Registry::new();
    let data_slice = data.as_slice()?;
    let n_samples = data_slice.len();
    let n_features = features.len();
    
    let features_refs: Vec<&str> = features.iter().map(|s| s.as_str()).collect();
    let result = expanding_inner::extract_expanding(data_slice, &features_refs, &registry);
    
    let array = ndarray::Array2::from_shape_vec((n_features, n_samples), result)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
    
    // Transpose to (n_samples, n_features)
    let transposed = array.reversed_axes();
    
    Ok(transposed.into_pyarray(_py).unbind())
}

#[pymodule]
fn _tsfast(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(extract_py, m)?)?;
    m.add_function(wrap_pyfunction!(extract_expanding_py, m)?)?;
    Ok(())
}
