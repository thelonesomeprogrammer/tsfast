//! The Python boundary shared by all three extractors: series come in as the
//! rows of a 2-D numpy array, features go out as a float32 numpy array.

use numpy::ndarray::{Array2, Array3};
use numpy::{
    IntoPyArray, PyArray2, PyArray3, PyReadonlyArrayDyn, PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

/// Calls `f` with the rows of `values`, a 2-D float32 or float64 numpy array of
/// shape (n_series, n_samples), as one slice per series, and with n_samples.
///
/// A C-contiguous float32 array is read in place; any other layout, and
/// float64, is copied to float32 first. Other dtypes are rejected rather than
/// converted, so integer or object data never silently turns into features.
pub(crate) fn with_rows<R>(
    values: &Bound<'_, PyAny>,
    f: impl FnOnce(&[&[f32]], usize) -> PyResult<R>,
) -> PyResult<R> {
    let owned: Vec<f32>;
    let borrowed: PyReadonlyArrayDyn<'_, f32>;
    let (data, shape): (&[f32], Vec<usize>) =
        if let Ok(array) = values.extract::<PyReadonlyArrayDyn<'_, f32>>() {
            borrowed = array;
            let shape = borrowed.shape().to_vec();
            match borrowed.as_slice() {
                Ok(slice) if borrowed.is_c_contiguous() => (slice, shape),
                _ => {
                    owned = borrowed.as_array().iter().copied().collect();
                    (owned.as_slice(), shape)
                }
            }
        } else if let Ok(array) = values.extract::<PyReadonlyArrayDyn<'_, f64>>() {
            owned = array.as_array().iter().map(|&v| v as f32).collect();
            (owned.as_slice(), array.shape().to_vec())
        } else {
            return Err(PyTypeError::new_err(format!(
                "expected a float32 or float64 numpy array, got {}",
                describe(values)
            )));
        };
    let &[n_series, n_samples] = shape.as_slice() else {
        return Err(PyValueError::new_err(format!(
            "expected a 2-D array of shape (n_series, n_samples), got {} dimension(s)",
            shape.len()
        )));
    };
    let rows: Vec<&[f32]> = if n_samples == 0 {
        vec![&[]; n_series]
    } else {
        data.chunks_exact(n_samples).collect()
    };
    f(&rows, n_samples)
}

fn describe(values: &Bound<'_, PyAny>) -> String {
    let dtype = values
        .getattr("dtype")
        .and_then(|d| d.str())
        .map(|d| format!(" with dtype {d}"))
        .unwrap_or_default();
    let type_name = values
        .get_type()
        .name()
        .map(|n| n.to_string())
        .unwrap_or_else(|_| "?".into());
    format!("{type_name}{dtype}")
}

/// Row-major `data` as a (rows, cols) numpy array, without copying.
pub(crate) fn to_numpy_2d(
    py: Python<'_>,
    rows: usize,
    cols: usize,
    data: Vec<f32>,
) -> PyResult<Bound<'_, PyArray2<f32>>> {
    Array2::from_shape_vec((rows, cols), data)
        .map(|a| a.into_pyarray(py))
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Row-major `data` as a numpy array of the given 3-D shape, without copying.
pub(crate) fn to_numpy_3d(
    py: Python<'_>,
    shape: (usize, usize, usize),
    data: Vec<f32>,
) -> PyResult<Bound<'_, PyArray3<f32>>> {
    Array3::from_shape_vec(shape, data)
        .map(|a| a.into_pyarray(py))
        .map_err(|e| PyValueError::new_err(e.to_string()))
}
