#![feature(portable_simd)]
use pyo3::prelude::*;

mod common;
mod expanding;
mod sliding;
mod static_ext;
mod types;

#[pymodule]
fn _tsfast(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<static_ext::Extractor>()?;
    Ok(())
}
