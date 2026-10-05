#![feature(portable_simd)]
use pyo3::prelude::*;

pub mod common;
pub mod expanding;
mod numpy_io;
pub mod sliding;
pub mod static_ext;
pub mod types;

pub use expanding::ExpandingExtractor;
pub use sliding::SlidingExtractor;
pub use static_ext::Extractor;

#[pymodule]
fn _tsfast(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<static_ext::Extractor>()?;
    m.add_class::<expanding::ExpandingExtractor>()?;
    m.add_class::<sliding::SlidingExtractor>()?;
    Ok(())
}
pub mod metrics;
pub mod context;
pub mod features;
pub mod spectral;
