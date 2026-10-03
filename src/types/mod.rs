mod compute;
mod feature;
mod parse;

// Flat re-exports: every existing `use crate::types::{…}` compiles unchanged.
pub use compute::Compute;
pub use feature::{AggAttr, AggFunc, Feature, FftAttr, compute_flags};
