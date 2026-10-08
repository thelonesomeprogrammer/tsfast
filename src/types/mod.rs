mod compat;
mod compute;
mod feature;
mod parse;
#[cfg(test)]
mod tests;

// Flat re-exports: every existing `use crate::types::{…}` compiles unchanged.
pub use compute::Compute;
pub use parse::{MetaFeatures, split_meta};
pub use feature::{AdfAttr, AggAttr, AggFunc, Feature, FftAggType, FftAttr, compute_flags};
