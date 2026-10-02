use crate::common::ColumnState;
use crate::types::Feature;

#[inline(always)]
pub fn eval_basic(
    feat: &Feature,
    n: f32,
    mean: f32,
    var: f32,
    std_dev: f32,
    state: &ColumnState,
) -> Option<f32> {
    match feat {
        Feature::TotalSum => Some(state.total_sum),
        Feature::Mean => Some(mean),
        Feature::Variance => Some(var),
        Feature::Std => Some(std_dev),
        Feature::Min => Some(state.min_value),
        Feature::Max => Some(state.max_value),
        Feature::AbsMax => Some(state.abs_max),
        Feature::Length => Some(n),
        _ => None,
    }
}
