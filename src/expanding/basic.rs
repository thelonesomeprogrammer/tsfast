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
        Feature::Length => Some(n as f32),
        Feature::Energy => Some(state.energy),
        Feature::Rms | Feature::RootMeanSquare => Some((state.energy / n).sqrt()),
        Feature::AbsMax => Some(state.abs_max),
        Feature::FirstLocMax => Some(state.first_max_idx as f32 / n),
        Feature::LastLocMax => Some((state.last_max_idx + 1) as f32 / n),
        Feature::FirstLocMin => Some(state.first_min_idx as f32 / n),
        Feature::LastLocMin => Some((state.last_min_idx + 1) as f32 / n),
        _ => None,
    }
}
