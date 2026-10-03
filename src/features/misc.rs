use crate::types::Feature;

#[inline(always)]
pub fn eval_misc(feat: &Feature, context: &mut crate::context::FeatureContext) -> Option<f32> {
    let n = context.n;
    let var = context.var;
    let state = &mut context.state;

    let res = match feat {
        Feature::PercentageOfReoccurringDatapointsToAllDatapoints => {
            if n == 0.0 {
                f32::NAN
            } else {
                state.reoccurring_datapoints as f32 / n
            }
        }
        Feature::PercentageOfReoccurringValuesToAllValues => {
            if n == 0.0 {
                f32::NAN
            } else {
                if state.value_counts.is_empty() {
                    0.0
                } else {
                    state.reoccurring_values as f32 / state.value_counts.len() as f32
                }
            }
        }
        Feature::RatioValueNumberToTimeSeriesLength => {
            if n == 0.0 {
                f32::NAN
            } else {
                state.value_counts.len() as f32 / n
            }
        }
        Feature::Length => n,
        Feature::VarianceLargerThanStandardDeviation => {
            if var > 1.0 {
                1.0
            } else {
                0.0
            }
        }
        _ => return None,
    };
    Some(res)
}
