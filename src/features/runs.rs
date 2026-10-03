use crate::types::Feature;

#[inline(always)]
pub fn eval_runs(feat: &Feature, context: &mut crate::context::FeatureContext) -> Option<f32> {
    let count_a = context.count_a;
    let count_b = context.count_b;
    let max_strike_a = context.max_strike_a;
    let max_strike_b = context.max_strike_b;

    let res = match feat {
        Feature::CountAboveMean => count_a as f32,
        Feature::CountBelowMean => count_b as f32,
        Feature::LongestStrikeAboveMean => max_strike_a as f32,
        Feature::LongestStrikeBelowMean => max_strike_b as f32,
        _ => return None,
    };
    Some(res)
}
