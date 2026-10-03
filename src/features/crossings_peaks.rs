use crate::types::Feature;

#[inline(always)]
pub fn eval_crossings_peaks(
    feat: &Feature,
    context: &mut crate::context::FeatureContext,
) -> Option<f32> {
    let n = context.n;
    let state = &mut *context.state;
    let zc_mean = context.zc_mean;
    let zc_std = context.zc_std;

    let res = match feat {
        Feature::ZeroCrossingRate => state.zcr_count as f32 / n,
        Feature::PeakCount => state.peaks as f32,
        Feature::ZeroCrossingMean => zc_mean,
        Feature::ZeroCrossingStd => zc_std,
        _ => return None,
    };
    Some(res)
}
