use crate::types::Feature;

#[inline(always)]
pub fn eval_complexity(
    feat: &Feature,
    context: &mut crate::context::FeatureContext,
) -> Option<f32> {
    let values = context.values;
    let state = &mut context.state;

    let res = match feat {
        Feature::ApproxEntropy(m, r_bits) if values.len() > *m as usize + 1 => {
            let m_val = *m as usize;
            let r = f32::from_bits(*r_bits);
            let mut buffer = std::mem::take(&mut state.approx_entropy_buffer);

            let res = crate::common::approx_entropy_phi(m_val, r, values, &mut buffer)
                - crate::common::approx_entropy_phi(m_val + 1, r, values, &mut buffer);
            state.approx_entropy_buffer = buffer;
            res
        }
        _ => return None,
    };
    Some(res)
}
