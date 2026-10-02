#[inline(always)]
pub fn compute_extrema_features(
    compute: &crate::types::Compute,
    values: &[f32],
    max_value: f32,
    min_value: f32,
    first_max_idx: &mut usize,
    last_max_idx: &mut usize,
    first_min_idx: &mut usize,
    last_min_idx: &mut usize,
) {
    if compute.intersects(crate::types::Compute::FIRST_LOC_MAX | crate::types::Compute::LAST_LOC_MAX | crate::types::Compute::FIRST_LOC_MIN | crate::types::Compute::LAST_LOC_MIN) {
        let mut found_max = false;
        let mut found_min = false;
        for (i, &v) in values.iter().enumerate() {
            if v == max_value {
                if !found_max {
                    *first_max_idx = i;
                    found_max = true;
                }
                *last_max_idx = i;
            }
            if v == min_value {
                if !found_min {
                    *first_min_idx = i;
                    found_min = true;
                }
                *last_min_idx = i;
            }
        }
    }
}
