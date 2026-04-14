use std::simd::f32x4;

pub(crate) struct ColumnState {
    pub total_sum: f32,
    pub min_value: f32,
    pub max_value: f32,
    pub energy: f32,
    pub sum_cubes: f32,
    pub sum_quads: f32,
    pub mac_sum_vec: f32x4,
    pub mc_sum_vec: f32x4,
    pub zcr_count: u32,
    pub sum_prod: f32,
    pub sum_sq_diff: f32,
    pub sum_ix: f32,
    pub auc_sum: f32,
    pub peaks: u32,
    pub zc_indices: Vec<f32>,
    pub paa_sums: Vec<Vec<f32>>,
    pub current_paa_segs: Vec<usize>,
    pub c3_sums: Vec<f64>,
    pub prev_last: f32,
}

impl ColumnState {
    pub fn new(unique_paa_totals: &[u16], unique_c3_lags: &[u16], first_val: f32) -> Self {
        Self {
            total_sum: 0.0,
            min_value: f32::INFINITY,
            max_value: f32::NEG_INFINITY,
            energy: 0.0,
            sum_cubes: 0.0,
            sum_quads: 0.0,
            mac_sum_vec: f32x4::splat(0.0),
            mc_sum_vec: f32x4::splat(0.0),
            zcr_count: 0,
            sum_prod: 0.0,
            sum_sq_diff: 0.0,
            sum_ix: 0.0,
            auc_sum: 0.0,
            peaks: 0,
            zc_indices: Vec::new(),
            paa_sums: unique_paa_totals.iter().map(|&total| vec![0.0; total as usize]).collect(),
            current_paa_segs: vec![0usize; unique_paa_totals.len()],
            c3_sums: vec![0.0; unique_c3_lags.len()],
            prev_last: first_val,
        }
    }
}
