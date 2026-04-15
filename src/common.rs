use smallvec::SmallVec;
use std::simd::f32x4;

pub const LANES: usize = 4;
pub struct ColumnState {
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
    pub zc_indices: SmallVec<[f32; 32]>,
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
            zc_indices: SmallVec::new(),
            paa_sums: unique_paa_totals
                .iter()
                .map(|&total| vec![0.0; total as usize])
                .collect(),
            current_paa_segs: vec![0usize; unique_paa_totals.len()],
            c3_sums: vec![0.0; unique_c3_lags.len()],
            prev_last: first_val,
        }
    }
}

use crate::types::{FastBitArray, Feature};

pub(crate) fn map_features_to_indices(features: &[Feature]) -> FastBitArray {
    let mut bits = FastBitArray::ZERO;
    for feat in features {
        match feat {
            Feature::TotalSum => bits.set_batch([0]),
            Feature::Mean => bits.set_batch([0, 1]),
            Feature::Variance => bits.set_batch([0, 1, 2, 12, 37]),
            Feature::Std => bits.set_batch([0, 1, 2, 3, 12, 37]),
            Feature::Min => bits.set_batch([4]),
            Feature::Max => bits.set_batch([5]),
            Feature::Median => bits.set_batch([6, 37]),
            Feature::Skew => bits.set_batch([0, 1, 2, 7, 12, 37]),
            Feature::Kurtosis => bits.set_batch([0, 1, 2, 8, 12, 37]),
            Feature::Mad => bits.set_batch([0, 1, 9, 37]),
            Feature::Iqr => bits.set_batch([4, 5, 6, 10, 37]),
            Feature::Entropy => bits.set_batch([4, 5, 6, 10, 11, 37]),
            Feature::Energy => bits.set_batch([12]),
            Feature::Rms => bits.set_batch([12, 13]),
            Feature::RootMeanSquare => bits.set_batch([12, 14]),
            Feature::ZeroCrossingRate => bits.set_batch([15]),
            Feature::PeakCount => bits.set_batch([16]),
            Feature::AutocorrLag1 => bits.set_batch([0, 1, 12, 17]),
            Feature::MeanAbsChange => bits.set_batch([0, 1, 18]),
            Feature::MeanChange => bits.set_batch([0, 1, 19]),
            Feature::CidCe => bits.set_batch([0, 1, 20]),
            Feature::Slope => bits.set_batch([0, 1, 21]),
            Feature::Intercept => bits.set_batch([0, 1, 21, 22]),
            Feature::Paa(_, _) => bits.set_batch([23]),
            Feature::AbsSumChange => bits.set_batch([24]),
            Feature::CountAboveMean => bits.set_batch([0, 1, 25, 37]),
            Feature::CountBelowMean => bits.set_batch([0, 1, 26, 37]),
            Feature::LongestStrikeAboveMean => bits.set_batch([0, 1, 27, 37]),
            Feature::LongestStrikeBelowMean => bits.set_batch([0, 1, 28, 37]),
            Feature::VariationCoefficient => bits.set_batch([0, 1, 2, 3, 12, 29, 37]),
            Feature::C3(_) => bits.set_batch([30]),
            Feature::Auc => bits.set_batch([31]),
            Feature::SlopeSignChange => bits.set_batch([16, 32]),
            Feature::TurningPoints => bits.set_batch([16, 33]),
            Feature::ZeroCrossingMean => bits.set_batch([0, 1, 15, 34, 36, 37]),
            Feature::ZeroCrossingStd => bits.set_batch([0, 1, 15, 34, 35, 36, 37]),
        }
    }
    bits
}
