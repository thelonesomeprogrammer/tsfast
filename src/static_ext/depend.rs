use crate::types::{FastBitArray, Feature};

pub(crate) fn map_features_to_indices(features: &[Feature]) -> FastBitArray {
    let mut bits = FastBitArray::ZERO;
    for feat in features {
        match feat {
            Feature::TotalSum => bits.set_batch([0]),
            Feature::Mean => bits.set_batch([0, 1]),
            Feature::Variance => bits.set_batch([0, 12]),
            Feature::Std => bits.set_batch([0, 12, 3]),
            Feature::Min => bits.set_batch([4]),
            Feature::Max => bits.set_batch([5]),
            Feature::Median => bits.set_batch([6, 37]),
            Feature::Skew => bits.set_batch([0, 12, 7]),
            Feature::Kurtosis => bits.set_batch([0, 12, 8]),
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
            Feature::VariationCoefficient => bits.set_batch([0, 12, 3, 29, 37]),
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
