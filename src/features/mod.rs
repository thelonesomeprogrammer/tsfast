pub mod autocorrelation;
pub mod changes;
pub mod complexity;
pub mod crossings_peaks;
pub mod cwt;
pub mod distribution;
pub mod dynamic;
pub mod energy;
pub mod lpc;
pub mod min_max;
pub mod misc;
pub mod moments;
pub mod runs;
pub mod stationarity;
pub mod subsequence;
pub mod transform;

use crate::context::FeatureContext;
use crate::types::Feature;

/// Evaluates one feature. Every engine calls this.
///
/// The match is exhaustive on purpose: adding a `Feature` variant without
/// routing it to an `eval_*` function here is a compile error.
pub fn eval(feat: &Feature, ctx: &mut FeatureContext) -> f32 {
    use Feature as F;
    let res = match feat {
        F::HurstExponent => complexity::eval_complexity(feat, ctx),
        F::MaximumFractalLength => complexity::eval_complexity(feat, ctx),
        F::BiasedFisherKurtosis
        | F::Iqr
        | F::Mad
        | F::Mean
        | F::Skew
        | F::BiasedSkew
        | F::Std
        | F::TotalSum
        | F::UnbiasedFisherKurtosis
        | F::Variance
        | F::VariationCoefficient
        | F::LargeStandardDeviation(..) => moments::eval_moments(feat, ctx),
        F::AbsMax
        | F::FirstLocMax
        | F::FirstLocMin
        | F::HasDuplicate
        | F::HasDuplicateMax
        | F::HasDuplicateMin
        | F::LastLocMax
        | F::LastLocMin
        | F::Max
        | F::MeanNAbsoluteMax(..)
        | F::Min
        | F::PkPkDistance => min_max::eval_min_max(feat, ctx),
        F::BenfordCorrelation
        | F::BinnedEntropy(..)
        | F::ChangeQuantiles(..)
        | F::Ecdf(..)
        | F::EcdfPercentile(..)
        | F::EcdfPercentileCount(..)
        | F::EcdfSlope(..)
        | F::Entropy
        | F::IndexMassQuantile(..)
        | F::Median
        | F::MedianAbsDeviation
        | F::Quantile(..)
        | F::RatioBeyondRSigma(..)
        | F::SumOfReoccurringDataPoints
        | F::SumOfReoccurringValues
        | F::ValueCount(..)
        | F::SymmetryLooking(..) => distribution::eval_distribution(feat, ctx),
        F::Energy
        | F::EnergyRatioByChunks(..)
        | F::HumanRangeEnergy(..)
        | F::Rms
        | F::RootMeanSquare => energy::eval_energy(feat, ctx),
        F::NegativeTurning
        | F::NumberCrossingM(..)
        | F::NumberPeaks(..)
        | F::PeakCount
        | F::PositiveTurning
        | F::ZeroCross
        | F::ZeroCrossingMean
        | F::ZeroCrossingRate
        | F::ZeroCrossingStd
        | F::SlopeSignChange
        | F::TurningPoints => crossings_peaks::eval_crossings_peaks(feat, ctx),
        F::AggAutocorrelation(..)
        | F::Autocorr(..)
        | F::AutocorrFirst1e
        | F::AutocorrLag1
        | F::PartialAutocorr(..)
        | F::TimeReversalAsymmetry(..) => autocorrelation::eval_autocorrelation(feat, ctx),
        F::AbsSumChange
        | F::AggLinearTrend(..)
        | F::Auc
        | F::CidCe
        | F::Intercept
        | F::LinearTrend(..)
        | F::MeanAbsChange
        | F::MeanChange
        | F::MedianAbsDiff
        | F::MedianDiff
        | F::SignalDistance
        | F::Slope
        | F::MeanSecondDerivativeCentral => changes::eval_changes(feat, ctx),
        F::CountAboveMean
        | F::CountBelowMean
        | F::CountAbove(..)
        | F::CountBelow(..)
        | F::RangeCount(..)
        | F::LongestStrikeAboveMean
        | F::LongestStrikeBelowMean => runs::eval_runs(feat, ctx),
        F::MatrixProfile(..) | F::QuerySimilarityCount(..) => {
            subsequence::eval_subsequence(feat, ctx)
        }
        F::C3(..)
        | F::CalcCentroid(..)
        | F::CwtCoefficients(..)
        | F::FftCoefficient(..)
        | F::Lpcc(..)
        | F::MaxPowerSpectrum
        | F::Mfcc(..)
        | F::NumberCwtPeaks(..)
        | F::Paa(..)
        | F::SpectralCentroid
        | F::SpectralDecrease
        | F::SpectralDistance
        | F::SpectralEntropy
        | F::SpectralKurtosis
        | F::SpectralRollOff
        | F::SpectralRollOn
        | F::SpectralSkewness
        | F::SpectralSlope
        | F::SpectralSpread
        | F::SpectrogramCoefficients(..)
        | F::SpktWelchDensity(..)
        | F::WaveletEnergy(..)
        | F::WaveletEntropy
        | F::MaxFrequency
        | F::MedianFrequency
        | F::FundamentalFrequency
        | F::WaveletFeatures(..) => transform::eval_transform(feat, ctx),
        F::ApproxEntropy(..)
        | F::HiguchiFd
        | F::LempelZiv
        | F::LempelZivComplexity(..)
        | F::PermutationEntropy(..)
        | F::SampleEntropy
        | F::Dfa
        | F::HurstExponent
        | F::MaximumFractalLength
        | F::Mse(..) => complexity::eval_complexity(feat, ctx),
        F::ArCoefficient(..) | F::FriedrichCoefficients(..) | F::MaxLangevinFixedPoint(..) => {
            dynamic::eval_dynamic(feat, ctx)
        }
        F::AugmentedDickeyFuller(..) => stationarity::eval_stationarity(feat, ctx),
        F::Length
        | F::PercentageOfReoccurringDatapointsToAllDatapoints
        | F::PercentageOfReoccurringValuesToAllValues
        | F::RatioValueNumberToTimeSeriesLength
        | F::VarianceLargerThanStandardDeviation => misc::eval_misc(feat, ctx),
        F::Dfa => complexity::eval_complexity(feat, ctx),
    };
    // `None` means "not computable for this window" (e.g. zero variance, window
    // shorter than a lag); engines have always reported that as 0.0.
    res.unwrap_or(0.0)
}
