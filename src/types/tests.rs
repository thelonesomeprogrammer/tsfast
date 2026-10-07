use std::collections::HashSet;

use strum::IntoEnumIterator;

use super::feature::FeatureDiscriminants;
use super::parse::UNIT_FEATURES;
use super::{AdfAttr, AggAttr, AggFunc, Feature, FftAggType, FftAttr};

fn f(x: f32) -> u32 {
    x.to_bits()
}

/// One valid instance of every parameterized variant. Unit variants come from
/// the `unit_features!` table. Adding a `Feature` variant without a sample here
/// fails `every_variant_has_a_sample`.
fn parameterized_samples() -> Vec<Feature> {
    use Feature as F;
    vec![
        F::LempelZivComplexity(3),
        F::Mse(3, 0),
        F::Mse(2, 10),
        F::BinnedEntropy(5),
        F::EnergyRatioByChunks(3, 1),
        F::NumberCrossingM(f(0.5)),
        F::NumberPeaks(3),
        F::Paa(4, 1),
        F::C3(2),
        F::Autocorr(3),
        F::AggAutocorrelation(AggFunc::Mean, 10),
        F::PartialAutocorr(2),
        F::TimeReversalAsymmetry(2),
        F::FftCoefficient(3, FftAttr::Abs),
        F::ApproxEntropy(2, f(0.1)),
        F::LinearTrend(AggAttr::Slope),
        F::AggLinearTrend(AggAttr::Intercept, 5, AggFunc::Max),
        F::Quantile(f(0.25)),
        F::ChangeQuantiles(f(0.2), f(0.8), true, AggFunc::Mean),
        F::IndexMassQuantile(f(0.5)),
        F::EcdfPercentile(f(0.5)),
        F::EcdfPercentileCount(f(0.5)),
        F::HistMode(10),
        F::EcdfSlope(f(0.2), f(0.5)),
        F::MaxLangevinFixedPoint(3, f(30.0)),
        F::ArCoefficient(10, 1),
        F::FriedrichCoefficients(3, f(30.0), 0),
        F::MeanNAbsoluteMax(7),
        F::QuerySimilarityCount(10, f(0.5)),
        F::MatrixProfile(10, AggFunc::Min),
        F::HumanRangeEnergy(f(100.0)),
        F::AveragePower(f(100.0)),
        F::WaveletFeatures(f(0.5), 1),
        F::SpectrogramCoefficients(2, f(0.5)),
        F::LargeStandardDeviation(f(0.05)),
        F::SymmetryLooking(f(0.05)),
        F::RatioBeyondRSigma(f(1.5)),
        F::Ecdf(10),
        F::PermutationEntropy(1, 3),
        F::ValueCount(f(1.0)),
        F::CalcCentroid(f(100.0)),
        F::Mfcc(3),
        F::Lpcc(3),
        F::WaveletEnergy(3),
        F::WaveletAbsMean(3),
        F::WaveletStd(3),
        F::WaveletVar(3),
        F::SpktWelchDensity(2),
        F::CwtCoefficients([2, 5, 10, 20, 0, 0, 0, 0], 4, 3, 5),
        F::NumberCwtPeaks(5),
        F::AugmentedDickeyFuller(AdfAttr::PValue),
        F::CountAbove(f(0.5)),
        F::CountBelow(f(0.5)),
        F::RangeCount(f(-1.0), f(1.0)),
        F::FourierEntropy(5),
        F::FftAggregated(FftAggType::Centroid),
        F::FftAggregated(FftAggType::Variance),
        F::FftAggregated(FftAggType::Skew),
        F::FftAggregated(FftAggType::Kurtosis),
    ]
}

fn samples() -> Vec<Feature> {
    UNIT_FEATURES
        .iter()
        .copied()
        .chain(parameterized_samples())
        .collect()
}

#[test]
fn every_variant_has_a_sample() {
    let covered: HashSet<FeatureDiscriminants> = samples()
        .into_iter()
        .map(FeatureDiscriminants::from)
        .collect();
    let missing: Vec<_> = FeatureDiscriminants::iter()
        .filter(|d| !covered.contains(d))
        .collect();
    assert!(
        missing.is_empty(),
        "add these to unit_features! (parse.rs) or parameterized_samples() (types/tests.rs): {missing:?}"
    );
}

#[test]
fn name_round_trips_through_parse() {
    let failures: Vec<String> = samples()
        .into_iter()
        .filter_map(|feat| {
            let name = feat.name();
            match name.parse::<Feature>() {
                Ok(parsed) if parsed == feat => None,
                other => Some(format!("{feat:?} -> {name:?} -> {other:?}")),
            }
        })
        .collect();
    assert!(
        failures.is_empty(),
        "name() doesn't parse back:\n{}",
        failures.join("\n")
    );
}

#[test]
fn invalid_parameters_are_rejected() {
    for s in [
        "paa-2-2",
        "mse-0",
        "hist_mode-0",
        "neighbourhood_peaks-0",
        "mse--1",
        "ar_coefficient-1-10",
        "ar_coefficient-0-0",
        "agg_linear_trend-slope-0-mean",
        "energy_ratio_by_chunks_num_segments_3__segment_focus_3",
        "human_range_energy-0",
        "average_power-0",
        "average_power--1",
        "ecdf_percentile-0.0",
        "ecdf_percentile-1.1",
        "ecdf_percentile_count-0.0",
        "ecdf_percentile_count-1.1",
        "ecdf_slope-0.5-0.5",
        "ecdf_slope-0.6-0.5",
        "ecdf_slope-0.0-0.5",
        "ecdf_slope-0.5-1.1",
        "count_above-nan",
        "count_below-nan",
        "range_count-nan-5",
        "range_count-5-nan",
        "range_count-5-0",
        "fourier_entropy-0",
        "fft_aggregated-mode",
        "not_a_feature",
    ] {
        assert!(s.parse::<Feature>().is_err(), "{s:?} should be rejected");
    }
}

#[test]
fn compute_flags_have_distinct_single_bits() {
    use super::Compute;

    let names: Vec<_> = Compute::all().iter_names().collect();
    for (name, flag) in &names {
        assert_eq!(flag.bits().count_ones(), 1, "{name} should be a single bit");
    }
    assert_eq!(
        Compute::all().bits().count_ones() as usize,
        names.len(),
        "two Compute flags share a bit"
    );
}

#[test]
fn python_feature_samples_cover_every_variant() {
    // tests/feature_samples.txt drives the cross-engine Python tests.
    let listed: HashSet<FeatureDiscriminants> = include_str!("../../tests/feature_samples.txt")
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(|l| {
            l.parse::<Feature>()
                .unwrap_or_else(|e| panic!("tests/feature_samples.txt: {e}"))
                .into()
        })
        .collect();
    let missing: Vec<_> = FeatureDiscriminants::iter()
        .filter(|d| !listed.contains(d))
        .collect();
    assert!(
        missing.is_empty(),
        "add these to tests/feature_samples.txt: {missing:?}"
    );
}
