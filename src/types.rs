use bitflags::bitflags;

// ─── Computation flags ──────────────────────────────────────────────────────

bitflags! {
    /// Bit-packed set of computation primitives needed to evaluate a feature set.
    ///
    /// Each flag gates a specific accumulation or post-processing step in the
    /// engine hot-loops. Features declare their dependencies via
    /// [`Feature::required_compute`], and the engine checks these flags to skip
    /// unnecessary work.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
    pub struct Compute: u128 {
        // ── Pass-1 accumulation gates ──────────────────────────
        const SUM            = 1 << 0;
        const MEAN           = 1 << 1;
        const VARIANCE       = 1 << 2;
        const STD            = 1 << 3;
        const MIN            = 1 << 4;
        const MAX            = 1 << 5;
        const MEDIAN         = 1 << 6;
        const SKEW           = 1 << 7;   // sum_cubes
        const KURTOSIS       = 1 << 8;   // sum_quads
        const MAD            = 1 << 9;
        const IQR            = 1 << 10;
        const ENTROPY        = 1 << 11;
        const ENERGY         = 1 << 12;  // sum of squares
        const RMS            = 1 << 13;
        const ROOT_MEAN_SQ   = 1 << 14;
        const ZERO_CROSS     = 1 << 15;
        const PEAKS          = 1 << 16;
        const AUTOCORR_LAG1  = 1 << 17;  // lag-1 product
        const MAC            = 1 << 18;  // mean abs change
        const MC             = 1 << 19;  // mean change
        const CID_CE         = 1 << 20;
        const SLOPE          = 1 << 21;  // sum_ix
        const INTERCEPT      = 1 << 22;
        const PAA            = 1 << 23;
        const ABS_SUM_CHG    = 1 << 24;
        const CNT_ABOVE_MEAN = 1 << 25;
        const CNT_BELOW_MEAN = 1 << 26;
        const STRIKE_ABOVE   = 1 << 27;
        const STRIKE_BELOW   = 1 << 28;
        const VAR_COEFF      = 1 << 29;
        const C3             = 1 << 30;
        const AUC            = 1 << 31;
        const SLOPE_SIGN_CHG = 1 << 32;
        const TURNING_PTS    = 1 << 33;
        const ZC_STATS       = 1 << 34;  // zero-crossing mean
        const ZC_STD         = 1 << 35;
        const ZC_INDICES     = 1 << 36;

        // ── Pass-2 / post-processing gates ─────────────────────
        const NEEDS_SORT     = 1 << 37;
        const ABS_MAX        = 1 << 38;
        const FIRST_LOC_MAX  = 1 << 39;
        const LAST_LOC_MAX   = 1 << 40;
        const FIRST_LOC_MIN  = 1 << 41;
        const LAST_LOC_MIN   = 1 << 42;
        const FULL_AUTOCORR  = 1 << 43;  // Wiener-Khinchin FFT autocorr
        const PACF           = 1 << 44;
        const TRA            = 1 << 45;  // time reversal asymmetry

        // ── FFT / spectral / complex feature gates ─────────────
        const FFT_COEFF      = 1 << 46;
        const APPROX_ENT     = 1 << 47;
        const AGG_LIN_TREND  = 1 << 48;
        const QUANTILE       = 1 << 49;
        const IDX_MASS_Q     = 1 << 50;
        const BENFORD        = 1 << 51;
        const LANGEVIN       = 1 << 52;
        const REOCCUR_VAL    = 1 << 53;
        const SPEC_CENTROID  = 1 << 54;
        const SPEC_DISTANCE  = 1 << 55;
        const SPEC_DECREASE  = 1 << 56;
        const SPEC_SLOPE     = 1 << 57;
        const SIG_DISTANCE   = 1 << 58;
        const WAVELET        = 1 << 59;
        const SPECTROGRAM    = 1 << 60;
        const ABS_SUM        = 1 << 61;
        const REOCCUR_DP     = 1 << 62;
        const MEAN_N_ABS_MAX = 1 << 63;
        const HUMAN_RANGE_E  = 1 << 64;
        const LENGTH         = 1 << 65;
        const VAR_GT_STD     = 1 << 66;
        const HAS_DUPLICATE  = 1 << 67;
        const HAS_DUP_MAX    = 1 << 68;
        const HAS_DUP_MIN    = 1 << 69;

        // ── Composite masks (zero bit-cost) ────────────────────
        const ANY_FFT = Self::FFT_COEFF.bits()
            | Self::HUMAN_RANGE_E.bits()
            | Self::SPEC_CENTROID.bits()
            | Self::SPEC_DISTANCE.bits()
            | Self::SPEC_DECREASE.bits()
            | Self::SPEC_SLOPE.bits()
            | Self::SPECTROGRAM.bits();

        const ANY_DIFF = Self::ZERO_CROSS.bits()
            | Self::AUTOCORR_LAG1.bits()
            | Self::MAC.bits()
            | Self::MC.bits()
            | Self::CID_CE.bits()
            | Self::AUC.bits();
    }
}

// ─── Feature enum ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Copy)]
pub enum Feature {
    TotalSum,
    Mean,
    Variance,
    Std,
    Min,
    Max,
    Median,
    Skew,
    UnbiasedFisherKurtosis, // tsfresh default
    BiasedFisherKurtosis,   // tsfel default
    Mad,
    Iqr,
    Entropy,
    Energy,
    Rms,
    RootMeanSquare,
    ZeroCrossingRate,
    PeakCount,
    AutocorrLag1,    // Centered (tsfresh default)
    AutocorrFirst1e, // tsfel 'Autocorrelation' feature
    MeanAbsChange,
    MeanChange,
    CidCe,
    Slope,
    Intercept,
    Paa(u16, u16),
    AbsSumChange,
    CountAboveMean,
    CountBelowMean,
    LongestStrikeAboveMean,
    LongestStrikeBelowMean,
    VariationCoefficient,
    C3(u16),
    Auc,
    SlopeSignChange,
    TurningPoints,
    ZeroCrossingMean,
    ZeroCrossingStd,
    AbsMax,
    FirstLocMax,
    LastLocMax,
    FirstLocMin,
    LastLocMin,
    Autocorr(u16),
    PartialAutocorr(u16),
    TimeReversalAsymmetry(u16),
    FftCoefficient(u16, FftAttr),
    ApproxEntropy(u8, u32), // r is encoded as u32 (fixed point or bitcast)
    AggLinearTrend(AggAttr, u16, AggFunc),
    Quantile(u32),          // q encoded as u32 bits
    IndexMassQuantile(u32), // q encoded as u32 bits
    BenfordCorrelation,
    MaxLangevinFixedPoint(u8, u32), // m, r as bits
    SumOfReoccurringValues,
    SumOfReoccurringDataPoints,
    MeanNAbsoluteMax(u16),
    Length,
    VarianceLargerThanStandardDeviation,
    HumanRangeEnergy(u32), // fs as bits
    SpectralCentroid,
    SpectralDistance,
    SpectralDecrease,
    SpectralSlope,
    SignalDistance,
    WaveletFeatures(u32, u16), // mother wavelet (freq stored as f32 bits), feature type
    SpectrogramCoefficients(u16, u32), // time, freq stored as f32 bits
    MeanSecondDerivativeCentral,
    LargeStandardDeviation(u32),
    SymmetryLooking(u32),
    HasDuplicateMax,
    HasDuplicateMin,
    HasDuplicate,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Copy)]
pub enum FftAttr {
    Real,
    Imag,
    Abs,
    Angle,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Copy)]
pub enum AggAttr {
    Slope,
    Intercept,
    Stderr,
    RValue,
    PValue,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Copy)]
pub enum AggFunc {
    Max,
    Min,
    Mean,
    Var,
}

// ─── Feature → Compute dependency mapping ───────────────────────────────────

impl Feature {
    /// Returns the set of computation flags this feature requires.
    pub fn required_compute(&self) -> Compute {
        use Compute as C;
        match self {
            Self::TotalSum => C::SUM,
            Self::Mean => C::SUM | C::MEAN,
            Self::Variance => C::SUM | C::MEAN | C::VARIANCE | C::ENERGY | C::NEEDS_SORT,
            Self::Std => C::SUM | C::MEAN | C::VARIANCE | C::STD | C::ENERGY | C::NEEDS_SORT,
            Self::Min => C::MIN,
            Self::Max => C::MAX,
            Self::Median => C::MEDIAN | C::NEEDS_SORT,
            Self::Skew => C::SUM | C::MEAN | C::VARIANCE | C::SKEW | C::ENERGY | C::NEEDS_SORT,
            Self::UnbiasedFisherKurtosis | Self::BiasedFisherKurtosis => {
                C::SUM | C::MEAN | C::VARIANCE | C::KURTOSIS | C::ENERGY | C::NEEDS_SORT
            }
            Self::Mad => C::SUM | C::MEAN | C::MAD | C::NEEDS_SORT,
            Self::Iqr => C::MIN | C::MAX | C::MEDIAN | C::IQR | C::NEEDS_SORT,
            Self::Entropy => {
                C::MIN | C::MAX | C::MEDIAN | C::IQR | C::ENTROPY | C::NEEDS_SORT
            }
            Self::Energy => C::ENERGY,
            Self::Rms => C::ENERGY | C::RMS,
            Self::RootMeanSquare => C::ENERGY | C::ROOT_MEAN_SQ,
            Self::ZeroCrossingRate => C::ZERO_CROSS,
            Self::PeakCount => C::PEAKS,
            Self::AutocorrLag1 => C::SUM | C::MEAN | C::ENERGY | C::AUTOCORR_LAG1,
            Self::AutocorrFirst1e => {
                C::SUM | C::MEAN | C::ENERGY | C::FULL_AUTOCORR | C::NEEDS_SORT
            }
            Self::MeanAbsChange => C::SUM | C::MEAN | C::MAC,
            Self::MeanChange => C::SUM | C::MEAN | C::MC,
            Self::CidCe => C::SUM | C::MEAN | C::CID_CE,
            Self::Slope => C::SUM | C::MEAN | C::SLOPE,
            Self::Intercept => C::SUM | C::MEAN | C::SLOPE | C::INTERCEPT,
            Self::Paa(_, _) => C::PAA,
            Self::AbsSumChange => C::ABS_SUM_CHG,
            Self::CountAboveMean => C::SUM | C::MEAN | C::CNT_ABOVE_MEAN | C::NEEDS_SORT,
            Self::CountBelowMean => C::SUM | C::MEAN | C::CNT_BELOW_MEAN | C::NEEDS_SORT,
            Self::LongestStrikeAboveMean => {
                C::SUM | C::MEAN | C::STRIKE_ABOVE | C::NEEDS_SORT
            }
            Self::LongestStrikeBelowMean => {
                C::SUM | C::MEAN | C::STRIKE_BELOW | C::NEEDS_SORT
            }
            Self::VariationCoefficient => {
                C::SUM | C::MEAN | C::VARIANCE | C::STD | C::ENERGY | C::VAR_COEFF | C::NEEDS_SORT
            }
            Self::C3(_) => C::C3,
            Self::Auc => C::AUC,
            Self::SlopeSignChange => C::PEAKS | C::SLOPE_SIGN_CHG,
            Self::TurningPoints => C::PEAKS | C::TURNING_PTS,
            Self::ZeroCrossingMean => {
                C::SUM | C::MEAN | C::ZERO_CROSS | C::ZC_STATS | C::ZC_INDICES | C::NEEDS_SORT
            }
            Self::ZeroCrossingStd => {
                C::SUM
                    | C::MEAN
                    | C::ZERO_CROSS
                    | C::ZC_STATS
                    | C::ZC_STD
                    | C::ZC_INDICES
                    | C::NEEDS_SORT
            }
            Self::AbsMax => C::ABS_MAX,
            Self::FirstLocMax => C::MAX | C::NEEDS_SORT | C::FIRST_LOC_MAX,
            Self::LastLocMax => C::MAX | C::NEEDS_SORT | C::LAST_LOC_MAX,
            Self::FirstLocMin => C::MIN | C::NEEDS_SORT | C::FIRST_LOC_MIN,
            Self::LastLocMin => C::MIN | C::NEEDS_SORT | C::LAST_LOC_MIN,
            Self::Autocorr(lag) => {
                if *lag == 1 {
                    C::SUM | C::MEAN | C::ENERGY | C::AUTOCORR_LAG1 | C::NEEDS_SORT
                } else {
                    C::SUM | C::MEAN | C::VARIANCE | C::ENERGY | C::FULL_AUTOCORR | C::NEEDS_SORT
                }
            }
            Self::PartialAutocorr(_) => {
                C::SUM | C::MEAN | C::VARIANCE | C::ENERGY | C::PACF | C::NEEDS_SORT
            }
            Self::TimeReversalAsymmetry(_) => C::TRA | C::NEEDS_SORT,
            Self::FftCoefficient(_, _) => C::FFT_COEFF | C::NEEDS_SORT,
            Self::ApproxEntropy(_, _) => C::APPROX_ENT | C::NEEDS_SORT,
            Self::AggLinearTrend(_, _, _) => C::AGG_LIN_TREND | C::NEEDS_SORT,
            Self::Quantile(_) => C::QUANTILE | C::NEEDS_SORT,
            Self::IndexMassQuantile(_) => C::ABS_SUM | C::IDX_MASS_Q | C::NEEDS_SORT,
            Self::BenfordCorrelation => C::BENFORD | C::NEEDS_SORT,
            Self::MaxLangevinFixedPoint(_, _) => C::LANGEVIN | C::NEEDS_SORT,
            Self::SumOfReoccurringValues => C::REOCCUR_VAL | C::NEEDS_SORT,
            Self::SumOfReoccurringDataPoints => C::REOCCUR_DP | C::NEEDS_SORT,
            Self::MeanNAbsoluteMax(_) => C::MEAN_N_ABS_MAX | C::NEEDS_SORT,
            Self::Length => C::LENGTH | C::NEEDS_SORT,
            Self::VarianceLargerThanStandardDeviation => {
                C::SUM | C::MEAN | C::VARIANCE | C::ENERGY | C::NEEDS_SORT | C::VAR_GT_STD
            }
            Self::HumanRangeEnergy(_) => C::HUMAN_RANGE_E | C::NEEDS_SORT,
            Self::SpectralCentroid => C::SPEC_CENTROID | C::NEEDS_SORT,
            Self::SpectralDistance => C::SPEC_DISTANCE | C::NEEDS_SORT,
            Self::SpectralDecrease => C::SPEC_DECREASE | C::NEEDS_SORT,
            Self::SpectralSlope => C::SPEC_SLOPE | C::NEEDS_SORT,
            Self::SignalDistance => C::SIG_DISTANCE | C::NEEDS_SORT,
            Self::WaveletFeatures(_, _) => C::WAVELET | C::NEEDS_SORT,
            Self::SpectrogramCoefficients(_, _) => C::SPECTROGRAM | C::NEEDS_SORT,
            Self::MeanSecondDerivativeCentral => C::LENGTH | C::NEEDS_SORT,
            Self::LargeStandardDeviation(_) => {
                C::SUM | C::MEAN | C::VARIANCE | C::MIN | C::MAX | C::ENERGY | C::NEEDS_SORT
            }
            Self::SymmetryLooking(_) => {
                C::SUM | C::MEAN | C::MIN | C::MAX | C::MEDIAN | C::NEEDS_SORT
            }
            Self::HasDuplicateMax => C::MAX | C::HAS_DUP_MAX | C::NEEDS_SORT,
            Self::HasDuplicateMin => C::MIN | C::HAS_DUP_MIN | C::NEEDS_SORT,
            Self::HasDuplicate => C::HAS_DUPLICATE | C::NEEDS_SORT,
        }
    }
}

// ─── Feature → Compute aggregation ─────────────────────────────────────────

/// Fold a feature list into a single `Compute` bitflag set.
pub fn compute_flags(features: &[Feature]) -> Compute {
    features
        .iter()
        .fold(Compute::empty(), |acc, f| acc | f.required_compute())
}

// ─── FromStr ────────────────────────────────────────────────────────────────

impl std::str::FromStr for Feature {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // 1. Exact matches (simple features)
        match s {
            "total_sum" | "value__sum_values" => return Ok(Feature::TotalSum),
            "mean" | "value__mean" => return Ok(Feature::Mean),
            "variance" | "value__variance" => return Ok(Feature::Variance),
            "std" | "std_dev" | "value__standard_deviation" => return Ok(Feature::Std),
            "min" | "min_value" | "value__minimum" => return Ok(Feature::Min),
            "max" | "max_value" | "value__maximum" => return Ok(Feature::Max),
            "median" | "value__median" => return Ok(Feature::Median),
            "skew" | "skewness" | "value__skewness" => return Ok(Feature::Skew),
            "kurtosis" | "value__kurtosis" | "unbiased_fisher_kurtosis" => {
                return Ok(Feature::UnbiasedFisherKurtosis)
            }
            "biased_fisher_kurtosis" => return Ok(Feature::BiasedFisherKurtosis),
            "mad" => return Ok(Feature::Mad),
            "iqr" => return Ok(Feature::Iqr),
            "entropy" => return Ok(Feature::Entropy),
            "energy" | "torque_Absolute energy" => return Ok(Feature::Energy),
            "rms" => return Ok(Feature::Rms),
            "root_mean_square" => return Ok(Feature::RootMeanSquare),
            "zero_crossing_rate" => return Ok(Feature::ZeroCrossingRate),
            "has_duplicate_max" => return Ok(Feature::HasDuplicateMax),
            "has_duplicate_min" => return Ok(Feature::HasDuplicateMin),
            "has_duplicate" => return Ok(Feature::HasDuplicate),
            "peak_count" => return Ok(Feature::PeakCount),
            "autocorr_lag1" | "centered_autocorr_lag1" => return Ok(Feature::AutocorrLag1),
            "autocorrelation" | "autocorr_first_1e" => return Ok(Feature::AutocorrFirst1e),
            "mean_abs_change" => return Ok(Feature::MeanAbsChange),
            "mean_change" | "mean_diff" | "torque_Mean diff" => return Ok(Feature::MeanChange),
            "cid_ce" => return Ok(Feature::CidCe),
            "slope" | "torque_Slope" => return Ok(Feature::Slope),
            "intercept" => return Ok(Feature::Intercept),
            "abs_sum_change" => return Ok(Feature::AbsSumChange),
            "count_above_mean" => return Ok(Feature::CountAboveMean),
            "count_below_mean" => return Ok(Feature::CountBelowMean),
            "longest_strike_above_mean" => return Ok(Feature::LongestStrikeAboveMean),
            "longest_strike_below_mean" => return Ok(Feature::LongestStrikeBelowMean),
            "variation_coefficient" => return Ok(Feature::VariationCoefficient),
            "auc" => return Ok(Feature::Auc),
            "slope_sign_change" => return Ok(Feature::SlopeSignChange),
            "turning_points" => return Ok(Feature::TurningPoints),
            "zero_crossing_mean" => return Ok(Feature::ZeroCrossingMean),
            "zero_crossing_std" => return Ok(Feature::ZeroCrossingStd),
            "abs_max" | "value__absolute_maximum" => return Ok(Feature::AbsMax),
            "first_loc_max" | "value__first_location_of_maximum" => {
                return Ok(Feature::FirstLocMax)
            }
            "last_loc_max" | "value__last_location_of_maximum" => {
                return Ok(Feature::LastLocMax)
            }
            "first_loc_min" | "value__first_location_of_minimum" => {
                return Ok(Feature::FirstLocMin)
            }
            "last_loc_min" | "value__last_location_of_minimum" => {
                return Ok(Feature::LastLocMin)
            }
            "benford_correlation" | "value__benford_correlation" => {
                return Ok(Feature::BenfordCorrelation)
            }
            "sum_of_reoccurring_values" | "value__sum_of_reoccurring_values" => {
                return Ok(Feature::SumOfReoccurringValues)
            }
            "sum_of_reoccurring_data_points" | "value__sum_of_reoccurring_data_points" => {
                return Ok(Feature::SumOfReoccurringDataPoints)
            }
            "length" | "value__length" => return Ok(Feature::Length),
            "variance_larger_than_standard_deviation"
            | "value__variance_larger_than_standard_deviation" => {
                return Ok(Feature::VarianceLargerThanStandardDeviation)
            }
            "spectral_centroid" | "torque_Centroid" => return Ok(Feature::SpectralCentroid),
            "spectral_distance" | "torque_Spectral distance" => {
                return Ok(Feature::SpectralDistance)
            }
            "spectral_decrease" | "torque_Spectral decrease" => {
                return Ok(Feature::SpectralDecrease)
            }
            "spectral_slope" | "torque_Spectral slope" => return Ok(Feature::SpectralSlope),
            "signal_distance" | "torque_Signal distance" => return Ok(Feature::SignalDistance),
            "human_range_energy" | "torque_Human range energy" => {
                return Ok(Feature::HumanRangeEnergy(100.0f32.to_bits())) // Default fs=100
            }
            "mean_second_derivative_central" => {
                return Ok(Feature::MeanSecondDerivativeCentral)
            }
            _ => {}
        }

        // 2. Parameterized features (prefix-based)
        if let Some(f) = parse_parameterized(s) {
            return Ok(f);
        }

        // 3. Legacy tsfresh / torque format
        if let Some(f) = parse_legacy_format(s) {
            return Ok(f);
        }

        Err(format!("Unknown feature: {}", s))
    }
}

// ─── Parameterized parsers ──────────────────────────────────────────────────

fn parse_parameterized(s: &str) -> Option<Feature> {
    if let Some(arg) = s.strip_prefix("paa-") {
        let (n, m) = arg.split_once('-')?;
        return Some(Feature::Paa(n.parse().ok()?, m.parse().ok()?));
    }
    if let Some(arg) = s.strip_prefix("c3-") {
        return Some(Feature::C3(arg.trim().parse().ok()?));
    }
    if let Some(arg) = s.strip_prefix("autocorr-") {
        return Some(Feature::Autocorr(arg.parse().ok()?));
    }
    if let Some(arg) = s.strip_prefix("partial_autocorr-") {
        return Some(Feature::PartialAutocorr(arg.parse().ok()?));
    }
    if let Some(arg) = s.strip_prefix("time_reversal_asymmetry-") {
        return Some(Feature::TimeReversalAsymmetry(arg.parse().ok()?));
    }
    if let Some(arg) = s.strip_prefix("fft_coeff-") {
        let (coeff_s, attr_s) = arg.split_once('-')?;
        let coeff: u16 = coeff_s.parse().ok()?;
        let attr = match attr_s {
            "real" => FftAttr::Real,
            "imag" => FftAttr::Imag,
            "abs" => FftAttr::Abs,
            "angle" => FftAttr::Angle,
            _ => return None,
        };
        return Some(Feature::FftCoefficient(coeff, attr));
    }
    if let Some(arg) = s.strip_prefix("approx_entropy-") {
        let (m_s, r_s) = arg.split_once('-')?;
        let m: u8 = m_s.parse().ok()?;
        let r: f32 = r_s.parse().ok()?;
        return Some(Feature::ApproxEntropy(m, r.to_bits()));
    }
    if let Some(arg) = s.strip_prefix("agg_linear_trend-") {
        let parts: Vec<&str> = arg.split('-').collect();
        if parts.len() != 3 {
            return None;
        }
        let attr = parse_agg_attr(parts[0])?;
        let chunk_len: u16 = parts[1].parse().ok()?;
        let func = parse_agg_func(parts[2])?;
        return Some(Feature::AggLinearTrend(attr, chunk_len, func));
    }
    if let Some(arg) = s.strip_prefix("quantile-") {
        let q: f32 = arg.parse().ok()?;
        return Some(Feature::Quantile(q.to_bits()));
    }
    if let Some(arg) = s.strip_prefix("index_mass_quantile-") {
        let q: f32 = arg.parse().ok()?;
        return Some(Feature::IndexMassQuantile(q.to_bits()));
    }
    if let Some(arg) = s.strip_prefix("max_langevin_fixed_point-") {
        let (m_s, r_s) = arg.split_once('-')?;
        let m: u8 = m_s.parse().ok()?;
        let r: f32 = r_s.parse().ok()?;
        return Some(Feature::MaxLangevinFixedPoint(m, r.to_bits()));
    }
    if let Some(arg) = s.strip_prefix("mean_n_absolute_max-") {
        return Some(Feature::MeanNAbsoluteMax(arg.parse().ok()?));
    }
    if let Some(arg) = s.strip_prefix("wavelet-") {
        let (w_s, f_s) = arg.split_once('-')?;
        let w: f32 = w_s.parse().ok()?;
        let f: u16 = f_s.parse().ok()?;
        return Some(Feature::WaveletFeatures(w.to_bits(), f));
    }
    if let Some(arg) = s.strip_prefix("spectrogram-") {
        let (t_s, f_s) = arg.split_once('-')?;
        let t: u16 = t_s.parse().ok()?;
        let f: f32 = f_s.parse().ok()?;
        return Some(Feature::SpectrogramCoefficients(t, f.to_bits()));
    }
    if let Some(arg) = s.strip_prefix("large_standard_deviation-") {
        let r: f32 = arg.parse().ok()?;
        return Some(Feature::LargeStandardDeviation(r.to_bits()));
    }
    if let Some(arg) = s.strip_prefix("symmetry_looking-") {
        let r: f32 = arg.parse().ok()?;
        return Some(Feature::SymmetryLooking(r.to_bits()));
    }
    None
}

// ─── Legacy tsfresh / torque format parsers ─────────────────────────────────

fn parse_legacy_format(s: &str) -> Option<Feature> {
    if s.contains("c3__lag_") {
        let pos = s.find("lag_")?;
        let n: u16 = s[pos + 4..].parse().ok()?;
        return Some(Feature::C3(n));
    }
    if s.contains("time_reversal_asymmetry_statistic__lag_") {
        let pos = s.find("lag_")?;
        let n: u16 = s[pos + 4..].parse().ok()?;
        return Some(Feature::TimeReversalAsymmetry(n));
    }
    if s.contains("agg_linear_trend__attr_") {
        let attr = if s.contains("attr_\"slope\"") {
            AggAttr::Slope
        } else if s.contains("attr_\"intercept\"") {
            AggAttr::Intercept
        } else if s.contains("attr_\"stderr\"") {
            AggAttr::Stderr
        } else if s.contains("attr_\"rvalue\"") {
            AggAttr::RValue
        } else if s.contains("attr_\"pvalue\"") {
            AggAttr::PValue
        } else {
            AggAttr::Slope
        };
        let chunk_len = if let Some(pos) = s.find("chunk_len_") {
            let sub = &s[pos + 10..];
            let end = sub.find("__").unwrap_or(sub.len());
            sub[..end].parse::<u16>().unwrap_or(5)
        } else {
            5
        };
        let func = if s.contains("f_agg_\"mean\"") {
            AggFunc::Mean
        } else if s.contains("f_agg_\"var\"") {
            AggFunc::Var
        } else if s.contains("f_agg_\"max\"") {
            AggFunc::Max
        } else if s.contains("f_agg_\"min\"") {
            AggFunc::Min
        } else {
            AggFunc::Mean
        };
        return Some(Feature::AggLinearTrend(attr, chunk_len, func));
    }
    if s.contains("value__quantile__q_") {
        let pos = s.find("q_")?;
        let q: f32 = s[pos + 2..].parse().ok()?;
        return Some(Feature::Quantile(q.to_bits()));
    }
    if s.contains("value__index_mass_quantile__q_") {
        let pos = s.find("q_")?;
        let q: f32 = s[pos + 2..].parse().ok()?;
        return Some(Feature::IndexMassQuantile(q.to_bits()));
    }
    if s.contains("value__max_langevin_fixed_point__m_") {
        let m = if let Some(pos) = s.find("m_") {
            let sub = &s[pos + 2..];
            let end = sub.find("__").unwrap_or(sub.len());
            sub[..end].parse::<u8>().unwrap_or(3)
        } else {
            3
        };
        let r = if let Some(pos) = s.find("r_") {
            let sub = &s[pos + 2..];
            let end = sub.find("__").unwrap_or(sub.len());
            sub[..end].parse::<f32>().unwrap_or(30.0)
        } else {
            30.0
        };
        return Some(Feature::MaxLangevinFixedPoint(m, r.to_bits()));
    }
    if s.contains("value__mean_n_absolute_max__number_of_maxima_") {
        let pos = s.find("number_of_maxima_")?;
        let n: u16 = s[pos + 17..].parse().ok()?;
        return Some(Feature::MeanNAbsoluteMax(n));
    }
    if s.contains("torque_Wavelet") {
        let f_type: u16 = if s.contains("absolute mean") { 0 } else { 1 };
        let freq = if let Some(pos) = s.rfind('_') {
            if let Some(end) = s.find("Hz") {
                s[pos + 1..end].parse::<f32>().unwrap_or(0.0)
            } else {
                0.0
            }
        } else {
            0.0
        };
        return Some(Feature::WaveletFeatures(freq.to_bits(), f_type));
    }
    if s.contains("torque_Spectrogram mean coefficient_") {
        if let Some(pos) = s.rfind('_') {
            if let Some(end) = s.find("Hz") {
                let freq = s[pos + 1..end].parse::<f32>().unwrap_or(0.0);
                return Some(Feature::SpectrogramCoefficients(0, freq.to_bits()));
            }
        }
    }
    if let Some(arg) = s.strip_prefix("large_standard_deviation__r_") {
        let r: f32 = arg.parse().ok()?;
        return Some(Feature::LargeStandardDeviation(r.to_bits()));
    }
    if let Some(arg) = s.strip_prefix("symmetry_looking__r_") {
        let r: f32 = arg.parse().ok()?;
        return Some(Feature::SymmetryLooking(r.to_bits()));
    }
    None
}

// ─── Shared sub-enum parsers ────────────────────────────────────────────────

fn parse_agg_attr(s: &str) -> Option<AggAttr> {
    match s {
        "slope" => Some(AggAttr::Slope),
        "intercept" => Some(AggAttr::Intercept),
        "stderr" => Some(AggAttr::Stderr),
        "rvalue" => Some(AggAttr::RValue),
        "pvalue" => Some(AggAttr::PValue),
        _ => None,
    }
}

fn parse_agg_func(s: &str) -> Option<AggFunc> {
    match s {
        "max" => Some(AggFunc::Max),
        "min" => Some(AggFunc::Min),
        "mean" => Some(AggFunc::Mean),
        "var" => Some(AggFunc::Var),
        _ => None,
    }
}

// ─── Feature::name() ───────────────────────────────────────────────────────

impl Feature {
    pub fn name(&self) -> String {
        match self {
            Feature::TotalSum => "total_sum".to_string(),
            Feature::Mean => "mean".to_string(),
            Feature::Variance => "variance".to_string(),
            Feature::Std => "std_dev".to_string(),
            Feature::Min => "min_value".to_string(),
            Feature::Max => "max_value".to_string(),
            Feature::Median => "median".to_string(),
            Feature::Skew => "skewness".to_string(),
            Feature::UnbiasedFisherKurtosis => "kurtosis".to_string(),
            Feature::BiasedFisherKurtosis => "biased_fisher_kurtosis".to_string(),
            Feature::Mad => "mad".to_string(),
            Feature::Iqr => "iqr".to_string(),
            Feature::Entropy => "entropy".to_string(),
            Feature::Energy => "energy".to_string(),
            Feature::Rms => "rms".to_string(),
            Feature::RootMeanSquare => "root_mean_square".to_string(),
            Feature::ZeroCrossingRate => "zero_crossing_rate".to_string(),
            Feature::PeakCount => "peak_count".to_string(),
            Feature::AutocorrLag1 => "autocorr_lag1".to_string(),
            Feature::AutocorrFirst1e => "autocorrelation".to_string(),
            Feature::MeanAbsChange => "mean_abs_change".to_string(),
            Feature::MeanChange => "mean_change".to_string(),
            Feature::CidCe => "cid_ce".to_string(),
            Feature::Slope => "slope".to_string(),
            Feature::Intercept => "intercept".to_string(),
            Feature::AbsSumChange => "abs_sum_change".to_string(),
            Feature::CountAboveMean => "count_above_mean".to_string(),
            Feature::CountBelowMean => "count_below_mean".to_string(),
            Feature::LongestStrikeAboveMean => "longest_strike_above_mean".to_string(),
            Feature::LongestStrikeBelowMean => "longest_strike_below_mean".to_string(),
            Feature::VariationCoefficient => "variation_coefficient".to_string(),
            Feature::Auc => "auc".to_string(),
            Feature::SlopeSignChange => "slope_sign_change".to_string(),
            Feature::TurningPoints => "turning_points".to_string(),
            Feature::ZeroCrossingMean => "zero_crossing_mean".to_string(),
            Feature::ZeroCrossingStd => "zero_crossing_std".to_string(),
            Feature::C3(lag) => format!("c3-{}", lag),
            Feature::Paa(total, index) => format!("paa-{}-{}", total, index),
            Feature::AbsMax => "abs_max".to_string(),
            Feature::FirstLocMax => "first_loc_max".to_string(),
            Feature::LastLocMax => "last_loc_max".to_string(),
            Feature::FirstLocMin => "first_loc_min".to_string(),
            Feature::LastLocMin => "last_loc_min".to_string(),
            Feature::Autocorr(lag) => format!("autocorr-{}", lag),
            Feature::PartialAutocorr(lag) => format!("partial_autocorr-{}", lag),
            Feature::TimeReversalAsymmetry(lag) => format!("time_reversal_asymmetry-{}", lag),
            Feature::FftCoefficient(coeff, attr) => {
                let attr_str = match attr {
                    FftAttr::Real => "real",
                    FftAttr::Imag => "imag",
                    FftAttr::Abs => "abs",
                    FftAttr::Angle => "angle",
                };
                format!("fft_coeff-{}-{}", coeff, attr_str)
            }
            Feature::ApproxEntropy(m, r_bits) => {
                format!("approx_entropy-{}-{}", m, f32::from_bits(*r_bits))
            }
            Feature::AggLinearTrend(attr, chunk_len, func) => {
                let attr_str = match attr {
                    AggAttr::Slope => "slope",
                    AggAttr::Intercept => "intercept",
                    AggAttr::Stderr => "stderr",
                    AggAttr::RValue => "rvalue",
                    AggAttr::PValue => "pvalue",
                };
                let func_str = match func {
                    AggFunc::Max => "max",
                    AggFunc::Min => "min",
                    AggFunc::Mean => "mean",
                    AggFunc::Var => "var",
                };
                format!("agg_linear_trend-{}-{}-{}", attr_str, chunk_len, func_str)
            }
            Feature::Quantile(q_bits) => format!("quantile-{}", f32::from_bits(*q_bits)),
            Feature::IndexMassQuantile(q_bits) => {
                format!("index_mass_quantile-{}", f32::from_bits(*q_bits))
            }
            Feature::BenfordCorrelation => "benford_correlation".to_string(),
            Feature::MaxLangevinFixedPoint(m, r_bits) => {
                format!("max_langevin_fixed_point-{}-{}", m, f32::from_bits(*r_bits))
            }
            Feature::SumOfReoccurringValues => "sum_of_reoccurring_values".to_string(),
            Feature::SumOfReoccurringDataPoints => "sum_of_reoccurring_data_points".to_string(),
            Feature::Length => "length".to_string(),
            Feature::VarianceLargerThanStandardDeviation => {
                "variance_larger_than_standard_deviation".to_string()
            }
            Feature::MeanNAbsoluteMax(n) => format!("mean_n_absolute_max-{}", n),
            Feature::HumanRangeEnergy(fs_bits) => {
                format!("human_range_energy-{}", f32::from_bits(*fs_bits))
            }
            Feature::SpectralCentroid => "spectral_centroid".to_string(),
            Feature::SpectralDistance => "spectral_distance".to_string(),
            Feature::SpectralDecrease => "spectral_decrease".to_string(),
            Feature::SpectralSlope => "spectral_slope".to_string(),
            Feature::SignalDistance => "signal_distance".to_string(),
            Feature::WaveletFeatures(w_bits, f) => {
                format!("wavelet-{}-{}", f32::from_bits(*w_bits), f)
            }
            Feature::SpectrogramCoefficients(t, f_bits) => {
                format!("spectrogram-{}-{}", t, f32::from_bits(*f_bits))
            }
            Feature::MeanSecondDerivativeCentral => "mean_second_derivative_central".to_string(),
            Feature::LargeStandardDeviation(r_bits) => {
                format!("large_standard_deviation-{}", f32::from_bits(*r_bits))
            }
            Feature::SymmetryLooking(r_bits) => {
                format!("symmetry_looking-{}", f32::from_bits(*r_bits))
            }
            Feature::HasDuplicateMax => "has_duplicate_max".to_string(),
            Feature::HasDuplicateMin => "has_duplicate_min".to_string(),
            Feature::HasDuplicate => "has_duplicate".to_string(),
        }
    }
}
