#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct FastBitArray(pub u128);

impl FastBitArray {
    pub const ZERO: Self = Self(0);

    #[inline]
    pub fn set(&mut self, index: usize) {
        self.0 |= 1 << index;
    }

    #[inline]
    pub fn unset(&mut self, index: usize) {
        self.0 &= !(1 << index);
    }

    #[inline]
    pub fn get(&self, index: usize) -> bool {
        (self.0 >> index) & 1 == 1
    }

    #[inline]
    pub fn set_batch<const N: usize>(&mut self, indices: [usize; N]) {
        let mut mask = 0u128;
        for i in indices {
            mask |= 1 << i;
        }
        self.0 |= mask;
    }

    #[inline]
    pub fn unset_batch<const N: usize>(&mut self, indices: [usize; N]) {
        let mut mask = 0u128;
        for i in indices {
            mask |= 1 << i;
        }
        self.0 &= !mask;
    }

    #[inline]
    pub fn any<const N: usize>(&self, indices: [usize; N]) -> bool {
        let mut mask = 0u128;
        for i in indices {
            mask |= 1 << i;
        }
        (self.0 & mask) != 0
    }

    #[inline]
    pub fn all<const N: usize>(&self, indices: [usize; N]) -> bool {
        let mut mask = 0u128;
        for i in indices {
            mask |= 1 << i;
        }
        (self.0 & mask) == mask
    }

    pub fn any_fft(&self) -> bool {
        // Bits: 46 (FftCoefficient), 64 (HumanRangeEnergy), 54 (SpectralCentroid), 
        // 55 (SpectralDistance), 56 (SpectralDecrease), 57 (SpectralSlope), 
        // 60 (SpectrogramCoefficients)
        let mask = (1u128 << 46) | (1u128 << 64) | (1u128 << 54) | (1u128 << 55) | 
                   (1u128 << 56) | (1u128 << 57) | (1u128 << 60);
        (self.0 & mask) != 0
    }
}

impl std::ops::Index<usize> for FastBitArray {
    type Output = bool;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        if (self.0 & (1 << index)) != 0 {
            &true
        } else {
            &false
        }
    }
}

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
    Kurtosis,
    Mad,
    Iqr,
    Entropy,
    Energy,
    Rms,
    RootMeanSquare,
    ZeroCrossingRate,
    PeakCount,
    AutocorrLag1,
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
    Quantile(u32),           // q encoded as u32 bits
    IndexMassQuantile(u32),  // q encoded as u32 bits
    BenfordCorrelation,
    MaxLangevinFixedPoint(u8, u32), // m, r as bits
    SumOfReoccurringValues,
    SumOfReoccurringDataPoints,
    MeanNAbsoluteMax(u16),
    HumanRangeEnergy(u32), // fs as bits
    SpectralCentroid,
    SpectralDistance,
    SpectralDecrease,
    SpectralSlope,
    SignalDistance,
    WaveletFeatures(u16, u16), // mother wavelet, feature type
    SpectrogramCoefficients(u16, u16), // time, freq
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

impl From<String> for Feature {
    fn from(s: String) -> Self {
        match s.as_str() {
            "total_sum" | "value__sum_values" => Feature::TotalSum,
            "mean" | "value__mean" => Feature::Mean,
            "variance" | "value__variance" => Feature::Variance,
            "std" | "std_dev" | "value__standard_deviation" => Feature::Std,
            "min" | "min_value" | "value__minimum" => Feature::Min,
            "max" | "max_value" | "value__maximum" => Feature::Max,
            "median" | "value__median" => Feature::Median,
            "skew" | "skewness" | "value__skewness" => Feature::Skew,
            "kurtosis" | "value__kurtosis" => Feature::Kurtosis,
            "mad" => Feature::Mad,
            "iqr" => Feature::Iqr,
            "entropy" => Feature::Entropy,
            "energy" | "torque_Absolute energy" => Feature::Energy,
            "rms" => Feature::Rms,
            "root_mean_square" => Feature::RootMeanSquare,
            "zero_crossing_rate" => Feature::ZeroCrossingRate,
            "peak_count" => Feature::PeakCount,
            "autocorr_lag1" => Feature::AutocorrLag1,
            "mean_abs_change" => Feature::MeanAbsChange,
            "mean_change" | "mean_diff" | "torque_Mean diff" => Feature::MeanChange,
            "cid_ce" => Feature::CidCe,
            "slope" | "torque_Slope" => Feature::Slope,
            "intercept" => Feature::Intercept,
            "abs_sum_change" => Feature::AbsSumChange,
            "count_above_mean" => Feature::CountAboveMean,
            "count_below_mean" => Feature::CountBelowMean,
            "longest_strike_above_mean" => Feature::LongestStrikeAboveMean,
            "longest_strike_below_mean" => Feature::LongestStrikeBelowMean,
            "variation_coefficient" => Feature::VariationCoefficient,
            "auc" => Feature::Auc,
            "slope_sign_change" => Feature::SlopeSignChange,
            "turning_points" => Feature::TurningPoints,
            "zero_crossing_mean" => Feature::ZeroCrossingMean,
            "zero_crossing_std" => Feature::ZeroCrossingStd,
            "abs_max" | "value__absolute_maximum" => Feature::AbsMax,
            "first_loc_max" | "value__first_location_of_maximum" => Feature::FirstLocMax,
            "last_loc_max" | "value__last_location_of_maximum" => Feature::LastLocMax,
            "first_loc_min" | "value__first_location_of_minimum" => Feature::FirstLocMin,
            "last_loc_min" | "value__last_location_of_minimum" => Feature::LastLocMin,
            "benford_correlation" | "value__benford_correlation" => Feature::BenfordCorrelation,
            "sum_of_reoccurring_values" | "value__sum_of_reoccurring_values" => Feature::SumOfReoccurringValues,
            "sum_of_reoccurring_data_points" | "value__sum_of_reoccurring_data_points" => Feature::SumOfReoccurringDataPoints,
            "spectral_centroid" | "torque_Centroid" => Feature::SpectralCentroid,
            "spectral_distance" | "torque_Spectral distance" => Feature::SpectralDistance,
            "spectral_decrease" | "torque_Spectral decrease" => Feature::SpectralDecrease,
            "spectral_slope" | "torque_Spectral slope" => Feature::SpectralSlope,
            "signal_distance" | "torque_Signal distance" => Feature::SignalDistance,
            "human_range_energy" | "torque_Human range energy" => Feature::HumanRangeEnergy(100.0f32.to_bits()), // Default fs=100
            e => {
                if let Some(arg) = e.strip_prefix("paa-") {
                    let params: Vec<&str> = arg.split('-').collect();
                    if params.len() == 2
                        && let (Ok(n), Ok(m)) = (params[0].parse::<u16>(), params[1].parse::<u16>())
                    {
                        return Feature::Paa(n, m);
                    }
                } else if let Some(arg) = e.strip_prefix("c3-") {
                    let param = arg.trim();
                    if let Ok(n) = param.parse::<u16>() {
                        return Feature::C3(n);
                    }
                } else if e.contains("c3__lag_") {
                    if let Some(pos) = e.find("lag_") {
                        if let Ok(n) = e[pos+4..].parse::<u16>() {
                            return Feature::C3(n);
                        }
                    }
                } else if let Some(arg) = e.strip_prefix("autocorr-") {
                    if let Ok(n) = arg.parse::<u16>() {
                        return Feature::Autocorr(n);
                    }
                } else if let Some(arg) = e.strip_prefix("partial_autocorr-") {
                    if let Ok(n) = arg.parse::<u16>() {
                        return Feature::PartialAutocorr(n);
                    }
                } else if let Some(arg) = e.strip_prefix("time_reversal_asymmetry-") {
                    if let Ok(n) = arg.parse::<u16>() {
                        return Feature::TimeReversalAsymmetry(n);
                    }
                } else if e.contains("time_reversal_asymmetry_statistic__lag_") {
                    if let Some(pos) = e.find("lag_") {
                        if let Ok(n) = e[pos+4..].parse::<u16>() {
                            return Feature::TimeReversalAsymmetry(n);
                        }
                    }
                } else if let Some(arg) = e.strip_prefix("fft_coeff-") {
                    let params: Vec<&str> = arg.split('-').collect();
                    if params.len() == 2 {
                        if let Ok(coeff) = params[0].parse::<u16>() {
                            let attr = match params[1] {
                                "real" => FftAttr::Real,
                                "imag" => FftAttr::Imag,
                                "abs" => FftAttr::Abs,
                                "angle" => FftAttr::Angle,
                                _ => panic!("Unknown FFT attribute: {}", params[1]),
                            };
                            return Feature::FftCoefficient(coeff, attr);
                        }
                    }
                } else if let Some(arg) = e.strip_prefix("approx_entropy-") {
                    let params: Vec<&str> = arg.split('-').collect();
                    if params.len() == 2 {
                        if let (Ok(m), Ok(r)) = (params[0].parse::<u8>(), params[1].parse::<f32>()) {
                            return Feature::ApproxEntropy(m, r.to_bits());
                        }
                    }
                } else if let Some(arg) = e.strip_prefix("agg_linear_trend-") {
                    let params: Vec<&str> = arg.split('-').collect();
                    if params.len() == 3 {
                        let attr = match params[0] {
                            "slope" => AggAttr::Slope,
                            "intercept" => AggAttr::Intercept,
                            "stderr" => AggAttr::Stderr,
                            "rvalue" => AggAttr::RValue,
                            "pvalue" => AggAttr::PValue,
                            _ => panic!("Unknown Agg attribute: {}", params[0]),
                        };
                        if let Ok(chunk_len) = params[1].parse::<u16>() {
                            let func = match params[2] {
                                "max" => AggFunc::Max,
                                "min" => AggFunc::Min,
                                "mean" => AggFunc::Mean,
                                "var" => AggFunc::Var,
                                _ => panic!("Unknown Agg function: {}", params[2]),
                            };
                            return Feature::AggLinearTrend(attr, chunk_len, func);
                        }
                    }
                } else if e.contains("agg_linear_trend__attr_") {
                    // value__agg_linear_trend__attr_"slope"__chunk_len_5__f_agg_"mean"
                    let attr = if e.contains("attr_\"slope\"") { AggAttr::Slope }
                              else if e.contains("attr_\"intercept\"") { AggAttr::Intercept }
                              else { AggAttr::Slope };
                    
                    let chunk_len = if let Some(pos) = e.find("chunk_len_") {
                        let sub = &e[pos+10..];
                        let end = sub.find("__").unwrap_or(sub.len());
                        sub[..end].parse::<u16>().unwrap_or(5)
                    } else { 5 };

                    let func = if e.contains("f_agg_\"mean\"") { AggFunc::Mean }
                               else if e.contains("f_agg_\"var\"") { AggFunc::Var }
                               else if e.contains("f_agg_\"max\"") { AggFunc::Max }
                               else if e.contains("f_agg_\"min\"") { AggFunc::Min }
                               else { AggFunc::Mean };
                    
                    return Feature::AggLinearTrend(attr, chunk_len, func);
                } else if let Some(arg) = e.strip_prefix("quantile-") {
                    if let Ok(q) = arg.parse::<f32>() {
                        return Feature::Quantile(q.to_bits());
                    }
                } else if e.contains("value__quantile__q_") {
                    if let Some(pos) = e.find("q_") {
                        if let Ok(q) = e[pos+2..].parse::<f32>() {
                            return Feature::Quantile(q.to_bits());
                        }
                    }
                } else if let Some(arg) = e.strip_prefix("index_mass_quantile-") {
                    if let Ok(q) = arg.parse::<f32>() {
                        return Feature::IndexMassQuantile(q.to_bits());
                    }
                } else if e.contains("value__index_mass_quantile__q_") {
                    if let Some(pos) = e.find("q_") {
                        if let Ok(q) = e[pos+2..].parse::<f32>() {
                            return Feature::IndexMassQuantile(q.to_bits());
                        }
                    }
                } else if let Some(arg) = e.strip_prefix("max_langevin_fixed_point-") {
                    let params: Vec<&str> = arg.split('-').collect();
                    if params.len() == 2 {
                        if let (Ok(m), Ok(r)) = (params[0].parse::<u8>(), params[1].parse::<f32>()) {
                            return Feature::MaxLangevinFixedPoint(m, r.to_bits());
                        }
                    }
                } else if e.contains("value__max_langevin_fixed_point__m_") {
                    // value__max_langevin_fixed_point__m_3__r_30
                    let m = if let Some(pos) = e.find("m_") {
                        let sub = &e[pos+2..];
                        let end = sub.find("__").unwrap_or(sub.len());
                        sub[..end].parse::<u8>().unwrap_or(3)
                    } else { 3 };
                    let r = if let Some(pos) = e.find("r_") {
                        let sub = &e[pos+2..];
                        let end = sub.find("__").unwrap_or(sub.len());
                        sub[..end].parse::<f32>().unwrap_or(30.0)
                    } else { 30.0 };
                    return Feature::MaxLangevinFixedPoint(m, r.to_bits());
                } else if e.contains("value__mean_n_absolute_max__number_of_maxima_") {
                    if let Some(pos) = e.find("number_of_maxima_") {
                        if let Ok(n) = e[pos+17..].parse::<u16>() {
                            return Feature::MeanNAbsoluteMax(n);
                        }
                    }
                } else if let Some(arg) = e.strip_prefix("mean_n_absolute_max-") {
                    if let Ok(n) = arg.parse::<u16>() {
                        return Feature::MeanNAbsoluteMax(n);
                    }
                } else if let Some(arg) = e.strip_prefix("wavelet-") {
                    let params: Vec<&str> = arg.split('-').collect();
                    if params.len() == 2 {
                        if let (Ok(w), Ok(f)) =
                            (params[0].parse::<u16>(), params[1].parse::<u16>())
                        {
                            return Feature::WaveletFeatures(w, f);
                        }
                    }
                } else if e.contains("torque_Wavelet") {
                    // torque_Wavelet absolute mean_104.17Hz
                    // torque_Wavelet variance_104.17Hz
                    let f_type = if e.contains("absolute mean") { 0 } else { 1 };
                    let freq = if let Some(pos) = e.find('_') {
                        if let Some(end) = e.find("Hz") {
                            e[pos+1..end].parse::<f32>().unwrap_or(0.0)
                        } else { 0.0 }
                    } else { 0.0 };
                    return Feature::WaveletFeatures(freq.to_bits() as u16, f_type); // Hacky storage
                } else if let Some(arg) = e.strip_prefix("spectrogram-") {
                    let params: Vec<&str> = arg.split('-').collect();
                    if params.len() == 2 {
                        if let (Ok(t), Ok(f)) =
                            (params[0].parse::<u16>(), params[1].parse::<u16>())
                        {
                            return Feature::SpectrogramCoefficients(t, f);
                        }
                    }
                } else if e.contains("torque_Spectrogram mean coefficient_") {
                    // torque_Spectrogram mean coefficient_322.58Hz
                    if let Some(pos) = e.rfind('_') {
                        if let Some(end) = e.find("Hz") {
                            let freq = e[pos+1..end].parse::<f32>().unwrap_or(0.0);
                            return Feature::SpectrogramCoefficients(0, freq.to_bits() as u16); // Hacky
                        }
                    }
                }
                panic!("Unknown feature: {}", e);
            }
        }
    }
}

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
            Feature::Kurtosis => "kurtosis".to_string(),
            Feature::Mad => "mad".to_string(),
            Feature::Iqr => "iqr".to_string(),
            Feature::Entropy => "entropy".to_string(),
            Feature::Energy => "energy".to_string(),
            Feature::Rms => "rms".to_string(),
            Feature::RootMeanSquare => "root_mean_square".to_string(),
            Feature::ZeroCrossingRate => "zero_crossing_rate".to_string(),
            Feature::PeakCount => "peak_count".to_string(),
            Feature::AutocorrLag1 => "autocorr_lag1".to_string(),
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
            Feature::MeanNAbsoluteMax(n) => format!("mean_n_absolute_max-{}", n),
            Feature::HumanRangeEnergy(fs_bits) => format!("human_range_energy-{}", f32::from_bits(*fs_bits)),
            Feature::SpectralCentroid => "spectral_centroid".to_string(),
            Feature::SpectralDistance => "spectral_distance".to_string(),
            Feature::SpectralDecrease => "spectral_decrease".to_string(),
            Feature::SpectralSlope => "spectral_slope".to_string(),
            Feature::SignalDistance => "signal_distance".to_string(),
            Feature::WaveletFeatures(w, f) => format!("wavelet-{}-{}", w, f),
            Feature::SpectrogramCoefficients(t, f) => format!("spectrogram-{}-{}", t, f),
        }
    }
}

