use super::feature::{AggAttr, AggFunc, Feature, FftAttr};

// ─── FromStr ────────────────────────────────────────────────────────────────

impl std::str::FromStr for Feature {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // 1. Exact matches (simple features)
        match s {
            "sample_entropy" => return Ok(Feature::SampleEntropy),
            "total_sum" | "value__sum_values" => return Ok(Feature::TotalSum),
            "mean" | "value__mean" => return Ok(Feature::Mean),
            "variance" | "value__variance" => return Ok(Feature::Variance),
            "std" | "std_dev" | "value__standard_deviation" => return Ok(Feature::Std),
            "min" | "min_value" | "value__minimum" => return Ok(Feature::Min),
            "max" | "max_value" | "value__maximum" => return Ok(Feature::Max),
            "median" | "value__median" => return Ok(Feature::Median),
            "skew" | "skewness" | "value__skewness" => return Ok(Feature::Skew),
            "kurtosis" | "value__kurtosis" | "unbiased_fisher_kurtosis" => {
                return Ok(Feature::UnbiasedFisherKurtosis);
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
                return Ok(Feature::FirstLocMax);
            }
            "last_loc_max" | "value__last_location_of_maximum" => return Ok(Feature::LastLocMax),
            "first_loc_min" | "value__first_location_of_minimum" => {
                return Ok(Feature::FirstLocMin);
            }
            "last_loc_min" | "value__last_location_of_minimum" => return Ok(Feature::LastLocMin),
            "benford_correlation" | "value__benford_correlation" => {
                return Ok(Feature::BenfordCorrelation);
            }
            "sum_of_reoccurring_values" | "value__sum_of_reoccurring_values" => {
                return Ok(Feature::SumOfReoccurringValues);
            }
            "sum_of_reoccurring_data_points" | "value__sum_of_reoccurring_data_points" => {
                return Ok(Feature::SumOfReoccurringDataPoints);
            }
            "percentage_of_reoccurring_datapoints_to_all_datapoints"
            | "value__percentage_of_reoccurring_datapoints_to_all_datapoints" => {
                return Ok(Feature::PercentageOfReoccurringDatapointsToAllDatapoints);
            }
            "percentage_of_reoccurring_values_to_all_values"
            | "value__percentage_of_reoccurring_values_to_all_values" => {
                return Ok(Feature::PercentageOfReoccurringValuesToAllValues);
            }
            "ratio_value_number_to_time_series_length"
            | "value__ratio_value_number_to_time_series_length" => {
                return Ok(Feature::RatioValueNumberToTimeSeriesLength);
            }
            "length" | "value__length" => return Ok(Feature::Length),
            "variance_larger_than_standard_deviation"
            | "value__variance_larger_than_standard_deviation" => {
                return Ok(Feature::VarianceLargerThanStandardDeviation);
            }
            "spectral_centroid" | "torque_Centroid" => return Ok(Feature::SpectralCentroid),
            "spectral_distance" | "torque_Spectral distance" => {
                return Ok(Feature::SpectralDistance);
            }
            "spectral_decrease" | "torque_Spectral decrease" => {
                return Ok(Feature::SpectralDecrease);
            }
            "spectral_slope" | "torque_Spectral slope" => return Ok(Feature::SpectralSlope),
            "spectral_spread" | "torque_Spectral spread" => return Ok(Feature::SpectralSpread),
            "spectral_entropy" | "torque_Spectral entropy" => return Ok(Feature::SpectralEntropy),
            "spectral_roll_on" | "torque_Spectral roll-on" => return Ok(Feature::SpectralRollOn),
            "spectral_roll_off" | "torque_Spectral roll-off" => {
                return Ok(Feature::SpectralRollOff);
            }
            "spectral_spread" | "torque_Spectral spread" => return Ok(Feature::SpectralSpread),
            "spectral_skewness" | "torque_Spectral skewness" => {
                return Ok(Feature::SpectralSkewness);
            }
            "spectral_kurtosis" | "torque_Spectral kurtosis" => {
                return Ok(Feature::SpectralKurtosis);
            }
            "signal_distance" | "torque_Signal distance" => return Ok(Feature::SignalDistance),
            "human_range_energy" | "torque_Human range energy" => {
                return Ok(Feature::HumanRangeEnergy(100.0f32.to_bits())); // Default fs=100
            }
            "pk_pk_distance" => return Ok(Feature::PkPkDistance),
            "zero_cross" | "torque_Zero_crossing_rate" => return Ok(Feature::ZeroCross),
            "wavelet_entropy" => return Ok(Feature::WaveletEntropy),
            "max_power_spectrum" => return Ok(Feature::MaxPowerSpectrum),
            "mean_second_derivative_central" => return Ok(Feature::MeanSecondDerivativeCentral),
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
    if let Some(arg) = s.strip_prefix("lpcc-") {
        return Some(Feature::Lpcc(arg.trim().parse().ok()?));
    }
    if let Some(arg) = s.strip_prefix("mfcc-") {
        return Some(Feature::Mfcc(arg.trim().parse().ok()?));
    }
    if let Some(arg) = s.strip_prefix("wavelet_energy-") {
        return Some(Feature::WaveletEnergy(arg.trim().parse().ok()?));
    }
    if let Some(arg) = s.strip_prefix("paa-") {
        let (n, m) = arg.split_once('-')?;
        let total = n.parse::<u16>().ok()?;
        let index = m.parse::<u16>().ok()?;
        if index >= total {
            return None;
        }
        return Some(Feature::Paa(total, index));
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
    if let Some(arg) = s.strip_prefix("linear_trend-") {
        let attr = parse_agg_attr(arg)?;
        return Some(Feature::LinearTrend(attr));
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
    if let Some(arg) = s.strip_prefix("ar_coefficient-") {
        let (k_s, p_s) = arg.split_once('-')?;
        let k: u16 = k_s.parse().ok()?;
        let p: u16 = p_s.parse().ok()?;
        return Some(Feature::ArCoefficient(k, p));
    }
    if let Some(arg) = s.strip_prefix("friedrich_coefficients-") {
        let parts: Vec<&str> = arg.split('-').collect();
        if parts.len() == 3 {
            let m: u8 = parts[0].parse().ok()?;
            let r: f32 = parts[1].parse().ok()?;
            let coeff: u16 = parts[2].parse().ok()?;
            return Some(Feature::FriedrichCoefficients(m, r.to_bits(), coeff));
        }
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
    if let Some(arg) = s.strip_prefix("binned_entropy__max_bins_")
        && let Ok(bins) = arg.parse::<u32>()
    {
        return Some(Feature::BinnedEntropy(bins));
    }
    if let Some(arg) = s.strip_prefix("ratio_beyond_r_sigma-") {
        let r: f32 = arg.parse().ok()?;
        return Some(Feature::RatioBeyondRSigma(r.to_bits()));
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
    if s.contains("linear_trend__attr_") && !s.contains("agg_linear_trend__attr_") {
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
        return Some(Feature::LinearTrend(attr));
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
    if s.contains("torque_Spectrogram mean coefficient_")
        && let Some(pos) = s.rfind('_')
        && let Some(end) = s.find("Hz")
    {
        let freq = s[pos + 1..end].parse::<f32>().unwrap_or(0.0);
        return Some(Feature::SpectrogramCoefficients(0, freq.to_bits()));
    }
    if let Some(arg) = s.strip_prefix("spkt_welch_density__coeff_") {
        let coeff: u16 = arg.parse().ok()?;
        return Some(Feature::SpktWelchDensity(coeff));
    }
    if let Some(arg) = s.strip_prefix("number_cwt_peaks__n_") {
        let n: u16 = arg.parse().ok()?;
        return Some(Feature::NumberCwtPeaks(n));
    }
    if s.starts_with("cwt_coefficients__coeff_") {
        if let Some(coeff_end) = s.find("__w_") {
            let coeff_str = &s[24..coeff_end];
            let coeff = coeff_str.parse::<u16>().ok()?;
            if let Some(w_end) = s.find("__widths_") {
                let w_str = &s[coeff_end + 4..w_end];
                let w = w_str.parse::<u16>().ok()?;
                let widths_str = &s[w_end + 9..];
                let cleaned = widths_str
                    .replace("(", "")
                    .replace(")", "")
                    .replace(" ", "");
                let parts: Vec<&str> = cleaned.split(',').collect();
                let mut widths = [0u16; 8];
                let len = parts.len().min(8) as u8;
                for i in 0..(len as usize) {
                    if let Ok(val) = parts[i].parse::<u16>() {
                        widths[i] = val;
                    }
                }
                return Some(Feature::CwtCoefficients(widths, len, coeff, w));
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
    if let Some(arg) = s.strip_prefix("binned_entropy__max_bins_")
        && let Ok(bins) = arg.parse::<u32>()
    {
        return Some(Feature::BinnedEntropy(bins));
    }
    if s.contains("value__ratio_beyond_r_sigma__r_") {
        let pos = s.find("r_")?;
        let r: f32 = s[pos + 2..].parse().ok()?;
        return Some(Feature::RatioBeyondRSigma(r.to_bits()));
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
            Feature::SampleEntropy => "sample_entropy".to_string(),
            Feature::BinnedEntropy(bins) => format!("binned_entropy__max_bins_{}", bins),
            Feature::ApproxEntropy(m, r_bits) => {
                format!("approx_entropy-{}-{}", m, f32::from_bits(*r_bits))
            }
            Feature::LinearTrend(attr) => {
                let attr_str = match attr {
                    AggAttr::Slope => "slope",
                    AggAttr::Intercept => "intercept",
                    AggAttr::Stderr => "stderr",
                    AggAttr::RValue => "rvalue",
                    AggAttr::PValue => "pvalue",
                };
                format!("linear_trend-{}", attr_str)
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
            Feature::PercentageOfReoccurringDatapointsToAllDatapoints => {
                "percentage_of_reoccurring_datapoints_to_all_datapoints".to_string()
            }
            Feature::PercentageOfReoccurringValuesToAllValues => {
                "percentage_of_reoccurring_values_to_all_values".to_string()
            }
            Feature::RatioValueNumberToTimeSeriesLength => {
                "ratio_value_number_to_time_series_length".to_string()
            }
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
            Feature::SpectralSpread => "spectral_spread".to_string(),
            Feature::SpectralEntropy => "spectral_entropy".to_string(),
            Feature::SpectralRollOn => "spectral_roll_on".to_string(),
            Feature::SpectralRollOff => "spectral_roll_off".to_string(),
            Feature::SpectralSpread => "spectral_spread".to_string(),
            Feature::SpectralSkewness => "spectral_skewness".to_string(),
            Feature::SpectralKurtosis => "spectral_kurtosis".to_string(),
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
            Feature::RatioBeyondRSigma(r_bits) => {
                format!("ratio_beyond_r_sigma-{}", f32::from_bits(*r_bits))
            }
            Feature::HasDuplicateMax => "has_duplicate_max".to_string(),
            Feature::HasDuplicateMin => "has_duplicate_min".to_string(),
            Feature::HasDuplicate => "has_duplicate".to_string(),
            Feature::PkPkDistance => "pk_pk_distance".to_string(),
            Feature::ZeroCross => "zero_cross".to_string(),
            Feature::MaxPowerSpectrum => "max_power_spectrum".to_string(),
            Feature::Lpcc(idx) => format!("lpcc-{}", idx),
            Feature::Mfcc(idx) => format!("mfcc-{}", idx),
            Feature::WaveletEnergy(idx) => format!("wavelet_energy-{}", idx),
            Feature::WaveletEntropy => "wavelet_entropy".to_string(),

            Feature::SpktWelchDensity(coeff) => format!("spkt_welch_density__coeff_{}", coeff),
            Feature::CwtCoefficients(widths, len, coeff, w) => {
                let mut w_str = String::from("(");
                for i in 0..*len as usize {
                    if i > 0 {
                        w_str.push_str(", ");
                    }
                    w_str.push_str(&widths[i].to_string());
                }
                w_str.push(')');
                format!(
                    "cwt_coefficients__coeff_{}__w_{}__widths_{}",
                    coeff, w, w_str
                )
            }
            Feature::NumberCwtPeaks(n) => format!("number_cwt_peaks__n_{}", n),
            Feature::ArCoefficient(k, p) => format!("ar_coefficient-{}-{}", k, p),
            Feature::FriedrichCoefficients(m, r_bits, coeff) => {
                format!(
                    "friedrich_coefficients-{}-{}-{}",
                    m,
                    f32::from_bits(*r_bits),
                    coeff
                )
            }
        }
    }
}
