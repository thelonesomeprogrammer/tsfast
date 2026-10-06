use super::feature::{AdfAttr, AggAttr, AggFunc, Feature, FftAttr};

// ─── Unit features: one line each ──────────────────────────────────────────
//
// `Variant => "canonical_name", ["alias", ...];`
// The canonical name is the output column name and is always accepted when
// parsing; aliases are additionally accepted (tsfresh/TSFEL/torque spellings).
// Features with parameters are parsed in `from_str` / named in `name()` below.

macro_rules! unit_features {
    ($($variant:ident => $name:literal $(, [$($alias:literal),* $(,)?])?;)*) => {
        fn parse_unit(s: &str) -> Option<Feature> {
            match s {
                $($name $($(| $alias)*)? => Some(Feature::$variant),)*
                _ => None,
            }
        }

        /// Every unit variant, for the coverage/round-trip tests.
        #[cfg(test)]
        pub(crate) const UNIT_FEATURES: &[Feature] = &[$(Feature::$variant),*];

        impl Feature {
            fn unit_name(&self) -> Option<&'static str> {
                match self {
                    $(Feature::$variant => Some($name),)*
                    _ => None,
                }
            }
        }
    };
}

unit_features! {
    TotalSum => "total_sum", ["value__sum_values"];
    Mean => "mean", ["value__mean"];
    Variance => "variance", ["value__variance"];
    Std => "std_dev", ["std", "value__standard_deviation"];
    Min => "min_value", ["min", "value__minimum"];
    Max => "max_value", ["max", "value__maximum"];
    Median => "median", ["value__median"];
    MedianAbsDeviation => "median_abs_deviation";
    Skew => "skewness", ["skew", "value__skewness"];
    BiasedSkew => "biased_skewness";
    UnbiasedFisherKurtosis => "kurtosis", ["value__kurtosis", "unbiased_fisher_kurtosis"];
    BiasedFisherKurtosis => "biased_fisher_kurtosis";
    Mad => "mad", ["mean_abs_deviation"];
    Iqr => "iqr";
    Entropy => "entropy";
    SampleEntropy => "sample_entropy";
    HiguchiFd => "higuchi_fd";
    Dfa => "dfa";
    HurstExponent => "hurst_exponent";
    MaximumFractalLength => "maximum_fractal_length";
    Energy => "energy", ["torque_Absolute energy"];
    Rms => "rms";
    RootMeanSquare => "root_mean_square";
    ZeroCrossingRate => "zero_crossing_rate";
    PeakCount => "peak_count";
    NegativeTurning => "negative_turning";
    PositiveTurning => "positive_turning";
    AutocorrLag1 => "autocorr_lag1", ["centered_autocorr_lag1"];
    AutocorrFirst1e => "autocorrelation", ["autocorr_first_1e"];
    MeanAbsChange => "mean_abs_change";
    MeanChange => "mean_change", ["mean_diff", "torque_Mean diff"];
    MedianDiff => "median_diff";
    MedianAbsDiff => "median_abs_diff";
    CidCe => "cid_ce";
    Slope => "slope", ["torque_Slope"];
    Intercept => "intercept";
    AbsSumChange => "abs_sum_change";
    CountAboveMean => "count_above_mean";
    CountBelowMean => "count_below_mean";
    LongestStrikeAboveMean => "longest_strike_above_mean";
    LongestStrikeBelowMean => "longest_strike_below_mean";
    VariationCoefficient => "variation_coefficient";
    Auc => "auc";
    SlopeSignChange => "slope_sign_change";
    TurningPoints => "turning_points";
    ZeroCrossingMean => "zero_crossing_mean";
    ZeroCrossingStd => "zero_crossing_std";
    AbsMax => "abs_max", ["value__absolute_maximum"];
    FirstLocMax => "first_loc_max", ["value__first_location_of_maximum"];
    LastLocMax => "last_loc_max", ["value__last_location_of_maximum"];
    FirstLocMin => "first_loc_min", ["value__first_location_of_minimum"];
    LastLocMin => "last_loc_min", ["value__last_location_of_minimum"];
    BenfordCorrelation => "benford_correlation", ["value__benford_correlation"];
    SumOfReoccurringValues => "sum_of_reoccurring_values", ["value__sum_of_reoccurring_values"];
    SumOfReoccurringDataPoints => "sum_of_reoccurring_data_points", ["value__sum_of_reoccurring_data_points"];
    PercentageOfReoccurringDatapointsToAllDatapoints => "percentage_of_reoccurring_datapoints_to_all_datapoints", ["value__percentage_of_reoccurring_datapoints_to_all_datapoints"];
    PercentageOfReoccurringValuesToAllValues => "percentage_of_reoccurring_values_to_all_values", ["value__percentage_of_reoccurring_values_to_all_values"];
    RatioValueNumberToTimeSeriesLength => "ratio_value_number_to_time_series_length", ["value__ratio_value_number_to_time_series_length"];
    Length => "length", ["value__length"];
    VarianceLargerThanStandardDeviation => "variance_larger_than_standard_deviation", ["value__variance_larger_than_standard_deviation"];
    SpectralCentroid => "spectral_centroid", ["torque_Centroid"];
    SpectralDistance => "spectral_distance", ["torque_Spectral distance"];
    SpectralDecrease => "spectral_decrease", ["torque_Spectral decrease"];
    SpectralSlope => "spectral_slope", ["torque_Spectral slope"];
    SpectralSpread => "spectral_spread", ["torque_Spectral spread"];
    SpectralEntropy => "spectral_entropy", ["torque_Spectral entropy"];
    SpectralRollOn => "spectral_roll_on", ["torque_Spectral roll-on"];
    SpectralRollOff => "spectral_roll_off", ["torque_Spectral roll-off"];
    SpectralSkewness => "spectral_skewness", ["torque_Spectral skewness"];
    SpectralKurtosis => "spectral_kurtosis", ["torque_Spectral kurtosis"];
    SignalDistance => "signal_distance", ["torque_Signal distance"];
    PkPkDistance => "pk_pk_distance";
    ZeroCross => "zero_cross", ["torque_Zero_crossing_rate"];
    MaxPowerSpectrum => "max_power_spectrum";
    MeanSecondDerivativeCentral => "mean_second_derivative_central";
    HasDuplicateMax => "has_duplicate_max";
    HasDuplicateMin => "has_duplicate_min";
    HasDuplicate => "has_duplicate";
    WaveletEntropy => "wavelet_entropy";
}

// ─── FromStr ────────────────────────────────────────────────────────────────

impl std::str::FromStr for Feature {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // 1. Unit features (table below)
        if let Some(f) = parse_unit(s) {
            return Ok(f);
        }

        // 2. Exact matches with fixed parameters
        match s {
            name if name.starts_with("energy_ratio_by_chunks_num_segments_") => {
                let parts: Vec<&str> = name.split("__").collect();
                if parts.len() == 2 {
                    let p1 = parts[0].strip_prefix("energy_ratio_by_chunks_num_segments_");
                    let p2 = parts[1].strip_prefix("segment_focus_");
                    if let (Some(num_s), Some(focus_s)) = (p1, p2) {
                        if let (Ok(num), Ok(focus)) = (num_s.parse::<u16>(), focus_s.parse::<u16>())
                        {
                            if focus < num {
                                return Ok(Feature::EnergyRatioByChunks(num, focus));
                            }
                        }
                    }
                }
            }
            "augmented_dickey_fuller-teststat" => {
                return Ok(Feature::AugmentedDickeyFuller(AdfAttr::TestStat));
            }
            "augmented_dickey_fuller-pvalue" => {
                return Ok(Feature::AugmentedDickeyFuller(AdfAttr::PValue));
            }
            "augmented_dickey_fuller-usedlag" => {
                return Ok(Feature::AugmentedDickeyFuller(AdfAttr::UsedLag));
            }
            "human_range_energy" | "torque_Human range energy" => {
                return Ok(Feature::HumanRangeEnergy(100.0f32.to_bits())); // Default fs=100
            }
            _ => {}
        }

        // 3. Parameterized features (prefix-based)
        if let Some(f) = parse_parameterized(s) {
            return Ok(f);
        }

        // 4. Legacy tsfresh / torque format
        if let Some(f) = parse_legacy_format(s) {
            return Ok(f);
        }

        if s.starts_with("query_similarity_count-") {
            let parts: Vec<&str> = s.split("-").collect();
            if parts.len() == 3 {
                if let (Ok(l), Ok(t)) = (parts[1].parse::<u16>(), parts[2].parse::<f32>()) {
                    return Ok(Feature::QuerySimilarityCount(l, t.to_bits()));
                }
            }
        }
        if s.starts_with("matrix_profile-") {
            let parts: Vec<&str> = s.split("-").collect();
            if parts.len() == 3 {
                if let Ok(l) = parts[1].parse::<u16>() {
                    if let Some(agg) = parse_agg_func(parts[2]) {
                        return Ok(Feature::MatrixProfile(l, agg));
                    }
                }
            }
        }
        Err(format!("Unknown feature: {}", s))
    }
}

// ─── Parameterized parsers ──────────────────────────────────────────────────

fn parse_parameterized(s: &str) -> Option<Feature> {
    if let Some(arg) = s.strip_prefix("human_range_energy-") {
        let fs: f32 = arg.trim().parse().ok()?;
        if !(fs.is_finite() && fs > 0.0) {
            return None;
        }
        return Some(Feature::HumanRangeEnergy(fs.to_bits()));
    }
    if let Some(arg) = s.strip_prefix("ecdf_slope-") {
        let (p1, p2) = arg.split_once('-')?;
        let p_init: f32 = p1.parse().ok()?;
        let p_end: f32 = p2.parse().ok()?;
        if p_init > 0.0 && p_init < p_end && p_end <= 1.0 {
            return Some(Feature::EcdfSlope(p_init.to_bits(), p_end.to_bits()));
        } else {
            return None;
        }
    }
    if let Some(arg) = s.strip_prefix("ecdf_percentile_count-") {
        let p: f32 = arg.parse().ok()?;
        if p > 0.0 && p <= 1.0 {
            return Some(Feature::EcdfPercentileCount(p.to_bits()));
        } else {
            return None;
        }
    }
    if let Some(arg) = s.strip_prefix("ecdf_percentile-") {
        let p: f32 = arg.parse().ok()?;
        if p > 0.0 && p <= 1.0 {
            return Some(Feature::EcdfPercentile(p.to_bits()));
        } else {
            return None;
        }
    }
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
    if let Some(arg) = s.strip_prefix("agg_autocorrelation-") {
        let parts: Vec<&str> = arg.split('-').collect();
        if parts.len() != 2 {
            return None;
        }
        let func = parse_agg_func(parts[0])?;
        let maxlag: u16 = parts[1].parse().ok()?;
        return Some(Feature::AggAutocorrelation(func, maxlag));
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
        if chunk_len == 0 {
            return None;
        }
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
        // ar_coefficient-<order k>-<coefficient p>; p = 0 is the intercept.
        let k: u16 = k_s.parse().ok()?;
        let p: u16 = p_s.parse().ok()?;
        if k == 0 || p > k {
            return None;
        }
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
    if let Some(arg) = s.strip_prefix("ecdf-") {
        let d: u32 = arg.parse().ok()?;
        return Some(Feature::Ecdf(d));
    }
    if let Some(arg) = s.strip_prefix("calc_centroid-") {
        let fs: f32 = arg.parse().ok()?;
        return Some(Feature::CalcCentroid(fs.to_bits()));
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
    if s.contains("number_crossing_m__m_") {
        let pos = s.rfind("__m_")?;
        let m: f32 = s[pos + 4..].parse().ok()?;
        return Some(Feature::NumberCrossingM(m.to_bits()));
    }
    if s.contains("number_peaks__n_") {
        let pos = s.find("n_")?;
        let n: u16 = s[pos + 2..].parse().ok()?;
        return Some(Feature::NumberPeaks(n));
    }
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
        if chunk_len == 0 {
            return None;
        }
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
    if s.starts_with("change_quantiles-") {
        let parts: Vec<&str> = s.split('-').collect();
        if parts.len() == 5 {
            let ql = parts[1].parse::<f32>().unwrap_or(0.0).to_bits();
            let qh = parts[2].parse::<f32>().unwrap_or(1.0).to_bits();
            let isabs = parts[3].to_lowercase() == "true";
            let func = parse_agg_func(parts[4]).unwrap_or(AggFunc::Mean);
            return Some(Feature::ChangeQuantiles(ql, qh, isabs, func));
        }
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
    if let Some(attr_str) = s.strip_prefix("augmented_dickey_fuller-") {
        return match attr_str {
            "teststat" => Some(Feature::AugmentedDickeyFuller(AdfAttr::TestStat)),
            "pvalue" => Some(Feature::AugmentedDickeyFuller(AdfAttr::PValue)),
            "usedlag" => Some(Feature::AugmentedDickeyFuller(AdfAttr::UsedLag)),
            _ => None,
        };
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

    if let Some(arg) = s.strip_prefix("permutation_entropy-") {
        let parts: Vec<&str> = arg.split('-').collect();
        if parts.len() == 2 {
            if let (Ok(tau), Ok(dim)) = (parts[0].parse::<u32>(), parts[1].parse::<u32>()) {
                return Some(Feature::PermutationEntropy(tau, dim));
            }
        }
    }

    if let Some(arg) = s.strip_prefix("value_count-") {
        if let Ok(v) = arg.parse::<f32>() {
            return Some(Feature::ValueCount(v.to_bits()));
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
        if let Some(name) = self.unit_name() {
            return name.to_string();
        }
        match self {
            Feature::EcdfPercentile(p) => format!("ecdf_percentile-{}", f32::from_bits(*p)),
            Feature::EcdfPercentileCount(p) => format!("ecdf_percentile_count-{}", f32::from_bits(*p)),
            Feature::EcdfSlope(p_init, p_end) => format!("ecdf_slope-{}-{}", f32::from_bits(*p_init), f32::from_bits(*p_end)),
            Feature::EnergyRatioByChunks(num, focus) => format!(
                "energy_ratio_by_chunks_num_segments_{}__segment_focus_{}",
                num, focus
            ),
            Feature::NumberCrossingM(m) => format!("number_crossing_m__m_{}", f32::from_bits(*m)),
            Feature::NumberPeaks(n) => format!("number_peaks__n_{}", n),
            Feature::C3(lag) => format!("c3-{}", lag),
            Feature::Paa(total, index) => format!("paa-{}-{}", total, index),
            Feature::Autocorr(lag) => format!("autocorr-{}", lag),
            Feature::AggAutocorrelation(func, maxlag) => {
                let func_str = match func {
                    AggFunc::Max => "max",
                    AggFunc::Min => "min",
                    AggFunc::Mean => "mean",
                    AggFunc::Var => "var",
                };
                format!("agg_autocorrelation-{}-{}", func_str, maxlag)
            }
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
            Feature::BinnedEntropy(bins) => format!("binned_entropy__max_bins_{}", bins),
            Feature::ApproxEntropy(m, r_bits) => {
                format!("approx_entropy-{}-{}", m, f32::from_bits(*r_bits))
            }
            Feature::PermutationEntropy(tau, dim) => {
                format!("permutation_entropy-{}-{}", tau, dim)
            }
            Feature::ValueCount(val_bits) => {
                format!("value_count-{}", f32::from_bits(*val_bits))
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
            Feature::ChangeQuantiles(ql_bits, qh_bits, isabs, func) => {
                let func_str = match func {
                    AggFunc::Max => "max",
                    AggFunc::Min => "min",
                    AggFunc::Mean => "mean",
                    AggFunc::Var => "var",
                };
                format!(
                    "change_quantiles-{}-{}-{}-{}",
                    f32::from_bits(*ql_bits),
                    f32::from_bits(*qh_bits),
                    if *isabs { "True" } else { "False" },
                    func_str
                )
            }
            Feature::Quantile(q_bits) => format!("quantile-{}", f32::from_bits(*q_bits)),
            Feature::IndexMassQuantile(q_bits) => {
                format!("index_mass_quantile-{}", f32::from_bits(*q_bits))
            }
            Feature::MaxLangevinFixedPoint(m, r_bits) => {
                format!("max_langevin_fixed_point-{}-{}", m, f32::from_bits(*r_bits))
            }
            Feature::QuerySimilarityCount(l, t) => {
                format!("query_similarity_count-{}-{}", l, f32::from_bits(*t))
            }
            Feature::MatrixProfile(l, agg) => {
                let agg_str = match agg {
                    AggFunc::Min => "min",
                    AggFunc::Max => "max",
                    AggFunc::Mean => "mean",
                    AggFunc::Var => "var",
                };
                format!("matrix_profile-{}-{}", l, agg_str)
            }
            Feature::MeanNAbsoluteMax(n) => format!("mean_n_absolute_max-{}", n),
            Feature::HumanRangeEnergy(fs_bits) => {
                format!("human_range_energy-{}", f32::from_bits(*fs_bits))
            }
            Feature::WaveletFeatures(w_bits, f) => {
                format!("wavelet-{}-{}", f32::from_bits(*w_bits), f)
            }
            Feature::SpectrogramCoefficients(t, f_bits) => {
                format!("spectrogram-{}-{}", t, f32::from_bits(*f_bits))
            }
            Feature::LargeStandardDeviation(r_bits) => {
                format!("large_standard_deviation-{}", f32::from_bits(*r_bits))
            }
            Feature::SymmetryLooking(r_bits) => {
                format!("symmetry_looking-{}", f32::from_bits(*r_bits))
            }
            Feature::RatioBeyondRSigma(r_bits) => {
                format!("ratio_beyond_r_sigma-{}", f32::from_bits(*r_bits))
            }
            Feature::Lpcc(idx) => format!("lpcc-{}", idx),
            Feature::Ecdf(d) => format!("ecdf-{}", d),
            Feature::CalcCentroid(fs) => format!("calc_centroid-{}", f32::from_bits(*fs)),
            Feature::Mfcc(idx) => format!("mfcc-{}", idx),
            Feature::WaveletEnergy(idx) => format!("wavelet_energy-{}", idx),

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
            Feature::AugmentedDickeyFuller(attr) => match attr {
                AdfAttr::TestStat => "augmented_dickey_fuller-teststat".to_string(),
                AdfAttr::PValue => "augmented_dickey_fuller-pvalue".to_string(),
                AdfAttr::UsedLag => "augmented_dickey_fuller-usedlag".to_string(),
            },
            Feature::ArCoefficient(k, p) => format!("ar_coefficient-{}-{}", k, p),
            Feature::FriedrichCoefficients(m, r_bits, coeff) => {
                format!(
                    "friedrich_coefficients-{}-{}-{}",
                    m,
                    f32::from_bits(*r_bits),
                    coeff
                )
            }
            _ => unreachable!("unit variant {self:?} missing from unit_features! table"),
        }
    }
}
