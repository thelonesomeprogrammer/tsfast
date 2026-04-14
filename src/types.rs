#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct FastBitArray(pub u64);

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
        let mut mask = 0u64;
        for i in indices {
            mask |= 1 << i;
        }
        self.0 |= mask;
    }

    #[inline]
    pub fn unset_batch<const N: usize>(&mut self, indices: [usize; N]) {
        let mut mask = 0u64;
        for i in indices {
            mask |= 1 << i;
        }
        self.0 &= !mask;
    }

    #[inline]
    pub fn any<const N: usize>(&self, indices: [usize; N]) -> bool {
        let mut mask = 0u64;
        for i in indices {
            mask |= 1 << i;
        }
        (self.0 & mask) != 0
    }

    #[inline]
    pub fn all<const N: usize>(&self, indices: [usize; N]) -> bool {
        let mut mask = 0u64;
        for i in indices {
            mask |= 1 << i;
        }
        (self.0 & mask) == mask
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
}

impl From<String> for Feature {
    fn from(s: String) -> Self {
        match s.as_str() {
            "total_sum" => Feature::TotalSum,
            "mean" => Feature::Mean,
            "variance" => Feature::Variance,
            "std" | "std_dev" => Feature::Std,
            "min" | "min_value" => Feature::Min,
            "max" | "max_value" => Feature::Max,
            "median" => Feature::Median,
            "skew" | "skewness" => Feature::Skew,
            "kurtosis" => Feature::Kurtosis,
            "mad" => Feature::Mad,
            "iqr" => Feature::Iqr,
            "entropy" => Feature::Entropy,
            "energy" => Feature::Energy,
            "rms" => Feature::Rms,
            "root_mean_square" => Feature::RootMeanSquare,
            "zero_crossing_rate" => Feature::ZeroCrossingRate,
            "peak_count" => Feature::PeakCount,
            "autocorr_lag1" => Feature::AutocorrLag1,
            "mean_abs_change" => Feature::MeanAbsChange,
            "mean_change" => Feature::MeanChange,
            "cid_ce" => Feature::CidCe,
            "slope" => Feature::Slope,
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
            Feature::ZeroCrossingMean => "zero_crossing_mean".to_string(),
            Feature::ZeroCrossingStd => "zero_crossing_std".to_string(),
            Feature::C3(lag) => format!("c3-{}", lag),
            Feature::Paa(total, index) => format!("paa-{}-{}", total, index),
            _ => "unknown".to_string(),
        }
    }
}
