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
        const SUM                  = 1 << 0;
        const MEAN                 = 1 << 1;
        const VARIANCE             = 1 << 2;
        const STD                  = 1 << 3;
        const MIN                  = 1 << 4;
        const MAX                  = 1 << 5;
        const MEDIAN               = 1 << 6;
        const SKEW                 = 1 << 7;   // sum_cubes
        const KURTOSIS             = 1 << 8;   // sum_quads
        const MAD                  = 1 << 9;
        const IQR                  = 1 << 10;
        const ENTROPY              = 1 << 11;
        const ENERGY               = 1 << 12;  // sum of squares
        const RMS                  = 1 << 13;
        const ROOT_MEAN_SQ         = 1 << 14;
        const ZERO_CROSS           = 1 << 15;
        const PEAKS                = 1 << 16;
        const AUTOCORR_LAG1        = 1 << 17;  // lag-1 product
        const MAC                  = 1 << 18;  // mean abs change
        const MC                   = 1 << 19;  // mean change
        const CID_CE               = 1 << 20;
        const SLOPE                = 1 << 21;  // sum_ix
        const INTERCEPT            = 1 << 22;
        const PAA                  = 1 << 23;
        const ABS_SUM_CHG          = 1 << 24;
        const CNT_ABOVE_MEAN       = 1 << 25;
        const CNT_BELOW_MEAN       = 1 << 26;
        const STRIKE_ABOVE         = 1 << 27;
        const STRIKE_BELOW         = 1 << 28;
        const VAR_COEFF            = 1 << 29;
        const C3                   = 1 << 30;
        const AUC                  = 1 << 31;
        const SLOPE_SIGN_CHG       = 1 << 32;
        const TURNING_PTS          = 1 << 33;
        const ZC_STATS             = 1 << 34;  // zero-crossing mean
        const ZC_STD               = 1 << 35;
        const ZC_INDICES           = 1 << 36;

        // ── Pass-2 / post-processing gates ─────────────────────
        const NEEDS_SORT           = 1 << 37;
        const ABS_MAX              = 1 << 38;
        const FIRST_LOC_MAX        = 1 << 39;
        const LAST_LOC_MAX         = 1 << 40;
        const FIRST_LOC_MIN        = 1 << 41;
        const LAST_LOC_MIN         = 1 << 42;
        const FULL_AUTOCORR        = 1 << 43;  // Wiener-Khinchin FFT autocorr
        const PACF                 = 1 << 44;
        const TRA                  = 1 << 45;  // time reversal asymmetry

        // ── FFT / spectral / complex feature gates ─────────────
        const FFT_COEFF            = 1 << 46;
        const APPROX_ENT           = 1 << 47;
        const AGG_LIN_TREND        = 1 << 48;
        const QUANTILE             = 1 << 49;
        const IDX_MASS_Q           = 1 << 50;
        const BENFORD              = 1 << 51;
        const LANGEVIN             = 1 << 52;
        const REOCCUR_VAL          = 1 << 53;
        const SPEC_CENTROID        = 1 << 54;
        const SPEC_DISTANCE        = 1 << 55;
        const SPEC_DECREASE        = 1 << 56;
        const SPEC_SLOPE           = 1 << 57;
        const SIG_DISTANCE         = 1 << 58;
        const WAVELET              = 1 << 59;
        const SPECTROGRAM          = 1 << 60;
        const ABS_SUM              = 1 << 61;
        const REOCCUR_DP           = 1 << 62;
        const MEAN_N_ABS_MAX       = 1 << 63;
        const HUMAN_RANGE_E        = 1 << 64;
        const LENGTH               = 1 << 65;
        const VAR_GT_STD           = 1 << 66;
        const HAS_DUPLICATE        = 1 << 67;
        const HAS_DUP_MAX          = 1 << 68;
        const HAS_DUP_MIN          = 1 << 69;
        const REOCCUR_RATIOS       = 1 << 70;
        const SPEC_SPREAD          = 1 << 71;
        const SPEC_SKEWNESS        = 1 << 72;
        const SPEC_KURTOSIS        = 1 << 73;
        const RATIO_BEYOND_R_SIGMA = 1 << 74;
        const SAMP_ENT             = 1 << 75;
        const BINNED_ENT           = 1 << 76;
        const SPEC_ROLLON          = 1 << 77;
        const SPEC_ROLLOFF         = 1 << 78;
        const AR_COEFF             = 1 << 79;
        const FRIEDRICH            = 1 << 80;
        const CWT                  = 1 << 81;
        const WELCH                = 1 << 82;
        const SPEC_ENTROPY         = 1 << 83;
        const MFCC                 = 1 << 84;
        const CWT_MEXH             = 1 << 85;
        const QUERY_SIMILARITY     = 1 << 86;
        const MATRIX_PROFILE       = 1 << 87;
        const MEDIAN_ABS_DEV       = 1 << 88;
        const CALC_CENTROID        = 1 << 89;
        const ADF                  = 1 << 90;
        const LPCC                 = 1 << 91;
        const TROUGHS              = 1 << 92;

        // ── Composite masks (zero bit-cost) ────────────────────
        const ANY_FFT = Self::FFT_COEFF.bits()
            | Self::HUMAN_RANGE_E.bits()
            | Self::SPEC_CENTROID.bits()
            | Self::SPEC_DISTANCE.bits()
            | Self::SPEC_DECREASE.bits()
            | Self::SPEC_SLOPE.bits()
            | Self::SPECTROGRAM.bits()
            | Self::MFCC.bits()
            | Self::LPCC.bits()
            | Self::SPEC_SPREAD.bits()
            | Self::SPEC_ENTROPY.bits()
            | Self::SPECTROGRAM.bits()
            | Self::WELCH.bits()
            | Self::SPEC_ROLLON.bits()
            | Self::SPEC_ROLLOFF.bits()
            | Self::SPEC_SKEWNESS.bits()
            | Self::SPEC_KURTOSIS.bits()
            | Self::SPECTROGRAM.bits();

        const ANY_DIFF = Self::ZERO_CROSS.bits()
            | Self::AUTOCORR_LAG1.bits()
            | Self::MAC.bits()
            | Self::MC.bits()
            | Self::CID_CE.bits()
            | Self::AUC.bits();
    }
}
