use bitflags::bitflags;

// ─── Computation flags ──────────────────────────────────────────────────────
//
// Add a new flag by NAME ONLY, next to related ones. Never write a bit index:
// a flag's bit is its position in this list, so flags added in parallel PRs
// can't collide. Bit values are internal and may change between builds.

macro_rules! compute_flags {
    ($($(#[$meta:meta])* $name:ident,)*) => {
        /// Implicit discriminants (0, 1, 2, …) give each flag its bit index.
        #[allow(non_camel_case_types, clippy::upper_case_acronyms, dead_code)]
        #[repr(u8)]
        enum ComputeBit {
            $($name,)*
            _COUNT,
        }

        const _: () = assert!(
            (ComputeBit::_COUNT as u32) <= 128,
            "Compute has more than 128 flags; widen its integer type"
        );

        bitflags! {
            /// Bit-packed set of computation primitives needed to evaluate a feature set.
            ///
            /// Each flag gates a specific accumulation or post-processing step in the
            /// engine hot-loops. Features declare their dependencies via
            /// [`Feature::required_compute`], and the engine checks these flags to skip
            /// unnecessary work.
            #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
            pub struct Compute: u128 {
                $($(#[$meta])* const $name = 1 << (ComputeBit::$name as u8);)*
            }
        }
    };
}

compute_flags! {
    // ── Pass-1 accumulation gates ──────────────────────────
    SUM,
    MEAN,
    VARIANCE,
    STD,
    MIN,
    MAX,
    MEDIAN,
    SKEW,                 // sum_cubes
    KURTOSIS,             // sum_quads
    MAD,
    IQR,
    ENERGY,               // sum of squares
    RMS,
    ROOT_MEAN_SQ,
    ZERO_CROSS,
    PEAKS,
    TROUGHS,
    NUMBER_PEAKS_CROSSINGS,
    MAC,                  // mean abs change
    MC,                   // mean change
    CID_CE,
    SLOPE,                // sum_ix
    INTERCEPT,
    PAA,
    CNT_ABOVE_MEAN,
    CNT_BELOW_MEAN,
    STRIKE_ABOVE,
    STRIKE_BELOW,
    VAR_COEFF,
    COUNT_ABOVE_T,
    COUNT_BELOW_T,
    RANGE_COUNT,
    C3,
    ZC_STATS,             // zero-crossing mean
    ZC_STD,
    ZC_INDICES,

    // ── Pass-2 / post-processing gates ─────────────────────
    NEEDS_SORT,
    ABS_MAX,
    FIRST_LOC_MAX,
    LAST_LOC_MAX,
    FIRST_LOC_MIN,
    LAST_LOC_MIN,
    FULL_AUTOCORR,        // Wiener-Khinchin FFT autocorr
    PACF,
    TRA,                  // time reversal asymmetry

    // ── FFT / spectral / complex feature gates ─────────────
    FFT_COEFF,
    APPROX_ENT,
    AGG_LIN_TREND,
    QUANTILE,
    BENFORD,
    LANGEVIN,
    REOCCUR_VAL,
    SPEC_CENTROID,
    SPEC_DISTANCE,
    SPEC_DECREASE,
    SPEC_SLOPE,
    SIG_DISTANCE,
    WAVELET,
    SPECTROGRAM,
    REOCCUR_DP,
    MEAN_N_ABS_MAX,
    HUMAN_RANGE_E,
    LENGTH,
    VAR_GT_STD,
    HAS_DUPLICATE,
    HAS_DUP_MAX,
    HAS_DUP_MIN,
    REOCCUR_RATIOS,
    SPEC_SPREAD,
    SPEC_SKEWNESS,
    SPEC_KURTOSIS,
    RATIO_BEYOND_R_SIGMA,
    SAMP_ENT,
    BINNED_ENT,
    SPEC_ROLLON,
    SPEC_ROLLOFF,
    AR_COEFF,
    FRIEDRICH,
    CWT,
    WELCH,
    SPEC_ENTROPY,
    MFCC,
    CWT_MEXH,
    QUERY_SIMILARITY,
    MATRIX_PROFILE,
    MEDIAN_ABS_DEV,
    CALC_CENTROID,
    ADF,
    LPCC,
}

// ── Composite masks (zero bit-cost) ────────────────────────────────────────

impl Compute {
    pub const ANY_FFT: Self = Self::from_bits_retain(
        Self::FFT_COEFF.bits()
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
            | Self::WELCH.bits()
            | Self::SPEC_ROLLON.bits()
            | Self::SPEC_ROLLOFF.bits()
            | Self::SPEC_SKEWNESS.bits()
            | Self::SPEC_KURTOSIS.bits(),
    );

    pub const ANY_DIFF: Self = Self::from_bits_retain(
        Self::ZERO_CROSS.bits() | Self::MAC.bits() | Self::MC.bits() | Self::CID_CE.bits(),
    );
}
