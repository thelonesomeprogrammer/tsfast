use num_complex::Complex;
use realfft::num_complex;
use smallvec::SmallVec;
use std::simd::f32x4;

pub const LANES: usize = 4;

pub struct SlidingDFT {
    pub n: usize,
    pub bins: Vec<Complex<f32>>,
    pub twiddles: Vec<Complex<f32>>,
}

impl SlidingDFT {
    pub fn new(n: usize) -> Self {
        let n_bins = n / 2 + 1;
        let mut twiddles = Vec::with_capacity(n_bins);
        let pi2 = 2.0 * std::f32::consts::PI;
        for k in 0..n_bins {
            let angle = pi2 * k as f32 / n as f32;
            twiddles.push(Complex::new(angle.cos(), angle.sin()));
        }
        Self {
            n,
            bins: vec![Complex::new(0.0, 0.0); n_bins],
            twiddles,
        }
    }

    #[inline(always)]
    pub fn update(&mut self, old_val: f32, new_val: f32) {
        let diff = new_val - old_val;
        for (k, twiddle) in self.twiddles.iter().enumerate() {
            // S_k(n+1) = twiddle_k * (S_k(n) + new - old)
            self.bins[k] = (self.bins[k] + diff) * twiddle;
        }
    }

    pub fn from_fft(fft_complex: Vec<Complex<f32>>, n: usize) -> Self {
        let mut s = Self::new(n);
        s.bins = fft_complex;
        s
    }
}
pub struct ColumnState {
    pub total_sum: f32,
    pub min_value: f32,
    pub max_value: f32,
    pub energy: f32,
    pub sum_cubes: f32,
    pub sum_quads: f32,
    pub mac_sum: f32,
    pub mc_sum: f32,
    pub sum_sq_diff: f32,
    pub sum_prod: f32,
    pub sum_ix: f32,
    pub auc_sum: f32,
    pub mac_sum_vec: f32x4,
    pub mc_sum_vec: f32x4,
    pub zcr_count: u32,
    pub peaks: u32,
    pub zc_indices: SmallVec<[f32; 32]>,
    pub paa_sums: Vec<Vec<f32>>,
    pub current_paa_segs: Vec<usize>,
    pub c3_sums: Vec<f32>,
    pub autocorr_sums: Vec<f32>,
    pub prefix_sums: Vec<f32>,
    pub prev_last: f32,
    pub prev_val: f32,
    pub prev_prev_val: f32,
    pub abs_max: f32,
    pub first_max_idx: usize,
    pub last_max_idx: usize,
    pub first_min_idx: usize,
    pub last_min_idx: usize,
    pub abs_sum: f32,
    pub benford_counts: [usize; 9],
    pub min_queue: Vec<(usize, f32)>,
    pub min_q_head: usize,
    pub min_q_tail: usize,
    pub max_queue: Vec<(usize, f32)>,
    pub max_q_head: usize,
    pub max_q_tail: usize,
    pub approx_entropy_buffer: Vec<usize>,
    // Incremental moments (Welford's or similar)
    pub n: f32,
    pub mean: f32,
    pub m2: f32,
    pub m3: f32,
    pub m4: f32,
    pub last_fft_n: usize,
    pub last_spectrum: Vec<f32>,
    pub last_fft_complex: Vec<num_complex::Complex<f32>>,
    pub fft_in_buffer: Vec<f32>,
    pub fft_out_buffer: Vec<num_complex::Complex<f32>>,
    pub sliding_dft: Option<SlidingDFT>,
}

pub fn next_good_fft_size(n: usize) -> usize {
    if n <= 2 {
        return n;
    }

    let limit = (n as f32 * 1.05) as usize;
    for candidate in n..=limit {
        if is_smooth(candidate) {
            return candidate;
        }
    }
    // Fallback to next power of 2 if no smooth number in 5% neighborhood
    n.next_power_of_two()
}

fn is_smooth(mut n: usize) -> bool {
    if n == 0 {
        return false;
    }
    for &p in &[2, 3, 5, 7] {
        while n % p == 0 {
            n /= p;
        }
    }
    n == 1
}

impl ColumnState {
    pub fn new(
        unique_paa_totals: &[u16],
        unique_c3_lags: &[u16],
        unique_autocorr_lags: &[u16],
        first_val: f32,
    ) -> Self {
        Self {
            total_sum: 0.0,
            min_value: f32::INFINITY,
            max_value: f32::NEG_INFINITY,
            energy: 0.0,
            sum_cubes: 0.0,
            sum_quads: 0.0,
            mac_sum: 0.0,
            mc_sum: 0.0,
            sum_sq_diff: 0.0,
            sum_prod: 0.0,
            sum_ix: 0.0,
            auc_sum: 0.0,
            mac_sum_vec: f32x4::splat(0.0),
            mc_sum_vec: f32x4::splat(0.0),
            zcr_count: 0,
            peaks: 0,
            zc_indices: SmallVec::new(),
            paa_sums: unique_paa_totals
                .iter()
                .map(|&total| vec![0.0; total as usize])
                .collect(),
            current_paa_segs: vec![0usize; unique_paa_totals.len()],
            c3_sums: vec![0.0; unique_c3_lags.len()],
            autocorr_sums: vec![0.0; unique_autocorr_lags.len()],
            prefix_sums: Vec::new(),
            prev_last: first_val,
            prev_val: first_val,
            prev_prev_val: first_val,
            abs_max: 0.0,
            first_max_idx: 0,
            last_max_idx: 0,
            first_min_idx: 0,
            last_min_idx: 0,
            abs_sum: 0.0,
            benford_counts: [0; 9],
            min_queue: Vec::new(),
            min_q_head: 0,
            min_q_tail: 0,
            max_queue: Vec::new(),
            max_q_head: 0,
            max_q_tail: 0,
            approx_entropy_buffer: Vec::new(),
            n: 0.0,
            mean: 0.0,
            m2: 0.0,
            m3: 0.0,
            m4: 0.0,
            last_fft_n: 0,
            last_spectrum: Vec::new(),
            last_fft_complex: Vec::new(),
            fft_in_buffer: Vec::new(),
            fft_out_buffer: Vec::new(),
            sliding_dft: None,
        }
    }
}

use crate::types::{FastBitArray, Feature};

pub(crate) fn map_features_to_indices(features: &[Feature]) -> FastBitArray {
    let mut bits = FastBitArray::ZERO;
    for feat in features {
        match feat {
            Feature::TotalSum => bits.set_batch([0]),
            Feature::Mean => bits.set_batch([0, 1]),
            Feature::Variance => bits.set_batch([0, 1, 2, 12, 37]),
            Feature::Std => bits.set_batch([0, 1, 2, 3, 12, 37]),
            Feature::Min => bits.set_batch([4]),
            Feature::Max => bits.set_batch([5]),
            Feature::Median => bits.set_batch([6, 37]),
            Feature::Skew => bits.set_batch([0, 1, 2, 7, 12, 37]),
            Feature::UnbiasedFisherKurtosis => bits.set_batch([0, 1, 2, 8, 12, 37]),
            Feature::BiasedFisherKurtosis => bits.set_batch([0, 1, 2, 8, 12, 37]),
            Feature::Mad => bits.set_batch([0, 1, 9, 37]),
            Feature::Iqr => bits.set_batch([4, 5, 6, 10, 37]),
            Feature::Entropy => bits.set_batch([4, 5, 6, 10, 11, 37]),
            Feature::Energy => bits.set_batch([12]),
            Feature::Rms => bits.set_batch([12, 13]),
            Feature::RootMeanSquare => bits.set_batch([12, 14]),
            Feature::ZeroCrossingRate => bits.set_batch([15]),
            Feature::PeakCount => bits.set_batch([16]),
            Feature::AutocorrLag1 => bits.set_batch([0, 1, 12, 17]),
            Feature::AutocorrFirst1e => bits.set_batch([0, 1, 12, 43, 37]),
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
            Feature::VariationCoefficient => bits.set_batch([0, 1, 2, 3, 12, 29, 37]),
            Feature::C3(_) => bits.set_batch([30]),
            Feature::Auc => bits.set_batch([31]),
            Feature::SlopeSignChange => bits.set_batch([16, 32]),
            Feature::TurningPoints => bits.set_batch([16, 33]),
            Feature::ZeroCrossingMean => bits.set_batch([0, 1, 15, 34, 36, 37]),
            Feature::ZeroCrossingStd => bits.set_batch([0, 1, 15, 34, 35, 36, 37]),
            Feature::AbsMax => bits.set_batch([38]),
            Feature::FirstLocMax => bits.set_batch([5, 39]),
            Feature::LastLocMax => bits.set_batch([5, 40]),
            Feature::FirstLocMin => bits.set_batch([4, 41]),
            Feature::LastLocMin => bits.set_batch([4, 42]),
            Feature::Autocorr(lag) => {
                if *lag == 1 {
                    bits.set_batch([0, 1, 12, 17, 37]);
                } else {
                    bits.set_batch([0, 1, 2, 12, 43, 37]);
                }
            }
            Feature::PartialAutocorr(_) => bits.set_batch([0, 1, 2, 12, 44, 37]),
            Feature::TimeReversalAsymmetry(_) => bits.set_batch([45, 37]),
            Feature::FftCoefficient(_, _) => bits.set_batch([46, 37]),
            Feature::ApproxEntropy(_, _) => bits.set_batch([47, 37]),
            Feature::AggLinearTrend(_, _, _) => bits.set_batch([48, 37]),
            Feature::Quantile(_) => bits.set_batch([49, 37]),
            Feature::IndexMassQuantile(_) => bits.set_batch([61, 50, 37]),
            Feature::BenfordCorrelation => bits.set_batch([51, 37]),
            Feature::MaxLangevinFixedPoint(_, _) => bits.set_batch([52, 37]),
            Feature::SumOfReoccurringValues => bits.set_batch([53, 37]),
            Feature::SumOfReoccurringDataPoints => bits.set_batch([62, 37]),
            Feature::MeanNAbsoluteMax(_) => bits.set_batch([63, 37]),
            Feature::HumanRangeEnergy(_) => bits.set_batch([64, 37]),
            Feature::SpectralCentroid => bits.set_batch([54, 37]),
            Feature::SpectralDistance => bits.set_batch([55, 37]),
            Feature::SpectralDecrease => bits.set_batch([56, 37]),
            Feature::SpectralSlope => bits.set_batch([57, 37]),
            Feature::SignalDistance => bits.set_batch([58, 37]),
            Feature::WaveletFeatures(_, _) => bits.set_batch([59, 37]),
            Feature::SpectrogramCoefficients(_, _) => bits.set_batch([60, 37]),
        }
    }
    bits
}
