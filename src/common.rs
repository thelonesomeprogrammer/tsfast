use num_complex::Complex;
use realfft::num_complex;
use smallvec::SmallVec;
use std::simd::f32x4;

pub const LANES: usize = 4;

#[derive(Clone, Debug)]
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

#[derive(Clone)]
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

    // SIMD accumulators
    pub sum_vec: f32x4,
    pub energy_vec: f32x4,
    pub sum_cubes_vec: f32x4,
    pub sum_quads_vec: f32x4,
    pub min_vec: f32x4,
    pub max_vec: f32x4,
    pub abs_sum_vec: f32x4,
    pub abs_max_vec: f32x4,
    pub sum_sq_diff_vec: f32x4,
    pub sum_prod_vec: f32x4,
    pub auc_sum_vec: f32x4,
    pub sum_ix_vec: f32x4,

    pub zcr_count: u32,
    pub peaks: u32,
    pub zc_indices: SmallVec<[f32; 32]>,
    pub paa_sums: Vec<Vec<f32>>,
    pub current_paa_segs: Vec<usize>,
    pub c3_sums: Vec<f32>,
    pub c3_sums_vec: Vec<f32x4>,
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
    pub min_q_len: usize,
    pub max_queue: Vec<(usize, f32)>,
    pub max_q_head: usize,
    pub max_q_tail: usize,
    pub max_q_len: usize,
    pub approx_entropy_buffer: Vec<usize>,
    pub agg_linear_trend_buffer: Vec<f32>,
    pub sort_buffer: Vec<f32>,
    pub pacf_buffer: Vec<f32>,
    pub value_counts: rustc_hash::FxHashMap<u32, u32>,
    pub reoccurring_datapoints: u32,
    pub reoccurring_values: u32,

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
    pub cwt_peaks: u16,
    pub welch_density: Vec<f32>,
    pub welch_planner: Option<std::sync::Arc<std::sync::Mutex<realfft::RealFftPlanner<f32>>>>,
    pub cwt_wavelets: rustc_hash::FxHashMap<u16, Vec<f32>>,
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
            sum_vec: f32x4::splat(0.0),
            energy_vec: f32x4::splat(0.0),
            sum_cubes_vec: f32x4::splat(0.0),
            sum_quads_vec: f32x4::splat(0.0),
            min_vec: f32x4::splat(f32::INFINITY),
            max_vec: f32x4::splat(f32::NEG_INFINITY),
            abs_sum_vec: f32x4::splat(0.0),
            abs_max_vec: f32x4::splat(0.0),
            sum_sq_diff_vec: f32x4::splat(0.0),
            sum_prod_vec: f32x4::splat(0.0),
            auc_sum_vec: f32x4::splat(0.0),
            sum_ix_vec: f32x4::splat(0.0),
            zcr_count: 0,
            peaks: 0,
            zc_indices: SmallVec::new(),
            paa_sums: unique_paa_totals
                .iter()
                .map(|&total| vec![0.0; total as usize])
                .collect(),
            current_paa_segs: vec![0usize; unique_paa_totals.len()],
            c3_sums: vec![0.0; unique_c3_lags.len()],
            c3_sums_vec: vec![f32x4::splat(0.0); unique_c3_lags.len()],
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
            min_q_len: 0,
            max_queue: Vec::new(),
            max_q_head: 0,
            max_q_tail: 0,
            max_q_len: 0,
            approx_entropy_buffer: Vec::new(),
            agg_linear_trend_buffer: Vec::new(),
            sort_buffer: Vec::new(),
            pacf_buffer: Vec::new(),
            value_counts: rustc_hash::FxHashMap::default(),
            reoccurring_datapoints: 0,
            reoccurring_values: 0,
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
            cwt_peaks: 0,
            welch_density: Vec::new(),
            welch_planner: None,
            cwt_wavelets: rustc_hash::FxHashMap::default(),
        }
    }
}

use crate::types::{Compute, Feature};

pub(crate) fn map_features_to_indices(features: &[Feature]) -> Compute {
    features
        .iter()
        .fold(Compute::empty(), |acc, f| acc | f.required_compute())
}

pub fn approx_entropy_phi(m: usize, r: f32, data: &[f32], sorted_idx: &mut Vec<usize>) -> f32 {
    let n = data.len();
    let mut result = 0.0;

    sorted_idx.clear();
    sorted_idx.extend(0..n - m + 1);
    sorted_idx.sort_unstable_by(|&a, &b| {
        data[a]
            .partial_cmp(&data[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    for i in 0..n - m + 1 {
        let mut count = 0;
        let target = data[i];
        let start_pos = sorted_idx.partition_point(|&idx| data[idx] < target - r);
        let end_pos = sorted_idx.partition_point(|&idx| data[idx] <= target + r);

        for &j in &sorted_idx[start_pos..end_pos] {
            let mut max_diff: f32 = 0.0;
            for k in 0..m {
                max_diff = max_diff.max((data[i + k] - data[j + k]).abs());
            }
            if max_diff <= r {
                count += 1;
            }
        }
        result += (count as f32 / (n - m + 1) as f32).ln();
    }
    result / (n - m + 1) as f32
}
