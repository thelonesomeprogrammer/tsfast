use num_complex::Complex;
use realfft::num_complex;
use smallvec::SmallVec;
use std::simd::f32x4;

pub const LANES: usize = 4;

#[derive(Clone, Debug)]
pub struct SlidingDFT {
    pub n: usize,
    /// Updates since the bins were last computed by a real FFT.
    pub updates: usize,
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
            updates: 0,
            bins: vec![Complex::new(0.0, 0.0); n_bins],
            twiddles,
        }
    }

    #[inline(always)]
    pub fn update(&mut self, old_val: f32, new_val: f32) {
        self.updates += 1;
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
    pub sum_ix: f32,
    pub t_energy: f32,
    pub mac_sum_vec: f32x4,
    pub mc_sum_vec: f32x4,

    // SIMD accumulators
    pub sum_vec: f32x4,
    pub energy_vec: f32x4,
    pub sum_cubes_vec: f32x4,
    pub sum_quads_vec: f32x4,
    pub min_vec: f32x4,
    pub max_vec: f32x4,
    pub abs_max_vec: f32x4,
    pub sum_sq_diff_vec: f32x4,
    pub sum_ix_vec: f32x4,
    pub t_energy_vec: f32x4,

    pub zcr_count: u32,
    pub peaks: u32,
    pub troughs: u32,
    pub zc_indices: SmallVec<[f32; 32]>,
    pub paa_sums: Vec<Vec<f32>>,
    pub current_paa_segs: Vec<usize>,
    pub c3_sums: Vec<f32>,
    pub c3_sums_vec: Vec<f32x4>,
    pub autocorr_sums: Vec<f32>,
    pub tra_sums: Vec<f32>,
    pub tra_sums_vec: Vec<f32x4>,
    pub prefix_sums: Vec<f32>,
    pub prev_last: f32,
    pub prev_val: f32,
    pub prev_prev_val: f32,
    pub abs_max: f32,
    pub first_max_idx: usize,
    pub last_max_idx: usize,
    pub first_min_idx: usize,
    pub last_min_idx: usize,
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
    pub binned_entropy_buffer: Vec<f32>,
    pub agg_linear_trend_buffer: Vec<f32>,
    pub workspace_f64_1: Vec<f64>,
    pub workspace_f64_2: Vec<f64>,
    pub workspace_f64_3: Vec<f64>,
    pub workspace_f64_4: Vec<f64>,
    pub workspace_f64_5: Vec<f64>,
    pub higuchi_lk: Vec<f64>,
    pub higuchi_k_values: Vec<f64>,
    pub sort_buffer: Vec<f32>,
    pub mad_buffer: Vec<f32>,
    pub diff_buffer: Vec<f32>,
    pub pacf_buffer: Vec<f32>,
    pub cwt_final_energy: Vec<f32>,
    pub cwt_final_sum_abs: Vec<f32>,
    pub cwt_final_clnc: Vec<f32>,
    pub value_counts: rustc_hash::FxHashMap<u32, u32>,
    pub reoccurring_datapoints: u32,
    pub reoccurring_values: u32,
    pub ar_coeffs: rustc_hash::FxHashMap<u16, Vec<f32>>,
    pub friedrich_coeffs: rustc_hash::FxHashMap<(u8, u32), Vec<f32>>,
    pub max_langevin_fixed_point_cache: rustc_hash::FxHashMap<(u8, u32), f32>,

    // Incremental moments (Welford's or similar)
    pub n: f32,
    pub mean: f32,
    pub m2: f32,
    pub m3: f32,
    pub m4: f32,
    pub last_fft_n: usize,
    pub last_fft_complex: Vec<num_complex::Complex<f32>>,
    pub fft_in_buffer: Vec<f32>,
    pub fft_out_buffer: Vec<num_complex::Complex<f32>>,
    pub fft_inv_buffer: Vec<f32>,
    pub sliding_dft: Option<SlidingDFT>,
    pub cwt_peaks: u16,
    pub welch_density: Vec<f32>,
    pub spectrum_buffer: Vec<f32>,
    pub adf_test_stat: f32,
    pub adf_p_value: f32,
    pub adf_used_lag: f32,
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
    /// Drop results cached while evaluating the previous window. Values or
    /// buffer addresses can't tell windows apart: the sliding engine reuses its
    /// history buffer in place, so every window has the same pointer and length.
    pub fn reset_window_caches(&mut self) {
        self.ar_coeffs.clear();
        self.friedrich_coeffs.clear();
        self.max_langevin_fixed_point_cache.clear();
        self.adf_test_stat = f32::NAN;
        self.adf_p_value = f32::NAN;
        self.adf_used_lag = f32::NAN;
        self.higuchi_lk.clear();
        self.higuchi_k_values.clear();
    }

    pub fn new(
        unique_paa_totals: &[u16],
        unique_c3_lags: &[u16],
        unique_autocorr_lags: &[u16],
        unique_tra_lags: &[u16],
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
            sum_ix: 0.0,
            mac_sum_vec: f32x4::splat(0.0),
            mc_sum_vec: f32x4::splat(0.0),
            sum_vec: f32x4::splat(0.0),
            energy_vec: f32x4::splat(0.0),
            sum_cubes_vec: f32x4::splat(0.0),
            sum_quads_vec: f32x4::splat(0.0),
            min_vec: f32x4::splat(f32::INFINITY),
            max_vec: f32x4::splat(f32::NEG_INFINITY),
            abs_max_vec: f32x4::splat(0.0),
            t_energy: 0.0,
            sum_sq_diff_vec: f32x4::splat(0.0),
            sum_ix_vec: f32x4::splat(0.0),
            t_energy_vec: f32x4::splat(0.0),
            zcr_count: 0,
            peaks: 0,
            troughs: 0,
            zc_indices: SmallVec::new(),
            paa_sums: unique_paa_totals
                .iter()
                .map(|&total| vec![0.0; total as usize])
                .collect(),
            current_paa_segs: vec![0usize; unique_paa_totals.len()],
            c3_sums: vec![0.0; unique_c3_lags.len()],
            c3_sums_vec: vec![f32x4::splat(0.0); unique_c3_lags.len()],
            autocorr_sums: vec![0.0; unique_autocorr_lags.len()],
            tra_sums: vec![0.0; unique_tra_lags.len()],
            tra_sums_vec: vec![f32x4::splat(0.0); unique_tra_lags.len()],
            prefix_sums: Vec::new(),
            prev_last: first_val,
            prev_val: first_val,
            prev_prev_val: first_val,
            abs_max: 0.0,
            first_max_idx: 0,
            last_max_idx: 0,
            first_min_idx: 0,
            last_min_idx: 0,
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
            binned_entropy_buffer: Vec::new(),
            agg_linear_trend_buffer: Vec::new(),
            workspace_f64_1: Vec::new(),
            workspace_f64_2: Vec::new(),
            workspace_f64_3: Vec::new(),
            workspace_f64_4: Vec::new(),
            workspace_f64_5: Vec::new(),
            higuchi_lk: Vec::new(),
            higuchi_k_values: Vec::new(),
            sort_buffer: Vec::new(),
            mad_buffer: Vec::new(),
            diff_buffer: Vec::new(),
            pacf_buffer: Vec::new(),
            cwt_final_energy: Vec::new(),
            cwt_final_sum_abs: Vec::new(),
            cwt_final_clnc: Vec::new(),
            value_counts: rustc_hash::FxHashMap::default(),
            reoccurring_datapoints: 0,
            reoccurring_values: 0,
            ar_coeffs: rustc_hash::FxHashMap::default(),
            friedrich_coeffs: rustc_hash::FxHashMap::default(),
            max_langevin_fixed_point_cache: rustc_hash::FxHashMap::default(),
            n: 0.0,
            mean: 0.0,
            m2: 0.0,
            m3: 0.0,
            m4: 0.0,
            last_fft_n: 0,
            last_fft_complex: Vec::new(),
            fft_in_buffer: Vec::new(),
            fft_out_buffer: Vec::new(),
            fft_inv_buffer: Vec::new(),
            sliding_dft: None,
            cwt_peaks: 0,
            welch_density: Vec::new(),
            spectrum_buffer: Vec::new(),
            adf_test_stat: std::f32::NAN,
            adf_p_value: std::f32::NAN,
            adf_used_lag: std::f32::NAN,
        }
    }
}

use crate::types::{Compute, Feature};

pub(crate) fn map_features_to_indices(features: &[Feature]) -> Compute {
    features
        .iter()
        .fold(Compute::empty(), |acc, f| acc | f.required_compute())
}

use std::simd::cmp::SimdPartialOrd;
use std::simd::num::SimdFloat;

pub fn sample_entropy_simd(data: &[f32], std_dev: f32) -> f32 {
    let n = data.len();
    if n <= 2 {
        return f32::NAN;
    }
    let m = 2;
    let r = 0.2 * std_dev;
    let r_vec = f32x4::splat(r);

    let mut b_count = 0;
    let mut a_count = 0;
    let end_m = n - m; // Number of templates of length m+1 is end_m.

    // We only need to check i < j and then multiply by 2 because distance is symmetric!
    for i in 0..end_m {
        // We broadcast data[i..i+m+1]
        let v_i0 = f32x4::splat(data[i]);
        let v_i1 = f32x4::splat(data[i + 1]);
        let v_i2 = f32x4::splat(data[i + 2]);

        let mut j = i + 1; // only check j > i
        while j + 4 <= end_m {
            let v_j0 = f32x4::from_slice(&data[j..j + 4]);
            let v_j1 = f32x4::from_slice(&data[j + 1..j + 5]);
            let v_j2 = f32x4::from_slice(&data[j + 2..j + 6]);

            let diff0 = (v_j0 - v_i0).abs();
            let diff1 = (v_j1 - v_i1).abs();
            let max_m = diff0.simd_max(diff1);
            let mask_m = max_m.simd_le(r_vec);
            b_count += mask_m.to_bitmask().count_ones();

            let diff2 = (v_j2 - v_i2).abs();
            let max_mp1 = max_m.simd_max(diff2);
            let mask_mp1 = max_mp1.simd_le(r_vec);
            a_count += mask_mp1.to_bitmask().count_ones();

            j += 4;
        }
        // Remainder
        for j_rem in j..end_m {
            let max_m = (data[j_rem] - data[i])
                .abs()
                .max((data[j_rem + 1] - data[i + 1]).abs());
            if max_m <= r {
                b_count += 1;
                let max_mp1 = max_m.max((data[j_rem + 2] - data[i + 2]).abs());
                if max_mp1 <= r {
                    a_count += 1;
                }
            }
        }
    }

    // B also uses the last length-m template (index n - m), which has no
    // length-(m+1) extension, so the loop above didn't visit it.
    let last = end_m;
    for j in 0..last {
        if (data[j] - data[last]).abs() <= r && (data[j + 1] - data[last + 1]).abs() <= r {
            b_count += 1;
        }
    }

    // TSFresh counts both (i, j) and (j, i) but excludes i == j.
    b_count *= 2;
    a_count *= 2;

    if b_count == 0 || a_count == 0 {
        return f32::NAN;
    }

    -(a_count as f32 / b_count as f32).ln()
}

pub fn approx_entropy_simd(m: usize, r: f32, data: &[f32]) -> f32 {
    let n = data.len();
    if n <= m {
        return 0.0;
    }
    let r_vec = f32x4::splat(r);

    let mut result = 0.0;
    let end = n - m + 1;
    for i in 0..end {
        let mut count = 0;

        let mut j = 0;
        while j + 4 <= end {
            let mut max_diff = f32x4::splat(0.0);
            for k in 0..m {
                let v_i = f32x4::splat(data[i + k]);
                let v_j = f32x4::from_slice(&data[j + k..j + k + 4]);
                let diff = (v_j - v_i).abs();
                max_diff = max_diff.simd_max(diff);
            }
            let mask = max_diff.simd_le(r_vec);
            count += mask.to_bitmask().count_ones();
            j += 4;
        }
        for j_rem in j..end {
            let mut max_diff = 0.0_f32;
            for k in 0..m {
                max_diff = max_diff.max((data[j_rem + k] - data[i + k]).abs());
            }
            if max_diff <= r {
                count += 1;
            }
        }

        result += (count as f32 / end as f32).ln();
    }

    result / end as f32
}

pub fn permutation_entropy(data: &[f32], tau: u32, dimension: u32) -> f32 {
    let tau = tau as usize;
    let dim = dimension as usize;
    let n = data.len();

    // tsfresh _into_subchunks implementation logic:
    // chunk length = dimension, starts every 'tau' (which they call every_n)
    // The window index in the original array is: start + j (where j < dim)
    // start shifts by 'tau' each time.
    if n < dim {
        return f32::NAN;
    }

    let num_shifts = (n - dim) / tau + 1;
    if num_shifts == 0 {
        return f32::NAN;
    }

    let mut counts = rustc_hash::FxHashMap::default();
    let mut perm_indices = vec![0usize; dim];

    for shift in 0..num_shifts {
        let start = shift * tau;
        for j in 0..dim {
            perm_indices[j] = j;
        }

        perm_indices.sort_by(|&a, &b| {
            let val_a = data[start + a];
            let val_b = data[start + b];

            match (val_a.is_nan(), val_b.is_nan()) {
                (true, true) => std::cmp::Ordering::Equal,
                (true, false) => std::cmp::Ordering::Greater,
                (false, true) => std::cmp::Ordering::Less,
                (false, false) => val_a.partial_cmp(&val_b).unwrap_or(std::cmp::Ordering::Equal),
            }
        });

        let mut inv_perm = vec![0usize; dim];
        for (rank, &idx) in perm_indices.iter().enumerate() {
            inv_perm[idx] = rank;
        }

        if dim <= 16 {
            let mut encoded: u64 = 0;
            for (shift, &idx) in inv_perm.iter().enumerate() {
                encoded |= (idx as u64) << (shift * 4);
            }
            *counts.entry(encoded).or_insert(0usize) += 1;
        } else {
            return f32::NAN;
        }
    }

    let mut entropy = 0.0;
    let num_windows_f = num_shifts as f32;
    for &count in counts.values() {
        let p = count as f32 / num_windows_f;
        if p > 0.0 {
            // TSFresh uses natural log here despite mathematical formulation calling for log2
            entropy -= p * p.ln();
        }
    }

    entropy
}
