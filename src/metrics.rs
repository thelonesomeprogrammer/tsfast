pub struct BaseMetrics {
    pub mean: f32,
    pub m2: f32,
    pub m3: f32,
    pub m4: f32,
    pub var: f32,
    pub std_dev: f32,
}

pub struct SortMetrics {
    pub first_max_idx: usize,
    pub last_max_idx: usize,
    pub first_min_idx: usize,
    pub last_min_idx: usize,
    pub median: f32,
    pub iqr: f32,
    pub entropy: f32,
}

pub struct MeanMetrics {
    pub mad_sum: f32,
    pub count_a: usize,
    pub count_b: usize,
    pub max_strike_a: usize,
    pub max_strike_b: usize,
}

pub struct ZcMetrics {
    pub zc_mean: f32,
    pub zc_std: f32,
}

pub struct FftResult {
    pub fft_complex: Vec<realfft::num_complex::Complex<f32>>,
    pub spectrum: Vec<f32>,
    pub freq_centroid: f32,
    pub spectral_decrease: f32,
    pub spectral_slope: f32,
    pub spectral_roll_on: f32,
    pub spectral_roll_off: f32,
    pub spectral_spread: f32,
    pub spectral_skewness: f32,
    pub spectral_kurtosis: f32,
    pub fft_autocorr: Vec<f32>,
}
