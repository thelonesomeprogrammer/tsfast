#[inline(always)]
pub fn compute_benford_correlation(
    compute: &crate::types::FastBitArray,
    values: &[f32],
) -> f32 {
    let mut benford_corr = 0.0;
    if compute[51] {
        let mut counts = [0.0; 9];
        for &v in values {
            let abs_v = v.abs();
            if abs_v > 0.0 {
                let first_digit =
                    (abs_v / 10.0_f32.powf(abs_v.log10().floor())).floor() as usize;
                if (1..=9).contains(&first_digit) {
                    counts[first_digit - 1] += 1.0;
                }
            }
        }
        let total: f32 = counts.iter().sum();
        if total > 0.0 {
            let p: Vec<f32> = counts.iter().map(|&c| c / total).collect();
            let b: Vec<f32> = (1..10).map(|i| (1.0 + 1.0 / i as f32).log10()).collect();
            let mu_p = p.iter().sum::<f32>() / 9.0;
            let mu_b = b.iter().sum::<f32>() / 9.0;
            let mut num = 0.0;
            let mut den_p = 0.0;
            let mut den_b = 0.0;
            for i in 0..9 {
                num += (p[i] - mu_p) * (b[i] - mu_b);
                den_p += (p[i] - mu_p).powi(2);
                den_b += (b[i] - mu_b).powi(2);
            }
            if den_p > 0.0 && den_b > 0.0 {
                benford_corr = num / (den_p * den_b).sqrt();
            }
        }
    }
    benford_corr
}
