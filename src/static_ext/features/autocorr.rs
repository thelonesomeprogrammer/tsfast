#[inline(always)]
pub fn compute_fft_autocorr(
    compute: &crate::types::FastBitArray,
    values: &[f32],
    n: f32,
    mean: f32,
    m2: f32,
) -> Vec<f32> {
    let mut fft_autocorr = Vec::new();
    if compute.any([43, 44]) && n > 1.0 {
        let n2 = values.len() * 2;
        let fft_size_ac = crate::common::next_good_fft_size(n2);
        let mut planner = realfft::RealFftPlanner::<f32>::new();
        let r2c_ac = planner.plan_fft_forward(fft_size_ac);
        let c2r_ac = planner.plan_fft_inverse(fft_size_ac);

        let mut indata = vec![0.0; fft_size_ac];
        for (i, &v) in values.iter().enumerate() {
            indata[i] = v - mean;
        }
        let mut outdata = r2c_ac.make_output_vec();
        r2c_ac.process(&mut indata, &mut outdata).unwrap();

        for c in &mut outdata {
            *c = realfft::num_complex::Complex::new(c.norm_sqr(), 0.0);
        }

        let mut outdata_inv = c2r_ac.make_output_vec();
        c2r_ac.process(&mut outdata, &mut outdata_inv).unwrap();

        let m2_ac = m2; // var * (n - 1.0)
        if m2_ac.abs() > 1e-9 {
            let scale = 1.0 / (fft_size_ac as f32);
            fft_autocorr = outdata_inv
                .into_iter()
                .take(values.len())
                .map(|v| (v * scale) / m2_ac)
                .collect();
        }
    }
    fft_autocorr
}
