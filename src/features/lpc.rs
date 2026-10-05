use std::f32::consts::PI;

pub fn compute_lpcc(values: &[f32]) -> Vec<f32> {
    let order = 11;
    let n_coeff = 12;

    if values.is_empty() {
        return vec![0.0; n_coeff];
    }

    // Uncentred autocorrelation (as TSFEL), not the mean-centred FFT one other
    // features share: that would make LPCC depend on which features are requested.
    let mut r = vec![0.0f32; order + 1];
    for (lag, r_lag) in r.iter_mut().enumerate().take(values.len()) {
        let s: f64 = values[lag..]
            .iter()
            .zip(values)
            .map(|(&a, &b)| a as f64 * b as f64)
            .sum();
        *r_lag = s as f32;
    }

    let mut a = vec![0.0f32; order + 1];
    let mut e = r[0];

    a[0] = 1.0;

    if e.abs() < 1e-9 {
        return vec![0.0; n_coeff];
    }

    for i in 1..=order {
        let mut sum = 0.0;
        for j in 1..i {
            sum += a[j] * r[i - j];
        }
        let k = (r[i] - sum) / e;

        let mut a_new = a.clone();
        a_new[i] = k;
        for j in 1..i {
            a_new[j] = a[j] - k * a[i - j];
        }
        a = a_new;
        e *= 1.0 - k * k;
        if e.abs() < 1e-9 {
            break;
        }
    }

    let mut lpc_coeffs = vec![0.0f32; n_coeff];
    lpc_coeffs[0] = 1.0;
    for i in 1..n_coeff {
        lpc_coeffs[i] = -a[i];
    }

    let n = lpc_coeffs.len();
    let mut fft_real = vec![0.0f32; n];
    let mut fft_imag = vec![0.0f32; n];

    for k in 0..n {
        for t in 0..n {
            let angle = -2.0 * PI * (k as f32) * (t as f32) / (n as f32);
            fft_real[k] += lpc_coeffs[t] * angle.cos();
            fft_imag[k] += lpc_coeffs[t] * angle.sin();
        }
    }

    let mut log_power = vec![0.0f32; n];
    for k in 0..n {
        let power = fft_real[k] * fft_real[k] + fft_imag[k] * fft_imag[k];
        log_power[k] = if power > 1e-9 { power.ln() } else { 0.0 };
    }

    let mut ifft_real = vec![0.0f32; n];
    let mut ifft_imag = vec![0.0f32; n];

    for t in 0..n {
        for k in 0..n {
            let angle = 2.0 * PI * (k as f32) * (t as f32) / (n as f32);
            ifft_real[t] += log_power[k] * angle.cos();
            ifft_imag[t] += log_power[k] * angle.sin();
        }
        ifft_real[t] /= n as f32;
        ifft_imag[t] /= n as f32;
    }

    let mut lpcc = vec![0.0f32; n_coeff];
    for i in 0..n {
        lpcc[i] = (ifft_real[i] * ifft_real[i] + ifft_imag[i] * ifft_imag[i]).sqrt();
    }
    lpcc
}
