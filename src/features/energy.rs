pub fn energy(data: &[f64]) -> f64 {
    data.iter().map(|&x| x * x).sum()
}

pub fn rms(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    (energy(data) / data.len() as f64).sqrt()
}
