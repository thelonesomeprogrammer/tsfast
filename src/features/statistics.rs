pub fn mean(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    data.iter().sum::<f64>() / data.len() as f64
}

pub fn variance(data: &[f64]) -> f64 {
    let n = data.len();
    if n < 2 {
        return 0.0;
    }
    let m = mean(data);
    let ss = data.iter().map(|&x| (x - m).powi(2)).sum::<f64>();
    ss / (n as f64)
}

pub fn std(data: &[f64]) -> f64 {
    variance(data).sqrt()
}

pub fn min(data: &[f64]) -> f64 {
    data.iter().cloned().fold(f64::INFINITY, f64::min)
}

pub fn max(data: &[f64]) -> f64 {
    data.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
}

pub fn median(data: &[f64]) -> f64 {
    let mut data = data.to_vec();
    data.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = data.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 0 {
        (data[n / 2 - 1] + data[n / 2]) / 2.0
    } else {
        data[n / 2]
    }
}

pub fn skew(data: &[f64]) -> f64 {
    let n = data.len();
    if n < 3 {
        return 0.0;
    }
    let m = mean(data);
    let s = std(data);
    if s == 0.0 {
        return 0.0;
    }
    let m3 = data.iter().map(|&x| (x - m).powi(3)).sum::<f64>() / n as f64;
    m3 / s.powi(3)
}

pub fn kurtosis(data: &[f64]) -> f64 {
    let n = data.len();
    if n < 4 {
        return 0.0;
    }
    let m = mean(data);
    let s = std(data);
    if s == 0.0 {
        return 0.0;
    }
    let m4 = data.iter().map(|&x| (x - m).powi(4)).sum::<f64>() / n as f64;
    m4 / s.powi(4) - 3.0
}

pub fn mad(data: &[f64]) -> f64 {
    let med = median(data);
    let deviations: Vec<f64> = data.iter().map(|&x| (x - med).abs()).collect();
    median(&deviations)
}

pub fn iqr(data: &[f64]) -> f64 {
    let mut data = data.to_vec();
    data.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = data.len();
    if n < 4 {
        return 0.0;
    }
    let q1 = median(&data[0..n/2]);
    let q3 = median(&data[(n + 1)/2..n]);
    q3 - q1
}

pub fn entropy(data: &[f64]) -> f64 {
    let n = data.len();
    if n == 0 {
        return 0.0;
    }
    // simple histogram approach (10 bins) for continuous data entropy estimation
    let min_val = min(data);
    let max_val = max(data);
    if (max_val - min_val) < 1e-9 {
        return 0.0;
    }
    let mut counts = vec![0; 10];
    let bin_width = (max_val - min_val) / 10.0;
    for &x in data {
        let mut bin = ((x - min_val) / bin_width) as usize;
        if bin >= 10 {
            bin = 9;
        }
        counts[bin] += 1;
    }
    let mut ent = 0.0;
    for count in counts {
        if count > 0 {
            let p = count as f64 / n as f64;
            ent -= p * p.log2();
        }
    }
    ent
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(mean(&data), 3.0);
    }

    #[test]
    fn test_variance() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(variance(&data), 2.0);
    }

    #[test]
    fn test_std() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(std(&data), 2.0f64.sqrt());
    }
}
