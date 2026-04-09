pub fn zero_crossing_rate(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return 0.0;
    }
    let mut count = 0;
    for i in 1..data.len() {
        if (data[i - 1] < 0.0 && data[i] >= 0.0) || (data[i - 1] >= 0.0 && data[i] < 0.0) {
            count += 1;
        }
    }
    count as f64 / (data.len() as f64 - 1.0)
}

pub fn peak_count(data: &[f64]) -> f64 {
    if data.len() < 3 {
        return 0.0;
    }
    let mut count = 0;
    for i in 1..data.len() - 1 {
        if data[i] > data[i - 1] && data[i] > data[i + 1] {
            count += 1;
        }
    }
    count as f64
}

pub fn autocorrelation(data: &[f64], lag: usize) -> f64 {
    let n = data.len();
    if n <= lag {
        return 0.0;
    }
    let m = crate::features::statistics::mean(data);
    let mut num = 0.0;
    let mut den = 0.0;
    for i in 0..n {
        let diff = data[i] - m;
        den += diff * diff;
        if i >= lag {
            num += (data[i] - m) * (data[i - lag] - m);
        }
    }
    if den == 0.0 {
        return 0.0;
    }
    num / den
}

pub fn autocorr_lag1(data: &[f64]) -> f64 {
    autocorrelation(data, 1)
}

pub fn mean_abs_change(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return 0.0;
    }
    let sum_diff: f64 = data.windows(2).map(|w| (w[1] - w[0]).abs()).sum();
    sum_diff / (data.len() - 1) as f64
}

pub fn mean_change(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return 0.0;
    }
    let sum_diff: f64 = data.windows(2).map(|w| w[1] - w[0]).sum();
    sum_diff / (data.len() - 1) as f64
}

pub fn cid_ce(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return 0.0;
    }
    let ce: f64 = data.windows(2).map(|w| (w[1] - w[0]).powi(2)).sum();
    ce.sqrt()
}

pub fn npaa(data: &[f64], segments: usize, idx: usize) -> f64 {
    let n = data.len();
    if n == 0 || segments == 0 || idx >= segments { return f64::NAN; }
    
    let mean = crate::features::statistics::mean(data);
    let std = crate::features::statistics::std(data);
    let std = if std == 0.0 { 1.0 } else { std };
    
    let start = idx * n / segments;
    let end = (idx + 1) * n / segments;
    if start >= end { return f64::NAN; }
    
    let mut sum = 0.0;
    for x in &data[start..end] {
        sum += (*x - mean) / std;
    }
    sum / (end - start) as f64
}

pub fn paa(data: &[f64], segments: usize, idx: usize) -> f64 {
    let n = data.len();
    if n == 0 || segments == 0 || idx >= segments { return f64::NAN; }
    
    let start = idx * n / segments;
    let end = (idx + 1) * n / segments;
    if start >= end { return f64::NAN; }
    
    let mut sum = 0.0;
    for x in &data[start..end] {
        sum += *x;
    }
    sum / (end - start) as f64
}

pub fn npta(data: &[f64], segments: usize, idx: usize) -> f64 {
    let n = data.len();
    if n == 0 || segments == 0 || idx >= segments { return f64::NAN; }
    
    let mean = crate::features::statistics::mean(data);
    let std = crate::features::statistics::std(data);
    let std = if std == 0.0 { 1.0 } else { std };
    
    let start = idx * n / segments;
    let end = (idx + 1) * n / segments;
    
    if end - start < 2 { return 0.0; }
    
    let norm_first = (data[start] - mean) / std;
    let norm_last = (data[end - 1] - mean) / std;
    (norm_last - norm_first) / (end - start - 1) as f64
}
