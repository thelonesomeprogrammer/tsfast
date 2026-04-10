use crate::features::statistics::mean;
use std::collections::HashMap;

pub fn zero_crossing_rate(data: &[f64], precalc: &mut HashMap<&str, f64>) -> f64 {
    if let Some(&z) = precalc.get("zero_crossing_rate") {
        return z;
    }
    if data.len() < 2 {
        return 0.0;
    }
    let mut count = 0;
    for i in 1..data.len() {
        if (data[i - 1] < 0.0 && data[i] >= 0.0) || (data[i - 1] >= 0.0 && data[i] < 0.0) {
            count += 1;
        }
    }
    let zcr = count as f64 / (data.len() as f64 - 1.0);
    precalc.insert("zero_crossing_rate", zcr);
    zcr
}

pub fn peak_count(data: &[f64], precalc: &mut HashMap<&str, f64>) -> f64 {
    if let Some(&p) = precalc.get("peak_count") {
        return p;
    }
    if data.len() < 3 {
        return 0.0;
    }
    let mut count = 0;
    for i in 1..data.len() - 1 {
        if data[i] > data[i - 1] && data[i] > data[i + 1] {
            count += 1;
        }
    }
    let peak = count as f64;
    precalc.insert("peak_count", peak);
    peak
}

fn autocorrelation(data: &[f64], lag: usize, precalc: &mut HashMap<&str, f64>) -> f64 {
    let n = data.len();
    if n <= lag {
        return 0.0;
    }
    let m = if let Some(&m) = precalc.get("mean") {
        m
    } else {
        mean(data, precalc)
    };
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

pub fn autocorr_lag1(data: &[f64], precalc: &mut HashMap<&str, f64>) -> f64 {
    if let Some(&a) = precalc.get("autocorr_lag1") {
        return a;
    }
    let autocorr = autocorrelation(data, 1, precalc);
    precalc.insert("autocorr_lag1", autocorr);
    autocorr
}

pub fn mean_abs_change(data: &[f64], precalc: &mut HashMap<&str, f64>) -> f64 {
    if let Some(&m) = precalc.get("mean_abs_change") {
        return m;
    }
    if data.len() < 2 {
        return 0.0;
    }
    let sum_diff: f64 = data.windows(2).map(|w| (w[1] - w[0]).abs()).sum();
    let mac = sum_diff / (data.len() - 1) as f64;
    precalc.insert("mean_abs_change", mac);
    mac
}

pub fn mean_change(data: &[f64], precalc: &mut HashMap<&str, f64>) -> f64 {
    if let Some(&m) = precalc.get("mean_change") {
        return m;
    }
    if data.len() < 2 {
        return 0.0;
    }
    let sum_diff: f64 = data.windows(2).map(|w| w[1] - w[0]).sum();
    let mc = sum_diff / (data.len() - 1) as f64;
    precalc.insert("mean_change", mc);
    mc
}

pub fn cid_ce(data: &[f64], precalc: &mut HashMap<&str, f64>) -> f64 {
    if let Some(&c) = precalc.get("cid_ce") {
        return c;
    }
    if data.len() < 2 {
        return 0.0;
    }
    let ce: f64 = data.windows(2).map(|w| (w[1] - w[0]).powi(2)).sum();
    let ces = ce.sqrt();
    precalc.insert("cid_ce", ces);
    ces
}

pub fn paa(data: &[f64], segments: usize, idx: usize, precalc: &mut HashMap<&str, f64>) -> f64 {
    let key = format!("paa-{}-{}", segments, idx);
    if let Some(&p) = precalc.get(key.as_str()) {
        return p;
    }

    let n = data.len();
    if n == 0 || segments == 0 || idx >= segments {
        return f64::NAN;
    }

    let start = idx * n / segments;
    let end = (idx + 1) * n / segments;
    if start >= end {
        return f64::NAN;
    }

    let mut sum = 0.0;
    for x in &data[start..end] {
        sum += *x;
    }
    let mean = sum / (end - start) as f64;
    precalc.insert(Box::leak(key.into_boxed_str()), mean);
    mean
}
