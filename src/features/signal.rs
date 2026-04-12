use crate::features::statistics::mean;
use crate::registry::Cache;

pub fn zero_crossing_rate(data: &[f64], cache: &mut Cache) {
    let mut count = 0;
    for i in 1..data.len() {
        if (data[i - 1] < 0.0 && data[i] >= 0.0) || (data[i - 1] >= 0.0 && data[i] < 0.0) {
            count += 1;
        }
    }
    let zcr = count as f64 / (data.len() as f64 - 1.0);
    cache.zero_crossing_rate = Some(zcr);
}

pub fn peak_count(data: &[f64], cache: &mut Cache) {
    let mut count = 0;
    for i in 1..data.len() - 1 {
        if data[i] > data[i - 1] && data[i] > data[i + 1] {
            count += 1;
        }
    }
    cache.peak_count = Some(count as f64);
}

fn autocorrelation(data: &[f64], lag: usize, cache: &mut Cache) -> f64 {
    let n = data.len();
    mean(data, cache);
    let m = cache.mean.unwrap();
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

pub fn autocorr_lag1(data: &[f64], cache: &mut Cache) {
    let autocorr = autocorrelation(data, 1, cache);
    cache.autocorr_lag1 = Some(autocorr);
}

pub fn mean_abs_change(data: &[f64], cache: &mut Cache) {
    let sum_diff: f64 = data.windows(2).map(|w| (w[1] - w[0]).abs()).sum();
    let mac = sum_diff / (data.len() - 1) as f64;
    cache.mean_abs_change = Some(mac);
}

pub fn mean_change(data: &[f64], cache: &mut Cache) {
    let sum_diff: f64 = data.windows(2).map(|w| w[1] - w[0]).sum();
    let mc = sum_diff / (data.len() - 1) as f64;
    cache.mean_change = Some(mc);
}

pub fn abs_sum_change(data: &[f64], cache: &mut Cache) {
    let sum_diff: f64 = data.windows(2).map(|w| (w[1] - w[0]).abs()).sum();
    cache.abs_sum_change = Some(sum_diff);
}

pub fn c3(data: &[f64], lag: usize, cache: &mut Cache) {
    let n = data.len();
    if n <= 2 * lag {
        cache.c3.insert(lag as u16, 0.0);
        return;
    }
    let mut sum = 0.0;
    for i in 0..n - 2 * lag {
        sum += data[i + 2 * lag] * data[i + lag] * data[i];
    }
    cache.c3.insert(lag as u16, sum / (n - 2 * lag) as f64);
}

pub fn cid_ce(data: &[f64], cache: &mut Cache) {
    let ce: f64 = data.windows(2).map(|w| (w[1] - w[0]).powi(2)).sum();
    let ces = ce.sqrt();
    cache.cid_ce = Some(ces);
}

pub fn auc(data: &[f64], cache: &mut Cache) {
    let mut area = 0.0;
    for i in 0..data.len() - 1 {
        area += (data[i].abs() + data[i + 1].abs()) / 2.0;
    }
    cache.auc = Some(area);
}

pub fn slope_sign_change(data: &[f64], cache: &mut Cache) {
    let mut count = 0;
    for i in 1..data.len() - 1 {
        let diff1 = data[i] - data[i - 1];
        let diff2 = data[i + 1] - data[i];
        if (diff1 > 0.0 && diff2 < 0.0) || (diff1 < 0.0 && diff2 > 0.0) {
            count += 1;
        }
    }
    cache.slope_sign_change = Some(count as f64);
}

pub fn turning_points(data: &[f64], cache: &mut Cache) {
    let mut count = 0;
    for i in 1..data.len() - 1 {
        if (data[i] > data[i - 1] && data[i] > data[i + 1])
            || (data[i] < data[i - 1] && data[i] < data[i + 1])
        {
            count += 1;
        }
    }
    cache.turning_points = Some(count as f64);
}

pub fn zero_crossing_derivations(data: &[f64], cache: &mut Cache) {
    let mut intervals = Vec::new();
    let mut last_crossing = None;

    for i in 1..data.len() {
        if (data[i - 1] < 0.0 && data[i] >= 0.0) || (data[i - 1] >= 0.0 && data[i] < 0.0) {
            if let Some(last) = last_crossing {
                intervals.push((i - last) as f64);
            }
            last_crossing = Some(i);
        }
    }

    if intervals.is_empty() {
        cache.zero_crossing_mean = Some(0.0);
        cache.zero_crossing_std = Some(0.0);
        return;
    }

    let n = intervals.len() as f64;
    let mean = intervals.iter().sum::<f64>() / n;
    let variance = intervals.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;

    cache.zero_crossing_mean = Some(mean);
    cache.zero_crossing_std = Some(variance.sqrt());
}

pub fn paa(data: &[f64], segments: usize, idx: usize, cache: &mut Cache) {
    let n = data.len();
    let start = idx * n / segments;
    let end = (idx + 1) * n / segments;

    let mut sum = 0.0;
    for x in &data[start..end] {
        sum += *x;
    }
    let mean = sum / (end - start) as f64;
    cache.paa.insert((segments as u16, idx as u16), mean);
}
