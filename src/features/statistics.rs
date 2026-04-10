use std::collections::HashMap;

pub fn mean(data: &[f64], precalc: &mut HashMap<&str, f64>) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let mean = data.iter().sum::<f64>() / data.len() as f64;
    precalc.insert("mean", mean);
    mean
}

pub fn variance(data: &[f64], precalc: &mut HashMap<&str, f64>) -> f64 {
    let n = data.len();
    if n < 2 {
        return 0.0;
    }

    // Welford's algorithm for numerically stable variance calculation
    let mut mean = 0.0;
    let mut m2 = 0.0;

    for (i, &x) in data.iter().enumerate() {
        let delta = x - mean;
        mean += delta / (i + 1) as f64;
        let delta2 = x - mean;
        m2 += delta * delta2;
    }

    precalc.insert("mean", mean);

    let var = m2 / n as f64;
    precalc.insert("variance", var);
    var
}

pub fn std(data: &[f64], precalc: &mut HashMap<&str, f64>) -> f64 {
    let var = if let Some(&v) = precalc.get("variance") {
        v
    } else {
        variance(data, precalc)
    };
    let std = var.sqrt();
    precalc.insert("std", std);
    std
}

pub fn min(data: &[f64], precalc: &mut HashMap<&str, f64>) -> f64 {
    let m = data.iter().copied().fold(f64::INFINITY, f64::min);
    precalc.insert("min", m);
    m
}

pub fn max(data: &[f64], precalc: &mut HashMap<&str, f64>) -> f64 {
    let m = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    precalc.insert("max", m);
    m
}
pub fn median(data: &[f64], precalc: &mut HashMap<&str, f64>) -> f64 {
    let mut v = data.to_vec();
    let median = sub_median(v.as_mut_slice());
    precalc.insert("median", median);
    median
}

fn sub_median(v: &mut [f64]) -> f64 {
    let mid = v.len() / 2;

    v.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap());

    if v.len() % 2 == 1 {
        v[mid]
    } else {
        let max_low = *v[..mid]
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        (max_low + v[mid]) / 2.0
    }
}

pub fn skew(data: &[f64], precalc: &mut HashMap<&str, f64>) -> f64 {
    let n = data.len();
    if n < 3 {
        return 0.0;
    }

    let m = if let Some(&m) = precalc.get("mean") {
        m
    } else {
        mean(data, precalc)
    };

    let s = if let Some(&s) = precalc.get("std") {
        s
    } else {
        std(data, precalc)
    };

    if s == 0.0 {
        return 0.0;
    }
    let m3 = data.iter().map(|&x| (x - m).powi(3)).sum::<f64>() / n as f64;
    let skew = m3 / s.powi(3);
    precalc.insert("skew", skew);
    skew
}

pub fn kurtosis(data: &[f64], precalc: &mut HashMap<&str, f64>) -> f64 {
    let n = data.len();
    if n < 4 {
        return 0.0;
    }
    let m = if let Some(&m) = precalc.get("mean") {
        m
    } else {
        mean(data, precalc)
    };
    let s = if let Some(&s) = precalc.get("std") {
        s
    } else {
        std(data, precalc)
    };

    if s == 0.0 {
        return 0.0;
    }
    let m4 = data.iter().map(|&x| (x - m).powi(4)).sum::<f64>() / n as f64;
    let kurtosis = m4 / s.powi(4) - 3.0;
    precalc.insert("kurtosis", kurtosis);
    kurtosis
}

pub fn mad(data: &[f64], precalc: &mut HashMap<&str, f64>) -> f64 {
    let med = if let Some(&m) = precalc.get("median") {
        m
    } else {
        median(data, precalc)
    };
    let mut deviations: Vec<f64> = data.iter().map(|&x| (x - med).abs()).collect();
    let mad = sub_median(deviations.as_mut_slice());
    precalc.insert("mad", mad);
    mad
}

pub fn iqr(data: &[f64], precalc: &mut HashMap<&str, f64>) -> f64 {
    let mut data = data.to_vec();
    let n = data.len();
    if n < 4 {
        return 0.0;
    }
    let q1 = sub_median(&mut data[..n / 2]);
    let q3 = sub_median(&mut data[(n + 1) / 2..]);
    let iqr = q3 - q1;
    precalc.insert("iqr", iqr);
    iqr
}

pub fn entropy(data: &[f64], precalc: &mut HashMap<&str, f64>) -> f64 {
    let n = data.len();
    if n == 0 {
        return 0.0;
    }

    let min_val = if let Some(&m) = precalc.get("min") {
        m
    } else {
        min(data, precalc)
    };
    let max_val = if let Some(&m) = precalc.get("max") {
        m
    } else {
        max(data, precalc)
    };
    let iqr_val = if let Some(&m) = precalc.get("iqr") {
        m
    } else {
        iqr(data, precalc)
    };

    if (max_val - min_val) < 1e-9 {
        return 0.0;
    }

    // Use Freedman-Diaconis rule for bin count selection
    let bin_width = 2.0 * iqr_val * (n as f64).powf(-1.0 / 3.0);
    let bin_count = ((max_val - min_val) / bin_width).ceil() as usize;
    let bin_count = bin_count.max(1);

    let mut counts = vec![0; bin_count];
    let actual_bin_width = (max_val - min_val) / bin_count as f64;

    for &x in data {
        let mut bin = ((x - min_val) / actual_bin_width) as usize;
        if bin >= bin_count {
            bin = bin_count - 1;
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
    precalc.insert("entropy", ent);
    ent
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean() {
        let mut precalc = HashMap::new();
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(mean(&data, &mut precalc), 3.0);
    }

    #[test]
    fn test_variance() {
        let mut precalc = HashMap::new();
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(variance(&data, &mut precalc), 2.0);
    }

    #[test]
    fn test_std() {
        let mut precalc = HashMap::new();
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(std(&data, &mut precalc), 2.0_f64.sqrt());
    }
}
