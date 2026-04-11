use crate::registry::Cache;
use std::collections::HashMap;

pub fn mean(data: &[f64], cache: &mut Cache) {
    let mean = data.iter().sum::<f64>() / data.len() as f64;
    cache.mean = Some(mean);
}

pub fn variance(data: &[f64], cache: &mut Cache) {
    let mut mean = 0.0;
    let mut m2 = 0.0;
    let n = data.len();

    for (i, &x) in data.iter().enumerate() {
        let delta = x - mean;
        mean += delta / (i + 1) as f64;
        let delta2 = x - mean;
        m2 += delta * delta2;
    }

    cache.mean = Some(mean);

    let var = m2 / n as f64;
    cache.variance = Some(var);
}

pub fn std(_data: &[f64], cache: &mut Cache) {
    if let Some(var) = cache.variance {
        cache.std = Some(var.sqrt());
    }
}

pub fn min(data: &[f64], cache: &mut Cache) {
    let m = data.iter().copied().fold(f64::INFINITY, f64::min);
    cache.min = Some(m);
}

pub fn max(data: &[f64], cache: &mut Cache) {
    let m = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    cache.max = Some(m);
}
pub fn median(data: &[f64], cache: &mut Cache) {
    let mut v = data.to_vec();
    let median = sub_median(v.as_mut_slice());
    cache.median = Some(median);
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

pub fn skew(data: &[f64], cache: &mut Cache) {
    let n = data.len();

    if let Some(m) = cache.mean
        && let Some(s) = cache.std
    {
        if s != 0.0 {
            let m3 = data.iter().map(|&x| (x - m).powi(3)).sum::<f64>() / n as f64;
            let skew = m3 / s.powi(3);
            cache.skew = Some(skew);
        } else {
            cache.skew = Some(0.0);
        }
    }
}

pub fn kurtosis(data: &[f64], cache: &mut Cache) {
    let n = data.len();

    if let Some(m) = cache.mean
        && let Some(s) = cache.std
    {
        if s != 0.0 {
            let m4 = data.iter().map(|&x| (x - m).powi(4)).sum::<f64>() / n as f64;
            let kurtosis = m4 / s.powi(4) - 3.0;
            cache.kurtosis = Some(kurtosis);
        } else {
            cache.kurtosis = Some(0.0);
        }
    }
}

pub fn mad(data: &[f64], cache: &mut Cache) {
    if let Some(med) = cache.median {
        let mut deviations: Vec<f64> = data.iter().map(|&x| (x - med).abs()).collect();
        let mad = sub_median(deviations.as_mut_slice());
        cache.mad = Some(mad);
    }
}

pub fn iqr(data: &[f64], cache: &mut Cache) {
    let mut data = data.to_vec();
    let n = data.len();
    let q1 = sub_median(&mut data[..n / 2]);
    let q3 = sub_median(&mut data[(n + 1) / 2..]);
    let iqr = q3 - q1;
    cache.iqr = Some(iqr);
}

pub fn entropy(data: &[f64], cache: &mut Cache) {
    let n = data.len();

    if let Some(min) = cache.min
        && let Some(max) = cache.max
        && let Some(iqr) = cache.iqr
    {
        if (max - min) < 1e-9 {
            cache.entropy = Some(0.0);
            return;
        }

        // Use Freedman-Diaconis rule for bin count selection
        let bin_width = 2.0 * iqr * (n as f64).powf(-1.0 / 3.0);
        let bin_count = ((max - min) / bin_width).ceil() as usize;
        let bin_count = bin_count.max(1);

        let mut counts = vec![0; bin_count];
        let actual_bin_width = (max - min) / bin_count as f64;

        for &x in data {
            let mut bin = ((x - min) / actual_bin_width) as usize;
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
        cache.entropy = Some(ent);
    }
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
