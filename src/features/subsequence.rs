use crate::context::FeatureContext;
use crate::types::{AggFunc, Feature};

pub fn eval_subsequence(feat: &Feature, context: &mut FeatureContext) -> Option<f32> {
    match feat {
        Feature::QuerySimilarityCount(l, threshold_bits) => {
            let l = *l as usize;
            if l == 0 || context.values.len() < l {
                return None;
            }
            let threshold = f32::from_bits(*threshold_bits);
            let q = &context.values[..l];
            let distances = mass_distance_profile(context.values, q);

            let mut count = 0;
            for &d in &distances {
                if d <= threshold {
                    count += 1;
                }
            }
            Some(count as f32)
        }
        Feature::MatrixProfile(l, agg) => {
            let l = *l as usize;
            if l == 0 || context.values.len() < l * 2 {
                if context.values.len() < l {
                    return None;
                }
            }
            let profile = stomp_matrix_profile(context.values, l);
            if profile.is_empty() {
                return None;
            }

            match agg {
                AggFunc::Min => Some(profile.iter().copied().fold(f32::INFINITY, f32::min)),
                AggFunc::Max => Some(profile.iter().copied().fold(f32::NEG_INFINITY, f32::max)),
                AggFunc::Mean => {
                    let sum: f32 = profile.iter().sum();
                    Some(sum / profile.len() as f32)
                }
                AggFunc::Var => {
                    let mean = profile.iter().sum::<f32>() / profile.len() as f32;
                    let var = profile.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / (profile.len() - 1).max(1) as f32;
                    Some(var)
                }
            }
        }
        _ => None,
    }
}

fn mass_distance_profile(x: &[f32], q: &[f32]) -> Vec<f32> {
    let m = q.len();
    let n = x.len();
    if n < m {
        return vec![];
    }

    let mut q_mean = 0.0;
    for &v in q {
        q_mean += v;
    }
    q_mean /= m as f32;
    let mut q_std = 0.0;
    for &v in q {
        q_std += (v - q_mean) * (v - q_mean);
    }
    q_std = (q_std / m as f32).sqrt();
    if q_std < 1e-6 {
        q_std = 1.0;
    }

    let mut q_norm = Vec::with_capacity(m);
    for &v in q {
        q_norm.push((v - q_mean) / q_std);
    }

    let mut distances = Vec::with_capacity(n - m + 1);

    for i in 0..=n - m {
        let window = &x[i..i+m];

        let mut w_mean = 0.0;
        for &v in window {
            w_mean += v;
        }
        w_mean /= m as f32;

        let mut w_std = 0.0;
        for &v in window {
            w_std += (v - w_mean) * (v - w_mean);
        }
        w_std = (w_std / m as f32).sqrt();
        if w_std < 1e-6 {
            w_std = 1.0;
        }

        let mut dot = 0.0;
        for j in 0..m {
            let w_norm = (window[j] - w_mean) / w_std;
            dot += q_norm[j] * w_norm;
        }

        let d = (2.0 * m as f32 * (1.0 - dot / m as f32).max(0.0)).sqrt();
        distances.push(d);
    }

    distances
}

fn stomp_matrix_profile(x: &[f32], m: usize) -> Vec<f32> {
    let n = x.len();
    if n <= m {
        return vec![0.0];
    }

    let l = n - m + 1;
    let mut means = vec![0.0; l];
    let mut stds = vec![0.0; l];

    for i in 0..l {
        let window = &x[i..i+m];
        let mut mean = 0.0;
        for &v in window {
            mean += v;
        }
        mean /= m as f32;

        let mut std = 0.0;
        for &v in window {
            std += (v - mean) * (v - mean);
        }
        std = (std / m as f32).sqrt();
        if std < 1e-6 {
            std = 1.0; // avoid div by zero, standardize flat line
        }
        means[i] = mean;
        stds[i] = std;
    }

    let mut profile = vec![f32::INFINITY; l];
    // stumpy / matrixprofile: neighbours within ceil(m / 4) are trivial matches.
    let exclusion_zone = m.div_ceil(4);

    for i in 0..l {
        for j in 0..l {
            if i.abs_diff(j) > exclusion_zone {
                let mut dot = 0.0;
                for k in 0..m {
                    let vi = (x[i + k] - means[i]) / stds[i];
                    let vj = (x[j + k] - means[j]) / stds[j];
                    dot += vi * vj;
                }
                let d = (2.0 * m as f32 * (1.0 - dot / m as f32).max(0.0)).sqrt();
                if d < profile[i] {
                    profile[i] = d;
                }
            }
        }
    }

    profile
}
