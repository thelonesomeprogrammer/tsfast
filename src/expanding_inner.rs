pub fn mean(data: &[f64], out: &mut [f64]) {
    let mut sum = 0.0;
    for (i, &x) in data.iter().enumerate() {
        sum += x;
        out[i] = sum / (i + 1) as f64;
    }
}

pub fn variance(data: &[f64], out: &mut [f64]) {
    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    for (i, &x) in data.iter().enumerate() {
        sum += x;
        sum_sq += x * x;
        let n = (i + 1) as f64;
        let m = sum / n;
        let var = (sum_sq / n) - m * m;
        out[i] = if var < 0.0 { 0.0 } else { var };
    }
}

pub fn std(data: &[f64], out: &mut [f64]) {
    variance(data, out);
    for x in out.iter_mut() {
        *x = x.sqrt();
    }
}

pub fn energy(data: &[f64], out: &mut [f64]) {
    let mut sum_sq = 0.0;
    for (i, &x) in data.iter().enumerate() {
        sum_sq += x * x;
        out[i] = sum_sq;
    }
}

pub fn rms(data: &[f64], out: &mut [f64]) {
    energy(data, out);
    for (i, x) in out.iter_mut().enumerate() {
        *x = (*x / (i + 1) as f64).sqrt();
    }
}

pub fn min(data: &[f64], out: &mut [f64]) {
    let mut current_min = f64::INFINITY;
    for (i, &x) in data.iter().enumerate() {
        if x < current_min {
            current_min = x;
        }
        out[i] = current_min;
    }
}

pub fn max(data: &[f64], out: &mut [f64]) {
    let mut current_max = f64::NEG_INFINITY;
    for (i, &x) in data.iter().enumerate() {
        if x > current_max {
            current_max = x;
        }
        out[i] = current_max;
    }
}

pub fn autocorr_lag1(data: &[f64], out: &mut [f64]) {
    let n_total = data.len();
    if n_total == 0 {
        return;
    }
    out[0] = 0.0;
    if n_total < 2 {
        return;
    }

    let mut sum_x = data[0];
    let mut sum_x_lag = 0.0;
    let mut sum_x2 = data[0] * data[0];
    let mut sum_xy = 0.0;

    for i in 1..n_total {
        let x = data[i];
        let x_lag = data[i - 1];
        sum_x += x;
        sum_x_lag += x_lag;
        sum_x2 += x * x;
        sum_xy += x * x_lag;

        let n = (i + 1) as f64;
        let m = sum_x / n;
        
        // Numerator: sum_{j=1..i} (x_j - m)*(x_{j-1} - m)
        // = sum(x_j*x_{j-1}) - m*sum(x_j) - m*sum(x_{j-1}) + (n-1)*m^2
        // Wait, the sum for autocorr is usually over all i, but we only have i-1 pairs.
        // Let's use the definition: sum_{j=1}^{i} (x_j - m)(x_{j-1} - m) / sum_{j=0}^{i} (x_j - m)^2
        
        // Actually, many definitions use: 
        // num = sum_{j=1}^{i} (x_j - m)(x_{j-1} - m)
        // den = sum_{j=0}^{i} (x_j - m)^2 = sum(x_j^2) - n*m^2
        
        let num = sum_xy - m * (sum_x - data[0]) - m * sum_x_lag + (n - 1.0) * m * m;
        let den = sum_x2 - n * m * m;
        
        out[i] = if den == 0.0 { 0.0 } else { num / den };
    }
}

pub fn mean_abs_change(data: &[f64], out: &mut [f64]) {
    let n = data.len();
    if n == 0 { return; }
    out[0] = 0.0;
    let mut sum_diff = 0.0;
    for i in 1..n {
        sum_diff += (data[i] - data[i-1]).abs();
        out[i] = sum_diff / i as f64;
    }
}

pub fn mean_change(data: &[f64], out: &mut [f64]) {
    let n = data.len();
    if n == 0 { return; }
    out[0] = 0.0;
    let mut sum_diff = 0.0;
    for i in 1..n {
        sum_diff += data[i] - data[i-1];
        out[i] = sum_diff / i as f64;
    }
}

pub fn expanding_npaa(data: &[f64], segments: usize, idx: usize, out: &mut [f64]) {
    let n_total = data.len();
    if n_total == 0 { return; }
    
    let mut cumsum = Vec::with_capacity(n_total + 1);
    let mut cumsum_sq = Vec::with_capacity(n_total + 1);
    cumsum.push(0.0);
    cumsum_sq.push(0.0);
    
    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    for &x in data {
        sum += x;
        sum_sq += x * x;
        cumsum.push(sum);
        cumsum_sq.push(sum_sq);
    }
    
    for i in 0..n_total {
        let n = i + 1;
        let start = idx * n / segments;
        let end = (idx + 1) * n / segments;
        
        if start >= end || idx >= segments {
            out[i] = f64::NAN;
            continue;
        }
        
        let m = cumsum[n] / n as f64;
        let var = (cumsum_sq[n] / n as f64) - m * m;
        let std = if var <= 0.0 { 1.0 } else { var.sqrt() };
        
        let seg_sum = cumsum[end] - cumsum[start];
        let seg_len = (end - start) as f64;
        
        let seg_mean = seg_sum / seg_len;
        out[i] = (seg_mean - m) / std;
    }
}

pub fn expanding_paa(data: &[f64], segments: usize, idx: usize, out: &mut [f64]) {
    let n_total = data.len();
    if n_total == 0 { return; }
    
    let mut cumsum = Vec::with_capacity(n_total + 1);
    cumsum.push(0.0);
    
    let mut sum = 0.0;
    for &x in data {
        sum += x;
        cumsum.push(sum);
    }
    
    for i in 0..n_total {
        let n = i + 1;
        let start = idx * n / segments;
        let end = (idx + 1) * n / segments;
        
        if start >= end || idx >= segments {
            out[i] = f64::NAN;
            continue;
        }
        
        let seg_sum = cumsum[end] - cumsum[start];
        let seg_len = (end - start) as f64;
        
        out[i] = seg_sum / seg_len;
    }
}

pub fn expanding_npta(data: &[f64], segments: usize, idx: usize, out: &mut [f64]) {
    let n_total = data.len();
    if n_total == 0 { return; }
    
    let mut cumsum = Vec::with_capacity(n_total + 1);
    let mut cumsum_sq = Vec::with_capacity(n_total + 1);
    cumsum.push(0.0);
    cumsum_sq.push(0.0);
    
    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    for &x in data {
        sum += x;
        sum_sq += x * x;
        cumsum.push(sum);
        cumsum_sq.push(sum_sq);
    }
    
    for i in 0..n_total {
        let n = i + 1;
        let start = idx * n / segments;
        let end = (idx + 1) * n / segments;
        
        if end - start < 2 || idx >= segments {
            out[i] = 0.0;
            continue;
        }
        
        let m = cumsum[n] / n as f64;
        let var = (cumsum_sq[n] / n as f64) - m * m;
        let std = if var <= 0.0 { 1.0 } else { var.sqrt() };
        
        let first = data[start];
        let last = data[end - 1];
        
        out[i] = (last - first) / (std * (end - start - 1) as f64);
    }
}

pub fn extract_expanding(
    data: &[f64],
    features: &[&str],
    registry: &crate::registry::Registry,
) -> Vec<f64> {
    let n_samples = data.len();
    let n_features = features.len();
    let mut out = vec![0.0; n_samples * n_features];

    for (feat_idx, &feat_name) in features.iter().enumerate() {
        let start_idx = feat_idx * n_samples;
        let out_slice = &mut out[start_idx..start_idx + n_samples];

        if let Some(feat_fn) = registry.expanding_features.get(feat_name) {
            feat_fn(data, out_slice);
        } else if feat_name.starts_with("npaa-") {
            let parts: Vec<&str> = feat_name.split('-').collect();
            if parts.len() == 3 {
                if let (Ok(segments), Ok(idx)) = (parts[1].parse::<usize>(), parts[2].parse::<usize>()) {
                    expanding_npaa(data, segments, idx, out_slice);
                    continue;
                }
            }
            out_slice.fill(f64::NAN);
        } else if feat_name.starts_with("paa-") {
            let parts: Vec<&str> = feat_name.split('-').collect();
            if parts.len() == 3 {
                if let (Ok(segments), Ok(idx)) = (parts[1].parse::<usize>(), parts[2].parse::<usize>()) {
                    expanding_paa(data, segments, idx, out_slice);
                    continue;
                }
            }
            out_slice.fill(f64::NAN);
        } else if feat_name.starts_with("npta-") {
            let parts: Vec<&str> = feat_name.split('-').collect();
            if parts.len() == 3 {
                if let (Ok(segments), Ok(idx)) = (parts[1].parse::<usize>(), parts[2].parse::<usize>()) {
                    expanding_npta(data, segments, idx, out_slice);
                    continue;
                }
            }
            out_slice.fill(f64::NAN);
        } else if let Some(batch_fn) = registry.batch_features.get(feat_name) {
            // Fallback for features that don't have an optimized expanding version.
            for i in 0..n_samples {
                out_slice[i] = batch_fn(&data[0..=i]);
            }
        } else {
            out_slice.fill(f64::NAN);
        }
    }
    out
}
