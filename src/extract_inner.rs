use crate::registry::Registry;

pub fn extract(data: &[f64], features: &[&str], registry: &Registry) -> Vec<f64> {
    let mut results = Vec::with_capacity(features.len());
    for &feat_name in features {
        if let Some(feat_fn) = registry.batch_features.get(feat_name) {
            results.push(feat_fn(data));
        } else if feat_name.starts_with("npaa-") {
            let parts: Vec<&str> = feat_name.split('-').collect();
            if parts.len() == 3 {
                if let (Ok(segments), Ok(idx)) = (parts[1].parse::<usize>(), parts[2].parse::<usize>()) {
                    results.push(crate::features::signal::npaa(data, segments, idx));
                    continue;
                }
            }
            results.push(f64::NAN);
        } else if feat_name.starts_with("paa-") {
            let parts: Vec<&str> = feat_name.split('-').collect();
            if parts.len() == 3 {
                if let (Ok(segments), Ok(idx)) = (parts[1].parse::<usize>(), parts[2].parse::<usize>()) {
                    results.push(crate::features::signal::paa(data, segments, idx));
                    continue;
                }
            }
            results.push(f64::NAN);
        } else if feat_name.starts_with("npta-") {
            let parts: Vec<&str> = feat_name.split('-').collect();
            if parts.len() == 3 {
                if let (Ok(segments), Ok(idx)) = (parts[1].parse::<usize>(), parts[2].parse::<usize>()) {
                    results.push(crate::features::signal::npta(data, segments, idx));
                    continue;
                }
            }
            results.push(f64::NAN);
        } else {
            // Handle missing feature? For now, maybe push 0.0 or NaN.
            results.push(f64::NAN);
        }
    }
    results
}
