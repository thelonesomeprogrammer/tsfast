use crate::features::signal::paa;
use crate::registry::Registry;
use std::collections::{HashMap, HashSet};

static DEPENDENCIES: &[(&str, &[&str])] = &[
    ("std", &["variance"]),
    ("skew", &["mean", "std"]),
    ("kurtosis", &["mean", "std"]),
    ("mad", &["median"]),
    ("entropy", &["min", "max", "iqr"]),
    ("rms", &["energy"]),
    ("autocorr_lag1", &["mean"]),
    ("intercept", &["slope"]), // Just to ensure they share linear regression result if both requested
];

pub fn extract(data: &[f64], features: &[&str], registry: &Registry) -> Vec<f64> {
    let mut sorted_features = features.to_vec();
    featuresort(&mut sorted_features);

    let mut computed = HashMap::new();

    for feat_name in &sorted_features {
        if computed.contains_key(feat_name) {
            continue;
        }

        if let Some(feat_fn) = registry.batch_features.get(feat_name) {
            feat_fn(data, &mut computed);
        } else if feat_name.starts_with("paa-") {
            let parts: Vec<&str> = feat_name.split('-').collect();
            if parts.len() == 3
                && let (Ok(segments), Ok(idx)) =
                    (parts[1].parse::<usize>(), parts[2].parse::<usize>())
            {
                paa(data, segments, idx, &mut computed);
            }
        }
    }

    // Return results in the original requested order
    features
        .iter()
        .map(|&f| *computed.get(f).unwrap_or(&f64::NAN))
        .collect()
}

fn featuresort(features: &mut Vec<&str>) {
    let mut sorted = Vec::new();
    let mut visited = HashSet::new();
    let mut temp_visited = HashSet::new();

    let dep_map: HashMap<&str, &[&str]> = DEPENDENCIES.iter().copied().collect();

    fn visit<'a>(
        feat: &'a str,
        dep_map: &HashMap<&'a str, &'a [&'a str]>,
        visited: &mut HashSet<&'a str>,
        temp_visited: &mut HashSet<&'a str>,
        sorted: &mut Vec<&'a str>,
    ) {
        if visited.contains(feat) {
            return;
        }
        if temp_visited.contains(feat) {
            // Cycle detected, but we assume DAG for features
            return;
        }

        temp_visited.insert(feat);

        if let Some(deps) = dep_map.get(feat) {
            for &dep in *deps {
                visit(dep, dep_map, visited, temp_visited, sorted);
            }
        }

        temp_visited.remove(feat);
        visited.insert(feat);
        sorted.push(feat);
    }

    let original_requested: Vec<&str> = features.clone();
    for &feat in &original_requested {
        visit(feat, &dep_map, &mut visited, &mut temp_visited, &mut sorted);
    }

    *features = sorted;
}
