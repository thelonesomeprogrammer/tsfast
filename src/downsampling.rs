use std::simd::prelude::*;
use std::simd::num::SimdFloat;

/// M4 downsampling algorithm.
/// For each bin, it extracts the first, last, minimum, and maximum values, 
/// preserving their original order.
pub fn m4_downsample(data: &[f64], n_bins: usize) -> Vec<f64> {
    let n = data.len();
    if n <= n_bins * 4 || n_bins == 0 {
        return data.to_vec();
    }

    let bin_size = n / n_bins;
    let mut result = Vec::with_capacity(n_bins * 4);

    for i in 0..n_bins {
        let start = i * bin_size;
        let end = if i == n_bins - 1 { n } else { (i + 1) * bin_size };
        let bin = &data[start..end];

        if bin.is_empty() {
            continue;
        }

        let first = bin[0];
        let last = bin[bin.len() - 1];

        let mut min_val = first;
        let mut min_idx = 0;
        let mut max_val = first;
        let mut max_idx = 0;

        // Linear pass to find min/max and their indices to preserve temporal order.
        // We use linear pass here for clarity and correctness in index tracking.
        for (j, &val) in bin.iter().enumerate().skip(1) {
            if val < min_val {
                min_val = val;
                min_idx = j;
            }
            if val > max_val {
                max_val = val;
                max_idx = j;
            }
        }

        let mut points = vec![(0, first), (bin.len() - 1, last), (min_idx, min_val), (max_idx, max_val)];
        // Sort by index to maintain original temporal order.
        points.sort_by_key(|p| p.0);
        // Deduplicate in case first/last/min/max are the same point.
        points.dedup_by_key(|p| p.0);

        for (_, val) in points {
            result.push(val);
        }
    }

    result
}
