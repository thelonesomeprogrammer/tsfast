use crate::common::ColumnState;

#[inline(always)]
pub fn compute_distribution_features(
    compute: &crate::types::FastBitArray,
    values: &[f32],
    state: &mut ColumnState,
    n: f32,
    median: &mut f32,
    iqr: &mut f32,
    entropy: &mut f32,
) -> Option<Vec<f32>> {
    let mut sorted_copy: Option<Vec<f32>> = None;

    if compute.any([6, 10, 11, 49]) {
        // ⚡ Bolt Optimization: Reuse sort_buffer to prevent inner loop memory allocations
        let mut copy = std::mem::take(&mut state.sort_buffer);
        copy.clear();
        copy.extend_from_slice(values);
        copy.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        if compute[6] {
            let n_len = copy.len();
            if n_len > 0 {
                if n_len % 2 == 1 {
                    *median = copy[n_len / 2];
                } else {
                    *median = (copy[n_len / 2] + copy[n_len / 2 - 1]) / 2.0;
                }
            }
        }
        if compute[10] {
            let n_f = n;
            let get_q = |q: f32, data: &[f32]| -> f32 {
                if data.is_empty() {
                    return 0.0;
                }
                let idx = q * (n_f - 1.0);
                let i = idx.floor() as usize;
                let f = idx - i as f32;
                if i >= data.len() - 1 {
                    data[data.len() - 1]
                } else {
                    (1.0 - f) * data[i] + f * data[i + 1]
                }
            };
            *iqr = get_q(0.75, &copy) - get_q(0.25, &copy);
        }
        if compute[11] {
            let range = state.max_value - state.min_value;
            if range > 1e-9 {
                let bins = 10;
                let mut counts = vec![0usize; bins];
                for &v in values {
                    let b = (((v - state.min_value) / range) * (bins as f32 - 1.0)) as usize;
                    counts[b.min(bins - 1)] += 1;
                }
                for &c in &counts {
                    if c > 0 {
                        let p = c as f32 / n;
                        *entropy -= p * p.ln();
                    }
                }
            }
        }
        sorted_copy = Some(copy);
    }
    sorted_copy
}
