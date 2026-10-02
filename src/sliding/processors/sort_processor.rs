use crate::common::ColumnState;
use crate::types::Compute;

pub struct SortMetrics {
    pub first_max_idx: usize,
    pub last_max_idx: usize,
    pub first_min_idx: usize,
    pub last_min_idx: usize,
    pub median: f32,
    pub iqr: f32,
    pub entropy: f32,
}

pub struct SortProcessor;

impl SortProcessor {
    pub fn finalize(
        compute: Compute,
        values: &[f32],
        n: f32,
        state: &mut ColumnState,
    ) -> SortMetrics {
        let mut first_max_idx = 0;
        let mut last_max_idx = 0;
        let mut first_min_idx = 0;
        let mut last_min_idx = 0;
        let mut median = 0.0;
        let mut iqr = 0.0;
        let mut entropy = 0.0;

        if compute.contains(Compute::NEEDS_SORT) {
            if compute.intersects(Compute::FIRST_LOC_MAX | Compute::LAST_LOC_MAX | Compute::FIRST_LOC_MIN | Compute::LAST_LOC_MIN) {
                let mut found_max = false;
                let mut found_min = false;
                for (i, &v) in values.iter().enumerate() {
                    if v == state.max_value {
                        if !found_max {
                            first_max_idx = i;
                            found_max = true;
                        }
                        last_max_idx = i;
                    }
                    if v == state.min_value {
                        if !found_min {
                            first_min_idx = i;
                            found_min = true;
                        }
                        last_min_idx = i;
                    }
                }
            }
            if compute.intersects(Compute::MEDIAN | Compute::IQR | Compute::ENTROPY) {
                let mut copy: Vec<f32> = std::mem::take(&mut state.sort_buffer);
                copy.clear();
                copy.extend_from_slice(values);
                
                if compute.contains(Compute::MEDIAN) {
                    let n_len = copy.len();
                    if n_len % 2 == 1 {
                        median = *copy
                            .select_nth_unstable_by(n_len / 2, |a: &f32, b: &f32| {
                                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                            })
                            .1;
                    } else {
                        let mid = n_len / 2;
                        let m1 = *copy
                            .select_nth_unstable_by(mid, |a: &f32, b: &f32| {
                                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                            })
                            .1;
                        let m2 = *copy[..mid]
                            .iter()
                            .max_by(|a: &&f32, b: &&f32| {
                                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                            })
                            .unwrap();
                        median = (m1 + m2) / 2.0;
                    }
                }
                if compute.contains(Compute::IQR) {
                    copy.sort_unstable_by(|a, b| {
                        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                    });
                    let n_len = copy.len() as f32;
                    let get_q = |q: f32, data: &[f32]| -> f32 {
                        let idx = q * (n_len - 1.0);
                        let i = idx.floor() as usize;
                        let f = idx - i as f32;
                        if i >= data.len() - 1 {
                            data[data.len() - 1]
                        } else {
                            (1.0 - f) * data[i] + f * data[i + 1]
                        }
                    };
                    iqr = get_q(0.75, &copy) - get_q(0.25, &copy);
                }
                if compute.contains(Compute::ENTROPY) {
                    let range = state.max_value - state.min_value;
                    if range > 1e-9 {
                        let bins = 10;
                        let mut counts = vec![0usize; bins];
                        for &v in values {
                            let b =
                                (((v - state.min_value) / range) * (bins as f32 - 1.0)) as usize;
                            counts[b.min(bins - 1)] += 1;
                        }
                        for &c in &counts {
                            if c > 0 {
                                let p = c as f32 / n;
                                entropy -= p * p.ln();
                            }
                        }
                    }
                }
                state.sort_buffer = copy;
            }
        }

        SortMetrics {
            first_max_idx,
            last_max_idx,
            first_min_idx,
            last_min_idx,
            median,
            iqr,
            entropy,
        }
    }
}
