use crate::types::Compute;

pub struct SortProcessor;

impl SortProcessor {
    pub fn process_running_sorted(
        compute: Compute,
        full_series: &[f32],
        running_sorted: &mut Vec<f32>,
    ) -> (f32, f32) {
        let mut median = 0.0;
        let mut iqr = 0.0;

        if compute.intersects(Compute::MEDIAN | Compute::IQR | Compute::QUANTILE | Compute::MEAN_N_ABS_MAX) {
            if running_sorted.len() < full_series.len() {
                let mut new_elements: Vec<f32> = full_series[running_sorted.len()..].to_vec();
                new_elements
                    .sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                if running_sorted.is_empty() {
                    *running_sorted = new_elements;
                } else {
                    let mut merged = Vec::with_capacity(running_sorted.len() + new_elements.len());
                    let mut i = 0;
                    let mut j = 0;
                    while i < running_sorted.len() && j < new_elements.len() {
                        if running_sorted[i] <= new_elements[j] {
                            merged.push(running_sorted[i]);
                            i += 1;
                        } else {
                            merged.push(new_elements[j]);
                            j += 1;
                        }
                    }
                    merged.extend_from_slice(&running_sorted[i..]);
                    merged.extend_from_slice(&new_elements[j..]);
                    *running_sorted = merged;
                }
            }
            let n_size = running_sorted.len();
            if n_size > 0 {
                if compute.contains(Compute::MEDIAN) {
                    if n_size % 2 == 1 {
                        median = running_sorted[n_size / 2];
                    } else {
                        let mid = n_size / 2;
                        median = (running_sorted[mid] + running_sorted[mid - 1]) / 2.0;
                    }
                }
                if compute.contains(Compute::IQR) {
                    let n_f = n_size as f32;
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
                    iqr = get_q(0.75, running_sorted) - get_q(0.25, running_sorted);
                }
            }
        }
        
        (median, iqr)
    }
}
