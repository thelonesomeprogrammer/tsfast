use crate::types::Compute;
use std::simd::f32x4;
use std::simd::num::SimdFloat;

pub struct SortProcessor;

impl SortProcessor {
    pub fn process_running_sorted(
        compute: Compute,
        full_series: &[f32],
        running_sorted: &mut Vec<f32>,
        state: &mut crate::common::ColumnState,
    ) -> (f32, f32, f32) {
        let mut median = 0.0;
        let mut iqr = 0.0;
        let mut median_abs_dev = 0.0;

        if compute.intersects(
            Compute::MEDIAN
                | Compute::IQR
                | Compute::QUANTILE
                | Compute::MEAN_N_ABS_MAX
                | Compute::MEDIAN_ABS_DEV,
        ) {
            if running_sorted.len() < full_series.len() {
                running_sorted.extend_from_slice(&full_series[running_sorted.len()..]);
                running_sorted
                    .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
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

        if compute.contains(Compute::MEDIAN_ABS_DEV) {
            let med_vec = f32x4::splat(median);
            let mut abs_devs: Vec<f32> = std::mem::take(&mut state.mad_buffer);
            abs_devs.clear();
            abs_devs.resize(full_series.len(), 0.0);

            let mut i = 0;
            while i + 4 <= full_series.len() {
                let chunk = f32x4::from_slice(&full_series[i..i + 4]);
                let diff = (chunk - med_vec).abs();
                diff.copy_to_slice(&mut abs_devs[i..i + 4]);
                i += 4;
            }
            for j in i..full_series.len() {
                abs_devs[j] = (full_series[j] - median).abs();
            }

            let n_len = abs_devs.len();
            if n_len == 0 {
                median_abs_dev = f32::NAN;
            } else if n_len > 0 {
                if n_len % 2 == 1 {
                    median_abs_dev = *abs_devs
                        .select_nth_unstable_by(n_len / 2, |a: &f32, b: &f32| {
                            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .1;
                } else {
                    let mid = n_len / 2;
                    let m1 = *abs_devs
                        .select_nth_unstable_by(mid, |a: &f32, b: &f32| {
                            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .1;
                    let m2 = *abs_devs[..mid]
                        .iter()
                        .max_by(|a: &&f32, b: &&f32| {
                            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .unwrap_or(&0.0);
                    median_abs_dev = (m1 + m2) / 2.0;
                }
            }
            state.mad_buffer = abs_devs;
        }

        (median, iqr, median_abs_dev)
    }
}
