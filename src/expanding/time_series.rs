use crate::common::ColumnState;
use crate::types::Feature;

#[inline(always)]
pub fn eval_time_series(
    feat: &Feature,
    n: f32,
    mean: f32,
    var: f32,
    m2: f32,
    mac_sum: f32,
    mc_sum: f32,
    state: &mut ColumnState,
    unique_c3_lags: &[u16],
    unique_paa_totals: &[u16],
    paa_boundaries: &[Vec<usize>],
    unique_autocorr_lags: &[u16],
    full_series: &[f32],
) -> Option<f32> {
    match feat {
        Feature::AutocorrLag1 if var > 1e-9 && n > 1.0 => {
            let x0 = full_series[0];
            let xn = full_series[full_series.len() - 1];
            let cov = state.sum_prod - mean * (2.0 * state.total_sum - x0 - xn)
                + (n - 1.0) * mean * mean;
            Some(cov / m2)
        }
        Feature::AutocorrFirst1e => {
            let n_val = full_series.len();
            if n_val > 1 {
                let n2 = n_val * 2;
                let fft_size_ac = crate::common::next_good_fft_size(n2);
                let mut planner = realfft::RealFftPlanner::<f32>::new();
                let r2c_ac = planner.plan_fft_forward(fft_size_ac);
                let c2r_ac = planner.plan_fft_inverse(fft_size_ac);

                let mut indata = vec![0.0; fft_size_ac];
                for (i, &v) in full_series.iter().enumerate() {
                    indata[i] = v - mean;
                }
                let mut outdata = r2c_ac.make_output_vec();
                r2c_ac.process(&mut indata, &mut outdata).unwrap();

                for c in &mut outdata {
                    *c = realfft::num_complex::Complex::new(c.norm_sqr(), 0.0);
                }

                let mut outdata_inv = c2r_ac.make_output_vec();
                c2r_ac.process(&mut outdata, &mut outdata_inv).unwrap();

                let m2_val = if n > 1.0 {
                    state.energy - (state.total_sum * state.total_sum) / n
                } else {
                    0.0
                };
                if m2_val.abs() > 1e-9 {
                    let scale = 1.0 / (fft_size_ac as f32);
                    let threshold = 0.36787944;
                    let mut found = false;
                    let mut first_lag = 0.0;
                    for l in 1..n_val {
                        let val = (outdata_inv[l] * scale) / m2_val;
                        if val < threshold {
                            first_lag = l as f32;
                            found = true;
                            break;
                        }
                    }
                    if found { Some(first_lag) } else { Some(0.0) }
                } else {
                    Some(0.0)
                }
            } else {
                Some(0.0)
            }
        }
        Feature::MeanAbsChange if n > 1.0 => Some(mac_sum / (n - 1.0)),
        Feature::MeanChange if n > 1.0 => Some(mc_sum / (n - 1.0)),
        Feature::CidCe => Some(state.sum_sq_diff.sqrt()),
        Feature::AbsSumChange => Some(mac_sum),
        Feature::Auc => Some(state.auc_sum),
        Feature::ZeroCrossingRate => Some(state.zcr_count as f32 / n),
        Feature::C3(lag) => {
            let l_idx = unique_c3_lags.iter().position(|&l| l == *lag).unwrap();
            let l = *lag as usize;
            if full_series.len() > 2 * l {
                Some(state.c3_sums[l_idx] / (full_series.len() - 2 * l) as f32)
            } else {
                Some(0.0)
            }
        }
        Feature::Paa(total, index) => {
            let t_idx = unique_paa_totals
                .iter()
                .position(|&t| t == *total)
                .unwrap();
            let b = &paa_boundaries[t_idx];
            let start = b[*index as usize];
            let end = b[*index as usize + 1];
            if start < end {
                let sum = if start == 0 {
                    state.prefix_sums[end - 1]
                } else {
                    state.prefix_sums[end - 1] - state.prefix_sums[start - 1]
                };
                Some(sum / (end - start) as f32)
            } else {
                Some(0.0)
            }
        }
        Feature::Autocorr(lag) if var > 1e-9 && full_series.len() > *lag as usize => {
            let l = *lag as usize;
            let l_idx = unique_autocorr_lags
                .iter()
                .position(|&lg| lg == *lag)
                .unwrap();
            let n_l = full_series.len() - l;

            let sum_xi = state.prefix_sums[n_l - 1];
            let sum_xil =
                state.prefix_sums[full_series.len() - 1] - state.prefix_sums[l - 1];
            let cov = state.autocorr_sums[l_idx] as f32 - mean * (sum_xi + sum_xil)
                + n_l as f32 * mean * mean;

            if var.abs() > 1e-9 {
                Some(cov / (n_l as f32 * var))
            } else {
                Some(0.0)
            }
        }
        Feature::TimeReversalAsymmetry(lag) if full_series.len() > 2 * *lag as usize => {
            let l = *lag as usize;
            let mut sum = 0.0;
            for i in 0..full_series.len() - 2 * l {
                sum += full_series[i + 2 * l].powi(2) * full_series[i + l]
                    - full_series[i + l] * full_series[i].powi(2);
            }
            Some(sum / (full_series.len() - 2 * l) as f32)
        }
        Feature::PartialAutocorr(lag)
            if var > 1e-9 && full_series.len() > *lag as usize =>
        {
            let l = *lag as usize;
            let mut r = std::mem::take(&mut state.pacf_buffer);
            r.clear();
            for k in 0..=l {
                if k == 0 {
                    r.push(1.0);
                    continue;
                }
                let k_idx = unique_autocorr_lags
                    .iter()
                    .position(|&lg| lg == k as u16)
                    .unwrap();
                let n_k = full_series.len() - k;
                let sum_xi = state.prefix_sums[n_k - 1];
                let sum_xk =
                    state.prefix_sums[full_series.len() - 1] - state.prefix_sums[k - 1];
                let cov = state.autocorr_sums[k_idx] as f32 - mean * (sum_xi + sum_xk)
                    + n_k as f32 * mean * mean;
                r.push(cov / (n_k as f32 * var));
            }
            let mut phi = vec![vec![0.0; l + 1]; l + 1];
            let mut error = r[0];
            if error.abs() < 1e-9 {
                return Some(0.0);
            }
            phi[1][1] = r[1] / r[0];
            error *= 1.0 - phi[1][1] * phi[1][1];
            for k in 1..l {
                let mut sum = 0.0;
                for i in 1..=k {
                    sum += phi[k][i] * r[k + 1 - i];
                }
                phi[k + 1][k + 1] = (r[k + 1] - sum) / error;
                for i in 1..=k {
                    phi[k + 1][i] = phi[k][i] - phi[k + 1][k + 1] * phi[k][k + 1 - i];
                }
                error *= 1.0 - phi[k + 1][k + 1] * phi[k + 1][k + 1];
                if error.abs() < 1e-9 {
                    break;
                }
            }
            let res = phi[l][l];
            state.pacf_buffer = r;
            Some(res)
        }
        Feature::ApproxEntropy(m, r_bits) if full_series.len() > *m as usize + 1 => {
            let m_val = *m as usize;
            let r = f32::from_bits(*r_bits);
            let mut buffer = std::mem::take(&mut state.approx_entropy_buffer);
            let res = crate::common::approx_entropy_phi(m_val, r, full_series, &mut buffer)
                - crate::common::approx_entropy_phi(m_val + 1, r, full_series, &mut buffer);
            state.approx_entropy_buffer = buffer;
            Some(res)
        }
        Feature::SumOfReoccurringValues => {
            use rustc_hash::FxHashMap;
            let mut counts = FxHashMap::default();
            for &v in full_series {
                let bits = v.to_bits();
                *counts.entry(bits).or_insert(0) += 1;
            }
            Some(counts
                .iter()
                .filter(|&(_, &count)| count > 1)
                .map(|(&bits, _)| f32::from_bits(bits))
                .sum())
        }
        Feature::SumOfReoccurringDataPoints => {
            use rustc_hash::FxHashMap;
            let mut counts = FxHashMap::default();
            for &v in full_series {
                let bits = v.to_bits();
                *counts.entry(bits).or_insert(0) += 1;
            }
            Some(counts
                .iter()
                .filter(|&(_, &count)| count > 1)
                .map(|(&bits, &count)| f32::from_bits(bits) * count as f32)
                .sum())
        }
        Feature::HasDuplicateMax => {
            let mut count = 0;
            for &v in full_series {
                if v == state.max_value {
                    count += 1;
                    if count > 1 {
                        break;
                    }
                }
            }
            Some(if count > 1 { 1.0 } else { 0.0 })
        }
        Feature::HasDuplicateMin => {
            let mut count = 0;
            for &v in full_series {
                if v == state.min_value {
                    count += 1;
                    if count > 1 {
                        break;
                    }
                }
            }
            Some(if count > 1 { 1.0 } else { 0.0 })
        }
        Feature::HasDuplicate => {
            let mut unique = rustc_hash::FxHashSet::default();
            let mut has_dup = false;
            for &v in full_series {
                if !unique.insert(v.to_bits()) {
                    has_dup = true;
                    break;
                }
            }
            Some(if has_dup { 1.0 } else { 0.0 })
        }
        _ => None,
    }
}
