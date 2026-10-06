import subprocess
import numpy as np

rust_code = """
use tsfast::common::ColumnState;

fn ensure_higuchi_lengths(values: &[f32], state: &mut ColumnState) {
    let n = values.len();
    if n < 10 {
        return;
    }
    let k_max = n / 10 - 1;
    if k_max < 2 {
        return;
    }

    let mut higuchi_lk = vec![];
    let mut higuchi_k_values = vec![];

    for k in 1..=k_max {
        let mut lmk_sum = 0.0f64;
        let mut m_sums = vec![0.0f64; k];

        for j in k..n {
            let m = (j % k) + 1;
            m_sums[m - 1] += (values[j] - values[j - k]).abs() as f64;
        }

        for m in 1..=k {
            let iters = (n - m) / k;
            let norm_factor = (n - 1) as f64 / (iters * k) as f64;
            let lmk = (m_sums[m - 1] * norm_factor) / k as f64;
            lmk_sum += lmk;
        }

        let lk = lmk_sum / k as f64;
        higuchi_lk.push(lk);
        higuchi_k_values.push(k as f64);
    }
    println!("Rust lk: {:?}", &higuchi_lk[..5]);
}
fn main() {
    let x_sine = vec![0.0; 200];
    // not important right now, we can just run the same algo in python to see difference
}
"""
