use std::time::Instant;

fn main() {
    let n = 2000;
    let data: Vec<f32> = (0..n).map(|i| (i as f32).sin()).collect();
    let m = 2;
    let r = 0.2;

    let start = Instant::now();
    let res = phi_old(m, r, &data) - phi_old(m + 1, r, &data);
    println!("Old: {} in {:?}", res, start.elapsed());

    let mut buf = Vec::new();
    let start = Instant::now();
    let res2 = phi_new(m, r, &data, &mut buf) - phi_new(m + 1, r, &data, &mut buf);
    println!("New: {} in {:?}", res2, start.elapsed());
}

fn phi_old(m: usize, r: f32, data: &[f32]) -> f32 {
    let n = data.len();
    let mut result = 0.0;
    for i in 0..n - m + 1 {
        let mut count = 0;
        for j in 0..n - m + 1 {
            let mut max_diff: f32 = 0.0;
            for k in 0..m {
                max_diff = max_diff.max((data[i + k] - data[j + k]).abs());
            }
            if max_diff <= r {
                count += 1;
            }
        }
        result += (count as f32 / (n - m + 1) as f32).ln();
    }
    result / (n - m + 1) as f32
}

fn phi_new(m: usize, r: f32, data: &[f32], sorted_idx: &mut Vec<usize>) -> f32 {
    let n = data.len();
    let mut result = 0.0;

    sorted_idx.clear();
    sorted_idx.extend(0..n - m + 1);
    sorted_idx.sort_unstable_by(|&a, &b| {
        data[a]
            .partial_cmp(&data[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    for i in 0..n - m + 1 {
        let mut count = 0;
        let target = data[i];
        let start_pos =
            sorted_idx.partition_point(|&idx| data[idx] < target - r);
        let end_pos =
            sorted_idx.partition_point(|&idx| data[idx] <= target + r);

        for &j in &sorted_idx[start_pos..end_pos] {
            let mut max_diff: f32 = 0.0;
            for k in 0..m {
                max_diff = max_diff.max((data[i + k] - data[j + k]).abs());
            }
            if max_diff <= r {
                count += 1;
            }
        }
        result += (count as f32 / (n - m + 1) as f32).ln();
    }
    result / (n - m + 1) as f32
}
