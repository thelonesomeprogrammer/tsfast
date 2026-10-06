//! Ports of the wavelet routines the reference libraries call, so tsfast
//! matches them: `pywt.cwt(x, scales, "mexh")` (TSFEL wavelet features,
//! tsfresh cwt_coefficients) and `scipy.signal.find_peaks_cwt` with tsfresh's
//! ricker wavelet (tsfresh number_cwt_peaks). Computed in f64.

use std::sync::OnceLock;

/// pywt.integrate_wavelet("mexh", precision=12) (pywt.cwt's default): cumulative
/// integral of the Mexican hat sampled at linspace(-8, 8, 4096), and the step.
fn mexh_integral() -> &'static (Vec<f64>, f64) {
    static INT_PSI: OnceLock<(Vec<f64>, f64)> = OnceLock::new();
    INT_PSI.get_or_init(|| {
        let len = 4096;
        let step = 16.0 / (len - 1) as f64;
        let c = 2.0 / (3.0f64.sqrt() * std::f64::consts::PI.powf(0.25));
        let mut acc = 0.0;
        let int_psi = (0..len)
            .map(|i| {
                let x = -8.0 + i as f64 * step;
                acc += c * (1.0 - x * x) * (-x * x / 2.0).exp();
                acc * step
            })
            .collect();
        (int_psi, step)
    })
}

/// One row of pywt.cwt(values, [scale], "mexh"): `values.len()` coefficients.
pub fn mexh_cwt(values: &[f32], scale: f64) -> Vec<f64> {
    let n = values.len();
    let (int_psi, step) = mexh_integral();
    // j = floor(arange(scale * 16 + 1) / (scale * step)), clipped to int_psi.
    let kernel: Vec<f64> = (0..=(scale * 16.0) as usize)
        .map(|i| (i as f64 / (scale * step)) as usize)
        .filter(|&j| j < int_psi.len())
        .map(|j| int_psi[j])
        .rev()
        .collect();
    if n == 0 || kernel.is_empty() {
        return vec![0.0; n];
    }

    // Full convolution, then coef = -sqrt(scale) * diff(conv), centre-trimmed to n.
    let conv_len = n + kernel.len() - 1;
    let conv_at = |k: usize| -> f64 {
        let lo = k.saturating_sub(kernel.len() - 1);
        let hi = k.min(n - 1);
        (lo..=hi).map(|m| values[m] as f64 * kernel[k - m]).sum()
    };
    let d = (conv_len - 1 - n) as f64 / 2.0;
    let start = d.floor() as usize;
    let sqrt_s = scale.sqrt();
    let mut prev = conv_at(start);
    (0..n)
        .map(|i| {
            let next = conv_at(start + i + 1);
            let c = -sqrt_s * (next - prev);
            prev = next;
            c
        })
        .collect()
}

/// tsfresh `_ricker(points, a)` (scipy's removed `ricker`).
fn ricker(points: usize, a: f64) -> Vec<f64> {
    let amp = 2.0 / ((3.0 * a).sqrt() * std::f64::consts::PI.powf(0.25));
    let centre = (points as f64 - 1.0) / 2.0;
    (0..points)
        .map(|i| {
            let x2 = (i as f64 - centre).powi(2);
            amp * (1.0 - x2 / (a * a)) * (-x2 / (2.0 * a * a)).exp()
        })
        .collect()
}

/// scipy's old `signal.cwt` row: convolve(data, ricker(min(10w, n), w)[::-1], "same").
fn ricker_cwt(values: &[f32], width: f64) -> Vec<f64> {
    let n = values.len();
    let points = ((10.0 * width) as usize).min(n);
    let mut wav = ricker(points, width);
    wav.reverse();
    let offset = (points - 1) / 2;
    (0..n)
        .map(|i| {
            let k = i + offset; // index into the full convolution
            let lo = k.saturating_sub(points - 1);
            let hi = k.min(n - 1);
            (lo..=hi).map(|m| values[m] as f64 * wav[k - m]).sum()
        })
        .collect()
}

/// `len(scipy.signal.find_peaks_cwt(values, widths=1..=max_width, wavelet=_ricker))`
/// with scipy's defaults: max_distances = widths / 4, gap_thresh = widths[0],
/// min_length = ceil(rows / 4), window_size = ceil(n / 20), min_snr = 1,
/// noise_perc = 10.
pub fn number_cwt_peaks(values: &[f32], max_width: usize) -> usize {
    let n = values.len();
    if n == 0 || max_width == 0 {
        return 0;
    }
    let widths: Vec<f64> = (1..=max_width).map(|w| w as f64).collect();
    let cwt: Vec<Vec<f64>> = widths.iter().map(|&w| ricker_cwt(values, w)).collect();
    let rows = cwt.len();

    // Relative maxima along each row; edges compare with themselves ("clip"), so never maxima.
    let is_max = |row: &[f64], i: usize| {
        let left = row[i.saturating_sub(1)];
        let right = row[(i + 1).min(n - 1)];
        row[i] > left && row[i] > right
    };
    let max_cols: Vec<Vec<usize>> = cwt
        .iter()
        .map(|row| (0..n).filter(|&i| is_max(row, i)).collect())
        .collect();

    // _identify_ridge_lines: follow maxima from the widest row down.
    struct Line {
        rows: Vec<usize>,
        cols: Vec<usize>,
        gap: usize,
    }
    let Some(start_row) = (0..rows).rev().find(|&r| !max_cols[r].is_empty()) else {
        return 0;
    };
    let gap_thresh = widths[0].ceil() as usize;
    let mut lines: Vec<Line> = max_cols[start_row]
        .iter()
        .map(|&c| Line {
            rows: vec![start_row],
            cols: vec![c],
            gap: 0,
        })
        .collect();
    let mut finished: Vec<Line> = Vec::new();
    for row in (0..start_row).rev() {
        for line in &mut lines {
            line.gap += 1;
        }
        let prev_cols: Vec<usize> = lines.iter().map(|l| *l.cols.last().unwrap_or(&0)).collect();
        let max_distance = widths[row] / 4.0;
        for &col in &max_cols[row] {
            // First closest previous ridge, as np.argmin.
            let closest = prev_cols
                .iter()
                .enumerate()
                .min_by_key(|&(_, &pc)| pc.abs_diff(col))
                .filter(|&(_, &pc)| pc.abs_diff(col) as f64 <= max_distance)
                .map(|(i, _)| i);
            match closest {
                Some(i) => {
                    lines[i].rows.push(row);
                    lines[i].cols.push(col);
                    lines[i].gap = 0;
                }
                None => lines.push(Line {
                    rows: vec![row],
                    cols: vec![col],
                    gap: 0,
                }),
            }
        }
        for i in (0..lines.len()).rev() {
            if lines[i].gap > gap_thresh {
                finished.push(lines.remove(i));
            }
        }
    }
    finished.extend(lines);

    // _filter_ridge_lines: minimum length and SNR against the 10th percentile
    // of the narrowest row in a window around the ridge's start.
    let min_length = (rows as f64 / 4.0).ceil() as usize;
    let window = (n as f64 / 20.0).ceil() as usize;
    let (half, odd) = (window / 2, window % 2);
    let row0 = &cwt[0];
    let noise_at = |col: usize| {
        let lo = col.saturating_sub(half);
        let hi = (col + half + odd).min(n);
        let mut w: Vec<f64> = row0[lo..hi].to_vec();
        w.sort_by(f64::total_cmp);
        let pos = 0.10 * (w.len() - 1) as f64;
        let (i, frac) = (pos.floor() as usize, pos.fract());
        w[i] + (w[(i + 1).min(w.len() - 1)] - w[i]) * frac
    };
    finished
        .iter()
        .filter(|line| {
            if line.rows.len() < min_length {
                return false;
            }
            // The ridge point on the lowest row (scipy sorts each line by row).
            let k = (0..line.rows.len())
                .min_by_key(|&k| line.rows[k])
                .unwrap_or(0);
            let (r, c) = (line.rows[k], line.cols[k]);
            // scipy drops a line only when snr < min_snr, so a NaN snr keeps it.
            !((cwt[r][c] / noise_at(c)).abs() < 1.0)
        })
        .count()
}
