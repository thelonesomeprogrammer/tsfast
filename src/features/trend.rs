use crate::registry::Cache;

fn linear_regression(data: &[f64]) -> (f64, f64) {
    let n = data.len();
    let n_f = n as f64;
    let sum_x = n_f * (n_f - 1.0) / 2.0;
    let sum_x2 = n_f * (n_f - 1.0) * (2.0 * n_f - 1.0) / 6.0;
    let mut sum_y = 0.0;
    let mut sum_xy = 0.0;
    for (i, &y) in data.iter().enumerate() {
        let x = i as f64;
        sum_y += y;
        sum_xy += x * y;
    }
    let denominator = n_f * sum_x2 - sum_x * sum_x;
    let slope = (n_f * sum_xy - sum_x * sum_y) / denominator;
    let intercept = (sum_y - slope * sum_x) / n_f;
    (slope, intercept)
}

pub fn slope(data: &[f64], cache: &mut Cache) {
    let (slope, intercept) = linear_regression(data);
    cache.slope = Some(slope);
    cache.intercept = Some(intercept);
}

pub fn intercept(data: &[f64], cache: &mut Cache) {
    let (slope, intercept) = linear_regression(data);
    cache.slope = Some(slope);
    cache.intercept = Some(intercept);
}
