use arrow::array::{Float32Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::pyarrow::PyArrowType;
use criterion::{BatchSize, BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use std::sync::Arc;
use std::time::Duration;
use tsfast::{ExpandingExtractor, Extractor, SlidingExtractor};

const FEATURES: &[&str] = &[
    "total_sum",
    "mean",
    "variance",
    "std_dev",
    "min_value",
    "max_value",
    "median",
    "skewness",
    "kurtosis",
    "biased_fisher_kurtosis",
    "mad",
    "iqr",
    "entropy",
    "energy",
    "rms",
    "root_mean_square",
    "zero_crossing_rate",
    "peak_count",
    "autocorr_lag1",
    "autocorrelation",
    "mean_abs_change",
    "mean_change",
    "median_diff",
    "median_abs_diff",
    "cid_ce",
    "slope",
    "intercept",
    "abs_sum_change",
    "count_above_mean",
    "count_below_mean",
    "longest_strike_above_mean",
    "longest_strike_below_mean",
    "variation_coefficient",
    "auc",
    "zero_crossing_mean",
    "zero_crossing_std",
    "abs_max",
    "first_loc_max",
    "last_loc_max",
    "first_loc_min",
    "last_loc_min",
    "benford_correlation",
    "spectral_centroid",
    "spectral_distance",
    "spectral_decrease",
    "spectral_slope",
    "signal_distance",
    "c3-1",
    "paa-2-0",
    "autocorr-1",
    "partial_autocorr-1",
    "time_reversal_asymmetry-1",
    "fft_coeff-1-real",
    "fft_coeff-1-imag",
    "fft_coeff-1-abs",
    "fft_coeff-1-angle",
    "approx_entropy-2-0.7",
    "agg_linear_trend-slope-50-mean",
    "quantile-0.5",
    "index_mass_quantile-0.5",
    "mean_n_absolute_max-5",
];

fn create_batch(data: &[f32]) -> RecordBatch {
    let schema = Schema::new(vec![Field::new("c1", DataType::Float32, false)]);
    let array = Float32Array::from(data.to_vec());
    RecordBatch::try_new(Arc::new(schema), vec![Arc::new(array)])
        .expect("Failed to build RecordBatch")
}

fn generate_synthetic_signal(n: usize) -> Vec<f32> {
    let mut data = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f32 * 0.01;
        let val = 0.5 * (t * 0.1).sin() + 0.3 * (t * 1.5).cos() + ((i % 7) as f32 - 3.0) * 0.1;
        data.push(val);
    }
    data
}

fn bench_static_features(c: &mut Criterion) {
    let data = generate_synthetic_signal(1000);
    let batch = create_batch(&data);

    let mut group = c.benchmark_group("static_features");
    group.warm_up_time(Duration::from_millis(100));
    group.measurement_time(Duration::from_millis(250));
    group.sample_size(20);

    for &feat in FEATURES {
        if let Ok(extractor) = Extractor::new(vec![feat.to_string()], None) {
            group.bench_with_input(BenchmarkId::new("static", feat), &feat, |b, _| {
                b.iter(|| {
                    let _ = extractor.process_2d_floats(PyArrowType(black_box(batch.clone())));
                });
            });
        }
    }
    group.finish();
}

fn bench_expanding_features(c: &mut Criterion) {
    let prime_data = generate_synthetic_signal(500);
    let update_data = generate_synthetic_signal(100);
    let prime_batch = create_batch(&prime_data);
    let update_batch = create_batch(&update_data);

    let mut group = c.benchmark_group("expanding_features");
    group.warm_up_time(Duration::from_millis(100));
    group.measurement_time(Duration::from_millis(250));
    group.sample_size(20);

    for &feat in FEATURES {
        if let Ok(mut ext) = ExpandingExtractor::new(vec![feat.to_string()], 1, None, 1) {
            let _ = ext.update(PyArrowType(prime_batch.clone()));
            group.bench_with_input(BenchmarkId::new("expanding", feat), &feat, |b, _| {
                b.iter_batched(
                    || ext.clone(),
                    |mut e| {
                        let _ = e.update(PyArrowType(black_box(update_batch.clone())));
                    },
                    BatchSize::SmallInput,
                );
            });
        }
    }
    group.finish();
}

fn bench_sliding_features(c: &mut Criterion) {
    let prime_data = generate_synthetic_signal(200);
    let update_data = generate_synthetic_signal(50);
    let prime_batch = create_batch(&prime_data);
    let update_batch = create_batch(&update_data);

    let mut group = c.benchmark_group("sliding_features");
    group.warm_up_time(Duration::from_millis(100));
    group.measurement_time(Duration::from_millis(250));
    group.sample_size(20);

    for &feat in FEATURES {
        if let Ok(mut ext) = SlidingExtractor::new(vec![feat.to_string()], 1, 200, 50) {
            let _ = ext.update(PyArrowType(prime_batch.clone()));
            group.bench_with_input(BenchmarkId::new("sliding", feat), &feat, |b, _| {
                b.iter_batched(
                    || ext.clone(),
                    |mut e| {
                        let _ = e.update(PyArrowType(black_box(update_batch.clone())));
                    },
                    BatchSize::SmallInput,
                );
            });
        }
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_static_features,
    bench_expanding_features,
    bench_sliding_features
);
criterion_main!(benches);
