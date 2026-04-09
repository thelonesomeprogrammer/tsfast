use criterion::{black_box, criterion_group, criterion_main, Criterion};
use tsfast::features::statistics::mean;

fn criterion_benchmark(c: &mut Criterion) {
    let data = vec![1.0; 1000];
    c.bench_function("mean", |b| b.iter(|| mean(black_box(&data))));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
