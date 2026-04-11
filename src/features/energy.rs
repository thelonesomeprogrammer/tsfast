use crate::registry::Cache;

pub fn energy(data: &[f64], cache: &mut Cache) {
    let energy = data.iter().map(|&x| x * x).sum();
    cache.energy = Some(energy);
}

pub fn rms(data: &[f64], cache: &mut Cache) {
    energy(data, cache);
    let energy = cache.energy.unwrap();
    let rms = (energy / data.len() as f64).sqrt();
    cache.rms = Some(rms);
}
