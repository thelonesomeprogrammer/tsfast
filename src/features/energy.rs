use std::collections::HashMap;

pub fn energy(data: &[f64], precalc: &mut HashMap<&str, f64>) -> f64 {
    if let Some(&e) = precalc.get("energy") {
        return e;
    }
    let energy = data.iter().map(|&x| x * x).sum();
    precalc.insert("energy", energy);
    energy
}

pub fn rms(data: &[f64], precalc: &mut HashMap<&str, f64>) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let energy = if let Some(&e) = precalc.get("energy") {
        e
    } else {
        energy(data, precalc)
    };
    let rms = (energy / data.len() as f64).sqrt();
    precalc.insert("rms", rms);
    rms
}
