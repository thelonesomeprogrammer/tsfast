use crate::types::Feature;

#[inline(always)]
pub fn eval_energy(feat: &Feature, context: &mut crate::context::FeatureContext) -> Option<f32> {
    let state = &mut *context.state;
    let n = context.n;
    let spectrum = context.spectrum;

    let res = match feat {
        Feature::Energy => state.energy as f32,
        Feature::Rms | Feature::RootMeanSquare => (state.energy / n).sqrt(),
        Feature::HumanRangeEnergy(fs_bits) => {
            if !spectrum.is_empty() {
                let fs = f32::from_bits(*fs_bits);
                let n_fft = (spectrum.len() - 1) * 2;
                let freq_step = fs / n_fft as f32;
                let start_idx = (0.6 / freq_step).ceil() as usize;
                let end_idx = (2.5 / freq_step).floor() as usize;

                let total_energy: f32 = spectrum.iter().map(|&s| s * s).sum();
                if total_energy > 0.0 {
                    let range_energy: f32 = spectrum
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| *i >= start_idx && *i <= end_idx)
                        .map(|(_, &s)| s * s)
                        .sum();
                    range_energy / total_energy
                } else {
                    0.0
                }
            } else {
                0.0
            }
        }
        _ => return None,
    };
    Some(res)
}
