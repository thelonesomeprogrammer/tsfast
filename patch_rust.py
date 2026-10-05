import sys

def replace_in_file(path, target, replacement):
    with open(path, "r") as f:
        content = f.read()
    if target in content:
        content = content.replace(target, replacement)
        with open(path, "w") as f:
            f.write(content)
        print(f"Replaced in {path}")
    else:
        print(f"Target not found in {path}")

# Patch src/types/feature.rs
feature_target = """    ChangeQuantiles(f32, f32, bool, bool),"""
feature_replacement = """    ChangeQuantiles(f32, f32, bool, bool),
    EnergyRatioByChunks(u16, u16),"""
replace_in_file("src/types/feature.rs", feature_target, feature_replacement)

# Patch compute requirements in src/types/feature.rs
compute_target = """            Feature::CountBelowMean | Feature::CountAboveMean => C::MEAN,
            Feature::CountBelow(_) | Feature::CountAbove(_) => C::NONE,"""
compute_replacement = """            Feature::CountBelowMean | Feature::CountAboveMean => C::MEAN,
            Feature::CountBelow(_) | Feature::CountAbove(_) => C::NONE,
            Feature::EnergyRatioByChunks(_, _) => C::ENERGY,"""
replace_in_file("src/types/feature.rs", compute_target, compute_replacement)

# Patch string formatting in src/types/feature.rs
format_target = """            Feature::ChangeQuantiles(ql, qh, isabs, f_agg) => {
                let f_agg_str = if *f_agg { "mean" } else { "var" };
                write!(f, "change_quantiles-{}-{}-{}-{}", ql, qh, isabs, f_agg_str)
            }"""
format_replacement = """            Feature::ChangeQuantiles(ql, qh, isabs, f_agg) => {
                let f_agg_str = if *f_agg { "mean" } else { "var" };
                write!(f, "change_quantiles-{}-{}-{}-{}", ql, qh, isabs, f_agg_str)
            }
            Feature::EnergyRatioByChunks(num_segments, segment_focus) => write!(f, "energy_ratio_by_chunks-{}-{}", num_segments, segment_focus),"""
replace_in_file("src/types/feature.rs", format_target, format_replacement)

# Patch parse logic in src/types/parse.rs
parse_target = """        } else if let Some(suffix) = s.strip_prefix("change_quantiles-") {
            let parts: Vec<&str> = suffix.split('-').collect();
            if parts.len() == 4 {
                if let (Ok(ql), Ok(qh), Ok(isabs)) = (parts[0].parse(), parts[1].parse(), parts[2].parse()) {
                    let f_agg = parts[3] == "mean";
                    Some(Feature::ChangeQuantiles(ql, qh, isabs, f_agg))
                } else { None }
            } else { None }"""
parse_replacement = """        } else if let Some(suffix) = s.strip_prefix("change_quantiles-") {
            let parts: Vec<&str> = suffix.split('-').collect();
            if parts.len() == 4 {
                if let (Ok(ql), Ok(qh), Ok(isabs)) = (parts[0].parse(), parts[1].parse(), parts[2].parse()) {
                    let f_agg = parts[3] == "mean";
                    Some(Feature::ChangeQuantiles(ql, qh, isabs, f_agg))
                } else { None }
            } else { None }
        } else if let Some(suffix) = s.strip_prefix("energy_ratio_by_chunks-") {
            let parts: Vec<&str> = suffix.split('-').collect();
            if parts.len() == 2 {
                if let (Ok(num_segments), Ok(segment_focus)) = (parts[0].parse(), parts[1].parse()) {
                    Some(Feature::EnergyRatioByChunks(num_segments, segment_focus))
                } else { None }
            } else { None }"""
replace_in_file("src/types/parse.rs", parse_target, parse_replacement)

# Patch src/features/energy.rs implementation
energy_target = """        Feature::Energy => state.energy,
        _ => unreachable!(),
    }
}"""
energy_replacement = """        Feature::Energy => state.energy,
        Feature::EnergyRatioByChunks(num_segments, segment_focus) => {
            let total = _values.iter().map(|&v| v * v).sum::<f32>();
            if total == 0.0 {
                return 0.0;
            }
            let n = _values.len();
            let num_segments = *num_segments as usize;
            let segment_focus = *segment_focus as usize;

            if num_segments == 0 || segment_focus >= num_segments {
                return 0.0;
            }

            let q = n / num_segments;
            let r = n % num_segments;

            let start_idx = if segment_focus < r {
                segment_focus * (q + 1)
            } else {
                r * (q + 1) + (segment_focus - r) * q
            };

            let end_idx = start_idx + q + if segment_focus < r { 1 } else { 0 };

            if start_idx >= n {
                return 0.0;
            }
            let end_idx = end_idx.min(n);

            let mut sum_sq = 0.0;
            let chunk = &_values[start_idx..end_idx];

            // SIMD optimization for chunk energy
            let mut i = 0;
            let mut sum_v = std::simd::f32x4::splat(0.0);
            while i + 4 <= chunk.len() {
                let v = std::simd::f32x4::from_slice(&chunk[i..i+4]);
                sum_v = sum_v + v * v;
                i += 4;
            }
            sum_sq += sum_v.reduce_sum();
            while i < chunk.len() {
                sum_sq += chunk[i] * chunk[i];
                i += 1;
            }

            sum_sq / total
        }
        _ => unreachable!(),
    }
}"""
replace_in_file("src/features/energy.rs", energy_target, energy_replacement)
