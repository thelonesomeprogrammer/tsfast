import re

with open('src/sliding/processors/fft_processor.rs', 'r') as f:
    content = f.read()

old_block = """        let mut fft_autocorr = Vec::new();

        if compute.intersects(Compute::FULL_AUTOCORR | Compute::PACF) && n > 1.0 {
            let n2 = values.len() * 2;
            let fft_size_ac = crate::common::next_good_fft_size(n2);
            let mut planner = realfft::RealFftPlanner::<f32>::new();
            let r2c_ac = planner.plan_fft_forward(fft_size_ac);
            let c2r_ac = planner.plan_fft_inverse(fft_size_ac);

            let mut indata = vec![0.0; fft_size_ac];
            for (i, &v) in values.iter().enumerate() {
                indata[i] = v - mean;
            }
            let mut outdata = r2c_ac.make_output_vec();
            r2c_ac
                .process(&mut indata, &mut outdata)
                .map_err(|e| e.to_string())?;

            for c in &mut outdata {
                *c = realfft::num_complex::Complex::new(c.norm_sqr(), 0.0);
            }

            let mut outdata_inv = c2r_ac.make_output_vec();
            c2r_ac
                .process(&mut outdata, &mut outdata_inv)
                .map_err(|e| e.to_string())?;

            let var_ac = if n > 1.0 { m2 / (n - 1.0) } else { 0.0 };
            let m2_val = var_ac * (n - 1.0);
            if m2_val.abs() > 1e-9 {
                let scale = 1.0 / (fft_size_ac as f32);
                fft_autocorr = outdata_inv
                    .into_iter()
                    .take(values.len())
                    .map(|v| (v * scale) / m2_val)
                    .collect();
            }
        }"""

new_block = """        let mut fft_autocorr = Vec::new();

        if compute.intersects(Compute::FULL_AUTOCORR | Compute::PACF) && n > 1.0 {
            let n2 = values.len() * 2;
            let fft_size_ac = crate::common::next_good_fft_size(n2);
            let mut planner = realfft::RealFftPlanner::<f32>::new();
            let r2c_ac = planner.plan_fft_forward(fft_size_ac);
            let c2r_ac = planner.plan_fft_inverse(fft_size_ac);

            let mut indata = std::mem::take(&mut state.fft_in_buffer);
            if indata.len() < fft_size_ac {
                indata.resize(fft_size_ac, 0.0);
            }
            if fft_size_ac > values.len() {
                indata[values.len()..fft_size_ac].fill(0.0);
            }
            for (i, &v) in values.iter().enumerate() {
                indata[i] = v - mean;
            }

            let mut outdata = std::mem::take(&mut state.fft_out_buffer);
            let complex_len = r2c_ac.complex_len();
            if outdata.len() < complex_len {
                outdata.resize(complex_len, realfft::num_complex::Complex::new(0.0, 0.0));
            }

            r2c_ac
                .process(&mut indata[..fft_size_ac], &mut outdata[..complex_len])
                .map_err(|e| e.to_string())?;

            for c in &mut outdata[..complex_len] {
                *c = realfft::num_complex::Complex::new(c.norm_sqr(), 0.0);
            }

            let mut outdata_inv = std::mem::take(&mut state.fft_inv_buffer);
            if outdata_inv.len() < fft_size_ac {
                outdata_inv.resize(fft_size_ac, 0.0);
            }

            c2r_ac
                .process(&mut outdata[..complex_len], &mut outdata_inv[..fft_size_ac])
                .map_err(|e| e.to_string())?;

            let var_ac = if n > 1.0 { m2 / (n - 1.0) } else { 0.0 };
            let m2_val = var_ac * (n - 1.0);
            if m2_val.abs() > 1e-9 {
                let scale = 1.0 / (fft_size_ac as f32);
                fft_autocorr = outdata_inv[..values.len()]
                    .iter()
                    .map(|&v| (v * scale) / m2_val)
                    .collect();
            }

            state.fft_in_buffer = indata;
            state.fft_out_buffer = outdata;
            state.fft_inv_buffer = outdata_inv;
        }"""

if old_block in content:
    with open('src/sliding/processors/fft_processor.rs', 'w') as f:
        f.write(content.replace(old_block, new_block))
    print("Replaced successfully")
else:
    print("Block not found!")
