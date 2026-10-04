import re
with open('src/features/stationarity.rs', 'r') as f:
    content = f.read()

# Fix unused variables that I accidentally removed instead of declaring properly
content = content.replace("                                let best_t_stat = (coeffs[0] as f64 / var_coeff0.sqrt()) as f32;", "                                best_t_stat = (coeffs[0] as f64 / var_coeff0.sqrt()) as f32;")
content = content.replace("                let final_lag = best_lag;", "                let final_lag = best_lag;\n                let mut best_t_stat = f32::NAN;")

with open('src/features/stationarity.rs', 'w') as f:
    f.write(content)
