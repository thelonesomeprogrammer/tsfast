import re
with open('src/features/stationarity.rs', 'r') as f:
    content = f.read()

# Add SIMD for OLS evaluation loop inside `if y_len > final_lag + 2 {` and the inner `for lag in (0..=maxlag).rev() {`
# The feedback explicitly asks for f64x8 inside the inner loops.

content = content.replace("use std::f32;", "use std::f32;\nuse std::simd::{f64x8, num::SimdFloat};")

with open('src/features/stationarity.rs', 'w') as f:
    f.write(content)
