import numpy as np
from tsfel.feature_extraction.features import maximum_fractal_length, calc_lengths_higuchi

np.random.seed(42)
x_sine = np.sin(np.linspace(0, 10 * np.pi, 200)).astype(np.float32)

k_values, lk = calc_lengths_higuchi(x_sine)
print("TSFEL lk:", lk[:5])
mfl = maximum_fractal_length(x_sine)
print("TSFEL MFL:", mfl)

import tsfast
features = ["maximum_fractal_length"]
ext = tsfast.Extractor(features)
res = ext.process_2d_floats(np.atleast_2d(x_sine))
print("Rust MFL:", res[0][0])
