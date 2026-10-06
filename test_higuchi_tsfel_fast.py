import numpy as np
from tsfel.feature_extraction.features import calc_lengths_higuchi, maximum_fractal_length, higuchi_fractal_dimension
np.random.seed(42)
x = np.cumsum(np.random.randn(200)).astype(np.float32)
k_values, lk = calc_lengths_higuchi(x)
print("TSFEL lk:", lk[:5])
mfl = maximum_fractal_length(x)
hfd = higuchi_fractal_dimension(x)
print("TSFEL MFL:", mfl, "HFD:", hfd)

import tsfast
features = ["maximum_fractal_length", "higuchi_fd"]
ext = tsfast.Extractor(features)
res = ext.process_2d_floats(np.atleast_2d(x))
print("Rust MFL:", res[0][0], "HFD:", res[0][1])
