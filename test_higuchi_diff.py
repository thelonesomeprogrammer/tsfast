import tsfast
import numpy as np
from tsfel.feature_extraction.features import calc_lengths_higuchi

np.random.seed(42)
x_sine = np.sin(np.linspace(0, 10 * np.pi, 200)).astype(np.float32)

k_values, lk = calc_lengths_higuchi(x_sine)
print("TSFEL lk:", lk[:5])
