from scipy.stats import skew, kurtosis
import numpy as np

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float32)

k_bias = kurtosis(x, fisher=True, bias=True)
k_nobias = kurtosis(x, fisher=True, bias=False)
k_pearson = kurtosis(x, fisher=False, bias=True)

print(f"k_bias: {k_bias}, k_nobias: {k_nobias}, k_pearson: {k_pearson}")
