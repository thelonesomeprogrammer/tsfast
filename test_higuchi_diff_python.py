import numpy as np
from tsfel.feature_extraction.features import calc_lengths_higuchi

np.random.seed(42)
x_sine = np.sin(np.linspace(0, 10 * np.pi, 200)).astype(np.float32)

k_values, lk = calc_lengths_higuchi(x_sine)

def rust_logic(values):
    n = len(values)
    k_max = n // 10 - 1

    higuchi_lk = []

    for k in range(1, k_max + 1):
        lmk_sum = 0.0
        m_sums = [0.0] * k

        for j in range(k, n):
            m = (j % k) + 1
            m_sums[m - 1] += abs(values[j] - values[j - k])

        for m in range(1, k + 1):
            iters = (n - m) // k
            norm_factor = (n - 1) / (iters * k)
            lmk = (m_sums[m - 1] * norm_factor) / k
            lmk_sum += lmk

        lk_val = lmk_sum / k
        higuchi_lk.append(lk_val)

    return higuchi_lk

rust_lk = rust_logic(x_sine)

print("TSFEL lk:", lk[:5])
print("Rust lk :", rust_lk[:5])
