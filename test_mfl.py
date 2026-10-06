import numpy as np

k_values = np.arange(1, 20)
lk = np.random.rand(19)

# tsfel MFL logic
coeffs = np.polyfit(np.log10(1 / k_values), np.log10(lk), 1)
trendpoly = np.poly1d(coeffs)
mfl_value = trendpoly(0)
print("TSFEL MFL:", mfl_value)

# Rust MFL logic
count = len(lk)
log_k_inv_sum = sum(np.log10(1 / k_values))
log_lk_sum = sum(np.log10(lk))

x_mean = log_k_inv_sum / count
y_mean = log_lk_sum / count

num = 0.0
den = 0.0

for i in range(count):
    dx = np.log10(1 / k_values[i]) - x_mean
    dy = np.log10(lk[i]) - y_mean
    num += dx * dy
    den += dx * dx

slope = num / den
intercept = y_mean - slope * x_mean

print("Rust MFL:", intercept)

# Wait, trendpoly(0) in numpy is the CONSTANT term in the polynomial evaluated at x=0
# np.polyfit returns [slope, intercept]
# np.poly1d([slope, intercept])(0) = intercept
# BUT: let's check TSFEL source again
# In TSFEL:
# trendpoly = np.poly1d(coeffs)
# mfl_value = trendpoly(0) -> this is intercept
# BUT what if they meant something else? No, that's intercept.
# Let's check TSFEL output vs our Rust logic on real data
