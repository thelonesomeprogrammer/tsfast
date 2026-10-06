import numpy as np

s = 10
y_all = np.random.randn(100)

y_pref = np.zeros(101, dtype=np.float64)
y2_pref = np.zeros(101, dtype=np.float64)
iy_pref = np.zeros(101, dtype=np.float64)

for i in range(100):
    y_pref[i+1] = y_pref[i] + y_all[i]
    y2_pref[i+1] = y2_pref[i] + y_all[i]**2
    iy_pref[i+1] = iy_pref[i] + i * y_all[i]

start = 20
y_seg = y_all[start:start+s]

# Standard OLS
coeff = np.polyfit(np.arange(s), y_seg, 1)
detrended = y_seg - np.polyval(coeff, np.arange(s))
rss_std = np.sum(detrended**2)

# Fast OLS
sum_y = y_pref[start+s] - y_pref[start]
sum_y2 = y2_pref[start+s] - y2_pref[start]
sum_iy_global = iy_pref[start+s] - iy_pref[start]
sum_jy = sum_iy_global - start * sum_y

x_bar = (s - 1) / 2
y_bar = sum_y / s
ss_xx = s * (s**2 - 1) / 12
ss_xy = sum_jy - s * x_bar * y_bar
beta1 = ss_xy / ss_xx
beta0 = y_bar - beta1 * x_bar

rss_fast = sum_y2 - beta0 * sum_y - beta1 * sum_jy

print(f"Standard RSS: {rss_std}")
print(f"Fast RSS: {rss_fast}")
