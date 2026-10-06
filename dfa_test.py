import numpy as np

def dfa_ref(signal):
    n = len(signal)
    accumulated_signal = np.cumsum(signal - np.mean(signal))
    # Python sets of ints are ordered by value for small positive integers!
    windows = sorted(list(set(np.linspace(4, n // 10, n // 2, dtype=int))))
    fluct = np.zeros(len(windows))

    for idx, window in enumerate(windows):
        num_windows = len(signal) // window
        rms = np.zeros(num_windows)
        for idx2 in np.arange(num_windows):
            start_idx = idx2 * window
            end_idx = start_idx + window
            windowed_signal = accumulated_signal[start_idx:end_idx]

            coeff = np.polyfit(np.arange(window), windowed_signal, 1)
            detrended_window = windowed_signal - np.polyval(coeff, np.arange(window))
            rms[idx2] = np.sqrt(np.mean(detrended_window**2))
        fluct[idx] = np.sqrt(np.mean(rms ** 2))

    def find_plateau(y, threshold=0.1, consecutive_points=5):
        dy = np.diff(y)
        for i in np.arange(len(dy) - consecutive_points + 1):
            if np.all(np.abs(dy[i : i + consecutive_points]) < threshold):
                plateau_value = np.mean(y[i : i + consecutive_points])
                if plateau_value > np.mean(y):
                    return i
        return len(y)

    i_plateau = find_plateau(np.log(fluct))
    fluct = fluct[0:i_plateau]
    windows = list(windows)[0:i_plateau]

    if len(windows) > 1:
        coeffs = np.polyfit(np.log(windows), np.log(fluct), 1)
        return coeffs[0]
    return np.nan

def hurst_ref(signal):
    n = len(signal)
    lags = sorted(list(set(np.linspace(4, n // 10, n // 2, dtype=int))))
    rs_vals = []

    for lag in lags:
        windowed_signal = np.reshape(signal[: n - n % lag], (-1, lag))
        mean_windows = np.mean(windowed_signal, axis=1)
        accumulated_windowed_signal = np.cumsum(
            windowed_signal - np.reshape(mean_windows, (-1, 1)),
            axis=1,
        )
        r = np.max(accumulated_windowed_signal, axis=1) - np.min(
            accumulated_windowed_signal,
            axis=1,
        )
        s = np.std(windowed_signal, axis=1)
        # to match tsfel exactly: rs = np.divide(r, s)
        with np.errstate(divide='ignore', invalid='ignore'):
            rs = np.divide(r, s)
        rs_vals.append(np.mean(rs))

    n_values = np.array(lags)[np.isfinite(rs_vals)]
    rs_vals = np.array(rs_vals)[np.isfinite(rs_vals)]

    if len(n_values) > 1:
        coeffs = np.polyfit(np.log10(n_values), np.log10(rs_vals), 1)
        return coeffs[0]
    return np.nan

print("Testing")
signal = np.random.randn(200)
print(dfa_ref(signal))
print(hurst_ref(signal))
