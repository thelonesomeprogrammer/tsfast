import tsfel
import numpy as np

data = np.array([1, 2, 3, 2, 1, 0, -1, 0, 1, 2, 1, 2, 3])
print("positive_turning:", tsfel.feature_extraction.features.positive_turning(data))
print("negative_turning:", tsfel.feature_extraction.features.negative_turning(data))

# tsfast PeakCount logic for this data:
peaks = 0
for i in range(1, len(data)-1):
    if data[i] > data[i-1] and data[i] > data[i+1]:
        peaks += 1
print("PeakCount:", peaks)
