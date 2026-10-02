import pyarrow as pa
import numpy as np
from tsfast._tsfast import ExpandingExtractor

features = ["mean", "std_dev", "skewness", "kurtosis"]
extractor = ExpandingExtractor(features, 1)

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float32)
data = pa.RecordBatch.from_arrays([pa.array(x, type=pa.float32())], names=['c'])
res = extractor.update(data).to_pandas().iloc[0]

print("Output:")
print(res)
