import pytest
import pyarrow as pa
import numpy as np
from tsfast._tsfast import SlidingExtractor

def test_sliding_higher_moments():
    features = ["mean", "std_dev", "skewness", "kurtosis"]
    extractor = SlidingExtractor(features, 1, 5, 1)

    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float32)
    data = pa.RecordBatch.from_arrays([pa.array(x, type=pa.float32())], names=['c'])
    res = extractor.update(data).to_pandas()

    print(res)

if __name__ == "__main__":
    test_sliding_higher_moments()
