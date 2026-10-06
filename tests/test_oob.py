import pytest
import numpy as np
import tsfast


def test_paa_oob():
    x = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    # total=2, index=2 -> boundaries has 3 elements [0, 1, 2], so b[index+1] -> b[3] is OOB!
    features = ["paa-2-2"]
    with pytest.raises(ValueError, match="Unknown feature"):
        extractor = tsfast.Extractor(features)
