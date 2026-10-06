import numpy as np
import tsfast

features = ['maximum_fractal_length']
ext = tsfast.Extractor(features)

np.random.seed(42)
x_rw = np.cumsum(np.random.randn(200)).astype(np.float32)

res = ext.process_2d_floats(np.atleast_2d(x_rw))
print('Rust MFL:', res[0][0])
