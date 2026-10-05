import sys

def append_to_file(path, text):
    with open(path, "a") as f:
        f.write(text)

with open("tests/test_tsfast.py", "r") as f:
    tsfast_content = f.read()

if "energy_ratio_by_chunks" not in tsfast_content:
    with open("tests/test_tsfast.py", "a") as f:
        f.write("""
def test_new_features():
    from tsfresh.feature_extraction.feature_calculators import energy_ratio_by_chunks
    import tsfast
    import numpy as np
    import pyarrow as pa

    x = np.array([3.0, 1.0, 4.0, 1.5, 9.0, 2.0, 6.0, 5.0, 3.5, 8.0, 9.0], dtype=np.float32)
    features = [
        "energy_ratio_by_chunks-10-5",
        "energy_ratio_by_chunks-10-9",
    ]
    extractor = tsfast.Extractor(features)
    batch = pa.RecordBatch.from_arrays([pa.array(x)], names=['c1'])
    result_batch = extractor.process_2d_floats(batch)
    results = result_batch.to_pandas().iloc[0].values

    expected = [
        energy_ratio_by_chunks(x, [{"num_segments": 10, "segment_focus": 5}])[0][1],
        energy_ratio_by_chunks(x, [{"num_segments": 10, "segment_focus": 9}])[0][1],
    ]

    for i in range(2):
        assert np.isclose(results[i], expected[i], rtol=1e-5, atol=1e-5), f"Feature {i} differs"
""")

with open("tests/test_sliding.py", "r") as f:
    sliding_content = f.read()

if "test_sliding_energy_ratio_by_chunks" not in sliding_content:
    with open("tests/test_sliding.py", "a") as f:
        f.write("""
def test_sliding_energy_ratio_by_chunks():
    from tsfresh.feature_extraction.feature_calculators import energy_ratio_by_chunks
    import tsfast
    import pyarrow as pa
    import numpy as np

    x = np.array([1.0, -2.0, 3.0, 4.0, 5.0, 1.0, 0.0, 5.0, -1.0, 3.0], dtype=np.float32)
    features = ["energy_ratio_by_chunks-10-5"]
    window_size = 7
    extractor = tsfast.SlidingExtractor(features, n_cols=1, window_size=window_size)

    batch = pa.RecordBatch.from_arrays([pa.array(x)], names=['c1'])
    df = extractor.update(batch).to_pandas()
    results = df.iloc[-1].values

    windowed_x = x[-window_size:]
    expected = energy_ratio_by_chunks(windowed_x, [{"num_segments": 10, "segment_focus": 5}])[0][1]

    assert np.isclose(results[0], expected, rtol=1e-5, atol=1e-5)
""")

with open("tests/test_expanding.py", "r") as f:
    expanding_content = f.read()

if "test_expanding_energy_ratio_by_chunks" not in expanding_content:
    with open("tests/test_expanding.py", "a") as f:
        f.write("""
def test_expanding_energy_ratio_by_chunks():
    from tsfresh.feature_extraction.feature_calculators import energy_ratio_by_chunks
    import tsfast
    import pyarrow as pa
    import numpy as np

    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 0.0, 5.0, -1.0, 3.0], dtype=np.float32)
    features = ["energy_ratio_by_chunks-5-2"]
    extractor = tsfast.ExpandingExtractor(features, n_cols=1)

    batch = pa.RecordBatch.from_arrays([pa.array(x[:5])], names=['c1'])
    df1 = extractor.update(batch).to_pandas()

    batch2 = pa.RecordBatch.from_arrays([pa.array(x[5:])], names=['c1'])
    df2 = extractor.update(batch2).to_pandas()

    results = df2.iloc[-1].values
    expected = energy_ratio_by_chunks(x, [{"num_segments": 5, "segment_focus": 2}])[0][1]

    assert np.isclose(results[0], expected, rtol=1e-5, atol=1e-5)
""")
