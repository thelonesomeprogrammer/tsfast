import tsfast
import numpy as np

def test_subsequence_features():
    import stumpy
    from tsfresh.feature_extraction.feature_calculators import query_similarity_count

    np.random.seed(42)
    x = np.random.randn(100).astype(np.float64)
    q = x[:10]

    # stumpy
    mp = stumpy.stump(x, m=10)
    profile = mp[:, 0]

    mp_min = float(np.min(profile))
    mp_max = float(np.max(profile))
    mp_mean = float(np.mean(profile))

    # tsfresh
    param = [{'query': q, 'threshold': 1.0}]
    qs = query_similarity_count(x, param)
    q_count = qs[0][1]

    features = [
        "matrix_profile-10-min",
        "matrix_profile-10-max",
        "matrix_profile-10-mean",
        "query_similarity_count-10-1.0"
    ]

    extractor = tsfast.Extractor(features)
    batch = np.stack([x.astype(np.float32)])
    out = extractor.process_2d_floats(batch)
    results = out[0]

    assert np.allclose(results[0], mp_min, atol=1e-4)
    assert np.allclose(results[1], mp_max, atol=1e-4)
    assert np.allclose(results[2], mp_mean, atol=1e-4)
    assert results[3] == q_count
