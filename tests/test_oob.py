import pytest
import numpy as np
import tsfast


def test_paa_oob():
    x = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    # total=2, index=2 -> boundaries has 3 elements [0, 1, 2], so b[index+1] -> b[3] is OOB!
    features = ["paa-2-2"]
    with pytest.raises(ValueError, match="Unknown feature"):
        extractor = tsfast.Extractor(features)

@pytest.mark.parametrize("invalid_feature", [
    # ECDF Slope invalid parameters
    "ecdf_slope-0.0-1.0", # p_init must be > 0
    "ecdf_slope-0.5-0.4", # p_init must be < p_end

    # ECDF Percentile Count invalid parameters
    "ecdf_percentile_count-0.0", # p must be > 0
    "ecdf_percentile_count-1.1", # p must be <= 1.0

    # Energy Ratio by Chunks invalid parameters
    "energy_ratio_by_chunks_num_segments_3__segment_focus_4", # focus < num_segments
    "energy_ratio_by_chunks_num_segments_0__segment_focus_0", # num_segments > 0

    # Agg Linear Trend invalid parameters
    "agg_linear_trend-slope-0-mean", # chunk_len > 0

    # AR Coefficient invalid parameters
    "ar_coefficient-0-0", # k > 0
    "ar_coefficient-2-3", # p <= k

    # Binned Entropy invalid parameters
    "binned_entropy__max_bins_0", # bins > 0

    # Peak finding features invalid parameters
    "number_cwt_peaks__n_0", # n > 0
    "number_peaks__n_0", # n > 0
])
def test_invalid_ffi_boundaries(invalid_feature):
    # Verify that these malformed string arguments fail gracefully at the
    # Rust parsing layer and return ValueError to Python instead of panicking
    with pytest.raises(ValueError, match="Unknown feature"):
        extractor = tsfast.Extractor([invalid_feature])
