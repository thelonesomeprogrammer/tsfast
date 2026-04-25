import tsfast
import numpy as np
import pyarrow as pa

def test_new_features():
    # List of some new features to test
    features = [
        "abs_max",
        "first_loc_max",
        "last_loc_max",
        "autocorr-1",
        "autocorr-2",
        "time_reversal_asymmetry-1",
        "fft_coeff-1-real",
        "fft_coeff-1-imag",
        "agg_linear_trend-slope-5-mean",
        "agg_linear_trend-intercept-5-mean"
    ]
    
    print(f"Testing features: {features}")
    
    # Static Extractor test with more interesting data for autocorr
    # data: linear trend + some periodicity
    data = np.array([1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0], dtype=np.float32)
    batch = pa.RecordBatch.from_arrays([pa.array(data)], names=["v"])
    
    extractor = tsfast.Extractor(features)
    result = extractor.process_2d_floats(batch)
    print("\nStatic Extraction Result (Periodic Data [1,2,1,2...]):")
    for feat in features:
        print(f"{feat}: {result.column(feat)[0].as_py()}")

    # Expanding Extractor test
    exp_extractor = tsfast.ExpandingExtractor(features, 1)
    
    print("\nExpanding Extraction Results:")
    # Update with first half
    batch1 = pa.RecordBatch.from_arrays([pa.array(data[:5])], names=["v"])
    res1 = exp_extractor.update(batch1)
    print(f"Batch 1 (first 5 elements, length {len(res1)}):")
    for feat in features:
         # Use the last row of the result
         print(f"{feat}: {res1.column(feat)[len(res1)-1].as_py()}")
    
    # Update with second half
    batch2 = pa.RecordBatch.from_arrays([pa.array(data[5:])], names=["v"])
    res2 = exp_extractor.update(batch2)
    print(f"\nBatch 2 (next 5 elements, total 10, length {len(res2)}):")
    for feat in features:
         print(f"{feat}: {res2.column(feat)[len(res2)-1].as_py()}")

if __name__ == "__main__":
    test_new_features()
