import pyarrow as pa
from tsfast._tsfast import SlidingExtractor

try:
    extractor = SlidingExtractor(["invalid"], 1, 3, 1)
    print("Success!")
except Exception as e:
    print(f"Error: {e}")
