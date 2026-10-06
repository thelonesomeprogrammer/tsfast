import sys

def process_file(filepath):
    with open(filepath, "r") as f:
        content = f.read()

    # We replace the batch usage with the numpy array
    content = content.replace("import pyarrow as pa\n", "")
    content = content.replace("batch = pa.RecordBatch.from_arrays([pa.array(x)], names=[\"f0\"])\n", "")
    content = content.replace("ext.update(batch)", "ext.update(np.stack([x]))")

    with open(filepath, "w") as f:
        f.write(content)

process_file("tests/test_sliding.py")
process_file("tests/test_expanding.py")
print("Replaced pyarrow with numpy arrays in tests.")
