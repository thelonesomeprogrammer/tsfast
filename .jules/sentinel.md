## 2024-05-24 - Fixed Hacky Frequency Storage in Features
**Learning:** Storing `f32` representations in `u16` fields using `as u16` bitcasting results in critical data loss (truncating exponent and mantissa). It should use `u32` for `f32::to_bits()` to maintain accuracy.
**Action:** Changed the storage type for frequencies from `u16` to `u32` within the `Feature` variants `SpectrogramCoefficients` and `WaveletFeatures`. Validated sizes, ensured matching string formatting behavior with `.from_bits()`, and fixed test assertions in static feature processing.

## 2023-10-02 - Variance edge case testing
**Learning:** Adding variance-dependent features requires ensuring edge cases are covered properly in sliding/expanding calculations. Variance might equal 0.0, or arrays might have insufficient length to calculate ddof=1 variance, leading to potential assertion failures if expected bounds are not set or float edge conditions are not handled.
**Action:** In `test_tsfast.py`, added a specific assertion to correctly verify that the condition var > 1.0 translates cleanly across the Rust FFI boundaries without panic or precision bugs.

## 2023-10-02 - Python FFI Panic on PyArrow Downcasts
**Vulnerability:** Calling `.expect()` or `.unwrap()` when downcasting user-provided data types across the PyO3 FFI boundary (like PyArrow Float64 to Float32) causes a hard panic. This terminates the host Python interpreter ungracefully, leading to a critical Denial of Service vulnerability on servers accepting user data.
**Learning:** Rust panics cannot be cleanly caught by Python unless explicitly converted into standard `PyResult` types mapped to Python Exceptions.
**Prevention:** Never use `.unwrap()` or `.expect()` on dynamically-typed external FFI inputs like DataFrames/RecordBatches. Always map errors (`.ok_or_else()`) into `PyTypeError` or `PyValueError` to return a `Result` type up the call stack to Python.

## 2024-05-25 - Denial of Service via Out Of Bounds Read in PAA Feature
**Vulnerability:** The `Feature::Paa(total, index)` parser failed to validate that `index < total`. Passing parameters like `paa-2-2` caused an out of bounds read on the `paa_boundaries` cache (which is sized `total + 1`). Because array accesses (`b[*index as usize + 1]`) lacked bounds checking and safety limits, this caused a Rust panic across the PyO3 boundary, crashing the host Python interpreter ungracefully (DoS).
**Learning:** Always validate indices supplied in string-parsed arguments against their expected maximum sizes immediately at the parsing layer, preventing invalid structures from entering the processing engine.
**Prevention:** In `src/types/parse.rs`, add bounds validation (`if index >= total { return None; }`) during parsing so that invalid inputs correctly map to a Python `ValueError("Unknown feature")` rather than panicking later.

## 2024-05-26 - Denial of Service via Zero Chunk Length in AggLinearTrend
**Vulnerability:** The `Feature::AggLinearTrend(attr, chunk_len, func)` parser failed to validate that `chunk_len > 0`. Parsing a feature string like `agg_linear_trend-slope-0-mean` passes `chunk_len = 0` to the processing engine. This triggers a panic in the `values.chunks_exact(0)` method inside `src/features/changes.rs`, causing the host Python interpreter to crash ungracefully (DoS).
**Learning:** Always validate sizes and chunk lengths supplied in string-parsed arguments immediately at the parsing layer, preventing invalid zero-length chunking requests from crashing the Rust engine via panic.
**Prevention:** In `src/types/parse.rs`, add bounds validation (`if chunk_len == 0 { return None; }`) during parsing so that inputs incorrectly requesting 0-length chunks correctly map to a Python `ValueError("Unknown feature")` rather than panicking.
