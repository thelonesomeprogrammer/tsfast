## 2024-05-24 - Fixed Hacky Frequency Storage in Features
**Learning:** Storing `f32` representations in `u16` fields using `as u16` bitcasting results in critical data loss (truncating exponent and mantissa). It should use `u32` for `f32::to_bits()` to maintain accuracy.
**Action:** Changed the storage type for frequencies from `u16` to `u32` within the `Feature` variants `SpectrogramCoefficients` and `WaveletFeatures`. Validated sizes, ensured matching string formatting behavior with `.from_bits()`, and fixed test assertions in static feature processing.

## 2023-10-02 - Variance edge case testing
**Learning:** Adding variance-dependent features requires ensuring edge cases are covered properly in sliding/expanding calculations. Variance might equal 0.0, or arrays might have insufficient length to calculate ddof=1 variance, leading to potential assertion failures if expected bounds are not set or float edge conditions are not handled.
**Action:** In `test_tsfast.py`, added a specific assertion to correctly verify that the condition var > 1.0 translates cleanly across the Rust FFI boundaries without panic or precision bugs.
