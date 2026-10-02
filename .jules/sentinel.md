## 2024-05-24 - Fixed Hacky Frequency Storage in Features
**Learning:** Storing `f32` representations in `u16` fields using `as u16` bitcasting results in critical data loss (truncating exponent and mantissa). It should use `u32` for `f32::to_bits()` to maintain accuracy.
**Action:** Changed the storage type for frequencies from `u16` to `u32` within the `Feature` variants `SpectrogramCoefficients` and `WaveletFeatures`. Validated sizes, ensured matching string formatting behavior with `.from_bits()`, and fixed test assertions in static feature processing.
