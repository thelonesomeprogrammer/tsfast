## 2026-10-02 - Reusing allocations in ColumnState
**Learning:** Repetitive inner-loop allocations (like `Vec::new()` or `Vec::with_capacity()` inside feature calculation blocks like `AggLinearTrend` and `PartialAutocorr`) can be a silent performance drain in this Rust codebase.
**Action:** Instead of allocating new vectors, add them to the persistent `ColumnState` struct, extract them safely during calculation using `std::mem::take()`, `.clear()` them for use, and store them back when done.

## 2024-05-15 - [Performance Insight: FFT memory allocation]
**Learning:** In the static extractor (`src/static_ext/extractor.rs`), the `r2c` processing logic created brand new `indata` and `outdata` vectors for every column processed (e.g. `vec![0.0; self.fft_size]`), causing significant overhead due to inner-loop heap allocations.
**Action:** Reused the `fft_in_buffer` and `fft_out_buffer` available on the `ColumnState` struct via `std::mem::take`, resizing them only when necessary, processing the FFT, and then restoring them back onto the `state` struct. NOTE: It is important to avoid `.to_vec()` on the returned `outdata` because `fft_complex` actually needs to return the values to the caller without stealing `outdata`. Oh wait, `complex_data` returns `Vec<Complex<f32>>` so `.to_vec()` is still currently allocating to give a copy to `fft_complex`.
## 2026-10-02 - Expanding Engine FFT Allocation Fix
Learning: The expanding engine was allocating a full_series.to_vec() every column in the FFT calculation loop for realfft, as well as making duplicate allocations with to_vec on fft_complex.
Action: Reused fft_in_buffer and fft_out_buffer for realfft processing and removed unneeded to_vec() by building the spectrum from the pre-sized output buffer, then cloning it safely at the end.
