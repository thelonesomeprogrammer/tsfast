1. **Update `src/types/feature.rs`**
   - Add `NegativeTurning` and `PositiveTurning` to `pub enum Feature`.
   - Update `required_compute` for them to return `C::PEAKS` (or better yet, `C::PEAKS` as they both calculate turning points using the same local context). Let's trace `TurningPoints`. `TurningPoints` requires `C::PEAKS | C::TURNING_PTS`. Maybe we should just add `C::TURNING_PTS`? No, wait. `C::TURNING_PTS` is literally only mapped to `TurningPoints`. Currently `TurningPoints` logic is actually missing in `eval_crossings_peaks`.
   Wait, if we can just reuse `C::PEAKS`, we can calculate both maxima and minima inside the `PEAKS` compute pass.
   - Let's check `src/types/compute.rs`. We can use `C::PEAKS`.
   - Let's add `troughs: u32` to `ColumnState` in `src/common.rs`.
   - Wait, `PositiveTurning` is exactly the same as `PeakCount`!
     Is `PositiveTurning` just an alias for `PeakCount`? The task says "verify whether existing features cover these. TSFEL's negative_turning and positive_turning split the total into minima vs. maxima counts separately." So `PositiveTurning` = `PeakCount`, and `NegativeTurning` = minima count (which is new). But the task explicitly says: "Add NegativeTurning and PositiveTurning variants to Feature enum." So we must add both explicitly to the `Feature` enum.

2. **Update `ColumnState` in `src/common.rs`**
   - Add `troughs: u32` next to `peaks: u32` in `ColumnState`.
   - Initialize it to `0` in `ColumnState::new`.

3. **Update `src/static_ext/processors/trend_processor.rs`**
   - In `process_simd`:
     ```rust
     let mask_peaks = chunk.simd_gt(left) & chunk.simd_gt(right);
     state.peaks += mask_peaks.to_bitmask().count_ones();
     let mask_troughs = chunk.simd_lt(left) & chunk.simd_lt(right);
     state.troughs += mask_troughs.to_bitmask().count_ones();
     ```
   - In `process_sequential`:
     ```rust
     if val > values[global_idx - 1] && val > values[global_idx + 1] {
         state.peaks += 1;
     }
     if val < values[global_idx - 1] && val < values[global_idx + 1] {
         state.troughs += 1;
     }
     ```

4. **Update `src/sliding/processors/diff_processor.rs`**
   - Around line 26: `state.peaks = 0; state.troughs = 0;`
   - Loop around line 30:
     ```rust
     if values[i] > values[i - 1] && values[i] > values[i + 1] {
         state.peaks += 1;
     }
     if values[i] < values[i - 1] && values[i] < values[i + 1] {
         state.troughs += 1;
     }
     ```

5. **Update `src/expanding/processors/trend_processor.rs`**
   - In `process_sequential`:
     ```rust
     if state.prev_val > state.prev_prev_val && state.prev_val > val {
         state.peaks += 1;
     }
     if state.prev_val < state.prev_prev_val && state.prev_val < val {
         state.troughs += 1;
     }
     ```

6. **Update `src/features/crossings_peaks.rs`**
   - Add mappings in `eval_crossings_peaks`:
     ```rust
     Feature::PositiveTurning => state.peaks as f32,
     Feature::NegativeTurning => state.troughs as f32,
     Feature::TurningPoints => (state.peaks + state.troughs) as f32,
     ```
   - Actually, wait, the existing `Feature::TurningPoints` implementation is missing! I should implement it here as well as a bonus.

7. **String parsing mapping in `src/types/parse.rs`**
   - `NegativeTurning` <-> `"negative_turning"`
   - `PositiveTurning` <-> `"positive_turning"`

8. **Benchmarks**
   - Add to `FEATURES` array in `benches/feature_benchmarks.rs`.
   - Add to `tests/test_tsfast.py` to assert against TSFEL.
