1. **Add to `Feature` enum**
   - Add `NegativeTurning` and `PositiveTurning` to `src/types/feature.rs` in the `Feature` enum.
   - Add to `required_compute` matching their dependencies (`C::TURNING_PTS | C::PEAKS`). Wait, let's just make it `C::PEAKS` (or `C::TURNING_PTS` if we use it, maybe `C::PEAKS | C::TURNING_PTS` is what `TurningPoints` already does).
   - Actually, let's check what TSFEL `negative_turning` and `positive_turning` exactly mean.
   - Wait, `negative_turning` is just local minima, and `positive_turning` is local maxima.
   - The current `PeakCount` is local maxima. TSFEL's `positive_turning` might be exactly `PeakCount`! Let's check TSFEL source or write a quick Python script to check if `tsfel.feature_extraction.features.positive_turning` is just `PeakCount`. No, the task says: "Note: tsfast already has TurningPoints (total turning points) and PeakCount — verify whether existing features cover these. TSFEL's negative_turning and positive_turning split the total into minima vs. maxima counts separately."

2. **Verify TSFEL `positive_turning`**
   - Write a quick python script to compare `tsfel` `positive_turning` with `tsfast` `PeakCount`.
   - If they are the same logic but different names, maybe we just alias `PositiveTurning` or something, but the prompt says: "Add NegativeTurning and PositiveTurning variants to Feature enum. ... TSFEL's negative_turning and positive_turning split the total into minima vs. maxima counts separately."
   - Wait, let's see how `TurningPoints` is implemented. Wait, `TurningPoints` is NOT implemented in `src/features/crossings_peaks.rs`! It just maps `Feature::TurningPoints => C::PEAKS | C::TURNING_PTS`, but the calculation is missing in `eval_crossings_peaks` (it returns `None`)? Let's check `src/features/crossings_peaks.rs` again. Yes, it returns `None` for `TurningPoints`! Or maybe it's in another module?

Let me investigate `TurningPoints` in `src/features/crossings_peaks.rs` or elsewhere.
