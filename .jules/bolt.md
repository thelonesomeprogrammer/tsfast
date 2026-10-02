## 2026-10-02 - Reusing allocations in ColumnState
**Learning:** Repetitive inner-loop allocations (like `Vec::new()` or `Vec::with_capacity()` inside feature calculation blocks like `AggLinearTrend` and `PartialAutocorr`) can be a silent performance drain in this Rust codebase.
**Action:** Instead of allocating new vectors, add them to the persistent `ColumnState` struct, extract them safely during calculation using `std::mem::take()`, `.clear()` them for use, and store them back when done.
