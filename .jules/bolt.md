## 2025-02-28 - Removed unnecessary Vector allocation inside hot loop
**Learning:** In the `SlidingExtractor`, memory allocation using `.to_vec()` creates a bottleneck within tight loops due to memory overhead when a simple slice reference is completely valid.
**Action:** Always prefer borrowing memory `&[T]` instead of explicitly cloning it with `to_vec()` or similar allocating methods when the underlying data (`history`) lives longer than the usage.
