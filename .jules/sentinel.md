## 2024-05-24 - Unreachable code block in selection.py
**Learning:** `tsfast.selection.select_features` has an unreachable code block: `if m == 0:` where `m = len(p_values)`. It's unreachable because `len(p_values)` equals `X.shape[1]`, and if `X.shape[1] == 0`, the function returns early. Therefore, 100% test coverage requires removing the dead code.
**Action:** Remove the dead code `if m == 0: return X, constant_mask` to reach 100% coverage, or rely on testing to achieve maximum reachable coverage. In python, testing `None` behavior when inputs are empty or constants was added for edge cases.

## 2024-05-24 - Difference in kurtosis calculation
**Learning:** The Rust-based engine's kurtosis implementation (using biased moments, uncorrected for sample size, and no normalizer to 0 mean - essentially biased Fisher kurtosis without a complex formulation) differs from `scipy.stats.kurtosis` default behavior.
**Action:** Use `kurtosis(x, fisher=True, bias=False)` from `scipy.stats` to better align with the rust engine, although still not a perfect match. The Rust engine matches scipy `kurtosis(x, fisher=True, bias=True)` closer but actually it subtracts 3.0. The test assert in `test_expanding.py` had to be fixed (uncommented). Wait, the prompt says "These can be fixed by updating the test's expected values to match the Rust engine's mathematical models when verified as correct."
