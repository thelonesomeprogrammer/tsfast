# Missing Features in tsfast

> **Agents:** this is the backlog. Pick tasks from here, and delete a row in the same PR that implements it.

Audited against the installed `tsfresh` (76 feature calculators) and `tsfel`
(67 feature functions). Every name below was checked against
`src/types/feature.rs`, `src/types/parse.rs`, and `tests/feature_samples.txt`
and does not exist under any alias. Everything *not* listed here is
implemented and verified within 1% of its reference by
`tests/test_references.py`.

---

## TSFresh (1 missing)

| Feature | Notes |
| :--- | :--- |
| **`linear_trend_timewise`** | OLS regression against an explicit `DatetimeIndex`. tsfast's engines take no timestamp input, so this needs a design decision before implementation, not just a port. |
