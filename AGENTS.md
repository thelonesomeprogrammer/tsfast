# AGENTS.md

tsfast: time-series feature extraction (tsfresh/TSFEL-compatible) in Rust with
PyO3 bindings. Three engines share one feature implementation: static
(`Extractor`), sliding (`SlidingExtractor`), expanding (`ExpandingExtractor`).

## Setup, build, test

```sh
uv sync                      # first time only; needs nightly Rust (rust-toolchain.toml pins it)
uv run pytest                # runs tests/ — rebuilds the extension automatically if src/ changed
uv run pytest tests/test_sliding.py -k energy   # narrow it down while iterating
cargo build                  # fast Rust-only type check
cargo test                   # Rust unit tests: feature name round-trips, coverage
uv run pytest benchmarks     # benchmarks (slow, not part of the normal test run)
uv run python scripts/coach_benchmark.py --skip-readme  # per-feature timings vs references -> .jules/feature_benchmarks.md
```

- **Do not run `maturin develop` or `pip install`.** uv owns the build: it
  compiles a release build into `tsfast/_tsfast*.so` whenever a `src/**/*.rs`,
  `Cargo.*` or `pyproject.toml` file changes. A second build path leaves a stale
  or debug `.so` around and tests silently run against old code.
- Nightly is required (`#![feature(portable_simd)]`). Don't try to make it build on stable.
- `cargo build` emits ~60 pre-existing warnings; don't fix unrelated ones in a feature PR.

## Layout

| Path | What |
| :--- | :--- |
| `src/types/feature.rs` | `Feature` enum + `required_compute()` (which `Compute` flags a feature needs) |
| `src/types/compute.rs` | `Compute` bitflags; add a flag by name to `compute_flags!`, bits are assigned automatically |
| `src/types/parse.rs` | `unit_features!` name table; `FromStr` and `Feature::name()` for parameterized features |
| `src/types/tests.rs` | round-trip/coverage tests over every `Feature` variant |
| `src/features/mod.rs` | `eval()`: exhaustive routing of every variant to its module |
| `src/features/*.rs` | `eval_*(feat, ctx) -> Option<f32>`: the actual feature math, grouped by domain |
| `src/common.rs` | `ColumnState`: per-column accumulators and reusable buffers |
| `src/{static_ext,sliding,expanding}/engine.rs` | engine drivers; call `features::eval` per feature |
| `src/{static_ext,sliding,expanding}/processors/` | accumulation passes (stats, diff, trend, sort, fft…) gated by `Compute` flags |
| `tsfast/` | Python package (`__init__.py`, `selection.py`); the built `.so` lands here |
| `tests/` | `test_tsfast.py` = static, `test_sliding.py`, `test_expanding.py`, plus topic files |
| `missing.md` | backlog of unimplemented tsfresh/TSFEL features |
| `.jules/*.md` | historical learning journals; may mention files that no longer exist |

## Adding a feature (checklist)

Each step is enforced: forget one and either the build or `cargo test` fails
and tells you where to look.

1. `src/types/feature.rs`: add the variant to `Feature`. Store float params as
   `u32` via `f32::to_bits()` (the enum must stay `Eq + Hash`), never `as u16`.
2. Same file: add an arm to `required_compute()` (**compile error** if missing).
   Use `C::empty()` if the feature computes directly from the raw window values.
3. Names, in `src/types/parse.rs`:
   - **No parameters**: add one line to the `unit_features!` table:
     `MyFeature => "my_feature", ["tsfresh_alias"];`. That's both parsing and naming done.
   - **With parameters**: add parsing (`parse_parameterized` for `name-p1-p2`
     style) **and** the `Feature::name()` arm, plus a sample instance in
     `parameterized_samples()` in `src/types/tests.rs`. `cargo test` checks that
     every variant has a sample and that `name()` parses back to the same value.
   - **Validate parameters while parsing** (zero lengths, `index < total`, …) and
     reject bad ones with `return None`. A bad parameter that reaches the engine
     panics and crashes the Python interpreter. Add the bad string to
     `invalid_parameters_are_rejected`.
4. `src/features/<domain>.rs`: add a match arm in that module's `eval_*` fn, then
   route the variant to it in `features::eval` (`src/features/mod.rs`): **compile
   error** if missing. All three engines call `features::eval`, so this covers
   static, sliding and expanding.
   Returning `None` from an `eval_*` arm means "not computable for this window"
   and is reported as `0.0`; return `Some(f32::NAN)` if NaN is what the
   reference library gives.
5. Only if the feature needs new accumulated state: add a `Compute` flag name
   to the `compute_flags!` list in `compute.rs` (never write a bit index), a field on `ColumnState` (`common.rs`), and update the relevant
   processor in **all three** engines (`*/processors/*.rs`). The sliding engine
   must also handle values leaving the window. Nothing enforces this step; the
   Python tests are the only check.
6. Tests: `cargo test`, plus one Python test per engine (`test_tsfast.py`,
   `test_sliding.py`, `test_expanding.py`) comparing against the tsfresh/TSFEL
   reference implementation (both are installed). Include an edge case
   (constant series or a short window). Also add a sample name to
   `tests/feature_samples.txt` (`cargo test` fails until you do): it drives
   `tests/test_engines.py`, which checks that all three engines agree and that
   the feature gives the same value alone as alongside every other feature.
   Then map it to its tsfresh/TSFEL function in `tests/references.py`
   (`REFERENCES`, or `NO_REFERENCE` with a reason): `tests/test_references.py`
   requires every feature within 1% of its reference. If tsfresh and TSFEL
   define the same quantity differently, add one feature per definition
   (e.g. `skewness` / `biased_skewness`) rather than picking one.
7. Delete the feature's row from `missing.md`.

Reference implementation: `EnergyRatioByChunks`. Before starting, grep
`src/types/parse.rs` for the feature name: several features have already been
implemented twice by parallel agents.

## Rules

- **Never delete, weaken, or comment out existing tests or asserts** to get a
  green run. If a reference value disagrees, find out why; if it's a genuine
  known difference, use `pytest.mark.xfail(reason=...)` and say so in the PR.
- **Edit tests in place; never paste a second `def test_x`** with the same
  name. Python keeps only the last definition, so the earlier copies silently
  stop running.
- Append new tests at the end of the file. Don't rewrite or reorder others.
- Don't commit scratch files: no `plan.md`, `patch_*.py`, helper scripts or
  result dumps in the repo root. Edit source files directly.
- No `.unwrap()`/`.expect()` on user-controlled input across the FFI boundary;
  map errors to `PyValueError`/`PyTypeError`.
- Hot loops: avoid per-column allocations. Reuse buffers on `ColumnState`
  (`std::mem::take` → use → put back).
- Before finishing: `cargo test` and `uv run pytest` pass, and the number of collected tests
  (`uv run pytest --collect-only -q | tail -1`) has not gone down.
