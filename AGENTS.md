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
uv run pytest benchmarks     # benchmarks (slow, not part of the normal test run)
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
| `src/types/compute.rs` | `Compute` bitflags (`u128`, check the last used bit before adding one) |
| `src/types/parse.rs` | `FromStr` (feature-name string → `Feature`) and `Feature::name()` (inverse) |
| `src/features/*.rs` | `eval_*(feat, ctx) -> Option<f32>` — the actual feature math, grouped by domain |
| `src/common.rs` | `ColumnState`: per-column accumulators and reusable buffers |
| `src/{static_ext,sliding,expanding}/engine.rs` | engine drivers; each calls every `eval_*` in turn |
| `src/{static_ext,sliding,expanding}/processors/` | accumulation passes (stats, diff, trend, sort, fft…) gated by `Compute` flags |
| `tsfast/` | Python package (`__init__.py`, `selection.py`); the built `.so` lands here |
| `tests/` | `test_tsfast.py` = static, `test_sliding.py`, `test_expanding.py`, plus topic files |
| `missing.md` | backlog of unimplemented tsfresh/TSFEL features |
| `.jules/*.md` | historical learning journals; may mention files that no longer exist |

## Adding a feature (checklist)

Reference implementation: `EnergyRatioByChunks` (`git show b346306`).

1. `src/types/feature.rs`: add the variant to `Feature`. Store float params as
   `u32` via `f32::to_bits()` (the enum must stay `Eq + Hash`), never `as u16`.
2. Same file: add an arm to `required_compute()`. Use `C::empty()` if the
   feature computes directly from the raw window values.
3. `src/types/parse.rs`: add parsing in `from_str`/`parse_parameterized` **and**
   the inverse in `Feature::name()`. **Validate parameters here** (zero lengths,
   `index < total`, …) and reject them by returning an error. A bad parameter
   that reaches the engine panics and crashes the Python interpreter.
4. `src/features/<domain>.rs`: add a match arm in the matching `eval_*`
   function. Every engine calls the same `eval_*` functions, so this one change
   covers static, sliding and expanding. **If no `eval_*` handles a variant, the
   engine silently returns `0.0`**, so a feature that always yields 0 isn't wired up.
5. Only if the feature needs new accumulated state: add a `Compute` flag in
   `compute.rs`, a field on `ColumnState` (`common.rs`), and update the relevant
   processor in **all three** engines (`*/processors/*.rs`). The sliding engine
   must also handle values leaving the window.
6. Tests: add one test per engine (`test_tsfast.py`, `test_sliding.py`,
   `test_expanding.py`) comparing against the tsfresh/TSFEL reference
   implementation (both are installed). Include an edge case (constant series or
   a short window).
7. Delete the feature's row from `missing.md`.

Before starting, grep `src/types/parse.rs` for the feature name: several
features have already been implemented twice by parallel agents.

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
- Before finishing: `uv run pytest` passes, and the number of collected tests
  (`uv run pytest --collect-only -q | tail -1`) has not gone down.
