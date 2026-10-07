import json
import os
import time
from pathlib import Path

import pytest
import numpy as np


@pytest.fixture(scope="session")
def basic_features():
    return [
        "mean",
        "variance",
        "std_dev",
        "min_value",
        "max_value",
        "total_sum",
        "energy",
        "rms",
    ]


@pytest.fixture(scope="session")
def advanced_features():
    return [
        "mean",
        "variance",
        "std_dev",
        "min_value",
        "max_value",
        "skewness",
        "kurtosis",
        "autocorr-1",
        "abs_max",
        "last_loc_max",
        "first_loc_max",
        "fft_coeff-1-real",
        "fft_coeff-1-imag",
        "c3-5",
        "cid_ce",
        "mean_abs_change",
    ]


@pytest.fixture(scope="session")
def full_30_features():
    return [
        "total_sum",
        "mean",
        "variance",
        "std_dev",
        "min_value",
        "max_value",
        "median",
        "skewness",
        "kurtosis",
        "mad",
        "iqr",
        "entropy",
        "energy",
        "rms",
        "zero_crossing_rate",
        "peak_count",
        "mean_abs_change",
        "mean_change",
        "cid_ce",
        "auc",
        "pk_pk_distance",
        "zero_cross",
        "max_power_spectrum",
        "abs_sum_change",
        "count_above_mean",
        "count_below_mean",
        "longest_strike_above_mean",
        "longest_strike_below_mean",
        "abs_max",
        "first_loc_max",
        "last_loc_max",
        "first_loc_min",
        "last_loc_min",
        "length",
        "variance_larger_than_standard_deviation",
        "percentage_of_reoccurring_datapoints_to_all_datapoints",
        "percentage_of_reoccurring_values_to_all_values",
        "ratio_value_number_to_time_series_length",
        "spectral_centroid",
        "spectral_spread",
        "spectral_entropy",
        'linear_trend__attr_"slope"',
        'agg_linear_trend__attr_"slope"__chunk_len_5__f_agg_"mean"',
        "time_reversal_asymmetry_statistic__lag_1",
        "spectral_slope",
        "spectral_roll_on",
        "spectral_roll_off",
        "approx_entropy-2-0.2",
        "sample_entropy",
        "binned_entropy__max_bins_10",
        "ratio_beyond_r_sigma-2.0",
        "index_mass_quantile-0.5",
        "c3-2",
        "permutation_entropy-1-3",
        "value_count-3.0",
    ]


@pytest.fixture(scope="session")
def single_series():
    np.random.seed(42)
    return np.random.randn(1000).astype(np.float32)


@pytest.fixture(scope="session")
def single_batch(single_series):
    return single_series[None, :]


@pytest.fixture(scope="session")
def batch_100_series():
    np.random.seed(42)
    data = np.random.randn(1000, 100).astype(np.float32)
    return np.ascontiguousarray(data.T)  # one series per row


@pytest.fixture(scope="session")
def batch_1000_series():
    np.random.seed(42)
    data = np.random.randn(1000, 1000).astype(np.float32)
    return np.ascontiguousarray(data.T)  # one series per row


@pytest.fixture(scope="session")
def streaming_chunks():
    np.random.seed(42)
    n_cols = 10
    total_len = 2000
    chunk_size = 100
    data = np.random.randn(n_cols, total_len).astype(np.float32)
    return [data[:, j : j + chunk_size].copy() for j in range(0, total_len, chunk_size)]


# ── per-feature regression gate (test_bench_features.py) ──
# Baseline: min seconds per benchmark, machine-local, under .benchmarks/ (gitignored).
FEATURE_BASELINE = Path(__file__).parent.parent / ".benchmarks" / "feature_baseline.json"


def pytest_addoption(parser):
    parser.addoption(
        "--save-feature-baseline",
        action="store_true",
        help="record per-feature benchmark times as the baseline to compare against",
    )


@pytest.fixture(scope="session")
def feature_gate(request):
    """Shared state for the gate: the stored baseline and this run's timings."""
    baseline = {}
    if FEATURE_BASELINE.exists():
        baseline = json.loads(FEATURE_BASELINE.read_text())
    state = {
        "save": request.config.getoption("--save-feature-baseline"),
        "threads": os.environ.get("RAYON_NUM_THREADS", ""),
        "baseline": baseline,
        "results": {},
    }
    request.config._feature_gate = state
    # Spin for a moment so a powersave governor has ramped up the clock before
    # the first benchmark; otherwise the first few read several % slow.
    end = time.perf_counter() + 1.0
    while time.perf_counter() < end:
        pass
    yield state
    if state["save"] and state["results"]:
        merged = dict(baseline.get("times", {})) if baseline.get("threads") == state["threads"] else {}
        merged.update(state["results"])
        FEATURE_BASELINE.parent.mkdir(exist_ok=True)
        FEATURE_BASELINE.write_text(
            json.dumps({"threads": state["threads"], "times": merged}, indent=1, sort_keys=True)
        )


def pytest_terminal_summary(terminalreporter, config):
    state = getattr(config, "_feature_gate", None)
    if not state or state["save"] or not state["baseline"]:
        return
    times = state["baseline"].get("times", {})
    by_engine = {}
    for name, got in state["results"].items():
        if name in times:
            engine = name.split("[")[0].removeprefix("test_bench_feature_")
            by_engine.setdefault(engine, []).append(got / times[name] - 1)
    if not by_engine:
        return
    tr = terminalreporter
    tr.section("feature regression gate (vs .benchmarks/feature_baseline.json)")
    for engine, changes in by_engine.items():
        c = sorted(changes)
        tr.write_line(
            f"{engine:>9}: median {c[len(c) // 2]:+.1%}, "
            f">5% slower: {sum(x > 0.05 for x in c)}, >10%: {sum(x > 0.10 for x in c)} "
            f"of {len(c)}"
        )
