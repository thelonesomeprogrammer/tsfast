"""Coach benchmark: tsfast vs the tsfresh/TSFEL reference, per feature and engine.

Every feature in tests/feature_samples.txt is timed on the same windows in the
static, sliding and expanding engines and, when it has one, against its
reference implementation from tests/references.py (the table the accuracy tests
use, so the benchmark covers exactly the features that are checked).

Outputs:
  .jules/feature_benchmarks.csv / .md  per-feature microseconds per window
  .jules/benchmarks.csv                appended aggregate history (all features
                                       in one extractor), plotted by the chart
  README.md "## Latest Results"        unless --skip-readme

    uv run python scripts/coach_benchmark.py [--quick] [--skip-readme]
"""
import argparse
import csv
import datetime
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import tsfast

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
from references import REFERENCES  # noqa: E402

warnings.filterwarnings("ignore")

FEATURES = [
    line.strip()
    for line in (ROOT / "tests" / "feature_samples.txt").read_text().splitlines()
    if line.strip() and not line.startswith("#")
]


def get_git_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()[:7]
    except:
        return 'unknown'

def append_to_csv(filepath, new_rows):
    import csv
    header = ["Date", "CommitHash", "Benchmark_Name", "Metric_Value", "Unit", "Delta_From_Last"]
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    file_exists = os.path.exists(filepath)

    if file_exists:
        df = pd.read_csv(filepath)
    else:
        df = pd.DataFrame(columns=header)

    date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    commit_hash = get_git_hash()

    with open(filepath, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(header)

        for row in new_rows:
            bench_name, metric_val, unit = row
            # Calculate Delta_From_Last
            past_runs = df[df['Benchmark_Name'] == bench_name]
            delta = 0.0
            if not past_runs.empty:
                last_val = past_runs.iloc[-1]['Metric_Value']
                if last_val != 0:
                    delta = ((metric_val - last_val) / last_val) * 100.0

            writer.writerow([date_str, commit_hash, bench_name, metric_val, unit, round(delta, 2)])

def generate_chart():
    """Trend of the all-features-in-one-extractor timings and feature counts."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    csv_path = ROOT / ".jules" / "benchmarks.csv"
    df = pd.read_csv(csv_path)
    series = {
        "tsfast static": "tsfast_static_all_features",
        "tsfast sliding": "tsfast_sliding_all_features",
        "tsfast expanding": "tsfast_expanding_all_features",
        "references (sum)": "reference_all_features",
    }
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    ax1.set_title("All features, one extractor: ms per window", fontweight="bold")
    for label, name in series.items():
        values = df[df["Benchmark_Name"] == name]["Metric_Value"].values
        if len(values):
            ax1.plot(range(1, len(values) + 1), values, marker="o", label=label)
    ax1.set_yscale("log")
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    ax1.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    ax2.set_title("Feature count", fontweight="bold")
    for label, name in [("tsfast", "tsfast_features"), ("with reference", "referenced_features")]:
        values = df[df["Benchmark_Name"] == name]["Metric_Value"].values
        if len(values):
            ax2.plot(range(1, len(values) + 1), values, marker="o", label=label)
    ax2.set_xlabel("Benchmark run")
    ax2.set_ylim(bottom=0)
    ax2.grid(True, ls="--", alpha=0.5)
    ax2.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    fig.tight_layout()
    chart_path = ROOT / ".jules" / "benchmark_trends.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    print(f"Chart saved to {chart_path}")


def update_readme():
    df = pd.read_csv('.jules/benchmarks.csv')
    latest_date = df['Date'].max()
    latest_df = df[df['Date'] == latest_date].copy()

    # Add emojis
    def get_emoji(val, unit):
        if unit == "count":
            return "⚪"

        # Check if lower is better (time, memory size)
        lower_is_better = unit in ["ms", "ns/iter", "MB"]

        if val > 5.0:
            return "🔴" if lower_is_better else "🟢"
        elif val < -5.0:
            return "🟢" if lower_is_better else "🔴"
        return "⚪"

    latest_df['Direction'] = latest_df.apply(lambda row: get_emoji(row['Delta_From_Last'], row['Unit']), axis=1)

    md_table = latest_df[['Date', 'CommitHash', 'Benchmark_Name', 'Metric_Value', 'Unit', 'Delta_From_Last', 'Direction']].to_markdown(index=False)

    try:
        with open('README.md', 'r') as f:
            readme_content = f.read()
    except FileNotFoundError:
        readme_content = "## Latest Results\n"

    start_index = readme_content.find('## Latest Results')
    if start_index != -1:
        new_content = readme_content[:start_index] + "## Latest Results\n" + md_table + "\n"
        with open('README.md', 'w') as f:
            f.write(new_content)
        print("README.md updated.")
    else:
        print("Could not find '## Latest Results' in README.md")

def make_batch(series):
    """Extractor input: one series per row, contiguous float32."""
    return np.ascontiguousarray(series, dtype=np.float32)


def per_call(fn, min_time):
    """Seconds per call of fn(), repeating until min_time has elapsed."""
    fn()  # warm-up (plans FFTs, fills caches)
    calls, start = 0, time.perf_counter()
    while True:
        fn()
        calls += 1
        elapsed = time.perf_counter() - start
        if elapsed >= min_time:
            return elapsed / calls


def bench_feature(feature, data, window, step, min_time):
    """Microseconds per window per series for each engine and the reference."""
    n_cols = data.shape[0]
    windows = make_batch(data[:, :window])
    prime = make_batch(data[:, :window])
    more = make_batch(data[:, window:window + step])

    static = tsfast.Extractor([feature])
    t_static = per_call(lambda: static.process_2d_floats(windows), min_time) / n_cols

    def sliding_step():
        ext = tsfast.SlidingExtractor([feature], n_cols, window)
        ext.update(prime)
        start = time.perf_counter()
        ext.update(more)  # `step` new windows per column
        return time.perf_counter() - start

    def expanding_step():
        ext = tsfast.ExpandingExtractor([feature], n_cols)
        ext.update(prime)
        start = time.perf_counter()
        ext.update(more)  # one update per column, series grows by `step`
        return time.perf_counter() - start

    def median_of(fn):
        samples, start = [], time.perf_counter()
        while not samples or time.perf_counter() - start < min_time:
            samples.append(fn())
        return float(np.median(samples))

    t_sliding = median_of(sliding_step) / (n_cols * step)
    t_expanding = median_of(expanding_step) / n_cols

    lib, t_ref = "", float("nan")
    if feature in REFERENCES:
        lib, ref = REFERENCES[feature]
        x = data[0, :window].astype(np.float64)
        t_ref = per_call(lambda: ref(x), min_time)

    us = 1e6
    return {
        "feature": feature,
        "static_us": t_static * us,
        "sliding_us": t_sliding * us,
        "expanding_us": t_expanding * us,
        "reference": lib,
        "reference_us": t_ref * us,
        "speedup": t_ref / t_static if t_static > 0 else float("nan"),
    }


def write_feature_table(rows, window, step, n_cols):
    out = ROOT / ".jules"
    out.mkdir(exist_ok=True)
    df = pd.DataFrame(rows).sort_values("feature")
    df.to_csv(out / "feature_benchmarks.csv", index=False, float_format="%.3f")

    def fmt(v, digits=1):
        return "" if pd.isna(v) else f"{v:,.{digits}f}"

    lines = [
        "# Per-feature benchmark",
        "",
        f"Microseconds per window per series. Window {window}, {n_cols} series, "
        f"{step} new values per sliding/expanding update. Commit {get_git_hash()}, "
        f"{datetime.datetime.now():%Y-%m-%d %H:%M}.",
        "Sliding = per new window; expanding = per update. Reference = the tsfresh/TSFEL",
        "function tsfast is tested against, called on one window; speed-up = reference / static.",
        "",
        "| feature | static | sliding | expanding | reference | ref µs | speed-up |",
        "|:--|--:|--:|--:|:--|--:|--:|",
    ]
    for r in df.itertuples():
        ref = r.reference or "—"
        lines.append(
            f"| `{r.feature}` | {fmt(r.static_us, 2)} | {fmt(r.sliding_us, 2)} | {fmt(r.expanding_us, 2)} "
            f"| {ref} | {fmt(r.reference_us)} | {fmt(r.speedup)}× |".replace("| × |", "| |")
        )
    (out / "feature_benchmarks.md").write_text("\n".join(lines) + "\n")
    return df


def bench_all_in_one(data, window, step, min_time):
    """All features in one extractor: the realistic deployment cost."""
    n_cols = data.shape[0]
    prime, more = make_batch(data[:, :window]), make_batch(data[:, window:window + step])
    static = tsfast.Extractor(FEATURES)
    t_static = per_call(lambda: static.process_2d_floats(prime), min_time) / n_cols

    def timed(make):
        ext = make()
        ext.update(prime)
        start = time.perf_counter()
        ext.update(more)
        return time.perf_counter() - start

    t_sliding = timed(lambda: tsfast.SlidingExtractor(FEATURES, n_cols, window)) / (n_cols * step)
    t_expanding = timed(lambda: tsfast.ExpandingExtractor(FEATURES, n_cols)) / n_cols
    return t_static, t_sliding, t_expanding


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="shorter timings, for a smoke run")
    parser.add_argument("--skip-readme", action="store_true", help="don't rewrite README.md")
    parser.add_argument("--skip-history", action="store_true", help="don't append .jules/benchmarks.csv")
    args = parser.parse_args()

    n_cols, window, step = 20, 256, 32
    min_time = 0.02 if args.quick else 0.2
    rng = np.random.default_rng(42)
    data = rng.normal(size=(n_cols, window + step)).astype(np.float32)

    rows = []
    for i, feature in enumerate(FEATURES, 1):
        rows.append(bench_feature(feature, data, window, step, min_time))
        print(f"[{i}/{len(FEATURES)}] {feature}", file=sys.stderr)
    df = write_feature_table(rows, window, step, n_cols)
    print(f"Wrote .jules/feature_benchmarks.md ({len(df)} features)")

    t_static, t_sliding, t_expanding = bench_all_in_one(data, window, step, min_time)
    ref_total = df["reference_us"].sum(skipna=True) / 1e3  # ms, sum over referenced features
    print(f"All {len(FEATURES)} features, ms per window: static {t_static * 1e3:.3f}, "
          f"sliding {t_sliding * 1e3:.3f}, expanding {t_expanding * 1e3:.3f}; "
          f"references (sum of {df['reference'].astype(bool).sum()}) {ref_total:.3f}")

    if not args.skip_history:
        append_to_csv(str(ROOT / ".jules" / "benchmarks.csv"), [
            ("tsfast_static_all_features", t_static * 1e3, "ms"),
            ("tsfast_sliding_all_features", t_sliding * 1e3, "ms"),
            ("tsfast_expanding_all_features", t_expanding * 1e3, "ms"),
            ("reference_all_features", ref_total, "ms"),
            ("tsfast_features", len(FEATURES), "count"),
            ("referenced_features", int(df["reference"].astype(bool).sum()), "count"),
        ])
        generate_chart()
    if not args.skip_readme:
        update_readme()


if __name__ == "__main__":
    main()
