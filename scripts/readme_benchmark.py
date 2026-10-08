"""The README's headline numbers: tsrocket vs tsfresh and TSFEL on the same
features, same data, same machine.

    uv run python scripts/readme_benchmark.py                        # all cores
    RAYON_NUM_THREADS=1 uv run python scripts/readme_benchmark.py    # one core

tsfresh runs with n_jobs=0 and TSFEL with n_jobs=None (both single process), so
compare them to the one-core run; the all-cores run shows what tsrocket's
built-in parallelism adds.
"""

import os
import platform
import time
import warnings

import numpy as np
import pandas as pd
import tsfel
from tsfresh import extract_features
from tsfresh.feature_extraction import ComprehensiveFCParameters

import tsrocket

warnings.filterwarnings("ignore")

N_SERIES, LENGTH = 100, 1000
WINDOW, STREAM = 256, 2000


def best_of(fn, repeat=3):
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return min(times)


def main():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(N_SERIES, LENGTH)).cumsum(axis=1).astype(np.float32)
    df = pd.DataFrame(
        {
            "id": np.repeat(np.arange(N_SERIES), LENGTH),
            "time": np.tile(np.arange(LENGTH), N_SERIES),
            "value": x.ravel().astype(np.float64),
        }
    )
    threads = os.environ.get("RAYON_NUM_THREADS", f"all ({os.cpu_count()})")
    print(f"{platform.processor() or platform.machine()}, tsrocket threads: {threads}")
    print(f"{N_SERIES} series x {LENGTH} samples\n")

    # tsfresh: every feature of its most complete settings.
    def run_tsfresh():
        return extract_features(
            df,
            column_id="id",
            column_sort="time",
            default_fc_parameters=ComprehensiveFCParameters(),
            disable_progressbar=True,
            n_jobs=0,
        )

    columns = run_tsfresh().columns
    t_ref = best_of(run_tsfresh, repeat=1)
    ext = tsrocket.Extractor(list(columns))
    t_rocket = best_of(lambda: ext.process_2d_floats(x))
    print(
        f"tsfresh ComprehensiveFCParameters ({len(columns)} features): "
        f"tsfresh {t_ref:.2f} s, tsrocket {t_rocket * 1e3:.1f} ms -> {t_ref / t_rocket:,.0f}x"
    )

    # TSFEL: every feature, fractal domain included.
    cfg = tsfel.get_features_by_domain()
    for domain in cfg.values():
        for feature in domain.values():
            feature["use"] = "yes"
    windows = [s.astype(np.float64) for s in x]
    run_tsfel = lambda: tsfel.time_series_features_extractor(cfg, windows, fs=100, verbose=0)
    tsfel_columns = run_tsfel().columns
    t_ref = best_of(run_tsfel, repeat=1)
    ext = tsrocket.Extractor(list(tsfel_columns), fs=100)
    t_rocket = best_of(lambda: ext.process_2d_floats(x))
    print(
        f"TSFEL, all domains ({len(tsfel_columns)} features): "
        f"TSFEL {t_ref:.2f} s, tsrocket {t_rocket * 1e3:.1f} ms -> {t_ref / t_rocket:,.0f}x"
    )

    # Streaming: one new sample arrives; features of the latest window.
    stream = x[:1, : WINDOW + STREAM]
    sliding = tsrocket.SlidingExtractor(list(columns), 1, WINDOW)
    sliding.update(stream[:, :WINDOW])
    start = time.perf_counter()
    for i in range(WINDOW, WINDOW + STREAM):
        sliding.update(stream[:, i : i + 1])
    t_rocket = (time.perf_counter() - start) / STREAM
    one = df[df.id == 0].iloc[:WINDOW]
    t_ref = best_of(
        lambda: extract_features(
            one,
            column_id="id",
            column_sort="time",
            default_fc_parameters=ComprehensiveFCParameters(),
            disable_progressbar=True,
            n_jobs=0,
        ),
        repeat=3,
    )
    print(
        f"Streaming, window {WINDOW}, all {len(columns)} tsfresh features per new sample: "
        f"tsfresh {t_ref * 1e3:.0f} ms, tsrocket {t_rocket * 1e6:.0f} us -> {t_ref / t_rocket:,.0f}x"
    )


if __name__ == "__main__":
    main()
