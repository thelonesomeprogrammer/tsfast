"""pandas front end: tsfresh's input formats in, a feature frame out."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from ._tsrocket import Extractor


def extract_features(
    timeseries,
    features: Iterable[str],
    column_id: str = "id",
    column_sort: str | None = None,
    column_kind: str | None = None,
    column_value: str | None = None,
    fs: float = 100.0,
):
    """Compute ``features`` for every series in a DataFrame.

    Takes the same input formats as ``tsfresh.extract_features``:

    - wide: one row per sample, a ``column_id`` column, an optional
      ``column_sort`` column and one column per signal;
    - long: ``column_kind`` names the signal and ``column_value`` holds the
      value.

    ``features`` are tsrocket, tsfresh or TSFEL names, typically the columns of
    the training frame. A tsfresh name's kind (``temperature__mean``) or a
    TSFEL name's channel (``temperature_Mean``) selects the signal it is
    computed on; names without one need a frame with a single signal.

    Returns a DataFrame with one row per id (sorted) and one float32 column
    per feature, named exactly as given, in the given order: the same layout
    as the frame the model was trained on.
    """
    import pandas as pd

    features = [str(f) for f in features]
    df = timeseries
    if column_id not in df.columns:
        raise ValueError(f"column_id {column_id!r} is not a column")
    if column_sort is not None:
        df = df.sort_values([column_id, column_sort], kind="stable")

    if column_kind is not None:
        if column_value is None:
            raise ValueError("column_kind needs column_value")
        position = df.groupby([column_id, column_kind], sort=False).cumcount()
        df = df.assign(_position=position).pivot(
            index=[column_id, "_position"], columns=column_kind, values=column_value
        )
        df = df.reset_index(level=column_id)
        signals = [c for c in df.columns if c != column_id]
    elif column_value is not None:
        signals = [column_value]
    else:
        signals = [c for c in df.columns if c not in (column_id, column_sort)]
    if not signals:
        raise ValueError("no signal columns to compute features on")

    groups = df.groupby(column_id, sort=True)
    ids = list(groups.groups)
    out = np.full((len(ids), len(features)), np.nan, dtype=np.float32)

    by_signal: dict = {}
    for i, name in enumerate(features):
        by_signal.setdefault(_signal_of(name, signals), []).append(i)

    lengths = groups.size().to_numpy()
    for signal, cols in by_signal.items():
        extractor = Extractor([features[i] for i in cols], fs=fs)
        series = [g[signal].to_numpy(dtype=np.float32) for _, g in groups]
        # The extractor takes equal-length rows: one batch per length.
        for length in np.unique(lengths):
            rows = np.flatnonzero(lengths == length)
            batch = np.stack([series[r] for r in rows])
            out[np.ix_(rows, cols)] = extractor.process_2d_floats(batch)

    return pd.DataFrame(out, index=pd.Index(ids, name=column_id), columns=features)


def _signal_of(name: str, signals: list):
    """The signal a feature name is computed on: the longest signal name that
    prefixes it as a tsfresh kind (``kind__``) or TSFEL channel (``ch_``)."""
    matches = [s for s in signals if name.startswith((f"{s}__", f"{s}_"))]
    if matches:
        return max(matches, key=lambda s: len(str(s)))
    if len(signals) == 1:
        return signals[0]
    raise ValueError(
        f"{name!r}: can't tell which of {signals} it is for; prefix it with the "
        f"column name, like tsfresh (`<column>__<feature>`) or TSFEL (`<column>_<Feature>`)"
    )
