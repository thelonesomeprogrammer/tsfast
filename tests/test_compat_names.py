"""tsfresh and TSFEL output column names are accepted unchanged.

The promise: train on tsfresh/TSFEL output, then deploy with
`tsrocket.Extractor(list(df.columns))` and get the same values. These tests
run both libraries with their most complete default settings and check every
column they produce: it must parse, and its value must match.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
import tsfel
from tsfresh import extract_features
from tsfresh.feature_extraction import ComprehensiveFCParameters

import tsrocket

RTOL = 0.01
ATOL = 1e-5



def _series(n, kind):
    rng = np.random.default_rng(n)
    if kind == "ties":  # repeated levels, like tests/test_references.py
        return rng.choice(rng.normal(0.5, 1.0, size=40), size=n).astype(np.float32)
    return rng.normal(size=n).cumsum().astype(np.float32)


def _mismatches(columns, reference, x, **kwargs):
    ext = tsrocket.Extractor(list(columns), **kwargs)
    got = ext.process_2d_floats(x[None])[0]
    return [
        (c, name, g, w)
        for c, name, g, w in zip(columns, ext.feature_names, got, reference)
        if not np.isclose(g, w, rtol=RTOL, atol=ATOL, equal_nan=True)
    ]


@pytest.mark.parametrize("kind", ["ties", "walk"])
@pytest.mark.parametrize("n", [200, 600])
def test_every_tsfresh_column_name(n, kind):
    x = _series(n, kind)
    df = pd.DataFrame({"id": 0, "time": np.arange(n), "value": x.astype(np.float64)})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = extract_features(
            df,
            column_id="id",
            column_sort="time",
            default_fc_parameters=ComprehensiveFCParameters(),
            disable_progressbar=True,
            n_jobs=0,
        )
    columns = list(ref.columns)
    if kind == "ties":
        # tsfresh ranks tied values with numpy's unstable SIMD argsort, so its
        # permutation entropy on ties depends on the CPU (see references.py).
        columns = [c for c in columns if "permutation_entropy" not in c]
    assert len(columns) > 700
    assert _mismatches(columns, ref[columns].iloc[0].to_numpy(), x) == []


@pytest.mark.parametrize("fs", [100.0, 50.0])
@pytest.mark.parametrize("n", [200, 600])
def test_every_tsfel_column_name(n, fs):
    x = _series(n, "ties")
    cfg = tsfel.get_features_by_domain()
    for domain in cfg.values():
        for feature in domain.values():
            feature["use"] = "yes"  # fractal features are off by default
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = tsfel.time_series_features_extractor(cfg, x.astype(np.float64), fs=fs, verbose=0)
    assert len(ref.columns) > 150
    assert _mismatches(ref.columns, ref.iloc[0].to_numpy(), x, fs=fs) == []


def test_names_without_kind_or_channel_and_with_other_prefixes():
    x = _series(300, "walk")[None]
    pairs = [
        ('fft_coefficient__attr_"abs"__coeff_3', "fft_coeff-3-abs"),
        ('my__sensor__fft_coefficient__attr_"abs"__coeff_3', "fft_coeff-3-abs"),
        ("cid_ce__normalize_True", "cid_ce_normalized"),
        ("Spectral centroid", "spectral_centroid"),
        ("acc_x_Spectral centroid", "spectral_centroid"),
        ("acc_x_LPCC_3", "lpcc-3"),
    ]
    ext = tsrocket.Extractor([a for a, _ in pairs])
    assert ext.feature_names == [b for _, b in pairs]
    want = tsrocket.Extractor([b for _, b in pairs]).process_2d_floats(x)
    np.testing.assert_array_equal(ext.process_2d_floats(x), want)


def test_frequency_suffixed_tsfel_names_follow_fs():
    # TSFEL labels wavelet scales and spectrogram bins by frequency, which
    # depends on the sampling frequency.
    assert tsrocket.Extractor(["0_Wavelet energy_25.0Hz"], fs=100).feature_names == [
        "wavelet_energy-0"
    ]
    assert tsrocket.Extractor(["0_Wavelet energy_25.0Hz"], fs=200).feature_names == [
        "wavelet_energy-1"
    ]
    with pytest.raises(ValueError, match="Unknown feature"):
        tsrocket.Extractor(["0_Wavelet energy_24.0Hz"], fs=100)


@pytest.mark.parametrize(
    "name",
    [
        'value__query_similarity_count__query_"abc"__threshold_0.0',
        'value__fft_coefficient__attr_"abs"__coeff_3__bogus_1',
        'value__agg_linear_trend__attr_"slope"__chunk_len_5__f_agg_"mode"',
        'value__change_quantiles__f_agg_"mean"__isabs_Maybe__qh_0.2__ql_0.0',
        'value__augmented_dickey_fuller__attr_"teststat"__autolag_"BIC"',
        "value__number_peaks__n_0",
        "0_ECDF Percentile_2",
        "0_Not a feature",
    ],
)
def test_bad_external_names_are_rejected(name):
    with pytest.raises(ValueError, match="Unknown feature"):
        tsrocket.Extractor([name])


def test_unknown_feature_suggests_a_name():
    with pytest.raises(ValueError, match="did you mean `spectral_centroid`"):
        tsrocket.Extractor(["spectral_centriod"])
    with pytest.raises(ValueError, match=r"did you mean `fft_coeff-\.\.\.`"):
        tsrocket.Extractor(["fft_coef-3-abs"])


def _sensor_frame(lengths, seed=0):
    """Wide tsfresh-style frame: id, time and two signals. Values are exact in
    float32 (a continuous one, and an integer one full of ties), so any
    difference is tsrocket's, not float32 rounding of the input."""
    rng = np.random.default_rng(seed)
    parts = []
    for i, n in enumerate(lengths):
        parts.append(
            pd.DataFrame(
                {
                    "id": i,
                    "time": np.arange(n),
                    "temp": rng.normal(size=n).astype(np.float32).astype(np.float64),
                    "count": np.round(rng.normal(size=n) * 3),
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def test_extract_features_reproduces_a_tsfresh_frame():
    # The deployment workflow: tsfresh features at training time, then
    # tsrocket.extract_features(df, X_train.columns) at inference.
    df = _sensor_frame([12, 37, 150, 400, 150])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X = extract_features(
            df,
            column_id="id",
            column_sort="time",
            default_fc_parameters=ComprehensiveFCParameters(),
            disable_progressbar=True,
            n_jobs=0,
        )
    # Shuffled rows: extract_features sorts by column_sort itself.
    Y = tsrocket.extract_features(
        df.sample(frac=1, random_state=0), X.columns, column_id="id", column_sort="time"
    )
    assert list(Y.columns) == list(X.columns)
    assert list(Y.index) == list(X.index)
    # Ties: see test_every_tsfresh_column_name.
    cols = [c for c in X.columns if not c.startswith("count__permutation_entropy")]
    # atol 1e-4: agg_linear_trend's rvalue of chunk variances is a near-zero
    # correlation, computed from float32 chunk statistics.
    close = np.isclose(Y[cols], X[cols], rtol=RTOL, atol=1e-4, equal_nan=True)
    bad = [(c, X.index[r]) for r, j in zip(*np.nonzero(~close)) for c in [cols[j]]]
    assert bad == []


def test_extract_features_input_formats():
    df = _sensor_frame([30, 50, 30], seed=1)
    names = ["temp__mean", "count__maximum", "temp_Spectral centroid", "count_Median"]
    wide = tsrocket.extract_features(df, names, column_id="id", column_sort="time")
    assert list(wide.columns) == names
    assert list(wide.index) == [0, 1, 2]
    for i, n in enumerate([30, 50, 30]):
        g = df[df.id == i]
        assert wide.loc[i, "temp__mean"] == pytest.approx(g.temp.mean(), rel=1e-5)
        assert wide.loc[i, "count__maximum"] == g["count"].max()

    long = df.melt(id_vars=["id", "time"], var_name="kind", value_name="value")
    from_long = tsrocket.extract_features(
        long.sample(frac=1, random_state=2),
        names,
        column_id="id",
        column_sort="time",
        column_kind="kind",
        column_value="value",
    )
    pd.testing.assert_frame_equal(from_long, wide)

    # One signal: names need no kind prefix.
    one = tsrocket.extract_features(df[["id", "time", "temp"]], ["mean", "0_Max"], column_sort="time")
    np.testing.assert_allclose(one["mean"], wide["temp__mean"])

    with pytest.raises(ValueError, match="can't tell which"):
        tsrocket.extract_features(df, ["mean"], column_sort="time")
    with pytest.raises(ValueError, match="column_value"):
        tsrocket.extract_features(long, ["value__mean"], column_kind="kind")

def test_extract_features_edge_cases():
    df = pd.DataFrame({
        "id": [1, 1, 2, 2],
        "time": [1, 2, 1, 2],
        "temp": [1.0, 2.0, 3.0, 4.0]
    })

    # Missing column_id
    with pytest.raises(ValueError, match="is not a column"):
        tsrocket.extract_features(df, ["mean"], column_id="missing_id")

    # column_value provided but column_kind is not
    result = tsrocket.extract_features(df, ["mean"], column_id="id", column_sort="time", column_value="temp")
    assert list(result.columns) == ["mean"]
    assert len(result) == 2

    # No signal columns
    df_no_signals = pd.DataFrame({
        "id": [1, 1],
        "time": [1, 2]
    })
    with pytest.raises(ValueError, match="no signal columns to compute features on"):
        tsrocket.extract_features(df_no_signals, ["mean"], column_id="id", column_sort="time")
