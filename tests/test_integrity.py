import numpy as np
import pandas as pd
import pytest
import torch
from cnx500.data import before_boundary, classify, inner_split, prepare, read_bars
from cnx500.metrics import score
from cnx500.models import CompactTransformer, Baseline, fit_neural, neural_probabilities


def synthetic_bars(n=1000):
    rng = np.random.default_rng(17)
    dates = pd.bdate_range("2010-01-01", periods=n)
    close = 100 * np.exp(np.cumsum(rng.normal(0, .01, n)))
    opened = close * np.exp(rng.normal(0, .003, n))
    timestamps = dates.tz_localize("Asia/Kolkata") + pd.Timedelta(hours=9, minutes=15)
    return pd.DataFrame({"timestamp": timestamps.tz_convert("UTC").as_unit("s").asi8,
                         "date": dates, "open": opened, "high": np.maximum(opened, close)*1.01,
                         "low": np.minimum(opened, close)*.99, "close": close,
                         "volume": rng.integers(1000, 5000, n).astype(float)})


def test_exact_label_boundaries_and_missing_targets():
    z = [-1.01, -1., -.251, -.25, 0., .25, .251, 1., 1.01, np.nan]
    assert classify(z).tolist() == [0, 1, 1, 2, 2, 2, 3, 3, 4, -1]


def test_future_perturbation_cannot_change_past_features_or_normalization():
    original = synthetic_bars()
    modified = original.copy()
    modified.loc[700:, ["open", "high", "low", "close"]] *= 10
    modified.loc[700:, "volume"] *= 100
    a, b = prepare(original), prepare(modified)
    np.testing.assert_allclose(a.x[:700], b.x[:700], equal_nan=True)
    assert a.fit_scaler(np.arange(311, 680), True) == b.fit_scaler(np.arange(311, 680), True)
    np.testing.assert_array_equal(a.y[:690], b.y[:690])


def test_labels_use_future_endpoint_and_past_volatility():
    market = prepare(synthetic_bars())
    i = 500
    close = market.bars.close.to_numpy()
    sigma = np.std(np.log(close[i-59:i+1] / close[i-60:i]), ddof=1)
    for k, h in enumerate((3, 5, 10)):
        expected = np.log(close[i+h]/close[i]) / (sigma*np.sqrt(h))
        assert market.z[i, k] == pytest.approx(expected)
    assert (market.y[-10:, 2] == -1).all()


def test_purge_and_inner_split_have_no_crossing_targets():
    indices = np.arange(300, 1000)
    purged = before_boundary(indices, 900)
    assert purged[-1] == 889
    fit, tune = inner_split(purged)
    assert len(tune) == 252
    assert fit[-1] + 10 < tune[0]
    assert tune[-1] + 10 < 900


def test_window_ends_at_prediction_date_and_common_eligibility():
    m = prepare(synthetic_bars())
    valid = m.eligible()
    assert valid[0] == 311
    assert valid[-1] == len(m.bars) - 11
    scaler = m.fit_scaler(valid[:300], False)
    for context in (20, 60, 120, 252):
        x = m.sequences(np.array([500]), context, False, scaler)
        expected = np.clip((m.x[500, :10]-scaler["mean"])/scaler["std"], -10, 10)
        np.testing.assert_allclose(x[0, -1], expected, rtol=1e-6)


def test_causal_attention_cannot_see_next_patch():
    torch.set_num_threads(2)
    torch.manual_seed(1)
    model = CompactTransformer(10, 20).eval()
    original = torch.randn(2, 20, 10)
    changed = original.clone()
    changed[:, 8:] += 100
    with torch.no_grad():
        a, b = model.encode(original), model.encode(changed)
    torch.testing.assert_close(a[:, :2], b[:, :2], atol=1e-5, rtol=1e-5)
    assert not torch.allclose(a[:, 2:], b[:, 2:])


def test_zero_volume_is_finite_missing_feature():
    df = synthetic_bars()
    df.loc[400, "volume"] = 0
    market = prepare(df)
    assert market.x[400, 11] == 1
    assert market.x[400, 10] == 0
    assert np.isfinite(market.x[400]).all()


def test_reader_rejects_duplicates_and_timestamp_date_mismatch(tmp_path):
    df = synthetic_bars(500)
    df["date"] = df.date.dt.strftime("%Y-%m-%d")
    path = tmp_path / "bars.csv"
    df.to_csv(path, index=False)
    assert len(read_bars(path)) == 500
    df.loc[1, "timestamp"] = df.loc[0, "timestamp"]
    df.to_csv(path, index=False)
    with pytest.raises(ValueError):
        read_bars(path)


@pytest.mark.parametrize("name", ["majority", "logistic", "histboost"])
def test_baseline_probabilities_and_metrics(name):
    torch.set_num_threads(2)
    m = prepare(synthetic_bars())
    idx = m.eligible()[:100]
    scaler = m.fit_scaler(idx, False)
    x = m.sequences(idx, 20, False, scaler)
    model = Baseline(name).fit(x, m.y[idx])
    p = model.predict_proba(x)
    assert p.shape == (100, 3, 5)
    np.testing.assert_allclose(p.sum(axis=-1), 1)
    results = score(m.y[idx], p)
    assert 0 <= results["mean_macro_f1"] <= 1


@pytest.mark.parametrize("name", ["mlp", "transformer", "pretrained_transformer"])
def test_cpu_neural_training_and_probability_contract(name):
    torch.set_num_threads(2)
    m = prepare(synthetic_bars())
    idx = m.eligible()[:64]
    scaler = m.fit_scaler(idx, False)
    x = m.sequences(idx, 20, False, scaler)
    config = {"model": name, "context": 20, "volume": False}
    model, detail = fit_neural(config, x, m.y[idx], 42, epochs=1, pretrain_epochs=1)
    p = neural_probabilities(model, config, x)
    assert p.shape == (64, 3, 5)
    np.testing.assert_allclose(p.sum(axis=-1), 1, atol=1e-6)
    assert next(model.parameters()).device.type == "cpu"
    assert detail["parameters"] < 100000


def test_perfect_probabilities_have_zero_error():
    y = np.tile(np.arange(5)[:, None], (1, 3))
    p = np.eye(5)[y]
    result = score(y, p)
    for h in ("3", "5", "10"):
        assert result[h]["macro_f1"] == 1
        assert result[h]["brier_sum_over_classes"] == 0
        assert result[h]["ece_10_equal_width_bins"] == 0


def test_metrics_reject_probability_sum_failure():
    with pytest.raises(ValueError):
        score(np.zeros((2, 3), dtype=int), np.ones((2, 3, 5)))
