import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
import pytest
import torch
from cnx500 import cli
from cnx500.data import prepare, read_bars
from test_integrity import synthetic_bars


def test_official_reconciliation_preserves_original_and_flags_fallbacks(tmp_path):
    original = synthetic_bars(500)
    raw_dir = tmp_path / "data/raw"
    cache = raw_dir / "official_verification"
    cache.mkdir(parents=True)
    tv = original.drop(index=350).copy()
    tv["date"] = tv.date.dt.strftime("%Y-%m-%d")
    tv_path = raw_dir / "cnx500_daily.csv"
    tv.to_csv(tv_path, index=False)
    before = tv_path.read_bytes()
    for year in original.date.dt.year.unique():
        rows = []
        for i, row in original.loc[original.date.dt.year == year].iterrows():
            rows.append({"HistoricalDate":row.date.strftime("%d %b %Y"),
                         "OPEN":row.open,"HIGH":row.close*.98 if i == 100 else row.high,
                         "LOW":row.low,"CLOSE":row.close})
        (cache / f"{year}.json").write_text(json.dumps(rows))
    script = Path(__file__).resolve().parents[1] / "scripts/reconcile_nse.py"
    completed = subprocess.run([sys.executable, str(script), "--root", str(tmp_path)], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert tv_path.read_bytes() == before
    result = read_bars(tmp_path / "data/processed/cnx500_daily_verified.csv")
    assert len(result) == 500
    manifest = json.loads((tmp_path / "reports/data_reconciliation.json").read_text())
    assert manifest["official_only_sessions_added"] == 1
    assert manifest["official_invalid_ohlc_fallback_count"] == 1
    assert result.loc[result.date == original.date.iloc[350], "volume"].iloc[0] == 0
    assert result.loc[result.date == original.date.iloc[100], "high"].iloc[0] == pytest.approx(original.high.iloc[100])
    # Changed TV data cannot silently reuse stale official caches.
    changed = tv.copy()
    changed.loc[1,"volume"] += 1
    changed.to_csv(tv_path,index=False)
    completed = subprocess.run([sys.executable, str(script), "--root", str(tmp_path)], capture_output=True, text=True)
    assert completed.returncode != 0
    assert "snapshot changed" in completed.stderr


def test_experiment_resumes_without_refitting_and_never_opens_holdout(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    df = synthetic_bars(2000)
    path = tmp_path / "data/raw/cnx500_daily.csv"
    path.parent.mkdir(parents=True)
    df.assign(date=df.date.dt.strftime("%Y-%m-%d")).to_csv(path,index=False)
    market = prepare(read_bars(path))
    monkeypatch.setattr(cli, "FOLDS", (("2016-01-01", "2016-12-31"),))
    configs = [{"model":"majority","context":20,"volume":False},
               {"model":"mlp","context":20,"volume":False}]
    monkeypatch.setattr(cli, "candidates", lambda contexts: iter(configs))
    args = argparse.Namespace(data="data/raw/cnx500_daily.csv",epochs=1,pretrain_epochs=0,seed=42,contexts=[20],threads=2)
    cli.run_experiment(tmp_path, market, args)
    assert not (tmp_path / "reports/holdout.json").exists()
    selection = json.loads((tmp_path / "reports/selection.json").read_text())
    assert selection["test_results_seen"] is False
    def cannot_fit(*args, **kwargs):
        raise AssertionError("Resume unexpectedly refit a model")
    monkeypatch.setattr(cli,"fit_at_boundary",cannot_fit)
    cli.run_experiment(tmp_path, market, args)
    assert not (tmp_path / "reports/holdout.json").exists()
