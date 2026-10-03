"""Diagnostic development-only seeds; never modify selection or open holdout."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from cnx500.data import FOLDS, HORIZONS, before_boundary, prepare, read_bars, write_json
from cnx500.cli import fit_at_boundary, probabilities
from cnx500.metrics import score

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", type=Path, default=Path.cwd())
parser.add_argument("--seeds", nargs="+", type=int, default=[7, 2026])
args = parser.parse_args()
root = args.root.resolve()
selection = json.loads((root / "reports/selection.json").read_text())
settings = json.loads((root / "reports/experiment_settings.json").read_text())
choice = selection["neural"]
config = choice["config"]
market = prepare(read_bars(root / "data/processed/cnx500_daily_verified.csv"))
eligible = market.eligible()
torch.set_num_threads(4)
torch.set_num_interop_threads(1)
records = []
with threadpool_limits(limits=4):
    for seed in args.seeds:
        for number, (start, end) in enumerate(FOLDS):
            boundary = int(np.searchsorted(market.dates, np.datetime64(start)))
            prefix = before_boundary(eligible, boundary)
            indices = eligible[(market.dates[eligible] >= np.datetime64(start)) & (market.dates[eligible + 10] <= np.datetime64(end))]
            model, scaler, training = fit_at_boundary(market, config, prefix, seed, settings["epochs"],
                                                      settings["pretrain_epochs"], choice["fixed_epochs"])
            p = probabilities(market, config, model, scaler, indices)
            record = {"seed":seed,"fold":number,"metrics":score(market.y[indices],p),"training":training}
            records.append(record)
            print(f"Seed {seed}, fold {number+1}: macro-F1={record['metrics']['mean_macro_f1']:.4f}",flush=True)
summary = {"config":config,"fixed_epochs":choice["fixed_epochs"],"diagnostic_only":True,
           "no_reselection":True,"no_holdout_used":True,
           "seed_mean_macro_f1":{str(seed):float(np.mean([r["metrics"]["mean_macro_f1"] for r in records if r["seed"]==seed])) for seed in args.seeds},
           "records":records}
write_json(root / "reports/seed_stability.json",summary)
