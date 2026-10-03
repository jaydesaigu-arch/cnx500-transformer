"""Save a development-selected architecture variant after final evaluation.

This creates current research weights and predictions. It does not evaluate or
reselect on the final test, and cannot replace the frozen primary selection.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from cnx500.data import CLASSES, HORIZONS, prepare, read_bars, write_json
from cnx500.cli import fit_at_boundary, probabilities, save_model

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", type=Path, default=Path.cwd())
parser.add_argument("--model", choices=("mlp", "transformer", "pretrained_transformer"), default="pretrained_transformer")
args = parser.parse_args()
root = args.root.resolve()
if not (root / "reports/holdout.json").exists():
    raise ValueError("Record the primary final holdout before training on all matured history")
summary = json.loads((root / "reports/development_summary.json").read_text())
settings = json.loads((root / "reports/experiment_settings.json").read_text())
choice = next(r for r in summary if r["config"]["model"] == args.model)
market = prepare(read_bars(root / "data/processed/cnx500_daily_verified.csv"))
eligible = market.eligible()
torch.set_num_threads(4)
torch.set_num_interop_threads(1)
with threadpool_limits(limits=4):
    model, scaler, training = fit_at_boundary(market, choice["config"], eligible, settings["seed"],
                                              settings["epochs"], settings["pretrain_epochs"], choice["fixed_epochs"])
    training["secondary_research_variant"] = True
    training["not_final_holdout_evaluated"] = True
    save_model(root, "current_" + args.model, choice["config"], model, scaler, training)
    p = probabilities(market, choice["config"], model, scaler, np.array([len(market.bars)-1]))[0]
    result = {"symbol":"NSE:CNX500", "as_of_completed_bar":str(market.bars.date.iloc[-1].date()),
              "config":choice["config"],"secondary_research_variant":True,"not_final_holdout_evaluated":True,
              "probabilities":"uncalibrated", "horizons":{str(h):{
                  "trend":CLASSES[int(p[k].argmax())], "probabilities":{CLASSES[j]:float(p[k,j]) for j in range(5)}} for k,h in enumerate(HORIZONS)}}
    write_json(root / "reports/local" / f"latest_{args.model}_prediction.json",result)
    print(json.dumps(result,indent=2),flush=True)
