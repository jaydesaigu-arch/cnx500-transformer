"""Command line workflow; every fitting boundary is explicit and saved."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import time
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from .data import CLASSES, CONTEXTS, FEATURES, FOLDS, HORIZONS, TEST_START, audit, before_boundary, inner_split, prepare, read_bars, write_json
from .metrics import score
from .models import Baseline, fit_neural, make_neural, neural_probabilities

BASELINES = ("majority", "logistic", "histboost")
NEURAL = ("mlp", "transformer", "pretrained_transformer")


def identifier(config):
    return f"{config['model']}_c{config['context']}_{'volume' if config['volume'] else 'price'}"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fingerprint(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def candidates(contexts=CONTEXTS):
    yield {"model": "majority", "context": 20, "volume": False}
    for model in ("logistic", "histboost") + NEURAL:
        for context in contexts:
            for volume in (False, True):
                yield {"model": model, "context": context, "volume": volume}


def fit_at_boundary(market, config, prefix, seed, max_epochs, pretrain_epochs, fixed_epochs=None):
    details = {}
    if config["model"] in NEURAL and fixed_epochs is None:
        fitting, tuning = inner_split(prefix)
        inner_scaler = market.fit_scaler(fitting, config["volume"])
        xf = market.sequences(fitting, config["context"], config["volume"], inner_scaler)
        xt = market.sequences(tuning, config["context"], config["volume"], inner_scaler)
        _, details["inner_fit"] = fit_neural(config, xf, market.y[fitting], seed, max_epochs,
                                             tuning=(xt, market.y[tuning]), pretrain_epochs=pretrain_epochs)
        epochs = details["inner_fit"]["best_epoch"]
        details["inner_fit_end_date"] = str(market.bars.date.iloc[fitting[-1]].date())
        details["inner_tuning_start_date"] = str(market.bars.date.iloc[tuning[0]].date())
        assert fitting[-1] + max(HORIZONS) < tuning[0]
    else:
        epochs = fixed_epochs if fixed_epochs is not None else max_epochs
    scaler = market.fit_scaler(prefix, config["volume"])
    x = market.sequences(prefix, config["context"], config["volume"], scaler)
    y = market.y[prefix]
    if config["model"] in BASELINES:
        model = Baseline(config["model"], seed).fit(x, y)
    else:
        model, details["full_refit"] = fit_neural(config, x, y, seed, epochs,
                                                  pretrain_epochs=pretrain_epochs)
    details["fixed_epochs"] = int(epochs)
    details["fitting_origins"] = len(prefix)
    details["last_fitting_origin"] = str(market.bars.date.iloc[prefix[-1]].date())
    details["last_supervised_target_date"] = str(market.bars.date.iloc[prefix[-1] + 10].date())
    return model, scaler, details


def probabilities(market, config, model, scaler, indices):
    x = market.sequences(indices, config["context"], config["volume"], scaler)
    return model.predict_proba(x) if config["model"] in BASELINES else neural_probabilities(model, config, x)


def save_model(root, role, config, model, scaler, training):
    directory = root / "models" / role
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "metadata.json", {"config": config, "scaler": scaler, "training": training,
                                             "classes": CLASSES, "horizons": HORIZONS, "model_version": "0.1.0",
                                             "probabilities": "uncalibrated"})
    if config["model"] in BASELINES:
        with (directory / "baseline.pkl").open("wb") as f:
            pickle.dump(model, f)
    else:
        torch.save(model.state_dict(), directory / "weights.pt")


def load_model(root, role):
    directory = root / "models" / role
    meta = read_json(directory / "metadata.json")
    config = meta["config"]
    if config["model"] in BASELINES:
        # Load only trusted, locally created artifacts; pickle is executable.
        with (directory / "baseline.pkl").open("rb") as f:
            model = pickle.load(f)
    else:
        shape = (1, config["context"], 12 if config["volume"] else 10)
        model = make_neural(config, np.zeros(shape, dtype=np.float32))
        model.load_state_dict(torch.load(directory / "weights.pt", map_location="cpu", weights_only=True))
    return model, meta


def run_experiment(root, market, args):
    audit_info = audit(root / args.data, market)
    code_digest = hashlib.sha256()
    for source in ("data.py", "models.py", "metrics.py", "cli.py"):
        code_digest.update((Path(__file__).parent / source).read_bytes())
    settings = {"epochs": args.epochs, "pretrain_epochs": args.pretrain_epochs, "seed": args.seed,
                "contexts": list(args.contexts), "folds": FOLDS, "threads": args.threads,
                "data_sha256": audit_info["csv_sha256"], "code_version": "0.1.0", "code_sha256": code_digest.hexdigest()}
    run_id = fingerprint(settings)
    settings["run_id"] = run_id
    settings_path = root / "reports" / "experiment_settings.json"
    if settings_path.exists() and read_json(settings_path) != json.loads(json.dumps(settings)):
        raise ValueError("Existing experiment has different settings. Use a separate output directory.")
    write_json(settings_path, settings)
    write_json(root / "reports" / "data_audit.json", audit_info)
    eligible = market.eligible()
    records = []
    for config in candidates(args.contexts):
        key = identifier(config)
        for fold_number, (start, end) in enumerate(FOLDS):
            cache = root / "reports" / "local" / "development" / f"{key}_fold{fold_number}.json"
            if cache.exists():
                record = read_json(cache)
                if record["run_id"] != run_id:
                    raise ValueError("Stale experiment cache")
                records.append(record)
                continue
            eval_idx = eligible[(market.dates[eligible] >= np.datetime64(start)) &
                                (market.dates[eligible + 10] <= np.datetime64(end))]
            if not len(eval_idx):
                raise ValueError("Empty development fold")
            boundary = int(np.searchsorted(market.dates, np.datetime64(start)))
            prefix = before_boundary(eligible, boundary)
            assert prefix[-1] + 10 < eval_idx[0]
            started = time.perf_counter()
            model, scaler, training = fit_at_boundary(market, config, prefix, args.seed,
                                                      args.epochs, args.pretrain_epochs)
            p = probabilities(market, config, model, scaler, eval_idx)
            record = {"run_id": run_id, "config": config, "fold": fold_number,
                      "evaluation_start": str(market.bars.date.iloc[eval_idx[0]].date()),
                      "evaluation_end": str(market.bars.date.iloc[eval_idx[-1]].date()),
                      "metrics": score(market.y[eval_idx], p), "training": training,
                      "seconds": time.perf_counter() - started}
            write_json(cache, record)
            records.append(record)
            print(f"{key} fold {fold_number+1}/3: macro-F1={record['metrics']['mean_macro_f1']:.4f}, {record['seconds']:.1f}s", flush=True)
    summary = []
    for config in candidates(args.contexts):
        group = [r for r in records if r["config"] == config]
        summary.append({"config": config, "mean_development_macro_f1": float(np.mean([r["metrics"]["mean_macro_f1"] for r in group])),
                        "mean_development_log_loss": float(np.mean([r["metrics"][str(h)]["log_loss"] for r in group for h in HORIZONS])),
                        "fold_macro_f1": [r["metrics"]["mean_macro_f1"] for r in group],
                        "fixed_epochs": max(1, int(np.median([r["training"]["fixed_epochs"] for r in group]))),
                        "total_seconds": sum(r["seconds"] for r in group)})
    # Deterministic ties: lower log loss, shorter context, price-only, model name.
    summary.sort(key=lambda r: (-r["mean_development_macro_f1"], r["mean_development_log_loss"],
                                 r["config"]["context"], r["config"]["volume"], r["config"]["model"]))
    write_json(root / "reports" / "development_summary.json", summary)
    selection = {"run_id": run_id, "data_sha256": audit_info["csv_sha256"],
                 "selection_rule": "development mean macro-F1 equally across 3 folds and 3 horizons; deterministic ties",
                 "baseline": next(r for r in summary if r["config"]["model"] in BASELINES),
                 "neural": next(r for r in summary if r["config"]["model"] in NEURAL),
                 "test_start": TEST_START, "test_results_seen": False}
    write_json(root / "reports" / "selection.json", selection)
    print("Configuration selection frozen. No final test metrics computed.", flush=True)


def write_prediction_rows(root, role, market, indices, p):
    import pandas as pd
    rows = []
    for i, origin in enumerate(indices):
        for k, h in enumerate(HORIZONS):
            row = {"date": str(market.bars.date.iloc[origin].date()), "horizon": h,
                   "actual_class": CLASSES[market.y[origin, k]], "predicted_class": CLASSES[int(p[i, k].argmax())]}
            row.update({CLASSES[j]: float(p[i, k, j]) for j in range(5)})
            rows.append(row)
    path = root / "reports" / "local" / f"holdout_{role}_predictions.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def finalize(root, market, args):
    selection = read_json(root / "reports" / "selection.json")
    settings = read_json(root / "reports" / "experiment_settings.json")
    if audit(root / args.data, market)["csv_sha256"] != selection["data_sha256"]:
        raise ValueError("Dataset changed after selection freeze")
    final_path = root / "reports" / "holdout.json"
    if final_path.exists():
        raise ValueError("Holdout already opened. Use saved results; do not repeatedly tune on the final test.")
    eligible = market.eligible()
    boundary = int(np.searchsorted(market.dates, np.datetime64(TEST_START)))
    fitting = before_boundary(eligible, boundary)
    testing = eligible[eligible >= boundary]
    results = {"selection_sha256": fingerprint(selection), "test_start": TEST_START,
               "test_end_origin": str(market.bars.date.iloc[testing[-1]].date()),
               "test_samples": len(testing), "static_weights": True, "roles": {}}
    for role in ("baseline", "neural"):
        choice = selection[role]
        config = choice["config"]
        model, scaler, training = fit_at_boundary(market, config, fitting, settings["seed"],
                                                  settings["epochs"], settings["pretrain_epochs"], choice["fixed_epochs"])
        assert fitting[-1] + 10 < testing[0]
        p = probabilities(market, config, model, scaler, testing)
        yearly = {}
        years = market.bars.date.iloc[testing].dt.year.to_numpy()
        for year in np.unique(years):
            mask = years == year
            yearly[str(year)] = score(market.y[testing[mask]], p[mask])
        results["roles"][role] = {"config": config, "training": training,
                                   "metrics": score(market.y[testing], p), "by_year": yearly}
        save_model(root, f"holdout_{role}", config, model, scaler, training)
        write_prediction_rows(root, role, market, testing, p)
    write_json(final_path, results)
    print("Final holdout recorded; now refitting selected configurations on matured history.", flush=True)
    for role in ("baseline", "neural"):
        choice = selection[role]
        config = choice["config"]
        model, scaler, training = fit_at_boundary(market, config, eligible, settings["seed"],
                                                  settings["epochs"], settings["pretrain_epochs"], choice["fixed_epochs"])
        save_model(root, f"current_{role}", config, model, scaler, training)
    predict(root, market)


def predict(root, market):
    idx = np.array([len(market.bars) - 1])
    result = {"symbol": "NSE:CNX500", "as_of_completed_bar": str(market.bars.date.iloc[-1].date()),
              "model_version": "0.1.0", "probabilities": "uncalibrated research outputs", "roles": {}}
    for role in ("baseline", "neural"):
        model, meta = load_model(root, f"current_{role}")
        p = probabilities(market, meta["config"], model, meta["scaler"], idx)[0]
        result["roles"][role] = {"config": meta["config"], "horizons": {
            str(h): {"trend": CLASSES[int(p[k].argmax())], "probabilities": {CLASSES[j]: float(p[k, j]) for j in range(5)}}
            for k, h in enumerate(HORIZONS)}}
    write_json(root / "reports" / "local" / "latest_prediction.json", result)
    print(json.dumps(result, indent=2), flush=True)


def benchmark(root, market, args):
    eligible = market.eligible()
    config = {"model": "pretrained_transformer", "context": 60, "volume": False}
    subset = eligible[:1000]
    scaler = market.fit_scaler(subset, False)
    x = market.sequences(subset, 60, False, scaler)
    started = time.perf_counter()
    _, details = fit_neural(config, x, market.y[subset], args.seed, epochs=2, pretrain_epochs=1)
    result = {"seconds": time.perf_counter() - started, "device": "cpu", "torch_threads": torch.get_num_threads(),
              "configuration": config, "details": details}
    write_json(root / "reports" / "cpu_benchmark.json", result)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit", "benchmark", "experiment", "finalize", "predict", "report"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--data", default="data/processed/cnx500_daily_verified.csv")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--pretrain-epochs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--contexts", nargs="+", type=int, default=list(CONTEXTS), choices=CONTEXTS)
    args = parser.parse_args()
    if args.threads < 1 or args.epochs < 1 or args.pretrain_epochs < 0:
        parser.error("Positive thread/epoch counts and nonnegative pretraining epochs required")
    root = args.root.resolve()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    with threadpool_limits(limits=args.threads):
        if args.command == "report":
            from .report import create_report
            create_report(root)
            return
        market = prepare(read_bars(root / args.data))
        if args.command == "audit":
            result = audit(root / args.data, market)
            write_json(root / "reports" / "data_audit.json", result)
            print(json.dumps(result, indent=2))
        elif args.command == "benchmark":
            benchmark(root, market, args)
        elif args.command == "experiment":
            run_experiment(root, market, args)
        elif args.command == "finalize":
            finalize(root, market, args)
        elif args.command == "predict":
            predict(root, market)


if __name__ == "__main__":
    main()
