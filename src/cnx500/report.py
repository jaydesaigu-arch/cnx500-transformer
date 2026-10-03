"""Generate aggregate research documentation and exportable figures."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .data import HORIZONS, CLASSES


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def label(config):
    return f"{config['model']} / {config['context']} days / {'price + volume' if config['volume'] else 'price'}"


def create_report(root):
    reports = root / "reports"
    audit = read(reports / "data_audit.json")
    development = read(reports / "development_summary.json")
    selection = read(reports / "selection.json")
    final = read(reports / "holdout.json")
    settings = read(reports / "experiment_settings.json")
    benchmark = read(reports / "cpu_benchmark.json")
    figures = reports / "figures"
    figures.mkdir(exist_ok=True)
    rows = ["# CNX 500 research results", "",
            "This is an initial CPU research experiment. Class probabilities are uncalibrated. No trading strategy or profitability claim is evaluated.", "",
            "## Data and target", "",
            f"The reconciled NSE:CNX500 dataset contains {audit['bars']:,} daily observations, {audit['first_date']} through {audit['last_date']}. Provider volume is unavailable on {audit['volume_unavailable_bars']} observations. OHLC relationships and unique ascending session dates passed checks. Session dates align with official published index history; provider index-volume aggregation semantics remain unverified.", "",
            "The five ordered labels describe the future endpoint log return divided by trailing 60-session daily volatility and square root of the horizon. Fixed thresholds are -1, -0.25, +0.25 and +1. See EXPERIMENT_SPEC.md for exact boundary inclusion.", "",
            "Context length counts daily feature rows. Rolling features require up to 60 preceding observations beyond that context. Horizons count observed provider sessions; missing exchange sessions could change their interpretation. All configurations use the same eligible origins and all three labels must be mature.", "",
            "Full official historical comparison found 11 sessions absent from the original 5,000-row TV snapshot and 14 material shared-date OHLC discrepancies. The canonical dataset adds the 11 official sessions and uses valid official OHLC, correcting 12 discrepant dates. Two official records fail OHLC shape validation and retain the already validated TV values, explicitly flagged. See reports/data_reconciliation.json. Separately, four daily archive spot checks found provider-volume differences, including about 4% on the latest checked date. Session reconciliation is against published index history, not an independent reconstruction of all exchange rules.", "",
            "## Development comparison", "",
            "Configuration selection used three expanding historical folds ending before October 2024. Neural epoch counts were selected on purged chronological inner tuning periods and then refit on each permitted training prefix. No final-test model scores were used to select configurations. An initial data audit computed full-history aggregate class counts; thresholds were already fixed and those counts were not used for selection.", "",
            "| Model / context / features | Mean development macro-F1 | Mean log loss |", "|---|---:|---:|"]
    for entry in development[:12]:
        rows.append(f"| {label(entry['config'])} | {entry['mean_development_macro_f1']:.4f} | {entry['mean_development_log_loss']:.4f} |")
    rows.extend(["", "Best configuration within each family (development only):", "",
                 "| Family | Selected context / features | Development macro-F1 |", "|---|---|---:|"])
    for family in ("majority", "logistic", "histboost", "mlp", "transformer", "pretrained_transformer"):
        entry = next(r for r in development if r["config"]["model"] == family)
        rows.append(f"| {family} | {entry['config']['context']} / {'price + volume' if entry['config']['volume'] else 'price'} | {entry['mean_development_macro_f1']:.4f} |")
    supervised_best = next(r for r in development if r["config"]["model"] == "transformer")
    pretrained_best = next(r for r in development if r["config"]["model"] == "pretrained_transformer")
    rows.extend(["", f"The best next-patch-pretrained Transformer scored {pretrained_best['mean_development_macro_f1']:.4f}, versus {supervised_best['mean_development_macro_f1']:.4f} for the best supervised Transformer. This bounded search did not establish a benefit from pretraining.", ""])
    rows.extend(["", f"Selected baseline: **{label(selection['baseline']['config'])}**.", "",
                 f"Selected neural model: **{label(selection['neural']['config'])}**.", "",
                 "## Final chronological holdout", "",
                 f"Final evaluation covers {final['test_samples']} common prediction origins starting {final['test_start']} and ending {final['test_end_origin']}, with fixed model weights throughout. Every fitting target ends before the first test origin. The latest ten origins are excluded from joint scoring because their 10-session labels are not yet available.", "",
                 "| Horizon | Baseline macro-F1 | Neural macro-F1 | Baseline accuracy | Neural accuracy | Baseline log loss | Neural log loss |", "|---|---:|---:|---:|---:|---:|---:|"])
    baseline, neural = (final["roles"][role]["metrics"] for role in ("baseline", "neural"))
    for h in HORIZONS:
        b, n = baseline[str(h)], neural[str(h)]
        rows.append(f"| {h} sessions | {b['macro_f1']:.4f} | {n['macro_f1']:.4f} | {b['accuracy']:.1%} | {n['accuracy']:.1%} | {b['log_loss']:.4f} | {n['log_loss']:.4f} |")
    delta = neural["mean_macro_f1"] - baseline["mean_macro_f1"]
    comparison = "higher" if delta > 0 else "lower"
    rows.extend(["", f"The selected neural model's mean holdout macro-F1 is **{abs(delta):.4f} {comparison}** than the selected baseline. This observed difference is not a statistical significance or profitability claim.", "",
                 "The confusion matrices show substantial class imbalance in the model's predictions: some heads almost never predict Bearish or Neutral. The 10-session head predicted Neutral zero times in this holdout. These are material weaknesses of this initial model; a category such as Strong Bullish describes the target's movement size and must not be read as strong predictive confidence.", "",
                 "![Holdout comparison](reports/figures/holdout_comparison.png)", "",
                 "![Neural confusion matrices](reports/figures/neural_confusion.png)", "",
                 "## Additional development seed diagnostics", ""])
    seed_path = reports / "seed_stability.json"
    if seed_path.exists():
        seed = read(seed_path)
        rows.extend(["The selected neural configuration was checked with additional initialization seeds using a fixed epoch count selected from the original development run. These diagnostics did not change configuration selection, current weights or final-test fitting. They are not a fresh hyperparameter search and are not directly identical to per-fold early-stopping scores.", "",
                     "| Seed | Mean development macro-F1 |", "|---|---:|"])
        for key, value in seed["seed_mean_macro_f1"].items():
            rows.append(f"| {key} | {value:.4f} |")
    else:
        rows.append("Additional seed diagnostics have not been recorded.")
    rows.extend(["",
                 "## Reproduction and artifact provenance", "",
                 f"Dataset SHA-256: `{audit['csv_sha256']}`.", "",
                 f"Computational source SHA-256: `{settings['code_sha256']}`.", "",
                 f"The initial CPU timing benchmark used 1,000 examples, a 60-row price-only context, one pretraining epoch and two supervised epochs: {benchmark['seconds']:.2f} seconds. It is a small benchmark, not a full-run runtime guarantee.", "",
                 f"Initial neural search used seed {settings['seed']}, {len(development)} total configurations, four CPU threads, up to {settings['epochs']} supervised epochs with patience four, and {settings['pretrain_epochs']} next-patch pretraining epochs where applicable. Contexts were selected from 20, 60, 120 and 252; no claim of globally optimal context is made. The baseline fixed_epochs field is an unused orchestration budget, not a boosting-iteration count; boosting uses 80 iterations as specified in models.py.", "",
                 "After recording this holdout, separate current research models were refit on all matured history. Current predictions are therefore produced by different weights from those evaluated above. Current predictions, date-level evaluation rows, weights and raw data are local-only artifacts excluded from Git.", "",
                 "## Limitations", "",
                 "One index is a small and dependent dataset. Evaluation windows and targets overlap. Historical market regimes change. Thresholds and model search are research choices. The initial search uses a single seed; subsequent development seed checks are diagnostic only. No economic backtest, execution-cost analysis, prospective paper-trading evidence, formal calibration or confidence intervals have been established. Baseline superiority or weak performance is a valid research outcome.", "",
                 "The aggregate report does not confer rights to redistribute underlying market data or trained weights. Raw source data and date-level derived records stay excluded from publication.", ""])
    (root / "REPORT.md").write_text("\n".join(rows), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    x = np.arange(3)
    for ax, metric, title in zip(axes, ("macro_f1", "log_loss"), ("Macro-F1 (higher is better)", "Log loss (lower is better)")):
        ax.bar(x-.18, [baseline[str(h)][metric] for h in HORIZONS], .36, label="Selected baseline", color="#607d8b")
        ax.bar(x+.18, [neural[str(h)][metric] for h in HORIZONS], .36, label="Selected neural model", color="#1565c0")
        ax.set_xticks(x, [f"{h} sessions" for h in HORIZONS])
        ax.set_title(title)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="outside lower center", ncol=2, fontsize=9, frameon=False)
    fig.suptitle("CNX 500: static chronological holdout")
    fig.savefig(figures / "holdout_comparison.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    short = ("SBear", "Bear", "Neutral", "Bull", "SBull")
    for k, h in enumerate(HORIZONS):
        matrix = np.array(neural[str(h)]["confusion_matrix_true_rows_predicted_columns"])
        axes[k].imshow(matrix, cmap="Blues")
        for i in range(5):
            for j in range(5):
                axes[k].text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=8,
                             color="white" if matrix[i, j] > matrix.max()/2 else "black")
        axes[k].set_xticks(range(5), short, rotation=45)
        axes[k].set_yticks(range(5), short)
        axes[k].set_title(f"{h}-session horizon")
        axes[k].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")
    fig.suptitle("Selected neural model: true rows, predicted columns")
    fig.savefig(figures / "neural_confusion.png", dpi=180)
    plt.close(fig)
    print(str(root / "REPORT.md"))
