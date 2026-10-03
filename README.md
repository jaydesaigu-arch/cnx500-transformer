# CNX 500: compact ANN and Transformer research

A CPU research pipeline for the **Nifty 500 index (`NSE:CNX500`)**. It classifies endpoint movement over the next **3, 5 and 10 observed trading sessions** as Strong Bearish, Bearish, Neutral, Bullish or Strong Bullish.

The project compares simple baselines, a small ANN, a compact causal Transformer and the same Transformer with **continuous next-patch pretraining**. This is LLM-inspired self-supervised training on numerical time-series features; it is not a large pretrained text model. The initial Transformer has approximately 21,000 parameters, depending on context and feature count.

See [the research report](REPORT.md), [the experiment specification](EXPERIMENT_SPEC.md) and [the model card](MODEL_CARD.md). Research results are reported even when a simpler baseline performs better. No trading execution or profitability claim is included.

## Install

Tested with Python 3.12 on Windows, an Intel i5-1235U and approximately 16 GB RAM. CPU-only PyTorch is sufficient. A GPU is not required.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
.\.venv\Scripts\python.exe -m pytest -q
```

On Linux/macOS, substitute `.venv/bin/python` for the Windows executable. `requirements-cpu.txt` pins the actual initial experiment environment. Optional dependencies may contain platform-specific package behavior; remote CI is reported separately from local test success.

## Data

Supply the original authorized TradingView CSV at `data/raw/cnx500_daily.csv` with columns:

```text
timestamp,date,open,high,low,close,volume
```

`timestamp` is UTC Unix seconds. `date` must match its session date in Asia/Kolkata, formatted YYYY-MM-DD. Rows must be unique, ascending, completed daily candles. Volume must be nonnegative; zero or unavailable volume is represented as unavailable input. Do not include a still-forming candle. No candles are invented or forward-filled.

Initial acquisition used the connected TradingView TVRemix MCP:

```json
{"symbol":"NSE:CNX500","interval":"1D","count":5000,"summary":false}
```

Its observed limit was 5,000 bars. This invocation returned history from July 2006 through October 2026. Exact reproduction requires the same authorized source snapshot and dataset hash; a later download changes the experiment. Do not assume the provider's index volume is a tradable share count or that the observed sessions have been independently reconciled with exchange calendars.

To import a saved connector response:

```powershell
.\.venv\Scripts\python.exe scripts/import_tv_response.py data/raw/tradingview_response.json
```

Raw data, date-level derived records, local forecasts and trained weights are excluded from Git. The code license does not grant rights to third-party market data or derived artifacts. Check the applicable provider/exchange terms before redistributing data or weights. Secrets are always excluded.

Before training, reconcile it against official published index history:

```powershell
.\.venv\Scripts\python.exe scripts/reconcile_nse.py
```

This reads public official history in chunks no longer than one year and saves local source responses. It creates `data/processed/cnx500_daily_verified.csv`, the default training input. The original TV snapshot is never overwritten. Initial reconciliation added 11 official sessions missing from TV and corrected 12 material OHLC discrepancies. Two official historical rows failed OHLC shape validation; the canonical file retains already validated TV prices on those explicitly flagged dates. TV volume is retained on shared dates and is unavailable on added sessions. Canonical timestamps anchor the daily session date, not historical special-session opening times. The manifest records source hashes and every exception.

If intentionally experimenting on another CSV, pass `--data relative/path.csv`; its provenance and session rules are your responsibility. Never silently combine an old official cache with a changed TV snapshot; use `scripts/reconcile_nse.py --refresh` when rebuilding from new source data.

## Run

From the project root:

```powershell
.\.venv\Scripts\python.exe -m cnx500.cli audit
.\.venv\Scripts\python.exe -m cnx500.cli benchmark
.\.venv\Scripts\python.exe -m cnx500.cli experiment --threads 4 --epochs 20 --pretrain-epochs 4
.\.venv\Scripts\python.exe -m cnx500.cli finalize
.\.venv\Scripts\python.exe -m cnx500.cli report
.\.venv\Scripts\python.exe -m cnx500.cli predict
```

`experiment` searches 41 configurations across three expanding historical folds. Contexts are 20, 60, 120 and 252 daily feature rows; rolling features need up to 60 preceding observations in addition. Price-only and provider-volume variants use identical eligible dates. Checkpoints permit resuming an interrupted experiment with identical source, data and settings. Use a separate project/results directory for a changed experiment.

`finalize` opens the final holdout once, using configurations already frozen in `reports/selection.json`. It records static-weight test results, then creates separate models refit on all matured history. Current model weights therefore differ from the evaluated holdout weights. Re-running `finalize` on an already opened test is rejected. Treat repeated experimentation against a known test as exploratory, not fresh validation.

`predict` uses the latest complete bar and the locally fitted current baseline and neural models. Outputs are research probabilities, not calibrated confidence. The forecasts do not assert a monotonically rising or falling price path. Neutral is a movement category, not a low-confidence signal.

Additional saved research variants can be built **after** the primary holdout is recorded:

```powershell
.\.venv\Scripts\python.exe scripts/train_research_variant.py --model pretrained_transformer
.\.venv\Scripts\python.exe scripts/train_research_variant.py --model mlp
.\.venv\Scripts\python.exe scripts/check_seed_stability.py
```

These variants use the best development configuration within their architecture family. They are refit on all matured history and have not separately been evaluated on the final holdout. They cannot replace the frozen primary selection. Additional seed checks use development data only and do not select a replacement model.

Only load trusted local baseline pickle files; pickle deserialization can execute code. Neural weights use PyTorch's weights-only loader.

## Evaluation

All training labels mature strictly before their evaluation boundary. Inner tuning uses a purged chronological split. Feature scaling and pretraining use only permitted fitting history. Three shared classification heads forecast 3/5/10 sessions. The target divides endpoint log return by trailing volatility times the square root of the horizon, with fixed five-class boundaries.

Metrics include five-class macro-F1, balanced accuracy, accuracy, log loss, summed multiclass Brier score, ten-bin expected calibration error, ordinal error and extreme reversals. The report includes confusion matrices and the machine-readable holdout file includes yearly results. Overlapping targets are dependent; single-seed model comparisons are preliminary.

The initial data-quality audit computed full-history aggregate class counts before model selection. Thresholds were already fixed and these counts did not guide selection; date-level holdout outcomes and holdout model scores were not used to select configurations. Subsequent audit class counts are limited to development data.

## Layout

```text
src/cnx500/           features, models, scoring, CLI and reporting
tests/               synthetic leakage and numerical integrity tests
scripts/             source-response importer
reports/             aggregate audit, selection and evaluation results
reports/local/       excluded caches and date-level predictions
data/raw/            excluded source data
models/              excluded local research weights
```

Research basis: [PatchTST](https://arxiv.org/abs/2211.14730), [original Chronos](https://www.amazon.science/publications/chronos-learning-the-language-of-time-series), [chronological validation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html), [NSE Indices history](https://www.niftyindices.com/reports/historical-data) and [NSE data policy](https://www.nseindia.com/static/market-data/nse-data-policy). General forecasting research does not establish stock-market predictive performance.
