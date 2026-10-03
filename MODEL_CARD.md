# Model card: CNX 500 CPU research, v0.1.0

## Intended use

Reproducible research into five-category endpoint trend classification for the Nifty 500 index at 3, 5 and 10 observed provider sessions. Predictions use completed daily history. Evaluation compares compact neural models with simpler baselines. No capital allocation or trading execution is implemented.

## Model families

Empirical class priors; logistic regression; histogram gradient boosting; a small shared multi-horizon ANN; a two-layer causal patch Transformer; the same Transformer with continuous next-patch pretraining. Numerical daily feature rows are grouped into non-overlapping four-row patches. Position j predicts patch j+1 under a causal attention mask during pretraining. Three five-class heads are fine-tuned jointly. No third-party foundation-model weights or training corpora are used.

## Inputs and targets

Past/current OHLC transformations, 5/20/60-session returns, 20/60-session volatility, and optionally relative provider volume and its missing flag. Context sizes count feature rows; rolling feature computation extends the effective raw-price lookback. Normalizers are fitted on permitted historical fitting prefixes and standardized features are clipped to +/-10.

Class order is Strong Bearish, Bearish, Neutral, Bullish, Strong Bullish. Target z is log(C[t+h]/C[t]) divided by max(trailing 60-return sample standard deviation, 1e-6) times sqrt(h). Boundaries: z < -1; -1 <= z < -0.25; -0.25 <= z <= 0.25; 0.25 < z <= 1; z > 1. The current return ending at t is included in the volatility estimate. This predicts endpoint movement, not the path between endpoints.

## Data provenance and splits

Initial source is TradingView via TVRemix, NSE:CNX500 daily bars, requested count 5,000. Training uses a 5,011-row canonical dataset reconciled with complete official published index history over the snapshot's date range. Eleven omitted sessions were added and twelve material OHLC differences corrected. Two official rows failed OHLC shape validation and retain validated original TV prices as explicit exceptions. Provider volume is retained where available and marked unavailable for added sessions. Session timestamps are date anchors. See reports/data_reconciliation.json and reports/data_audit.json for source policies and hashes. Session order matches published index history, not an independent reconstruction of all exchange rules. Data is excluded from Git.

Three expanding development folds: 2016-2018, 2019-2021, 2022 through September 2024. Purge fitting origins whose 10-session targets cross the evaluation boundary. Inner epoch tuning uses the last 252 eligible origins of each permitted prefix with the same purge. Final holdout starts October 2024. Configuration and median development epoch count are frozen before final model metrics are computed. Final test models have static weights. Current research models are separate refits performed after test results are archived.

An initial audit computed aggregate class counts over the complete dataset; no thresholds or configurations were selected using those aggregates. This is disclosed rather than claiming that every property of the final-test data was unseen.

Independent NSE daily archive spot checks on four available dates matched OHLC within 0.011 index points. Provider volume did not exactly match NSE on these dates and differed by roughly 4% on the latest checked date. Volume is retained as a provider-specific experimental feature rather than substituted or represented as an identical official series. See reports/source_verification_summary.json; detailed date-level differences remain local. Two older archive URLs returned no data; this is not proof that the underlying sessions or data are absent.

## Evaluation and known limits

See REPORT.md and reports/holdout.json for measured results. Five-class probabilities are raw and uncalibrated. Calibration metrics are diagnostics; no calibrated-confidence claim is made. Initial configuration search uses one seed. The index provides only a small number of dependent observations. Forecast horizons overlap, regimes change, and there is no established prospective trading evidence, cost-adjusted profitability, formal statistical significance or worst-case guarantee.

## Distribution

Source is MIT licensed. Raw market data, date-level derived outputs, baseline pickle files and neural weights are local-only artifacts until applicable distribution rights are confirmed. Documentation inclusion never overrides content exclusions. Credentials and secrets must never be published. Only deserialize trusted local artifacts.
