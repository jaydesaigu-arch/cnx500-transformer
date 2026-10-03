"""Causal features, immutable-data checks, labels and chronological splits."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

HORIZONS = (3, 5, 10)
CLASSES = ("Strong Bearish", "Bearish", "Neutral", "Bullish", "Strong Bullish")
CONTEXTS = (20, 60, 120, 252)
FEATURES = ("return_1", "overnight_gap", "candle_body", "candle_range", "close_location",
            "return_5", "return_20", "return_60", "volatility_20", "volatility_60",
            "relative_volume", "volume_missing")
FOLDS = (("2016-01-01", "2018-12-31"), ("2019-01-01", "2021-12-31"),
         ("2022-01-01", "2024-09-30"))
TEST_START = "2024-10-01"


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8")


def classify(z):
    z = np.asarray(z, dtype=float)
    y = np.select([z < -1, z < -.25, z <= .25, z <= 1], [0, 1, 2, 3], default=4)
    return np.where(np.isfinite(z), y, -1).astype(np.int64)


def read_bars(path):
    df = pd.read_csv(path)
    required = {"timestamp", "date", "open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns: {required - set(df.columns)}")
    if len(df) < 400:
        raise ValueError("At least 400 daily bars required")
    dates = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_convert("Asia/Kolkata")
    derived = dates.dt.strftime("%Y-%m-%d")
    if not derived.equals(df["date"].astype(str)):
        raise ValueError("Session dates disagree with timestamps in Asia/Kolkata")
    if df["timestamp"].duplicated().any() or df["date"].duplicated().any():
        raise ValueError("Duplicate daily bars")
    if not df["timestamp"].is_monotonic_increasing:
        raise ValueError("Bars must arrive in ascending timestamp order")
    prices = df[["open", "high", "low", "close"]].to_numpy(float)
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("Non-finite or non-positive prices")
    if ((df.low > df[["open", "close"]].min(axis=1)) |
        (df.high < df[["open", "close"]].max(axis=1)) | (df.low > df.high)).any():
        raise ValueError("Invalid OHLC relationships")
    if (df.volume < 0).any():
        raise ValueError("Negative volume")
    df["date"] = pd.to_datetime(df["date"])
    return df


@dataclass
class MarketData:
    bars: pd.DataFrame
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    sigma: np.ndarray

    @property
    def dates(self):
        return self.bars.date.to_numpy()

    def eligible(self, max_context=252):
        finite = np.isfinite(self.x).all(axis=1)
        valid_window = pd.Series(finite.astype(int)).rolling(max_context).sum().to_numpy() == max_context
        return np.flatnonzero(valid_window & (self.y >= 0).all(axis=1))

    def fit_scaler(self, fitting_indices, volume):
        if len(fitting_indices) == 0:
            raise ValueError("Empty fitting set")
        f = self.x[:int(max(fitting_indices)) + 1, :12 if volume else 10]
        f = f[np.isfinite(f).all(axis=1)]
        return {"mean": f.mean(axis=0).tolist(), "std": np.maximum(f.std(axis=0), 1e-6).tolist()}

    def sequences(self, indices, context, volume, scaler):
        if context % 4:
            raise ValueError("Context must be divisible by patch size 4")
        f = self.x[:, :12 if volume else 10]
        f = np.clip((f - np.asarray(scaler["mean"])) / np.asarray(scaler["std"]), -10, 10)
        indices = np.asarray(indices, dtype=int)
        if (indices < context - 1).any():
            raise ValueError("Not enough historical input")
        x = f[indices[:, None] - np.arange(context - 1, -1, -1)[None, :]]
        if not np.isfinite(x).all():
            raise ValueError("Non-finite input window")
        return x.astype(np.float32)


def prepare(df):
    c, o, h, l = (df[k].astype(float) for k in ("close", "open", "high", "low"))
    r = np.log(c / c.shift(1))
    sigma = r.rolling(60, min_periods=60).std(ddof=1).clip(lower=1e-6)
    v = df.volume.where(df.volume > 0)
    vref = v.rolling(20, min_periods=5).median()
    relative_v = np.log(v / vref).fillna(0)
    features = pd.DataFrame({
        "return_1": r, "overnight_gap": np.log(o / c.shift(1)),
        "candle_body": np.log(c / o), "candle_range": np.log(h / l),
        "close_location": ((c - l) / (h - l).replace(0, np.nan) - .5).fillna(0),
        "return_5": np.log(c / c.shift(5)), "return_20": np.log(c / c.shift(20)),
        "return_60": np.log(c / c.shift(60)), "volatility_20": r.rolling(20).std(ddof=1),
        "volatility_60": sigma, "relative_volume": relative_v,
        "volume_missing": v.isna().astype(float),
    })
    z = np.stack([(np.log(c.shift(-horizon) / c) / (sigma * np.sqrt(horizon))).to_numpy()
                  for horizon in HORIZONS], axis=1)
    return MarketData(df, features[list(FEATURES)].to_numpy(float), classify(z), z, sigma.to_numpy())


def before_boundary(indices, boundary, max_horizon=10):
    """Strict purge: every supervised target ends before boundary index."""
    indices = np.asarray(indices, dtype=int)
    return indices[indices + max_horizon < boundary]


def inner_split(prefix, tuning_size=252):
    if len(prefix) < tuning_size + 300:
        raise ValueError("Not enough history for chronological tuning")
    tuning = prefix[-tuning_size:]
    fitting = before_boundary(prefix[:-tuning_size], int(tuning[0]))
    return fitting, tuning


def audit(path, market):
    df = market.bars
    gaps = df.date.diff().dt.days
    gap_indices = np.flatnonzero(gaps.to_numpy() > 4)
    missing_volume = ~(df.volume > 0)
    valid = market.eligible()
    development = valid[market.dates[valid + 10] < np.datetime64(TEST_START)]
    return {
        "symbol": "NSE:CNX500", "source": "Official NSE historical OHLC/session dates joined with TradingView TVRemix volume" if "price_source" in df else "TradingView via TVRemix get_ohlcv",
        "source_interval": "1D", "requested_bar_limit": 5000,
        "retrieved_date": "2026-10-03", "retrieval_time_note": "Exact connector retrieval timestamp was not returned",
        "csv_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "bars": len(df), "first_date": str(df.date.iloc[0].date()), "last_date": str(df.date.iloc[-1].date()),
        "volume_available_bars": int((~missing_volume).sum()), "volume_unavailable_bars": int(missing_volume.sum()),
        "first_positive_volume_date": str(df.loc[~missing_volume, "date"].iloc[0].date()),
        "weekday_sessions_present": df.date.dt.day_name().value_counts().to_dict(),
        "calendar_gaps_over_four_days": [{"before": str(df.date.iloc[i-1].date()),
                                         "after": str(df.date.iloc[i].date()), "calendar_days": int(gaps.iloc[i])}
                                        for i in gap_indices],
        "completeness": "Aligned with official published Nifty 500 historical index sessions; exchange session rules not independently reconstructed" if "price_source" in df else "NOT independently verified against historical NSE session calendars; gaps may be holidays or missing bars",
        "volume_semantics": "UNVERIFIED provider index-volume field; do not interpret as tradable index share volume",
        "ohlc_validation": "passed", "unique_ascending_dates": True,
        "common_eligible_origins": int(len(valid)), "latest_ten_origins_excluded_from_joint_evaluation": True,
        "class_counts_development_only": {str(h): {CLASSES[j]: int((market.y[development, k] == j).sum()) for j in range(5)}
                         for k, h in enumerate(HORIZONS)},
    }
