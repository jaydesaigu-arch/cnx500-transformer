"""Canonical published NSE OHLC/session dates, joined to original TV volume.

Reads public official historical data in <=365-day chunks when cache is missing.
Raw responses and derived market data remain local. Never overwrite the TV CSV.
"""
import argparse
import datetime
import hashlib
import http.cookiejar
import json
from pathlib import Path
import urllib.request
import numpy as np
import pandas as pd
from cnx500.data import read_bars, write_json

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", type=Path, default=Path.cwd())
parser.add_argument("--refresh", action="store_true", help="Refetch official cached chunks")
args = parser.parse_args()
root = args.root.resolve()
tv_path = root / "data/raw/cnx500_daily.csv"
tv = read_bars(tv_path)
prior_manifest = root / "reports/data_reconciliation.json"
if prior_manifest.exists() and not args.refresh:
    previous = json.loads(prior_manifest.read_text(encoding="utf-8"))
    if previous["tv_csv_sha256"] != hashlib.sha256(tv_path.read_bytes()).hexdigest():
        raise ValueError("TV snapshot changed; use --refresh to rebuild official caches")
cache = root / "data/raw/official_verification"
cache.mkdir(parents=True, exist_ok=True)
opener = None
all_rows = []
chunk_counts = []
base = "https://www.niftyindices.com"
endpoint = base + "/BackPage/getHistoricaldatatabletoString"
for year in range(tv.date.min().year, tv.date.max().year + 1):
    path = cache / f"{year}.json"
    if args.refresh or not path.exists():
        if opener is None:
            opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
            opener.addheaders = [("User-Agent", "Mozilla/5.0"), ("Referer", base + "/reports/historical-data")]
            opener.open(base + "/reports/historical-data", timeout=25).read()
        start = max(tv.date.min(), pd.Timestamp(year=year, month=1, day=1))
        end = min(tv.date.max(), pd.Timestamp(year=year, month=12, day=31))
        info = {"name":"NIFTY 500", "indexName":"NIFTY 500", "startDate":start.strftime("%d-%b-%Y"), "endDate":end.strftime("%d-%b-%Y")}
        request = urllib.request.Request(endpoint, data=json.dumps({"cinfo":json.dumps(info)}).encode(),
                                         headers={"Content-Type":"application/json; charset=utf-8"})
        path.write_bytes(opener.open(request, timeout=25).read())
    rows = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(rows, dict):
        rows = rows.get("d", rows)
    if isinstance(rows, str):
        rows = json.loads(rows)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Missing official history for {year}")
    all_rows.extend(rows)
    chunk_counts.append({"year":year,"rows":len(rows),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
official = pd.DataFrame(all_rows)
official["date"] = pd.to_datetime(official.HistoricalDate, format="%d %b %Y")
official = official.loc[(official.date >= tv.date.min()) & (official.date <= tv.date.max())].sort_values("date")
if official.date.duplicated().any():
    raise ValueError("Duplicate official historical session dates")
official = official.rename(columns={"OPEN":"open", "HIGH":"high", "LOW":"low", "CLOSE":"close"})
official = official[["date", "open", "high", "low", "close"]].copy()
for key in ("open", "high", "low", "close"):
    official[key] = pd.to_numeric(official[key], errors="raise")
joined = official.merge(tv, on="date", how="outer", suffixes=("", "_tv"), indicator=True)
if (joined._merge == "right_only").any():
    raise ValueError("TV dates absent from official history; investigate before reconciling")
missing = joined._merge == "left_only"
invalid_official = ((joined.low > joined[["open", "close"]].min(axis=1)) |
                    (joined.high < joined[["open", "close"]].max(axis=1)) |
                    (joined.low > joined.high) | (joined[["open", "high", "low", "close"]] <= 0).any(axis=1))
if (invalid_official & missing).any():
    raise ValueError("Official-only sessions contain invalid OHLC; no trustworthy fallback")
mismatch = np.zeros(len(joined), dtype=bool)
for key in ("open", "high", "low", "close"):
    mismatch |= ((joined[key] - joined[key + "_tv"]).abs() > .011).fillna(False).to_numpy()
canonical = joined[["date", "open", "high", "low", "close", "volume"]].copy()
for key in ("open", "high", "low", "close"):
    canonical.loc[invalid_official, key] = joined.loc[invalid_official, key + "_tv"]
canonical["volume"] = canonical.volume.fillna(0)
# Date anchors, not claimed intraday opening times; historical special sessions
# need not have started at 09:15. Daily modeling uses the session date only.
anchors = (canonical.date.dt.tz_localize("Asia/Kolkata") + pd.Timedelta(hours=9, minutes=15)).dt.tz_convert("UTC")
canonical["timestamp"] = anchors.dt.as_unit("s").astype("int64")
canonical["price_source"] = np.where(invalid_official, "TradingView: official OHLC failed shape validation", "NSE published index history")
canonical["volume_source"] = np.where(missing, "unavailable", "TradingView TVRemix")
canonical["official_only_session"] = missing.astype(int)
canonical["tv_ohlc_differs_beyond_rounding"] = mismatch.astype(int)
canonical["date"] = canonical.date.dt.strftime("%Y-%m-%d")
canonical = canonical[["timestamp", "date", "open", "high", "low", "close", "volume", "price_source", "volume_source", "official_only_session", "tv_ohlc_differs_beyond_rounding"]]
destination = root / "data/processed/cnx500_daily_verified.csv"
destination.parent.mkdir(parents=True, exist_ok=True)
canonical.to_csv(destination, index=False)
read_bars(destination)  # Enforce the same schema and OHLC rules as training.
report = {"created_at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
          "official_source":base + "/reports/historical-data", "official_endpoint":endpoint,
          "original_tv_rows":len(tv), "canonical_rows":len(canonical),
          "official_only_sessions_added":int(missing.sum()),
          "official_only_dates":joined.loc[missing,"date"].dt.strftime("%Y-%m-%d").tolist(),
          "shared_dates_with_material_ohlc_corrections":int((mismatch & ~invalid_official).sum()),
          "material_ohlc_correction_dates":joined.loc[mismatch & ~invalid_official,"date"].dt.strftime("%Y-%m-%d").tolist(),
          "official_invalid_ohlc_fallback_count":int(invalid_official.sum()),
          "official_invalid_ohlc_fallback_dates":joined.loc[invalid_official,"date"].dt.strftime("%Y-%m-%d").tolist(),
          "ohlc_policy":"Official published NSE OHLC where shape-valid; retain previously validated TV OHLC on flagged official-inconsistent dates. Original TV CSV retained unchanged.",
          "volume_policy":"Retain provider-specific TV volume on shared dates; zero denotes unavailable volume for official-only sessions",
          "timestamp_policy":"09:15 Asia/Kolkata session-date anchors, not claimed actual special-session opening times",
          "tv_csv_sha256":hashlib.sha256(tv_path.read_bytes()).hexdigest(),
          "canonical_csv_sha256":hashlib.sha256(destination.read_bytes()).hexdigest(),
          "official_chunks":chunk_counts}
write_json(root / "reports/data_reconciliation.json", report)
print(json.dumps({k:v for k,v in report.items() if k != "official_chunks"},indent=2))
