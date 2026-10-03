"""Convert a locally saved TVRemix OHLCV response into the input CSV."""
import argparse
import json
from pathlib import Path
import pandas as pd

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("response", type=Path)
parser.add_argument("--out", type=Path, default=Path("data/raw/cnx500_daily.csv"))
args = parser.parse_args()
payload = json.loads(args.response.read_text(encoding="utf-8-sig"))
payload = payload.get("structuredContent", payload)
payload = payload.get("data", payload)
if payload.get("symbol") != "NSE:CNX500" or payload.get("interval") != "1D":
    raise ValueError("Only NSE:CNX500 daily source responses accepted")
df = pd.DataFrame(payload["bars"]).rename(columns={"t": "timestamp", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
df["date"] = pd.to_datetime(df.timestamp, unit="s", utc=True).dt.tz_convert("Asia/Kolkata").dt.strftime("%Y-%m-%d")
args.out.parent.mkdir(parents=True, exist_ok=True)
df[["timestamp", "date", "open", "high", "low", "close", "volume"]].to_csv(args.out, index=False)
print(f"Wrote {len(df)} rows to {args.out}")
