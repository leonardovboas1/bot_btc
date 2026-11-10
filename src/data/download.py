"""Idempotent, resumable Binance public-data downloader (no API keys).

Usage (from project root):
    .venv/Scripts/python -m src.data.download all
    .venv/Scripts/python -m src.data.download klines --market perp --intervals 1h 4h
    .venv/Scripts/python -m src.data.download funding | futdata | metrics-archive | bookdepth
"""
from __future__ import annotations

import argparse
import io
import re
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
SYMBOL = "BTCUSDT"
FAPI = "https://fapi.binance.com"
SPOT = "https://api.binance.com"
VISION = "https://data.binance.vision"
S3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"

PERP_START = datetime(2019, 9, 8, tzinfo=timezone.utc)
SPOT_START = datetime(2017, 8, 17, tzinfo=timezone.utc)

INTERVAL_MS = {"5m": 300_000, "15m": 900_000, "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}
KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
              "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]
FUTDATA = {  # endpoint -> value columns
    "openInterestHist": ["sumOpenInterest", "sumOpenInterestValue"],
    "globalLongShortAccountRatio": ["longShortRatio", "longAccount", "shortAccount"],
    "topLongShortAccountRatio": ["longShortRatio", "longAccount", "shortAccount"],
    "topLongShortPositionRatio": ["longShortRatio", "longAccount", "shortAccount"],
    "takerlongshortRatio": ["buySellRatio", "buyVol", "sellVol"],
}

_sess = requests.Session()
_sess.headers["User-Agent"] = "bot_trade-research/1.0"


def now_ms() -> int:
    return int(time.time() * 1000)


def ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def api_get(base: str, path: str, params: dict, weight_limit: int) -> requests.Response:
    """GET with weight-aware throttling and 429/418 back-off."""
    for attempt in range(8):
        try:
            r = _sess.get(base + path, params=params, timeout=30)
        except requests.RequestException:
            time.sleep(min(2 ** attempt, 60))
            continue
        if r.status_code in (429, 418):
            wait = int(r.headers.get("Retry-After", 60 if r.status_code == 429 else 600))
            print(f"[rate-limit {r.status_code}] sleeping {wait}s", file=sys.stderr)
            time.sleep(wait + 1)
            continue
        if r.status_code >= 500:
            time.sleep(min(2 ** attempt, 60))
            continue
        r.raise_for_status()
        used = int(r.headers.get("x-mbx-used-weight-1m") or 0)  # headers are case-insensitive
        # near the cap: wait for the 1-minute window to roll over
        time.sleep(15 if used > 0.8 * weight_limit else 0.05)
        return r
    raise RuntimeError(f"giving up on {path} {params}")


def out_path(market: str, name: str) -> Path:
    base = RAW if SYMBOL == "BTCUSDT" else RAW / SYMBOL  # BTC keeps legacy paths
    p = base / market / f"{name}.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def load(path: Path) -> pd.DataFrame | None:
    return pd.read_parquet(path) if path.exists() else None


def save(path: Path, new: pd.DataFrame, old: pd.DataFrame | None, dedupe: bool = True) -> pd.DataFrame:
    df = new if old is None or old.empty else pd.concat([old, new])
    if dedupe:  # bookDepth has several rows per timestamp (one per depth level): skip
        df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index(kind="stable")
    df.to_parquet(path, compression="zstd")
    return df


def to_utc_index(s: pd.Series, name: str) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(s.astype("int64"), unit="ms", utc=True), name=name)


def last_ms(df: pd.DataFrame) -> int:
    return int(df.index.max().timestamp() * 1000)


# --------------------------------------------------------------------- klines
def download_klines(market: str, interval: str) -> None:
    perp = market == "perp"
    base, path, limit, wl = (FAPI, "/fapi/v1/klines", 1500, 2400) if perp else (SPOT, "/api/v3/klines", 1000, 6000)
    folder = "um_futures" if perp else "spot"
    p = out_path(folder, f"klines_{interval}")
    old = load(p)
    step = INTERVAL_MS[interval]
    start = ms(PERP_START if perp else SPOT_START)
    if old is not None and len(old):
        start = last_ms(old) + step
    end = now_ms()
    rows: list = []
    est = max((end - start) // (step * limit), 0) + 1
    with tqdm(total=est, desc=f"{folder} klines {interval}", leave=False) as bar:
        while start < end:
            r = api_get(base, path, {"symbol": SYMBOL, "interval": interval, "startTime": start, "limit": limit}, wl).json()
            if not r:
                break
            rows.extend(r)
            start = r[-1][0] + step
            bar.update(1)
            if len(r) < limit:
                break
    if not rows:
        print(f"{folder}/klines_{interval}: up to date")
        return
    df = pd.DataFrame(rows, columns=KLINE_COLS)
    df = df[df["close_time"].astype("int64") < end]  # drop the still-forming candle
    if df.empty:
        print(f"{folder}/klines_{interval}: up to date")
        return
    df.index = to_utc_index(df["open_time"], "open_time")
    df = df.drop(columns=["open_time", "close_time", "ignore"]).astype("float64")
    out = save(p, df, old)
    print(f"{folder}/klines_{interval}: +{len(df)} rows -> {len(out)} total")


# -------------------------------------------------------------------- funding
def download_funding() -> None:
    p = out_path("um_futures", "funding_rate")
    old = load(p)
    start = ms(PERP_START)
    if old is not None and len(old):
        start = last_ms(old) + 1
    rows: list = []
    while True:
        r = api_get(FAPI, "/fapi/v1/fundingRate", {"symbol": SYMBOL, "startTime": start, "limit": 1000}, 2400).json()
        if not r:
            break
        rows.extend(r)
        start = r[-1]["fundingTime"] + 1
        if len(r) < 1000:
            break
    if not rows:
        print("funding_rate: up to date")
        return
    df = pd.DataFrame(rows)
    # fundingTime has few-ms jitter after the hour: round to the second
    df.index = to_utc_index(df["fundingTime"], "funding_time").round("s")
    df = df[["fundingRate", "markPrice"]].replace("", float("nan")).astype("float64")
    out = save(p, df, old)
    print(f"funding_rate: +{len(df)} rows -> {len(out)} total")


# ------------------------------------------------- futures/data (~30d window)
def download_futdata(periods=("5m", "1h")) -> None:
    for ep, cols in FUTDATA.items():
        for period in periods:
            p = out_path("um_futures", f"{ep}_{period}")
            old = load(p)
            step = INTERVAL_MS[period]
            end = now_ms()
            start = end - 29 * 86_400_000  # REST only serves ~30 days
            if old is not None and len(old):
                start = max(start, last_ms(old) + step)
            rows: list = []
            while start < end:
                r = api_get(FAPI, f"/futures/data/{ep}",
                            {"symbol": SYMBOL, "period": period, "startTime": start,
                             "endTime": min(end, start + 499 * step), "limit": 500}, 2400).json()
                if not isinstance(r, list):
                    raise RuntimeError(f"{ep}: {r}")
                if r:
                    nxt = r[-1]["timestamp"] + step
                    if nxt <= start:  # endpoint echoes the latest row when start is past the data: stop
                        break
                    rows.extend(x for x in r if x["timestamp"] >= start)
                    start = nxt
                else:
                    start += 500 * step
            if not rows:
                print(f"{ep}_{period}: up to date")
                continue
            df = pd.DataFrame(rows)
            df.index = to_utc_index(df["timestamp"], "timestamp")
            df = df[cols].astype("float64")
            out = save(p, df, old)
            print(f"{ep}_{period}: +{len(df)} rows -> {len(out)} total")


# ------------------------------------------------ data.binance.vision archive
def s3_list(prefix: str) -> list[str]:
    """List all .zip keys under prefix (paginated S3 XML listing)."""
    keys, marker = [], ""
    while True:
        r = _sess.get(S3, params={"delimiter": "/", "prefix": prefix, "marker": marker}, timeout=60)
        r.raise_for_status()
        found = re.findall(r"<Key>([^<]+)</Key>", r.text)
        keys += [k for k in found if k.endswith(".zip")]
        if "<IsTruncated>true</IsTruncated>" not in r.text or not found:
            return keys
        marker = found[-1]


def _fetch_zip_csv(key: str) -> pd.DataFrame | None:
    for attempt in range(4):
        try:
            r = _sess.get(f"{VISION}/{key}", timeout=60)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                return pd.read_csv(z.open(z.namelist()[0]))
        except (requests.RequestException, zipfile.BadZipFile):
            time.sleep(2 ** attempt)
    return None


def _download_archive(dataset: str, since: str, name: str, ts_col: str, workers: int = 8) -> None:
    """Download daily archive zips of a UM-futures dataset; skip days already stored.

    Monthly zips do not exist for metrics/bookDepth, only daily ones.
    """
    p = out_path("um_futures", name)
    old = load(p)
    have = set(old.index.strftime("%Y-%m-%d")) if old is not None else set()
    keys = s3_list(f"data/futures/um/daily/{dataset}/{SYMBOL}/")
    todo = []
    for k in keys:
        d = re.search(r"(\d{4}-\d{2}-\d{2})\.zip$", k).group(1)
        if d >= since and d not in have:
            todo.append(k)
    print(f"{dataset}: {len(keys)} archive days listed, {len(todo)} to fetch")
    if not todo:
        return
    frames = []
    with ThreadPoolExecutor(workers) as ex:
        for df in tqdm(ex.map(_fetch_zip_csv, todo), total=len(todo), desc=dataset, leave=False):
            if df is not None and len(df):
                frames.append(df)
    if not frames:
        return
    df = pd.concat(frames, ignore_index=True)
    if ts_col == "timestamp":
        ts = pd.to_datetime(df[ts_col], utc=True)
    else:
        ts = pd.to_datetime(df[ts_col], utc=True)
    df.index = pd.DatetimeIndex(ts, name=ts_col)
    df = df.drop(columns=[ts_col, "symbol"], errors="ignore")
    df = df.apply(pd.to_numeric, errors="coerce").astype("float64")
    out = save(p, df, old, dedupe=(ts_col != "timestamp"))
    print(f"{name}: +{len(df)} rows -> {len(out)} total")


def download_metrics_archive(since: str = "2021-01-01") -> None:
    """metrics: OI, top-trader and global L/S ratios, taker vol ratio at 5m; daily zips."""
    _download_archive("metrics", since, "metrics_archive_5m", "create_time")


def download_bookdepth(since: str = "2023-01-01") -> None:
    """bookDepth: +-1..5% order-book depth snapshots (~30s) for liquidity heatmaps. Large."""
    _download_archive("bookDepth", since, "bookdepth_archive", "timestamp", workers=6)


# ------------------------------------------------------------------------ CLI
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["all", "klines", "funding", "futdata", "metrics-archive", "bookdepth"])
    ap.add_argument("--market", choices=["perp", "spot", "both"], default="both")
    ap.add_argument("--intervals", nargs="+", help="override kline intervals")
    ap.add_argument("--symbol", default="BTCUSDT", help="e.g. ETHUSDT (BTCUSDT keeps legacy paths)")
    ap.add_argument("--since", help="archive start date YYYY-MM-DD")
    a = ap.parse_args(argv)
    global SYMBOL
    SYMBOL = a.symbol.upper()

    if a.what in ("all", "klines"):
        jobs = []
        if a.market in ("perp", "both"):
            jobs += [("perp", i) for i in (a.intervals or ["5m", "15m", "1h", "4h", "1d"])]
        if a.market in ("spot", "both"):
            jobs += [("spot", i) for i in (a.intervals or ["1h", "4h", "1d"])]
        for m, i in jobs:
            download_klines(m, i)
    if a.what in ("all", "funding"):
        download_funding()
    if a.what in ("all", "futdata"):
        download_futdata()
    if a.what in ("all", "metrics-archive"):
        download_metrics_archive(a.since or "2021-01-01")
    if a.what == "bookdepth":  # not part of "all": large
        download_bookdepth(a.since or "2023-01-01")


if __name__ == "__main__":
    main()
