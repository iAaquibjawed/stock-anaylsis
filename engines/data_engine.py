"""
Data Engine (v2, hardened)
==========================
Pulls, cleans, validates, caches, and serves OHLCV + corporate-action data for
an Indian equity universe (default: Nifty 500) using free yfinance (.NS tickers).

v2 changes (from code review):
  1. Survivorship bias  -> point-in-time universe via universe/YYYY-MM.csv +
                           load_universe(date). Backtests see the constituents
                           as of the rebalance date, not today's list.
  2. Incremental update -> fetch_symbol reads last cached date and only pulls
                           the delta, then appends.
  3. Retry/backoff      -> 3 attempts with 2s/4s/8s exponential backoff.
  4. Corporate actions  -> caches dividends & splits alongside prices.
  5. Data validation    -> duplicate dates, non-positive prices, OHLC sanity,
                           and abnormal (>70%) non-split jumps are flagged.
  6. Metadata           -> JSON, not TXT (extensible/versioned).
  7. Parallel downloads -> ThreadPoolExecutor over the universe.

Point-in-time honesty: as_of() is PIT-correct for PRICES only. yfinance serves
restated FUNDAMENTALS, so do not treat fundamental fields as point-in-time.

Research scaffolding, not investment advice. Outputs are candidates to research,
never buy signals.

Dependencies:
    pip install yfinance pandas pyarrow
"""
from __future__ import annotations

import json
import time
import datetime as dt
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent  # project root, not cwd
CACHE_DIR = ROOT / "cache" / "prices"
ACTIONS_DIR = ROOT / "cache" / "actions"
META_DIR = ROOT / "cache" / "meta"
UNIVERSE_DIR = ROOT / "universe"
for d in (CACHE_DIR, ACTIONS_DIR, META_DIR, UNIVERSE_DIR):
    d.mkdir(parents=True, exist_ok=True)

SCHEMA_VERSION = 2

SAMPLE_UNIVERSE = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK",
]


# ---------------------------------------------------------------------------
# 1. Point-in-time universe (survivorship bias fix)
# ---------------------------------------------------------------------------
def _universe_files() -> list[Path]:
    return sorted(UNIVERSE_DIR.glob("[0-9][0-9][0-9][0-9]-[0-9][0-9].csv"))


def save_universe_snapshot(symbols: list[str], date: str | dt.date) -> Path:
    """
    Store the universe as it was on a given month: universe/YYYY-MM.csv.
    Call this each time you fetch NSE's live constituents, building history
    forward. A backtest then reads the snapshot effective at each rebalance.
    """
    d = pd.to_datetime(date)
    path = UNIVERSE_DIR / f"{d.strftime('%Y-%m')}.csv"
    pd.DataFrame({"Symbol": symbols}).to_csv(path, index=False)
    return path


def load_universe(date: str | dt.date | None = None) -> list[str]:
    """
    Return the universe effective on `date` (the latest snapshot on or before it).
    If no snapshots exist, fall back to the built-in sample so the engine runs.
    If date is None, return the most recent snapshot (or the sample).
    """
    files = _universe_files()
    if not files:
        return SAMPLE_UNIVERSE
    if date is None:
        chosen = files[-1]
    else:
        cutoff = pd.to_datetime(date)
        eligible = [f for f in files
                    if pd.to_datetime(f.stem + "-01") <= cutoff]
        if not eligible:
            return SAMPLE_UNIVERSE  # date precedes our earliest snapshot
        chosen = eligible[-1]
    df = pd.read_csv(chosen)
    col = "Symbol" if "Symbol" in df.columns else df.columns[0]
    return df[col].astype(str).str.strip().tolist()


def _to_yf(symbol: str) -> str:
    return f"{symbol.strip().upper()}.NS"


# ---------------------------------------------------------------------------
# 5. Validation
# ---------------------------------------------------------------------------
def validate(df: pd.DataFrame, symbol: str) -> list[str]:
    """Return a list of human-readable data-quality warnings (empty = clean)."""
    warns: list[str] = []
    if df.empty:
        return ["empty frame"]
    if df.index.duplicated().any():
        warns.append(f"{int(df.index.duplicated().sum())} duplicate dates")
    if (df["Close"] <= 0).any():
        warns.append("non-positive Close present")
    # OHLC sanity: High should be the max, Low the min of the bar
    bad_hl = (df["High"] < df["Low"]).sum()
    if bad_hl:
        warns.append(f"{int(bad_hl)} bars with High < Low")
    for col in ("Open", "Close"):
        out = ((df[col] > df["High"]) | (df[col] < df["Low"])).sum()
        if out:
            warns.append(f"{int(out)} bars with {col} outside [Low, High]")
    # Abnormal jump without a recorded split -> possible bad data
    if "AdjClose" in df.columns and len(df) > 1:
        pct = df["AdjClose"].pct_change().abs()
        big = pct[pct > 0.70]
        if len(big):
            warns.append(f"{len(big)} daily moves >70% (check for split/error)")
    return warns


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=["Close"])
    df = df[df["Close"] > 0]
    if "Volume" in df.columns:
        df["Volume"] = df["Volume"].fillna(0)
    return df


# ---------------------------------------------------------------------------
# 3. Retry with exponential backoff
# ---------------------------------------------------------------------------
def _download_with_retry(tkr: str, start: str, end: str, attempts: int = 3) -> pd.DataFrame:
    delay = 2.0
    last_err: Exception | None = None
    for i in range(attempts):
        try:
            raw = yf.download(
                tkr, start=start, end=end,
                auto_adjust=False, progress=False, threads=False,
            )
            if raw is not None and not raw.empty:
                return raw
        except Exception as e:  # noqa: BLE001
            last_err = e
        if i < attempts - 1:
            time.sleep(delay)
            delay *= 2  # 2s -> 4s -> 8s
    if last_err:
        raise last_err
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# 4. Corporate actions
# ---------------------------------------------------------------------------
def _fetch_actions(symbol: str) -> None:
    """Cache dividends and splits. Best-effort: never fatal to a price fetch."""
    try:
        t = yf.Ticker(_to_yf(symbol))
        actions = t.actions  # DataFrame with Dividends, Stock Splits
        if actions is not None and not actions.empty:
            actions.index = pd.to_datetime(actions.index).tz_localize(None)
            actions.to_parquet(ACTIONS_DIR / f"{symbol}.parquet")
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# 2. Incremental fetch + cache
# ---------------------------------------------------------------------------
def fetch_symbol(
    symbol: str,
    start: str = "2015-01-01",
    end: str | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """
    Fetch one symbol incrementally, cache as Parquet, return the cleaned frame.
    Keeps both raw Close and split/dividend-adjusted AdjClose.
    """
    cache_file = CACHE_DIR / f"{symbol}.parquet"
    end = end or dt.date.today().isoformat()

    existing = pd.DataFrame()
    fetch_start = start
    if cache_file.exists() and not force:
        existing = pd.read_parquet(cache_file)
        if not existing.empty:
            last = existing.index.max()
            # Already current?
            if last.date() >= (pd.to_datetime(end) - pd.Timedelta(days=1)).date():
                return existing
            fetch_start = (last + pd.Timedelta(days=1)).date().isoformat()

    raw = _download_with_retry(_to_yf(symbol), fetch_start, end)
    df = _clean(raw)
    if df.empty:
        return existing  # nothing new; keep what we had

    if "Adj Close" in df.columns:
        df = df.rename(columns={"Adj Close": "AdjClose"})
    else:
        df["AdjClose"] = df["Close"]
    keep = [c for c in ["Open", "High", "Low", "Close", "AdjClose", "Volume"] if c in df.columns]
    df = df[keep]
    df["Symbol"] = symbol

    if not existing.empty:
        df = pd.concat([existing, df])
        df = df[~df.index.duplicated(keep="last")].sort_index()

    warns = validate(df, symbol)
    df.to_parquet(cache_file)
    _fetch_actions(symbol)

    meta = {
        "symbol": symbol,
        "rows": int(len(df)),
        "first": df.index.min().date().isoformat(),
        "last": df.index.max().date().isoformat(),
        "last_update": dt.datetime.now().isoformat(timespec="seconds"),
        "source": "yahoo",
        "schema_version": SCHEMA_VERSION,
        "warnings": warns,
    }
    (META_DIR / f"{symbol}.json").write_text(json.dumps(meta, indent=2))
    return df


# ---------------------------------------------------------------------------
# 7. Parallel universe build
# ---------------------------------------------------------------------------
def build_cache(
    symbols: list[str],
    start: str = "2015-01-01",
    force: bool = False,
    max_workers: int = 8,
) -> dict[str, int]:
    """Fetch a universe concurrently. Returns {symbol: rows}. Failures -> 0."""
    report: dict[str, int] = {}

    def _job(sym: str) -> tuple[str, int, str]:
        try:
            df = fetch_symbol(sym, start=start, force=force)
            return sym, len(df), ("ok" if len(df) else "EMPTY")
        except Exception as e:  # noqa: BLE001
            return sym, 0, f"FAIL ({type(e).__name__})"

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_job, s): s for s in symbols}
        for i, fut in enumerate(as_completed(futures), 1):
            sym, rows, status = fut.result()
            report[sym] = rows
            print(f"[{i:>3}/{len(symbols)}] {sym:<12} {status}")
    return report


# ---------------------------------------------------------------------------
# Access
# ---------------------------------------------------------------------------
def as_of(symbol: str, date: str | dt.date, price_col: str = "AdjClose") -> pd.DataFrame:
    """All bars up to and including `date` — the anti-look-ahead gateway (prices only)."""
    cache_file = CACHE_DIR / f"{symbol}.parquet"
    if not cache_file.exists():
        raise FileNotFoundError(f"No cache for {symbol}; run build_cache first.")
    df = pd.read_parquet(cache_file)
    sliced = df[df.index <= pd.to_datetime(date)]
    if price_col not in sliced.columns:
        raise KeyError(f"{price_col} not in {list(sliced.columns)}")
    return sliced


def get(symbol: str) -> pd.DataFrame:
    """Full cached frame — for research/features, NOT inside a backtest loop."""
    cache_file = CACHE_DIR / f"{symbol}.parquet"
    if not cache_file.exists():
        raise FileNotFoundError(f"No cache for {symbol}; run build_cache first.")
    return pd.read_parquet(cache_file)


def get_actions(symbol: str) -> pd.DataFrame:
    f = ACTIONS_DIR / f"{symbol}.parquet"
    return pd.read_parquet(f) if f.exists() else pd.DataFrame()


def get_meta(symbol: str) -> dict:
    f = META_DIR / f"{symbol}.json"
    return json.loads(f.read_text()) if f.exists() else {}


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    universe = load_universe()
    print(f"Building cache for {len(universe)} symbols (parallel)...\n")
    report = build_cache(universe, start="2018-01-01", max_workers=8)
    ok = {k: v for k, v in report.items() if v > 0}
    print(f"\nCached {len(ok)}/{len(universe)} symbols with data.")
    if ok:
        sym = next(iter(ok))
        print("\nMeta:", json.dumps(get_meta(sym), indent=2))
        pit = as_of(sym, "2020-06-01")
        print(f"\nas_of(2020-06-01) last visible bar: {pit.index.max().date()}")
