"""
Feature Engine (v1)
===================
Turns cached OHLCV (from data_engine) into a tidy, look-ahead-safe feature
matrix per symbol, cached to Parquet. Features are computed ONCE here so the
Scoring Engine can try hundreds of ranking formulas without recomputation.

Design contract — every feature is CAUSAL:
  A feature value on bar t uses only data available at the close of bar t
  (no centered windows, no future leakage). This is what makes it safe to feed
  into a backtest that decides at t and trades at t+1.

Feature matrix columns (per symbol):
  Returns/momentum : ret_1d, ret_5d, ret_20d, ret_60d, ret_120d
  Trend            : ema_20, ema_50, ema_200, ema_dist_50, golden (50>200)
  Momentum osc.    : rsi_14, macd, macd_signal, macd_hist
  Volatility       : vol_20d (annualized), atr_14_pct
  Liquidity        : adv_20 (avg daily value traded), dollar_vol_z
  Volume           : vol_spike (vol / 20d avg)

NOTE on fundamentals: deliberately omitted. yfinance fundamentals are restated
(not point-in-time), so mixing them into a historical feature matrix silently
introduces look-ahead bias. Add them only with a PIT-correct source.

Research scaffolding, not investment advice.

Dependencies:
    pip install pandas numpy pyarrow
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import data_engine as de

FEATURE_DIR = de.ROOT / "cache" / "features"
FEATURE_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_SCHEMA_VERSION = 3  # v3: + classic 12-1 momentum (own group, 0 default weight)

# ---------------------------------------------------------------------------
# Feature metadata registry
# ---------------------------------------------------------------------------
# Each feature declares: group, direction (higher_better/lower_better/neutral),
# and the default cross-sectional normalization the Scoring Engine should apply.
# This makes the Scoring Engine GENERIC — it reads this registry instead of
# hard-coding which columns mean what. Add a feature here and scoring picks it up.
FEATURE_META: dict[str, dict] = {
    # momentum
    "ret_5d":        {"group": "momentum",   "direction": "higher_better", "normalize": "rank"},
    "ret_20d":       {"group": "momentum",   "direction": "higher_better", "normalize": "rank"},
    "ret_60d":       {"group": "momentum",   "direction": "higher_better", "normalize": "rank"},
    "ret_120d":      {"group": "momentum",   "direction": "higher_better", "normalize": "rank"},
    "consec_up":     {"group": "momentum",   "direction": "higher_better", "normalize": "rank"},
    # classic academic 12-1 momentum — own group so it's OPT-IN (zero weight in the
    # default config; only strategies that name "classic_momentum" use it).
    "mom_12_1":      {"group": "classic_momentum", "direction": "higher_better", "normalize": "rank"},
    # trend
    "ema_dist_50":   {"group": "trend",      "direction": "higher_better", "normalize": "rank"},
    "golden":        {"group": "trend",      "direction": "higher_better", "normalize": "none"},
    "dist_52w_high": {"group": "trend",      "direction": "higher_better", "normalize": "rank"},
    "adx_14":        {"group": "trend",      "direction": "higher_better", "normalize": "rank"},
    "donchian_break":{"group": "trend",      "direction": "higher_better", "normalize": "none"},
    # momentum oscillators (mean-reverting interpretation handled by scorer config)
    "rsi_14":        {"group": "oscillator", "direction": "higher_better", "normalize": "rank"},
    "macd_hist":     {"group": "oscillator", "direction": "higher_better", "normalize": "rank"},
    # volatility (lower is generally preferred for a quality tilt)
    "vol_20d":       {"group": "volatility", "direction": "lower_better",  "normalize": "rank"},
    "atr_14_pct":    {"group": "volatility", "direction": "lower_better",  "normalize": "rank"},
    "gap_pct":       {"group": "volatility", "direction": "lower_better",  "normalize": "rank"},
    # liquidity (higher is safer/tradeable)
    "adv_20":        {"group": "liquidity",  "direction": "higher_better", "normalize": "rank"},
    "vol_spike":     {"group": "liquidity",  "direction": "higher_better", "normalize": "rank"},
}


def feature_columns() -> list[str]:
    """Scorable feature columns (everything in the registry)."""
    return list(FEATURE_META.keys())


# ---------------------------------------------------------------------------
# Indicator primitives (all causal)
# ---------------------------------------------------------------------------
def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    # Wilder's smoothing
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    # When avg_loss == 0 (only gains), RSI is 100
    rsi = rsi.where(avg_loss != 0, 100.0)
    return rsi


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd = _ema(close, fast) - _ema(close, slow)
    sig = macd.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return macd, sig, macd - sig


def _true_range(df: pd.DataFrame) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    return pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)


def _atr_pct(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = _true_range(df)
    atr = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return atr / df["Close"]  # as a fraction of price


def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average Directional Index (Wilder). Causal — uses only past/current bars."""
    high, low = df["High"], df["Low"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    tr = _true_range(df)
    atr = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def _consec_up(daily_ret: pd.Series) -> pd.Series:
    """Signed run length: +N for N up-days in a row, -N for down-days. Causal."""
    sign = np.sign(daily_ret).fillna(0.0)
    out = np.zeros(len(sign))
    run = 0.0
    vals = sign.to_numpy()
    for i, s in enumerate(vals):
        if s > 0:
            run = run + 1 if run > 0 else 1.0
        elif s < 0:
            run = run - 1 if run < 0 else -1.0
        else:
            run = 0.0
        out[i] = run
    return pd.Series(out, index=daily_ret.index)


# ---------------------------------------------------------------------------
# Feature matrix builder
# ---------------------------------------------------------------------------
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the causal feature matrix from a cleaned OHLCV frame.
    Uses AdjClose for returns/trend/momentum; raw Volume*Close for liquidity.
    """
    if df.empty:
        return pd.DataFrame()

    px = df["AdjClose"]
    out = pd.DataFrame(index=df.index)

    # Returns / momentum (trailing, so causal)
    for n in (1, 5, 20, 60, 120):
        out[f"ret_{n}d"] = px.pct_change(n)

    # Classic 12-1 momentum: return from ~12 months ago to ~1 month ago.
    # Skipping the most recent ~21 trading days avoids the well-known short-term
    # reversal. Uses only past data (both legs are shifted) -> causal.
    out["mom_12_1"] = px.shift(21) / px.shift(252) - 1.0

    # Trend
    out["ema_20"] = _ema(px, 20)
    out["ema_50"] = _ema(px, 50)
    out["ema_200"] = _ema(px, 200)
    out["ema_dist_50"] = px / out["ema_50"] - 1.0          # % above/below 50EMA
    out["golden"] = (out["ema_50"] > out["ema_200"]).astype("float")

    # Momentum oscillators
    out["rsi_14"] = _rsi(px, 14)
    macd, sig, hist = _macd(px)
    out["macd"] = macd
    out["macd_signal"] = sig
    out["macd_hist"] = hist

    # Trend (quality)
    roll_high_252 = px.rolling(252, min_periods=126).max()
    out["dist_52w_high"] = px / roll_high_252 - 1.0       # 0 at high, negative below
    out["adx_14"] = _adx(df, 14)
    # Donchian breakout: close exceeds the prior 20-bar high (shift to avoid self)
    prior_high_20 = df["High"].rolling(20, min_periods=20).max().shift(1)
    out["donchian_break"] = (df["Close"] > prior_high_20).astype("float")

    # Volatility
    daily_ret = px.pct_change()
    out["vol_20d"] = daily_ret.rolling(20, min_periods=20).std() * np.sqrt(252)
    out["atr_14_pct"] = _atr_pct(df, 14)
    out["gap_pct"] = (df["Open"] / df["Close"].shift(1) - 1.0).abs()
    out["consec_up"] = _consec_up(daily_ret)

    # Liquidity / volume (value traded = close * volume)
    dollar_vol = df["Close"] * df["Volume"]
    out["adv_20"] = dollar_vol.rolling(20, min_periods=20).mean()
    mu = dollar_vol.rolling(60, min_periods=20).mean()
    sd = dollar_vol.rolling(60, min_periods=20).std()
    out["dollar_vol_z"] = (dollar_vol - mu) / sd.replace(0.0, np.nan)
    vavg = df["Volume"].rolling(20, min_periods=20).mean()
    out["vol_spike"] = df["Volume"] / vavg.replace(0.0, np.nan)

    out["Symbol"] = df["Symbol"].iloc[0] if "Symbol" in df.columns else None
    return out


def build_symbol_features(symbol: str, save: bool = True) -> pd.DataFrame:
    """Read prices from data_engine cache, compute features, cache them."""
    prices = de.get(symbol)
    feats = compute_features(prices)
    if save and not feats.empty:
        feats.to_parquet(FEATURE_DIR / f"{symbol}.parquet")
        _FEATURE_CACHE[symbol] = feats   # keep cache fresh after a rebuild
    return feats


def build_all_features(symbols: list[str] | None = None) -> dict[str, int]:
    symbols = symbols or [p.stem for p in de.CACHE_DIR.glob("*.parquet")]
    report: dict[str, int] = {}
    for sym in symbols:
        try:
            feats = build_symbol_features(sym)
            report[sym] = len(feats)
            print(f"{sym:<12} {len(feats)} rows")
        except Exception as e:  # noqa: BLE001
            report[sym] = 0
            print(f"{sym:<12} FAIL ({type(e).__name__})")
    return report


# ---------------------------------------------------------------------------
# Access (look-ahead-safe gateways for the Scoring Engine)
# ---------------------------------------------------------------------------
# In-memory cache so backtests/validation don't re-read parquet every rebalance.
_FEATURE_CACHE: dict[str, pd.DataFrame] = {}


def clear_cache() -> None:
    _FEATURE_CACHE.clear()


def get_features(symbol: str) -> pd.DataFrame:
    cached = _FEATURE_CACHE.get(symbol)
    if cached is not None:
        return cached
    f = FEATURE_DIR / f"{symbol}.parquet"
    if not f.exists():
        raise FileNotFoundError(f"No features for {symbol}; run build_all_features.")
    df = pd.read_parquet(f)
    _FEATURE_CACHE[symbol] = df
    return df


def features_as_of(symbol: str, date) -> pd.Series | None:
    """
    The single most recent feature row at or before `date` — the row a scorer
    is allowed to see when ranking on that rebalance date. None if no data yet.
    """
    feats = get_features(symbol)
    sliced = feats[feats.index <= pd.to_datetime(date)]
    return None if sliced.empty else sliced.iloc[-1]


def cross_section(symbols: list[str], date) -> pd.DataFrame:
    """
    Build the cross-sectional feature table (one row per symbol) as of `date`.
    This is exactly what the Scoring Engine consumes each rebalance.
    """
    rows = {}
    for sym in symbols:
        try:
            r = features_as_of(sym, date)
            if r is not None:
                rows[sym] = r
        except FileNotFoundError:
            continue
    return pd.DataFrame(rows).T


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    syms = [p.stem for p in de.CACHE_DIR.glob("*.parquet")]
    if not syms:
        print("No cached prices. Run data_engine.py first.")
    else:
        print(f"Building features for {len(syms)} symbols...\n")
        build_all_features(syms)
        sample = syms[0]
        f = get_features(sample)
        print(f"\n{sample} feature columns:\n{list(f.columns)}")
        print(f"\nTail:\n{f[['ret_20d','rsi_14','macd_hist','vol_20d','adv_20']].tail()}")
