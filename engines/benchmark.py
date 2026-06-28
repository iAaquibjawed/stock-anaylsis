"""
Benchmark module
================
Index-relative performance and relative-strength features. Everything here needs
a benchmark price series (e.g. Nifty 50 '^NSEI' or Nifty 500 '^CRSLDX' via
yfinance). Cache it like any other symbol through data_engine, then pass it in.

Until a real index series is cached these functions simply aren't called — no
faked beta/alpha. That's the honest stance: index-dependent numbers appear only
when index data actually exists.

Provides:
  benchmark_returns(symbol)                  -> daily returns of the cached index
  benchmark_metrics(equity, bench_close)     -> alpha, beta, tracking error, IR
  relative_strength(symbol_px, bench_px, n)  -> a causal RS feature series

Research scaffolding, not investment advice.

Dependencies:
    pip install pandas numpy
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import data_engine as de


def benchmark_returns(bench_symbol: str) -> pd.Series:
    """Daily returns of a cached benchmark price series (AdjClose)."""
    df = de.get(bench_symbol)               # raises if not cached — by design
    return df["AdjClose"].pct_change().dropna()


def benchmark_metrics(equity: pd.Series, bench_close: pd.Series,
                      rf_annual: float = 0.0) -> dict:
    """
    Index-relative metrics from a strategy equity curve and a benchmark close.
    Aligns on common dates. rf_annual is an optional risk-free (annual, decimal).
    """
    strat_r = equity.pct_change().dropna()
    bench_r = bench_close.pct_change().dropna()
    df = pd.concat([strat_r.rename("s"), bench_r.rename("b")], axis=1).dropna()
    if len(df) < 30:
        return {"error": "insufficient overlapping data"}

    rf_daily = rf_annual / 252.0
    s = df["s"] - rf_daily
    b = df["b"] - rf_daily

    var_b = b.var()
    beta = float(s.cov(b) / var_b) if var_b else np.nan
    # CAPM alpha (annualized)
    alpha_daily = s.mean() - beta * b.mean()
    alpha_ann = float(alpha_daily * 252)

    active = df["s"] - df["b"]
    te = float(active.std() * np.sqrt(252))             # tracking error
    ir = float(active.mean() / active.std() * np.sqrt(252)) if active.std() else np.nan

    # benchmark + strategy CAGR over the overlap, for context
    def _cagr(r):
        cum = (1 + r).prod()
        yrs = max(len(r) / 252.0, 1e-9)
        return float(cum ** (1 / yrs) - 1)

    return {
        "Beta": round(beta, 3),
        "Alpha (ann)": round(alpha_ann, 4),
        "Tracking Error": round(te, 4),
        "Information Ratio": round(ir, 3) if ir == ir else None,
        "Strategy CAGR": round(_cagr(df["s"]), 4),
        "Benchmark CAGR": round(_cagr(df["b"]), 4),
        "Excess CAGR": round(_cagr(df["s"]) - _cagr(df["b"]), 4),
        "Overlap Days": int(len(df)),
    }


def relative_strength(symbol_px: pd.Series, bench_px: pd.Series, window: int = 63) -> pd.Series:
    """
    Causal relative-strength feature: trailing return of the stock minus the
    trailing return of the benchmark over `window` bars. Positive = outperforming.
    Align both on the stock's index; only past data is used (pct_change over a
    trailing window), so it's safe for the feature matrix.
    """
    bench = bench_px.reindex(symbol_px.index).ffill()
    rs = symbol_px.pct_change(window) - bench.pct_change(window)
    rs.name = f"rel_strength_{window}"
    return rs
