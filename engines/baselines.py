"""
Baselines module
================
Before trusting a sophisticated strategy, compare it against naive ones. Beating
the index is not enough; if the engine can't beat simple, cheap strategies after
costs, the finding is "the strategy needs work" (not "the code is broken").

Baselines (all run through the SAME ExecutionSimulator -> after-cost, apples-to-
apples with the real strategy, same next-open timing):

  - buy_hold_index        : invest once in the benchmark, hold
  - buy_hold_equalweight  : equal-weight the whole universe once, hold
  - equal_weight          : equal-weight the universe, rebalanced each period
  - top_momentum(N)       : each period pick top-N by trailing 12m return
  - random_portfolio(N)   : pick N at random each period; run MANY sims to get a
                            distribution, so you can see where the strategy ranks
                            vs luck (its percentile among random portfolios)

compare_baselines(...) returns a tidy table (CAGR / Sharpe / MaxDD) with the
strategy row included and the strategy's percentile vs the random cloud.

Research scaffolding, not investment advice.

Dependencies:
    pip install pandas numpy
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import backtest_engine as bt
from execution_engine import ExecutionSimulator, ExecConfig


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def metrics_from_equity(eq: pd.Series) -> dict:
    eq = eq.dropna()
    if len(eq) < 5:
        return {"CAGR": None, "Sharpe": None, "Max Drawdown": None}
    r = eq.pct_change().dropna()
    yrs = max(len(eq) / 252.0, 1e-9)
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1
    sd = r.std()
    sharpe = (r.mean() / sd * np.sqrt(252)) if sd else np.nan
    return {
        "CAGR": round(float(cagr), 4),
        "Sharpe": round(float(sharpe), 3) if sharpe == sharpe else None,
        "Max Drawdown": round(bt._max_drawdown(eq), 4),
    }


# ---------------------------------------------------------------------------
# Selection-driven runner (reuses backtest timing: signal close -> next open)
# ---------------------------------------------------------------------------
def _available_prices(panel: dict, sig) -> pd.Series:
    c = panel["close"].loc[:sig]
    if c.empty:
        return pd.Series(dtype=float)
    return c.iloc[-1].dropna()


def _trailing_return(panel: dict, sig, lookback: int) -> pd.Series:
    c = panel["close"].loc[:sig]
    if len(c) < 2:
        return pd.Series(dtype=float)
    last = c.iloc[-1]
    past = c.iloc[-(lookback + 1)] if len(c) > lookback else c.iloc[0]
    return (last / past - 1).dropna()


def _run_periodic(panel, days, emap, selector, exec_cfg, cash, rng=None):
    opens, closes, adv = panel["open"], panel["close"], panel["adv_shares"]
    sim = ExecutionSimulator(cash, exec_cfg)
    curve = []
    for d in days:
        if d in emap:
            w = selector(emap[d], panel, rng)
            if w:
                sim.rebalance(d, opens.loc[d].dropna(), w,
                              adv.loc[d] if d in adv.index else None)
        curve.append((d, sim.equity(closes.loc[d].dropna())))
    return pd.Series(dict(curve)).sort_index(), sim


def _run_once(panel, days, emap, weights, exec_cfg, cash):
    """Buy at the first execution day, then hold (no further trades)."""
    opens, closes, adv = panel["open"], panel["close"], panel["adv_shares"]
    sim = ExecutionSimulator(cash, exec_cfg)
    first = sorted(emap)[0] if emap else None
    curve = []
    for d in days:
        if d == first and weights:
            sim.rebalance(d, opens.loc[d].dropna(), weights,
                          adv.loc[d] if d in adv.index else None)
        curve.append((d, sim.equity(closes.loc[d].dropna())))
    return pd.Series(dict(curve)).sort_index(), sim


# ---------------------------------------------------------------------------
# Selectors
# ---------------------------------------------------------------------------
def _sel_equal_weight(sig, panel, rng=None):
    px = _available_prices(panel, sig)
    return {s: 1.0 / len(px) for s in px.index} if len(px) else {}


def _make_sel_top_momentum(n: int, lookback: int = 252):
    def sel(sig, panel, rng=None):
        tr = _trailing_return(panel, sig, lookback)
        top = tr.sort_values(ascending=False).head(n)
        return {s: 1.0 / len(top) for s in top.index} if len(top) else {}
    return sel


def _make_sel_random(n: int):
    def sel(sig, panel, rng):
        px = _available_prices(panel, sig)
        syms = list(px.index)
        if not syms:
            return {}
        k = min(n, len(syms))
        pick = rng.choice(syms, size=k, replace=False)
        return {s: 1.0 / k for s in pick}
    return sel


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def compare_baselines(
    symbols: list[str],
    start: str,
    end: str | None,
    strategy_equity: pd.Series,
    bench_close: pd.Series | None = None,
    exec_cfg: ExecConfig | None = None,
    freq: str = "M",
    cash: float = 1_000_000.0,
    n_momentum: int = 20,
    n_random: int = 100,
    random_size: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Build the baseline comparison table. The strategy row is included, plus the
    strategy's percentile among the random portfolios (higher = better than more
    random portfolios).
    """
    panel = bt.build_panel(symbols)
    closes = panel["close"].loc[start:end]
    days = closes.index
    emap = bt._rebalance_map(days, freq)
    random_size = random_size or n_momentum

    rows = []

    # The strategy itself
    rows.append({"strategy": "** THIS STRATEGY **", **metrics_from_equity(strategy_equity)})

    # Buy & hold index
    if bench_close is not None:
        bc = bench_close.loc[start:end].dropna()
        if len(bc) > 5:
            fee = ((exec_cfg or ExecConfig()).commission_bps +
                   (exec_cfg or ExecConfig()).slippage_bps) / 1e4
            bh = cash * (1 - fee) * (bc / bc.iloc[0])
            rows.append({"strategy": "buy_hold_index", **metrics_from_equity(bh)})

    # Buy & hold equal weight (once)
    first = sorted(emap)[0] if emap else None
    if first is not None:
        w0 = _sel_equal_weight(emap[first], panel)
        eq, _ = _run_once(panel, days, emap, w0, exec_cfg, cash)
        rows.append({"strategy": "buy_hold_equalweight", **metrics_from_equity(eq)})

    # Equal weight rebalanced
    eq, _ = _run_periodic(panel, days, emap, _sel_equal_weight, exec_cfg, cash)
    rows.append({"strategy": "equal_weight_rebal", **metrics_from_equity(eq)})

    # Top-N 12m momentum
    eq, _ = _run_periodic(panel, days, emap, _make_sel_top_momentum(n_momentum),
                          exec_cfg, cash)
    rows.append({"strategy": f"top{n_momentum}_momentum_12m", **metrics_from_equity(eq)})

    # Random portfolios (distribution)
    sel_rand = _make_sel_random(random_size)
    rand_cagr, rand_sharpe = [], []
    for i in range(n_random):
        rng = np.random.default_rng(seed + i)
        eqr, _ = _run_periodic(panel, days, emap, sel_rand, exec_cfg, cash, rng)
        m = metrics_from_equity(eqr)
        if m["CAGR"] is not None:
            rand_cagr.append(m["CAGR"])
        if m["Sharpe"] is not None:
            rand_sharpe.append(m["Sharpe"])
    if rand_cagr:
        rows.append({"strategy": f"random_x{n_random} (median)",
                     "CAGR": round(float(np.median(rand_cagr)), 4),
                     "Sharpe": round(float(np.median(rand_sharpe)), 3) if rand_sharpe else None,
                     "Max Drawdown": None})
        rows.append({"strategy": f"random_x{n_random} (95th pct)",
                     "CAGR": round(float(np.percentile(rand_cagr, 95)), 4),
                     "Sharpe": round(float(np.percentile(rand_sharpe, 95)), 3) if rand_sharpe else None,
                     "Max Drawdown": None})

    df = pd.DataFrame(rows)

    # Strategy percentile vs the random cloud (by CAGR and Sharpe)
    s_m = metrics_from_equity(strategy_equity)
    pct_cagr = (float(np.mean([s_m["CAGR"] >= x for x in rand_cagr])) * 100
                if rand_cagr and s_m["CAGR"] is not None else np.nan)
    pct_sharpe = (float(np.mean([s_m["Sharpe"] >= x for x in rand_sharpe])) * 100
                  if rand_sharpe and s_m["Sharpe"] is not None else np.nan)
    df.attrs["strategy_percentile"] = {
        "vs_random_CAGR_pct": round(pct_cagr, 1) if pct_cagr == pct_cagr else None,
        "vs_random_Sharpe_pct": round(pct_sharpe, 1) if pct_sharpe == pct_sharpe else None,
        "n_random": n_random,
    }
    return df


if __name__ == "__main__":
    print("Baselines module — call compare_baselines() from the runner or a script.")
