"""
Backtest Engine (Engine 4)
==========================
Simulates the strategy through history with correct timing, delegating all fill
mechanics to the Execution Simulator. The cardinal rule, enforced structurally:

    Features/scores are computed on the SIGNAL day's close.
    Orders are filled on the NEXT trading day's OPEN.
    -> we never trade on the same close used to generate the signal.

Flow per rebalance:
    month-end close (signal)  ->  score()  ->  top N  ->  fill next open  ->  hold

Price handling: valuation uses AdjClose (split/dividend consistent). Fills use an
"adjusted open" = Open * AdjClose/Close, so opens live on the same adjusted scale
as the equity curve (avoids a raw-vs-adjusted mismatch on corporate-action days).

Metrics reported (from day one):
  Performance : CAGR, ann. return, ann. vol, Sharpe, Sortino, Calmar, max DD,
                win rate (per rebalance period), profit factor
  Trading     : turnover (ann.), avg holding period, num trades, commission,
                slippage, cash drag
  Portfolio   : avg position size, largest position, concentration (HHI)
  (Beta vs Nifty needs an index series — left as a hook, not faked.)

Research scaffolding, not investment advice. A good backtest is a reason to keep
testing (out-of-sample, costs sensitivity), never a reason to trade.

Dependencies:
    pip install pandas numpy
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import data_engine as de
import feature_engine as fe
import scoring_engine as se
from execution_engine import ExecutionSimulator, ExecConfig


# ---------------------------------------------------------------------------
# Price panel
# ---------------------------------------------------------------------------
def build_panel(symbols: list[str]) -> dict[str, pd.DataFrame]:
    """Aligned date x symbol frames for adjusted open, adjusted close, ADV(shares)."""
    opens, closes, advsh = {}, {}, {}
    for sym in symbols:
        try:
            df = de.get(sym)
        except FileNotFoundError:
            continue
        if df.empty:
            continue
        adj = df["AdjClose"]
        factor = (df["AdjClose"] / df["Close"]).replace([np.inf, -np.inf], np.nan)
        opens[sym] = df["Open"] * factor          # open on adjusted scale
        closes[sym] = adj
        # ADV in shares, lagged one day so an open trade only knows the past
        advsh[sym] = df["Volume"].rolling(20, min_periods=10).mean().shift(1)
    return {
        "open": pd.DataFrame(opens).sort_index(),
        "close": pd.DataFrame(closes).sort_index(),
        "adv_shares": pd.DataFrame(advsh).sort_index(),
    }


def _rebalance_map(trading_days: pd.DatetimeIndex, freq: str) -> dict[pd.Timestamp, pd.Timestamp]:
    """
    Map each EXECUTION day -> its SIGNAL day. Signal = last trading day of each
    period; execution = the next trading day after the signal. This is what makes
    'signal close, fill next open' structural rather than a convention.
    """
    s = pd.Series(trading_days, index=trading_days)
    period = trading_days.to_period(freq)
    signal_days = s.groupby(period).last().values
    signal_days = pd.DatetimeIndex(signal_days)
    exec_map: dict[pd.Timestamp, pd.Timestamp] = {}
    for sig in signal_days:
        later = trading_days[trading_days > sig]
        if len(later):
            exec_map[later[0]] = sig
    return exec_map


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------
@dataclass
class BacktestResult:
    equity: pd.Series
    period_returns: pd.Series
    metrics: dict
    fills: pd.DataFrame
    importance: pd.Series          # avg factor attribution across rebalances
    sim: ExecutionSimulator
    snapshots: pd.DataFrame = None         # per-rebalance PortfolioSnapshot rows
    journal: pd.DataFrame = None           # one row per realized round trip


def build_trade_journal(sim: ExecutionSimulator) -> pd.DataFrame:
    """One permanent record per realized round trip (entry/exit/return/holding)."""
    if not sim.round_trips:
        return pd.DataFrame()
    j = pd.DataFrame(sim.round_trips)
    return j.sort_values("exit_date").reset_index(drop=True)


def run_backtest(
    symbols: list[str],
    start: str,
    end: str | None = None,
    strat: se.StrategyConfig | None = None,
    exec_cfg: ExecConfig | None = None,
    starting_cash: float = 1_000_000.0,
    freq: str = "M",
) -> BacktestResult:
    strat = strat or se.StrategyConfig()
    panel = build_panel(symbols)
    if panel["close"].empty:
        raise ValueError("No price data. Run data_engine then feature_engine first.")

    closes = panel["close"].loc[start:end]
    opens = panel["open"]
    advsh = panel["adv_shares"]
    days = closes.index
    exec_map = _rebalance_map(days, freq)

    sim = ExecutionSimulator(starting_cash, exec_cfg)
    equity_curve: list[tuple[pd.Timestamp, float]] = []
    importances: list[pd.Series] = []
    snapshots: list[dict] = []
    peak = -np.inf

    for d in days:
        if d in exec_map:
            signal_date = exec_map[d]
            cs = fe.cross_section(symbols, signal_date)        # causal
            ranked = se.score(cs, strat)
            top = ranked.head(strat.top_n)
            if not top.empty:
                w = {sym: 1.0 / len(top) for sym in top.index}  # equal weight
                day_open = opens.loc[d].dropna()
                day_adv = advsh.loc[d] if d in advsh.index else None
                turn_before = sim.turnover_notional
                sim.rebalance(d, day_open, w, day_adv)
                importances.append(se.feature_importance(top))
                # Portfolio snapshot at this rebalance (close-valued)
                cl = closes.loc[d].dropna()
                eq_now = sim.equity(cl)
                peak = max(peak, eq_now)
                wts = sim.weights(cl)
                snapshots.append({
                    "date": d, "signal_date": signal_date,
                    "cash": round(sim.cash, 2), "equity": round(eq_now, 2),
                    "n_positions": len(wts),
                    "gross_exposure": round(sum(wts.values()), 4),
                    "largest_position": round(max(wts.values()), 4) if wts else 0.0,
                    "hhi": round(sum(v * v for v in wts.values()), 4) if wts else 0.0,
                    "rebalance_turnover": round(sim.turnover_notional - turn_before, 2),
                    "drawdown": round(eq_now / peak - 1.0, 4) if peak > 0 else 0.0,
                })
        equity_curve.append((d, sim.equity(closes.loc[d].dropna())))

    eq = pd.Series(dict(equity_curve)).sort_index()
    metrics = _compute_metrics(eq, sim, exec_map, starting_cash, panel, days, strat)
    imp = (pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False)
           if importances else pd.Series(dtype=float))
    fills_df = pd.DataFrame(sim.fills)
    snaps_df = pd.DataFrame(snapshots)
    journal = build_trade_journal(sim)
    return BacktestResult(eq, metrics["_period_returns"], metrics, fills_df, imp, sim,
                          snaps_df, journal)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _max_drawdown(eq: pd.Series) -> float:
    roll_max = eq.cummax()
    dd = eq / roll_max - 1.0
    return float(dd.min())


def _compute_metrics(eq, sim, exec_map, starting_cash, panel, days, strat) -> dict:
    daily_ret = eq.pct_change().dropna()
    n = len(eq)
    years = max(n / 252.0, 1e-9)
    total_ret = eq.iloc[-1] / eq.iloc[0] - 1.0
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1.0

    ann_ret = daily_ret.mean() * 252
    ann_vol = daily_ret.std() * np.sqrt(252)
    downside = daily_ret[daily_ret < 0].std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol else np.nan
    sortino = ann_ret / downside if downside else np.nan
    mdd = _max_drawdown(eq)
    calmar = cagr / abs(mdd) if mdd else np.nan

    # Per-rebalance-period returns (win rate / profit factor on the cadence we trade)
    exec_days = pd.DatetimeIndex(sorted(exec_map.keys()))
    exec_days = exec_days[exec_days.isin(eq.index)]
    period_eq = eq.reindex(exec_days)
    period_returns = period_eq.pct_change().dropna()
    wins = period_returns[period_returns > 0]
    losses = period_returns[period_returns < 0]
    win_rate = len(wins) / len(period_returns) if len(period_returns) else np.nan
    profit_factor = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else np.nan

    # Trading
    avg_equity = eq.mean()
    n_rebal = len(exec_days)
    turnover_ann = (sim.turnover_notional / avg_equity / years) if avg_equity else np.nan
    avg_hold = float(np.mean(sim.holding_periods)) if sim.holding_periods else np.nan

    # Portfolio: exact per-day weights need daily snapshots; we report
    # realized-trade-based figures that are exact, plus end-of-test concentration.
    closes = panel["close"]
    realized = pd.Series([t["pnl"] for t in sim.round_trips], dtype=float)
    end_prices = closes.loc[days[-1]].dropna()
    end_w = pd.Series(sim.weights(end_prices))
    largest_pos = float(end_w.max()) if len(end_w) else np.nan
    hhi = float((end_w ** 2).sum()) if len(end_w) else np.nan

    metrics = {
        "Performance": {
            "Total Return": round(total_ret, 4),
            "CAGR": round(cagr, 4),
            "Ann Return": round(ann_ret, 4),
            "Ann Volatility": round(ann_vol, 4),
            "Sharpe": round(sharpe, 3) if sharpe == sharpe else None,
            "Sortino": round(sortino, 3) if sortino == sortino else None,
            "Calmar": round(calmar, 3) if calmar == calmar else None,
            "Max Drawdown": round(mdd, 4),
            "Win Rate (period)": round(win_rate, 3) if win_rate == win_rate else None,
            "Profit Factor": round(profit_factor, 3) if profit_factor == profit_factor else None,
        },
        "Trading": {
            "Rebalances": n_rebal,
            "Num Fills": len(sim.fills),
            "Num Round Trips": len(sim.round_trips),
            "Turnover (ann)": round(turnover_ann, 3) if turnover_ann == turnover_ann else None,
            "Avg Holding (days)": round(avg_hold, 1) if avg_hold == avg_hold else None,
            "Commission Paid": round(sim.commission_paid, 2),
            "Slippage Paid": round(sim.slippage_paid, 2),
            "Cash Drag (avg cash wt)": round(_avg_cash_weight(sim, closes, days), 4),
        },
        "Portfolio": {
            "Target Positions": strat.top_n,
            "Realized Trade Win Rate": round((realized > 0).mean(), 3) if len(realized) else None,
            "Avg Realized PnL": round(realized.mean(), 2) if len(realized) else None,
            "End Largest Position": round(largest_pos, 3) if largest_pos == largest_pos else None,
            "End Concentration (HHI)": round(hhi, 3) if hhi == hhi else None,
            "Beta vs Nifty": "needs index series (hook)",
        },
        "_period_returns": period_returns,
    }
    return metrics


def _avg_cash_weight(sim, closes, days) -> float:
    # exact cash drag requires daily snapshots; approximate with final cash weight
    last = days[-1]
    return sim.cash_weight(closes.loc[last].dropna())


def print_report(res: BacktestResult) -> None:
    for section, vals in res.metrics.items():
        if section.startswith("_"):
            continue
        print(f"\n== {section} ==")
        for k, v in vals.items():
            print(f"  {k:<26} {v}")
    print("\n== Factor Importance (avg attribution) ==")
    print(res.importance.round(4).to_string())
    print("\nNOTE: research output. Validate out-of-sample before trusting it.")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    syms = [p.stem for p in fe.FEATURE_DIR.glob("*.parquet")]
    if not syms:
        print("No features cached. Run data_engine.py then feature_engine.py.")
    else:
        res = run_backtest(syms, start="2019-01-01", strat=se.StrategyConfig(top_n=5))
        print_report(res)
