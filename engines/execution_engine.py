"""
Execution Engine (Execution Simulator)
======================================
A reusable, stateful broker simulator. The Backtest Engine feeds it target
weights and the day's prices; it models the *mechanics* of actually trading:

  - next-OPEN fills (the caller passes open prices; the simulator never peeks)
  - commissions (bps of traded notional)
  - slippage (bps; buys fill higher, sells fill lower)
  - lot sizes (integer share lots; fractional optional)
  - liquidity caps (a trade can't exceed X% of recent ADV in shares)
  - cash balance tracking (no leverage; under-funded targets scale to cash)
  - realized P&L via average-cost (for win rate / profit factor)
  - holding-period tracking (entry->full-exit, in calendar days)

Why separate from the Backtest Engine? The same simulator drives both historical
backtests and future paper trading — identical fill logic, so paper results are
directly comparable to the backtest. The Backtest Engine just consumes the
executed trades and the equity it reports.

Research scaffolding, not investment advice.

Dependencies:
    pip install pandas numpy
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class ExecConfig:
    commission_bps: float = 3.0       # 0.03% per side
    slippage_bps: float = 5.0         # 0.05% adverse fill
    lot_size: int = 1
    max_participation: float = 0.10   # max fraction of ADV (shares) per trade
    allow_fractional: bool = False


@dataclass
class _Position:
    shares: float = 0.0
    cost_basis: float = 0.0           # total cost of current shares (incl. costs)
    entry_date: pd.Timestamp | None = None


class ExecutionSimulator:
    """Long-only portfolio simulator. Deterministic given inputs."""

    def __init__(self, starting_cash: float = 1_000_000.0, cfg: ExecConfig | None = None):
        self.cfg = cfg or ExecConfig()
        self.cash = float(starting_cash)
        self.positions: dict[str, _Position] = {}
        # logs
        self.fills: list[dict] = []
        self.round_trips: list[dict] = []   # realized P&L per full/partial close
        self.holding_periods: list[int] = []
        self.commission_paid = 0.0
        self.slippage_paid = 0.0
        self.turnover_notional = 0.0        # cumulative traded notional

    # -- valuation -------------------------------------------------------
    def equity(self, prices: pd.Series) -> float:
        val = self.cash
        for sym, pos in self.positions.items():
            p = prices.get(sym)
            if p is not None and p > 0:
                val += pos.shares * p
        return val

    def weights(self, prices: pd.Series) -> dict[str, float]:
        eq = self.equity(prices)
        if eq <= 0:
            return {}
        return {s: (p.shares * prices.get(s, 0.0)) / eq
                for s, p in self.positions.items() if p.shares > 0}

    def cash_weight(self, prices: pd.Series) -> float:
        eq = self.equity(prices)
        return self.cash / eq if eq > 0 else 1.0

    # -- trading ---------------------------------------------------------
    def _round_lot(self, shares: float) -> float:
        if self.cfg.allow_fractional:
            return shares
        return math.floor(shares / self.cfg.lot_size) * self.cfg.lot_size

    def rebalance(
        self,
        date: pd.Timestamp,
        open_prices: pd.Series,
        target_weights: dict[str, float],
        adv_shares: pd.Series | None = None,
    ) -> list[dict]:
        """
        Move the book toward target_weights at this day's open prices.
        Positions not in target_weights are exited. Returns the day's fills.
        """
        date = pd.Timestamp(date)
        equity = self.equity(open_prices)
        day_fills: list[dict] = []

        # Desired share counts
        targets: dict[str, float] = {}
        for sym, w in target_weights.items():
            p = open_prices.get(sym)
            if p is None or p <= 0:
                continue
            targets[sym] = self._round_lot((w * equity) / p)
        # Anything currently held but not targeted -> exit
        for sym in self.positions:
            targets.setdefault(sym, 0.0)

        # Execute sells first (frees cash), then buys
        deltas = {s: targets[s] - self.positions.get(s, _Position()).shares for s in targets}
        order = sorted(deltas, key=lambda s: deltas[s])  # negative (sells) first

        for sym in order:
            delta = deltas[sym]
            if delta == 0:
                continue
            p = open_prices.get(sym)
            if p is None or p <= 0:
                continue

            # liquidity cap (in shares)
            if adv_shares is not None and sym in adv_shares.index:
                cap = self.cfg.max_participation * float(adv_shares.get(sym, 0.0))
                if cap > 0 and abs(delta) > cap:
                    delta = math.copysign(self._round_lot(cap), delta)
            if delta == 0:
                continue

            side = 1 if delta > 0 else -1
            fill_price = p * (1 + side * self.cfg.slippage_bps / 1e4)
            notional = abs(delta) * fill_price
            commission = notional * self.cfg.commission_bps / 1e4
            slip_cost = abs(delta) * p * self.cfg.slippage_bps / 1e4

            # cash guard on buys: scale down if insufficient cash
            if side > 0 and (notional + commission) > self.cash:
                affordable = self._round_lot(
                    max(0.0, self.cash / (fill_price * (1 + self.cfg.commission_bps / 1e4)))
                )
                if affordable <= 0:
                    continue
                delta = affordable
                notional = abs(delta) * fill_price
                commission = notional * self.cfg.commission_bps / 1e4
                slip_cost = abs(delta) * p * self.cfg.slippage_bps / 1e4

            pos = self.positions.setdefault(sym, _Position())

            if side > 0:  # BUY
                if pos.shares == 0:
                    pos.entry_date = date
                pos.shares += delta
                pos.cost_basis += notional + commission
                self.cash -= notional + commission
            else:         # SELL
                sold = min(abs(delta), pos.shares)
                avg_cost = pos.cost_basis / pos.shares if pos.shares else 0.0
                proceeds = sold * fill_price - commission
                realized = proceeds - sold * avg_cost
                trip_ret = (fill_price - avg_cost) / avg_cost if avg_cost else 0.0
                hold_days = (date - pos.entry_date).days if pos.entry_date else None
                self.round_trips.append({
                    "symbol": sym,
                    "entry_date": pos.entry_date,
                    "exit_date": date,
                    "shares": sold,
                    "avg_cost": avg_cost,        # per-share cost incl. buy commission
                    "exit_price": fill_price,    # gross fill (sell commission in pnl)
                    "pnl": realized,
                    "return": trip_ret,
                    "holding_days": hold_days,
                })
                pos.shares -= sold
                pos.cost_basis -= sold * avg_cost
                self.cash += proceeds
                if pos.shares <= 1e-9:  # fully closed
                    if pos.entry_date is not None:
                        self.holding_periods.append((date - pos.entry_date).days)
                    pos.shares = 0.0
                    pos.cost_basis = 0.0
                    pos.entry_date = None

            self.commission_paid += commission
            self.slippage_paid += slip_cost
            self.turnover_notional += notional
            fill = {
                "date": date, "symbol": sym, "side": "BUY" if side > 0 else "SELL",
                "shares": abs(delta), "price": fill_price, "commission": commission,
                "slippage": slip_cost,
            }
            self.fills.append(fill)
            day_fills.append(fill)

        # drop empty positions
        self.positions = {s: p for s, p in self.positions.items() if p.shares > 0}
        return day_fills
